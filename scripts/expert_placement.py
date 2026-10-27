#!/usr/bin/env python3
"""Generate static MoE placement files and benchmark equal-budget candidates."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return json.load(source)


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")


def load_profiles(paths):
    metadata = None
    counts = None
    identity = ("num_hidden_layers", "num_experts", "top_k", "hidden_size",
                "intermediate_size", "expert_size_bytes")
    for path in paths:
        report = read_json(path)
        if not report.get("completed"):
            raise ValueError(f"incomplete calibration report: {path}")
        placement = report.get("expert_placement")
        if not placement or any(not isinstance(placement.get(key), int)
                                or placement[key] <= 0 for key in identity):
            raise ValueError(f"missing MoE model metadata: {path}")
        if metadata is None:
            metadata = {key: placement[key] for key in identity}
            counts = {phase: [[0] * metadata["num_experts"]
                              for _ in range(metadata["num_hidden_layers"])]
                      for phase in ("prefill", "decode")}
        elif any(metadata[key] != placement[key] for key in identity):
            raise ValueError("calibration model dimensions do not match")
        runs = 0
        for case in report["cases"]:
            for run in case["runs"]:
                profile = run.get("expert_profile", [])
                if len(profile) != metadata["num_hidden_layers"]:
                    raise ValueError(f"missing expert profile; rerun with --profile-experts: {path}")
                seen_layers = set()
                for layer in profile:
                    index = layer["layer"]
                    if (not isinstance(index, int) or index in seen_layers
                            or not 0 <= index < metadata["num_hidden_layers"]):
                        raise ValueError("invalid or duplicate profile layer")
                    seen_layers.add(index)
                    for phase in counts:
                        entry = layer[phase]
                        total = 0
                        seen_experts = set()
                        for expert in entry["experts"]:
                            expert_id, tokens = expert["id"], expert["tokens"]
                            if (not isinstance(expert_id, int)
                                    or not 0 <= expert_id < metadata["num_experts"]
                                    or expert_id in seen_experts
                                    or not isinstance(tokens, int) or tokens < 0):
                                raise ValueError("invalid expert routing counts")
                            seen_experts.add(expert_id)
                            counts[phase][index][expert_id] += tokens
                            total += tokens
                        if (total != entry["input_tokens"] * metadata["top_k"]
                                or total != entry["gpu_assignments"] + entry["cpu_assignments"]):
                            raise ValueError("inconsistent expert assignment totals")
                runs += 1
        if not runs:
            raise ValueError(f"calibration report contains no measured runs: {path}")
    return metadata, counts


def make_placement(metadata, counts, *, phase, prefill_weight, strategy,
                   budget_mib=None, experts_per_layer=None):
    layers, experts = metadata["num_hidden_layers"], metadata["num_experts"]
    expert_bytes = metadata["expert_size_bytes"]
    totals = {name: sum(map(sum, values)) for name, values in counts.items()}
    weights = {"prefill": float(phase == "prefill"), "decode": float(phase == "decode")}
    if phase == "mixed":
        weights = {"prefill": prefill_weight, "decode": 1 - prefill_weight}
    for name, weight in weights.items():
        if weight and not totals[name]:
            raise ValueError(f"no {name} samples in calibration; select another --phase")
    scores = [[sum(weights[name] * counts[name][layer][expert] / totals[name]
                   for name in weights if weights[name])
               for expert in range(experts)] for layer in range(layers)]
    if experts_per_layer is not None:
        if not 0 <= experts_per_layer <= experts:
            raise ValueError(f"experts per layer must be in [0, {experts}]")
        slots = experts_per_layer * layers
        budget_bytes = slots * expert_bytes
    else:
        if not math.isfinite(budget_mib) or budget_mib < 0:
            raise ValueError("budget must be a finite nonnegative MiB value")
        budget_bytes = int(budget_mib * 2**20)
        slots = min(budget_bytes // expert_bytes, layers * experts)
    selected = [[] for _ in range(layers)]
    if strategy == "uniform" or experts_per_layer is not None:
        for layer in range(layers):
            count = slots // layers + int(layer < slots % layers)
            order = range(experts)
            if strategy == "hot":
                order = sorted(order, key=lambda expert: (-scores[layer][expert], expert))
            selected[layer] = sorted(list(order)[:count])
    else:
        candidates = sorted(((-scores[layer][expert], layer, expert)
                             for layer in range(layers) for expert in range(experts)))
        for _, layer, expert in candidates[:slots]:
            selected[layer].append(expert)
        for ids in selected:
            ids.sort()
    coverage = {name: (sum(values[layer][expert] for layer in range(layers)
                          for expert in selected[layer]) / totals[name]
                       if totals[name] else None)
                for name, values in counts.items()}
    return {"schema_version": 1, **metadata, "default": [],
            "layers": {str(layer): ids for layer, ids in enumerate(selected)},
            "strategy": strategy, "phase": phase, "prefill_weight": prefill_weight,
            "budget_bytes": budget_bytes, "cache_bytes": slots * expert_bytes,
            "gpu_slots": slots, "calibration_assignment_coverage": coverage}


def case_summary(report):
    result = []
    for case in report["cases"]:
        summary = case["summary"]
        result.append({key: case[key] for key in ("input_tokens", "output_tokens", "batch_size")} | {
            "prefill_ms": summary["prefill_ms"]["p50"],
            "decode_tokens_per_second": summary["decode_output_tokens_per_second"]["mean"],
            "ttft_ms": summary["ttft_ms"]["p50"],
            "tpot_ms": summary["tpot_ms"]["p50"]})
    return result


def add_speedups(rows):
    uniform = {(row["budget"], case["input_tokens"], case["output_tokens"], case["batch_size"]): case
               for row in rows if row["strategy"] == "uniform" and row["status"] == "ok"
               for case in row["cases"]}
    for row in rows:
        for case in row.get("cases", []):
            baseline = uniform.get((row["budget"], case["input_tokens"],
                                    case["output_tokens"], case["batch_size"]))
            if baseline:
                before, after = baseline["prefill_ms"], case["prefill_ms"]
                case["prefill_speedup_vs_uniform"] = before / after if before and after else None
                before, after = baseline["decode_tokens_per_second"], case["decode_tokens_per_second"]
                case["decode_speedup_vs_uniform"] = after / before if before and after else None


def sweep(args, metadata, counts):
    forwarded = args.bench_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    reserved = {"--gpu-expert-ids", "--gpu-expert-config", "--profile-experts", "--output", "-o"}
    if any(value.split("=", 1)[0] in reserved or value.startswith("-o")
           and not value.startswith("--") for value in forwarded):
        raise ValueError("sweep controls placement/output and disables profiling; remove conflicting benchmark flags")
    executable = str(Path(args.bench_exe).resolve())
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    strategies = args.strategies.split(",")
    if not strategies or any(strategy not in ("uniform", "hot") for strategy in strategies):
        raise ValueError("strategies must be uniform and/or hot")
    values = (args.counts if args.counts is not None else args.budgets_mib).split(",")
    rows = []
    comparison_path = directory / "comparison.json"
    for value in values:
        limit = {"experts_per_layer": int(value)} if args.counts is not None else {"budget_mib": float(value)}
        label = ("count-" if args.counts is not None else "mib-") + str(next(iter(limit.values())))
        for strategy in dict.fromkeys(strategies):
            config = make_placement(metadata, counts, phase=args.phase,
                                    prefill_weight=args.prefill_weight, strategy=strategy, **limit)
            config["calibration_reports"] = [str(Path(path).resolve()) for path in args.profile]
            name = f"{label}-{strategy}"
            config_path, report_path = directory / f"{name}.config.json", directory / f"{name}.bench.json"
            save_json(config_path, config)
            command = [executable, *forwarded, "--gpu-expert-config", str(config_path),
                       "--output", str(report_path)]
            report_path.unlink(missing_ok=True)
            print(f"\n{name}: {config['cache_bytes'] / 2**20:.2f} MiB routed expert cache", flush=True)
            row = {"budget": label, "strategy": strategy, "cache_bytes": config["cache_bytes"],
                   "config": str(config_path), "report": str(report_path),
                   "calibration_assignment_coverage": config["calibration_assignment_coverage"],
                   "command": command}
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                row.update(status="failed", returncode=completed.returncode)
            else:
                report = read_json(report_path)
                if (not report.get("completed") or report["settings"].get("profile_experts")
                        or report["expert_placement"]["layers"] != config["layers"]):
                    raise ValueError(f"incomplete or mismatched benchmark: {report_path}")
                row.update(status="ok", cases=case_summary(report),
                           memory_after_load=report.get("memory_after_load"))
            rows.append(row)
            add_speedups(rows)
            save_json(comparison_path, {"schema_version": 1, "runs": rows})
    print("\nCandidate                      I/O/B          Cache MiB   Prefill ms   Decode tok/s")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['budget']}-{row['strategy']}: FAILED (exit {row['returncode']})")
        for case in row.get("cases", []):
            workload = f"{case['input_tokens']}/{case['output_tokens']}/{case['batch_size']}"
            decode = case["decode_tokens_per_second"]
            decode_text = f"{decode:.2f}" if decode is not None else "n/a"
            print(f"{row['budget'] + '-' + row['strategy']:<30} {workload:<14}"
                  f"{row['cache_bytes'] / 2**20:>9.2f} {case['prefill_ms']:>12.2f} {decode_text:>14}")
    print(f"Comparison: {comparison_path}")
    return int(any(row["status"] != "ok" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "sweep"):
        command = commands.add_parser(name)
        command.add_argument("--profile", nargs="+", required=True,
                             help="completed benchmark reports collected with --profile-experts")
        command.add_argument("--phase", choices=("prefill", "decode", "mixed"), default="mixed")
        command.add_argument("--prefill-weight", type=float, default=0.5,
                             help="mixed phase weight after normalizing each phase separately (default: 0.5)")
        limits = command.add_mutually_exclusive_group(required=True)
        if name == "plan":
            limits.add_argument("--budget-mib", type=float, help="global routed expert cache budget only")
            limits.add_argument("--experts-per-layer", type=int)
            command.add_argument("--strategy", choices=("uniform", "hot"), default="hot")
            command.add_argument("-o", "--output", required=True)
        else:
            limits.add_argument("--budgets-mib", help="comma-separated global routed expert cache budgets")
            limits.add_argument("--counts", help="comma-separated GPU expert counts per layer")
            command.add_argument("--strategies", default="uniform,hot")
            command.add_argument("--bench-exe", required=True)
            command.add_argument("--output-dir", required=True)
            command.add_argument("bench_args", nargs=argparse.REMAINDER,
                                 help="after --, ordinary benchmark arguments including --model and --num-blocks")
    args = parser.parse_args()
    try:
        if not 0 <= args.prefill_weight <= 1:
            raise ValueError("prefill weight must be in [0, 1]")
        metadata, counts = load_profiles(args.profile)
        if args.command == "sweep":
            return sweep(args, metadata, counts)
        config = make_placement(metadata, counts, phase=args.phase,
                                prefill_weight=args.prefill_weight, strategy=args.strategy,
                                budget_mib=args.budget_mib, experts_per_layer=args.experts_per_layer)
        config["calibration_reports"] = [str(Path(path).resolve()) for path in args.profile]
        save_json(args.output, config)
        print(f"{config['gpu_slots']} GPU slots; {config['cache_bytes'] / 2**20:.2f} MiB; "
              f"calibration assignment coverage: {config['calibration_assignment_coverage']}")
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
