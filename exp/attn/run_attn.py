#!/usr/bin/env python3
"""Run reproducible attention comparisons without mixing dtype or task boundaries."""

# Cache locations must be selected before importing Torch/FlashInfer modules.
# ruff: noqa: E402

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT / "build/attn-cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "build/attn-cache/torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "build/attn-cache/triton"))
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("MAX_JOBS", "4")

import torch

from baselines import (
    classify_failure,
    custom,
    flash_attention_optional,
    flashinfer,
    sdpa,
    verify_sdpa_backend,
)
from bridge import Bridge, default_library
from cases import Bank, Case, DTYPES, SPEC, expand_cases
from correctness import canonical, compare, references, validate
from measure import measure_interleaved, prepare_timer

TASKS = ("pipeline", "prefill")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def source_hashes():
    files = [
        ROOT / "CMakeLists.txt",
        ROOT / "src/kernels/cuda/attention.cu",
        ROOT / "src/kernels/cuda/common.cuh",
        ROOT / "src/kernels/cuda/reduce.cuh",
    ]
    for pattern in ("*.py", "*.cc", "*.cu", "*.hh", "suite.json"):
        files += sorted(Path(__file__).parent.glob(pattern))
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def command_output(command):
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except OSError as error:
        return {"error": str(error)}


def gpu_snapshot():
    return command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,utilization.gpu,"
            "temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
    )


def metadata(args, bridge):
    properties = torch.cuda.get_device_properties(0)
    versions = {}
    for package in (
        "torch",
        "flashinfer-python",
        "flash-attn",
        "nvidia-cudnn-frontend",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": {
            "gpu": properties.name,
            "sm": [properties.major, properties.minor],
            "vram_bytes": properties.total_memory,
            "l2_bytes": properties.L2_cache_size,
        },
        "software": {
            "packages": versions,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "build_type": bridge.build_type,
        },
        "options": {
            key: value
            for key, value in vars(args).items()
            if key not in ("worker", "output", "output_dir")
        },
        "contract": {
            "page_size": 16,
            "causal": True,
            "scale": "1/sqrt(D)",
            "rope": False,
            "dropout": 0,
            "timing": "CUDA Graph + CUDA events; fixed lengths per replay",
            "tolerances": SPEC["correctness"]["tolerances"],
            "production_optimized": False,
        },
        "source_sha256": source_hashes(),
        "library_sha256": sha256(bridge.path),
        "rows": [],
    }


def configure_cuda():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setter = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    if setter:
        setter(False)


def edge_checks(bridge):
    records = []
    shapes = [(1, 5), (2, 5), (3, 20), (5, 17), (17, 32), (65, 322), (0, 0)]
    for dtype_name, (hq, hkv, dim), causal in itertools.product(
        DTYPES,
        ((8, 8, 64), (8, 1, 80), (8, 2, 96), (16, 8, 128), (16, 2, 256)),
        (True, False),
    ):
        for query, key in shapes:
            case = Case(
                f"edge_q{query}_k{key}",
                "correctness",
                "edge",
                hq,
                hkv,
                dim,
                (query,),
                (key,),
                causal=causal,
            )
            bank = Bank(case, DTYPES[dtype_name], 42, deterministic=query <= 3)
            reference = references(bank)
            implementation = custom(bridge, [bank])
            result = validate(implementation, bank, reference, dtype_name)
            records.append(
                {
                    "dtype": dtype_name,
                    "hq": hq,
                    "hkv": hkv,
                    "dim": dim,
                    "causal": causal,
                    "q": query,
                    "k": key,
                    "implementation": implementation.name,
                    **result,
                }
            )
            del bank, reference, implementation
    # Mixed empty/nonempty requests and empty batches need no external backend.
    for qlens, klens in (((0, 1, 3), (0, 17, 36)), ((), ())):
        case = Case("empty_batch_edges", "correctness", "edge", 8, 2, 64, qlens, klens)
        bank = Bank(case, torch.float32, 42)
        reference = references(bank)
        implementation = custom(bridge, [bank])
        validate(implementation, bank, reference, "float32")
    torch.cuda.synchronize()
    return {
        "status": "ok",
        "checks": len(records) + 2,
        "max_abs_error": max(row["max_abs_error"] for row in records),
        "per_dtype": {
            name: {
                "max_abs_error": max(
                    row["max_abs_error"] for row in records if row["dtype"] == name
                ),
                "max_tolerance_ratio": max(
                    row["max_tolerance_ratio"]
                    for row in records
                    if row["dtype"] == name
                ),
            }
            for name in DTYPES
        },
        "note": "Deterministic V includes KV-head offsets up to 448; BF16 absolute error reflects that scale.",
    }


def rotating_banks(bank, cache, max_banks, l2_bytes, workspace_bytes):
    if cache == "reuse":
        return [bank], {
            "banks": 1,
            "target_bytes": 0,
            "target_reached": True,
            "working_set_bytes": bank.bytes(),
        }
    target = max(64 * 2**20, 2 * l2_bytes)
    free_bytes, _ = torch.cuda.mem_get_info()
    limit = max(1, int((free_bytes - workspace_bytes) * 0.25) // bank.bytes())
    count = min(max_banks, limit, max(1, math.ceil(target / bank.bytes())))
    banks = [bank] + [bank.clone_addresses() for _ in range(count - 1)]
    return banks, {
        "banks": count,
        "target_bytes": target,
        "target_reached": count * bank.bytes() >= target,
        "working_set_bytes": count * bank.bytes(),
        "bank_limit_bytes": int(free_bytes * 0.25),
    }


def tune_flashinfer(banks, workspace, reference, dtype_name, args):
    case = banks[0].case
    configs = (
        [
            {"backend": backend, "tensor_cores": tensor_cores}
            for backend in ("auto", "fa2")
            for tensor_cores in (False, True)
        ]
        if case.phase in ("decode", "ragged_decode")
        else [{"backend": backend} for backend in ("auto", "fa2")]
    )
    trials, selected, selected_config = [], None, None
    best = math.inf
    seen = set()
    for config in configs:
        record = {"configuration": config}
        implementation = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                implementation = flashinfer(banks, workspace, config)
                record["metadata"] = implementation.metadata
                identity = (
                    tuple(implementation.metadata["resolved_backends"]),
                    config.get("tensor_cores"),
                )
                if identity in seen:
                    record["status"] = "duplicate_configuration"
                    trials.append(record)
                    continue
                seen.add(identity)
                implementation.correctness = validate(
                    implementation, banks[0], reference, dtype_name
                )
                timer = prepare_timer(
                    implementation.launches, 3, min(args.max_calls, 256), 2
                )
                samples = [timer.sample() for _ in range(3)]
                del timer
                elapsed = statistics.median(samples)
                record.update(
                    status="ok",
                    samples_us=samples,
                    median_us=elapsed,
                    correctness=implementation.correctness,
                    warnings=[str(item.message) for item in caught],
                )
                if elapsed < best:
                    best, selected_config = elapsed, config
        except Exception as error:
            record.update(status=classify_failure(error), reason=str(error))
        finally:
            del implementation
        trials.append(record)
    if selected_config is not None:
        selected = flashinfer(banks, workspace, selected_config)
        selected.correctness = validate(selected, banks[0], reference, dtype_name)
    return selected, trials


def run_row(case, task, dtype_name, cache, bridge, args, l2_bytes):
    bank = Bank(case, DTYPES[dtype_name], args.seed, identity=task == "prefill")
    reference = references(bank)
    bank.reset()
    banks, working_set = rotating_banks(
        bank,
        cache,
        min(args.max_banks, args.max_calls),
        l2_bytes,
        args.workspace_mib * 2**20,
    )
    row = {
        "case": case.document(),
        "task": task,
        "dtype": dtype_name,
        "cache": cache,
        "bank": bank.metadata(),
        "working_set": working_set,
        "implementations": {},
        "calibration": [],
    }
    implementations = {}
    own = custom(bridge, banks)
    own.correctness = validate(own, bank, reference, dtype_name)
    implementations[own.name] = own
    if task == "pipeline" and not args.custom_only:
        workspace = torch.empty(
            args.workspace_mib * 2**20, dtype=torch.uint8, device="cuda"
        )
        selected, trials = tune_flashinfer(
            banks, workspace, reference, dtype_name, args
        )
        row["calibration"] = trials
        if selected:
            implementations[selected.name] = selected
        else:
            statuses = [trial["status"] for trial in trials]
            row["implementations"]["flashinfer"] = {
                "status": "not_installed"
                if "not_installed" in statuses
                else "incorrect"
                if "incorrect" in statuses
                else "error"
                if "error" in statuses
                else "unsupported",
                "reason": "no correct supported configuration; see calibration",
            }
    if task == "prefill" and not args.custom_only:
        for backend in ("math", "flash", "cudnn", "flash_attention_optional"):
            name = (
                backend if backend == "flash_attention_optional" else "sdpa_" + backend
            )
            try:
                implementation = (
                    flash_attention_optional(banks)
                    if backend == "flash_attention_optional"
                    else sdpa(banks, backend)
                )
                if backend != "flash_attention_optional":
                    verify_sdpa_backend(implementation)
                implementation.correctness = validate(
                    implementation, bank, reference, dtype_name
                )
                implementations[name] = implementation
            except Exception as error:
                row["implementations"][name] = {
                    "status": classify_failure(error),
                    "reason": str(error),
                }
    for name in list(implementations):
        implementation = implementations[name]
        try:
            # Each rotated address gets the expected final cache before capture.
            for rotated in banks:
                rotated.reset()
            for launch in implementation.launches:
                launch()
            torch.cuda.synchronize()
            compare(
                canonical(implementation.launches[-1](), banks[-1]),
                reference[0],
                dtype_name,
            )
            banks[-1].assert_cache_and_guards()
            implementation.timer = prepare_timer(
                implementation.launches, args.warmup, args.max_calls, args.target_ms
            )
        except Exception as error:
            row["implementations"][name] = {
                "status": classify_failure(error),
                "reason": str(error),
                "metadata": implementation.metadata,
            }
            del implementations[name]
            if name.startswith("custom"):
                raise
    measurements, orders = measure_interleaved(
        implementations,
        args.samples,
        args.seed + sum(case.kv_lengths) + sum(case.query_lengths),
    )
    for name, measurement in measurements.items():
        implementation = implementations[name]
        row["implementations"][name] = {
            "status": "ok",
            **measurement,
            "correctness": implementation.correctness,
            "metadata": implementation.metadata,
        }
    row["sampling_order"] = orders
    return row


def applicable(case, task):
    return task == "pipeline" or (task == "prefill" and case.phase == "prefill")


def worker(args):
    configure_cuda()
    bridge = Bridge(args.library)
    if bridge.build_type not in ("Release", "RelWithDebInfo"):
        raise RuntimeError("performance comparisons require a Release build")
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        if args.check_only:
            result = edge_checks(bridge)
            print(json.dumps(result), flush=True)
            if args.output:
                save(args.output, result)
            return
        document = metadata(args, bridge)
        tasks = args.tasks.split(",")
        caches = ("reuse", "rotating") if args.cache == "both" else (args.cache,)
        rows = [
            (case, task, dtype, cache)
            for case in expand_cases(args.suite)
            for task in tasks
            if applicable(case, task)
            for dtype in args.dtypes.split(",")
            for cache in caches
            if not args.only or args.only in case.name
        ]
        random.Random(args.seed).shuffle(rows)
        document["expected_rows"] = len(rows)
        if not rows:
            raise ValueError("no cases matched the selected task/dtype/filter")
        save(args.output, document)
        try:
            for index, (case, task, dtype, cache) in enumerate(rows):
                started = time.monotonic()
                print(
                    f"[{index + 1}/{len(rows)}] {task} {case.name} {dtype} {cache}",
                    flush=True,
                )
                measured = run_row(
                    case,
                    task,
                    dtype,
                    cache,
                    bridge,
                    args,
                    document["hardware"]["l2_bytes"],
                )
                measured["elapsed_seconds"] = time.monotonic() - started
                document["rows"].append(measured)
                custom_name = "custom"
                custom_us = measured["implementations"][custom_name]["median_us"]
                baseline_times = {
                    name: value["median_us"]
                    for name, value in measured["implementations"].items()
                    if not name.startswith("custom") and value["status"] == "ok"
                }
                print(
                    f"  custom {custom_us:.3f} us; baselines {baseline_times}",
                    flush=True,
                )
                save(args.output, document)
                gc.collect()
                torch.cuda.empty_cache()
            document["status"] = "complete"
            document["source_sha256_after"] = source_hashes()
            document["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            save(args.output, document)
        except Exception as error:
            document["status"] = "failed"
            document["error"] = str(error)
            save(args.output, document)
            raise


def controller(args):
    from report import generate_report

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = Path(args.output_dir or ROOT / "build/attn-results" / stamp).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("run-*.json")):
        raise ValueError(
            "use a new output directory; measurements are never overwritten"
        )
    library = Path(args.library).resolve()
    manifest = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": source_hashes(),
        "library_sha256": sha256(library),
        "python": sys.executable,
        "gpu_before": gpu_snapshot(),
        "processes": [],
    }
    save(directory / "manifest.json", manifest)
    for index in range(args.runs):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--library",
            str(library),
            "--suite",
            args.suite,
            "--tasks",
            args.tasks,
            "--dtypes",
            args.dtypes,
            "--cache",
            args.cache,
            "--seed",
            str(args.seed + index),
            "--samples",
            str(args.samples),
            "--target-ms",
            str(args.target_ms),
            "--max-calls",
            str(args.max_calls),
            "--max-banks",
            str(args.max_banks),
            "--warmup",
            str(args.warmup),
            "--workspace-mib",
            str(args.workspace_mib),
            "--output",
            str(directory / f"run-{index + 1:03}.json"),
        ]
        if args.only:
            command += ["--only", args.only]
        if args.custom_only:
            command += ["--custom-only"]
        record = {"command": command, "gpu_before": gpu_snapshot()}
        started = time.monotonic()
        print(f"Process {index + 1}/{args.runs}", flush=True)
        with (directory / f"run-{index + 1:03}.txt").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                record["exit_code"] = process.wait()
        record["elapsed_seconds"] = time.monotonic() - started
        record["gpu_after"] = gpu_snapshot()
        manifest["processes"].append(record)
        save(directory / "manifest.json", manifest)
        if record["exit_code"]:
            raise RuntimeError(f"attention benchmark failed; inspect {directory}")
    manifest["source_sha256_after"] = source_hashes()
    manifest["source_changed_during_run"] = (
        manifest["source_sha256_after"] != manifest["source_sha256"]
    )
    manifest["library_sha256_after"] = sha256(library)
    manifest["gpu_after"] = gpu_snapshot()
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(directory / "manifest.json", manifest)
    generate_report(directory)
    print(directory / "report.md", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=str(default_library()))
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--dtypes", default=",".join(DTYPES))
    parser.add_argument(
        "--cache", choices=("reuse", "rotating", "both"), default="both"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--target-ms", type=float, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=1024)
    parser.add_argument("--max-banks", type=int, default=32)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="")
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    parser.add_argument("--report-only")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--custom-only", action="store_true")
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.samples < 3
        or args.target_ms < 1
        or args.max_calls < 1
        or args.max_banks < 1
        or args.warmup < 1
        or args.workspace_mib < 1
    ):
        parser.error("invalid sampling parameters")
    if len(set(args.tasks.split(","))) != len(args.tasks.split(",")) or len(
        set(args.dtypes.split(","))
    ) != len(args.dtypes.split(",")):
        parser.error("duplicate task or dtype")
    if not set(args.tasks.split(",")) <= set(TASKS) or not set(
        args.dtypes.split(",")
    ) <= set(DTYPES):
        parser.error("unknown task or dtype")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.report_only:
        from report import generate_report

        generate_report(Path(args.report_only))
    elif args.worker or args.check_only:
        worker(args)
    else:
        controller(args)
