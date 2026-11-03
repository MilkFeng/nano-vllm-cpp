#!/usr/bin/env python3
"""Interleaved before/after BF16 decode benchmark, including KV append."""

import argparse
import collections
import csv
import gc
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from baselines import custom
from bridge import Bridge, ROOT, default_library
from cases import Bank, expand_cases
from correctness import references, validate
from measure import measure_interleaved, prepare_timer
from report import geometric_mean
from run_attn import (
    configure_cuda,
    gpu_snapshot,
    save,
    sha256,
    source_hashes,
    rotating_banks,
)


def worker(args, output):
    configure_cuda()
    before, after = Bridge(args.before_library), Bridge(args.after_library)
    if before.build_type != "Release" or after.build_type != "Release":
        raise ValueError("both libraries must be Release builds")
    hashes = source_hashes()
    for name in (
        "src/kernels/cuda/kernels.hh",
        "src/core/forward_context.hh",
        "src/engine/model_runner.cc",
        "src/engine/model_runner.hh",
        "src/models/qwen3.cc",
        "src/models/qwen35.cc",
    ):
        hashes[name] = sha256(ROOT / name)
    document = {
        "status": "running",
        "seed": args.seed,
        "samples": args.samples,
        "dtype": "bfloat16",
        "append_included": True,
        "before_library_sha256": sha256(args.before_library),
        "after_library_sha256": sha256(args.after_library),
        "source_sha256": hashes,
        "gpu_before": gpu_snapshot(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "rows": [],
    }
    stream = torch.cuda.Stream()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    cases = [
        (case, cache)
        for case in expand_cases("full")
        if case.phase in ("decode", "ragged_decode")
        for cache in ("reuse", "rotating")
    ]
    random.Random(args.seed).shuffle(cases)
    document["expected_rows"] = len(cases)
    save(output, document)
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            for index, (case, cache) in enumerate(cases):
                bank = Bank(case, torch.bfloat16, args.seed)
                reference = references(bank)
                bank.reset()
                old_banks, working_set = rotating_banks(bank, cache, 32, l2, 0)
                new_banks = [item.clone_addresses() for item in old_banks]
                implementations = {
                    "before": custom(before, old_banks),
                    "after": custom(after, new_banks),
                }
                for name, banks in (("before", old_banks), ("after", new_banks)):
                    implementation = implementations[name]
                    implementation.correctness = validate(
                        implementation, banks[0], reference, "bfloat16"
                    )
                    implementation.timer = prepare_timer(
                        implementation.launches, warmup=20, max_calls=1024, target_ms=10
                    )
                measured, order = measure_interleaved(
                    implementations, args.samples, args.seed + index
                )
                row = {
                    "case": case.document(),
                    "cache": cache,
                    "working_set": working_set,
                    "measurements": measured,
                    "sample_order": order,
                    "correctness": {
                        n: x.correctness for n, x in implementations.items()
                    },
                    "workspace_bytes": {
                        n: x.metadata["workspace_bytes"]
                        for n, x in implementations.items()
                    },
                }
                document["rows"].append(row)
                save(output, document)
                print(
                    f"[{index+1}/{len(cases)}] {case.name} {cache}: "
                    f"{measured['before']['median_us']:.2f} -> {measured['after']['median_us']:.2f} us",
                    flush=True,
                )
                del (
                    implementations,
                    implementation,
                    banks,
                    bank,
                    reference,
                    old_banks,
                    new_banks,
                )
                gc.collect()
                torch.cuda.empty_cache()
        assert hashes == {name: sha256(ROOT / name) for name in hashes}
        assert document["after_library_sha256"] == sha256(args.after_library)
        assert document["before_library_sha256"] == sha256(args.before_library)
        document["status"] = "complete"
        document["gpu_after"] = gpu_snapshot()
        save(output, document)
    except Exception as error:
        document.update(status="failed", error=str(error))
        save(output, document)
        raise


def report(directory):
    documents = [
        json.loads(p.read_text()) for p in sorted(directory.glob("run-*.json"))
    ]
    grouped = collections.defaultdict(list)
    for document in documents:
        assert (
            document["status"] == "complete"
            and len(document["rows"]) == document["expected_rows"]
        )
        for row in document["rows"]:
            grouped[(row["case"]["name"], row["cache"])].append(row)
    rows = []
    for key, records in sorted(grouped.items()):
        assert len(records) == len(documents)
        old = statistics.median(
            r["measurements"]["before"]["median_us"] for r in records
        )
        new = statistics.median(
            r["measurements"]["after"]["median_us"] for r in records
        )
        rows.append(
            {
                "name": key[0],
                "cache": key[1],
                "case": records[0]["case"],
                "before_us": old,
                "after_us": new,
                "speedup": old / new,
            }
        )
    summaries = []
    for profile in ("qwen3_0p6b", "qwen35_full_attention"):
        for cache in ("reuse", "rotating"):
            selected = [
                r
                for r in rows
                if r["case"]["profile"] == profile
                and r["case"]["phase"] == "decode"
                and r["cache"] == cache
            ]
            summaries.append(
                {
                    "profile": profile,
                    "cache": cache,
                    "shapes": len(selected),
                    "geomean_speedup": geometric_mean(r["speedup"] for r in selected),
                }
            )
    save(directory / "summary.json", {"groups": summaries, "cases": rows})
    with (directory / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=["name", "cache", "before_us", "after_us", "speedup"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "case"} for r in rows)
    lines = [
        "# Split-KV decode：同进程交错对比",
        "",
        f'{len(documents)} 个独立进程，BF16，每项每实现 {documents[0]["samples"]} 组 CUDA Graph/event 样本。',
        "两端都包含 KV 追加；新版本的分片计算和归并均计时。预分配、正确性验证、Graph 构建不计时。",
        "每项先取样本中位数，再取进程中位数；逐 shape 的旧/新延迟等权取几何平均。",
        "",
        "| 配置 | 缓存 | decode shapes | 相较单 block 版本 |",
        "| --- | --- | ---: | ---: |",
    ]
    for group in summaries:
        lines.append(
            f"| {group['profile']} | {group['cache']} | {group['shapes']} | {group['geomean_speedup']:.2f}× |"
        )
    lines += [
        "",
        "## BF16 / reuse / B=1 / KV=8192",
        "",
        "| 配置 | 单 block | Split-KV | 加速 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        if (
            r["cache"] == "reuse"
            and r["case"]["query_lengths"] == [1]
            and r["case"]["kv_lengths"] == [8192]
        ):
            lines.append(
                f"| {r['case']['profile']} | {r['before_us']:.2f} μs | {r['after_us']:.2f} μs | {r['speedup']:.2f}× |"
            )
    lines += [
        "",
        "完整 shape 延迟见 summary.csv；原始样本、每组执行顺序、误差、工作集与工作区大小见 run-*.json。",
        "rotating 为地址轮换，未达到目标工作集的项保留标记，不保证全冷缓存。",
        "这是 BF16 decode 算子消融，不包含 FlashInfer 重测或端到端生成性能。",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-library",
        default=str(ROOT / "build/attn-split-kv-before/libnano-vllm-cpp-attn-exp.so"),
    )
    parser.add_argument("--after-library", default=str(default_library()))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--samples", default=15, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.samples < 3:
        parser.error("positive runs and at least three samples required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, args.output_dir / f"run-{args.seed-41:03}.json")
    else:
        if list(args.output_dir.glob("run-*.json")):
            raise ValueError("use a new output directory")
        for seed in range(args.seed, args.seed + args.runs):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--before-library",
                args.before_library,
                "--after-library",
                args.after_library,
                "--output-dir",
                str(args.output_dir),
                "--samples",
                str(args.samples),
                "--seed",
                str(seed),
            ]
            subprocess.run(command, check=True, cwd=ROOT)
        report(args.output_dir)


if __name__ == "__main__":
    main()
