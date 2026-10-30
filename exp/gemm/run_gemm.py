#!/usr/bin/env python3
"""Run independent GEMM benchmark processes and publish a reproducible report."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATIONS = ("custom", "tiled_16x16", "cublas_sgemm",
                   "cublaslt_tuned", "cublaslt_legacy")
MANDATORY = set(IMPLEMENTATIONS) - {"cublaslt_tuned"}


def command_output(command):
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                   text=True, check=False)
        return {"command": command, "exit_code": completed.returncode,
                "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except OSError as error:
        return {"command": command, "error": str(error)}


def gpu_snapshot():
    return command_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate",
        "--format=csv,noheader,nounits",
    ])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot():
    paths = [ROOT / "CMakeLists.txt", ROOT / "src/kernels/cuda/linear.cu",
             ROOT / "src/kernels/cuda/common.cuh", ROOT / "src/kernels/cuda/reduce.cuh"]
    paths += sorted((ROOT / "exp/gemm").glob("*.cc"))
    paths += sorted((ROOT / "exp/gemm").glob("*.cu"))
    paths += sorted((ROOT / "exp/gemm").glob("*.hh"))
    paths += sorted((ROOT / "exp/gemm").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def save_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires positive finite samples")
    return math.exp(statistics.mean(math.log(value) for value in values))


def load_runs(directory):
    paths = sorted(directory.glob("run-*.json"))
    if not paths:
        raise ValueError("no run-*.json results found")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validate_runs(documents)
    return documents


def validate_runs(documents):
    if not documents:
        raise ValueError("no independent process measurements")
    expected_keys = None
    reference_metadata = None
    for document in documents:
        if document["status"] != "complete" or not document["correctness"]["passed"]:
            raise ValueError("incomplete or failed measurements cannot enter the report")
        contract = document["contract"]
        if (contract["dtype"] != "float32"
                or contract["strict_sgemm_math"] != "CUBLAS_PEDANTIC_MATH"
                or contract["strict_lt_compute"] != "CUBLAS_COMPUTE_32F_PEDANTIC"):
            raise ValueError("this report requires strict FP32 measurements")
        options = {key: value for key, value in document["options"].items() if key != "seed"}
        metadata = (document["hardware"], document["software"], document["contract"], options)
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError("hardware, software, precision, and measurement options must match")
        keys = set()
        for case in document["cases"]:
            key = (case["cache"], case["name"], case["group"], case["m"], case["n"], case["k"])
            if key in keys:
                raise ValueError("duplicate benchmark case")
            keys.add(key)
            if not MANDATORY <= set(case["implementations"]):
                raise ValueError("a mandatory implementation is missing")
            for name, measurement in case["implementations"].items():
                samples = measurement["samples_us"]
                if len(samples) != document["options"]["samples"]:
                    raise ValueError("sample count mismatch")
                if any(not math.isfinite(value) or value <= 0 for value in samples):
                    raise ValueError("non-positive or non-finite latency")
                if not math.isclose(statistics.median(samples), measurement["median_us"], rel_tol=1e-9):
                    raise ValueError("stored median does not match raw samples")
                if measurement["correctness"]["max_tolerance_ratio"] > 1:
                    raise ValueError("incorrect result cannot enter the report")
        expected_count = 33 if document["options"]["suite"] == "full" else 5
        expected_count *= 2 if document["options"]["cache"] == "both" else 1
        if len(keys) != expected_count:
            raise ValueError("the predeclared suite has missing cases")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("all process runs must include exactly the same cases")


def aggregate(documents):
    validate_runs(documents)
    by_key = {}
    for run_index, document in enumerate(documents):
        for case in document["cases"]:
            key = (case["cache"], case["name"])
            by_key.setdefault(key, []).append((run_index, case))
    rows = []
    for key, entries in sorted(by_key.items()):
        prototype = entries[0][1]
        row = {field: prototype[field] for field in ("name", "group", "cache", "m", "n", "k")}
        row["slots"] = prototype["slots"]
        row["working_set_bytes"] = prototype["working_set_bytes"]
        row["target_reached"] = all(case["target_reached"] for _, case in entries)
        row["latency_us"] = {}
        for name in IMPLEMENTATIONS:
            available = [case["implementations"][name]["median_us"]
                         for _, case in entries if name in case["implementations"]]
            row["latency_us"][name] = statistics.median(available) if len(available) == len(entries) else None
        best = []
        process_ratios = []
        for _, case in entries:
            metrics = case["implementations"]
            latency = min(metrics[name]["median_us"] for name in ("cublas_sgemm", "cublaslt_tuned")
                          if name in metrics)
            best.append(latency)
            process_ratios.append(latency / metrics["custom"]["median_us"])
        custom = row["latency_us"]["custom"]
        row["best_strict_blas_us"] = statistics.median(best)
        row["speedup_blas"] = row["best_strict_blas_us"] / custom
        row["speedup_tiled"] = row["latency_us"]["tiled_16x16"] / custom
        row["tflops"] = 2 * row["m"] * row["n"] * row["k"] / (custom * 1e6)
        row["process_blas_ratios"] = process_ratios
        row["custom_process_medians_us"] = [case["implementations"]["custom"]["median_us"] for _, case in entries]
        row["max_custom_cv"] = max(case["implementations"]["custom"]["cv"] for _, case in entries)
        row["max_cv_by_implementation"] = {
            name: max(case["implementations"][name]["cv"] for _, case in entries
                      if name in case["implementations"])
            for name in IMPLEMENTATIONS
            if any(name in case["implementations"] for _, case in entries)
        }
        row["max_custom_abs_error"] = max(case["implementations"]["custom"]["correctness"]["max_abs_error"]
                                          for _, case in entries)
        rows.append(row)
    return rows


def group_summary(rows):
    process_geomeans = [geometric_mean(row["process_blas_ratios"][i] for row in rows)
                        for i in range(len(rows[0]["process_blas_ratios"]))]
    return {"cases": len(rows),
            "geomean_speedup_blas": geometric_mean(row["speedup_blas"] for row in rows),
            "geomean_speedup_tiled": geometric_mean(row["speedup_tiled"] for row in rows),
            "process_geomean_min": min(process_geomeans),
            "process_geomean_max": max(process_geomeans),
            "wins_blas": sum(row["speedup_blas"] > 1 for row in rows),
            "peak_tflops": max(row["tflops"] for row in rows)}


def generate_report(directory, documents, manifest):
    rows = aggregate(documents)
    first = documents[0]
    groups = {}
    for cache in sorted({row["cache"] for row in rows}):
        selected = [row for row in rows if row["cache"] == cache]
        groups[f"{cache}/all"] = group_summary(selected)
        for group in sorted({row["group"] for row in selected}):
            groups[f"{cache}/{group}"] = group_summary([row for row in selected if row["group"] == group])
    save_json(directory / "summary.json", {
        "schema_version": 1, "groups": groups, "cases": rows,
        "report_script_sha256": sha256(Path(__file__)),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })

    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["cache", "name", "group", "M", "N", "K", *[name + "_us" for name in IMPLEMENTATIONS],
                         "best_strict_blas_us", "custom_TFLOPS", "speedup_blas", "speedup_tiled", "max_custom_cv"])
        for row in rows:
            writer.writerow([row["cache"], row["name"], row["group"], row["m"], row["n"], row["k"],
                             *[row["latency_us"][name] for name in IMPLEMENTATIONS],
                             row["best_strict_blas_us"], row["tflops"], row["speedup_blas"],
                             row["speedup_tiled"], row["max_custom_cv"]])

    hardware = first["hardware"]
    software = first["software"]
    options = first["options"]
    lines = ["# FP32 GEMM 性能报告", "",
             f"GPU：{hardware['gpu']}，SM {hardware['sm_major']}.{hardware['sm_minor']}，"
             f"{hardware['multiprocessors']} SM，显存 {hardware['vram_bytes'] / 2**30:.2f} GiB，"
             f"L2 {hardware['l2_bytes'] / 2**20:.1f} MiB。", "",
             f"CUDA runtime {software['cuda_runtime']}，driver API {software['cuda_driver_api']}，"
             f"cuBLASLt {software['cublaslt_version']}，nvcc {software['cuda_compiler']}，"
             f"{software['build_type']} 构建。CPU/驱动快照和源码、二进制 SHA256 见 `manifest.json`。", "",
             f"完成 {len(documents)} 次独立进程运行，每进程每实现 {options['samples']} 组 event 样本，"
             f"目标每组 {options['target_ms']:g} ms，最多 {options['max_calls']} 次 GEMM；"
             f"每项预热至少 {options['warmup']} 次或一次完整轮换。seed 分别为 "
             + ", ".join(str(document["options"]["seed"]) for document in documents) + "。", "",
             "## 口径", "",
             "- `X[M,K] @ W[N,K]^T`，FP32 输入、累加、输出，alpha=1、beta=0，无 bias。",
             "- CUDA Graph + CUDA events 测 GPU 执行，不含分配、H2D、CPU 提交、handle/descriptor 初始化及算法选择。",
             "- 主流对照为严格 FP32 SGEMM 与严格 FP32 cuBLASLt 调优结果的逐 case 较快者。"
             f"二者使用 {options['workspace_mib']} MiB workspace；Lt 最多返回并实测 "
             f"{options['heuristic_candidates']} 个启发式候选，各用独立的 3 组校准样本选算法。",
             "- 原默认算法、零 workspace 的 cuBLASLt 仍单列为 legacy，主结论不使用它替代更快的严格 FP32 对照。",
             "- CUDA 分块基线为 16×16 block、K tile=16、每线程一个 FP32 累加器；"
             "使用共享内存，W 的 shared tile 加一列 padding。它不是未优化的朴素点积。",
             "- hot 表示重复使用同一组矩阵；rotating 轮换最多 256 组地址，目标工作集为 max(64 MiB, 2×L2)。"
             "未达到目标的 case 明确标注，轮换测试不宣称每次访问都为冷缓存。",
             "- 每进程先取 event 样本中位数，再取进程间中位数；对各 shape 加速比做等权几何平均。"
             "进程范围为每个独立进程几何平均的最小值和最大值，不是置信区间。",
             "- 每个 shape 各实现全量输出与 SGEMM 比较，并检查最多 1024 个独立 FP64 点积。"
             "容差为 |error| <= 2e-4 + 2e-4×|reference|。另通过 180 组生产 kernel 和 30 组分块基线检查。",
             "- 结果涵盖预先声明的完整 shape 集；不等价于整模型 prefill/decode 加速，且不与 TF32/FP16 Tensor Core 路径混比。", "",
             "## 汇总", "",
             "比值大于 1 表示自定义实现更快。", "",
             "| 访存/组别 | case 数 | 相对严格 BLAS 几何平均 | 进程范围 | 相对 16×16 分块 | 胜过 BLAS | 峰值 TFLOPS |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, summary in groups.items():
        lines.append(f"| {name} | {summary['cases']} | {summary['geomean_speedup_blas']:.3f}× | "
                     f"{summary['process_geomean_min']:.3f}–{summary['process_geomean_max']:.3f}× | "
                     f"{summary['geomean_speedup_tiled']:.3f}× | {summary['wins_blas']}/{summary['cases']} | "
                     f"{summary['peak_tflops']:.3f} |")
    lines += ["", "## 逐项数据", "", "所有延迟单位为 μs；Lt 为候选选择后的严格 FP32 cuBLASLt。", "",
              "| 访存 | shape M×N×K | 分组 | 自定义 | 16×16 分块 | SGEMM | Lt | legacy | 自定义 TFLOPS | BLAS/自定义 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        latency = row["latency_us"]
        values = [f"{latency[name]:.3f}" if latency[name] is not None else "unsupported" for name in IMPLEMENTATIONS]
        marker = "†" if row["cache"] == "rotating" and not row["target_reached"] else ""
        lines.append(f"| {row['cache']}{marker} | {row['m']}×{row['n']}×{row['k']} | {row['group']} | "
                     + " | ".join(values) + f" | {row['tflops']:.3f} | {row['speedup_blas']:.3f}× |")
    noisy = [row for row in rows if row["max_custom_cv"] > 0.05]
    lines += ["", "† 表示轮换地址的总工作集未达到预设目标；实际 slots/字节数见 summary.json。", "",
              "## 稳定性与正确性", "",
              f"自定义 kernel 全量对照的最大绝对误差：{max(row['max_custom_abs_error'] for row in rows):.6g}。"
              f"{len(noisy)} 个 case 至少有一轮自定义 event 样本的变异系数超过 5%；"
              "所有样本均保留，未删除慢样本。", ""]
    if noisy:
        lines.append("存在波动的项：" + "、".join(f"{row['cache']}/{row['name']}" for row in noisy) + "。")
        lines.append("")
    lines += ["所有实现的波动一并保留。下表按每项三个进程中的最大 event 样本变异系数统计；"
              "尤其不要将 hot 单行 GEMV 的个别高加速比外推为整模型 decode 收益。", "",
              "| 实现 | CV 超过 5% 的项数 | 最大 CV |",
              "| --- | ---: | ---: |"]
    for name in IMPLEMENTATIONS:
        cvs = [row["max_cv_by_implementation"][name] for row in rows
               if name in row["max_cv_by_implementation"]]
        if cvs:
            lines.append(f"| {name} | {sum(value > 0.05 for value in cvs)}/{len(cvs)} | {max(cvs):.1%} |")
    lines.append("")
    if manifest.get("source_changed_during_run"):
        lines += ["源码在测试过程中发生变化。结果仅对应 manifest 中记录的二进制；"
                  "重新构建并复测后再使用简历表述。", ""]
    lines += ["## 可用于简历的表述", "",
              "以下为完整测试范围的限定表述，不将最好的一项扩展为整体收益。", ""]
    for cache in sorted({row["cache"] for row in rows}):
        summary = groups[f"{cache}/all"]
        lines += [f"> 实现 FP32 CUDA GEMM/GEMV，采用寄存器分块、float4 向量化、共享内存双缓冲、"
                  f"寄存器预取与 XOR swizzle，支持 X@Wᵀ 并按 M 分派；在 {hardware['gpu']}、"
                  f"{summary['cases']} 种预定义尺寸的 {cache} 测试中，"
                  f"吞吐几何平均为 16×16 共享内存分块基线的 {summary['geomean_speedup_tiled']:.2f} 倍、"
                  f"严格 FP32 cuBLAS 对照的 {summary['geomean_speedup_blas'] * 100:.1f}%，"
                  f"峰值 {summary['peak_tflops']:.2f} TFLOPS；通过数值一致性检查。", ""]
    lines += ["## 原始证据与复现", "",
              "- `run-*.json`：每组原始耗时、采样顺序、算法校准、误差、调用次数与工作集。",
              "- `run-*.log`（归档为 `run-*.txt`）：进程输出。`manifest.json`：完整命令、系统、GPU 快照、git 状态、源码与二进制 SHA256。",
              "- `memcheck.txt` / `racecheck.txt`：归档的内存与同步检查日志。`summary.json` 另记录当前报告生成脚本的 SHA256。",
              "- `summary.csv` / `summary.json`：完整逐项与分组数据，可用表格工具继续分析。", "",
              "重跑命令（在仓库根目录）：", "", "```text",
              manifest.get("reproduction_command", "python exp/gemm/run_gemm.py --suite full --runs 3"), "```", "",
              "严格 FP32 数学模式依据 [NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)。", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return groups


def default_executable():
    candidates = [ROOT / "build/nano-vllm-cpp-comp-gemm",
                  ROOT / "build/windows/Release/nano-vllm-cpp-comp-gemm.exe",
                  ROOT / "build/Release/nano-vllm-cpp-comp-gemm.exe"]
    return next((path for path in candidates if path.exists()), candidates[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=default_executable())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-only", type=Path, help="Regenerate report from an existing results directory")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--cache", choices=("hot", "rotating", "both"), default="both")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--target-ms", type=float, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace-mib", type=int, default=32)
    parser.add_argument("--heuristic-candidates", type=int, default=8)
    args = parser.parse_args()
    if args.report_only:
        directory = args.report_only.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        generate_report(directory, load_runs(directory), manifest)
        print(directory / "report.md")
        return
    if args.runs < 1 or args.samples < 3 or args.target_ms < 1 or args.warmup < 1:
        parser.error("runs/warmup must be positive, samples >= 3, target-ms >= 1")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable not found: {executable}; build the comp-gemm target first")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = (args.output_dir or ROOT / "build/gemm-results" / stamp).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("run-*.json")):
        parser.error("output directory already contains measurements; use a fresh directory")
    manifest = {"schema_version": 1, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "platform": platform.platform(), "cpu": platform.processor(), "python": sys.version,
                "git_head": command_output(["git", "rev-parse", "HEAD"]),
                "git_status": command_output(["git", "status", "--short"]),
                "executable": str(executable), "executable_sha256": sha256(executable),
                "source_sha256": source_snapshot(), "gpu_before": gpu_snapshot(), "runs": [],
                "reproduction_command": subprocess.list2cmdline([sys.executable, "exp/gemm/run_gemm.py",
                    "--executable", str(executable), "--suite", args.suite, "--cache", args.cache,
                    "--runs", str(args.runs), "--samples", str(args.samples),
                    "--target-ms", str(args.target_ms), "--warmup", str(args.warmup),
                    "--seed", str(args.seed), "--workspace-mib", str(args.workspace_mib),
                    "--heuristic-candidates", str(args.heuristic_candidates)])}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        manifest["cpu_model"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                                      if line.startswith("model name")), "unknown")
    environment = os.environ.copy()
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    manifest["environment_overrides"] = {"NVIDIA_TF32_OVERRIDE": "0"}
    save_json(directory / "manifest.json", manifest)
    for index in range(args.runs):
        command = [str(executable), "--suite", args.suite, "--cache", args.cache,
                   "--output", str(directory / f"run-{index + 1:03}.json"),
                   "--samples", str(args.samples), "--target-ms", str(args.target_ms),
                   "--warmup", str(args.warmup), "--seed", str(args.seed + index),
                   "--workspace-mib", str(args.workspace_mib),
                   "--heuristic-candidates", str(args.heuristic_candidates)]
        record = {"command": command, "gpu_before": gpu_snapshot()}
        started = time.monotonic()
        print(f"Process {index + 1}/{args.runs}: {subprocess.list2cmdline(command)}", flush=True)
        with (directory / f"run-{index + 1:03}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as child:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                record["exit_code"] = child.wait()
        record["elapsed_seconds"] = time.monotonic() - started
        record["gpu_after"] = gpu_snapshot()
        manifest["runs"].append(record)
        save_json(directory / "manifest.json", manifest)
        if record["exit_code"]:
            raise RuntimeError(f"benchmark failed; see {directory / f'run-{index + 1:03}.log'}")
    manifest["source_sha256_after"] = source_snapshot()
    manifest["source_changed_during_run"] = manifest["source_sha256"] != manifest["source_sha256_after"]
    manifest["executable_sha256_after"] = sha256(executable)
    if manifest["executable_sha256_after"] != manifest["executable_sha256"]:
        raise RuntimeError("benchmark binary changed during measurement")
    manifest["gpu_after"] = gpu_snapshot()
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_json(directory / "manifest.json", manifest)
    groups = generate_report(directory, load_runs(directory), manifest)
    for name in sorted(key for key in groups if key.endswith("/all")):
        summary = groups[name]
        print(f"{name}: {summary['cases']} shapes; geometric mean "
              f"{summary['geomean_speedup_blas']:.3f}x vs strict BLAS, "
              f"{summary['geomean_speedup_tiled']:.3f}x vs tiled; "
              f"peak {summary['peak_tflops']:.3f} TFLOPS")
    print(f"Report: {directory / 'report.md'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
