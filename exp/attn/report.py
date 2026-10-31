"""Aggregate only matching tasks, precision, head configurations and cache modes."""

import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("positive finite ratios are required")
    return math.exp(statistics.mean(math.log(x) for x in values))


def row_key(row):
    return row["task"], row["case"]["name"], row["dtype"], row["cache"]


def aggregate(documents):
    if not documents:
        raise ValueError("no runs")
    keys = None
    metadata = None
    groups = collections.defaultdict(list)
    for document in documents:
        if (
            document["status"] != "complete"
            or len(document["rows"]) != document["expected_rows"]
        ):
            raise ValueError("an incomplete run cannot enter the report")
        options = {
            key: value for key, value in document["options"].items() if key != "seed"
        }
        identity = (
            document["hardware"],
            document["software"],
            document["contract"],
            options,
            document.get("library_sha256"),
            document.get("source_sha256"),
        )
        if document.get(
            "source_sha256_after", document.get("source_sha256")
        ) != document.get("source_sha256"):
            raise ValueError("benchmark source changed during a process run")
        if metadata is not None and metadata != identity:
            raise ValueError("hardware/software/contract mismatch")
        metadata = identity
        current = {row_key(row) for row in document["rows"]}
        if len(current) != len(document["rows"]) or (
            keys is not None and current != keys
        ):
            raise ValueError("case sets differ or contain duplicates")
        keys = current
        for row in document["rows"]:
            for measurement in row["implementations"].values():
                if measurement["status"] != "ok":
                    continue
                values = measurement["samples_us"]
                if len(values) != document["options"]["samples"] or any(
                    not math.isfinite(x) or x <= 0 for x in values
                ):
                    raise ValueError("invalid samples")
                if not math.isclose(
                    statistics.median(values), measurement["median_us"], rel_tol=1e-8
                ):
                    raise ValueError("median does not match raw samples")
                if measurement["correctness"]["max_tolerance_ratio"] > 1:
                    raise ValueError("incorrect output")
            groups[row_key(row)].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        prototype = rows[0]
        own_name = "custom"
        if any(
            row["implementations"].get(own_name, {}).get("status") != "ok"
            for row in rows
        ):
            raise ValueError("custom measurement missing")
        own = statistics.median(
            row["implementations"][own_name]["median_us"] for row in rows
        )
        names = set().union(*(set(row["implementations"]) for row in rows)) - {own_name}
        for name in sorted(names):
            measurements = [
                row["implementations"].get(name, {"status": "missing"}) for row in rows
            ]
            valid = all(measurement["status"] == "ok" for measurement in measurements)
            base = (
                statistics.median(m["median_us"] for m in measurements)
                if valid
                else None
            )
            case = prototype["case"]
            result.append(
                {
                    "task": key[0],
                    "name": key[1],
                    "dtype": key[2],
                    "cache": key[3],
                    "profile": case["profile"],
                    "phase": case["phase"],
                    "case": case,
                    "baseline": name,
                    "status": "ok" if valid else "unavailable",
                    "process_statuses": [m["status"] for m in measurements],
                    "reasons": sorted(
                        {
                            m.get("reason", "")
                            for m in measurements
                            if m["status"] != "ok"
                        }
                    ),
                    "custom_us": own,
                    "baseline_us": base,
                    "speedup": base / own if valid else None,
                    "process_ratios": [
                        m["median_us"] / row["implementations"][own_name]["median_us"]
                        for m, row in zip(measurements, rows)
                    ]
                    if valid
                    else [],
                    "custom_query_tokens_per_second": sum(case["query_lengths"])
                    / (own * 1e-6),
                    "max_custom_cv": max(
                        row["implementations"][own_name]["cv"] for row in rows
                    ),
                    "max_baseline_cv": max(m["cv"] for m in measurements)
                    if valid
                    else None,
                    "max_custom_abs_error": max(
                        row["implementations"][own_name]["correctness"]["max_abs_error"]
                        for row in rows
                    ),
                    "rotation_target_reached": all(
                        row["working_set"]["target_reached"] for row in rows
                    ),
                }
            )
    return result


def generate_report(directory):
    directory = Path(directory)
    paths = sorted([*directory.glob("run-*.json"), *directory.glob("run-*.json.gz")])
    documents = [
        json.loads(
            gzip.decompress(path.read_bytes())
            if path.suffix == ".gz"
            else path.read_text(encoding="utf-8")
        )
        for path in paths
    ]
    rows = aggregate(documents)
    grouped = collections.defaultdict(list)
    for row in rows:
        key = tuple(
            row[field]
            for field in ("task", "profile", "phase", "dtype", "cache", "baseline")
        )
        grouped[key].append(row)
    summaries = []
    for key, selected in sorted(grouped.items()):
        valid = [row for row in selected if row["status"] == "ok"]
        process_means = (
            [
                geometric_mean(row["process_ratios"][i] for row in valid)
                for i in range(len(documents))
            ]
            if valid
            else []
        )
        summaries.append(
            {
                **dict(
                    zip(("task", "profile", "phase", "dtype", "cache", "baseline"), key)
                ),
                "supported": len(valid),
                "total": len(selected),
                "geomean_speedup": geometric_mean(row["speedup"] for row in valid)
                if valid
                else None,
                "process_min": min(process_means) if valid else None,
                "process_max": max(process_means) if valid else None,
            }
        )
    summary = {
        "schema_version": 1,
        "groups": summaries,
        "cases": rows,
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        fields = (
            "task",
            "profile",
            "phase",
            "dtype",
            "cache",
            "name",
            "baseline",
            "status",
            "custom_us",
            "baseline_us",
            "speedup",
            "custom_query_tokens_per_second",
            "max_custom_cv",
            "max_baseline_cv",
            "rotation_target_reached",
        )
        writer.writerow([*fields, "query_lengths", "kv_lengths", "Hq", "Hkv", "D"])
        for row in rows:
            case = row["case"]
            writer.writerow(
                [
                    *[row[field] for field in fields],
                    json.dumps(case["query_lengths"]),
                    json.dumps(case["kv_lengths"]),
                    case["hq"],
                    case["hkv"],
                    case["dim"],
                ]
            )
    from plots import export_plots

    figures = export_plots(directory, rows)
    first = documents[0]
    options = first["options"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures = collections.Counter(
        status for row in rows for status in row["process_statuses"] if status != "ok"
    )
    noise = sum(
        row["status"] == "ok"
        and (row["max_custom_cv"] > 0.05 or row["max_baseline_cv"] > 0.05)
        for row in rows
    )
    lines = [
        "# Attention 性能比较",
        "",
        f"GPU：{first['hardware']['gpu']}，SM {first['hardware']['sm']}；"
        f"PyTorch {first['software']['packages']['torch']}，CUDA {first['software']['torch_cuda']}，"
        f"FlashInfer {first['software']['packages']['flashinfer-python']}，{first['software']['build_type']}。",
        "",
        f"{len(documents)} 个独立进程，seed="
        + ",".join(str(d["options"]["seed"]) for d in documents)
        + f"；每项每实现 {options['samples']} 组 CUDA event 样本，目标每批 {options['target_ms']} ms，"
        f"最多 {options['max_calls']} 次调用，预热至少 {options['warmup']} 次。",
        "",
        "自定义侧统一通过 kernels.hh 调用 src/kernels/cuda/attention.cu；实验目录不包含另一份 CUDA attention 算法。"
        "本轮未修改生产算法，不报告内部优化收益。",
        "",
        "## 计时与任务边界",
        "",
        "- pipeline：追加 KV + paged GQA attention。当前实现为融合调用；FlashInfer 包含 append 与 paged run。",
        "- prefill：同样调用原生产算子，使用 identity 页表。SDPA 对照计入 Torch index_copy_ 的 K/V 追加、"
        "相同 cache 零拷贝 dense view 上的 attention，以及写回 NHD 预分配输出的 packing/copy。",
        "- FP32/FP16/BF16、profile、phase、task、缓存条件分别统计，不合并百分比。"
        "scale=1/sqrt(D)，dropout=0，不含 RoPE、QK norm、线性投影或 gate。",
        "- Graph + events 测 GPU 执行；不包含分配、输入生成、H2D、plan/JIT、算法选择、"
        "正确性与 profiler。实际 backend、workspace、配置校准与误差见 run JSON。",
        "- reuse 复用一组矩阵；rotating 轮换地址，目标 max(64 MiB,2×L2)，最多 32 个 bank，"
        "受显存预算限制。未达到目标的项在 CSV 标注，轮换不保证每次访问均为冷缓存。",
        "- replay 固定长度、反复写同一批新 token 槽位，不在重复计时中增长 KV。所有实现使用同一非默认 stream。",
        "- 两个任务均包含 KV append，所有实现都先从干净 cache 验证全量 O、完整 cache 更新及哨兵、重复调用；"
        "O 对比禁用 TF32 的 FP32 SDPA-Math，并与独立 FP64 sampled rows 交叉验证。",
        "- FlashInfer 在计时外对 auto/fa2、适用的 SIMT/TC 配置进行独立校准；"
        "SDPA 强制单 backend，并记录 profiler 观察到的实际 op。没有把 Math fallback 算成 Flash。",
        "- 进程内取样本中位数，再取进程间中位数；逐 shape 比值等权取几何平均。"
        "进程范围不是置信区间，不删除慢样本。",
        "",
        "## 分组结果",
        "",
        "吞吐比例 = baseline_time / custom_time；100% 表示等速。覆盖率不满时只表示所列共同支持集合。",
        "",
        "| 任务 | Profile | 阶段 | dtype | 缓存 | 对照 | 覆盖率 | 吞吐比例 | 进程范围 |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for group in summaries:
        ratio = group["geomean_speedup"]
        value = f"{ratio * 100:.2f}%" if ratio is not None else "—"
        span = (
            f"{group['process_min'] * 100:.2f}–{group['process_max'] * 100:.2f}%"
            if ratio is not None
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                str(group[field])
                for field in ("task", "profile", "phase", "dtype", "cache", "baseline")
            )
            + f" | {group['supported']}/{group['total']} | {value} | {span} |"
        )
    for filename in figures:
        lines += ["", f"![{filename}]({filename})", ""]
    lines += [
        "",
        "## 完整数据与限制",
        "",
        "逐项延迟、吞吐比、误差、变异系数、工作集与所有 unavailable 记录见 `summary.csv`、"
        "`summary.json` 和 `run-*.json`（归档可为 .json.gz）。进程日志为 `run-*.txt`；版本、命令、GPU 快照和 SHA256 在 `manifest.json`。",
        "",
        f"基线非成功状态计数（逐 case、逐进程）：{dict(failures)}。"
        f"{noise} 个有效对照项至少有一端的样本 CV 超过 5%，均保留。",
        "",
        "FP64 误差检验使用完整 K 维；长 Q 采样固定行，全量 O 另与 FP32 数学参考比较。"
        "所有容差来自 suite.json，没有为保留快结果而放宽。",
        "",
        "query tok/s 只表示 attention 算子的输入 query 数 / 时间，不是模型生成速度。"
        "本报告没有端到端 prefill/decode 加速结论。",
        "",
    ]
    if manifest.get("source_changed_during_run"):
        lines += [
            "源码在进程运行期间发生变化，结果只对应记录的二进制。正式引用前应核对测量代码。",
            "",
        ]
    lines += [
        "## 简历用语",
        "",
        "必须选择一行完整限定的任务/精度/头配置/缓存条件，并注明覆盖的 shape 数，不能把"
        "FP32 Math、FP16 Flash 和 paged decode 的结果混在一起。可用模板：",
        "",
        "> 实现分页 GQA attention，支持 KV cache 更新与变长 batch；在 RTX 5060 Ti、"
        "[dtype、profile、phase、cache、shape 集] 下，吞吐几何平均达到同精度 "
        "[FlashInfer append+attention / Torch append+SDPA Flash+NHD输出] 对照的 Y%，"
        "并通过输出及 cache 更新一致性验证。",
        "",
        "数值 Y 取上表对应分组的吞吐比例。本轮没有修改生产算法，因此不写“相较原始实现降低 X%”。",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
