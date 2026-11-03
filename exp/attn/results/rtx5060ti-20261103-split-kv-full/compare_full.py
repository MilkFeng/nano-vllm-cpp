"""Compare the completed 408-case full suite with the preceding full run."""

import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'exp/attn/run_attn.py').is_file())
PREVIOUS = ROOT / 'exp/attn/results/rtx5060ti-20261103'
CURRENT = Path(__file__).resolve().parent


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def unique(summary):
    result = {}
    for row in summary['cases']:
        key = tuple(row[f] for f in ('task', 'name', 'dtype', 'cache'))
        if key in result:
            assert result[key]['custom_us'] == row['custom_us']
        result[key] = row
    return result


def main():
    old_summary, new_summary = read(PREVIOUS / 'summary.json'), read(CURRENT / 'summary.json')
    old, new = unique(old_summary), unique(new_summary)
    assert old.keys() == new.keys() and len(new) == 408
    documents = [read(p) for p in sorted([*CURRENT.glob('run-*.json'), *CURRENT.glob('run-*.json.gz')])]
    assert len(documents) == 3
    manifest, build = read(CURRENT / 'manifest.json'), read(CURRENT / 'build-info.json')
    assert not manifest['source_changed_during_run']
    assert manifest['library_sha256'] == manifest['library_sha256_after'] == build['library_sha256']
    for d in documents:
        assert d['status'] == 'complete' and len(d['rows']) == d['expected_rows'] == 408
        assert d['source_sha256'] == d['source_sha256_after'] == manifest['source_sha256']
        assert d['library_sha256'] == manifest['library_sha256']
    for name, expected in build['extra_source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(ROOT / 'build/libnano-vllm-cpp-attn-exp.so') == build['library_sha256']
    records = []
    fields = ('task', 'profile', 'phase', 'dtype', 'cache')
    for key in sorted(new):
        before, after = old[key], new[key]
        assert before['case'] == after['case']
        records.append({**{f: after[f] for f in (*fields, 'name')},
                        'before_us': before['custom_us'], 'after_us': after['custom_us'],
                        'old_over_new': before['custom_us'] / after['custom_us']})
    grouped = collections.defaultdict(list)
    for r in records:
        grouped[tuple(r[f] for f in fields)].append(r)
    groups = [{**dict(zip(fields, k)), 'shapes': len(rows),
               'geomean_old_over_new': geomean(r['old_over_new'] for r in rows)}
              for k, rows in sorted(grouped.items())]
    with (CURRENT / 'comparison.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (CURRENT / 'comparison.json').write_text(json.dumps({'groups': groups, 'cases': records}, ensure_ascii=False, indent=2) + '\n')

    def baseline(summary, profile, phase, task, name, cache='reuse'):
        return next(r for r in summary['groups'] if r['profile'] == profile and r['phase'] == phase
                    and r['task'] == task and r['baseline'] == name and r['dtype'] == 'bfloat16' and r['cache'] == cache)

    lines = ['# Split-KV 完整 attention 测试', '',
             '每进程 408 项，3 个独立进程，共 1224 项；FP32/FP16/BF16、reuse/rotating，包含 pipeline 和连续 prefill。',
             '使用原有 run_attn.py --suite full，无 shape 或 baseline 过滤；每项每实现 15 组样本，seed=42/43/44。',
             '包含 KV 追加，Split-KV 的计算与归并均计时；plan/JIT、分配、校验和 Graph 构建在计时外。', '',
             '## BF16 / reuse：对照吞吐比例', '',
             '| 配置 | Decode / FlashInfer（12） | Paged prefill / FlashInfer（6） | 连续 prefill / SDPA Flash（6） |',
             '| --- | ---: | ---: | ---: |']
    for profile in ('qwen3_0p6b', 'qwen35_full_attention'):
        values = []
        for phase, task, name in (('decode', 'pipeline', 'flashinfer'), ('prefill', 'pipeline', 'flashinfer'), ('prefill', 'prefill', 'sdpa_flash')):
            group = baseline(new_summary, profile, phase, task, name)
            value = f"{group['geomean_speedup']*100:.2f}%" if group['geomean_speedup'] is not None else '—'
            values.append(f"{value} ({group['supported']}/{group['total']})")
        lines.append('| ' + profile + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '吞吐比例 = baseline_time / project_time × 100%，100% 为等速；各任务、精度和缓存条件分别统计。', '',
              '## 与上一轮单 block 完整记录对比：BF16 / reuse', '',
              '| 配置 | 阶段 | shapes | 匹配 shape 的旧/新延迟几何平均 | 对照吞吐比例：旧 → 新 |',
              '| --- | --- | ---: | ---: | ---: |']
    for profile in ('qwen3_0p6b', 'qwen35_full_attention'):
        for phase, task, name in (('decode', 'pipeline', 'flashinfer'), ('prefill', 'prefill', 'sdpa_flash')):
            speed = next(g for g in groups if g['profile'] == profile and g['phase'] == phase
                         and g['task'] == task and g['dtype'] == 'bfloat16' and g['cache'] == 'reuse')
            before, after = baseline(old_summary, profile, phase, task, name), baseline(new_summary, profile, phase, task, name)
            fmt = lambda r: f"{r['geomean_speedup']*100:.2f}%" if r['geomean_speedup'] is not None else '—'
            lines.append(f"| {profile} | {phase} | {speed['shapes']} | {speed['geomean_old_over_new']:.2f}× | "
                         f"{fmt(before)} → {fmt(after)} |")
    lines += ['', '上一完整记录为 ../rtx5060ti-20261103。这是两次完整测试的对比；频率、系统负载、缓存和 baseline 选择会波动。',
              '同进程交错的 Split-KV 消融见 ../rtx5060ti-20261103-split-kv；两类结果保留各自的口径。Prefill 生产计算路径未改，其变化不归因于 decode 分片。', '',
              '## 完整结果', '', '[所有精度、阶段、缓存和 baseline 分组](report.md) · [逐项 baseline 延迟 CSV](summary.csv) · [匹配上一完整记录的 CSV](comparison.csv)', '',
              '![BF16 batch=1 decode](decode_latency.png)', '', '![BF16 batch=1 prefill](prefill_latency.png)', '',
              '## 正确性与完整性', '',
              '- 全部 1224 个项目侧测量项均通过全量输出、完整 KV cache 更新、哨兵和重复调用检查；容差保持不变。',
              '- FP32 的 FlashInfer 等不支持项保留原状态和原因，不转换精度，也不纳入不匹配的基线均值。',
              '- 运行前后记录的源码与共享库 SHA256 一致，额外核对 API/header/模型调用方；版本与 GPU 快照见 manifest.json，编译参数见 build-info.json。',
              '- 原始 15 组样本、backend 选择、误差、CV、工作集和未支持项完整保存在 run-*.json（归档为 .json.gz）。rotating 不保证全冷缓存。',
              '- 算子时间包含 KV 追加；这些不是模型端到端生成 tok/s。', '']
    (CURRENT / 'comparison.md').write_text('\n'.join(lines))
    tolerance = max(r['implementations']['custom']['correctness']['max_tolerance_ratio'] for d in documents for r in d['rows'])
    validation = {'status': 'ok', 'runs': 3, 'cases_per_run': 408, 'total_cases': 1224,
                  'max_custom_tolerance_ratio': tolerance,
                  'source_and_library_unchanged': True, 'extra_production_sources_match': True,
                  'kernel_sha256': build['source_sha256'], 'library_sha256': build['library_sha256'],
                  'matched_previous_cases': len(records),
                  'compare_script_sha256': sha(Path(__file__)),
                  'artifacts_sha256': {n: sha(CURRENT / n) for n in ('summary.json', 'summary.csv', 'comparison.json', 'comparison.csv')}}
    (CURRENT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'validation': validation, 'BF16_reuse': [g for g in groups if g['dtype'] == 'bfloat16' and g['cache'] == 'reuse']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
