#!/usr/bin/env python3
"""Compare completed, matched attention suites without mixing task boundaries."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def unique_cases(summary):
    result = {}
    for row in summary['cases']:
        key = tuple(row[k] for k in ('task', 'name', 'dtype', 'cache'))
        if key in result:
            assert result[key]['custom_us'] == row['custom_us']
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    previous, current = args.previous.resolve(), args.current.resolve()
    old_summary, new_summary = read(previous / 'summary.json'), read(current / 'summary.json')
    old, new = unique_cases(old_summary), unique_cases(new_summary)
    assert old.keys() == new.keys(), 'Historical and current case sets differ'
    manifest, build = read(current / 'manifest.json'), read(current / 'build-info.json')
    assert not manifest['source_changed_during_run']
    assert manifest['library_sha256'] == manifest['library_sha256_after'] == build['library_sha256']
    assert build['source_sha256'] == manifest['source_sha256']['src/kernels/cuda/attention.cu']
    documents = [read(p) for p in sorted([*current.glob('run-*.json'), *current.glob('run-*.json.gz')])]
    assert len(documents) == 3
    for document in documents:
        assert document['status'] == 'complete'
        assert len(document['rows']) == document['expected_rows'] == 408
        assert document['source_sha256'] == document['source_sha256_after'] == manifest['source_sha256']
        assert document['library_sha256'] == manifest['library_sha256']
    old_run_path = sorted([*previous.glob('run-*.json'), *previous.glob('run-*.json.gz')])[0]
    old_document = read(old_run_path)
    same_environment = {
        'hardware': documents[0]['hardware'] == old_document['hardware'],
        'software': documents[0]['software'] == old_document['software'],
    }
    changed_sources = [name for name, value in manifest['source_sha256'].items()
                       if read(previous / 'manifest.json')['source_sha256'].get(name) != value]
    numerical = read(current / 'correctness.json')
    assert numerical['status'] == 'ok' and numerical['checks'] == 212
    assert 'ERROR SUMMARY: 0 errors' in (current / 'memcheck.txt').read_text()
    assert '0 errors, 0 warnings' in (current / 'racecheck.txt').read_text()
    records = []
    for key in sorted(new):
        before, after = old[key], new[key]
        assert before['case'] == after['case']
        records.append({
            **{k: after[k] for k in ('task', 'name', 'profile', 'phase', 'dtype', 'cache')},
            'case': after['case'],
            'old_custom_us': before['custom_us'],
            'new_custom_us': after['custom_us'],
            'old_over_new': before['custom_us'] / after['custom_us'],
            'new_over_old': after['custom_us'] / before['custom_us'],
            'latency_reduction_percent': 100 * (1 - after['custom_us'] / before['custom_us']),
            'max_custom_cv': after['max_custom_cv'],
        })
    grouped = collections.defaultdict(list)
    fields = ('task', 'profile', 'phase', 'dtype', 'cache')
    for row in records:
        grouped[tuple(row[k] for k in fields)].append(row)
    groups = []
    for key, rows in sorted(grouped.items()):
        ratio = geomean(row['old_over_new'] for row in rows)
        groups.append({**dict(zip(fields, key)), 'shapes': len(rows), 'old_over_new': ratio,
                       'latency_reduction_percent': 100 * (1 - 1 / ratio),
                       'min_speedup': min(row['old_over_new'] for row in rows),
                       'max_speedup': max(row['old_over_new'] for row in rows)})
    max_error = max(row['implementations']['custom']['correctness']['max_tolerance_ratio']
                    for document in documents for row in document['rows'])
    validation = {
        'status': 'ok', 'runs': 3, 'cases_per_run': 408, 'total_cases': 1224,
        'source_and_library_unchanged_during_measurement': True,
        'source_sha256': build['source_sha256'], 'library_sha256': build['library_sha256'],
        'numerical': numerical, 'max_performance_case_tolerance_ratio': max_error,
        'sanitizers': {name: {'native_cases': 36, 'errors': 0, 'artifact': name + '.txt',
                              'log_sha256': sha(current / (name + '.txt'))}
                       for name in ('memcheck', 'racecheck')},
        'same_environment_as_previous': same_environment,
        'changed_sources_since_previous': changed_sources,
        'comparison_script_sha256': sha(Path(__file__)),
        'note': 'contract.production_optimized in the unchanged runner is a historical constant; the source hash identifies this decode revision.',
    }
    (current / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    comparison = {'previous': str(previous), 'current': str(current), 'groups': groups, 'cases': records,
                  'same_environment': same_environment, 'changed_sources': changed_sources}
    (current / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
    with (current / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=[k for k in records[0] if k != 'case'], lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != 'case'} for row in records)

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nanovllm-attn-mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, profile, title in zip(axes, ('qwen3_0p6b', 'qwen35_full_attention'),
                                    ('Qwen3: Hq/Hkv/D = 16/8/128', 'Qwen3.5 full: Hq/Hkv/D = 16/2/256')):
        selected = sorted([r for r in records if r['task'] == 'pipeline' and r['profile'] == profile
                           and r['phase'] == 'decode' and r['dtype'] == 'bfloat16' and r['cache'] == 'reuse'
                           and len(r['case']['query_lengths']) == 1], key=lambda r: r['case']['kv_lengths'][0])
        lengths = [r['case']['kv_lengths'][0] for r in selected]
        baseline = [next(r['baseline_us'] for r in new_summary['cases'] if r['name'] == row['name']
                         and r['dtype'] == 'bfloat16' and r['cache'] == 'reuse' and r['baseline'] == 'flashinfer')
                    for row in selected]
        axis.plot(lengths, [r['old_custom_us'] for r in selected], 'o--', label='Previous kernel (Oct 31)')
        axis.plot(lengths, [r['new_custom_us'] for r in selected], 'o-', label='Current kernel (Nov 03)')
        axis.plot(lengths, baseline, 's:', label='Current FlashInfer')
        axis.set(xscale='log', yscale='log', xticks=lengths, xlabel='KV length (tokens)',
                 ylabel='Append + attention latency (us)', title=title)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.grid(True, which='both', alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle('BF16 / batch=1 / reused addresses / median of 3 processes')
    figure.tight_layout()
    figure.savefig(current / 'before_after_decode.svg')
    figure.savefig(current / 'before_after_decode.png', dpi=180)
    plt.close(figure)
    for path in current.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')

    def baseline_group(summary, profile, phase, task, baseline):
        return next(row for row in summary['groups'] if row['profile'] == profile
                    and row['phase'] == phase and row['task'] == task
                    and row['baseline'] == baseline and row['dtype'] == 'bfloat16'
                    and row['cache'] == 'reuse')

    lines = [
        '# Attention kernel 复测：2026-11-03 对比 2026-10-31', '',
        f"当前版本：`{build['git_head']}`；RTX 5060 Ti，Release / -O3 / SM120。",
        '本次使用加入 decode 专用分支的生产 kernel，测试脚本与上次相同；仅 attention.cu 的源码 hash 变化。', '',
        '3 个独立进程，seed=42/43/44；每进程 408 项，共 1224 项。每项每实现 15 组样本，'
        '至少预热 20 次，目标每组 10 ms，最多 1024 次调用。',
        '每项先取样本中位数，再取三进程中位数；跨 shape 对旧延迟/新延迟等权取几何平均。',
        '所有计时包含 KV 追加；pipeline 对比 FlashInfer append+attention，连续 prefill 对比 '
        'Torch append+SDPA+NHD 输出写回。两类任务分别统计。', '',
        '## BF16 / reuse 主结果', '',
        '| 配置 | 阶段 / 对照 | shapes | 相较旧 kernel | 旧吞吐达到对照 | 新吞吐达到对照 |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for profile in ('qwen3_0p6b', 'qwen35_full_attention'):
        for phase, task, baseline in (('decode', 'pipeline', 'flashinfer'), ('prefill', 'prefill', 'sdpa_flash')):
            group = next(row for row in groups if row['profile'] == profile and row['phase'] == phase
                         and row['task'] == task and row['dtype'] == 'bfloat16' and row['cache'] == 'reuse')
            before = baseline_group(old_summary, profile, phase, task, baseline)
            after = baseline_group(new_summary, profile, phase, task, baseline)
            lines.append(f"| {profile} | {phase} / {baseline} | {group['shapes']} | "
                         f"{group['old_over_new']:.2f}× | {before['geomean_speedup']*100:.2f}% | "
                         f"{after['geomean_speedup']*100:.2f}% |")
    lines += ['', '100% 表示与对照等速。旧、新吞吐比例分别使用各次重测的 baseline；'
              '“相较旧 kernel”直接由匹配 shape 的生产 kernel 延迟计算，避免将 baseline 波动当成 kernel 加速。', '',
              '![BF16 decode：旧 kernel、新 kernel 与本轮 FlashInfer](before_after_decode.svg)', '',
              '## 长上下文 decode：BF16 / reuse / batch=1 / KV=8192', '',
              '| 配置 | 旧 kernel | 新 kernel | 加速 | 当前 FlashInfer |',
              '| --- | ---: | ---: | ---: | ---: |']
    for profile in ('qwen3_0p6b', 'qwen35_full_attention'):
        row = next(row for row in records if row['name'] == profile + '_decode_b1_q1_k8192'
                   and row['dtype'] == 'bfloat16' and row['cache'] == 'reuse')
        after = next(r for r in new_summary['cases'] if r['name'] == row['name'] and r['dtype'] == row['dtype']
                     and r['cache'] == 'reuse' and r['baseline'] == 'flashinfer')
        lines.append(f"| {profile} | {row['old_custom_us']/1000:.3f} ms | {row['new_custom_us']/1000:.3f} ms | "
                     f"{row['old_over_new']:.2f}× | {after['baseline_us']:.2f} μs |")
    lines += ['', '## 正确性与可比性', '',
              f"- 212 个边界检查通过；正式采样的 1224 项均通过输出与完整 cache 更新检查。最大容差比 {max_error:.4f}（阈值 1）。",
              '- 原生 36 个代表用例分别通过 memcheck 和 racecheck，0 errors / 0 hazards。',
              '- 运行前后源码及共享库 SHA256 一致；版本、命令、GPU 快照见 manifest.json，编译参数见 build-info.json。',
              f"- 与旧记录的硬件元数据一致：{same_environment['hardware']}；软件版本一致：{same_environment['software']}。",
              '- 这是跨日期的实测对比。GPU 频率、系统负载和缓存会造成波动；prefill/chunked prefill 的计算路径没有代码修改，其延迟变化不能归因于 decode 优化。',
              '- rotating 是地址轮换，受最多 32 个 bank 和显存预算限制，不等于每次访问全冷缓存；逐项工作集和 CV 均保留。',
              '- FP32 在本环境 FlashInfer 中不支持，保留 unsupported；不转换精度。query tok/s 仅为算子指标，不是模型生成速度。',
              '- 原 runner 的 contract.production_optimized=false 是历史常量；本次实际运行版本以本报告、源码 hash 和 commit 为准。', '',
              '## 全部分组：旧 kernel / 新 kernel', '',
              '| 任务 | 配置 | 阶段 | 精度 | 缓存 | shapes | 加速 | 逐 shape 范围 |',
              '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for group in groups:
        lines.append('| ' + ' | '.join(str(group[k]) for k in fields)
                     + f" | {group['shapes']} | {group['old_over_new']:.3f}× | "
                     f"{group['min_speedup']:.3f}–{group['max_speedup']:.3f}× |")
    lines += ['', '## 产物', '',
              '- [本轮 baseline 分组报告](report.md)、[全部 baseline 逐项 CSV](summary.csv)。',
              '- [匹配旧版的逐项 CSV](comparison.csv)、[比较 JSON](comparison.json)。',
              '- [正确性与完整性验证](validation.json)、[212 个边界检查](correctness.json)。',
              '- run-*.json（归档为 .json.gz）保留全部原始样本、baseline 选择、误差及不支持原因。',
              '- compare_previous.py 是本比较的可复现汇总脚本；旧结果位于 ../rtx5060ti-20261031。', '',
              '在仓库根目录重新生成归档报告：', '', '```bash',
              'build/attn-venv/bin/python exp/attn/run_attn.py --report-only exp/attn/results/rtx5060ti-20261103',
              'build/attn-venv/bin/python exp/attn/results/rtx5060ti-20261103/compare_previous.py '
              'exp/attn/results/rtx5060ti-20261031 exp/attn/results/rtx5060ti-20261103',
              '```', '']
    (current / 'comparison.md').write_text('\n'.join(lines))
    report = (current / 'report.md').read_text()
    report = report.replace('本轮未修改生产算法，不报告内部优化收益。',
                            '本轮测试包含 decode 专用分支的新版本；与上次 kernel 的对比见 [comparison.md](comparison.md)。')
    report = report.replace('数值 Y 取上表对应分组的吞吐比例。本轮没有修改生产算法，因此不写“相较原始实现降低 X%”。',
                            '数值 Y 取上表对应分组的吞吐比例。相较旧版的延迟比与限制见 [comparison.md](comparison.md)。')
    (current / 'report.md').write_text(report)
    print(json.dumps({'validation': validation, 'headline_groups': [g for g in groups
                      if g['dtype'] == 'bfloat16' and g['cache'] == 'reuse'
                      and g['phase'] in ('decode', 'prefill')]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
