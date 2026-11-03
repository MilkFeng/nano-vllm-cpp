# Attention 性能比较

GPU：NVIDIA GeForce RTX 5060 Ti，SM [12, 0]；PyTorch 2.11.0，CUDA 13.0，FlashInfer 0.7.2.post1，Release。

3 个独立进程，seed=42,43,44；每项每实现 15 组 CUDA event 样本，目标每批 10.0 ms，最多 1024 次调用，预热至少 20 次。

自定义侧统一通过 kernels.hh 调用 src/kernels/cuda/attention.cu；实验目录不包含另一份 CUDA attention 算法。具体 kernel 版本由 manifest 中的源码与共享库 SHA256 标识；历史版本的内部加速需另做匹配对比。

## 计时与任务边界

- pipeline：追加 KV + paged GQA attention。当前实现为融合调用；FlashInfer 包含 append 与 paged run。
- prefill：同样调用原生产算子，使用 identity 页表。SDPA 对照计入 Torch index_copy_ 的 K/V 追加、相同 cache 零拷贝 dense view 上的 attention，以及写回 NHD 预分配输出的 packing/copy。
- FP32/FP16/BF16、profile、phase、task、缓存条件分别统计，不合并百分比。scale=1/sqrt(D)，dropout=0，不含 RoPE、QK norm、线性投影或 gate。
- Graph + events 测 GPU 执行；不包含分配、输入生成、H2D、plan/JIT、算法选择、正确性与 profiler。实际 backend、workspace、配置校准与误差见 run JSON。
- reuse 复用一组矩阵；rotating 轮换地址，目标 max(64 MiB,2×L2)，最多 32 个 bank，受显存预算限制。未达到目标的项在 CSV 标注，轮换不保证每次访问均为冷缓存。
- replay 固定长度、反复写同一批新 token 槽位，不在重复计时中增长 KV。所有实现使用同一非默认 stream。
- 两个任务均包含 KV append，所有实现都先从干净 cache 验证全量 O、完整 cache 更新及哨兵、重复调用；O 对比禁用 TF32 的 FP32 SDPA-Math，并与独立 FP64 sampled rows 交叉验证。
- FlashInfer 在计时外对 auto/fa2、适用的 SIMT/TC 配置进行独立校准；SDPA 强制单 backend，并记录 profiler 观察到的实际 op。没有把 Math fallback 算成 Flash。
- 进程内取样本中位数，再取进程间中位数；逐 shape 比值等权取几何平均。进程范围不是置信区间，不删除慢样本。

## 分组结果

吞吐比例 = baseline_time / custom_time；100% 表示等速。覆盖率不满时只表示所列共同支持集合。

| 任务 | Profile | 阶段 | dtype | 缓存 | 对照 | 覆盖率 | 吞吐比例 | 进程范围 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| pipeline | qwen35_full_attention | chunked_prefill | bfloat16 | reuse | flashinfer | 8/8 | 1.42% | 1.37–1.45% |
| pipeline | qwen35_full_attention | chunked_prefill | bfloat16 | rotating | flashinfer | 8/8 | 1.42% | 1.39–1.42% |
| pipeline | qwen35_full_attention | chunked_prefill | float16 | reuse | flashinfer | 8/8 | 1.40% | 1.38–1.43% |
| pipeline | qwen35_full_attention | chunked_prefill | float16 | rotating | flashinfer | 8/8 | 1.41% | 1.39–1.42% |
| pipeline | qwen35_full_attention | chunked_prefill | float32 | reuse | flashinfer | 0/8 | — | — |
| pipeline | qwen35_full_attention | chunked_prefill | float32 | rotating | flashinfer | 0/8 | — | — |
| pipeline | qwen35_full_attention | decode | bfloat16 | reuse | flashinfer | 12/12 | 8.47% | 8.24–8.41% |
| pipeline | qwen35_full_attention | decode | bfloat16 | rotating | flashinfer | 12/12 | 8.68% | 8.66–8.73% |
| pipeline | qwen35_full_attention | decode | float16 | reuse | flashinfer | 12/12 | 8.57% | 8.20–8.73% |
| pipeline | qwen35_full_attention | decode | float16 | rotating | flashinfer | 12/12 | 8.56% | 8.26–8.66% |
| pipeline | qwen35_full_attention | decode | float32 | reuse | flashinfer | 0/12 | — | — |
| pipeline | qwen35_full_attention | decode | float32 | rotating | flashinfer | 0/12 | — | — |
| pipeline | qwen35_full_attention | prefill | bfloat16 | reuse | flashinfer | 6/6 | 2.03% | 2.01–2.02% |
| pipeline | qwen35_full_attention | prefill | bfloat16 | rotating | flashinfer | 6/6 | 2.17% | 2.14–2.17% |
| pipeline | qwen35_full_attention | prefill | float16 | reuse | flashinfer | 6/6 | 2.04% | 2.04–2.06% |
| pipeline | qwen35_full_attention | prefill | float16 | rotating | flashinfer | 6/6 | 2.18% | 2.18–2.21% |
| pipeline | qwen35_full_attention | prefill | float32 | reuse | flashinfer | 0/6 | — | — |
| pipeline | qwen35_full_attention | prefill | float32 | rotating | flashinfer | 0/6 | — | — |
| pipeline | qwen35_full_attention | ragged_chunked_prefill | bfloat16 | reuse | flashinfer | 1/1 | 1.24% | 1.18–1.25% |
| pipeline | qwen35_full_attention | ragged_chunked_prefill | bfloat16 | rotating | flashinfer | 1/1 | 1.22% | 1.19–1.23% |
| pipeline | qwen35_full_attention | ragged_chunked_prefill | float16 | reuse | flashinfer | 1/1 | 1.19% | 1.18–1.25% |
| pipeline | qwen35_full_attention | ragged_chunked_prefill | float16 | rotating | flashinfer | 1/1 | 1.21% | 1.18–1.23% |
| pipeline | qwen35_full_attention | ragged_chunked_prefill | float32 | reuse | flashinfer | 0/1 | — | — |
| pipeline | qwen35_full_attention | ragged_chunked_prefill | float32 | rotating | flashinfer | 0/1 | — | — |
| pipeline | qwen35_full_attention | ragged_decode | bfloat16 | reuse | flashinfer | 1/1 | 2.73% | 2.61–2.78% |
| pipeline | qwen35_full_attention | ragged_decode | bfloat16 | rotating | flashinfer | 1/1 | 4.68% | 4.67–4.82% |
| pipeline | qwen35_full_attention | ragged_decode | float16 | reuse | flashinfer | 1/1 | 2.56% | 2.49–2.67% |
| pipeline | qwen35_full_attention | ragged_decode | float16 | rotating | flashinfer | 1/1 | 4.47% | 4.44–4.72% |
| pipeline | qwen35_full_attention | ragged_decode | float32 | reuse | flashinfer | 0/1 | — | — |
| pipeline | qwen35_full_attention | ragged_decode | float32 | rotating | flashinfer | 0/1 | — | — |
| pipeline | qwen3_0p6b | chunked_prefill | bfloat16 | reuse | flashinfer | 8/8 | 2.66% | 2.60–2.65% |
| pipeline | qwen3_0p6b | chunked_prefill | bfloat16 | rotating | flashinfer | 8/8 | 2.79% | 2.74–2.84% |
| pipeline | qwen3_0p6b | chunked_prefill | float16 | reuse | flashinfer | 8/8 | 2.65% | 2.60–2.63% |
| pipeline | qwen3_0p6b | chunked_prefill | float16 | rotating | flashinfer | 8/8 | 2.85% | 2.82–2.85% |
| pipeline | qwen3_0p6b | chunked_prefill | float32 | reuse | flashinfer | 0/8 | — | — |
| pipeline | qwen3_0p6b | chunked_prefill | float32 | rotating | flashinfer | 0/8 | — | — |
| pipeline | qwen3_0p6b | decode | bfloat16 | reuse | flashinfer | 12/12 | 16.31% | 15.75–16.47% |
| pipeline | qwen3_0p6b | decode | bfloat16 | rotating | flashinfer | 12/12 | 20.30% | 19.93–20.49% |
| pipeline | qwen3_0p6b | decode | float16 | reuse | flashinfer | 12/12 | 15.56% | 14.96–15.55% |
| pipeline | qwen3_0p6b | decode | float16 | rotating | flashinfer | 12/12 | 20.82% | 20.53–21.12% |
| pipeline | qwen3_0p6b | decode | float32 | reuse | flashinfer | 0/12 | — | — |
| pipeline | qwen3_0p6b | decode | float32 | rotating | flashinfer | 0/12 | — | — |
| pipeline | qwen3_0p6b | prefill | bfloat16 | reuse | flashinfer | 6/6 | 3.36% | 3.30–3.42% |
| pipeline | qwen3_0p6b | prefill | bfloat16 | rotating | flashinfer | 6/6 | 3.71% | 3.69–3.70% |
| pipeline | qwen3_0p6b | prefill | float16 | reuse | flashinfer | 6/6 | 3.35% | 3.32–3.37% |
| pipeline | qwen3_0p6b | prefill | float16 | rotating | flashinfer | 6/6 | 3.74% | 3.67–3.80% |
| pipeline | qwen3_0p6b | prefill | float32 | reuse | flashinfer | 0/6 | — | — |
| pipeline | qwen3_0p6b | prefill | float32 | rotating | flashinfer | 0/6 | — | — |
| pipeline | qwen3_0p6b | ragged_chunked_prefill | bfloat16 | reuse | flashinfer | 1/1 | 3.05% | 2.94–3.19% |
| pipeline | qwen3_0p6b | ragged_chunked_prefill | bfloat16 | rotating | flashinfer | 1/1 | 2.96% | 2.96–3.02% |
| pipeline | qwen3_0p6b | ragged_chunked_prefill | float16 | reuse | flashinfer | 1/1 | 3.00% | 2.81–3.06% |
| pipeline | qwen3_0p6b | ragged_chunked_prefill | float16 | rotating | flashinfer | 1/1 | 2.99% | 2.99–3.00% |
| pipeline | qwen3_0p6b | ragged_chunked_prefill | float32 | reuse | flashinfer | 0/1 | — | — |
| pipeline | qwen3_0p6b | ragged_chunked_prefill | float32 | rotating | flashinfer | 0/1 | — | — |
| pipeline | qwen3_0p6b | ragged_decode | bfloat16 | reuse | flashinfer | 1/1 | 14.15% | 13.67–14.40% |
| pipeline | qwen3_0p6b | ragged_decode | bfloat16 | rotating | flashinfer | 1/1 | 13.69% | 13.44–14.07% |
| pipeline | qwen3_0p6b | ragged_decode | float16 | reuse | flashinfer | 1/1 | 14.42% | 14.18–14.84% |
| pipeline | qwen3_0p6b | ragged_decode | float16 | rotating | flashinfer | 1/1 | 14.17% | 14.17–14.54% |
| pipeline | qwen3_0p6b | ragged_decode | float32 | reuse | flashinfer | 0/1 | — | — |
| pipeline | qwen3_0p6b | ragged_decode | float32 | rotating | flashinfer | 0/1 | — | — |
| prefill | qwen35_full_attention | prefill | bfloat16 | reuse | flash_attention_optional | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | bfloat16 | reuse | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | bfloat16 | reuse | sdpa_flash | 6/6 | 2.14% | 2.10–2.17% |
| prefill | qwen35_full_attention | prefill | bfloat16 | reuse | sdpa_math | 6/6 | 22.16% | 22.00–22.13% |
| prefill | qwen35_full_attention | prefill | bfloat16 | rotating | flash_attention_optional | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | bfloat16 | rotating | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | bfloat16 | rotating | sdpa_flash | 6/6 | 2.25% | 2.24–2.27% |
| prefill | qwen35_full_attention | prefill | bfloat16 | rotating | sdpa_math | 6/6 | 21.61% | 21.10–21.84% |
| prefill | qwen35_full_attention | prefill | float16 | reuse | flash_attention_optional | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float16 | reuse | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float16 | reuse | sdpa_flash | 6/6 | 2.15% | 2.13–2.15% |
| prefill | qwen35_full_attention | prefill | float16 | reuse | sdpa_math | 6/6 | 21.93% | 21.76–22.07% |
| prefill | qwen35_full_attention | prefill | float16 | rotating | flash_attention_optional | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float16 | rotating | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float16 | rotating | sdpa_flash | 6/6 | 2.25% | 2.20–2.26% |
| prefill | qwen35_full_attention | prefill | float16 | rotating | sdpa_math | 6/6 | 21.85% | 21.41–21.89% |
| prefill | qwen35_full_attention | prefill | float32 | reuse | flash_attention_optional | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float32 | reuse | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float32 | reuse | sdpa_flash | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float32 | reuse | sdpa_math | 6/6 | 20.37% | 20.36–20.71% |
| prefill | qwen35_full_attention | prefill | float32 | rotating | flash_attention_optional | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float32 | rotating | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float32 | rotating | sdpa_flash | 0/6 | — | — |
| prefill | qwen35_full_attention | prefill | float32 | rotating | sdpa_math | 6/6 | 20.90% | 20.66–20.97% |
| prefill | qwen3_0p6b | prefill | bfloat16 | reuse | flash_attention_optional | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | bfloat16 | reuse | sdpa_cudnn | 6/6 | 3.09% | 3.04–3.11% |
| prefill | qwen3_0p6b | prefill | bfloat16 | reuse | sdpa_flash | 6/6 | 3.03% | 2.85–3.05% |
| prefill | qwen3_0p6b | prefill | bfloat16 | reuse | sdpa_math | 6/6 | 34.98% | 34.55–35.18% |
| prefill | qwen3_0p6b | prefill | bfloat16 | rotating | flash_attention_optional | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | bfloat16 | rotating | sdpa_cudnn | 6/6 | 3.37% | 3.29–3.38% |
| prefill | qwen3_0p6b | prefill | bfloat16 | rotating | sdpa_flash | 6/6 | 3.45% | 3.39–3.51% |
| prefill | qwen3_0p6b | prefill | bfloat16 | rotating | sdpa_math | 6/6 | 33.59% | 33.51–34.40% |
| prefill | qwen3_0p6b | prefill | float16 | reuse | flash_attention_optional | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float16 | reuse | sdpa_cudnn | 6/6 | 3.05% | 2.91–3.13% |
| prefill | qwen3_0p6b | prefill | float16 | reuse | sdpa_flash | 6/6 | 3.17% | 3.17–3.20% |
| prefill | qwen3_0p6b | prefill | float16 | reuse | sdpa_math | 6/6 | 34.41% | 34.38–34.77% |
| prefill | qwen3_0p6b | prefill | float16 | rotating | flash_attention_optional | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float16 | rotating | sdpa_cudnn | 6/6 | 3.35% | 3.35–3.36% |
| prefill | qwen3_0p6b | prefill | float16 | rotating | sdpa_flash | 6/6 | 3.44% | 3.41–3.46% |
| prefill | qwen3_0p6b | prefill | float16 | rotating | sdpa_math | 6/6 | 33.68% | 32.77–34.67% |
| prefill | qwen3_0p6b | prefill | float32 | reuse | flash_attention_optional | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float32 | reuse | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float32 | reuse | sdpa_flash | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float32 | reuse | sdpa_math | 6/6 | 31.84% | 31.42–31.98% |
| prefill | qwen3_0p6b | prefill | float32 | rotating | flash_attention_optional | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float32 | rotating | sdpa_cudnn | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float32 | rotating | sdpa_flash | 0/6 | — | — |
| prefill | qwen3_0p6b | prefill | float32 | rotating | sdpa_math | 6/6 | 32.15% | 31.38–32.19% |

![decode_latency.svg](decode_latency.svg)


![prefill_latency.svg](prefill_latency.svg)


## 完整数据与限制

逐项延迟、吞吐比、误差、变异系数、工作集与所有 unavailable 记录见 `summary.csv`、`summary.json` 和 `run-*.json`（归档可为 .json.gz）。进程日志为 `run-*.txt`；版本、命令、GPU 快照和 SHA256 在 `manifest.json`。

基线非成功状态计数（逐 case、逐进程）：{'unsupported': 552, 'not_installed': 216}。348 个有效对照项至少有一端的样本 CV 超过 5%，均保留。

FP64 误差检验使用完整 K 维；长 Q 采样固定行，全量 O 另与 FP32 数学参考比较。所有容差来自 suite.json，没有为保留快结果而放宽。

query tok/s 只表示 attention 算子的输入 query 数 / 时间，不是模型生成速度。本报告没有端到端 prefill/decode 加速结论。

## 简历用语

必须选择一行完整限定的任务/精度/头配置/缓存条件，并注明覆盖的 shape 数，不能把FP32 Math、FP16 Flash 和 paged decode 的结果混在一起。可用模板：

> 实现分页 GQA attention，支持 KV cache 更新与变长 batch；在 RTX 5060 Ti、[dtype、profile、phase、cache、shape 集] 下，吞吐几何平均达到同精度 [FlashInfer append+attention / Torch append+SDPA Flash+NHD输出] 对照的 Y%，并通过输出及 cache 更新一致性验证。

数值 Y 取上表对应分组的吞吐比例。相较历史 kernel 的提升必须来自匹配任务、精度和 shape 的独立对比。
