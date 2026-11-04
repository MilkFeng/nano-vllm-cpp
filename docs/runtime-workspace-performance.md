# Qwen3.5 专家缓存与临时 Tensor 复用（2026-11-04）

先在原实现中对照每层 3/8/16 个 GPU 专家槽位，再实现引擎级临时 Tensor 池和 CPU 专家结果批量上传，并对三个预算逐一复测。

## 结果

输入 128、输出 128、batch=1；每个新引擎预热 1 轮、正式 3 轮。下表是正式轮次平均值。

| 每层 GPU 槽位 | GPU assignment ratio | 原 decode tok/s | 优化后 tok/s | 本次提升 | 原/新 CPU 等待 ms/token | 原/新 Tensor 显存峰值 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 14.41% | 9.70 | 10.61 | 9.3% | 42.96 / 41.27 | 10.222 / 10.238 |
| 8 | 26.65% | 9.75 | 11.17 | 14.6% | 41.02 / 37.66 | 10.810 / 10.828 |
| 16 | 36.34% | 9.93 | 11.68 | 17.7% | 38.73 / 33.36 | 11.748 / 11.766 |

![主对照吞吐和 CPU 等待](../profile/cache-workspace-20261104/comparison.png)

## 16 槽位的相邻进程复核

为核对测试顺序和轮间波动，另用两个新进程相邻运行原实现和优化实现，每端预热 1 轮、正式 3 轮。结果为 **10.64 → 11.26 tok/s，+5.8%**；本次复核的 384 个 token ID 也完全一致。

16 槽位两组对照合计每端 6 轮，均值为 **10.28 → 11.47 tok/s（+11.5%）**。两组均值增幅分别为 17.7% 和 5.8%，说明实际收益有明显波动，不能把最好一组写成稳定的 17.7% 加速。慢样本全部保留。

复核原始数据：[before-confirm16.json](../profile/cache-workspace-20261104/before-confirm16.json)、[after-confirm16.json](../profile/cache-workspace-20261104/after-confirm16.json)。


## 缓存容量的结论

原实现中 3→16 槽位将 GPU assignment ratio 从约 14.4% 提高到 36.3%，CPU 专家等待从约 42.96 降至 38.73 ms/token，但平均吞吐仅提高约 2.3%；该差异小于本次轮间波动，不能据此宣称稳定加速。

同层多个 CPU 专家并行，层完成时间取决于最后一个任务。减少 CPU assignments 不会同比缩短等待；GPU 专家仍逐个执行，也增加了 GPU 工作及提交。固定的全模型换入上限还会影响大缓存的学习速度。这是当前短输出和固定替换预算下的结果，不能外推到所有提示词或更长任务。

## 实现

- `TensorPool` 绑定一个 CUDA stream，在 `ModelRunner::run` 中启用。`Tensor::empty` 从池中取未被 Tensor 持有的存储，按容量复用；仍被 Tensor 或 Tensor slice 引用的块不复用。
- 所有 GPU 使用必须提交到绑定 stream；同一块的前后使用按 stream 顺序执行。正常步骤仍显式同步，异常退出时作用域在 host 元数据销毁前同步。
- 按 2 的幂归类容量，避免变长请求为每个长度永久保留独立块。池内存保留至引擎销毁，因此峰值按保留容量计入。
- 单 token CPU 专家改用持久 pinned 输入/输出缓冲区。CPU 输出连续打包，只上传一次；CPU/GPU 混合时用 permutation/scatter 恢复原 top-k 顺序；全 CPU 或全 GPU 时不需要 scatter。
- 未改变专家放置策略、默认缓存预算、GEMM/Attention 内核、模型精度或采样规则。

三个预算的稳态 decode 均从每 token **901 次 cudaMalloc + 901 次 cudaFree** 降为 **0 + 0**。第一次遇到新容量时仍可能扩容，表中测试经过预热。

## 正确性与检查

- 主矩阵 **1152 个 token ID** 与对应原实现逐项一致，连同 16 槽位相邻复核共 **1536 个 token ID** 一致。主矩阵的 CPU/GPU assignments、swaps 和专家上传字节数也完全一致。
- 原生 GPU 测试覆盖 Tensor slice 保活、同 stream 复用、不同 stream 隔离、嵌套作用域、异常 DMA 同步、变长容量复用。
- 专家检查覆盖全 CPU、全 GPU、交错混合 rank、batch→decode、top-k=1，并逐 assignment 对照 CPU 参考。
- 小型完整 Qwen3 模型在 prefill/decode 和不同长度间切换，复用前后 logits 完全一致。
- Release 构建、原生 GPU 功能测试、compute-sanitizer memcheck 和 initcheck 通过，均为 0 errors；具体日志见原始证据。

## 测量边界

- RTX 5060 Ti 16 GB，Ryzen 7 9700X，WSL2，Release / CUDA 13.3；8 个 CPU 专家线程。
- 初始专家 ID 使用嵌套偶数名单：3 槽位为 0,2,4；8 槽位为 0,2,…,14；16 槽位为 0,2,…,30。
- KV block size=16、num_blocks=32、max_batched_tokens=128，greedy、ignore EOS、seed=42、无前缀缓存。每 4 个 decode 步替换，最多全模型 8 次交换，其余策略参数保持默认。
- 原实现顺序 3→8→16，优化后顺序 16→8→3。每个 case 创建新引擎；同一引擎内重复请求保留在线策略学习状态，三轮不作为三个独立进程实验。
- 使用轻量计数版源文件副本记录 CPU future 等待、Storage cudaMalloc/free 次数和字节高水位。没有 Nsight tracing；两侧使用同一测量驱动。
- Tensor 显存峰值为 Storage 实际请求的 CUDA 分配高水位，包含模型、KV、状态、激活和池保留块；不含 CUDA context、驱动和桌面进程。不能与 nvidia-smi 的全卡占用混用。
- 权重复制到 `build/perf-model-cache`（约 35 GB）以缩短重复加载；加载和输入构造不计入 inference。
- 不删除慢样本。终端和 JSON 保留每轮数据，结果仅限本机该合成工作负载。

## 逐轮范围

| GPU 槽位 | 原 decode tok/s min–max | 优化后 min–max | 原 prefill ms | 优化后 prefill ms |
| --- | ---: | ---: | ---: | ---: |
| 3 | 9.39–10.16 | 10.39–10.84 | 2150.67 | 1964.53 |
| 8 | 9.33–10.10 | 11.07–11.27 | 2237.92 | 1923.30 |
| 16 | 9.75–10.03 | 11.57–11.76 | 1873.79 | 1677.19 |

## 复现与原始证据

- [原实现逐轮数据](../profile/cache-workspace-20261104/before.json)、[优化后逐轮数据](../profile/cache-workspace-20261104/after.json)、[汇总 JSON](../profile/cache-workspace-20261104/summary.json)。原始数据含生成 token IDs。
- `profile/cache-workspace-20261104/prepare.py`、`benchmark.cc`、`counters.hh` 保存测量程序；before/after 目录保存测量源文件副本、二进制和哈希。
- [GPU memcheck](../profile/cache-workspace-20261104/memcheck.txt)、[GPU initcheck](../profile/cache-workspace-20261104/initcheck.txt)。
- 测试：`cmake --build build --target nano-vllm-cpp-runtime-workspace-test -j4`，然后 `./build/nano-vllm-cpp-runtime-workspace-test`。
- 原吞吐对照二进制为 `profile/cache-workspace-20261104/before/bench`；优化后为 `after/bench`。参数依次为模型目录、JSON 输出路径、槽位列表、预热轮数、正式轮数。
