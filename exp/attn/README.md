# Attention 比较

自定义侧直接调用 **`src/kernels/cuda/attention.cu`** 的
`nanovllm::kernels::cuda::flash_attn_with_kvcache()`。
`attn_bridge.cc` include 项目的 `kernels/cuda/kernels.hh`，CMake 将该原始 `.cu` 与
TensorView 支持代码构建为实验共享库。`exp/attn` 不包含另一份 CUDA attention
算法，也没有复制、冻结或改写生产 kernel；本轮不宣称内部优化收益。

## 已测结果：RTX 5060 Ti

2026-10-31 在 RTX 5060 Ti / WSL2 上完成 **3 个独立进程、1224 个测试项**，
项目 kernel 全部通过输出与 cache 更新检查。测量期间源码和共享库 SHA256 一致。
编译为 Release、`-O3`、SM120；每项每实现 15 组样本。

以下为 **BF16、reuse**，逐 shape 吞吐比的等权几何平均；所有对照均包含 KV 追加：

| 头配置 Hq/Hkv/D | paged decode / FlashInfer（12 shapes） | paged prefill / FlashInfer（6 shapes） | 连续 prefill / SDPA Flash（6 shapes） |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B：16/8/128 | 4.35% | 3.37% | 3.20% |
| Qwen3.5 full attention：16/2/256 | 1.84% | 2.03% | 2.14% |

百分比为 `baseline_time / project_time * 100%`，表示项目吞吐达到 baseline 的比例。
这些是算子结果，不能作为模型端到端 decode/prefill tok/s。当前生产 kernel 与主流
融合实现差距较大，建议先优化生产 kernel，再把这一项作为简历的性能亮点。

[完整报告与曲线](results/rtx5060ti-20261031/report.md)包含各精度、阶段、缓存条件的
结果、覆盖率和三进程范围。[逐项 CSV](results/rtx5060ti-20261031/summary.csv)、
`run-*.json.gz` 原始样本、`manifest.json` 版本/命令/校验、`build-info.json` 编译记录、
`correctness.json` 数值边界检查及 memcheck/racecheck 日志均保存在同一目录。

## 运行

先使用已经具有 CUDA PyTorch 的 Python 准备独立环境，复用其安装：

```bash
python exp/attn/setup_env.py --torch-python /path/to/python-with-cuda-torch
cmake -S . -B build
cmake --build build --target nano-vllm-cpp-attn-exp --config Release -j 4

# 正确性：边界、GQA、右下 causal、cache 更新、哨兵、重复调用
build/attn-venv/bin/python exp/attn/run_attn.py --check-only --output build/attn-checks.json

# 完整比较：3 个独立进程，15 组样本，两个任务、三种精度、两种缓存条件
build/attn-venv/bin/python exp/attn/run_attn.py --runs 3 --output-dir build/attn-results/full
```

完整测试负载较大，原 kernel 在最大 prefill shape 上单次约数百毫秒，三轮可能
需要数十分钟。`--suite quick --runs 1 --samples 3 --warmup 3 --target-ms 2`
用于冒烟；不能用它代替完整测试结论。输出目录已有数据时拒绝覆盖。
FlashInfer 0.7.2.post1 本次在 Linux/WSL 上验证；Windows 可构建 bridge 并测 CUDA SDPA，
FlashInfer 可用性需另验。CLI 的 `--library` 可指定对应 DLL/so。

`requirements-measured.txt` 记录本次主要依赖版本。环境及 JIT 缓存位于 `build/`，
不将某个外部训练项目的绝对路径写成仓库依赖。首次 FlashInfer JIT 会花额外时间，
不计入 kernel 时间。没有安装的可选独立 `flash-attn` 会记录 not_installed。

## 比较同一项工作

当前 API 融合了 **追加 KV + 分页 GQA attention**，没有纯 attention 开关。
因此本实验所有性能比较均包含 KV 更新，不通过自写只读 kernel 去替代它。

| runner task | 页表 | 项目侧 | Baseline |
| --- | --- | --- | --- |
| `pipeline` | 随机打散的物理页 | 原生产 API | FlashInfer append_paged_kv_cache + paged prefill/decode run |
| `prefill` | identity，连续 KV | 同一个原生产 API | Torch index_copy_ 追加 K/V + 强制 SDPA Math/Flash/cuDNN + NHD 输出写回 |

SDPA 使用同一底层 NHD cache 的零拷贝 dense view。其输出布局整理及复制到预分配
NHD O 的成本纳入 Graph。pipeline 的 FlashInfer 原生 NHD tuple 对应本项目独立
K/V cache，不做额外 gather。两个任务分开统计，不把 pipeline 比例叫作纯 attention 性能。

本次 FlashInfer 安装中的 JIT dtype map 不含 float32，FP32 对应项明确 unsupported，
不暗中转成 FP16。FP32 的性能对照是连续 prefill 的 SDPA Math；FP16/BF16 分别
对比同精度的 FlashInfer 或 SDPA Flash。SM120 的 SDK/shape 支持以实际运行记录为准。

FlashInfer 对适用的 auto/fa2 与 SIMT/tensor-core 配置做独立校准，选最快的正确候选，
再正式采样；保存候选耗时、实际 backend、失败原因和选中配置。workspace 统一为
256 MiB：最大 FA2 prefill 的临时 V 就需要约 134.6 MB，128 MiB 不够。
`plan()` 和 JIT 计时外完成，planning 时间、workspace 大小另记。
SDPA 每次仅启用指定 backend，以 profiler 观察到的 aten op 核对，拒绝隐式 fallback。

## 输入、mask 与 shape

```text
Q / O:        [sum(Q_i), Hq, D]
new K / V:    [sum(Q_i), Hkv, D]
cache K / V:  [pages, page_size, Hkv, D]，两个独立连续张量
block_table:  [batch,max_pages]，int32
cu_q / cu_k:  [batch+1]，int32，分别是 query 长度和追加后 KV 总长度的前缀和
prefix_i = K_i-Q_i
causal: key_position <= prefix_i + local_query_position
KV head = query_head // (Hq/Hkv)
```

cu_k 不是 FlashInfer 的 page indptr；harness 从有效 block table 生成 CSR indptr、
indices 和 last_page_len。RoPE、QK norm、projection、Qwen3.5 sigmoid gate 和
GatedDeltaNet 都不在本算子内。scale=1/sqrt(D)，dropout=0，不含 ALiBi、soft cap。

SDPA 的 `is_causal=True` 在非方阵中为左上对齐，不能直接用于有 prefix 的 chunked
prefill。本实验 SDPA 性能对照仅限 Q=K 的完整 prefill；其他阶段由原分页 API 与
FlashInfer 比较。独立数学参考使用显式的右下对齐 mask。

机器可读矩阵为 `suite.json`。两个 profile：Qwen3-0.6B 的 Hq/Hkv/D=16/8/128，
Qwen3.5 full attention 的 16/2/256（不涵盖 linear attention）。每个 profile：

| 阶段 | 固定组合 | 数量 |
| --- | --- | ---: |
| prefill | B=1/4，Q=K=128/512/2048 | 6 |
| decode | B=1/4/16，Q=1，K=128/512/2048/8192 | 12 |
| chunked prefill | B=1/4，Q=16/128，prefix=512/2048 | 8 |
| ragged decode | Q=[1,1,1,1]，K=[127,513,2047,8191] | 1 |
| ragged chunked | Q=[1,3,17,65]，prefix=[0,15,511,2047] | 1 |

共 56 种逻辑 shape。pipeline 每种缓存条件 168 项（3 dtype）；连续 prefill
任务每种缓存条件另有 36 项。合计每进程 **408 项**，三进程 **1224 项**。
page_size=16 为主集，其他页大小及 8192 prefill 属扩展，不加入主集合汇总。

## 正确性

每个实现分别从干净 cache 开始，目标新槽位填 poison。检查全量 O、完整 cache
更新、旧 prefix/未分配页/页尾 padding 不变、输出哨兵和重复调用。
全量 O 对比禁用 TF32 的 FP32 SDPA Math；另对短序列全量、长序列固定 sampled
query rows 的完整 K 维做独立 FP64 stable softmax 参考。输入先量化到目标 dtype。
容差固定在 suite.json，记录最大绝对误差、relative L2 和最大容差比。

额外边界检查包含 Q=0/1/2/3/5/17/65、空 batch/混合空请求、跨页 prefix，
Hq=Hkv、MQA、GQA，D=64/80/96/128/256，causal true/false，以及非默认 stream。
确定性 Q/K 全零、V=token位置+KV head编码的样例专门检查 mask 与 head 映射。
输入中 K<Q 或非法头配置在调用前拒绝；生成的页表没有共享可写页。

```bash
build/attn-venv/bin/python -m unittest discover -s exp/attn -p test_attn.py -v
cmake --build build --target nano-vllm-cpp-attn-check --config Release -j 4
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-attn-check
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/nano-vllm-cpp-attn-check
```

原生检查驱动只有输入构造、CPU 参考和 API 调用，通过同一个共享库执行原生产
kernel；它在 36 个代表性用例上完成 memcheck/racecheck，避免 Python/CUDA 模块
加载在 sanitizer 下的巨大开销。更全面的 212 个数值边界用例仍由 Python runner 检查。

## 计时、统计与产物

输入生成、分配、H2D、页表转换、plan/JIT、校准、正确性、profiler 和 Graph 构建
不计入 GPU 时间；每个 task 必要的 append 与输出处理在 Graph 内。使用同一个非默认
stream，C++ 直接借用 Torch GPU 指针和当前 stream，无 LibTorch 或输入文件搬运。

每项预热至少 20 次或完整一轮地址轮换；每组目标约 10 ms，最多 1024 次调用，
实际次数保留。每项 15 组样本，随机交错实现、shape 和缓存条件；seed=42/43/44
分别在独立进程运行。固定 seq lengths 与 append 槽位，replay 中不让 KV 长度增长。

reuse 复用一组地址。rotating 轮换最多 32 个 bank，目标 max(64 MiB,2×L2)，
受空闲显存 25% 预算约束；未达到目标会标注，不宣称全冷缓存。

每进程先取样本中位数，再取进程间中位数；跨 shape 比值用等权几何平均。
按 task、profile、phase、dtype、cache、baseline 分组，并记录覆盖率 k/N。
unsupported/not_installed/memory_limit/incorrect/error 均保留，不删除慢样本。
范围是独立进程范围，不是置信区间。源代码及共享库 SHA256、版本、完整命令和
GPU 时钟/温度/使用率快照保存在 manifest；候选校准及全部样本在 run JSON。

输出 `report.md`、`summary.csv/json`、`run-*.json/txt`、环境 manifest；安装了
matplotlib 时额外输出 BF16 batch=1 decode/prefill 的 SVG/PNG 延迟图。
`baseline_time/custom_time*100%` 表示项目吞吐达到对照的比例。query tok/s 只是该
算子的 query 数/时间，不是模型生成速度。当前没有改变生产算法，不填内部优化 X%。

```bash
build/attn-venv/bin/python exp/attn/run_attn.py --report-only build/attn-results/full
```

接口参考：[FlashInfer](https://docs.flashinfer.ai/api/attention.html)、
[append NHD tuple](https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html)、
[PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)。
初期小样例探测记录 `results/rtx5060ti-20261031-capabilities.json` 只反映当时环境和
特定 eager 样例，不是性能结果，也不代表之后安装的 FlashInfer 不可用。



完整比较已完成：三轮共 1224 项，另通过 212 个边界检查及 memcheck/racecheck。

RTX 5060 Ti、BF16、地址复用、包含 KV 追加，吞吐比几何平均如下：

| 配置 | Decode / FlashInfer（12 shapes） | Prefill / SDPA Flash（6 shapes） |
|---|---:|---:|
| Qwen3-0.6B | 4.35% | 3.20% |
| Qwen3.5 full attention | 1.84% | 2.14% |

完整报告、曲线与原始数据 (exp/attn/results/rtx5060ti-20261031/report.md)已归档。当前性能差距较大，建议先优化生产 attention，再作为简历上的性能亮点。