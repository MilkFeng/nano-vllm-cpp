# Attention 比较

自定义侧直接调用 **`src/kernels/cuda/attention.cu`** 的
`nanovllm::kernels::cuda::flash_attn_with_kvcache()`。
`attn_bridge.cc` include 项目的 `kernels/cuda/kernels.hh`，CMake 将该原始 `.cu` 与
TensorView 支持代码构建为实验共享库。`exp/attn` 不包含另一份 CUDA attention
算法，也没有复制、冻结或改写生产 kernel。

## Tensor Core 路径：2026-11-04

在 RTX 5060 Ti / SM120、Release 上完成 **3 个独立进程、672 项**正式测量：
全部 56 个分页 shape × FP16/BF16 × reuse/rotating，每项每实现 15 组样本。
输出、cache 更新、源码/共享库 SHA256 校验全部通过。按既有的等权几何平均口径，
**40 个配置/阶段/精度/缓存分组全部超过 FlashInfer 的 50%，最低 55.49%**。

以下为 **BF16/reuse、包含 KV 追加**：

| Hq/Hkv/D | Decode（12 shapes） | Paged prefill（6 shapes） | Chunked prefill（8 shapes） |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B：16/8/128 | **83.52%** | **105.51%** | **94.50%** |
| Qwen3.5 full attention：16/2/256 | **95.06%** | **69.86%** | **61.07%** |

这是分组目标；**5/224 个单项仍低于 50%**，均为 D=256、Q=16、prefix=2048，
逐项保留在报告中。FP32 模型端到端速度不属于本次结论。
[验收说明与限制](results/rtx5060ti-20261104-tensor-core/overview.md) ·
[完整分组报告](results/rtx5060ti-20261104-tensor-core/report.md) ·
[逐项 CSV](results/rtx5060ti-20261104-tensor-core/summary.csv) ·
[可运行验收脚本](results/rtx5060ti-20261104-tensor-core/verify_results.py)。

生产算子新增 FP16/BF16、D=128/256 的 Tensor Core 路径：`mma.sync` 执行 QK/PV，
softmax 状态和输出累加保持 FP32，PV 的概率转换为输入精度；`cp.async` 搬运
Q/K/V，`ldmatrix` 加载矩阵片段。同组 GQA head 共用 KV tile，分页地址每个 token
计算一次。D=128 decode 使用向量化 SIMT；D=256 decode 的 warp 分摊输出维度。
Prefill 和 decode 均可沿 KV 分片，FP32 scratch 保存未归一化输出、最大值和分母，
第二个 kernel 稳定归并。新 KV 始终从输入读取，cache 由唯一的 tile/head/分片写入。

FP32、其他 D 及不支持该 Tensor Core 配置的 GPU 保留原 SIMT 路径。
新路径要求 SM80+ 且 opt-in shared memory 至少 96 KiB，已实测 SM120；SM75 编译通过。
算子内不分配显存、不做全局同步；硬件属性及 shared-memory 配置按进程缓存。

引擎在 capture 外用 **`flash_attn_workspace_elements(max_batch, Hq, D, max_k)`**
分配通用 FP32 工作区，覆盖所有 query 长度和不超过 max_batch 的 batch。
Tensor 分片的 packed query/head/split 行数上限为 16384；D=128/256 的工作区约
8.1/16.1 MiB。旧 `flash_attn_decode_workspace_elements()` 仍只保证 decode 容量。
不传工作区时使用单阶段路径；传入不足的工作区会拒绝调用。
工作区按 stream 独立拥有、生命周期覆盖 Graph 重放。

正式对比沿用 `suite.json` 的全部 56 个分页 shape，包含 KV 追加，FP16/BF16 各自
比较同精度 FlashInfer，reuse/rotating 各自统计，3 个独立进程、每项 15 组样本。
FP32 不在 FlashInfer 可比集合内，正确性仍由边界、原生和 Split-KV 测试覆盖。
结果与完整性验证归档到 `results/rtx5060ti-20261104-tensor-core/`。

```bash
cmake --build build --target nano-vllm-cpp-attn-check nano-vllm-cpp -j 4
build/attn-venv/bin/python -m unittest discover -s exp/attn -p 'test_*.py' -v
build/attn-venv/bin/python exp/attn/run_attn.py --suite full --tasks pipeline \
  --dtypes float16,bfloat16 --cache both --runs 3 \
  --output-dir exp/attn/results/rtx5060ti-20261104-tensor-core
```

## Split-KV 完整 profile：2026-11-03，RTX 5060 Ti

直接运行原有 `run_attn.py --suite full`，每进程 **408 项**，3 个独立进程共 **1224 项**，
耗时约 **44 分钟**。覆盖 FP32/FP16/BF16、reuse/rotating、paged decode/prefill/chunked prefill
及变长请求、连续 prefill；所有项目侧输出和 cache 检查通过，计时前后源码及共享库 SHA256 一致。

以下是 **BF16/reuse** 的吞吐比例几何平均，包含 KV 追加及必要的归并/输出处理：

| 配置 Hq/Hkv/D | Decode / FlashInfer（12 shapes） | Paged prefill / FlashInfer（6 shapes） | 连续 prefill / SDPA Flash（6 shapes） |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B：16/8/128 | **16.31%** | 3.36% | 3.03% |
| Qwen3.5 full attention：16/2/256 | **8.47%** | 2.03% | 2.14% |

100% 表示与对照等速。相较上一份单 block 完整记录，BF16/reuse decode 的项目侧延迟
几何平均均约改善 **1.67×**；连续 prefill 本轮延迟约高 **10–11%**，其计算路径没有改动。
这是两次完整测试的对比，测量环境、缓存及 baseline 选择的波动需一并考虑。

[完整分组报告](results/rtx5060ti-20261103-split-kv-full/report.md) ·
[与上一完整记录对比及曲线](results/rtx5060ti-20261103-split-kv-full/comparison.md) ·
[全部逐项 CSV](results/rtx5060ti-20261103-split-kv-full/summary.csv)。
原始样本、日志、编译参数和完整性验证均保存在同目录。下面一节是另外保留的同进程交错消融。

## Split-KV 交错消融：2026-11-03，RTX 5060 Ti

生产 decode 支持多个 block 协作计算一个 query head：每片保存 FP32 的未归一化输出、
softmax 最大值和分母，第二个 kernel 稳定归并。分片数随 batch、head 数和 KV 长度变化，
目标约 256 个 block，最多 32 片，限制每片平均至少 256 个 token；短序列或 block 数充足时
仍走单阶段路径。Prefill 的计算路径保持不变。

`flash_attn_with_kvcache()` 末尾增加可选 FP32 `workspace`；不提供时仍支持原单 block 调用。
使用 `flash_attn_decode_workspace_elements(max_batch, Hq, D, max_k)` 查询容量，返回的是
**float 元素数**，覆盖所有不超过 `max_batch` 的 batch，避免只按最大 batch 估算导致小 batch
分片更多时容量不足。工作区在 capture 外分配，生命周期覆盖 stream/Graph 使用，
不同并发 stream 使用不同工作区。引擎已预分配并通过 `ForwardContext` 传入，两种 Qwen 模型
的 full attention 层在同一 stream 上复用它；算子内部不分配显存、不做全局同步。

与修改前的单 block 库在同一进程交错计时，3 个独立进程、每项每实现 15 组样本，
共 156 个 BF16 decode 比较项，两端均包含 KV 追加，新版本包含归并：

| 配置 | reuse：12 shapes 几何平均 | rotating：12 shapes 几何平均 | B=1、KV=8192、reuse |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | **1.86×** | **1.74×** | 4.401 ms → 0.397 ms，**11.10×** |
| Qwen3.5 full attention | **1.94×** | **1.82×** | 7.366 ms → 0.517 ms，**14.25×** |

这是相较本次修改前版本的算子加速；本轮没有重测 FlashInfer 或端到端生成。
完整结果见 [Split-KV 消融报告](results/rtx5060ti-20261103-split-kv/report.md)。
短序列和大 batch 保留单阶段路径，逐项延迟及波动均保留。

验证包含 212 个既有边界检查、新增 Split-KV/GQA/非二次幂页大小/空分片/极端分数/
双 stream Graph 重放测试，以及 132 个原生用例的 memcheck/racecheck。

```bash
cmake --build build --target nano-vllm-cpp-attn-check nano-vllm-cpp --config Release -j 4
build/attn-venv/bin/python -m unittest discover -s exp/attn -p 'test_*.py' -v

# 先保留修改前的库，再编译新版；两端应使用相同 Release/SM 配置。
build/attn-venv/bin/python exp/attn/bench_split_kv.py \
  --before-library /path/to/before/libnano-vllm-cpp-attn-exp.so \
  --after-library build/libnano-vllm-cpp-attn-exp.so \
  --output-dir build/attn-results/split-kv-new --runs 3 --samples 15
```

## 单 block decode 复测：2026-11-03，RTX 5060 Ti

重新编译 `159e84a` 的生产 kernel（包含 decode 专用分支），沿用上次完整测试设置：
3 个独立进程、每项每实现 15 组样本，共 1224 项，约 38 分钟。
测试脚本、用例、软硬件版本与上次一致，记录的源码中仅 `attention.cu` 变化。
212 个数值边界检查和正式测试的全量输出/cache 检查通过；36 个原生用例分别通过
memcheck/racecheck，运行前后源码及共享库 SHA256 一致。

以下为 **BF16、reuse、包含 KV 追加**；加速为匹配 shape 的旧延迟/新延迟的等权几何平均：

| 配置 Hq/Hkv/D | Decode 相较旧记录（12 shapes） | Decode / FlashInfer 吞吐比例：旧 → 新 | 连续 prefill / SDPA Flash：旧 → 新（6 shapes） |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B：16/8/128 | **2.13×** | 4.35% → **9.03%** | 3.20% → 3.24% |
| Qwen3.5 full attention：16/2/256 | **2.69×** | 1.84% → **4.78%** | 2.14% → 2.17% |

Prefill 的生产路径未改，延迟约有 3% 的跨日期变化，不归因于 decode 优化。
完整的新旧对比、长上下文延迟、曲线和限制见
[复测报告](results/rtx5060ti-20261103/comparison.md)；本轮全部 baseline 分组见
[性能报告](results/rtx5060ti-20261103/report.md)，原始样本、逐项 CSV、编译记录及验证结果均在同目录。
下文的 2026-10-31 数据和优化讨论对应旧版 kernel。

## 初测结果：2026-10-31，RTX 5060 Ti

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

我建议先做 decode 专用 kernel，再做 Tensor Core prefill。当前代码已经实现在线 softmax 和融合 KV 更新，接下来最有价值的是调整并行方式。

下面是结合源码和测量结果的判断，具体瓶颈占比还需要 Nsight Compute 验证。

1. Decode：沿 KV 长度并行，加入自适应 Split-KV，优先级最高。

   当前一个 block 处理 4 个 query，每个 warp 负责一个 query。Decode 的 Q=1 时，只有一个 warp 计算输出，其他 warp 参与加载；B=1、Hq=16 总共只有 16 个 block，每个 block 串行遍历整个 KV。对应代码 (src/kernels/cuda/attention.cu:284)

   可以先让多个 warp 协同处理同一个 query，再把长 KV 分给多个 block。每个分片保存局部 softmax 状态和加权输出，最后稳定归并。这也是 Flash-Decoding 的核心思路。分片数应根据 batch、head 数和 KV 长度选择，短序列保留单阶段路径。

   这项值得先做：BF16、B=1、KV=8192 的 Qwen3 配置，目前约 7.23 ms，FlashInfer 约 40 μs。
2. GQA：让多个 query head 共享同一份 KV tile。

   现在每个 query head 单独启动 block。同组 query head 使用相同 KV head，却分别加载、处理相同数据。Qwen3.5 的 Hq/Hkv=16/2，每个 KV head 对应 8 个 query head。

   可以把同组的若干 query head 放进一个 block，共享 KV tile。它需要和 Split-KV 一起设计，兼顾数据复用和 block 数量；实际显存流量收益还要考虑缓存命中。
3. Prefill：把 QKᵀ 和 PV 改成 Tensor Core 矩阵运算。

   当前计算循环 (src/kernels/cuda/attention.cu:207)逐个 key 做 FP32 标量点积，每个点积都进行 warp reduction。FP16/BF16 输入也走这条计算路径。

   建议扩大 query tile，让更多 query 复用 KV，并用 mma.sync 完成 QKᵀ、PV，softmax 和累加保持 FP32。这对应 FlashAttention-2 强调的矩阵计算效率和 warp 工作划分，是 prefill 的主要优化方向。
4. 随后优化搬运、同步和特化。
   - KV shared memory 保留 FP16/BF16，减少容量占用。
   - 配合向量化加载、cp.async 和双缓冲，重叠搬运与计算。NVIDIA 数据流说明
   - 对 D=128/256、decode/prefill 做模板特化。
   - 根据 profiler 结果再调整 shared memory 布局和 swizzle。

   注意：当前 score 每个 lane 只保存一个值，不能直接把 kKeyTile 从 16 改成 64，需要同时重写分数存储和归约布局。

实施顺序我会选：decode 协同计算 → 自适应 Split-KV → GQA 复用 → Tensor Core prefill → 搬运流水线。每一步单独测消融，保留这次结果作为改前 baseline。这样简历可以同时写“相较原实现提升 X 倍”和“达到 FlashInfer/SDPA Flash 的 Y%”。

所有优化都应写进 src/kernels/cuda，exp/attn 继续只调用生产接口做比较。
