# 性能基线与简历指标计划

目标是形成两条可复现结论：固定专家缓存预算下的整模型收益，以及真实模型
shape 下的算子收益。除第 4 节已接入的 GEMM 与对照程序外，新增策略和后续算子
优化仍是实验计划；最终性能指标需按下述口径正式测量。

## 1. 当前能证明什么

当前已实现专家分组执行、CPU AVX2/FMA 点积、GPU FP8 权重反量化融合、
CPU/GPU 专家重叠执行，以及带衰减热度、替换阈值和全局换入上限的在线缓存。

此前 RTX 5060 Ti 的短测，第二轮 static/adaptive decode 分别为
8.2455 / 9.4573 tok/s，差约 14.7%。但每端只有两轮、无预热，第一轮波动大，
不能据此写成稳定的优化结论。这些数据仅证明功能运行，不作为最终简历指标。

`expert_placement.cache_bytes` 只包含专家权重和 scales。
`memory_after_load` 是设备显存快照，均不能当作模型运行峰值显存。
35B 能在 16 GB 显卡上运行时，还需注明模型精度、CPU 和系统 RAM 条件。

## 2. 专家调度：先形成整模型指标

| 对照 | 方法 | 要回答的问题 |
| --- | --- | --- |
| 静态名单 | 相同初始 ID、相同每层槽位数，`--expert-adapt-every 0` | 在线替换是否有端到端收益 |
| 当前策略 | 衰减访问次数 + 最小收益阈值 + 每轮替换上限 | 热度迁移是否抵消了拷贝开销 |
| LRU，后续实现 | 按近期访问淘汰，匹配替换周期与传输上限 | 频率信息是否优于单纯的近期性 |
| 消融，后续实验 | 关闭衰减、提高替换上限或去掉阈值，每次只改一项 | 收益具体来自哪里 |

静态初始名单是最低成本的第一条基线。若只比较较差的初始 ID，结论应明确写为
“相较该静态放置基线”，不扩大成优于所有静态热点方案。
后续可用独立校准集得到热点名单并冻结，在另一组文本上验证泛化，避免用测试集
挑选静态名单。LRU 对照也必须计入拷贝和等待，不能仅比较命中率。

先做最小闭环：同一个模型和二进制，batch=1，输入 128、输出 256 或 512，
每端预热 2 轮、正式测量 5 轮。相同 seed、文本轮换、CPU 线程、KV 配置和
初始 expert IDs；关闭完整 profiling，greedy、忽略 EOS，不复用前缀。
每端从新 engine 开始，后续再用反向顺序复测，减少温度和运行顺序影响。
同一 engine 的多轮 generation 有学习状态关联，不视为独立的进程重复实验。

先固定一个预算；验证收益后再扩展每层 3/8/16 个槽位、batch=1/4 和不同文本。
35B 每层 16 个槽位、40 层约 1.875 GiB routed 专家缓存，不包含其他设备内存。
扩大预算前检查实际剩余显存，不默认把剩余空间全部用掉。

主指标选 decode TPOT 或 decode tok/s，它们在 batch=1 时高度相关，属于同一类
收益。辅指标记录 GPU assignment ratio、上传字节数和替换耗时占 decode 时间的比例。
TTFT 单独报告，不能将预热带来的加载、页面或 kernel 初始化差异归因于调度算法。
报告所有工作负载的结果和汇总，不只挑最快的一轮。

更深入的策略是优化“预计节省的关键路径时间”。当前策略按访问次数选专家，
没有考虑 CPU/GPU 实际耗时、组大小、竞争及传输成本。可以先校准不同组大小
`M_e` 的两端成本，再评估候选替换：

```text
预计收益 = 未来 H 步的层完成时间缩减 - 换入时间
层完成时间 ≈ 路由与输入传输 + max(CPU 工作, GPU 工作) + 输出传输与合并
```

预测必须包含被淘汰专家的未来代价。优先记录实际阶段耗时，验证 CPU 是否在
关键路径上；不能简单将“CPU 单专家时间 - GPU 单专家时间”全部算作端到端收益。
再比较同步换入与 copy stream/event 的异步换入，单独做消融，避免混合归因。

## 3. Attention：区分 prefill 与 paged decode

当前 `attention.cu` 支持 FP32/FP16/BF16，Qwen 模型运行链路主要使用 FP32。
kernel 采用 online softmax、共享内存分块和分页 KV，包含新 KV 写入；
它没有 Tensor Core MMA。应明确称为当前 CUDA 实现，不根据函数名宣称具有
FlashAttention-2 的性能。

| 场景 | 正确性参考 | 主流性能对照 |
| --- | --- | --- |
| 常规 causal prefill | FP32/FP64 数学参考 | PyTorch SDPA Flash 后端；设备支持时加入 Dao-AILab FlashAttention |
| 分页 GQA decode | 相同位置和页表的数学参考 | FlashInfer paged-KV decode |
| 当前 FP32 运行链路 | 相同 FP32 语义的参考 | SDPA math 等同精度路径，单独命名报告 |

PyTorch SDPA 会自动选择后端，测试时要指定并记录实际 backend，失败时明确
记录 unsupported，不能把 math fallback 当作 FlashAttention。FlashAttention-2
CUDA 官方接口的主要输入精度是 FP16/BF16。做主流 fused 对照时双方使用相同
输入输出精度、GQA 头数、causal 规则、布局和误差容差。

当前实现还写入 KV cache：纯 attention 与 KV append+attention 分别计时；
组合任务的 baseline 也计入 append，或使用具有对应功能的接口。
Paged decode 对照使用同样页大小、长度分布和页表条件，不额外要求只有 baseline
做 gather。单 token decode 的 query/KV 长度不同，尤其要检查 causal mask 对齐。

建议 shape：batch=1/4/16，context=128/512/2048/8192；prefill 的 query 长度
等于输入长度，decode 的 query 长度为 1。覆盖 Qwen3 的 Hq/Hkv/D=16/8/128 和
Qwen3.5 的 16/2/256。只测 inference forward，不和训练 forward+backward 混比。

从代码推测的优先优化点，需要用 Nsight 验证：

- `kQueryTile=4` 在单 query decode 时只使用其中一个 query warp 执行主计算。
  可实现 decode 专用路径，并考察 split-KV 增加长上下文的并行度。
- GQA 的多个 query head 会分别加载相同 KV head，可尝试跨头复用 KV tile。
- Prefill 若以接近主流实现为目标，需进一步考虑 Tensor Core 分块与流水线，
  仅调大现有 tile 不足以保证竞争力。

参考：[PyTorch SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)、
[FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)、
[FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html)。
版本、SM 架构与实际可用后端必须记录；不能把 H100/B200 的支持直接套到 SM120。

## 4. FP32 GEMM：以用户提供的 kernel 对比 cuBLAS

算子主线改为普通 FP32 GEMM，不继续 FP8 linear benchmark。
用户提供的 kernel 计算 row-major `C[M,N] = A[M,K] @ B[K,N]`，
采用 128x128 block tile、8x8 register tile、float4 向量化、共享内存双缓冲、
寄存器预取和 XOR swizzle。它使用 FP32 FMA，没有 Tensor Core MMA。

当前项目 `src/kernels/cuda/linear.cu` 已接入该 GEMM 的 `X @ W^T` 版本：
沿 `[N,K]` 权重的 K 维加载，再转置写入 shared memory，无需预打包。
原来的 cuBLASLt 实现保留为 `exp/gemm/comp_gemm.cc` 的 `cublas_linear()`。
实验构建、正确性检查和计时口径见 [exp/gemm/README.md](../exp/gemm/README.md)。
独立 kernel 的性能和引擎 Linear 性能仍需分开描述。

RTX 5060 Ti 的完整 33-shape、3 进程测量已保存为
[GEMM 报告](../exp/gemm/results/rtx5060ti-20261030/report.md)。同矩阵复用条件下，
吞吐几何平均为 16×16 共享内存分块基线的 2.63 倍、严格 FP32 cuBLAS 对照的
70.2%，峰值 9.85 TFLOPS。轮换矩阵条件下分别为 2.15 倍、64.2%、9.71 TFLOPS。
主流对照取严格 SGEMM 与从启发式候选实测选择的 cuBLASLt 中较快者；完整数据
包含慢于分块基线与 cuBLAS 的尺寸，不支持“全面超越 cuBLAS”的表述。

| 对照 | 计时与用途 |
| --- | --- |
| cuBLAS strict FP32 | 使用匹配的 FP32 输入、累加和输出，禁用 TF32/低精度降格，作为主流性能参考 |
| 朴素 CUDA GEMM，可加入 | 明确展示分块、访存和复用带来的收益，不能代替 cuBLAS 对照 |
| 用户的优化 kernel | 分别测对齐和尾块 shape；当前集成版通过边界判断处理尾块，无 padding buffers |
| 消融 | 保持 tile 等条件一致，分别移除 swizzle、预取或双缓冲，解释收益来源 |

cuBLAS handle、算法选择、workspace 分配和输入生成在计时外。
两端统一 `alpha=1,beta=0`、矩阵布局及误差检查；使用 CUDA events。
若另测 TF32 Tensor Core 路径，应单列精度和数学模式。

先测方阵 512/1024/2048/4096，再补矩形和尾块，如
`(M,N,K)=(128,2048,2048)、(128,8192,2048)、(129,513,257)`。
报告逐 shape 延迟、逻辑 `2*M*N*K/time` TFLOPS、对 cuBLAS 的吞吐比例和
声明范围内的几何平均加速比，不只报最好的一种尺寸。

最初提供的 `solve()` 会在非对齐尺寸上申请/释放 padding buffers；当前集成版
已改为补零读取与边界写回，不包含这些分配和 D2D 拷贝，可直接用于 Graph capture。

当前集成版对小 M 使用较小 tile，对 `M=1` 使用独立 GEMV。
这些分支保留了推理所需的形状覆盖，但不预设所有 Linear shape 都比 cuBLAS 更快。

参考：[cuBLAS 数学模式与数据类型](https://docs.nvidia.com/cuda/cublas/index.html)。

## 5. 统一测量规则与简历表述

算子使用 CUDA events，预分配数据和 workspace，充分预热后分批采样。
排除首次编译、JIT、分配和 H2D 输入搬运；需要测这些开销时另列 pipeline 指标。
Python 与 C++ 同时测 kernel 时间，不能一端包含 Python 调度而另一端只测 GPU。
热缓存与大权重工作集分开测，避免单个小权重常驻 L2 的结果被误当作整模型收益。
先通过正确性检查，记录绝对误差和相对误差，再比较耗时。

每份结果保存：GPU/CPU、系统、CUDA/库版本、git commit、dtype/数学模式、shape、
完整命令、预热与重复次数、原始样本及聚合方法。不同 shape 的加速比用几何平均，
保留逐 shape 数据，不只报最佳值。主流算子只作为性能对照，不预设必须被击败。

获得实测结果后，简历可采用以下模板，X/Y/Z 均需由结果填充：

> 实现 CPU/GPU 异构 MoE 专家调度，以衰减热度和换入预算维护 GPU 缓存；在
> [硬件、模型、工作负载]、相同专家缓存预算下，相较静态放置将 TPOT 降低 X%。

> 实现 FP32 CUDA GEMM，采用寄存器分块、float4 向量化、共享内存双缓冲与
> XOR swizzle；在 [矩阵尺寸范围] 达到 cuBLAS strict FP32 吞吐的 Y%，
> 相较 [明确的原始 CUDA 基线] 提升 Z 倍。

若最终选择 attention，可替换第二条：

> 优化分页 GQA decode，通过 [实际实现的方法] 将 [shape 范围] 的延迟降低 Z%；
> 同精度对比 FlashInfer，达到其 [实测比例] 的吞吐，并通过数值一致性验证。

最终增加整模型结果：[硬件、CPU/RAM、模型精度、缓存预算、输入/输出长度、batch]
下 prefill 达到 P tok/s、decode 达到 D tok/s。单序列速度和批量聚合吞吐分别注明。

实施顺序：先完成 FP32 GEMM 与 cuBLAS 的独立基准，同时正式复测 MoE 静态/动态
对照；随后对 attention 做独立基准和定向优化，最后回到整模型验证。
保留调度、GEMM、attention、整模型四层证据，避免将算子局部收益直接当作引擎收益。
