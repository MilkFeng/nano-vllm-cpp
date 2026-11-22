# Qwen3.5 混合精度

Qwen3.5 文本推理默认采用 BF16 主干激活和缓存、FP8 大矩阵权重，以及 FP32
累加、残差和递推状态。使用原始 checkpoint，无需修改 `config.json` 或重新量化。
`ModelConfig::dtype` 仍描述 checkpoint 未量化参数的 dtype；
`Qwen35ForCausalLM::dtype()` 返回运行时主干激活精度 BF16。

| 部分 | 运行时存储 / 输入 | 内部计算与输出 |
| --- | --- | --- |
| Embedding | BF16 权重 | BF16 输出 |
| Attention / DeltaNet 大投影 | checkpoint 的 FP8 权重、FP32 scale，BF16 激活 | 融合反量化、FP32 乘加，BF16 输出 |
| GPU routed / shared 专家 | FP8 权重、BF16 激活与中间张量 | FP32 累加；routed 专家部分结果用 FP32 合并 |
| CPU routed 专家 | FP8 权重、FP32 scale、BF16 输入 | FP32 SIMD 计算、SwiGLU 和输出 |
| Router、共享专家门控 | FP32 权重、BF16 输入 | FP32 logits、softmax、路由权重和输出合并 |
| DeltaNet a/b 小投影 | FP32 权重，BF16 输入 | FP32 乘加与输出 |
| DeltaNet A_log、dt_bias | FP32 | FP32 指数、softplus 和衰减计算 |
| RMSNorm、Q/K norm | BF16 输入输出，普通 norm 权重 BF16 | 平方求和、归一化和权重乘法 FP32 |
| DeltaNet gated RMSNorm | BF16 输入输出、FP32 权重 | FP32 归一化与 SiLU |
| 残差流 | FP32 | BF16 分支输出与 FP32 残差在 FP32 中相加 |
| Full attention | BF16 Q/K/V 与 KV cache | FP32 累加和 softmax 统计；BF16 输出 |
| DeltaNet depthwise conv | BF16 权重、输入输出与历史 | FP32 累加与 SiLU |
| DeltaNet recurrent state | FP32 | FP32 状态更新；BF16 核心输出 |
| LM head | BF16 权重与输入 | FP32 累加，直接输出 FP32 logits |

GPU FP8 Linear 接受 BF16 或 FP32 输入，权重在 kernel 内解码成 FP32。
本次没有加入动态 FP8 激活量化；投影和专家仍采用现有 SIMT GEMM/GEMV，
完整注意力的 BF16 路径沿用已有 Tensor Core / SIMT 分派。降为 BF16 的张量
采用 round-to-nearest-even。转换为 FP32 的 BF16/FP8 数值不会恢复量化前的信息。

`MoE` 直接将 BF16 激活下载到 CPU。AVX-512 单 token 的对齐 Gate/Up 内核加载 BF16
后在寄存器中展开为 FP32，其余路径在专家内核入口展开一次。
普通前向和 CUDA Graph 的合并 kernel 都在 FP32 中累加专家输出和共享分支，
并在写出主干输出时舍入为 BF16。
Qwen3 的 LM head 默认输出精度保持原接口行为，Qwen3.5 显式请求 FP32 logits。

## 内存

35B-A3B 的 10 个 full attention 层采用 BF16 K/V，每个逻辑 token 共占
`2 × 10 × 2 × 256 × 2 = 20480` 字节，即 **20 KiB**，是原 FP32 K/V 的一半。

30 个 linear attention 层的每序列状态为：

- FP32 递推矩阵：`30 × 32 × 128 × 128 × 4 = 62914560` 字节。
- BF16 卷积历史：`30 × 8192 × 3 × 2 = 1474560` 字节。
- 合计 **64389120 字节，约 61.4 MiB / 活跃序列**。

专家 FP8 权重和 scale 的占用保持原计算口径。历史性能文档中的 FP32 激活、
40 KiB/token KV 和约 62.8 MiB 状态记录对应修改前实现。

## 验证

```bash
cmake --build build --target nano-vllm-cpp-precision-check \
  nano-vllm-cpp-moe-sync-check nano-vllm-cpp-compact-check -j4
./build/nano-vllm-cpp-precision-check
./build/nano-vllm-cpp-moe-sync-check
./build/nano-vllm-cpp-compact-check
```

`precision-check` 覆盖非整块 GEMM/GEMV、实际 hidden size=2048 的 FP8/BF16
矩阵计算、BF16 LM head 输入直接产生 FP32 logits、FP32 残差，以及独立 FP64
递推参考。513 个 token 的单 token decode 和每块 37 token 的分块 prefill
必须得到逐位相同的 BF16 输出、卷积历史和 FP32 递推状态，未使用的状态槽位保持零。
BF16 输出按 `2e-5 + 0.008 × |reference|` 检查，FP32 递推状态按
`2e-6 + 2e-4 × |reference|` 检查。

`moe-sync-check` 覆盖 BF16 MoE 的全 CPU / 混合 / 全 GPU 专家、共享专家、
不同 batch、同一 stream 上的工作区复用、热点专家替换、profiling 和异步暂存寿命。
混合 BF16 MoE 相对 CPU FP32 参考的阈值为 `3e-4 + 0.02 × |reference|`。
普通执行、分段图和整步图的 Decoder 输出、残差及递推状态要求逐位一致。
`compact-check` 保留原 FP32 路径的 120 组紧凑权重和 9 组融合专家逐位回归检查。

在 RTX 5060 Ti 上，`precision-check` 和 `moe-sync-check` 均通过
`compute-sanitizer --tool memcheck --error-exitcode 99`，各报告 0 errors。
完整本地 `Qwen3.5-35B-A3B-FP8` 使用每层 GPU 专家 `0,2,4`、单序列、
128 token prefill 上限、32 token greedy 输出和两轮请求完成文本生成。
16 个块、每块 16 token 的 KV 分配由 10485760 降至 5242880 字节，
单槽位状态由 65863680 降至 64389120 字节。

输入“用中文简要解释什么是矩阵乘法。”时，BF16 输出与原 FP32 基线的前
14 个 token 相同，随后措辞发生变化。CLI 的 `--repeat` 仅输出最后一轮文本，
两轮均成功运行不代表已比较两轮 token 的逐位一致性。
通用算子修改后的 Qwen3-0.6B 回归生成也通过，输入 `What is 2+2?` 输出 `2 + 2 = 4.`。

这些是算子与执行路径检查，不代表完整模型任务准确率评测；BF16 舍入可能改变
接近的路由或 token 排序。
