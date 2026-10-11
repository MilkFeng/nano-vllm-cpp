# nano-vllm-cpp

使用 C++ 和 CUDA 实现 nano-vllm 的单 GPU Qwen3 推理链路。

## 构建

需要 CMake 3.28+、支持 C++20 的主机编译器和标准库、CUDA Toolkit、Rust/Cargo，以及 `vcpkg.json` 中列出的 C++ 依赖。首次配置会获取 tokenizers-cpp 和固定版本的 minja。

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
```

将架构 `120` 改为实际 GPU 对应值。C++ 和 CUDA 源文件均使用 C++20。

## 生成文本

模型目录需要包含 `config.json`、`model.safetensors`、`tokenizer.json` 和
`tokenizer_config.json`。

```bash
./build/nano-vllm-cpp --model "$HOME/models/Qwen3-0.6B" \
  --prompt 'Hello' \
  --temperature 0 --max-tokens 16 \
  --max-model-len 128 --max-seqs 4 --max-batched-tokens 128 \
  --num-blocks 16 --block-size 16
```

- `--prompt` 默认是一条用户消息，程序读取模型的 `chat_template` 并自动应用。重复 `--prompt` 可批量提交独立对话，输出顺序与输入一致。
- 默认关闭思考模式；`--thinking` 开启模型模板中的 `enable_thinking`。
- `--raw-prompt` 用于原始文本续写或已经格式化的输入，与 `--thinking` 互斥。
- `--temperature 0` 使用 greedy 解码，正数使用温度采样；`--seed` 设置随机种子。
- `--max-batched-tokens` 限制每轮 token 数，较长输入会分块 prefill。
- `--num-blocks` 必填，指定全部请求共享的 KV 块池大小；`--block-size` 指定每块 token 数，默认 256。程序按指定数量分配，不再自动估算显存容量。
- `--json` 输出生成文本和 token ID；`--repeat 2` 在同一引擎中重复请求，以使用前缀缓存。
- 达到 EOS、`--max-tokens` 或总长度 `--max-model-len` 时结束；`--ignore-eos` 可忽略 EOS。

KV 块数和 block size 必须大于零，块池至少容纳一条 `max_model_len` 长度的序列，总 token 容量不得超过 Int32 slot 索引范围。启动日志显示块数和全部 K/V 缓存的字节数。当前 dense Qwen3 的缓存字节数为 `2 * num_hidden_layers * num_blocks * block_size * num_key_value_heads * head_dim * sizeof(dtype)`；配置时还需给权重、激活和计算工作区留出显存，缓存分配成功不保证最大 batch 能运行。`--gpu-memory-utilization` 已移除。

C++ 调用方也必须设置 `EngineParams::num_kvcache_blocks`；默认值 `0` 会报参数错误。例如 `EngineParams params; params.num_kvcache_blocks = 16;`，配合默认 block size 256，可容纳一条默认长度 4096 的序列。

默认温度为 1，只做全词表温度采样；当前不会加载 `generation_config.json` 中的 temperature、top-k 或 top-p 配置。`--max-model-len` 限制总上下文长度；输出长度仍由 `--max-tokens` 控制，默认最多生成 64 个 token。

### C++ 聊天模板接口

[LLMEngine::apply_chat_template](include/nanovllm/nanovllm.hh) 使用 [minja](https://github.com/google/minja) 在 C++ 中渲染 checkpoint 的真实 Jinja 模板，运行时不依赖 Python。当前支持 system、user、assistant 的纯文本消息。

已有 `LLMEngine engine` 时，可以传入对话历史：

```cpp
const std::array messages{
    nanovllm::ChatMessage{"system", "You are a helpful assistant."},
    nanovllm::ChatMessage{"user", "Hello"},
};
const std::array prompts{
    engine.apply_chat_template(messages, {.enable_thinking = false}),
};
auto outputs = engine.generate(prompts, {.temperature = 0.0f});
```

模板接口默认 `add_generation_prompt=true`、`enable_thinking=true`，与 nano-vllm 示例的调用一致；CLI 显式传入关闭思考模式的选项。库的 `generate()` 接受已格式化的 prompt，不会重复套模板。同一个引擎的 `generate()` 应串行调用，跨调用保留可复用的完整 KV 块。

输出只包含新生成的 token。`<|im_start|>assistant` 已属于输入，不会回显；`<|im_end|>` 是新生成的正常结束标记，当前文本解码保留特殊 token。

## 代码组织

`src/layers/` 放通用的 `Linear`、`Embedding`、`RMSNorm`、`SwiGLUMLP`、`LMHead` 和 `RotaryEmbedding`。这些层接收具体维度，不依赖 `ModelConfig`；CUDA 算子和头文件放在 `src/kernels/cuda/`；CPU 算子和参考计算放在 `src/kernels/cpu/`。

`src/models/qwen3.hh/.cc` 负责 Qwen3 Attention、DecoderLayer 和模型组装。`compute_logits()` 在 prefill 时选出每条序列最后一个 token 的 hidden state，再交给只做词表投影的 `LMHead`。`src/core/forward_context.hh` 保存由 ModelRunner 准备的执行元数据，包括 KV block table 和 CUDA stream。

## 已实现链路

```text
Token IDs → Embedding
  → [RMSNorm → Q/K/V Linear → Q/K Norm → RoPE
     → KV Cache → Causal Attention → O Linear
     → Residual + RMSNorm → Gate/Up Linear → SiLU × Up → Down Linear] × layers
  → Residual + RMSNorm → Last-token Gather → LM Head → Sampling → Token IDs
```

基础算子支持 FP32、FP16 和 BF16。Linear 使用 cuBLAS；attention 使用 FP32 在线 softmax，支持 GQA、变长 prefill、缓存前缀和分页 decode。引擎包含分块 prefill、KV 块管理、前缀缓存、显存不足时的请求抢占和停止条件处理。

内部 CUDA 算子只接受连续 Tensor，并使用调用线程已经选好的 device 0。通用模型配置约束，以及模型和算子内部的 shape、dtype 和索引约束使用 Debug 断言；Qwen3 支持范围、用户参数和 CUDA/cuBLAS 运行错误在 Release 中也会检查。缓冲区重叠关系由调用者保证。RoPE 直接原地修改 Q/K。

当前支持 CUDA device 0 上的 dense Qwen3、单文件 safetensors 权重、未缩放的 RoPE，以及偶数且不超过 256 的 head dimension。Attention 是用于打通计算链路的基础实现；FlashAttention 级优化、CUDA Graph、张量并行和分片权重加载仍待实现。KV 块数由命令行或 `EngineParams` 手动指定。

FP16/BF16 的舍入会受 GEMM 的 batch 大小影响；候选分数接近时，生成 token 可能随批量配置变化，也不保证与 Transformers 逐 token 相同。当前融合 residual 和 RMSNorm 的计算顺序沿用 nano-vllm；详细对比结果见测试文档。

## 学习用 MoE

`src/layers/moe.hh/.cc` 实现一个独立的 routed MoE 层，支持 FP32 的 `[T,H]`
输入，`T > 0`。维度由 `MoEConfig` 指定，例如 `H=128, I=64, E=16, K=2`。
`T` 是本次输入的 token 总数；每个 token 的 FFN 独立计算，可以来自不同请求，
不需要 attention 的序列边界或 KV 元数据。

```text
x[T,H] → Router Linear → logits[T,E]
       → 每个 token 独立 softmax、top-k、选中概率重新归一化
       → 逐 token 执行 K 个 SwiGLUMLP
       → 每个 token 按自己的路由权重求和 → y[T,H]
```

`MoE::forward()` 返回最终输出和以下中间结果：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `hidden_states` | `[T,H]` | routed 与可选共享分支的最终输出 |
| `router_logits` | `[T,E]` | softmax 之前的 Router 输出 |
| `expert_ids` | `[T,K]` | 每个 token 按 Router 排名选出的专家编号 |
| `expert_weights` | `[T,K]` | 每行独立归一化，和为 1 |
| `expert_outputs` | `[T*K,H]` | 第 `t*K+rank` 行是 token `t` 的第 `rank` 个专家原始输出 |
| `shared_expert_output` | `[T,H]` | 未乘门控的共享专家输出；关闭分支时为空 |
| `shared_gate_logits` | `[T,1]` | sigmoid 之前的共享门控；关闭分支时为空 |

专家输出按 token、Router 排名排列；路由权重只在 combine 时应用一次。
不同 token 可以选中同一专家。`T=1` 时保留原有形状，包括 `[K,H]` 的专家输出。
权重可通过 `Module::named_parameters()` 初始化；测试展示了完整的运行方式。

全部 routed 专家权重保存在 RAM 中的 `ExpertStore`，模型参数名仍为
`experts.N.gate_proj.weight` 等。GPU 的 `ExpertCache` 只分配固定名单对应的
专家槽位，缓存副本不注册为额外模型参数。Router 在 GPU 上计算，命中的专家
在 GPU 槽位上执行，未命中的专家在 CPU 上执行；CPU 输出复制回 GPU 后统一
combine。Router 的整批专家编号一次下载到 CPU；对每个有未命中的 token，
只下载一次输入向量，由它的全部 CPU 专家复用。

```text
批量 Router → expert_ids[T,K]
            → for each token: ExpertExecutor(input[1,H], ids[1,K])
                ├── ExpertCache 命中 → GPU SwiGLUMLP
                └── 未命中 → ExpertStore → CPU expert_forward → 上传输出
            → 按 token 和排名收集 expert_outputs[T*K,H]
            → CUDA combine → y[T,H]
```

这一版复用单 token `ExpertExecutor`，逐 token 分派，每个 token 执行完才处理
下一个。每次执行返回的专家输出复制到整批输出中；源 Tensor 保留到最终 stream
同步完成，保证异步复制的源仍然有效。这会额外暂存 `T*K*H*sizeof(float)` 字节
的专家输出，尚未实现按专家聚合多个 token 的执行方式。

固定驻留名单通过第三个构造参数传入。例如缓存 4 个专家：

```cpp
MoE moe({128, 64, 16, 2}, Device::cuda(0),
        std::vector<std::int32_t>{0, 2, 4, 6});
```

初始化 `named_parameters()` 返回的权重后，调用
`moe.populate_expert_cache(stream)`，再调用 `moe.forward(x, stream)`。
routed 专家参数在 CPU，Router 参数在所选设备上；复制参数时按目标设备选择
`memcpy` 或 CUDA 复制。RAM 权重修改后需要再次填充缓存。

省略驻留名单（`std::nullopt`）缓存全部 routed 专家；传入显式空向量
`std::vector<std::int32_t>{}` 分配零个 GPU 专家槽位，全部选中 routed 专家在 CPU 执行。
CPU 构造只创建参考权重。`MoEOutput` 的 `gpu_experts`、`cpu_experts` 记录
整批 `T*K` 次 routed 专家选择的执行数量，不包含共享分支，同一专家被不同
token 选中会分别计数；`expert_cache().num_slots()`
和 `size_bytes()` 只统计 routed 缓存槽位和权重，不包含 Router、共享专家及工作区。

当前支持多 token、FP32、1..1024 个专家、`1 <= top_k <= E`。缓存名单固定，
populate/forward 同步传入的 stream，调用需串行。量化、动态换入换出作为后续
步骤；当前文本生成使用原有 Qwen3 模型。

### 可选共享专家

`MoEConfig::shared_expert_intermediate_size` 默认是 `0`，关闭共享分支。
设为正数时，增加一个 SwiGLUMLP 和无 bias 的 `H → 1` Linear：

```text
y = routed_sum + sigmoid(shared_expert_gate(x)) * shared_expert(x)
```

共享专家对整批 token 执行，每个 token 使用自己的门控，不参与 Router 的
top-k，也不参与选中专家概率的归一化。它的中间维度可以与 routed 专家不同。
门控 logit 为 `0` 时，sigmoid
系数是 `0.5`；关闭分支应将配置维度设为 `0`。

```cpp
MoEConfig config{
    .hidden_size = 128,
    .intermediate_size = 64,
    .num_experts = 16,
    .top_k = 2,
    .shared_expert_intermediate_size = 32,
};
MoE moe(config, Device::cuda(0), std::vector<std::int32_t>{0, 2, 4, 6},
        ExpertExecutionMode::Overlap);
```

共享分支直接挂在 `MoE` 下，注册以下参数：

```text
shared_expert.gate_proj.weight   [Is,H]
shared_expert.up_proj.weight     [Is,H]
shared_expert.down_proj.weight   [H,Is]
shared_expert_gate.weight        [1,H]
```

GPU 构造时这四个参数常驻 GPU，不占 routed expert cache 的槽位；CPU 构造时
它们位于 RAM，供独立参考实现使用。共享分支额外的 FP32 权重字节数是
`(3 * H * Is + H) * sizeof(float)`，不含激活和临时工作区。初始化参数仍然通过
`named_parameters()`；`populate_expert_cache()` 只上传 routed 专家的缓存副本。

这一版先完成 routed dispatch 和 combine，再用现有层计算共享专家、门控，并
通过 `kernels::cuda::moe_add_shared()` 原地加入输出。同步和 Overlap 模式使用
相同共享分支；共享分支尚未加入 CPU/GPU 计算重叠，工作区使用现有层的临时分配。

`MoEOutput::shared_expert_output` 返回未经门控的 `[T,H]` 输出，
`shared_gate_logits` 返回 sigmoid 之前的 `[T,1]` logits；关闭分支时两者为空。
最终 `hidden_states` 包含 routed 和共享两部分。CPU 参考接口的可选
`SharedExpertWeights` 传入这四个权重，默认省略时保持原有 routed 计算。

### CPU/GPU 重叠执行

第四个构造参数选择执行方式，默认 `ExpertExecutionMode::Synchronous` 保留
按 Router 排名逐个分派的路径。开启重叠执行：

```cpp
MoE moe({128, 64, 16, 2}, Device::cuda(0),
        std::vector<std::int32_t>{0, 2, 4, 6},
        ExpertExecutionMode::Overlap);
```

对每个 token，Overlap 先划分 GPU hits / CPU misses，有 misses 时下载一次
该 token 的输入并等待完成，随后提交它的全部 GPU hits，再由调用线程计算 CPU
misses。CPU 输出上传到原始排名对应的行；一个 stream 保证 GPU 专家、上传和
combine 的顺序。重叠发生在同一个 token 内，token 之间仍逐个执行。`forward()`
完成后才返回，同一个对象的调用仍需串行，完成后可以改用另一个 stream。

`SwiGLUMLP::forward_into()` 使用预先分配的 gate/up/activated 工作区，直接
写入输出行。所有 GPU 专家共用这三个缓冲区，依靠同一 stream 排序；提交 GPU
专家时不会分配或释放 CUDA Tensor。CPU 输入与输出暂存区也会复用，输出区
按需要增长。返回的输出拥有独立存储，后续调用不会覆盖先前结果。

这一版只重叠 CPU/GPU 的计算：输入下载在 GPU hits 之前完成，CPU 输出在
CPU 计算结束后上传，因此暂存区使用普通 RAM。CPU 专家在调用线程串行计算。
数学结果和执行时间线的检查方法见测试文档。

三个辅助类与 `MoE` 一起放在 `src/layers/moe.hh/.cc`。建议在 `.cc` 中按
`ExpertStore` 的 RAM 权重 → `ExpertCache` 的固定 GPU 槽位 →
`ExpertExecutor` 的分派 → `MoE::forward` 的路由与合并阅读。CPU 参考实现
继续独立使用 double 点积累加，运行时 CPU 专家使用 FP32 累加。

## 验证

```bash
ctest --test-dir build --output-on-failure
```

小尺寸 MoE 的 CPU/GPU 对照和 CUDA 内存检查方法见 [tests/README.md](tests/README.md)。
