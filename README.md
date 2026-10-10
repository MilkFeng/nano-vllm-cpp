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
  --max-model-len 128 --max-seqs 4 --max-batched-tokens 128
```

- `--prompt` 默认是一条用户消息，程序读取模型的 `chat_template` 并自动应用。重复 `--prompt` 可批量提交独立对话，输出顺序与输入一致。
- 默认关闭思考模式；`--thinking` 开启模型模板中的 `enable_thinking`。
- `--raw-prompt` 用于原始文本续写或已经格式化的输入，与 `--thinking` 互斥。
- `--temperature 0` 使用 greedy 解码，正数使用温度采样；`--seed` 设置随机种子。
- `--max-batched-tokens` 限制每轮 token 数，较长输入会分块 prefill。
- `--json` 输出生成文本和 token ID；`--repeat 2` 在同一引擎中重复请求，以使用前缀缓存。
- 达到 EOS、`--max-tokens` 或总长度 `--max-model-len` 时结束；`--ignore-eos` 可忽略 EOS。

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

当前支持 CUDA device 0 上的 dense Qwen3、单文件 safetensors 权重、未缩放的 RoPE，以及偶数且不超过 256 的 head dimension。Attention 是用于打通计算链路的基础实现；FlashAttention 级优化、CUDA Graph、张量并行和分片权重加载仍待实现。KV 容量根据可用显存和工作区估算确定。

FP16/BF16 的舍入会受 GEMM 的 batch 大小影响；候选分数接近时，生成 token 可能随批量配置变化，也不保证与 Transformers 逐 token 相同。当前融合 residual 和 RMSNorm 的计算顺序沿用 nano-vllm；详细对比结果见测试文档。

## 学习用 MoE

`src/layers/moe.hh/.cc` 实现一个独立的 routed MoE 层，先支持单 token、FP32，
维度由 `MoEConfig` 指定。例如 `H=128, I=64, E=16, K=2`：

```text
x[1,H] → Router Linear → softmax → top-k → 选中概率重新归一化
       → K 个 SwiGLUMLP → 按路由权重求和 → y[1,H]
```

`MoE::forward()` 返回最终输出，以及 Router logits、专家编号、路由权重和
未加权的专家输出。专家输出的行顺序与 Router 的排名一致；路由权重只在
combine 时应用一次。权重可通过 `Module::named_parameters()` 初始化；测试
展示了完整的权重初始化和运行方式。

全部专家权重保存在 RAM 中的 `ExpertStore`，模型参数名仍为
`experts.N.gate_proj.weight` 等。GPU 的 `ExpertCache` 只分配固定名单对应的
专家槽位，缓存副本不注册为额外模型参数。Router 在 GPU 上计算，命中的专家
在 GPU 槽位上执行，未命中的专家在 CPU 上执行；CPU 输出复制回 GPU 后统一
combine。有未命中时，每次 forward 只下载一次输入向量。

```text
Router → expert_ids
       → ExpertExecutor
           ├── ExpertCache 命中 → GPU SwiGLUMLP
           └── 未命中 → ExpertStore → CPU expert_forward → 上传输出
       → CUDA combine → y
```

固定驻留名单通过第三个构造参数传入。例如缓存 4 个专家：

```cpp
MoE moe({128, 64, 16, 2}, Device::cuda(0),
        std::vector<std::int32_t>{0, 2, 4, 6});
```

初始化 `named_parameters()` 返回的权重后，调用
`moe.populate_expert_cache(stream)`，再调用 `moe.forward(x, stream)`。
专家参数在 CPU，Router 参数在所选设备上；复制参数时按目标设备选择
`memcpy` 或 CUDA 复制。RAM 权重修改后需要再次填充缓存。

省略驻留名单（`std::nullopt`）缓存全部专家；传入显式空向量
`std::vector<std::int32_t>{}` 分配零个 GPU 专家槽位，全部专家在 CPU 执行。
CPU 构造只创建参考权重。`MoEOutput` 的 `gpu_experts`、`cpu_experts` 记录
本次选中专家的执行数量；`expert_cache().num_slots()` 和 `size_bytes()`
可以检查缓存容量和实际 GPU 专家权重字节数。

当前支持单 token、FP32、1..1024 个专家、`1 <= top_k <= E`。缓存名单固定，
populate/forward 同步传入的 stream，调用需串行。量化、共享专家、动态换入
换出和 CPU/GPU 并行作为后续步骤；当前文本生成使用原有 Qwen3 模型。

三个辅助类与 `MoE` 一起放在 `src/layers/moe.hh/.cc`。建议在 `.cc` 中按
`ExpertStore` 的 RAM 权重 → `ExpertCache` 的固定 GPU 槽位 →
`ExpertExecutor` 的分派 → `MoE::forward` 的路由与合并阅读。CPU 参考实现
继续独立使用 double 点积累加，运行时 CPU 专家使用 FP32 累加。

## 验证

```bash
ctest --test-dir build --output-on-failure
```

小尺寸 MoE 的 CPU/GPU 对照和 CUDA 内存检查方法见 [tests/README.md](tests/README.md)。
