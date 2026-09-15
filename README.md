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

## 已实现链路

```text
Token IDs → Embedding
  → [RMSNorm → Q/K/V Linear → Q/K Norm → RoPE
     → KV Cache → Causal Attention → O Linear
     → Residual + RMSNorm → Gate/Up Linear → SiLU × Up → Down Linear] × layers
  → Residual + RMSNorm → Last-token Gather → LM Head → Sampling → Token IDs
```

基础算子支持 FP32、FP16 和 BF16。Linear 使用 cuBLAS；attention 使用 FP32 在线 softmax，支持 GQA、变长 prefill、缓存前缀和分页 decode。引擎包含分块 prefill、KV 块管理、前缀缓存、显存不足时的请求抢占和停止条件处理。

当前支持 CUDA device 0 上的 dense Qwen3、单文件 safetensors 权重、未缩放的 RoPE，以及偶数且不超过 256 的 head dimension。Attention 是用于打通计算链路的基础实现；FlashAttention 级优化、CUDA Graph、张量并行和分片权重加载仍待实现。KV 容量根据可用显存和工作区估算确定。

FP16/BF16 的舍入会受 GEMM 的 batch 大小影响；候选分数接近时，生成 token 可能随批量配置变化，也不保证与 Transformers 逐 token 相同。当前融合 residual 和 RMSNorm 的计算顺序沿用 nano-vllm；详细对比结果见测试文档。

## 验证

```bash
ctest --test-dir build --output-on-failure
```

算子覆盖、CUDA 内存检查和真实模型对比方法见 [tests/README.md](tests/README.md)。
