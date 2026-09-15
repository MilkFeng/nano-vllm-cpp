# nano-vllm-cpp

使用 C++ 和 CUDA 实现 nano-vllm 的单 GPU Qwen3 推理链路。

## 构建

需要 CMake 3.28+、支持 C++23 的主机编译器、CUDA Toolkit、Rust/Cargo，以及
`vcpkg.json` 中列出的 C++ 依赖。首次配置会获取 tokenizers-cpp。

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
```

将架构 `120` 改为实际 GPU 对应值。C++ 源文件使用 C++23，CUDA 源文件使用 C++20。

## 生成文本

模型目录需要包含 `config.json`、`model.safetensors`、`tokenizer.json` 和
`tokenizer_config.json`。

```bash
./build/nano-vllm-cpp --model "$HOME/models/Qwen3-0.6B" \
  --prompt 'The capital of France is' \
  --temperature 0 --max-tokens 16 \
  --max-model-len 128 --max-seqs 4 --max-batched-tokens 128
```

- 重复 `--prompt` 可批量提交请求，输出顺序与输入一致。
- 输入是原始文本；使用对话格式时，需要自行应用模型的 chat template。
- `--temperature 0` 使用 greedy 解码，正数使用温度采样；`--seed` 设置随机种子。
- `--max-batched-tokens` 限制每轮 token 数，较长输入会分块 prefill。
- `--json` 输出生成文本和 token ID；`--repeat 2` 在同一引擎中重复请求，以使用前缀缓存。
- 达到 EOS、`--max-tokens` 或总长度 `--max-model-len` 时结束；`--ignore-eos` 可忽略 EOS。

C++ 调用入口为 [LLMEngine::generate](include/nanovllm/nanovllm.hh)。同一个引擎的
`generate()` 应串行调用，跨调用保留可复用的完整 KV 块。

## 已实现链路

```text
Token IDs → Embedding
  → [RMSNorm → Q/K/V Linear → Q/K Norm → RoPE
     → KV Cache → Causal Attention → O Linear
     → Residual + RMSNorm → Gate/Up Linear → SiLU × Up → Down Linear] × layers
  → Residual + RMSNorm → Last-token Gather → LM Head → Sampling → Token IDs
```

基础算子支持 FP32、FP16 和 BF16。Linear 使用 cuBLAS；attention 使用 FP32
在线 softmax，支持 GQA、变长 prefill、缓存前缀和分页 decode。引擎包含分块
prefill、KV 块管理、前缀缓存、显存不足时的请求抢占和停止条件处理。

当前支持 CUDA device 0 上的 dense Qwen3、单文件 safetensors 权重、未缩放的
RoPE，以及偶数且不超过 256 的 head dimension。Attention 是用于打通计算链路的
基础实现；FlashAttention 级优化、CUDA Graph、张量并行和分片权重加载仍待实现。
KV 容量根据可用显存和工作区估算确定。

FP16/BF16 的舍入会受 GEMM 的 batch 大小影响；候选分数接近时，生成 token
可能随批量配置变化，也不保证与 Transformers 逐 token 相同。当前融合 residual
和 RMSNorm 的计算顺序沿用 nano-vllm；详细对比结果见测试文档。

## 验证

```bash
ctest --test-dir build --output-on-failure
```

算子覆盖、CUDA 内存检查和真实模型对比方法见 [tests/README.md](tests/README.md)。
