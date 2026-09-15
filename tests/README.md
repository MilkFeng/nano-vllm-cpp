# Tests

Build with tests enabled and a CUDA architecture supported by your GPU:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For example, an RTX 5060 Ti uses `-DCMAKE_CUDA_ARCHITECTURES=120`.
The existing project's dependencies/toolchain must already be configured.

## CTest coverage

| Test | Coverage |
| --- | --- |
| `kernels` | First batch: RMSNorm, fused residual + RMSNorm, SiLU + multiply, embedding and RoPE. CPU numerical references, strides, in-place execution, rounding, empty inputs and invalid metadata. |
| `inference_kernels` | Second batch: cuBLAS Linear with optional bias, last-token gather and sampling. Non-square/strided GEMM, bias rounding, unequal sequence lengths, greedy ties, deterministic sampling and temperature-dependent frequencies. |
| `attention_kernels` | Second batch: KV cache writes, packed/cached prefill and paged decode. CPU references, GQA, causal masks, cached prefixes/chunks, strides and empty inputs. Fixed Q/K/V also compare full/chunked execution, shared prefixes and reordered batches with exact equality. |
| `model-kernels` | Qwen3 embedding and normalization wrappers, plus tensor-layout validation. |
| `scheduler` | CPU scheduling/block management: chunked prefill, EOS/length limits, prefix reuse, shared blocks, memory pressure/preemption and invalid requests. Shadow KV slots check full causal prefixes and prevent writes to shared blocks. |

The three kernel suites cover FP32, FP16 and BF16. GPU tests use a non-default
CUDA stream and return CTest's skip code 77 if a CUDA device is unavailable;
`scheduler` runs on the CPU.

## GPU memory checks

To check GPU memory accesses:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-kernels-test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-inference_kernels-test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-attention_kernels-test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/nano-vllm-cpp-model-kernels-test
```

Kernel APIs and layout/lifetime requirements are documented in
[`kernels.hh`](../src/kernels/kernels.hh),
[`inference.hh`](../src/kernels/inference.hh) and
[`attention.hh`](../src/kernels/attention.hh). Host metadata errors throw;
device-side index/value preconditions use debug assertions without adding host
synchronization. Tests exercise valid device data and invalid host metadata.

## Local Qwen3 generation against Transformers

This optional integration check requires a local Qwen3 checkpoint, CUDA-enabled
PyTorch and Transformers. It is separate from CTest and does not download models.

```sh
python tests/verify_generation.py \
  --model ~/models/Qwen3-0.6B \
  --binary ./build/nano-vllm-cpp
```

For the existing local environment, use
`/home/feng/miniconda3/envs/nanovllm/bin/python` in place of `python`.

The script uses three raw prompts of different lengths, including a shared
prefix, and generates up to eight tokens with temperature zero. It compares
ordinary prefill, chunked prefill, and a second generation pass on the same
engine (`--repeat 2`) against single-prompt HF greedy decoding. Native processes
finish before the HF model loads, avoiding simultaneous model allocations.

HF uses the checkpoint dtype, eager attention, and
`tokenizer.encode(add_special_tokens=False)` without a chat template. Every mode
must produce identical generated token IDs. A mismatch fails the check and
reports the first differing token, HF's top-two logit margin, and the HF logit
for the C++ choice; close logits do not relax equality. Use repeated `--prompt`
arguments to override the defaults, or `--max-tokens` to change the output length.

### Recorded local results (2026-09-15)

Environment: RTX 5060 Ti, system CUDA 13.3, PyTorch 2.11.0+cu130 and
Transformers 5.15.1; local Qwen3-0.6B checkpoint.

- All five CTests pass. The new inference and attention suites also pass
  Compute Sanitizer memcheck with zero errors.
- Casting the same stored checkpoint tensors to FP32 and setting its config
  dtype to `float32` gives exact agreement with HF for all three prompts and
  all three execution modes (eight generated tokens per prompt).
- With the original BF16 checkpoint, ordinary prefill matches all three HF
  outputs. Chunked prefill and prefix reuse diverge on the sixth generated
  token of the first prompt: `Italy` instead of `France`. HF's two candidate
  logits are 16.5 and 16.625. **The strict BF16 comparison currently exits 1**;
  the test does not suppress this difference or relax token equality.

The divergence also occurs in batches without shared prefixes. Fixed-input
attention remains exactly equal across full/chunked execution and batch sizes
1/2/3, while shadow-cache tests verify slot contents and write ownership.
An independent cuBLAS probe finds that changing the batch size can change
BF16 GEMM rounding even with FP32 reductions. These findings are consistent
with accumulated rounding differences; they do not establish bitwise BF16
equivalence with HF. The fused residual/RMSNorm calculation also follows
nano-vllm's FP32-add semantics, which differ from separate BF16 residual adds.
