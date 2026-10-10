#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

#include "core/tensor_view.hh"

namespace nanovllm::kernels {

// All tensors are contiguous on CUDA device 0, which the caller selects.
// Floating point tensors use Float32, Float16 or BFloat16. Outputs do not
// overlap inputs. Preconditions use debug assertions. Operations enqueue work
// on stream; callers retain buffers until it completes.

// X: [M,K], W: [N,K], Y: [M,N], optional bias: [N]. Computes X @ W^T + bias
// with FP32 accumulation.
// FP32 reduction order can vary with GEMM batch shape, changing output
// rounding. The first call on a host thread initializes a reusable
// cuBLAS handle; subsequent calls reuse it. Initialize before CUDA Graph
// capture.
void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias = std::nullopt,
            cudaStream_t stream = nullptr);

// hidden: [T,H], cu_seqlens_q: Int32 [B+1],
// output: [B,H]. Boundaries start at 0, end at T, and are strictly increasing;
// each output row gathers hidden[cu_seqlens_q[b+1]-1]. B=T=0 is supported.
void gather_last_hidden(ConstTensorView hidden, ConstTensorView cu_seqlens_q,
                        TensorView output, cudaStream_t stream = nullptr);

// logits: [B,V]; temperatures: Float32 [B]; output: Int32 [B].
// Temperatures are finite and nonnegative.
// A zero temperature selects the largest logit (lowest index breaks ties).
// Positive temperatures use the exponential-race sampler from nano-vllm,
// evaluated in log space with Philox(seed, flat token index, step). Logits may
// include -infinity, but each row must have at least one finite value and no
// NaN/+infinity. Equal seed/step/input reproduce the same results; advance step
// for a new draw. Random bit sequences are not identical to PyTorch's sampler.
void sample(ConstTensorView logits, ConstTensorView temperatures,
            TensorView output, std::uint64_t seed, std::uint64_t step,
            cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels
