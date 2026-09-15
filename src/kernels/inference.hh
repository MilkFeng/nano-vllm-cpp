#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

#include "model/tensor_view.hh"

namespace nanovllm::kernels {

// Floating point tensors use Float32, Float16, or BFloat16 on one CUDA device.
// Outputs are contiguous and may not overlap inputs. Operations enqueue work
// on stream; callers retain buffers until it completes. Invalid host metadata
// throws std::invalid_argument. Invalid device index/temperature values are
// device-side precondition failures and never require a host synchronization.

// X: [M,K], W: [N,K], Y: [M,N], optional bias: [N]. Computes X @ W^T + bias
// with FP32 accumulation. X may have a padded row stride; W is contiguous.
// FP32 reduction order can vary with GEMM batch shape, changing output rounding.
// The first call on a host thread/device initializes a reusable cuBLAS handle;
// subsequent calls reuse it. Initialize before CUDA Graph capture.
void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias = std::nullopt,
            cudaStream_t stream = nullptr);

// hidden: [T,H] with contiguous columns, cu_seqlens_q: contiguous Int32 [B+1],
// output: [B,H]. Boundaries start at 0, end at T, and are strictly increasing;
// each output row gathers hidden[cu_seqlens_q[b+1]-1]. B=T=0 is supported.
void gather_last_hidden(ConstTensorView hidden, ConstTensorView cu_seqlens_q,
                        TensorView output, cudaStream_t stream = nullptr);

// logits: [B,V] with contiguous columns; temperatures: contiguous Float32 [B];
// output: contiguous Int32 [B]. Temperatures are finite and nonnegative.
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
