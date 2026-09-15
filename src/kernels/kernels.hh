#pragma once

#include <cuda_runtime_api.h>

#include "model/tensor_view.hh"

namespace nanovllm::kernels {

// All operations enqueue work on stream without allocating or synchronizing.
// Floating point inputs/outputs use Float32, Float16 or BFloat16 on one CUDA
// device. Outputs are contiguous; callers retain buffers until work completes.
// Shape/dtype/layout errors throw std::invalid_argument; CUDA errors throw
// std::runtime_error. Indices must be in range (a device-side precondition).
// Partial input/output overlap is unsupported. Exact in-place use is allowed
// for normalization, activation and RoPE when the aliased input is contiguous.

// input: [D], [T,D] or [T,H,D]; weight: [D]. Last dimension is contiguous;
// rank-3 heads are contiguous, while the token/row stride may include padding.
// Accumulation is FP32. The normalized value is rounded to the input dtype
// before multiplying by weight, matching nano-vllm's RMSNorm.
void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream = nullptr);

// residual has the same shape as input. Both outputs have that shape and must
// be disjoint. Statistics use the FP32 sum; residual_output stores its rounded
// value. output may alias input and residual_output may alias residual.
void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream = nullptr);

// gate, up, output: [T,I]. Inputs may be slices of a packed [T,2I] tensor,
// with contiguous columns and independent row strides.
void silu_and_mul(ConstTensorView gate, ConstTensorView up, TensorView output,
                  cudaStream_t stream = nullptr);

// ids: contiguous Int32 [T], in [0,V); weight: contiguous [V,D]; output: [T,D].
// Output must not overlap either input.
void embedding(ConstTensorView input_ids, ConstTensorView weight,
               TensorView output, cudaStream_t stream = nullptr);

// cache: Float32 [max_positions,D], D positive and even. Stores cos in the
// first D/2 columns, sin in the second; theta must be finite and positive.
void build_rope_cache(TensorView cos_sin_cache, float theta,
                      cudaStream_t stream = nullptr);

// positions: contiguous Int32 [T], in [0,max_positions).
// query/key: [T,Nq,D] / [T,Nkv,D]. Each token's heads are contiguous;
// token strides may include padding. Each output has its input's shape.
// Rotates pairs (i, i+D/2), using the FP32 cache built above.
void apply_rotary_embedding(ConstTensorView positions, ConstTensorView query,
                            ConstTensorView key, ConstTensorView cos_sin_cache,
                            TensorView query_output, TensorView key_output,
                            cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels
