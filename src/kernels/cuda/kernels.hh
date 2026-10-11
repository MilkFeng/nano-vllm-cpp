#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

// All operations enqueue work on stream without allocating or synchronizing.
// All tensors are contiguous on CUDA device 0, which the caller selects.
// Floating point tensors use Float32, Float16 or BFloat16. Shape/dtype/index
// preconditions use debug assertions; CUDA errors throw std::runtime_error.
// Callers retain buffers until work completes and prevent partial overlap.
// Normalization and activation allow exact in-place use; RoPE is in-place.

// input: [D], [T,D] or [T,H,D]; weight: [D]; output matches input shape.
// Accumulation is FP32. The normalized value is rounded to the input dtype
// before multiplying by weight, matching nano-vllm's RMSNorm. With
// zero_centered=true, multiply by (1 + weight) in FP32 before the output cast.
void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream = nullptr,
              bool zero_centered = false);

// residual has the same shape as input. Both outputs have that shape and must
// be disjoint. Statistics use the FP32 sum; residual_output stores its rounded
// value. zero_centered=true also rounds the sum before computing statistics,
// matching Qwen3.5 residual addition. Outputs may alias their respective
// inputs.
void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream = nullptr,
                        bool zero_centered = false);

// FP32 input/gate/output: [D], [T,D] or [T,H,D]; weight: FP32 [D].
// output = RMSNorm(input) * weight * SiLU(gate). Weight is used directly.
void gated_rms_norm(ConstTensorView input, ConstTensorView gate,
                    ConstTensorView weight, TensorView output, float epsilon,
                    cudaStream_t stream = nullptr);

// gate, up, output: [T,I].
void silu_and_mul(ConstTensorView gate, ConstTensorView up, TensorView output,
                  cudaStream_t stream = nullptr);

// ids: contiguous Int32 [T], in [0,V); weight: contiguous [V,D]; output: [T,D].
// Output must not overlap either input.
void embedding(ConstTensorView input_ids, ConstTensorView weight,
               TensorView output, cudaStream_t stream = nullptr);

// cache: Float32 [max_positions,R], R positive and even. Stores cos in the
// first R/2 columns, sin in the second; theta must be finite and positive.
void build_rope_cache(TensorView cos_sin_cache, float theta,
                      cudaStream_t stream = nullptr);

// positions: contiguous Int32 [T], in [0,max_positions).
// query/key: [T,Nq,D] / [T,Nkv,D], disjoint from each other, cache and
// positions. Cache width R <= D; rotates pairs (i, i+R/2) in the first
// R columns using the FP32 cache above. Columns [R,D) are untouched.
void apply_rotary_embedding(ConstTensorView positions, TensorView query,
                            TensorView key, ConstTensorView cos_sin_cache,
                            cudaStream_t stream = nullptr);

inline constexpr std::size_t kMaxMoEExperts = 1024;

// Per-token routing. Finite Float32 logits: [T,E], T>0, 1 <= E <= 1024.
// expert_ids: Int32 [T,K]; expert_weights: Float32 [T,K], 1 <= K <= E.
// One CUDA block per token; T must fit CUDA's grid.x limit (INT32_MAX).
// Computes softmax over all experts, picks top-k by score (lowest ID on ties),
// and renormalizes selected probabilities to sum to 1. All views are disjoint.
void moe_route(ConstTensorView logits, TensorView expert_ids,
               TensorView expert_weights, cudaStream_t stream = nullptr);

// expert_outputs: Float32 [T*K,H], row t*K+rank, still unweighted.
// expert_weights: Float32 [T,K]; output: Float32 [T,H]. T,K,H > 0.
// Applies each token's router weights exactly once. All views are disjoint.
void moe_combine(ConstTensorView expert_outputs, ConstTensorView expert_weights,
                 TensorView output, cudaStream_t stream = nullptr);

// In-place addition: output += sigmoid(gate_logits) * shared_output.
// shared_output/output: Float32 [T,H]; gate_logits: Float32 [T,1], finite.
// T,H > 0; each token uses its own gate. All views are disjoint.
void moe_add_shared(ConstTensorView shared_output, ConstTensorView gate_logits,
                    TensorView output, cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
