#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

// FP32, contiguous CUDA 0 tensors. No allocation/synchronization. Calls on
// the same state are serialized. cu_seqlens: Int32 [B+1] for packed prefill;
// empty for decode (one token per sequence). Optional Int32 state_indices[B]
// maps batch rows to persistent state slots; empty means slot = batch row.
// Each used slot is distinct and in range. Caller resets new sequence slots.

// input/output: [T,C], weight: [C,1,K], state: [slots,C,K-1]. Applies causal
// depthwise convolution followed by SiLU. State stores raw projected inputs,
// oldest first, and is updated after each token. K=1 needs no history.
void causal_conv1d(ConstTensorView input, ConstTensorView weight,
                   TensorView state, ConstTensorView cu_seqlens,
                   ConstTensorView state_indices, TensorView output,
                   cudaStream_t stream = nullptr);

// qkv: [T,2*Hk*Dk+Hv*Dv], a/b: [T,Hv], A_log/dt_bias: [Hv],
// state: [slots,Hv,Dk,Dv], output: [T,Hv,Dv]. Dk,Dv in [1,256], Hv % Hk=0.
// Q/K L2 normalization uses eps=1e-6; repeats each key head Hv/Hk times.
// q /= sqrt(Dk); decay = exp(-exp(A_log)*softplus(a+dt_bias)); beta=sigmoid(b).
// S *= decay; delta=(v-k^T*S)*beta; S += k*delta^T; output=q^T*S.
// Prefill and decode deliberately share this simple sequential recurrence.
void gated_delta_rule(ConstTensorView qkv, ConstTensorView a, ConstTensorView b,
                      ConstTensorView A_log, ConstTensorView dt_bias,
                      TensorView state, ConstTensorView cu_seqlens,
                      ConstTensorView state_indices, std::size_t key_heads,
                      TensorView output, cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
