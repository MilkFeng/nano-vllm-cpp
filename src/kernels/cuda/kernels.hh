#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

// input @ weight^T + bias
void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias = std::nullopt,
            cudaStream_t stream = nullptr);

// hidden state of the last token of each sequence
//
// hidden:       [num_tokens, hidden_size]
// cu_seqlens_q: [batch + 1]
// output:       [batch, hidden_size]
//
// output[seq][i] = hidden[cu_seqlens_q[seq + 1] - 1][i]
void gather_last_hidden(ConstTensorView hidden, ConstTensorView cu_seqlens_q,
                        TensorView output, cudaStream_t stream = nullptr);

// logits:       [batch, vocab_size]
// temperatures: [batch]
// output:       [batch]
//
// for scores, temperature in logits, temperatures:
//   if temperature == 0:
//     token = argmax(scores)
//   else:
//     centered = (scores - max(scores)) / temperature
//     uniform = Random.uniform(size=len(scores))
//     exponential = maximum(-log(uniform), 1e-10)
//     token = argmax(centered - log(exponential))
//   output.append(token)
void sample(ConstTensorView logits, ConstTensorView temperatures,
            TensorView output, std::uint64_t seed, std::uint64_t step,
            cudaStream_t stream = nullptr);

// causal depthwise convolution + SiLU
//
// input/output:  [tokens, channels]
// weight:        [channels, 1, kernel_size]
// state:         [slots, channels, kernel_size - 1]
// cu_seqlens:    [batch + 1]
// state_indices: [batch]
//
// for token in a sequence:
//   current = input[token, channel]
//
//   # conv
//   total = current * weight[channel, K - 1]
//   for j in range(K - 1):
//     total += history[j] * weight[channel, j]
//
//   # update history (state[slots, channels, :])
//   history[:-1] = history[1:]
//   history[-1] = current
//
//   # apply SiLU
//   output[token, channel] = silu(total)
void causal_conv1d(ConstTensorView input, ConstTensorView weight,
                   TensorView state, ConstTensorView cu_seqlens,
                   ConstTensorView state_indices, TensorView output,
                   cudaStream_t stream = nullptr);

// gated delta rule
//
// qkv:           [tokens, 2 * key_heads * key_dim + value_heads * value_dim]
//                ([[Q Heads, K Heads, V Heads], ...])
// a/b:           [tokens, value_heads]
// A_log/dt_bias: [value_heads]
// state:         [slots, value_heads, key_dim, value_dim]
// cu_seqlens:    [batch + 1]
// state_indices: [batch]
// output:        [tokens, value_heads, value_dim]
//
// for token in sequence:
//     q = normalize(q) / sqrt(key_dim)
//     k = normalize(k)
//
//     decay = exp(-exp(A_log) * softplus(a + dt_bias))
//     beta = sigmoid(b)
//
//     state *= decay
//     prediction = k @ state
//     delta = beta * (v - prediction)
//
//     state += outer(k, delta)
//     output[token] = q @ state
void gated_delta_rule(ConstTensorView qkv, ConstTensorView a, ConstTensorView b,
                      ConstTensorView A_log, ConstTensorView dt_bias,
                      TensorView state, ConstTensorView cu_seqlens,
                      ConstTensorView state_indices, std::size_t key_heads,
                      TensorView output, cudaStream_t stream = nullptr);

// input:  [tokens, hidden_size]
// weight: [hidden_size]
// output: [tokens, hidden_size]
//
// if zero-centered:
//   output = RMSNorm(input) * (1 + weight)
// else:
//   output = RMSNorm(input) * weight
void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream = nullptr,
              bool zero_centered = false);

// input:           [tokens, hidden_size]
// residual:        [tokens, hidden_size]
// weight:          [hidden_size]
// residual_output: [tokens, hidden_size]
// output:          [tokens, hidden_size]
//
// residual_output = input + residual
// output = RMSNorm(residual_output) * weight
void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream = nullptr,
                        bool zero_centered = false);

// input:  [tokens, hidden_size]
// gate:   [hidden_size]
// weight: [hidden_size]
// output: [tokens, hidden_size]
//
// output = RMSNorm(input) * weight * SiLU(gate)
void gated_rms_norm(ConstTensorView input, ConstTensorView gate,
                    ConstTensorView weight, TensorView output, float epsilon,
                    cudaStream_t stream = nullptr);

// gate/up/output: [tokens, intermediate_size]
//
// output = SiLU(gate) * up
void silu_and_mul(ConstTensorView gate, ConstTensorView up, TensorView output,
                  cudaStream_t stream = nullptr);

// input_ids: [tokens]
// weight:    [vocab_size, hidden_size]
// output:    [tokens, hidden_size]
//
// output[row][col] = weight[input_ids[row]][col];
void embedding(ConstTensorView input_ids, ConstTensorView weight,
               TensorView output, cudaStream_t stream = nullptr);

void build_rope_cache(TensorView cos_sin_cache, float theta,
                      cudaStream_t stream = nullptr);

void apply_rotary_embedding(ConstTensorView positions, TensorView query,
                            TensorView key, ConstTensorView cos_sin_cache,
                            cudaStream_t stream = nullptr);

// packed: [tokens, heads * 2 * head_dim] ([[Q, G, Q, G, ...], ...])
// query:  [tokens, heads, head_dim]
// gate:   [tokens, heads, head_dim]
void split_attention_gate(ConstTensorView packed, TensorView query,
                          TensorView gate, cudaStream_t stream = nullptr);

// gate/output: [tokens, heads, head_dim]
//
// output = Sigmoid(gate)
void apply_sigmoid_gate(TensorView output, ConstTensorView gate,
                        cudaStream_t stream = nullptr);

constexpr std::size_t kMaxMoEExperts = 1024;

void moe_route(ConstTensorView logits, TensorView expert_ids,
               TensorView expert_weights, cudaStream_t stream = nullptr);

void moe_combine(ConstTensorView expert_outputs, ConstTensorView expert_weights,
                 TensorView output, cudaStream_t stream = nullptr);

void moe_add_shared(ConstTensorView shared_output, ConstTensorView gate_logits,
                    TensorView output, cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
