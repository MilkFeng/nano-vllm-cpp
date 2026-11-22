#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>
#include <span>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

// Convert contiguous floating-point tensors, rounding only at the destination.
void cast(ConstTensorView input, TensorView output,
          cudaStream_t stream = nullptr);

// input @ weight^T + bias
void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias = std::nullopt,
            cudaStream_t stream = nullptr);

// input:            FP32/BF16 [M, K]
// weight:           FP8  [N, K]
// weight_scale_inv: FP32 [N/128, K/128]
// output:           same as input, or FP32 [M, N]; FP32 accumulation
//
// weight_dequant[i][j] = weight[i][j] * weight_scale_inv[i/128][j/128]
// input @ weight_dequant^T + bias
void linear_fp8(ConstTensorView input, ConstTensorView weight,
                ConstTensorView weight_scale_inv, TensorView output,
                cudaStream_t stream = nullptr);

struct FP8ExpertWeights {
  const std::uint8_t *gate, *up, *down;
  const float *gate_scale, *up_scale, *down_scale;
};

struct FP8ExpertBatch {
  FP8ExpertWeights weights[8];
  unsigned rows[8];
  unsigned count = 0;
};

// All resident decode experts in two launches: fused Gate/Up/SiLU, then Down.
// input/activated share FP32 or BF16 dtype; output parts are FP32 for merging.
void fp8_experts_decode(ConstTensorView input, const FP8ExpertBatch& batch,
                        TensorView activated, TensorView output,
                        cudaStream_t stream = nullptr);

void fp8_experts_decode_routed(ConstTensorView input, ConstTensorView ids,
                               ConstTensorView routes, TensorView activated,
                               TensorView output, cudaStream_t stream);

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
// input/weight/state/output share FP32 or BF16 dtype; accumulation is FP32.
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
// qkv/output may be BF16; a/b/A_log/dt_bias/state and internal math are FP32.
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
// residual/residual_output may be FP32 with BF16 input/output. The residual
// input and output may use different dtypes (e.g. a BF16 residual input that
// starts an FP32 residual stream).
// if zero-centered:
//   output = RMSNorm(residual_output) * (1 + weight)
// else:
//   output = RMSNorm(residual_output) * weight
void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream = nullptr,
                        bool zero_centered = false);

// input:  [tokens, hidden_size]
// gate:   [hidden_size, hidden_size]
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

// cos_sin_cache: [max_positions, rotary_dim]
//                ([cos(0, 0), ..., cos(0, rotary_dim/2−1),
//                  sin(0, 0), ..., sin(0, rotary_dim/2−1)], ...)
//
// phi(position, i) = position * theta^(-2i/rotary_dim)
void build_rope_cache(TensorView cos_sin_cache, float theta,
                      cudaStream_t stream = nullptr);

// apply rotary embedding for query/key
//
// positions:     [tokens]
// query:         [q_tokens, query_heads, head_dim]
// key            [tokens, key_heads, head_dim]
// cos_sin_cache: [max_positions, rotary_dim]
//
// def rotate(x, positions, cache):
//   H = cache.shape[1] // 2
//   for t in range(tokens):
//     p = positions[t]
//     for h in range(x.shape[1]):
//       for i in range(H):
//         a = float(x[t, h, i]), b = float(x[t, h, i + H])
//         c = cache[p, i],       s = cache[p, i + H]
//         x[t, h, i]     = a * c - b * s
//         x[t, h, i + H] = b * c + a * s
// rotate(query, positions, cache)
// rotate(key, positions, cache)
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
// output *= Sigmoid(gate)
void apply_sigmoid_gate(TensorView output, ConstTensorView gate,
                        cudaStream_t stream = nullptr);

// logits:         [tokens, experts]
// expert_ids:     [tokens, topk]
// expert_weights: [tokens, topk]
//
// logits of a token -> Softmax -> TopK -> norm -> experts and weights
void moe_route(ConstTensorView logits, TensorView expert_ids,
               TensorView expert_weights, cudaStream_t stream = nullptr);

// input:   [T,H]
// order:   [G] Int32
// grouped: [G,H]
//
// grouped[q,d] = input[order[q] / top_k,d]
void moe_gather(ConstTensorView input, ConstTensorView order, std::size_t top_k,
                TensorView grouped, cudaStream_t stream = nullptr);

// grouped/output: [T*K,H], order: Int32 [T*K], a permutation of [0,T*K).
// output[order[q],d] = grouped[q,d]; unique destinations require no atomics.
void moe_scatter(ConstTensorView grouped, ConstTensorView order,
                 TensorView output, cudaStream_t stream = nullptr);

// expert_outputs: [tokens * topk, hidden_size]
// expert_weights: [tokens, topk]
// output:         BF16 [tokens, hidden_size]
// shared_output:  BF16 [tokens, hidden_size], optional
// gate_logits:    FP32 [tokens, 1] when shared_output is provided
//
// output[tok][i] = sum(
//   expert_weights[tok][j] * expert_outputs[tok * topk + j][i])
//   + sigmoid(gate_logits[tok]) * shared_output[tok][i] (when provided)
void moe_combine(ConstTensorView expert_outputs, ConstTensorView expert_weights,
                 TensorView output, ConstTensorView shared_output,
                 ConstTensorView gate_logits, cudaStream_t stream = nullptr);

// CPU host-node rows are mapped and materialized in expert_outputs. The final
// output is BF16, with optional BF16 shared_output and FP32 gate_logits.
void moe_combine_routed(TensorView expert_outputs, const float* cpu_parts,
                        ConstTensorView ids, ConstTensorView routes,
                        ConstTensorView expert_weights, TensorView output,
                        ConstTensorView shared_output,
                        ConstTensorView gate_logits, cudaStream_t stream,
                        const volatile std::uint32_t* ready = nullptr,
                        volatile std::uint32_t* error = nullptr);
void moe_publish_decode(ConstTensorView input, ConstTensorView ids,
                        std::uint16_t* mapped_input, std::int32_t* mapped_ids,
                        volatile std::uint32_t* ready, cudaStream_t stream,
                        bool publish_only = false);

// query/output:              [q_tokens, query_heads, head_dim]
// key_cache/value_cache:     [blocks, block_tokens, key_heads, head_dim]
// key/value (new tokens):    [q_tokens, key_heads, head_dim]
// cu_seqlens_q/cu_seqlens_k: [batch + 1]
// block_table:               [batch, max_blocks]
void flash_attn_with_kvcache(
    ConstTensorView query, TensorView key_cache, TensorView value_cache,
    ConstTensorView key, ConstTensorView value, ConstTensorView cu_seqlens_q,
    ConstTensorView cu_seqlens_k, std::size_t max_seqlen_q,
    std::size_t max_seqlen_k, ConstTensorView block_table, TensorView output,
    bool causal, cudaStream_t stream = nullptr, TensorView workspace = {});

[[nodiscard]] std::size_t flash_attn_workspace_elements(
    std::size_t max_batch, std::size_t query_heads, std::size_t head_dim,
    std::size_t max_seqlen_k);

} // namespace nanovllm::kernels::cuda
