#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "core/forward_context.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/linear.hh"
#include "layers/rms_norm.hh"
#include "layers/rotary_embedding.hh"

namespace nanovllm {

template <bool Gated, bool FP8>
class SelfAttention final : public Module {
 public:
  SelfAttention(std::size_t hidden_size, std::size_t num_heads,
                std::size_t num_kv_heads, std::size_t head_dim,
                float rms_norm_eps, std::size_t max_positions, float rope_theta,
                DataType dtype, Device device, std::size_t rotary_dim = 0,
                bool zero_centered_norm = false)
      : num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        q_size_(num_heads * head_dim),
        kv_size_(num_kv_heads * head_dim),
        q_proj_(make_projection(hidden_size, Gated ? 2 * q_size_ : q_size_,
                                dtype, device)),
        k_proj_(make_projection(hidden_size, kv_size_, dtype, device)),
        v_proj_(make_projection(hidden_size, kv_size_, dtype, device)),
        o_proj_(make_projection(q_size_, hidden_size, dtype, device)),
        q_norm_(head_dim, rms_norm_eps, dtype, device, zero_centered_norm),
        k_norm_(head_dim, rms_norm_eps, dtype, device, zero_centered_norm),
        rotary_emb_(max_positions, head_dim, rope_theta, device, rotary_dim) {
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("o_proj", o_proj_);
    register_module("q_norm", q_norm_);
    register_module("k_norm", k_norm_);
  }

  void build_rope_cache(cudaStream_t stream) {
    rotary_emb_.build_cache(stream);
  }

  [[nodiscard]] Tensor forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const ForwardContext& context = {}) const {
    const auto tokens = hidden_states.shape()[0];
    const auto stream = context.stream;
    const auto device = hidden_states.device();
    const auto dtype = hidden_states.dtype();

    const auto query_projection = q_proj_.forward(hidden_states, stream);
    const std::array query_shape{tokens, num_heads_, head_dim_};
    auto query_heads =
        Gated ? Tensor::empty({tokens, num_heads_, head_dim_}, dtype, device)
              : query_projection.reshape(query_shape);
    Tensor gate;
    if constexpr (Gated) {
      gate = Tensor::empty({tokens, num_heads_, head_dim_}, dtype, device);
      kernels::cuda::split_attention_gate(
          query_projection.view(), query_heads.view(), gate.view(), stream);
    }

    const auto key_projection = k_proj_.forward(hidden_states, stream);
    const auto value_projection = v_proj_.forward(hidden_states, stream);

    const std::array kv_shape{tokens, num_kv_heads_, head_dim_};
    const auto key_heads = key_projection.reshape(kv_shape);
    const auto value_heads = value_projection.reshape(kv_shape);

    auto query = q_norm_.forward(query_heads.view(), stream);
    auto key = k_norm_.forward(key_heads.view(), stream);

    rotary_emb_.forward(positions, query.view(), key.view(), stream);

    auto attention_output =
        Tensor::empty({tokens, num_heads_, head_dim_}, dtype, device);
    kernels::cuda::flash_attn_with_kvcache(
        query.view(), key_cache_, value_cache_, key.view(), value_heads.view(),
        context.cu_seqlens_q, context.cu_seqlens_k, context.max_seqlen_q,
        context.max_seqlen_k, context.block_tables, attention_output.view(),
        true, stream, context.attention_workspace);

    if constexpr (Gated) {
      kernels::cuda::apply_sigmoid_gate(attention_output.view(), gate.view(),
                                        stream);
    }

    const auto flattened_output =
        attention_output.reshape(std::array{tokens, q_size_});
    return o_proj_.forward(flattened_output.view(), stream);
  }

  void set_kv_cache(TensorView key_cache, TensorView value_cache) {
    key_cache_ = key_cache;
    value_cache_ = value_cache;
  }

  void clear_kv_cache() noexcept {
    key_cache_ = {};
    value_cache_ = {};
  }

  [[nodiscard]] ConstTensorView key_cache() const { return key_cache_; }
  [[nodiscard]] ConstTensorView value_cache() const { return value_cache_; }

  [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
  [[nodiscard]] std::size_t num_kv_heads() const noexcept {
    return num_kv_heads_;
  }
  [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
  [[nodiscard]] std::size_t q_size() const noexcept { return q_size_; }
  [[nodiscard]] std::size_t kv_size() const noexcept { return kv_size_; }

 private:
  using Projection = std::conditional_t<FP8, FP8Linear, Linear>;

  static Projection make_projection(std::size_t in_features,
                                    std::size_t out_features, DataType dtype,
                                    Device device) {
    if constexpr (FP8) {
      return FP8Linear(in_features, out_features, device);
    } else {
      return Linear(in_features, out_features, false, dtype, device);
    }
  }

 private:
  std::size_t num_heads_, num_kv_heads_, head_dim_, q_size_, kv_size_;

  Projection q_proj_, k_proj_, v_proj_, o_proj_;
  RMSNorm q_norm_, k_norm_;
  RotaryEmbedding rotary_emb_;

  TensorView key_cache_, value_cache_;
};

} // namespace nanovllm
