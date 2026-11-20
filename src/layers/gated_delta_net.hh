#pragma once

#include <cuda_runtime_api.h>

#include "core/model_config.hh"
#include "layers/causal_conv1d.hh"
#include "layers/gated_rms_norm.hh"
#include "layers/linear.hh"

namespace nanovllm {

class GatedDeltaNet final : public Module {
 public:
  GatedDeltaNet(const ModelConfig& config, Device device,
                DataType activation_dtype = DataType::Float32);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               ConstTensorView cu_seqlens = {},
                               ConstTensorView state_indices = {},
                               cudaStream_t stream = nullptr) const;

  void set_state_cache(TensorView conv, TensorView recurrent);
  void clear_state_cache() noexcept;

  [[nodiscard]] ConstTensorView conv_state() const;
  [[nodiscard]] ConstTensorView recurrent_state() const;

 private:
  // hidden_size_:            Hidden Dim
  // key_heads_/key_dim_:     Q/K 的 Head 数、Head Dim
  // value_heads_/value_dim_: V 的 Head 数、Head Dim
  //
  // GatedDeltaNet 多个 V head 共享一组 Q/K
  std::size_t hidden_size_, key_heads_, value_heads_, key_dim_, value_dim_;

  std::size_t conv_dim_, conv_kernel_;

  // in_proj_qkv_: [2*qk_heads*qk_head_dim+v_heads*v_head_dim, hidden_size]
  // in_proj_z_:   [v_heads*v_head_dim,                        hidden_size]
  // in_proj_a_:   [v_heads,                                   hidden_size]
  // in_proj_b_:   [v_heads,                                   hidden_size]
  //
  // out_proj_:    [hidden_size,                        v_heads*v_head_dim]
  Linear in_proj_qkv_, in_proj_z_, in_proj_a_, in_proj_b_, out_proj_;

  CausalConv1d conv1d_;
  Parameter A_log_, dt_bias_;
  GatedRMSNorm norm_;

  // conv_state_:      [slots, 2*qk_heads*qk_head_dim+v_heads*v_head_dim,
  //                    conv_kernel-1]
  // recurrent_state_: [slots, v_heads, qk_head_dim, v_head_dim]
  TensorView conv_state_, recurrent_state_;
};
} // namespace nanovllm
