#pragma once

#include "core/forward_context.hh"
#include "core/model_config.hh"
#include "layers/gated_rms_norm.hh"
#include "layers/linear.hh"

namespace nanovllm {

// Qwen3.5 linear attention, FP32 computation. Parameters use checkpoint names.
// Recurrent/conv caches are borrowed runtime data, never model parameters.
class GatedDeltaNet final : public Module {
 public:
  GatedDeltaNet(const ModelConfig& config, Device device);
  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               const ForwardContext& context) const;
  // conv: [slots,2*Hk*Dk+Hv*Dv,K-1]; recurrent: [slots,Hv,Dk,Dv].
  // The owner keeps these alive and resets slots before a new sequence.
  void set_state_cache(TensorView conv, TensorView recurrent);
  void clear_state_cache() noexcept;
  [[nodiscard]] ConstTensorView conv_state() const;
  [[nodiscard]] ConstTensorView recurrent_state() const;

 private:
  std::size_t hidden_size_, key_heads_, value_heads_, key_dim_, value_dim_;
  std::size_t conv_dim_, conv_kernel_;
  Linear in_proj_qkv_, in_proj_z_, in_proj_a_, in_proj_b_, out_proj_;
  Module conv1d_;
  Parameter conv_weight_, A_log_, dt_bias_;
  GatedRMSNorm norm_;
  TensorView conv_state_, recurrent_state_;
};
} // namespace nanovllm
