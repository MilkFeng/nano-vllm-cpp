#include "layers/gated_delta_net.hh"

#include <array>
#include <cassert>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

GatedDeltaNet::GatedDeltaNet(const ModelConfig& config, Device device)
    : hidden_size_(config.hidden_size),
      key_heads_(config.linear_num_key_heads),
      value_heads_(config.linear_num_value_heads),
      key_dim_(config.linear_key_head_dim),
      value_dim_(config.linear_value_head_dim),
      conv_dim_(2 * key_heads_ * key_dim_ + value_heads_ * value_dim_),
      conv_kernel_(config.linear_conv_kernel_dim),
      in_proj_qkv_(hidden_size_, conv_dim_, device),
      in_proj_z_(hidden_size_, value_heads_ * value_dim_, device),
      in_proj_a_(hidden_size_, value_heads_, false, DataType::Float32, device),
      in_proj_b_(hidden_size_, value_heads_, false, DataType::Float32, device),
      out_proj_(value_heads_ * value_dim_, hidden_size_, device),
      conv1d_(conv_dim_, conv_kernel_, DataType::BFloat16, device),
      A_log_(Tensor::empty({value_heads_}, DataType::Float32, device)),
      dt_bias_(Tensor::empty({value_heads_}, DataType::Float32, device)),
      norm_(value_dim_, config.rms_norm_eps, device) {
  register_module("in_proj_qkv", in_proj_qkv_);
  register_module("in_proj_z", in_proj_z_);
  register_module("in_proj_a", in_proj_a_);
  register_module("in_proj_b", in_proj_b_);
  register_module("out_proj", out_proj_);
  register_module("norm", norm_);
  register_module("conv1d", conv1d_);

  register_parameter("A_log", A_log_);
  register_parameter("dt_bias", dt_bias_);
}

Tensor GatedDeltaNet::forward(ConstTensorView hidden_states,
                              ConstTensorView cu_seqlens,
                              ConstTensorView state_indices,
                              cudaStream_t stream) const {
  assert(conv_state_.rank() == 3 && recurrent_state_.numel() != 0);

  const auto tokens = hidden_states.shape()[0];

  // input: [tokens, hidden_size]
  //
  // qkv:   [tokens, 2*qk_heads*qk_head_dim+v_heads*v_head_dim]
  // z:     [tokens,                        v_heads*v_head_dim]
  // a:     [tokens,                                   v_heads]
  // b:     [tokens,                                   v_heads]

  auto qkv = in_proj_qkv_.forward(hidden_states, stream);
  auto z = in_proj_z_.forward(hidden_states, stream);
  auto a = in_proj_a_.forward(hidden_states, stream);
  auto b = in_proj_b_.forward(hidden_states, stream);

  // 对 qkv 做一下 Causal Depthwise Convolution
  //
  // convolved: [tokens, 2*qk_heads*qk_head_dim+v_heads*v_head_dim]
  //            最后一维按照顺序存放 Q、K、V
  auto convolved = conv1d_.forward(qkv.view(), conv_state_, cu_seqlens,
                                   state_indices, stream);

  auto output = Tensor::empty({tokens, value_heads_, value_dim_},
                              hidden_states.dtype(), hidden_states.device());

  // 更新递归记忆，并生成每个 token 的输出
  kernels::cuda::gated_delta_rule(convolved.view(), a.view(), b.view(),
                                  A_log_.view(), dt_bias_.view(),
                                  recurrent_state_, cu_seqlens, state_indices,
                                  key_heads_, output.view(), stream);

  // z:       [tokens, v_heads*v_head_dim]
  // z_heads: [tokens, v_heads, v_head_dim]
  const auto z_heads = z.reshape(std::array{tokens, value_heads_, value_dim_});

  // 每个 head 内做 RMSNorm，再应用 z 门控
  //
  // normalized: [tokens, v_heads, v_head_dim]
  auto normalized = norm_.forward(output.view(), z_heads.view(), stream);

  // 最后两维合并一下
  const auto flattened =
      normalized.reshape(std::array{tokens, value_heads_ * value_dim_});

  // 应用 out_proj_
  return out_proj_.forward(flattened.view(), stream);
}

void GatedDeltaNet::set_state_cache(TensorView conv, TensorView recurrent) {
  conv_state_ = conv;
  recurrent_state_ = recurrent;
}

void GatedDeltaNet::clear_state_cache() noexcept {
  conv_state_ = {};
  recurrent_state_ = {};
}

ConstTensorView GatedDeltaNet::conv_state() const {
  return ConstTensorView(conv_state_);
}

ConstTensorView GatedDeltaNet::recurrent_state() const {
  return ConstTensorView(recurrent_state_);
}

} // namespace nanovllm
