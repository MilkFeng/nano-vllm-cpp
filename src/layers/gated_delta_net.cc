#include "layers/gated_delta_net.hh"

#include <array>
#include <cassert>

#include "common/cuda_utils.hh"
#include "kernels/cuda/gated_delta_net.hh"

namespace nanovllm {

GatedDeltaNet::GatedDeltaNet(const ModelConfig& config, Device device)
    : hidden_size_(config.hidden_size),
      key_heads_(config.linear_num_key_heads),
      value_heads_(config.linear_num_value_heads),
      key_dim_(config.linear_key_head_dim),
      value_dim_(config.linear_value_head_dim),
      conv_dim_(2 * key_heads_ * key_dim_ + value_heads_ * value_dim_),
      conv_kernel_(config.linear_conv_kernel_dim),
      in_proj_qkv_(hidden_size_, conv_dim_, false, DataType::Float32, device),
      in_proj_z_(hidden_size_, value_heads_ * value_dim_, false,
                 DataType::Float32, device),
      in_proj_a_(hidden_size_, value_heads_, false, DataType::Float32, device),
      in_proj_b_(hidden_size_, value_heads_, false, DataType::Float32, device),
      out_proj_(value_heads_ * value_dim_, hidden_size_, false,
                DataType::Float32, device),
      conv_weight_(Tensor::empty({conv_dim_, 1, conv_kernel_},
                                 DataType::Float32, device)),
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

  conv1d_.register_parameter("weight", conv_weight_);
}

Tensor GatedDeltaNet::forward(ConstTensorView hidden_states,
                              const ForwardContext& context) const {
  const auto tokens = hidden_states.shape()[0];
  const auto stream = context.stream;

  const auto cu = context.is_prefill ? context.cu_seqlens_q : ConstTensorView{};

  auto qkv = in_proj_qkv_.forward(hidden_states, stream);
  auto z = in_proj_z_.forward(hidden_states, stream);
  auto a = in_proj_a_.forward(hidden_states, stream);
  auto b = in_proj_b_.forward(hidden_states, stream);

  Tensor local_conv, local_recurrent;
  auto conv = conv_state_, recurrent = recurrent_state_;
  auto state_indices = context.state_indices;
  if (recurrent.numel() == 0) {
    const auto batch = cu.shape()[0] - 1;

    local_conv = Tensor::empty({batch, conv_dim_, conv_kernel_ - 1},
                               DataType::Float32, hidden_states.device());
    local_recurrent = Tensor::empty({batch, value_heads_, key_dim_, value_dim_},
                                    DataType::Float32, hidden_states.device());

    if (local_conv.size_bytes()) {
      check_cuda(cudaMemsetAsync(local_conv.data(), 0, local_conv.size_bytes(),
                                 stream),
                 "zero local conv history");
    }

    check_cuda(cudaMemsetAsync(local_recurrent.data(), 0,
                               local_recurrent.size_bytes(), stream),
               "zero local recurrent state");

    conv = local_conv.view();
    recurrent = local_recurrent.view();
    state_indices = {};
  }
  auto convolved = Tensor::empty({tokens, conv_dim_}, DataType::Float32,
                                 hidden_states.device());

  kernels::cuda::causal_conv1d(qkv.view().as_const(), conv_weight_.view(), conv,
                               cu, state_indices, convolved.view(), stream);

  auto output = Tensor::empty({tokens, value_heads_, value_dim_},
                              DataType::Float32, hidden_states.device());

  kernels::cuda::gated_delta_rule(
      convolved.view().as_const(), a.view().as_const(), b.view().as_const(),
      A_log_.view(), dt_bias_.view(), recurrent, cu, state_indices, key_heads_,
      output.view(), stream);

  const auto z_heads = z.reshape(std::array{tokens, value_heads_, value_dim_});

  auto normalized =
      norm_.forward(output.view().as_const(), z_heads.view(), stream);

  const auto flattened =
      normalized.reshape(std::array{tokens, value_heads_ * value_dim_});
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
