#include "layers/linear.hh"

#include <cassert>
#include <optional>
#include <utility>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias,
               DataType dtype, Device device)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(Tensor::empty({out_features_, in_features_}, dtype, device)) {
  register_parameter("weight", weight_);
  if (bias) {
    bias_.emplace(Tensor::empty({out_features_}, dtype, device));
    register_parameter("bias", *bias_);
  }
}

Tensor Linear::forward(ConstTensorView input, cudaStream_t stream) const {
  const auto weight = weight_.view();

  // FP32 parameters are used by the small, sensitive routing/decay projections.
  const auto output_dtype =
      weight.dtype() == DataType::Float32 ? DataType::Float32 : input.dtype();
  auto output = Tensor::empty({input.shape()[0], out_features_}, output_dtype,
                              input.device());

  std::optional<ConstTensorView> bias;
  if (bias_) {
    bias = std::as_const(*bias_).view();
  }

  kernels::cuda::linear(input, weight, output.view(), bias, stream);
  return output;
}

std::size_t Linear::in_features() const noexcept {
  return in_features_;
}

std::size_t Linear::out_features() const noexcept {
  return out_features_;
}

Parameter& Linear::weight() noexcept {
  return weight_;
}

const Parameter& Linear::weight() const noexcept {
  return weight_;
}

Parameter* Linear::bias() noexcept {
  return bias_ ? &*bias_ : nullptr;
}

const Parameter* Linear::bias() const noexcept {
  return bias_ ? &*bias_ : nullptr;
}

FP8Linear::FP8Linear(std::size_t in_features, std::size_t out_features,
                     Device device)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(Tensor::empty({out_features_, in_features_},
                            DataType::Float8E4M3FN, device)),
      weight_scale_inv_(Tensor::empty(
          {(out_features_ + 127) / 128, (in_features_ + 127) / 128},
          DataType::Float32, device)) {
  register_parameter("weight", weight_);
  register_parameter("weight_scale_inv", weight_scale_inv_);
}

Tensor FP8Linear::forward(ConstTensorView input, cudaStream_t stream) const {
  auto output = Tensor::empty({input.shape()[0], out_features_}, input.dtype(),
                              input.device());
  forward_into(input, output.view(), stream);
  return output;
}

void FP8Linear::forward_into(ConstTensorView input, TensorView output,
                             cudaStream_t stream) const {
  kernels::cuda::linear_fp8(input, weight_.view(), weight_scale_inv_.view(),
                            output, stream);
}

std::size_t FP8Linear::in_features() const noexcept {
  return in_features_;
}

std::size_t FP8Linear::out_features() const noexcept {
  return out_features_;
}

Parameter& FP8Linear::weight() noexcept {
  return weight_;
}

const Parameter& FP8Linear::weight() const noexcept {
  return weight_;
}

Parameter& FP8Linear::weight_scale_inv() noexcept {
  return weight_scale_inv_;
}

const Parameter& FP8Linear::weight_scale_inv() const noexcept {
  return weight_scale_inv_;
}

} // namespace nanovllm
