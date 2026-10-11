#include "layers/linear.hh"

#include <cassert>
#include <optional>
#include <utility>

#include "kernels/cuda/inference.hh"

namespace nanovllm {

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias,
               DataType dtype, Device device)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(Tensor::empty({out_features_, in_features_}, dtype, device)) {
  assert(in_features_ > 0 && out_features_ > 0);
  register_parameter("weight", weight_);
  if (bias) {
    bias_ = std::make_unique<Parameter>(
        Tensor::empty({out_features_}, dtype, device));
    register_parameter("bias", *bias_);
  }
}

Tensor Linear::forward(ConstTensorView input, cudaStream_t stream) const {
  const auto weight = weight_.view();
  assert(input.rank() == 2 && input.shape()[1] == in_features_);
  assert(input.device().type == DeviceType::Cuda);
  assert(same_device(input, weight));
  assert(same_dtype(input, weight));
  auto output = Tensor::empty({input.shape()[0], out_features_}, input.dtype(),
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
  return bias_.get();
}

const Parameter* Linear::bias() const noexcept {
  return bias_.get();
}

} // namespace nanovllm
