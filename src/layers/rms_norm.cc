#include "layers/rms_norm.hh"

#include <cassert>
#include <cmath>
#include <vector>

#include "kernels/kernels.hh"

namespace nanovllm {

RMSNorm::RMSNorm(std::size_t normalized_size, float epsilon, DataType dtype,
                 Device device)
    : normalized_size_(normalized_size),
      epsilon_(epsilon),
      weight_(Tensor::empty({normalized_size_}, dtype, device)) {
  assert(normalized_size_ > 0);
  assert(std::isfinite(epsilon_) && epsilon_ > 0);
  register_parameter("weight", weight_);
}

Tensor RMSNorm::forward(ConstTensorView input, cudaStream_t stream) const {
  const auto weight = weight_.view();
  assert(input.rank() >= 1 && input.rank() <= 3);
  assert(input.shape().back() == normalized_size_);
  assert(input.device().type == DeviceType::Cuda);
  assert(input.device() == weight.device() && same_dtype(input, weight));
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              input.dtype(), input.device());
  kernels::rms_norm(input, weight, output.view(), epsilon_, stream);
  return output;
}

RMSNormOutput RMSNorm::forward(ConstTensorView input, ConstTensorView residual,
                               cudaStream_t stream) const {
  const auto weight = weight_.view();
  assert(input.rank() >= 1 && input.rank() <= 3);
  assert(input.shape().back() == normalized_size_);
  assert(input.device().type == DeviceType::Cuda);
  assert(input.device() == weight.device()
         && input.device() == residual.device());
  assert(same_shape(input, residual));
  assert(same_dtype(input, residual, weight));
  const std::vector<std::size_t> shape(input.shape().begin(),
                                       input.shape().end());
  RMSNormOutput result{Tensor::empty(shape, input.dtype(), input.device()),
                       Tensor::empty(shape, input.dtype(), input.device())};
  kernels::fused_add_rms_norm(input, residual, weight,
                              result.hidden_states.view(),
                              result.residual.view(), epsilon_, stream);
  return result;
}

std::size_t RMSNorm::normalized_size() const noexcept {
  return normalized_size_;
}

float RMSNorm::epsilon() const noexcept {
  return epsilon_;
}

Parameter& RMSNorm::weight() noexcept {
  return weight_;
}

const Parameter& RMSNorm::weight() const noexcept {
  return weight_;
}

} // namespace nanovllm
