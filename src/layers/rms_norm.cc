#include "layers/rms_norm.hh"

#include <cassert>
#include <cmath>
#include <vector>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

RMSNorm::RMSNorm(std::size_t normalized_size, float epsilon, DataType dtype,
                 Device device, bool zero_centered)
    : normalized_size_(normalized_size),
      epsilon_(epsilon),
      zero_centered_(zero_centered),
      weight_(Tensor::empty({normalized_size_}, dtype, device)) {
  register_parameter("weight", weight_);
}

Tensor RMSNorm::forward(ConstTensorView input, cudaStream_t stream) const {
  const auto weight = weight_.view();
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              input.dtype(), input.device());
  kernels::cuda::rms_norm(input, weight, output.view(), epsilon_, stream,
                          zero_centered_);
  return output;
}

RMSNormOutput RMSNorm::forward(ConstTensorView input, ConstTensorView residual,
                               cudaStream_t stream) const {
  const auto weight = weight_.view();
  const std::vector<std::size_t> shape(input.shape().begin(),
                                       input.shape().end());
  RMSNormOutput result{Tensor::empty(shape, input.dtype(), input.device()),
                       Tensor::empty(shape, input.dtype(), input.device())};
  kernels::cuda::fused_add_rms_norm(
      input, residual, weight, result.hidden_states.view(),
      result.residual.view(), epsilon_, stream, zero_centered_);
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
