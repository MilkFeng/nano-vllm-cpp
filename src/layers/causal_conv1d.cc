#include "layers/causal_conv1d.hh"

#include <cassert>
#include <utility>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

CausalConv1d::CausalConv1d(std::size_t channels, std::size_t kernel_size,
                           DataType dtype, Device device)
    : channels_(channels),
      kernel_size_(kernel_size),
      weight_(Tensor::empty({channels_, 1, kernel_size_}, dtype, device)) {
  assert(channels_ > 0 && kernel_size_ > 0);
  register_parameter("weight", weight_);
}

Tensor CausalConv1d::forward(ConstTensorView input, TensorView state,
                             ConstTensorView cu_seqlens,
                             ConstTensorView state_indices,
                             cudaStream_t stream) const {
  assert(input.rank() == 2 && input.shape()[1] == channels_);
  auto output = Tensor::empty({input.shape()[0], channels_}, input.dtype(),
                              input.device());
  // Forward the owned view metadata without copying its shape/stride vectors.
  kernels::cuda::causal_conv1d(std::move(input), weight_.view(),
                               std::move(state), std::move(cu_seqlens),
                               std::move(state_indices), output.view(), stream);
  return output;
}

std::size_t CausalConv1d::channels() const noexcept {
  return channels_;
}

std::size_t CausalConv1d::kernel_size() const noexcept {
  return kernel_size_;
}

Parameter& CausalConv1d::weight() noexcept {
  return weight_;
}

const Parameter& CausalConv1d::weight() const noexcept {
  return weight_;
}

} // namespace nanovllm
