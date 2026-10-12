#include "layers/gated_rms_norm.hh"

#include <cassert>
#include <cmath>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

GatedRMSNorm::GatedRMSNorm(std::size_t normalized_size, float epsilon,
                           Device device)
    : normalized_size_(normalized_size),
      epsilon_(epsilon),
      weight_(Tensor::empty({normalized_size}, DataType::Float32, device)) {
  register_parameter("weight", weight_);
}

Tensor GatedRMSNorm::forward(ConstTensorView input, ConstTensorView gate,
                             cudaStream_t stream) const {
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              input.dtype(), input.device());
  kernels::cuda::gated_rms_norm(input, gate, weight_.view(), output.view(),
                                epsilon_, stream);
  return output;
}

} // namespace nanovllm
