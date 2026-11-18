#pragma once

#include "core/tensor.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {

inline Tensor cast_tensor(ConstTensorView input, DataType dtype,
                          cudaStream_t stream) {
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              dtype, input.device());
  kernels::cuda::cast(input, output.view(), stream);
  return output;
}

} // namespace nanovllm
