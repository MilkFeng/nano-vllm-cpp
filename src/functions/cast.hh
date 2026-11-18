#pragma once

#include <cuda_runtime_api.h>

#include "core/tensor.hh"

namespace nanovllm {

[[nodiscard]] Tensor cast_tensor(ConstTensorView input, DataType dtype,
                                 cudaStream_t stream);

} // namespace nanovllm
