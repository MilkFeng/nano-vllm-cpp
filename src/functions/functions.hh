#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "core/tensor.hh"

namespace nanovllm::functions {

void zero_tensor(TensorView tensor, cudaStream_t stream);

[[nodiscard]] Tensor cast_tensor(ConstTensorView input, DataType dtype,
                                 cudaStream_t stream);

[[nodiscard]] Tensor sample(ConstTensorView logits,
                            ConstTensorView temperatures, std::uint64_t seed,
                            std::uint64_t step, cudaStream_t stream);

} // namespace nanovllm::functions
