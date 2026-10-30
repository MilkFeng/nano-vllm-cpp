#pragma once

#include <cuda_runtime_api.h>

#include <optional>

#include "core/tensor_view.hh"

namespace nanovllm::exp {

// Reference implementation of input @ weight^T + bias using cuBLASLt.
void cublas_linear(ConstTensorView input, ConstTensorView weight,
                   TensorView output,
                   std::optional<ConstTensorView> bias = std::nullopt,
                   cudaStream_t stream = nullptr);

} // namespace nanovllm::exp
