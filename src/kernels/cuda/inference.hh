#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias = std::nullopt,
            cudaStream_t stream = nullptr);

void gather_last_hidden(ConstTensorView hidden, ConstTensorView cu_seqlens_q,
                        TensorView output, cudaStream_t stream = nullptr);

void sample(ConstTensorView logits, ConstTensorView temperatures,
            TensorView output, std::uint64_t seed, std::uint64_t step,
            cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
