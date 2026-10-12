#pragma once

#include <cstddef>
#include <cstdint>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cpu {

[[nodiscard]] float fp8_e4m3fn_to_float(std::uint8_t bits);

void dequantize_fp8(ConstTensorView weight, ConstTensorView scale_inv,
                    TensorView output, std::size_t block_rows = 128,
                    std::size_t block_columns = 128);

} // namespace nanovllm::kernels::cpu
