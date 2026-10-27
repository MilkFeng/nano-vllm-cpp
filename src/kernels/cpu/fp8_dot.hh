#pragma once

#include <cstddef>
#include <cstdint>

namespace nanovllm::kernels::cpu {

float fp8_dot_kernel(const std::uint8_t* weights, const float* scales,
                     const float* input, std::size_t columns,
                     std::size_t block_size);

void fp8_dot_batch_kernel(const std::uint8_t* weights, const float* scales,
                          const float* input, std::size_t input_row_stride,
                          std::size_t rows, std::size_t columns,
                          std::size_t block_size, float* output);

} // namespace nanovllm::kernels::cpu
