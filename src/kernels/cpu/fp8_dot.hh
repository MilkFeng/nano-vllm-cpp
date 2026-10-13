#pragma once

#include <cstddef>
#include <cstdint>

namespace nanovllm::kernels::cpu {

float fp8_dot_kernel(const std::uint8_t* weights, const float* scales,
                     const float* input, std::size_t columns,
                     std::size_t block_size);

} // namespace nanovllm::kernels::cpu
