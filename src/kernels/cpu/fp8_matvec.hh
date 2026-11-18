#pragma once

#include <cstddef>
#include <cstdint>

namespace nanovllm::kernels::cpu {

// The override NANOVLLM_CPU_ISA=avx2|avx512 is read once, before first use.
const char* fp8_cpu_isa();

void fp8_matvec(const std::uint8_t* weights, const float* scales,
                const float* input, std::size_t rows, std::size_t columns,
                std::size_t block_size, float* output);
void fp8_gate_up(const std::uint8_t* gate, const std::uint8_t* up,
                 const float* gate_scale, const float* up_scale,
                 const float* input, std::size_t rows, std::size_t columns,
                 std::size_t block_size, float* activated);

namespace detail {
using Matvec = decltype(&fp8_matvec);
using GateUp = decltype(&fp8_gate_up);
void matvec_avx2(const std::uint8_t*, const float*, const float*, std::size_t,
                 std::size_t, std::size_t, float*);
void gate_up_avx2(const std::uint8_t*, const std::uint8_t*, const float*,
                  const float*, const float*, std::size_t, std::size_t,
                  std::size_t, float*);
void matvec_avx512(const std::uint8_t*, const float*, const float*, std::size_t,
                   std::size_t, std::size_t, float*);
void gate_up_avx512(const std::uint8_t*, const std::uint8_t*, const float*,
                    const float*, const float*, std::size_t, std::size_t,
                    std::size_t, float*);
} // namespace detail
} // namespace nanovllm::kernels::cpu
