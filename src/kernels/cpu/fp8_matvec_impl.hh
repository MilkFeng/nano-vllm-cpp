// Included once by each ISA translation unit. Only the full 128-column scale
// blocks use this path; arbitrary sizes retain the general AVX2 dot product.
#include <cmath>
#include <utility>

#include "kernels/cpu/fp8_dot.hh"

namespace nanovllm::kernels::cpu::detail {
namespace {

float reduce8(__m256 sum) {
  auto v = _mm_add_ps(_mm256_castps256_ps128(sum),
                       _mm256_extractf128_ps(sum, 1));
  v = _mm_hadd_ps(v, v);
  v = _mm_hadd_ps(v, v);
  return _mm_cvtss_f32(v);
}

// A 16-lane accumulator keeps exactly the old two 8-lane accumulation chains.
// The final reduction therefore has the same order on both ISAs.
template <std::size_t Rows, bool GateUp>
void tile(const std::uint8_t* weight, const std::uint8_t* up,
          const float* scales, const float* up_scales, const float* input,
          std::size_t first, std::size_t columns, float* output) {
  constexpr auto indices = std::make_index_sequence<Rows>{};
  Wide sums[Rows];
  Wide up_sums[Rows];
  [&]<std::size_t... R>(std::index_sequence<R...>) {
    ((sums[R] = zero()), ...);
    if constexpr (GateUp) ((up_sums[R] = zero()), ...);
  }(indices);
  const auto blocks = columns / 128;
  for (std::size_t block = 0; block < blocks; ++block) {
    Wide scale[Rows], up_scale[Rows];
    [&]<std::size_t... R>(std::index_sequence<R...>) {
      ((scale[R] = broadcast(scales[((first + R) / 128) * blocks + block])),
       ...);
      if constexpr (GateUp) {
        ((up_scale[R] =
              broadcast(up_scales[((first + R) / 128) * blocks + block])),
         ...);
      }
    }(indices);
    for (std::size_t col = block * 128; col < (block + 1) * 128; col += 16) {
      const auto x = load(input + col);
      [&]<std::size_t... R>(std::index_sequence<R...>) {
        ((sums[R] = madd(decode(weight + (first + R) * columns + col),
                          scale[R], x, sums[R])), ...);
        if constexpr (GateUp) {
          ((up_sums[R] = madd(decode(up + (first + R) * columns + col),
                               up_scale[R], x, up_sums[R])), ...);
        }
      }(indices);
    }
  }
  [&]<std::size_t... R>(std::index_sequence<R...>) {
    if constexpr (GateUp) {
      ((output[first + R] = (reduce(sums[R]) /
                            (1.0f + std::exp(-reduce(sums[R])))) *
                           reduce(up_sums[R])), ...);
    } else {
      ((output[first + R] = reduce(sums[R])), ...);
    }
  }(indices);
}

void matvec_impl(const std::uint8_t* weights, const float* scales,
                 const float* input, std::size_t rows, std::size_t columns,
                 std::size_t block_size, float* output) {
  std::size_t row = 0;
  if (block_size == 128 && columns % 128 == 0) {
    for (; row + kOutputRows <= rows; row += kOutputRows) {
      tile<kOutputRows, false>(weights, nullptr, scales, nullptr, input, row,
                               columns, output);
    }
  }
  const auto blocks = (columns + block_size - 1) / block_size;
  for (; row < rows; ++row) {
    output[row] = fp8_dot_kernel(weights + row * columns,
                                 scales + (row / block_size) * blocks, input,
                                 columns, block_size);
  }
}

void gate_up_impl(const std::uint8_t* gate, const std::uint8_t* up,
                  const float* gate_scale, const float* up_scale,
                  const float* input, std::size_t rows, std::size_t columns,
                  std::size_t block_size, float* activated) {
  std::size_t row = 0;
  if (block_size == 128 && columns % 128 == 0) {
    for (; row + kGateRows <= rows; row += kGateRows) {
      tile<kGateRows, true>(gate, up, gate_scale, up_scale, input, row, columns,
                            activated);
    }
  }
  const auto blocks = (columns + block_size - 1) / block_size;
  for (; row < rows; ++row) {
    const auto offset = (row / block_size) * blocks;
    const auto g = fp8_dot_kernel(gate + row * columns, gate_scale + offset,
                                  input, columns, block_size);
    const auto u = fp8_dot_kernel(up + row * columns, up_scale + offset,
                                  input, columns, block_size);
    activated[row] = (g / (1.0f + std::exp(-g))) * u;
  }
}

} // namespace
} // namespace nanovllm::kernels::cpu::detail
