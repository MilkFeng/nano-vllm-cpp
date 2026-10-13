#include "kernels/cpu/fp8_dot.hh"

#include <immintrin.h>
#if defined(_MSC_VER)
  #include <intrin.h>
#endif

#include <algorithm>

#include "kernels/cpu/fp8_dequant.hh"

namespace nanovllm::kernels::cpu {
namespace {

__m256 decode(const std::uint8_t* weights, const float* table) {
  const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights));
  const auto indices = _mm256_cvtepu8_epi32(bytes);
  return _mm256_i32gather_ps(table, indices, sizeof(float));
}

float fp8_dot_avx2(const std::uint8_t* weights, const float* scales,
                   const float* input, std::size_t columns,
                   std::size_t block_size) {
  const auto* table = fp8_e4m3fn_table().data();
  auto sum0 = _mm256_setzero_ps();
  auto sum1 = _mm256_setzero_ps();
  float tail = 0.0f;
  for (std::size_t begin = 0; begin < columns; begin += block_size) {
    const auto end = std::min(begin + block_size, columns);
    const float scale = scales[begin / block_size];
    const auto scale8 = _mm256_set1_ps(scale);
    auto column = begin;
    for (; column + 16 <= end; column += 16) {
      const auto w0 = _mm256_mul_ps(decode(weights + column, table), scale8);
      const auto w1 =
          _mm256_mul_ps(decode(weights + column + 8, table), scale8);
      sum0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(input + column), sum0);
      sum1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(input + column + 8), sum1);
    }
    for (; column + 8 <= end; column += 8) {
      const auto w = _mm256_mul_ps(decode(weights + column, table), scale8);
      sum0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(input + column), sum0);
    }
    for (; column < end; ++column) {
      tail += (table[weights[column]] * scale) * input[column];
    }
  }
  const auto sum = _mm256_add_ps(sum0, sum1);
  auto reduced =
      _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  return _mm_cvtss_f32(reduced) + tail;
}

} // namespace

float fp8_dot_kernel(const std::uint8_t* weights, const float* scales,
                     const float* input, std::size_t columns,
                     std::size_t block_size) {
  return fp8_dot_avx2(weights, scales, input, columns, block_size);
}

} // namespace nanovllm::kernels::cpu