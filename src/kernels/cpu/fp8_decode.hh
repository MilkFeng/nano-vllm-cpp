#pragma once

#include <immintrin.h>
#include <cstdint>

namespace nanovllm::kernels::cpu::detail {

// E4M3FN normal numbers map directly to FP32 exponent/mantissa bits.
// Handle subnormals, signed zero and the two NaN encodings explicitly.
inline __m256 decode_fp8_avx2(const std::uint8_t* weights) {
  const auto b = _mm256_cvtepu8_epi32(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights)));
  const auto magnitude = _mm256_and_si256(b, _mm256_set1_epi32(127));
  auto bits = _mm256_add_epi32(_mm256_slli_epi32(magnitude, 20),
                               _mm256_set1_epi32(0x3c000000));
  const auto small = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), magnitude);
  const auto subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude),
                                       _mm256_set1_ps(0x1p-9f));
  bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(subnormal), small);
  bits = _mm256_blendv_epi8(
      bits, _mm256_set1_epi32(0x7fc00000),
      _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(127)));
  return _mm256_castsi256_ps(_mm256_or_si256(
      bits, _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(128)), 24)));
}

} // namespace nanovllm::kernels::cpu::detail
