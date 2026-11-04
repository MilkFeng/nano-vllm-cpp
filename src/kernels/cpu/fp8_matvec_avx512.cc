#include "kernels/cpu/fp8_matvec.hh"
#include <immintrin.h>

namespace nanovllm::kernels::cpu::detail {
namespace {
using Wide = __m512;
Wide zero() { return _mm512_setzero_ps(); }
Wide broadcast(float x) { return _mm512_set1_ps(x); }
Wide load(const float* x) { return _mm512_loadu_ps(x); }
Wide decode(const std::uint8_t* w) {
  const auto b = _mm512_cvtepu8_epi32(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
  const auto mag = _mm512_and_si512(b, _mm512_set1_epi32(127));
  auto bits = _mm512_add_epi32(_mm512_slli_epi32(mag, 20),
                               _mm512_set1_epi32(0x3c000000));
  const auto small = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(8));
  const auto subnormal = _mm512_mul_ps(_mm512_cvtepi32_ps(mag),
                                       _mm512_set1_ps(0x1p-9f));
  bits = _mm512_mask_mov_epi32(bits, small, _mm512_castps_si512(subnormal));
  bits = _mm512_mask_mov_epi32(
      bits, _mm512_cmpeq_epi32_mask(mag, _mm512_set1_epi32(127)),
      _mm512_set1_epi32(0x7fc00000));
  return _mm512_castsi512_ps(_mm512_or_si512(bits, _mm512_slli_epi32(
      _mm512_and_si512(b, _mm512_set1_epi32(128)), 24)));
}
Wide madd(Wide w, Wide s, Wide x, Wide sum) {
  return _mm512_fmadd_ps(_mm512_mul_ps(w, s), x, sum);
}
float reduce(Wide);
constexpr std::size_t kOutputRows = 8;
constexpr std::size_t kGateRows = 4;
} // namespace
} // namespace nanovllm::kernels::cpu::detail

#include "kernels/cpu/fp8_matvec_impl.hh"

namespace nanovllm::kernels::cpu::detail {
namespace {
float reduce(Wide x) {
  return reduce8(_mm256_add_ps(_mm512_castps512_ps256(x),
                                _mm512_extractf32x8_ps(x, 1)));
}
}
void matvec_avx512(const std::uint8_t* w, const float* s, const float* x,
                    std::size_t n, std::size_t k, std::size_t b, float* y) {
  matvec_impl(w, s, x, n, k, b, y);
}
void gate_up_avx512(const std::uint8_t* g, const std::uint8_t* u,
                     const float* gs, const float* us, const float* x,
                     std::size_t n, std::size_t k, std::size_t b, float* y) {
  gate_up_impl(g, u, gs, us, x, n, k, b, y);
}
} // namespace nanovllm::kernels::cpu::detail
