#include <immintrin.h>

#include <array>
#include <cassert>

#include "kernels/cpu/fp8_matvec.hh"

namespace nanovllm::kernels::cpu::detail { namespace {
using Wide = __m512;
Wide zero() {
  return _mm512_setzero_ps();
}
Wide broadcast(float x) {
  return _mm512_set1_ps(x);
}
Wide load(const float* x) {
  return _mm512_loadu_ps(x);
}
Wide load(const std::uint16_t* x) {
  const auto packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
}
alignas(64) constexpr auto kHalfValues = [] {
  std::array<std::uint16_t, 128> table{};
  for (unsigned i = 0; i < 128; ++i) {
    const unsigned e = i >> 3, f = i & 7;
    table[i] = e        ? ((e + 8) << 10) | (f << 7)
               : f == 0 ? 0
               : f == 1 ? 6 << 10
               : f < 4  ? (7 << 10) | ((f - 2) << 9)
                        : (8 << 10) | ((f - 4) << 8);
  }
  table[127] = 0x7e00;
  return table;
}();
Wide decode(const std::uint8_t* w) {
  const auto b = _mm512_cvtepu8_epi16(_mm256_zextsi128_si256(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(w))));
  const auto lo =
      _mm512_permutex2var_epi16(_mm512_load_si512(kHalfValues.data()), b,
                                _mm512_load_si512(kHalfValues.data() + 32));
  const auto hi =
      _mm512_permutex2var_epi16(_mm512_load_si512(kHalfValues.data() + 64), b,
                                _mm512_load_si512(kHalfValues.data() + 96));
  const auto positive = _mm512_mask_blend_epi16(
      _mm512_test_epi16_mask(b, _mm512_set1_epi16(64)), lo, hi);
  const auto sign =
      _mm512_slli_epi16(_mm512_and_si512(b, _mm512_set1_epi16(128)), 8);
  return _mm512_cvtph_ps(
      _mm512_castsi512_si256(_mm512_or_si512(positive, sign)));
}
Wide madd(Wide w, Wide s, Wide x, Wide sum) {
  return _mm512_fmadd_ps(_mm512_mul_ps(w, s), x, sum);
}
float reduce(Wide);
constexpr std::size_t kOutputRows = 8;
constexpr std::size_t kGateRows = 4;
}} // namespace nanovllm::kernels::cpu::detail

#include "kernels/cpu/fp8_matvec_impl.hh"

namespace nanovllm::kernels::cpu::detail {
namespace {
float reduce(Wide x) {
  return reduce8(
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1)));
}
} // namespace
void matvec_avx512(const std::uint8_t* w, const float* s, const float* x,
                   std::size_t n, std::size_t k, std::size_t b, float* y) {
  matvec_impl(w, s, x, n, k, b, y);
}
void gate_up_avx512(const std::uint8_t* g, const std::uint8_t* u,
                    const float* gs, const float* us, const float* x,
                    std::size_t n, std::size_t k, std::size_t b, float* y) {
  gate_up_impl(g, u, gs, us, x, n, k, b, y);
}
void gate_up_bf16_avx512(const std::uint8_t* g, const std::uint8_t* u,
                         const float* gs, const float* us,
                         const std::uint16_t* x, std::size_t n, std::size_t k,
                         float* y) {
  assert(n % kGateRows == 0 && k % 128 == 0);
  for (std::size_t row = 0; row < n; row += kGateRows) {
    tile<kGateRows, true>(g, u, gs, us, x, row, k, y);
  }
}
} // namespace nanovllm::kernels::cpu::detail
