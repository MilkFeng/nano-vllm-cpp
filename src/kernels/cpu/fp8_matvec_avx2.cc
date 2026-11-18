#include "kernels/cpu/fp8_decode.hh"
#include "kernels/cpu/fp8_matvec.hh"

namespace nanovllm::kernels::cpu::detail { namespace {
struct Wide {
  __m256 lo, hi;
};
Wide zero() {
  return {_mm256_setzero_ps(), _mm256_setzero_ps()};
}
Wide broadcast(float x) {
  return {_mm256_set1_ps(x), _mm256_set1_ps(x)};
}
Wide load(const float* x) {
  return {_mm256_loadu_ps(x), _mm256_loadu_ps(x + 8)};
}
Wide decode(const std::uint8_t* w) {
  return {decode_fp8_avx2(w), decode_fp8_avx2(w + 8)};
}
Wide madd(Wide w, Wide scale, Wide x, Wide sum) {
  return {_mm256_fmadd_ps(_mm256_mul_ps(w.lo, scale.lo), x.lo, sum.lo),
          _mm256_fmadd_ps(_mm256_mul_ps(w.hi, scale.hi), x.hi, sum.hi)};
}
float reduce(Wide);
constexpr std::size_t kOutputRows = 2;
constexpr std::size_t kGateRows = 1;
}} // namespace nanovllm::kernels::cpu::detail

#include "kernels/cpu/fp8_matvec_impl.hh"

namespace nanovllm::kernels::cpu::detail {
namespace {
float reduce(Wide x) {
  return reduce8(_mm256_add_ps(x.lo, x.hi));
}
} // namespace
void matvec_avx2(const std::uint8_t* w, const float* s, const float* x,
                 std::size_t n, std::size_t k, std::size_t b, float* y) {
  matvec_impl(w, s, x, n, k, b, y);
}
void gate_up_avx2(const std::uint8_t* g, const std::uint8_t* u, const float* gs,
                  const float* us, const float* x, std::size_t n, std::size_t k,
                  std::size_t b, float* y) {
  gate_up_impl(g, u, gs, us, x, n, k, b, y);
}
} // namespace nanovllm::kernels::cpu::detail
