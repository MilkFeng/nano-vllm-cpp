#include "kernels/cpu/fp8_matvec.hh"

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#if defined(_MSC_VER)
  #include <intrin.h>
#endif

namespace nanovllm::kernels::cpu {
namespace {
bool has_avx512() {
#if defined(_MSC_VER)
  int regs[4];
  __cpuid(regs, 1);
  if ((regs[2] & (1 << 27)) == 0 || (_xgetbv(0) & 0xe6) != 0xe6) {
    return false;
  }
  __cpuidex(regs, 7, 0);
  constexpr unsigned mask = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
  return (static_cast<unsigned>(regs[1]) & mask) == mask;
#else
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
         && __builtin_cpu_supports("avx512dq")
         && __builtin_cpu_supports("avx512vl");
#endif
}
struct Backend {
  detail::Matvec matvec;
  detail::GateUp gate_up;
  const char* name;
};
const Backend& backend() {
  static const Backend selected = [] {
    const auto* env = std::getenv("NANOVLLM_CPU_ISA");
    const std::string_view requested = env ? env : "auto";
    if (requested != "auto" && requested != "avx2" && requested != "avx512") {
      throw std::invalid_argument(
          "NANOVLLM_CPU_ISA must be auto, avx2 or avx512");
    }
    const bool available = has_avx512();
    if (requested == "avx512" && !available) {
      throw std::runtime_error("AVX-512 is unavailable on this CPU/OS");
    }
    if (requested != "avx2" && available) {
      return Backend{detail::matvec_avx512, detail::gate_up_avx512, "avx512"};
    }
    return Backend{detail::matvec_avx2, detail::gate_up_avx2, "avx2"};
  }();
  return selected;
}
} // namespace
const char* fp8_cpu_isa() {
  return backend().name;
}
void fp8_matvec(const std::uint8_t* w, const float* s, const float* x,
                std::size_t n, std::size_t k, std::size_t b, float* y) {
  backend().matvec(w, s, x, n, k, b, y);
}
void fp8_gate_up(const std::uint8_t* g, const std::uint8_t* u, const float* gs,
                 const float* us, const float* x, std::size_t n, std::size_t k,
                 std::size_t b, float* y) {
  backend().gate_up(g, u, gs, us, x, n, k, b, y);
}
} // namespace nanovllm::kernels::cpu
