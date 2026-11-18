#include "kernels/cpu/fp8_dot.hh"

#include <immintrin.h>
#if defined(_MSC_VER)
  #include <intrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <utility>

#include "kernels/cpu/fp8_decode.hh"
#include "kernels/cpu/fp8_dequant.hh"

namespace nanovllm::kernels::cpu {
namespace {

__m256 decode(const std::uint8_t* weights, const float* table) {
  (void)table;
  return detail::decode_fp8_avx2(weights);
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

float horizontal_sum(__m256 value) {
  auto sum = _mm_add_ps(_mm256_castps256_ps128(value),
                        _mm256_extractf128_ps(value, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

template <std::size_t Rows, bool FullBlocks = false>
void fp8_dot_batch_avx2(const std::uint8_t* weights, const float* scales,
                        const float* input, std::size_t input_row_stride,
                        std::size_t columns, std::size_t block_size,
                        float* output) {
  const auto* table = fp8_e4m3fn_table().data();
  __m256 sum0[Rows], sum1[Rows];
  float tail[Rows]{};
  [&]<std::size_t... R>(std::index_sequence<R...>) {
    ((sum0[R] = _mm256_setzero_ps(), sum1[R] = _mm256_setzero_ps()), ...);
  }(std::make_index_sequence<Rows>{});

  for (std::size_t begin = 0; begin < columns; begin += block_size) {
    const auto end =
        FullBlocks ? begin + 128 : std::min(begin + block_size, columns);
    const float scale = scales[begin / block_size];
    const auto scale8 = _mm256_set1_ps(scale);
    auto column = begin;
    for (; column + 16 <= end; column += 16) {
      const auto w0 = _mm256_mul_ps(decode(weights + column, table), scale8);
      const auto w1 =
          _mm256_mul_ps(decode(weights + column + 8, table), scale8);
      // Constant row indices let the compiler keep accumulators in registers.
      [&]<std::size_t... R>(std::index_sequence<R...>) {
        ((sum0[R] = _mm256_fmadd_ps(
              w0, _mm256_loadu_ps(input + R * input_row_stride + column),
              sum0[R]),
          sum1[R] = _mm256_fmadd_ps(
              w1, _mm256_loadu_ps(input + R * input_row_stride + column + 8),
              sum1[R])),
         ...);
      }(std::make_index_sequence<Rows>{});
    }
    if constexpr (!FullBlocks) {
      for (; column + 8 <= end; column += 8) {
        const auto w = _mm256_mul_ps(decode(weights + column, table), scale8);
        [&]<std::size_t... R>(std::index_sequence<R...>) {
          ((sum0[R] = _mm256_fmadd_ps(
                w, _mm256_loadu_ps(input + R * input_row_stride + column),
                sum0[R])),
           ...);
        }(std::make_index_sequence<Rows>{});
      }
      for (; column < end; ++column) {
        const float w = table[weights[column]] * scale;
        [&]<std::size_t... R>(std::index_sequence<R...>) {
          ((tail[R] += w * input[R * input_row_stride + column]), ...);
        }(std::make_index_sequence<Rows>{});
      }
    }
  }
  [&]<std::size_t... R>(std::index_sequence<R...>) {
    ((output[R] = horizontal_sum(_mm256_add_ps(sum0[R], sum1[R])) + tail[R]),
     ...);
  }(std::make_index_sequence<Rows>{});
}

} // namespace

float fp8_dot_kernel(const std::uint8_t* weights, const float* scales,
                     const float* input, std::size_t columns,
                     std::size_t block_size) {
  return fp8_dot_avx2(weights, scales, input, columns, block_size);
}

void fp8_dot_batch_kernel(const std::uint8_t* weights, const float* scales,
                          const float* input, std::size_t input_row_stride,
                          std::size_t rows, std::size_t columns,
                          std::size_t block_size, float* output) {
  assert((rows == 2 || rows == 4) && block_size > 0);
  assert(input_row_stride >= columns);
  // Qwen's 2048/512-column projections have no remainder. Removing tail paths
  // lets the compiler retain the four-row accumulators across scale blocks.
  if (block_size == 128 && columns % 128 == 0) {
    if (rows == 4) {
      fp8_dot_batch_avx2<4, true>(weights, scales, input, input_row_stride,
                                  columns, 128, output);
    } else {
      fp8_dot_batch_avx2<2, true>(weights, scales, input, input_row_stride,
                                  columns, 128, output);
    }
  } else if (rows == 4) {
    fp8_dot_batch_avx2<4>(weights, scales, input, input_row_stride, columns,
                          block_size, output);
  } else {
    fp8_dot_batch_avx2<2>(weights, scales, input, input_row_stride, columns,
                          block_size, output);
  }
}

} // namespace nanovllm::kernels::cpu
