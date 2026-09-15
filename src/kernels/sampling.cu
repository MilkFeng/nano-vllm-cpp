#include "kernels/inference.hh"

#include <curand_kernel.h>
#include <math_constants.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common.cuh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

__device__ bool better(float score, int index, float best, int best_index) {
  return score > best || (score == best && index < best_index);
}

__device__ void reduce_best(float* scores, int* indices) {
  for (unsigned offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset && better(scores[threadIdx.x + offset],
        indices[threadIdx.x + offset], scores[threadIdx.x], indices[threadIdx.x])) {
      scores[threadIdx.x] = scores[threadIdx.x + offset];
      indices[threadIdx.x] = indices[threadIdx.x + offset];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void sample_kernel(const T* logits, const float* temperatures,
                              std::int32_t* output, std::size_t batch,
                              std::size_t vocab, std::size_t stride,
                              std::uint64_t seed, std::uint64_t step) {
  __shared__ float scores[kThreads];
  __shared__ int indices[kThreads];
  for (std::size_t row = blockIdx.x; row < batch; row += gridDim.x) {
    const float temperature = temperatures[row];
    const bool valid_temperature = isfinite(temperature) && temperature >= 0.0f;
    assert(valid_temperature && "sampling temperature must be finite and nonnegative");
    if (!valid_temperature) {
      if (threadIdx.x == 0) output[row] = -1;
      continue;
    }
    float best = -CUDART_INF_F;
    int best_index = INT_MAX;
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      const float logit = detail::to_float(logits[row * stride + col]);
      const bool valid_logit = !isnan(logit) && logit != CUDART_INF_F;
      assert(valid_logit && "sampling logits cannot contain NaN or positive infinity");
      if (valid_logit && better(logit, static_cast<int>(col), best, best_index)) {
        best = logit;
        best_index = static_cast<int>(col);
      }
    }
    scores[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    reduce_best(scores, indices);
    const float maximum = scores[0];
    const int greedy_index = indices[0];
    const bool valid_row = isfinite(maximum);
    assert(valid_row && "sampling requires at least one finite logit per row");
    // Every thread must finish reading the reduction result before reuse.
    __syncthreads();
    if (temperature == 0.0f || !valid_row) {
      if (threadIdx.x == 0) output[row] = valid_row ? greedy_index : -1;
      continue;
    }

    best = -CUDART_INF_F;
    best_index = INT_MAX;
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, row * vocab + col, step, &state);
      const float exponential = fmaxf(-logf(curand_uniform(&state)), 1.0e-10f);
      const float logit = detail::to_float(logits[row * stride + col]);
      // Divide first when temperature can shrink finite extreme logits;
      // subtract first for small temperatures to avoid infinity - infinity.
      const float centered = temperature >= 1.0f
          ? logit / temperature - maximum / temperature
          : (logit - maximum) / temperature;
      const float score = centered - logf(exponential);
      if (better(score, static_cast<int>(col), best, best_index)) {
        best = score;
        best_index = static_cast<int>(col);
      }
    }
    scores[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    reduce_best(scores, indices);
    if (threadIdx.x == 0) output[row] = indices[0];
    __syncthreads();
  }
}

} // namespace

void sample(ConstTensorView logits, ConstTensorView temperatures,
            TensorView output, std::uint64_t seed, std::uint64_t step,
            cudaStream_t stream) {
  detail::require_cuda(logits);
  detail::require_cuda(temperatures);
  detail::require_cuda(output.as_const());
  detail::require_float(logits);
  detail::require(temperatures.device() == logits.device()
                      && output.device() == logits.device()
                      && temperatures.dtype() == DataType::Float32
                      && output.dtype() == DataType::Int32,
                  "sample requires Float32 temperatures and Int32 output on the logits device");
  detail::require(logits.rank() == 2 && temperatures.rank() == 1 && output.rank() == 1,
                  "sample expects logits[B,V], temperatures[B], and output[B]");
  const auto batch = logits.shape()[0], vocab = logits.shape()[1];
  detail::require(vocab > 0
                      && vocab <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
                      && temperatures.shape()[0] == batch && output.shape()[0] == batch,
                  "sample batch sizes must match and vocabulary must fit positive Int32");
  detail::require(logits.strides().size() == 2 && logits.strides()[1] == 1
                      && logits.strides()[0] >= vocab && temperatures.is_contiguous()
                      && output.is_contiguous(),
                  "sample requires contiguous logits columns, temperatures, and output");
  detail::require(!detail::overlaps(logits, output.as_const())
                      && !detail::overlaps(temperatures, output.as_const()),
                  "sample output must not overlap its inputs");
  (void)detail::checked_mul(batch, vocab);
  if (batch == 0) return;
  const detail::DeviceGuard guard(logits.device().index);
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(batch, 65535));
  detail::dispatch_float(logits.dtype(), [&]<typename T> {
    sample_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<const T*>(logits.data()),
        static_cast<const float*>(temperatures.data()), static_cast<std::int32_t*>(output.data()),
        batch, vocab, logits.strides()[0], seed, step);
  });
  detail::check_cuda(cudaGetLastError(), "sample launch");
}

} // namespace nanovllm::kernels
