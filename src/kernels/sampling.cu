#include <curand_kernel.h>
#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/common.cuh"
#include "kernels/inference.hh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

__device__ bool better(float score, int index, float best, int best_index) {
  return score > best || (score == best && index < best_index);
}

__device__ void reduce_best(float* scores, int* indices) {
  for (unsigned offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset
        && better(scores[threadIdx.x + offset], indices[threadIdx.x + offset],
                  scores[threadIdx.x], indices[threadIdx.x])) {
      scores[threadIdx.x] = scores[threadIdx.x + offset];
      indices[threadIdx.x] = indices[threadIdx.x + offset];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void sample_kernel(const T* logits, const float* temperatures,
                              std::int32_t* output, std::size_t batch,
                              std::size_t vocab, std::uint64_t seed,
                              std::uint64_t step) {
  __shared__ float scores[kThreads];
  __shared__ int indices[kThreads];
  for (std::size_t row = blockIdx.x; row < batch; row += gridDim.x) {
    const float temperature = temperatures[row];
    assert(isfinite(temperature) && temperature >= 0.0f);
    float best = -CUDART_INF_F;
    int best_index = INT_MAX;
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      const float logit = detail::to_float(logits[row * vocab + col]);
      assert(!isnan(logit) && logit != CUDART_INF_F);
      if (better(logit, static_cast<int>(col), best, best_index)) {
        best = logit;
        best_index = col;
      }
    }
    scores[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;

    __syncthreads();

    reduce_best(scores, indices);

    const float maximum = scores[0];
    const int greedy_index = indices[0];
    assert(isfinite(maximum)
           && "sampling requires at least one finite logit per row");
    // Every thread must finish reading the reduction result before reuse.
    __syncthreads();
    if (temperature == 0.0f) {
      if (threadIdx.x == 0) output[row] = greedy_index;
      continue;
    }

    best = -CUDART_INF_F;
    best_index = INT_MAX;
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, row * vocab + col, step, &state);
      const float exponential = fmaxf(-logf(curand_uniform(&state)), 1.0e-10f);
      const float logit = detail::to_float(logits[row * vocab + col]);
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

    if (threadIdx.x == 0) {
      output[row] = indices[0];
    }

    __syncthreads();
  }
}

} // namespace

void sample(ConstTensorView logits, ConstTensorView temperatures,
            TensorView output, std::uint64_t seed, std::uint64_t step,
            cudaStream_t stream) {
  detail::assert_tensor(logits);
  detail::assert_tensor(temperatures);
  detail::assert_tensor(output);
  assert(temperatures.dtype() == DataType::Float32);
  assert(output.dtype() == DataType::Int32);
  assert(logits.rank() == 2 && temperatures.rank() == 1 && output.rank() == 1);
  const auto batch = logits.shape()[0], vocab = logits.shape()[1];
  assert(vocab > 0 && vocab <= INT_MAX);
  assert(temperatures.shape()[0] == batch && output.shape()[0] == batch);
  if (batch == 0) {
    return;
  }

  const auto dtype = logits.dtype();
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>(batch, 65535));
  detail::dispatch_float(dtype, [&]<typename T> {
    sample_kernel<<<blocks, kThreads, 0, stream>>>(
        logits.data<T>(), temperatures.data<float>(),
        output.data<std::int32_t>(), batch, vocab, seed, step);
  });

  check_cuda(cudaGetLastError(), "sample launch");
}

} // namespace nanovllm::kernels
