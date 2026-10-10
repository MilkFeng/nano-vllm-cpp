#include <math_constants.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

__global__ void route_kernel(const float* logits, std::int32_t* ids,
                             float* weights, std::size_t experts,
                             std::size_t top_k) {
  extern __shared__ float storage[];
  float* scores = storage;
  float* probabilities = storage + experts;

  float maximum = -CUDART_INF_F;
  for (std::size_t e = threadIdx.x; e < experts; e += blockDim.x) {
    scores[e] = logits[e];
    maximum = fmaxf(maximum, logits[e]);
  }
  maximum = detail::block_max_reduce<float, kThreads>(maximum);

  float sum = 0.0f;
  for (std::size_t e = threadIdx.x; e < experts; e += blockDim.x) {
    probabilities[e] = expf(scores[e] - maximum);
    sum += probabilities[e];
  }
  sum = detail::block_sum_reduce<float, kThreads>(sum);
  for (std::size_t e = threadIdx.x; e < experts; e += blockDim.x) {
    probabilities[e] /= sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float selected_sum = 0.0f;
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      std::size_t best = 0;
      float best_score = -CUDART_INF_F;
#pragma unroll 1
      for (std::size_t e = 0; e < experts; ++e) {
        if (scores[e] > best_score) {
          best = e;
          best_score = scores[e];
        }
      }
      ids[rank] = static_cast<std::int32_t>(best);
      weights[rank] = probabilities[best];
      selected_sum += weights[rank];
      scores[best] = -CUDART_INF_F; // Do not select this expert again.
    }
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      weights[rank] /= selected_sum;
    }
  }
}

__global__ void combine_kernel(const float* parts, const float* weights,
                               float* output, std::size_t hidden_size,
                               std::size_t top_k) {
  for (std::size_t d = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       d < hidden_size; d += std::size_t{blockDim.x} * gridDim.x) {
    float sum = 0.0f;
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      sum += weights[rank] * parts[rank * hidden_size + d];
    }
    output[d] = sum;
  }
}

} // namespace

void moe_route(ConstTensorView logits, TensorView expert_ids,
               TensorView expert_weights, cudaStream_t stream) {
  detail::assert_tensor(logits);
  detail::assert_tensor(expert_ids);
  detail::assert_tensor(expert_weights);
  assert(logits.dtype() == DataType::Float32 && logits.rank() == 2
         && logits.shape()[0] == 1);
  assert(expert_ids.dtype() == DataType::Int32 && expert_ids.rank() == 2
         && expert_ids.shape()[0] == 1);
  assert(expert_weights.dtype() == DataType::Float32
         && same_shape(expert_ids, expert_weights));
  const auto experts = logits.shape()[1], top_k = expert_ids.shape()[1];
  assert(experts > 0 && experts <= kMaxMoEExperts && top_k > 0
         && top_k <= experts);
  route_kernel<<<1, kThreads, 2 * experts * sizeof(float), stream>>>(
      logits.data<float>(), expert_ids.data<std::int32_t>(),
      expert_weights.data<float>(), experts, top_k);
  check_cuda(cudaGetLastError(), "MoE route kernel launch");
}

void moe_combine(ConstTensorView expert_outputs, ConstTensorView expert_weights,
                 TensorView output, cudaStream_t stream) {
  detail::assert_tensor(expert_outputs);
  detail::assert_tensor(expert_weights);
  detail::assert_tensor(output);
  assert(same_dtype(expert_outputs, expert_weights, output)
         && output.dtype() == DataType::Float32);
  assert(expert_outputs.rank() == 2 && expert_weights.rank() == 2
         && expert_weights.shape()[0] == 1 && output.rank() == 2
         && output.shape()[0] == 1);
  const auto top_k = expert_outputs.shape()[0];
  const auto hidden_size = expert_outputs.shape()[1];
  assert(top_k > 0 && hidden_size > 0 && expert_weights.shape()[1] == top_k
         && output.shape()[1] == hidden_size);
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((hidden_size + kThreads - 1) / kThreads, 65535));
  combine_kernel<<<blocks, kThreads, 0, stream>>>(
      expert_outputs.data<float>(), expert_weights.data<float>(),
      output.data<float>(), hidden_size, top_k);
  check_cuda(cudaGetLastError(), "MoE combine kernel launch");
}

} // namespace nanovllm::kernels::cuda
