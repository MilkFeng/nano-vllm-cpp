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

__global__ void route_kernel(const float* __restrict__ logits,
                             std::int32_t* __restrict__ ids,
                             float* __restrict__ weights, std::size_t experts,
                             std::size_t top_k) {
  const std::size_t token = blockIdx.x;
  logits += token * experts;
  ids += token * top_k;
  weights += token * top_k;
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
      scores[best] = -CUDART_INF_F;
    }
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      weights[rank] /= selected_sum;
    }
  }
}

__global__ void combine_kernel(const float* __restrict__ parts,
                               const float* __restrict__ weights,
                               float* __restrict__ output,
                               std::size_t hidden_size, std::size_t top_k,
                               std::size_t tokens) {
  const std::size_t start_d =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start_d; index < tokens * hidden_size;
       index += step) {
    const auto token = index / hidden_size;
    const auto d = index % hidden_size;
    float sum = 0.0f;
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      sum += weights[token * top_k + rank]
             * parts[(token * top_k + rank) * hidden_size + d];
    }
    output[index] = sum;
  }
}

__global__ void add_shared_kernel(const float* __restrict__ shared_output,
                                  const float* __restrict__ gate_logits,
                                  float* __restrict__ output,
                                  std::size_t hidden_size, std::size_t tokens) {
  const std::size_t start_d =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start_d; index < tokens * hidden_size;
       index += step) {
    const float logit = gate_logits[index / hidden_size];
    const float exponential = expf(-fabsf(logit));
    const float weight = logit >= 0.0f ? 1.0f / (1.0f + exponential)
                                       : exponential / (1.0f + exponential);
    output[index] += weight * shared_output[index];
  }
}

} // namespace

void moe_route(ConstTensorView logits, TensorView expert_ids,
               TensorView expert_weights, cudaStream_t stream) {
  detail::assert_tensor(logits);
  detail::assert_tensor(expert_ids);
  detail::assert_tensor(expert_weights);
  assert(same_device(logits, expert_ids, expert_weights));
  assert(logits.dtype() == DataType::Float32);
  assert(logits.rank() == 2);
  const auto tokens = logits.shape()[0], experts = logits.shape()[1];
  assert(tokens > 0);
  assert(tokens <= std::numeric_limits<std::int32_t>::max());
  assert(experts > 0);
  assert(experts <= kMaxMoEExperts);
  assert(expert_ids.dtype() == DataType::Int32);
  assert(expert_ids.rank() == 2);
  assert(expert_ids.shape()[0] == tokens);
  assert(same_dtype(logits, expert_weights));
  assert(same_shape(expert_ids, expert_weights));
  const auto top_k = expert_ids.shape()[1];
  assert(top_k > 0);
  assert(top_k <= experts);
  route_kernel<<<static_cast<unsigned>(tokens), kThreads,
                 2 * experts * sizeof(float), stream>>>(
      logits.data<float>(), expert_ids.data<std::int32_t>(),
      expert_weights.data<float>(), experts, top_k);
  check_cuda(cudaGetLastError(), "MoE route kernel launch");
}

void moe_combine(ConstTensorView expert_outputs, ConstTensorView expert_weights,
                 TensorView output, cudaStream_t stream) {
  detail::assert_tensor(expert_outputs);
  detail::assert_tensor(expert_weights);
  detail::assert_tensor(output);
  assert(same_device(expert_outputs, expert_weights, output));
  assert(same_dtype(expert_outputs, expert_weights, output));
  assert(output.dtype() == DataType::Float32);
  assert(output.rank() == 2);
  const auto tokens = output.shape()[0], hidden_size = output.shape()[1];
  assert(tokens > 0);
  assert(hidden_size > 0);
  assert(expert_weights.rank() == 2);
  assert(expert_weights.shape()[0] == tokens);
  const auto top_k = expert_weights.shape()[1];
  assert(top_k > 0);
  assert(expert_outputs.rank() == 2);
  assert(expert_outputs.shape()[0] == tokens * top_k);
  assert(expert_outputs.shape()[1] == hidden_size);
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((output.numel() + kThreads - 1) / kThreads, 65535));
  combine_kernel<<<blocks, kThreads, 0, stream>>>(
      expert_outputs.data<float>(), expert_weights.data<float>(),
      output.data<float>(), hidden_size, top_k, tokens);
  check_cuda(cudaGetLastError(), "MoE combine kernel launch");
}

void moe_add_shared(ConstTensorView shared_output, ConstTensorView gate_logits,
                    TensorView output, cudaStream_t stream) {
  detail::assert_tensor(shared_output);
  detail::assert_tensor(gate_logits);
  detail::assert_tensor(output);
  assert(same_device(shared_output, gate_logits, output));
  assert(same_dtype(shared_output, gate_logits, output));
  assert(output.dtype() == DataType::Float32);
  assert(output.rank() == 2);
  const auto tokens = output.shape()[0], hidden_size = output.shape()[1];
  assert(tokens > 0);
  assert(hidden_size > 0);
  assert(same_shape(shared_output, output));
  assert(gate_logits.rank() == 2);
  assert(gate_logits.shape()[0] == tokens);
  assert(gate_logits.shape()[1] == 1);
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((output.numel() + kThreads - 1) / kThreads, 65535));
  add_shared_kernel<<<blocks, kThreads, 0, stream>>>(
      shared_output.data<float>(), gate_logits.data<float>(),
      output.data<float>(), hidden_size, tokens);
  check_cuda(cudaGetLastError(), "MoE shared branch kernel launch");
}

} // namespace nanovllm::kernels::cuda
