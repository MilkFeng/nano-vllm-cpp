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
constexpr std::size_t kMaxExperts = 1024;
constexpr unsigned kExpertsPerThread = (kMaxExperts + kThreads - 1) / kThreads;

__global__ void route_kernel(const float* __restrict__ logits,
                             std::int32_t* __restrict__ ids,
                             float* __restrict__ weights, std::size_t experts,
                             std::size_t top_k) {
  const std::size_t token = blockIdx.x;

  logits += token * experts;
  ids += token * top_k;
  weights += token * top_k;

  extern __shared__ float storage[];

  float* probs = storage;
  float scores[kExpertsPerThread];
  auto local_best = detail::ArgMaxOp::identity();

  float maximum = -CUDART_INF_F;
#pragma unroll
  for (unsigned item = 0; item < kExpertsPerThread; ++item) {
    const auto e = threadIdx.x + item * kThreads;
    if (e < experts) {
      const float logit = logits[e];
      scores[item] = logit;
      maximum = fmaxf(maximum, logit);
      local_best =
          detail::ArgMaxOp::apply(local_best, {logit, static_cast<int>(e)});
    }
  }
  maximum = detail::block_max_reduce<float, kThreads>(maximum);

  float sum = 0.0f;
#pragma unroll
  for (unsigned item = 0; item < kExpertsPerThread; ++item) {
    const auto e = threadIdx.x + item * kThreads;
    if (e < experts) {
      probs[e] = __expf(scores[item] - maximum);
      sum += probs[e];
    }
  }
  sum = detail::block_sum_reduce<float, kThreads>(sum);

  for (std::size_t e = threadIdx.x; e < experts; e += blockDim.x) {
    probs[e] /= sum;
  }

  __syncthreads();

  float selected_sum = 0.0f;
  unsigned selected_mask = 0;
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    const auto best = detail::block_argmax_reduce<kThreads>(local_best);

    if (threadIdx.x == 0) {
      ids[rank] = best.index;
      weights[rank] = probs[best.index];
      selected_sum += weights[rank];
    }

    if (rank + 1 < top_k && best.index == local_best.index) {
      const auto selected_item = (best.index - threadIdx.x) / kThreads;

      selected_mask |= 1u << selected_item;
      local_best = detail::ArgMaxOp::identity();

#pragma unroll
      for (unsigned item = 0; item < kExpertsPerThread; ++item) {
        const auto e = threadIdx.x + item * kThreads;
        if (e < experts && (selected_mask & (1u << item)) == 0) {
          local_best = detail::ArgMaxOp::apply(
              local_best, {scores[item], static_cast<int>(e)});
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      weights[rank] /= selected_sum;
    }
  }
}

template <typename T>
__global__ void moe_gather_kernel(const T* __restrict__ input,
                                  const std::int32_t* __restrict__ order,
                                  T* __restrict__ grouped, std::size_t count,
                                  std::size_t width, std::size_t tokens,
                                  std::size_t top_k) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (auto index = start; index < count; index += step) {
    const auto rank = order[index / width];
    assert(rank >= 0 && rank / top_k < tokens);
    grouped[index] =
        input[(static_cast<std::size_t>(rank) / top_k) * width + index % width];
  }
}

__global__ void moe_scatter_kernel(const float* __restrict__ grouped,
                                   const std::int32_t* __restrict__ order,
                                   float* __restrict__ output,
                                   std::size_t count, std::size_t width,
                                   std::size_t rows) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (auto index = start; index < count; index += step) {
    const auto rank = order[index / width];
    assert(rank >= 0 && rank < rows);
    output[static_cast<std::size_t>(rank) * width + index % width] =
        grouped[index];
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
      const std::size_t index = token * top_k + rank;
      sum += weights[index] * parts[index * hidden_size + d];
    }
    output[index] = sum;
  }
}

template <typename T>
__global__ void add_shared_kernel(const T* __restrict__ shared_output,
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
    output[index] += weight * detail::to_float(shared_output[index]);
  }
}

template <typename T>
__global__ void combine_mapped_kernel(
    float* parts, const float* cpu_parts, const float* weights, float* output,
    const T* shared, const float* gate, std::size_t hidden, std::size_t top_k,
    const std::int32_t* ids, const FP8ExpertWeights* routes,
    const volatile std::uint32_t* ready, volatile std::uint32_t* error) {
  __shared__ int usable;
  if (threadIdx.x == 0) {
    usable = 1;
    if (ready) {
      unsigned long long start, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      std::uint32_t value;
      while ((value = *ready) == 0) {
        __nanosleep(64);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - start > 200000000ull) {
          if (error) {
            error[blockIdx.x] = 1;
          }
          value = 2;
          break;
        }
      }
      __threadfence_system();
      usable = value == 1;
    }
  }
  __syncthreads();
  const auto d = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (d >= hidden) {
    return;
  }
  if (!usable) {
    output[d] = 0;
    return;
  }
  float sum = 0;
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    float value;
    const int cpu_row = routes[ids[rank]].gate ? -1 : int(rank);
    if (cpu_row >= 0) {
      value = cpu_parts[std::size_t(cpu_row) * hidden + d];
      parts[rank * hidden + d] = value;
    } else {
      value = parts[rank * hidden + d];
    }
    sum += weights[rank] * value;
  }
  if (shared) {
    const float logit = gate[0];
    const float exponential = expf(-fabsf(logit));
    const float weight = logit >= 0.0f ? 1.0f / (1.0f + exponential)
                                       : exponential / (1.0f + exponential);
    sum += weight * detail::to_float(shared[d]);
  }
  output[d] = sum;
}

__global__ void publish_decode_kernel(const float* input,
                                      const std::int32_t* ids,
                                      float* host_input, std::int32_t* host_ids,
                                      volatile std::uint32_t* ready,
                                      std::size_t hidden, std::size_t top_k) {
  for (std::size_t d = threadIdx.x; d < hidden; d += blockDim.x) {
    host_input[d] = input[d];
  }
  for (std::size_t r = threadIdx.x; r < top_k; r += blockDim.x) {
    host_ids[r] = ids[r];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *ready = 1;
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
  assert(experts <= kMaxExperts);
  assert(expert_ids.dtype() == DataType::Int32);
  assert(expert_ids.rank() == 2);
  assert(expert_ids.shape()[0] == tokens);
  assert(same_dtype(logits, expert_weights));
  assert(same_shape(expert_ids, expert_weights));

  const auto top_k = expert_ids.shape()[1];

  assert(top_k > 0);
  assert(top_k <= experts);

  route_kernel<<<static_cast<unsigned>(tokens), kThreads,
                 experts * sizeof(float), stream>>>(
      logits.data<float>(), expert_ids.data<std::int32_t>(),
      expert_weights.data<float>(), experts, top_k);
  check_cuda(cudaGetLastError(), "MoE route kernel launch");
}

void moe_gather(ConstTensorView input, ConstTensorView order, std::size_t top_k,
                TensorView grouped, cudaStream_t stream) {
  detail::assert_tensor(input);
  detail::assert_tensor(order);
  detail::assert_tensor(grouped);

  assert(same_device(input, order, grouped));
  assert(same_dtype(input, grouped));
  assert(input.rank() == 2 && grouped.rank() == 2 && order.rank() == 1);
  assert(order.dtype() == DataType::Int32
         && order.numel() == grouped.shape()[0]);
  assert(input.shape()[1] > 0 && grouped.shape()[1] == input.shape()[1]);
  assert(top_k > 0);

  if (grouped.numel() == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(
      (grouped.numel() + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    moe_gather_kernel<<<blocks, kThreads, 0, stream>>>(
        input.data<T>(), order.data<std::int32_t>(), grouped.data<T>(),
        grouped.numel(), input.shape()[1], input.shape()[0], top_k);
  });
  check_cuda(cudaGetLastError(), "MoE gather kernel launch");
}

void moe_scatter(ConstTensorView grouped, ConstTensorView order,
                 TensorView output, cudaStream_t stream) {
  detail::assert_tensor(grouped);
  detail::assert_tensor(order);
  detail::assert_tensor(output);

  assert(same_device(grouped, order, output));
  assert(grouped.dtype() == DataType::Float32 && same_dtype(grouped, output));
  assert(grouped.rank() == 2 && same_shape(grouped, output)
         && order.rank() == 1);
  assert(order.dtype() == DataType::Int32
         && order.numel() == grouped.shape()[0]);
  assert(grouped.shape()[1] > 0);

  if (output.numel() == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((output.numel() + kThreads - 1) / kThreads, 65535));
  moe_scatter_kernel<<<blocks, kThreads, 0, stream>>>(
      grouped.data<float>(), order.data<std::int32_t>(), output.data<float>(),
      output.numel(), output.shape()[1], output.shape()[0]);
  check_cuda(cudaGetLastError(), "MoE scatter kernel launch");
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
  assert(same_dtype(gate_logits, output));
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
  detail::dispatch_float(shared_output.dtype(), [&]<typename T>() {
    add_shared_kernel<<<blocks, kThreads, 0, stream>>>(
        shared_output.data<T>(), gate_logits.data<float>(),
        output.data<float>(), hidden_size, tokens);
  });
  check_cuda(cudaGetLastError(), "MoE shared branch kernel launch");
}

void moe_combine_routed(TensorView expert_outputs, const float* cpu_parts,
                        ConstTensorView ids, ConstTensorView routes,
                        ConstTensorView expert_weights, TensorView output,
                        ConstTensorView shared_output,
                        ConstTensorView gate_logits, cudaStream_t stream,
                        const volatile std::uint32_t* ready,
                        volatile std::uint32_t* error) {
  const auto hidden = output.shape()[1];
  const auto dtype =
      shared_output.numel() ? shared_output.dtype() : DataType::Float32;
  detail::dispatch_float(dtype, [&]<typename T>() {
    combine_mapped_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        expert_outputs.data<float>(), cpu_parts, expert_weights.data<float>(),
        output.data<float>(), shared_output.data<T>(),
        gate_logits.data<float>(), hidden, ids.numel(),
        ids.data<std::int32_t>(),
        reinterpret_cast<const FP8ExpertWeights*>(routes.data()), ready, error);
  });
  check_cuda(cudaGetLastError(), "routed mapped combine launch");
}

void moe_publish_decode(ConstTensorView input, ConstTensorView ids,
                        float* mapped_input, std::int32_t* mapped_ids,
                        volatile std::uint32_t* ready, cudaStream_t stream,
                        bool publish_only) {
  publish_decode_kernel<<<1, publish_only ? 1 : 256, 0, stream>>>(
      input.data<float>(), ids.data<std::int32_t>(), mapped_input, mapped_ids,
      ready, publish_only ? 0 : input.shape()[1],
      publish_only ? 0 : ids.numel());
  check_cuda(cudaGetLastError(), "publish mapped decode input");
}

} // namespace nanovllm::kernels::cuda
