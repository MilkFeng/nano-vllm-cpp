#include <cuda_fp8.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kWarps = 4;
constexpr std::size_t kBlockSize = kWarps * 32;

constexpr unsigned kVectorSize = 4;

__device__ __forceinline__ float4
load_input4_or_zero(const float* __restrict__ input, std::size_t M,
                    std::size_t K, std::size_t m, std::size_t k) {
  if (m >= M || k >= K) {
    return make_float4(0, 0, 0, 0);
  }

  const auto* address = input + m * K + k;

  if (K - k >= kVectorSize
      && (reinterpret_cast<std::uintptr_t>(address) & 15) == 0) {
    return *reinterpret_cast<const float4*>(address);
  } else {
    return make_float4(address[0], K - k > 1 ? address[1] : 0.0f,
                       K - k > 2 ? address[2] : 0.0f,
                       K - k > 3 ? address[3] : 0.0f);
  }
}

__device__ __forceinline__ std::uint32_t load_weight4_or_zero(
    const std::uint8_t* __restrict__ weight, std::size_t N, std::size_t K,
    std::size_t n, std::size_t k, std::size_t weight_stride) {
  if (n >= N || k >= K) {
    return 0;
  }

  const auto* address = weight + n * weight_stride + k;
  if (K - k >= kVectorSize
      && (reinterpret_cast<std::uintptr_t>(address) & 3) == 0) {
    return *reinterpret_cast<const std::uint32_t*>(address);
  }

  std::uint32_t packed = 0;
#pragma unroll
  for (unsigned i = 0; i < kVectorSize; ++i) {
    if (i < K - k) {
      packed |= static_cast<std::uint32_t>(address[i]) << (8 * i);
    }
  }
  return packed;
}

__device__ __forceinline__ float4 dequantize4(std::uint32_t packed,
                                              float scale) {
  __nv_fp8x4_e4m3 values;
  values.__x = packed;
  const auto decoded = static_cast<float4>(values);
  return make_float4(decoded.x * scale, decoded.y * scale, decoded.z * scale,
                     decoded.w * scale);
}

__global__ void gemv_kernel(const float* __restrict__ input,
                            const std::uint8_t* __restrict__ weight,
                            const float* __restrict__ weight_scale_inv,
                            float* __restrict__ output, std::size_t N,
                            std::size_t K, std::size_t weight_stride) {
  const unsigned lane = threadIdx.x % 32;
  const auto n =
      static_cast<std::size_t>(blockIdx.x) * kWarps + threadIdx.x / 32;

  if (n >= N) {
    return;
  }

  const auto k_blocks = (K + kBlockSize - 1) / kBlockSize;

  float sum = 0.0f;
  for (std::size_t k = lane * kVectorSize; k < K; k += 32 * kVectorSize) {
    const auto a = load_input4_or_zero(input, 1, K, 0, k);
    const auto packed = load_weight4_or_zero(weight, N, K, n, k, weight_stride);
    const auto scale =
        weight_scale_inv[(n / kBlockSize) * k_blocks + k / kBlockSize];

    const auto b = dequantize4(packed, scale);

    sum = fmaf(a.x, b.x, sum);
    sum = fmaf(a.y, b.y, sum);
    sum = fmaf(a.z, b.z, sum);
    sum = fmaf(a.w, b.w, sum);
  }

  sum = detail::warp_sum_reduce<float, false>(sum);

  if (lane == 0) {
    output[n] = sum;
  }
}

template <unsigned BX, unsigned BY, unsigned TM, unsigned TN, unsigned KS>
__global__ void matmul_kernel(const float* __restrict__ input,
                              const std::uint8_t* __restrict__ weight,
                              const float* __restrict__ weight_scale_inv,
                              float* __restrict__ output, std::size_t M,
                              std::size_t N, std::size_t K,
                              std::size_t weight_stride) {
  constexpr unsigned BM = BY * TM;
  constexpr unsigned BN = BX * TN;

  constexpr unsigned num_threads = BX * BY;

  constexpr unsigned a_vectors = BM * KS / kVectorSize;
  constexpr unsigned b_vectors = BN * KS / kVectorSize;
  constexpr unsigned a_loads = (a_vectors + num_threads - 1) / num_threads;
  constexpr unsigned b_loads = (b_vectors + num_threads - 1) / num_threads;

  static_assert(KS % kVectorSize == 0 && kBlockSize % KS == 0);
  static_assert(kBlockSize % BN == 0);

  __align__(16) __shared__ float tile_a[2][BM][KS + kVectorSize];
  __align__(16) __shared__ float tile_b[2][KS][BN + kVectorSize];

  const unsigned tid = threadIdx.x + threadIdx.y * BX;

  const auto m_begin = static_cast<std::size_t>(blockIdx.y) * BM;
  const auto n_begin = static_cast<std::size_t>(blockIdx.x) * BN;
  const unsigned thread_m = threadIdx.y * TM;
  const unsigned thread_n = threadIdx.x * TN;

  const auto k_blocks = (K + kBlockSize - 1) / kBlockSize;

  unsigned stage = 0;

  float acc[TM][TN] = {};

  float4 prefetched_a[a_loads];
  std::uint32_t prefetched_b[b_loads];

#pragma unroll
  for (unsigned i = 0; i < a_loads; ++i) {
    const unsigned vector = tid + i * num_threads;
    if (vector < a_vectors) {
      const unsigned m = vector / (KS / kVectorSize);
      const unsigned k = (vector % (KS / kVectorSize)) * kVectorSize;
      const auto a = load_input4_or_zero(input, M, K, m_begin + m, k);
      *reinterpret_cast<float4*>(&tile_a[stage][m][k]) = a;
    }
  }

  const float scale = weight_scale_inv[(n_begin / kBlockSize) * k_blocks];

#pragma unroll
  for (unsigned i = 0; i < b_loads; ++i) {
    const unsigned vector = tid + i * num_threads;
    if (vector < b_vectors) {
      const unsigned n = vector / (KS / kVectorSize);
      const unsigned k = (vector % (KS / kVectorSize)) * kVectorSize;
      const auto packed =
          load_weight4_or_zero(weight, N, K, n_begin + n, k, weight_stride);
      const auto b = dequantize4(packed, scale);
      tile_b[stage][k][n] = b.x;
      tile_b[stage][k + 1][n] = b.y;
      tile_b[stage][k + 2][n] = b.z;
      tile_b[stage][k + 3][n] = b.w;
    }
  }

  __syncthreads();

  for (std::size_t k = 0; k < K; k += KS) {
    const auto next_k = k + KS;

    // prefetch
    if (next_k < K) {
#pragma unroll
      for (unsigned i = 0; i < a_loads; ++i) {
        const unsigned vector = tid + i * num_threads;
        if (vector < a_vectors) {
          const unsigned m = vector / (KS / kVectorSize);
          const unsigned k = (vector % (KS / kVectorSize)) * kVectorSize;
          prefetched_a[i] =
              load_input4_or_zero(input, M, K, m_begin + m, next_k + k);
        }
      }
#pragma unroll
      for (unsigned i = 0; i < b_loads; ++i) {
        const unsigned vector = tid + i * num_threads;
        if (vector < b_vectors) {
          const unsigned n = vector / (KS / kVectorSize);
          const unsigned k = (vector % (KS / kVectorSize)) * kVectorSize;
          prefetched_b[i] = load_weight4_or_zero(weight, N, K, n_begin + n,
                                                 next_k + k, weight_stride);
        }
      }
    }

    // compute
#pragma unroll
    for (unsigned t = 0; t < KS; ++t) {
      float reg_a[TM];
      float reg_b[TN];
#pragma unroll
      for (unsigned i = 0; i < TM; ++i) {
        reg_a[i] = tile_a[stage][thread_m + i][t];
      }
#pragma unroll
      for (unsigned j = 0; j < TN; ++j) {
        reg_b[j] = tile_b[stage][t][thread_n + j];
      }
#pragma unroll
      for (unsigned i = 0; i < TM; ++i) {
#pragma unroll
        for (unsigned j = 0; j < TN; ++j) {
          acc[i][j] = fmaf(reg_a[i], reg_b[j], acc[i][j]);
        }
      }
    }

    // store
    if (next_k < K) {
#pragma unroll
      for (unsigned i = 0; i < a_loads; ++i) {
        const unsigned vector = tid + i * num_threads;
        if (vector < a_vectors) {
          const unsigned m = vector / (KS / kVectorSize);
          const unsigned k = (vector % (KS / kVectorSize)) * kVectorSize;
          *reinterpret_cast<float4*>(&tile_a[stage ^ 1][m][k]) =
              prefetched_a[i];
        }
      }

      const float scale = weight_scale_inv[(n_begin / kBlockSize) * k_blocks
                                           + next_k / kBlockSize];
#pragma unroll
      for (unsigned i = 0; i < b_loads; ++i) {
        const unsigned vector = tid + i * num_threads;
        if (vector < b_vectors) {
          const unsigned n = vector / (KS / kVectorSize);
          const unsigned k = (vector % (KS / kVectorSize)) * kVectorSize;
          const auto b = dequantize4(prefetched_b[i], scale);
          tile_b[stage ^ 1][k][n] = b.x;
          tile_b[stage ^ 1][k + 1][n] = b.y;
          tile_b[stage ^ 1][k + 2][n] = b.z;
          tile_b[stage ^ 1][k + 3][n] = b.w;
        }
      }
    }

    __syncthreads();
    stage ^= 1;
  }

#pragma unroll
  for (unsigned i = 0; i < TM; ++i) {
    const auto m = m_begin + thread_m + i;
#pragma unroll
    for (unsigned j = 0; j < TN; ++j) {
      const auto n = n_begin + thread_n + j;
      if (m < M && n < N) {
        output[m * N + n] = acc[i][j];
      }
    }
  }
}

__global__ void expert_gate_up_kernel(const float* input, FP8ExpertBatch batch,
                                       float* activated, std::size_t hidden,
                                       std::size_t intermediate,
                                       const FP8ExpertWeights* routes = nullptr,
                                       const std::int32_t* ids = nullptr) {
  const auto expert = blockIdx.y;
  const auto n = std::size_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  const auto lane = threadIdx.x % 32;
  if (n >= intermediate) return;
  const auto weights = routes ? routes[ids[expert]] : batch.weights[expert];
  if (!weights.gate) return;
  const auto blocks = (hidden + 127) / 128;
  float gate = 0, up = 0;
  for (std::size_t k = lane * 4; k < hidden; k += 128) {
    const auto x = load_input4_or_zero(input, 1, hidden, 0, k);
    const auto index = (n / 128) * blocks + k / 128;
    const auto g = dequantize4(load_weight4_or_zero(weights.gate, intermediate, hidden, n, k, hidden), weights.gate_scale[index]);
    const auto u = dequantize4(load_weight4_or_zero(weights.up, intermediate, hidden, n, k, hidden), weights.up_scale[index]);
    gate = fmaf(x.x, g.x, gate); gate = fmaf(x.y, g.y, gate);
    gate = fmaf(x.z, g.z, gate); gate = fmaf(x.w, g.w, gate);
    up = fmaf(x.x, u.x, up); up = fmaf(x.y, u.y, up);
    up = fmaf(x.z, u.z, up); up = fmaf(x.w, u.w, up);
  }
  gate = detail::warp_sum_reduce<float, false>(gate);
  up = detail::warp_sum_reduce<float, false>(up);
  if (lane == 0) activated[expert * intermediate + n] = (gate / (1.0f + __expf(-gate))) * up;
}

__global__ void expert_down_kernel(const float* activated, FP8ExpertBatch batch,
                                    float* output, std::size_t hidden,
                                    std::size_t intermediate,
                                    const FP8ExpertWeights* routes = nullptr,
                                    const std::int32_t* ids = nullptr) {
  const auto expert = blockIdx.y;
  const auto n = std::size_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  const auto lane = threadIdx.x % 32;
  if (n >= hidden) return;
  const auto weights = routes ? routes[ids[expert]] : batch.weights[expert];
  if (!weights.down) return;
  const auto blocks = (intermediate + 127) / 128;
  float sum = 0;
  for (std::size_t k = lane * 4; k < intermediate; k += 128) {
    const auto x = load_input4_or_zero(activated, batch.count, intermediate, expert, k);
    const auto w = dequantize4(load_weight4_or_zero(weights.down, hidden, intermediate, n, k, intermediate), weights.down_scale[(n / 128) * blocks + k / 128]);
    sum = fmaf(x.x, w.x, sum); sum = fmaf(x.y, w.y, sum);
    sum = fmaf(x.z, w.z, sum); sum = fmaf(x.w, w.w, sum);
  }
  sum = detail::warp_sum_reduce<float, false>(sum);
  if (lane == 0) output[std::size_t(routes ? expert : batch.rows[expert]) * hidden + n] = sum;
}

} // namespace

void fp8_experts_decode(ConstTensorView input, const FP8ExpertBatch& batch,
                         TensorView activated, TensorView output, cudaStream_t stream) {
  assert(batch.count > 0 && batch.count <= 8);
  assert(input.rank() == 2 && input.shape()[0] == 1);
  assert(input.dtype() == DataType::Float32 && same_dtype(input, activated, output));
  assert(activated.rank() == 2 && activated.shape()[0] >= batch.count);
  const auto hidden = input.shape()[1], intermediate = activated.shape()[1];
  expert_gate_up_kernel<<<dim3((intermediate + 3) / 4, batch.count), 128, 0, stream>>>(input.data<float>(), batch, activated.data<float>(), hidden, intermediate);
  check_cuda(cudaGetLastError(), "batched FP8 Gate/Up decode launch");
  expert_down_kernel<<<dim3((hidden + 3) / 4, batch.count), 128, 0, stream>>>(activated.data<float>(), batch, output.data<float>(), hidden, intermediate);
  check_cuda(cudaGetLastError(), "batched FP8 Down decode launch");
}

void fp8_experts_decode_routed(ConstTensorView input, ConstTensorView ids,
                                ConstTensorView routes, TensorView activated,
                                TensorView output, cudaStream_t stream) {
  FP8ExpertBatch batch{};
  batch.count = ids.numel();
  const auto hidden = input.shape()[1], intermediate = activated.shape()[1];
  const auto* table = reinterpret_cast<const FP8ExpertWeights*>(routes.data());
  expert_gate_up_kernel<<<dim3((intermediate + 3) / 4, batch.count), 128, 0, stream>>>(
      input.data<float>(), batch, activated.data<float>(), hidden, intermediate, table, ids.data<std::int32_t>());
  check_cuda(cudaGetLastError(), "routed FP8 Gate/Up launch");
  expert_down_kernel<<<dim3((hidden + 3) / 4, batch.count), 128, 0, stream>>>(
      activated.data<float>(), batch, output.data<float>(), hidden, intermediate, table, ids.data<std::int32_t>());
  check_cuda(cudaGetLastError(), "routed FP8 Down launch");
}

void linear_fp8(ConstTensorView input, ConstTensorView weight,
                ConstTensorView weight_scale_inv, TensorView output,
                cudaStream_t stream) {
  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(weight_scale_inv);
  detail::assert_tensor(output);

  assert(input.dtype() == DataType::Float32
         && output.dtype() == DataType::Float32);
  assert(weight.dtype() == DataType::Float8E4M3FN);
  assert(weight_scale_inv.dtype() == DataType::Float32);
  assert(same_device(input, weight, weight_scale_inv, output));
  assert(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2);
  assert(weight_scale_inv.rank() == 2);

  const auto M = input.shape()[0];
  const auto K = input.shape()[1];
  const auto N = output.shape()[1];

  [[maybe_unused]] const auto k_blocks = (K + kBlockSize - 1) / kBlockSize;
  [[maybe_unused]] const auto n_blocks = (N + kBlockSize - 1) / kBlockSize;

  assert(output.shape()[0] == M);
  assert(weight.shape()[0] >= N && weight.shape()[1] >= K);
  assert(weight_scale_inv.shape()[0] == n_blocks
         && weight_scale_inv.shape()[1] == k_blocks);

  if (M == 0 || N == 0 || K == 0) {
    return;
  }

  if (M == 1) {
    gemv_kernel<<<static_cast<unsigned>((N + kWarps - 1) / kWarps), kWarps * 32,
                  0, stream>>>(input.data<float>(), weight.data<std::uint8_t>(),
                               weight_scale_inv.data<float>(),
                               output.data<float>(), N, K, weight.shape()[1]);
  } else {
    constexpr unsigned BX = 16, BY = 8, TM = 2, TN = 4, KS = 32;

    const dim3 block(BX, BY);
    const dim3 grid((N + BX * TN - 1) / (BX * TN),
                    (M + BY * TM - 1) / (BY * TM));

    matmul_kernel<BX, BY, TM, TN, KS><<<grid, block, 0, stream>>>(
        input.data<float>(), weight.data<std::uint8_t>(),
        weight_scale_inv.data<float>(), output.data<float>(), M, N, K,
        weight.shape()[1]);
  }
  check_cuda(cudaGetLastError(), "fused FP8 weight dequantization GEMM launch");
}

} // namespace nanovllm::kernels::cuda
