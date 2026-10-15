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

} // namespace

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
