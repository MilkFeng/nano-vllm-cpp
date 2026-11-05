#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr int kVectorSize = 4;
constexpr int kTileK = 16;

__host__ __device__ constexpr int ceil_div(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}

// Vectorize aligned FP32 rows; partial or unaligned vectors use scalar loads.
template <typename T>
__device__ __forceinline__ float4 load_vector(const T* matrix, int row,
                                              std::size_t col, int rows,
                                              int cols) {
  float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (row >= rows || col >= static_cast<std::size_t>(cols)) {
    return result;
  }

  const T* pointer = matrix + static_cast<std::size_t>(row) * cols + col;

  if constexpr (std::is_same_v<T, float>) {
    const bool aligned = (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;

    if (aligned && col + kVectorSize <= static_cast<std::size_t>(cols)) {
      return *reinterpret_cast<const float4*>(pointer);
    }
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if ((reinterpret_cast<std::uintptr_t>(pointer) & 7) == 0
        && col + kVectorSize <= static_cast<std::size_t>(cols)) {
      const uint2 packed = *reinterpret_cast<const uint2*>(pointer);
      return make_float4(__uint_as_float(packed.x << 16),
                         __uint_as_float(packed.x & 0xffff0000u),
                         __uint_as_float(packed.y << 16),
                         __uint_as_float(packed.y & 0xffff0000u));
    }
  }

  result.x = detail::to_float(pointer[0]);

  if (col + 1 < static_cast<std::size_t>(cols)) {
    result.y = detail::to_float(pointer[1]);
  }

  if (col + 2 < static_cast<std::size_t>(cols)) {
    result.z = detail::to_float(pointer[2]);
  }

  if (col + 3 < static_cast<std::size_t>(cols)) {
    result.w = detail::to_float(pointer[3]);
  }

  return result;
}

template <typename T>
__device__ __forceinline__ void store_vector(T* output, const T* bias,
                                             float4 value, int row, int col,
                                             int m, int n) {
  if (row >= m || col >= n) {
    return;
  }

  if (bias != nullptr) {
    const float4 offset = load_vector(bias, 0, col, 1, n);

    value.x += offset.x;
    value.y += offset.y;
    value.z += offset.z;
    value.w += offset.w;
  }

  T* pointer = output + static_cast<std::size_t>(row) * n + col;

  if constexpr (std::is_same_v<T, float>) {
    const bool aligned = (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;

    if (aligned && col + kVectorSize <= n) {
      *reinterpret_cast<float4*>(pointer) = value;
      return;
    }
  }

  pointer[0] = detail::from_float<T>(value.x);

  if (col + 1 < n) {
    pointer[1] = detail::from_float<T>(value.y);
  }

  if (col + 2 < n) {
    pointer[2] = detail::from_float<T>(value.z);
  }

  if (col + 3 < n) {
    pointer[3] = detail::from_float<T>(value.w);
  }
}

template <int TM>
__device__ __forceinline__ int swizzle_input_col(int row, int col) {
  constexpr int groups = kTileK / kVectorSize;
  const int group = (row / TM) & (groups - 1);

  return col ^ (group * kVectorSize);
}

__host__ __device__ constexpr int ilog2_pow2(int value) {
  int result = 0;

  while (value > 1) {
    value >>= 1;
    ++result;
  }

  return result;
}

template <int TN>
__device__ __forceinline__ int swizzle_weight_col(int row, int col) {
  constexpr int d = ilog2_pow2(TN / kVectorSize);
  constexpr int shift = d > 3 ? d : 3;
  constexpr int bits = d < 3 ? d : 3;
  constexpr int mask = (1 << bits) - 1;

  const int vector = col / kVectorSize;
  const int swizzled_vector = vector ^ ((vector >> shift) & mask);

  // W is loaded along K, then scattered into shared memory as [K,N].
  // Different K-vector lanes occupy different bank groups on this store.
  // The low two bits stay intact for float4 reads during computation.
  const int k_group = (row / kVectorSize) & 3;

  return ((swizzled_vector * kVectorSize) | (col & 3)) ^ (k_group << 3);
}

template <int BX, int BY, int TM, int TN, typename T, typename W>
__global__ void linear_gemm_kernel(const T* __restrict__ input,
                                   const W* __restrict__ weight,
                                   T* __restrict__ output,
                                   const T* __restrict__ bias, int m, int n,
                                   int k) {
  constexpr int threads = BX * BY;
  constexpr int BM = BY * TM;
  constexpr int BN = BX * TN;
  constexpr int vectors_per_row = kTileK / kVectorSize;
  constexpr int input_vectors = BM * vectors_per_row;
  constexpr int weight_vectors = BN * vectors_per_row;
  constexpr int input_loads = ceil_div(input_vectors, threads);
  constexpr int weight_loads = ceil_div(weight_vectors, threads);

  static_assert(TN % kVectorSize == 0 && BN >= 32);
  static_assert((TN & (TN - 1)) == 0);

  __shared__ __align__(16) float tile_input[2][BM][kTileK];
  __shared__ __align__(16) float tile_weight[2][kTileK][BN];

  const int tid = threadIdx.x + threadIdx.y * BX;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int thread_row = threadIdx.y * TM;
  const int thread_col = threadIdx.x * TN;

  float4 prefetched_input[input_loads];
  float4 prefetched_weight[weight_loads];

#pragma unroll
  for (int i = 0; i < input_loads; ++i) {
    const int vector = tid + i * threads;

    if (vector < input_vectors) {
      const int row = vector / vectors_per_row;
      const int col = (vector % vectors_per_row) * kVectorSize;
      const float4 value = load_vector(input, block_row + row, col, m, k);

      *reinterpret_cast<float4*>(
          &tile_input[0][row][swizzle_input_col<TM>(row, col)]) = value;
    }
  }

#pragma unroll
  for (int i = 0; i < weight_loads; ++i) {
    const int vector = tid + i * threads;

    if (vector < weight_vectors) {
      const int col = vector / vectors_per_row;
      const int row = (vector % vectors_per_row) * kVectorSize;
      const float4 value = load_vector(weight, block_col + col, row, n, k);

      tile_weight[0][row][swizzle_weight_col<TN>(row, col)] = value.x;
      tile_weight[0][row + 1][swizzle_weight_col<TN>(row + 1, col)] = value.y;
      tile_weight[0][row + 2][swizzle_weight_col<TN>(row + 2, col)] = value.z;
      tile_weight[0][row + 3][swizzle_weight_col<TN>(row + 3, col)] = value.w;
    }
  }

  __syncthreads();

  float acc[TM][TN] = {};
  int stage = 0;

  for (std::size_t k_begin = 0; k_begin < static_cast<std::size_t>(k);
       k_begin += kTileK) {
    const bool has_next = k_begin + kTileK < static_cast<std::size_t>(k);

    if (has_next) {
#pragma unroll
      for (int i = 0; i < input_loads; ++i) {
        const int vector = tid + i * threads;

        if (vector < input_vectors) {
          const int row = vector / vectors_per_row;
          const int col = (vector % vectors_per_row) * kVectorSize;

          prefetched_input[i] =
              load_vector(input, block_row + row, k_begin + kTileK + col, m, k);
        }
      }

#pragma unroll
      for (int i = 0; i < weight_loads; ++i) {
        const int vector = tid + i * threads;

        if (vector < weight_vectors) {
          const int col = vector / vectors_per_row;
          const int row = (vector % vectors_per_row) * kVectorSize;

          prefetched_weight[i] = load_vector(weight, block_col + col,
                                             k_begin + kTileK + row, n, k);
        }
      }
    }

#pragma unroll
    for (int t = 0; t < kTileK; ++t) {
      float reg_input[TM];
      float reg_weight[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int row = thread_row + i;

        reg_input[i] = tile_input[stage][row][swizzle_input_col<TM>(row, t)];
      }

#pragma unroll
      for (int j = 0; j < TN; j += kVectorSize) {
        const int col = thread_col + j;
        const float4 value = *reinterpret_cast<const float4*>(
            &tile_weight[stage][t][swizzle_weight_col<TN>(t, col)]);

        reg_weight[j] = value.x;
        reg_weight[j + 1] = value.y;
        reg_weight[j + 2] = value.z;
        reg_weight[j + 3] = value.w;
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = fmaf(reg_input[i], reg_weight[j], acc[i][j]);
        }
      }
    }

    if (has_next) {
#pragma unroll
      for (int i = 0; i < input_loads; ++i) {
        const int vector = tid + i * threads;

        if (vector < input_vectors) {
          const int row = vector / vectors_per_row;
          const int col = (vector % vectors_per_row) * kVectorSize;

          *reinterpret_cast<float4*>(
              &tile_input[stage ^ 1][row][swizzle_input_col<TM>(row, col)]) =
              prefetched_input[i];
        }
      }

#pragma unroll
      for (int i = 0; i < weight_loads; ++i) {
        const int vector = tid + i * threads;

        if (vector < weight_vectors) {
          const int col = vector / vectors_per_row;
          const int row = (vector % vectors_per_row) * kVectorSize;
          const float4 value = prefetched_weight[i];

          tile_weight[stage ^ 1][row][swizzle_weight_col<TN>(row, col)] =
              value.x;
          tile_weight[stage ^ 1][row + 1]
                     [swizzle_weight_col<TN>(row + 1, col)] = value.y;
          tile_weight[stage ^ 1][row + 2]
                     [swizzle_weight_col<TN>(row + 2, col)] = value.z;
          tile_weight[stage ^ 1][row + 3]
                     [swizzle_weight_col<TN>(row + 3, col)] = value.w;
        }
      }
    }

    stage ^= 1;
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = block_row + thread_row + i;

#pragma unroll
    for (int j = 0; j < TN; j += kVectorSize) {
      const int col = block_col + thread_col + j;
      const float4 value =
          make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);

      store_vector(output, bias, value, row, col, m, n);
    }
  }
}

// Decode avoids filling a GEMM tile with a single input row.
template <typename T, typename W>
__global__ void linear_gemv_kernel(const T* __restrict__ input,
                                   const W* __restrict__ weight,
                                   T* __restrict__ output,
                                   const T* __restrict__ bias, int n, int k) {
  constexpr int warps_per_block = 4;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp;

  if (row >= n) {
    return;
  }

  float sum = 0.0f;

  for (std::size_t col = lane * kVectorSize; col < static_cast<std::size_t>(k);
       col += 32 * kVectorSize) {
    const float4 x = load_vector(input, 0, col, 1, k);
    const float4 w = load_vector(weight, row, col, n, k);

    sum = fmaf(x.x, w.x, sum);
    sum = fmaf(x.y, w.y, sum);
    sum = fmaf(x.z, w.z, sum);
    sum = fmaf(x.w, w.w, sum);
  }

  sum = detail::warp_sum_reduce<float, false>(sum);

  if (lane == 0) {
    if (bias != nullptr) {
      sum += detail::to_float(bias[row]);
    }

    output[row] = detail::from_float<T>(sum);
  }
}

template <int BX, int BY, int TM, int TN, typename T, typename W>
void launch_gemm(const T* input, const W* weight, T* output, const T* bias,
                 int m, int n, int k, cudaStream_t stream) {
  const dim3 block(BX, BY);
  const dim3 grid(ceil_div(n, BX * TN), ceil_div(m, BY * TM));

  linear_gemm_kernel<BX, BY, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, output, bias, m, n, k);
}

} // namespace

void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias, cudaStream_t stream) {
  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(output);

  assert(same_dtype(input, output));
  assert(weight.dtype() == input.dtype()
         || (input.dtype() == DataType::Float32
             && weight.dtype() == DataType::BFloat16));
  assert(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2);

  const auto m = input.shape()[0];
  const auto k = input.shape()[1];
  const auto n = weight.shape()[0];

  assert(k > 0 && n > 0 && weight.shape()[1] == k);
  assert(output.shape()[0] == m && output.shape()[1] == n);
  assert(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX);

  if (bias) {
    detail::assert_tensor(*bias);
    assert(same_dtype(*bias, input));
    assert(bias->rank() == 1 && bias->shape()[0] == n);
  }

  if (m == 0) {
    return;
  }

  const auto launch = [&]<typename T, typename W>() {
    const T* x = input.data<T>();
    const W* w = weight.data<W>();
    const T* b = bias ? bias->data<T>() : nullptr;
    T* y = output.data<T>();

    if (m == 1) {
      linear_gemv_kernel<<<ceil_div(n, 4), 128, 0, stream>>>(x, w, y, b, n, k);
    } else if (m <= 8) {
      launch_gemm<16, 4, 2, 4>(x, w, y, b, m, n, k, stream);
    } else if (m < 128) {
      launch_gemm<16, 8, 4, 8>(x, w, y, b, m, n, k, stream);
    } else {
      launch_gemm<16, 16, 8, 8>(x, w, y, b, m, n, k, stream);
    }
  };
  if (input.dtype() == DataType::Float32
      && weight.dtype() == DataType::BFloat16) {
    launch.template operator()<float, __nv_bfloat16>();
  } else {
    detail::dispatch_float(input.dtype(), [&]<typename T>() {
      launch.template operator()<T, T>();
    });
  }

  check_cuda(cudaGetLastError(), "linear kernel launch");
}

} // namespace nanovllm::kernels::cuda
