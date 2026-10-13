
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cuda/std/limits>
#include <type_traits>

namespace nanovllm::kernels::cuda::detail {

template <typename T>
struct SumOp {
  static __device__ __forceinline__ T identity() { return T(0); }
  static __device__ __forceinline__ T apply(T a, T b) { return a + b; }
};

template <typename T>
struct MaxOp {
  static __device__ __forceinline__ T identity() {
    if constexpr (std::is_same_v<T, float>) {
      return -CUDART_INF_F;
    } else if constexpr (std::is_same_v<T, double>) {
      return -CUDART_INF;
    } else if constexpr (std::is_same_v<T, __half>
                         || std::is_same_v<T, __nv_bfloat16>) {
      return T(-CUDART_INF_F);
    } else {
      return ::cuda::std::numeric_limits<T>::lowest();
    }
  }

  static __device__ __forceinline__ T apply(T a, T b) {
    if constexpr (std::is_same_v<T, float>) {
      return fmaxf(a, b);
    } else if constexpr (std::is_same_v<T, double>) {
      return fmax(a, b);
    } else if constexpr (std::is_same_v<T, __half>
                         || std::is_same_v<T, __nv_bfloat16>) {
      return __hmax(a, b);
    } else {
      return max(a, b);
    }
  }
};

struct ArgMaxResult {
  float value;
  int index;
};

struct ArgMaxOp {
  static __device__ __forceinline__ ArgMaxResult identity() {
    return {-CUDART_INF_F, ::cuda::std::numeric_limits<int>::max()};
  }

  static __device__ __forceinline__ ArgMaxResult apply(ArgMaxResult a,
                                                       ArgMaxResult b) {
    if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
      return b;
    }
    return a;
  }
};

template <bool Xor, typename T>
__device__ __forceinline__ T warp_shuffle(T value, int offset) {
  if constexpr (Xor) {
    return __shfl_xor_sync(0xffffffffu, value, offset);
  } else {
    return __shfl_down_sync(0xffffffffu, value, offset);
  }
}

template <bool Xor>
__device__ __forceinline__ ArgMaxResult warp_shuffle(ArgMaxResult value,
                                                     int offset) {
  return {
      warp_shuffle<Xor>(value.value, offset),
      warp_shuffle<Xor>(value.index, offset),
  };
}

template <typename T, typename Op, bool Broadcast = true>
__device__ __forceinline__ T warp_reduce(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const T other = warp_shuffle<Broadcast>(value, offset);
    value = Op::apply(value, other);
  }

  return value;
}

template <typename T, int Threads, typename Op, bool Broadcast = true>
__device__ __forceinline__ T block_reduce(T value) {
  static_assert(Threads >= 32 && Threads <= 1024);
  static_assert(Threads % 32 == 0);

  if constexpr (Threads == 32) {
    return warp_reduce<T, Op, Broadcast>(value);
  } else {
    constexpr int num_warps = Threads >> 5;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ T warp_values[num_warps];

    value = warp_reduce<T, Op, false>(value);

    if (lane == 0) {
      warp_values[warp] = value;
    }
    __syncthreads();

    T result = Op::identity();

    if (warp == 0) {
      if (lane < num_warps) {
        result = warp_values[lane];
      }

      result = warp_reduce<T, Op, false>(result);
    }

    if constexpr (Broadcast) {
      if (threadIdx.x == 0) {
        warp_values[0] = result;
      }
      __syncthreads();
      T output = warp_values[0];
      __syncthreads();

      return output;
    } else {
      __syncthreads();
      return result;
    }
  }
}

template <typename T, bool Broadcast = true>
__device__ __forceinline__ T warp_sum_reduce(T value) {
  return warp_reduce<T, SumOp<T>, Broadcast>(value);
}

template <typename T, bool Broadcast = true>
__device__ __forceinline__ T warp_max_reduce(T value) {
  return warp_reduce<T, MaxOp<T>, Broadcast>(value);
}

template <typename T, int Threads, bool Broadcast = true>
__device__ __forceinline__ T block_sum_reduce(T value) {
  return block_reduce<T, Threads, SumOp<T>, Broadcast>(value);
}

template <typename T, int Threads, bool Broadcast = true>
__device__ __forceinline__ T block_max_reduce(T value) {
  return block_reduce<T, Threads, MaxOp<T>, Broadcast>(value);
}

template <int Threads, bool Broadcast = true>
__device__ __forceinline__ ArgMaxResult
block_argmax_reduce(ArgMaxResult value) {
  return block_reduce<ArgMaxResult, Threads, ArgMaxOp, Broadcast>(value);
}

} // namespace nanovllm::kernels::cuda::detail
