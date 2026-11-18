#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

#include "kernels/cuda/attn/attn_shape.cuh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda::attn {
namespace {

template <typename T>
__device__ __forceinline__ void attn_load8(const T* src, float* dst) {
  const uint4 bits = *reinterpret_cast<const uint4*>(src);
  const T* data = reinterpret_cast<const T*>(&bits);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    dst[j] = detail::to_float(data[j]);
  }
}

template <typename T, int HeadDim, int Warps, bool SplitKV>
__global__ void attn_vector_decode_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_q,
    const std::int32_t* __restrict__ cu_k,
    const std::int32_t* __restrict__ table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* __restrict__ partials, int splits,
    int split_tokens, float scale) {
  constexpr int Threads = Warps << 5;

  constexpr int GroupSize = 4;
  constexpr int Groups = Threads / GroupSize;
  constexpr int ThreadHeadDim = HeadDim / GroupSize;

  const int lane = threadIdx.x & 31;
  const int group_lane = threadIdx.x % GroupSize;
  const int group = threadIdx.x / GroupSize;

  const unsigned group_mask = ((1u << GroupSize) - 1)
                              << (lane / GroupSize * GroupSize);

  const int batch = blockIdx.z;
  const int head = blockIdx.y;
  const int split = blockIdx.x;

  const int qs = cu_q[batch];
  const int qlen = cu_q[batch + 1] - qs;
  const int klen = cu_k[batch + 1] - cu_k[batch];
  const int prefix = klen - qlen;

  if (qlen == 0) {
    return;
  }

  const int page_tokens = shape.block_tokens;
  const int hgroup = shape.query_heads / shape.key_heads;
  const int kh = head / hgroup;
  const int q_offset = (qs * shape.query_heads + head) * HeadDim;

  float q[ThreadHeadDim];

#pragma unroll
  for (int c = 0; c < ThreadHeadDim / 8; ++c) {
    const int d = (c * GroupSize + group_lane) * 8;
    attn_load8(query + q_offset + d, q + c * 8);
  }

  const int begin = SplitKV ? split * split_tokens : 0;
  const int end = SplitKV ? min(klen, begin + split_tokens) : klen;

  float acc[ThreadHeadDim] = {};

  float m = -CUDART_INF_F;
  float l = 0.0f;

  for (int pos = begin + group; pos < end; pos += Groups) {
    const int page = table[batch * shape.max_blocks + pos / page_tokens];
    const int slot =
        static_cast<std::size_t>(page) * page_tokens + pos % page_tokens;
    const int offset = (slot * shape.key_heads + kh) * HeadDim;

    const int new_pos = pos - prefix;
    const int input_offset = ((qs + new_pos) * shape.key_heads + kh) * HeadDim;

    float k[ThreadHeadDim], v[ThreadHeadDim];
#pragma unroll
    for (int c = 0; c < ThreadHeadDim / 8; ++c) {
      const int d = (c * GroupSize + group_lane) * 8;

      if (pos < prefix) {
        attn_load8(key_cache + offset + d, k + c * 8);
        attn_load8(value_cache + offset + d, v + c * 8);
      } else {
        attn_load8(key + input_offset + d, k + c * 8);
        attn_load8(value + input_offset + d, v + c * 8);

        if (head % hgroup == 0) {
          *reinterpret_cast<uint4*>(key_cache + offset + d) =
              *reinterpret_cast<const uint4*>(key + input_offset + d);
          *reinterpret_cast<uint4*>(value_cache + offset + d) =
              *reinterpret_cast<const uint4*>(value + input_offset + d);
        }
      }
    }

    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < ThreadHeadDim; ++c) {
      s += q[c] * k[c];
    }

#pragma unroll
    for (int mask = GroupSize / 2; mask; mask /= 2) {
      s += __shfl_xor_sync(group_mask, s, mask, GroupSize);
    }
    s *= scale;

    const float nm = fmaxf(m, s);
    const float alpha = __expf(m - nm);
    const float p = __expf(s - nm);
    l = l * alpha + p;
    m = nm;

#pragma unroll
    for (int c = 0; c < ThreadHeadDim; ++c) {
      acc[c] = acc[c] * alpha + v[c] * p;
    }
  }

  __shared__ float s_m[Groups];
  __shared__ float s_l[Groups];
  __shared__ float s_out[Groups][HeadDim];
  __shared__ float global_m, global_l;

  if (group_lane == 0) {
    s_m[group] = m;
    s_l[group] = l;
  }

  __syncthreads();

  if (threadIdx.x < 32) {
    const float group_m = lane < Groups ? s_m[lane] : -CUDART_INF_F;
    const float group_l = lane < Groups ? s_l[lane] : 0.0f;

    const float gm = detail::warp_max_reduce(group_m);
    const float gl = detail::warp_sum_reduce(
        group_l > 0.0f ? group_l * __expf(group_m - gm) : 0.0f);

    if (lane == 0) {
      global_m = gm;
      global_l = gl;
    }
  }

  __syncthreads();

  const float gm = global_m, gl = global_l;
  const float weight = l > 0.0f ? __expf(m - gm) : 0.0f;

#pragma unroll
  for (int c = 0; c < ThreadHeadDim; ++c) {
    const int d = (c / 8 * GroupSize + group_lane) * 8 + c % 8;
    s_out[group][d] = acc[c] * weight;
  }

  __syncthreads();

  const int split_row = (batch * shape.query_heads + head) * splits + split;
  const int rows = shape.batch * shape.query_heads * splits;

  for (int d = threadIdx.x; d < HeadDim; d += Threads) {
    float acc = 0.f;
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
      acc += s_out[g][d];
    }
    if constexpr (SplitKV) {
      partials[split_row * HeadDim + d] = acc;
    } else {
      output[q_offset + d] = detail::from_float<T>(gl > 0.f ? acc / gl : 0.f);
    }
  }

  if constexpr (SplitKV) {
    if (threadIdx.x == 0) {
      partials[rows * HeadDim + split_row] = gm;
      partials[rows * (HeadDim + 1) + split_row] = gl;
    }
  }
}

template <typename T>
__global__ void merge_decode_splits_kernel(
    const float* __restrict__ partials,
    const std::int32_t* __restrict__ cu_seqlens_q, T* __restrict__ output,
    AttentionShape shape, int num_splits) {
  const auto head = blockIdx.x;
  const auto batch = blockIdx.y;

  const int query_start = cu_seqlens_q[batch];

  if (cu_seqlens_q[batch + 1] == query_start) {
    return;
  }

  const auto rows = shape.batch * shape.query_heads * num_splits;
  const auto row_begin = (batch * shape.query_heads + head) * num_splits;
  const float* maxima = partials + rows * shape.head_dim;
  const float* sums = maxima + rows;

  __shared__ float weights[32], inv_sum;

  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;

    const float m =
        lane < num_splits ? maxima[row_begin + lane] : -CUDART_INF_F;
    const float l = lane < num_splits ? sums[row_begin + lane] : 0.0f;

    const float global_m = detail::warp_max_reduce(m);
    const float weight = l > 0.0f ? __expf(m - global_m) : 0.0f;
    const float global_l = detail::warp_sum_reduce(l * weight);

    if (lane < num_splits) {
      weights[lane] = weight;
    }

    if (lane == 0) {
      inv_sum = global_l > 0.0f ? 1.0f / global_l : 0.0f;
    }
  }

  __syncthreads();

  const auto offset = (query_start * shape.query_heads + head) * shape.head_dim;

  for (int d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      acc += partials[(row_begin + s) * shape.head_dim + d] * weights[s];
    }
    output[offset + d] = detail::from_float<T>(acc * inv_sum);
  }
}

} // namespace

template <typename T, int HeadDim, int Warps, bool SplitKV>
void launch_attn_vector_decode(ConstTensorView query, TensorView key_cache,
                               TensorView value_cache, ConstTensorView key,
                               ConstTensorView value, ConstTensorView cu_q,
                               ConstTensorView cu_k, ConstTensorView table,
                               TensorView output, AttentionShape shape,
                               bool causal, cudaStream_t stream,
                               float* partials, int splits) {
  constexpr int block = Warps << 5;
  const dim3 grid(splits, shape.query_heads, shape.batch);

  static_assert(HeadDim == 128);
  static_assert(block == 128);

  const int tokens = (shape.max_seqlen_k + splits - 1) / splits;
  const float scale = rsqrtf(HeadDim);

  attn_vector_decode_kernel<T, HeadDim, Warps, SplitKV>
      <<<grid, block, 0, stream>>>(
          query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
          key.data<T>(), value.data<T>(), cu_q.data<std::int32_t>(),
          cu_k.data<std::int32_t>(), table.data<std::int32_t>(),
          output.data<T>(), shape, causal, partials, splits, tokens, scale);

  if constexpr (SplitKV) {
    const dim3 grid(shape.query_heads, shape.batch);
    constexpr int block = 128;

    merge_decode_splits_kernel<T><<<grid, block, 0, stream>>>(
        partials, cu_q.data<std::int32_t>(), output.data<T>(), shape, splits);
  }
}

} // namespace nanovllm::kernels::cuda::attn
