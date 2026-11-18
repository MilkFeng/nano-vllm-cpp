#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

#include "kernels/cuda/attn/attn_shape.cuh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda::attn {
namespace {

// 通用 decode：每个 block 处理一个 query head 的一段 KV，各 warp 分摊 token
template <typename T, int Warps, bool SplitKV>
__global__ void attn_simt_decode_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_q,
    const std::int32_t* __restrict__ cu_k,
    const std::int32_t* __restrict__ table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* __restrict__ partials, int splits,
    int split_tokens, float scale) {
  constexpr int Threads = Warps << 5;
  constexpr int ThreadHeadDim = MaxHeadDim / 32;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

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

  const int HeadDim = shape.head_dim;
  const int page_tokens = shape.block_tokens;
  const int hgroup = shape.query_heads / shape.key_heads;
  const int kh = head / hgroup;
  const auto q_offset = (qs * shape.query_heads + head) * HeadDim;

  const int visible = causal ? min(klen, prefix + 1) : klen;
  const int begin = SplitKV ? split * split_tokens : 0;
  const int end = SplitKV ? static_cast<int>(min(
                                static_cast<std::size_t>(visible),
                                static_cast<std::size_t>(begin) + split_tokens))
                          : visible;

  extern __shared__ __align__(16) unsigned char memory[];
  float* s_out = reinterpret_cast<float*>(memory);
  float* s_m = s_out + HeadDim;
  float* s_l = s_m + Warps;

  for (int d = threadIdx.x; d < HeadDim; d += Threads) {
    s_out[d] = 0.0f;
  }

  float q[ThreadHeadDim] = {};
  float acc[ThreadHeadDim] = {};
#pragma unroll
  for (int c = 0; c < ThreadHeadDim; ++c) {
    const int d = lane + c * 32;
    if (d < HeadDim) {
      q[c] = detail::to_float(query[q_offset + d]);
    }
  }

  float m = -CUDART_INF_F;
  float l = 0.0f;
  for (int pos = begin + warp; pos < end; pos += Warps) {
    const int page = table[batch * shape.max_blocks + pos / page_tokens];
    const auto slot =
        static_cast<std::size_t>(page) * page_tokens + pos % page_tokens;
    const auto offset = (slot * shape.key_heads + kh) * HeadDim;

    float v[ThreadHeadDim] = {};
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < ThreadHeadDim; ++c) {
      const int d = lane + c * 32;
      if (d < HeadDim) {
        T k, value_element;
        if (pos < prefix) {
          k = key_cache[offset + d];
          value_element = value_cache[offset + d];
        } else {
          const int new_pos = pos - prefix;
          const auto input_offset =
              ((qs + new_pos) * shape.key_heads + kh) * HeadDim + d;
          k = key[input_offset];
          value_element = value[input_offset];

          if (head % hgroup == 0 && new_pos == 0) {
            key_cache[offset + d] = k;
            value_cache[offset + d] = value_element;
          }
        }
        s += q[c] * detail::to_float(k);
        v[c] = detail::to_float(value_element);
      }
    }

    s = detail::warp_sum_reduce(s) * scale;
    const float nm = fmaxf(m, s);
    const float alpha = __expf(m - nm);
    const float p = __expf(s - nm);
    l = l * alpha + p;
    m = nm;
#pragma unroll
    for (int c = 0; c < ThreadHeadDim; ++c) {
      acc[c] = acc[c] * alpha + p * v[c];
    }
  }

  if (lane == 0) {
    s_m[warp] = m;
    s_l[warp] = l;
  }
  __syncthreads();

  // 先合并 block 内各 warp 的最大值和分母，再缩放各自的输出分子。
  float gm = -CUDART_INF_F;
  float gl = 0.0f;
#pragma unroll
  for (int w = 0; w < Warps; ++w) {
    gm = fmaxf(gm, s_m[w]);
  }
#pragma unroll
  for (int w = 0; w < Warps; ++w) {
    if (s_l[w] > 0.0f) {
      gl += s_l[w] * expf(s_m[w] - gm);
    }
  }

  const float weight = l > 0.0f ? expf(m - gm) : 0.0f;
#pragma unroll
  for (int c = 0; c < ThreadHeadDim; ++c) {
    const int d = lane + c * 32;
    if (d < HeadDim) {
      atomicAdd(s_out + d, acc[c] * weight);
    }
  }
  __syncthreads();

  if constexpr (SplitKV) {
    const auto split_row = (batch * shape.query_heads + head) * splits + split;
    const auto rows = shape.batch * shape.query_heads * splits;
    for (int d = threadIdx.x; d < HeadDim; d += Threads) {
      partials[split_row * HeadDim + d] = s_out[d];
    }
    if (threadIdx.x == 0) {
      // 空分片也覆盖旧 workspace，写入 (acc=0, m=-inf, l=0)。
      partials[rows * HeadDim + split_row] = gm;
      partials[rows * (HeadDim + 1) + split_row] = gl;
    }
  } else {
    const float inv_sum = gl > 0.0f ? 1.0f / gl : 0.0f;
    for (int d = threadIdx.x; d < HeadDim; d += Threads) {
      output[q_offset + d] = detail::from_float<T>(s_out[d] * inv_sum);
    }
  }
}

// Decode 的中间结果按 [batch, query head, split] 排列。
// 与 tensor 路径按实际 query token 排列的中间结果不同，空请求也占行。
template <typename T>
__global__ void merge_attn_simt_decode_splits_kernel(
    const float* __restrict__ partials, const std::int32_t* __restrict__ cu_q,
    T* __restrict__ output, AttentionShape shape, int splits) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int qs = cu_q[batch];
  const int qlen = cu_q[batch + 1] - qs;

  if (qlen == 0) {
    return;
  }

  const auto rows = shape.batch * shape.query_heads * splits;
  const auto row_begin = (batch * shape.query_heads + head) * splits;
  const float* maxima = partials + rows * shape.head_dim;
  const float* sums = maxima + rows;

  __shared__ float weights[32], inv_sum;

  if (threadIdx.x < 32) {
    const int split = threadIdx.x;
    const float m = split < splits ? maxima[row_begin + split] : -CUDART_INF_F;
    const float l = split < splits ? sums[row_begin + split] : 0.0f;
    const float gm = detail::warp_max_reduce(m);
    const float weight = l > 0.0f ? __expf(m - gm) : 0.0f;
    const float gl = detail::warp_sum_reduce(l * weight);

    if (split < splits) {
      weights[split] = weight;
    }
    if (split == 0) {
      inv_sum = gl > 0.0f ? 1.0f / gl : 0.0f;
    }
  }

  __syncthreads();

  const auto q_offset = (qs * shape.query_heads + head) * shape.head_dim;
  for (int d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int split = 0; split < splits; ++split) {
      acc +=
          partials[(row_begin + split) * shape.head_dim + d] * weights[split];
    }
    output[q_offset + d] = detail::from_float<T>(acc * inv_sum);
  }
}

} // namespace

template <typename T, int Warps, bool SplitKV>
void launch_attn_simt_decode(ConstTensorView query, TensorView key_cache,
                             TensorView value_cache, ConstTensorView key,
                             ConstTensorView value, ConstTensorView cu_q,
                             ConstTensorView cu_k, ConstTensorView table,
                             TensorView output, AttentionShape shape,
                             bool causal, cudaStream_t stream, float* partials,
                             int splits) {
  static_assert(Warps > 0 && Warps <= 32);

  constexpr int block = Warps << 5;
  const int bytes = (shape.head_dim + 2 * Warps) * sizeof(float);
  const dim3 grid(splits, shape.query_heads, shape.batch);
  const int split_tokens = (shape.max_seqlen_k + splits - 1) / splits;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));

  assert(shape.head_dim <= MaxHeadDim);

  attn_simt_decode_kernel<T, Warps, SplitKV><<<grid, block, bytes, stream>>>(
      query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
      key.data<T>(), value.data<T>(), cu_q.data<std::int32_t>(),
      cu_k.data<std::int32_t>(), table.data<std::int32_t>(), output.data<T>(),
      shape, causal, partials, splits, split_tokens, scale);

  if constexpr (SplitKV) {
    const dim3 merge_grid(shape.query_heads, shape.batch);
    merge_attn_simt_decode_splits_kernel<T><<<merge_grid, 128, 0, stream>>>(
        partials, cu_q.data<std::int32_t>(), output.data<T>(), shape, splits);
  }
}

} // namespace nanovllm::kernels::cuda::attn
