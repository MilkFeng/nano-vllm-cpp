#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

#include "kernels/cuda/attn/attn_shape.cuh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda::attn {
namespace {
// 通用 prefill：每个 warp 处理一行 query，block 共用一个 KV tile
template <typename T, int Warps, int Keys>
__global__ void attn_simt_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_q,
    const std::int32_t* __restrict__ cu_k,
    const std::int32_t* __restrict__ table, T* __restrict__ output,
    AttentionShape shape, bool causal, float scale) {
  constexpr int Threads = Warps << 5;
  constexpr int ThreadHeadDim = MaxHeadDim / 32;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int batch = blockIdx.z;
  const int head = blockIdx.y;
  const int q_tile = blockIdx.x;

  const int q_base = q_tile * Warps;

  const int qs = cu_q[batch];
  const int qlen = cu_q[batch + 1] - qs;
  const int klen = cu_k[batch + 1] - cu_k[batch];
  const int prefix = klen - qlen;

  if (q_base >= qlen) {
    return;
  }

  const int q_pos = q_base + warp;
  const int HeadDim = shape.head_dim;
  const int page_tokens = shape.block_tokens;
  const int hgroup = shape.query_heads / shape.key_heads;
  const int kh = head / hgroup;
  const auto q_offset = ((qs + q_pos) * shape.query_heads + head) * HeadDim;

  float q[ThreadHeadDim] = {};
  float acc[ThreadHeadDim] = {};
#pragma unroll
  for (int c = 0; c < ThreadHeadDim; ++c) {
    const int d = lane + c * 32;
    if (q_pos < qlen && d < HeadDim) {
      q[c] = detail::to_float(query[q_offset + d]);
    }
  }

  extern __shared__ __align__(16) unsigned char memory[];
  float* s_k = reinterpret_cast<float*>(memory);
  float* s_v = s_k + Keys * HeadDim;

  float m = -CUDART_INF_F;
  float l = 0.0f;
  const int visible = causal ? min(klen, prefix + q_base + Warps) : klen;
  const int row_end = causal ? min(klen, prefix + q_pos + 1) : klen;

  for (int k_base = 0; k_base < visible; k_base += Keys) {
    const int key_count = min(Keys, visible - k_base);

    // 读取当前 tile；新 token 从输入读取，并由唯一的 query tile 写入 cache。
    for (int i = threadIdx.x; i < key_count * HeadDim; i += Threads) {
      const int pos = k_base + i / HeadDim;
      const int d = i % HeadDim;
      const int page = table[batch * shape.max_blocks + pos / page_tokens];
      const auto slot =
          static_cast<std::size_t>(page) * page_tokens + pos % page_tokens;
      const auto offset = (slot * shape.key_heads + kh) * HeadDim + d;

      T k, v;
      if (pos < prefix) {
        k = key_cache[offset];
        v = value_cache[offset];
      } else {
        const int new_pos = pos - prefix;
        const auto input_offset =
            ((qs + new_pos) * shape.key_heads + kh) * HeadDim + d;
        k = key[input_offset];
        v = value[input_offset];

        if (head % hgroup == 0 && new_pos / Warps == q_tile) {
          key_cache[offset] = k;
          value_cache[offset] = v;
        }
      }
      s_k[i] = detail::to_float(k);
      s_v[i] = detail::to_float(v);
    }
    __syncthreads();

    if (q_pos < qlen && k_base < row_end) {
      const int row_keys = min(key_count, row_end - k_base);
      float s = -CUDART_INF_F;
      for (int j = 0; j < row_keys; ++j) {
        float dot = 0.0f;
#pragma unroll
        for (int c = 0; c < ThreadHeadDim; ++c) {
          const int d = lane + c * 32;
          if (d < HeadDim) {
            dot += q[c] * s_k[j * HeadDim + d];
          }
        }
        dot = detail::warp_sum_reduce(dot);
        if (lane == j) {
          s = dot * scale;
        }
      }

      const float nm = fmaxf(m, detail::warp_max_reduce(s));
      const float alpha = expf(m - nm);
      const float p = expf(s - nm);
      l = l * alpha + detail::warp_sum_reduce(p);
      m = nm;

#pragma unroll
      for (int c = 0; c < ThreadHeadDim; ++c) {
        acc[c] *= alpha;
      }
      for (int j = 0; j < row_keys; ++j) {
        const float weight = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
        for (int c = 0; c < ThreadHeadDim; ++c) {
          const int d = lane + c * 32;
          if (d < HeadDim) {
            acc[c] += weight * s_v[j * HeadDim + d];
          }
        }
      }
    }
    __syncthreads();
  }

  if (q_pos < qlen) {
    const float inv_sum = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int c = 0; c < ThreadHeadDim; ++c) {
      const int d = lane + c * 32;
      if (d < HeadDim) {
        output[q_offset + d] = detail::from_float<T>(acc[c] * inv_sum);
      }
    }
  }
}

} // namespace

template <typename T, int Warps, int Keys>
void launch_attn_simt(ConstTensorView query, TensorView key_cache,
                      TensorView value_cache, ConstTensorView key,
                      ConstTensorView value, ConstTensorView cu_q,
                      ConstTensorView cu_k, ConstTensorView table,
                      TensorView output, AttentionShape shape, bool causal,
                      cudaStream_t stream) {
  static_assert(Warps > 0 && Warps <= 32);
  static_assert(Keys > 0 && Keys <= 32);

  constexpr int block = Warps << 5;
  const int bytes = 2 * Keys * shape.head_dim * sizeof(float);
  const dim3 grid((shape.max_seqlen_q + Warps - 1) / Warps, shape.query_heads,
                  shape.batch);
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));

  assert(shape.head_dim <= MaxHeadDim);

  attn_simt_kernel<T, Warps, Keys><<<grid, block, bytes, stream>>>(
      query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
      key.data<T>(), value.data<T>(), cu_q.data<std::int32_t>(),
      cu_k.data<std::int32_t>(), table.data<std::int32_t>(), output.data<T>(),
      shape, causal, scale);
}

} // namespace nanovllm::kernels::cuda::attn
