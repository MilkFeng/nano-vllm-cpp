#include <math_constants.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {

namespace {

constexpr int kQueryTile = 4;
constexpr int kKeyTile = 16;
constexpr int kThreads = kQueryTile * 32;
constexpr int kHeadElements = 256 / 32;

struct AttentionShape {
  std::size_t batch;
  std::size_t query_tokens;
  std::size_t max_seqlen_q, max_seqlen_k;
  std::size_t query_heads, key_heads;
  std::size_t head_dim;
  std::size_t blocks, block_tokens;
  std::size_t max_blocks;
};

AttentionShape attn_shape(ConstTensorView query, TensorView key_cache,
                          TensorView value_cache, ConstTensorView key,
                          ConstTensorView value, ConstTensorView cu_seqlens_q,
                          ConstTensorView cu_seqlens_k,
                          std::size_t max_seqlen_q, std::size_t max_seqlen_k,
                          ConstTensorView block_table, TensorView output) {
  detail::assert_tensor(query);
  detail::assert_tensor(key_cache);
  detail::assert_tensor(value_cache);
  detail::assert_tensor(key);
  detail::assert_tensor(value);
  detail::assert_tensor(cu_seqlens_q);
  detail::assert_tensor(cu_seqlens_k);
  detail::assert_tensor(block_table);
  detail::assert_tensor(output);

  assert(same_dtype(query, key_cache, value_cache, key, value, output));
  assert(same_device(query, key_cache, value_cache, key, value, cu_seqlens_q,
                     cu_seqlens_k, block_table, output));
  assert(query.dtype() == DataType::Float32
         || query.dtype() == DataType::Float16
         || query.dtype() == DataType::BFloat16);

  AttentionShape shape = {};

  assert(same_shape(output, query));
  assert(query.rank() == 3);

  shape.query_tokens = query.shape()[0];
  shape.query_heads = query.shape()[1];
  shape.head_dim = query.shape()[2];
  assert(shape.query_heads > 0 && shape.head_dim > 0 && shape.head_dim <= 256);

  assert(same_shape(key_cache, value_cache));
  assert(key_cache.rank() == 4);

  shape.blocks = key_cache.shape()[0];
  shape.block_tokens = key_cache.shape()[1];
  shape.key_heads = key_cache.shape()[2];

  assert(shape.block_tokens > 0);
  assert(shape.key_heads > 0 && shape.query_heads % shape.key_heads == 0);
  assert(key_cache.shape()[3] == shape.head_dim);

  assert(same_shape(key, value));
  assert(key.rank() == 3);
  assert(key.shape()[0] == shape.query_tokens);
  assert(key.shape()[1] == shape.key_heads && key.shape()[2] == shape.head_dim);

  assert(same_shape(cu_seqlens_q, cu_seqlens_k));
  assert(same_dtype(cu_seqlens_q, cu_seqlens_k));
  assert(cu_seqlens_q.dtype() == DataType::Int32);
  assert(cu_seqlens_q.rank() == 1);
  assert(cu_seqlens_q.shape()[0] >= 1);

  shape.batch = cu_seqlens_q.shape()[0] - 1;

  shape.max_seqlen_q = max_seqlen_q;
  shape.max_seqlen_k = max_seqlen_k;

  assert(shape.batch <= INT_MAX && shape.query_tokens <= INT_MAX);
  assert(max_seqlen_q <= INT_MAX && max_seqlen_k <= INT_MAX);
  assert(shape.query_tokens == 0 || (shape.batch > 0 && max_seqlen_q > 0));
  assert(shape.query_tokens <= shape.batch * max_seqlen_q);

  assert(block_table.dtype() == DataType::Int32);
  assert(block_table.rank() == 2);
  assert(block_table.shape()[0] == shape.batch);

  shape.max_blocks = block_table.shape()[1];

  assert(max_seqlen_k == 0 || shape.blocks > 0);
  assert(max_seqlen_k <= shape.max_blocks * shape.block_tokens);

  return shape;
}

template <typename T>
__global__ void flash_attn_with_kvcache_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_seqlens_q,
    const std::int32_t* __restrict__ cu_seqlens_k,
    const std::int32_t* __restrict__ block_table, T* __restrict__ output,
    AttentionShape shape, bool causal) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int query_begin = blockIdx.x * kQueryTile;
  const std::size_t query_head = blockIdx.y;
  const std::size_t batch = blockIdx.z;

  const int query_start = cu_seqlens_q[batch];
  const int query_length = cu_seqlens_q[batch + 1] - query_start;
  const int key_length = cu_seqlens_k[batch + 1] - cu_seqlens_k[batch];
  const int cached_length = key_length - query_length;

  if (query_begin >= query_length) {
    return;
  }

  const int query_pos = query_begin + warp;
  const int head_dim = shape.head_dim;
  const auto group_size = shape.query_heads / shape.key_heads;
  const auto key_head = query_head / group_size;
  const auto query_offset =
      ((query_start + query_pos) * shape.query_heads + query_head) * head_dim;
  const float scale = rsqrtf(head_dim);

  float q[kHeadElements] = {};
  float acc[kHeadElements] = {};

#pragma unroll
  for (int i = 0; i < kHeadElements; ++i) {
    const int d = lane + i * 32;
    if (query_pos < query_length && d < head_dim) {
      q[i] = detail::to_float(query[query_offset + d]);
    }
  }

  extern __shared__ float shared[];
  float* tile_k = shared;
  float* tile_v = shared + kKeyTile * head_dim;

  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;

  int key_end = key_length;
  int row_end = key_length;
  if (causal) {
    key_end =
        min(key_length, key_length - query_length + query_begin + kQueryTile);
    row_end = min(key_length, key_length - query_length + query_pos + 1);
  }

  for (int key_begin = 0; key_begin < key_end; key_begin += kKeyTile) {
    const int tile_size = min(kKeyTile, key_end - key_begin);

    for (int i = threadIdx.x; i < tile_size * head_dim; i += kThreads) {
      const int key_pos = key_begin + i / head_dim;
      const int d = i % head_dim;
      const auto page =
          block_table[batch * shape.max_blocks + key_pos / shape.block_tokens];
      const auto cache_token =
          page * shape.block_tokens + key_pos % shape.block_tokens;
      const auto offset =
          (cache_token * shape.key_heads + key_head) * head_dim + d;

      T k;
      T v;
      if (key_pos < cached_length) {
        k = key_cache[offset];
        v = value_cache[offset];
      } else {
        const int new_pos = key_pos - cached_length;
        const auto input_offset =
            ((query_start + new_pos) * shape.key_heads + key_head) * head_dim
            + d;
        k = key[input_offset];
        v = value[input_offset];

        if (query_head % group_size == 0
            && new_pos / kQueryTile == blockIdx.x) {
          key_cache[offset] = k;
          value_cache[offset] = v;
        }
      }

      tile_k[i] = detail::to_float(k);
      tile_v[i] = detail::to_float(v);
    }
    __syncthreads();

    if (query_pos < query_length && key_begin < row_end) {
      const int key_count = min(tile_size, row_end - key_begin);
      float score = -CUDART_INF_F;

      for (int j = 0; j < key_count; ++j) {
        float dot = 0.0f;

#pragma unroll
        for (int i = 0; i < kHeadElements; ++i) {
          const int d = lane + i * 32;
          if (d < head_dim) {
            dot += q[i] * tile_k[j * head_dim + d];
          }
        }

        dot = detail::warp_sum_reduce(dot);
        if (lane == j) {
          score = dot * scale;
        }
      }

      const float tile_max = detail::warp_max_reduce(score);
      const float next_max = fmaxf(row_max, tile_max);
      const float rescale = expf(row_max - next_max);
      const float probability = expf(score - next_max);
      const float tile_sum = detail::warp_sum_reduce(probability);

      row_sum = row_sum * rescale + tile_sum;
      row_max = next_max;

#pragma unroll
      for (int i = 0; i < kHeadElements; ++i) {
        acc[i] *= rescale;
      }

      for (int j = 0; j < key_count; ++j) {
        const float weight = __shfl_sync(0xffffffffu, probability, j);

#pragma unroll
        for (int i = 0; i < kHeadElements; ++i) {
          const int d = lane + i * 32;
          if (d < head_dim) {
            acc[i] += weight * tile_v[j * head_dim + d];
          }
        }
      }
    }
    __syncthreads();
  }

  if (query_pos < query_length) {
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

#pragma unroll
    for (int i = 0; i < kHeadElements; ++i) {
      const int d = lane + i * 32;
      if (d < head_dim) {
        output[query_offset + d] = detail::from_float<T>(acc[i] * inv_sum);
      }
    }
  }
}

} // namespace

void flash_attn_with_kvcache(ConstTensorView query, TensorView key_cache,
                             TensorView value_cache, ConstTensorView key,
                             ConstTensorView value,
                             ConstTensorView cu_seqlens_q,
                             ConstTensorView cu_seqlens_k,
                             std::size_t max_seqlen_q, std::size_t max_seqlen_k,
                             ConstTensorView block_table, TensorView output,
                             bool causal, cudaStream_t stream) {
  const auto shape =
      attn_shape(query, key_cache, value_cache, key, value, cu_seqlens_q,
                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table, output);

  if (shape.query_tokens == 0) {
    return;
  }

  const dim3 grid((max_seqlen_q + kQueryTile - 1) / kQueryTile,
                  shape.query_heads, shape.batch);
  const auto shared_bytes = 2 * kKeyTile * shape.head_dim * sizeof(float);

  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    flash_attn_with_kvcache_kernel<T><<<grid, kThreads, shared_bytes, stream>>>(
        query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
        key.data<T>(), value.data<T>(), cu_seqlens_q.data<std::int32_t>(),
        cu_seqlens_k.data<std::int32_t>(), block_table.data<std::int32_t>(),
        output.data<T>(), shape, causal);
  });
  check_cuda(cudaGetLastError(), "launch flash_attn_with_kvcache");
}

} // namespace nanovllm::kernels::cuda
