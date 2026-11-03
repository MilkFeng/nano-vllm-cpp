#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>

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
constexpr int kMaxDecodeSplits = 32;
constexpr std::size_t kDecodeTargetBlocks = 256;
constexpr std::size_t kMinSplitTokens = 256;

int decode_splits(std::size_t batch, std::size_t heads,
                  std::size_t max_seqlen_k) {
  if (batch == 0 || heads == 0 || max_seqlen_k < 2 * kMinSplitTokens
      || heads >= kDecodeTargetBlocks
      || batch >= (kDecodeTargetBlocks + heads - 1) / heads) {
    return 1;
  }
  const auto blocks = batch * heads;
  const auto wanted = (kDecodeTargetBlocks + blocks - 1) / blocks;
  int splits = 1;
  while (splits < wanted && splits < kMaxDecodeSplits
         && 2 * splits <= max_seqlen_k / kMinSplitTokens) {
    splits *= 2;
  }
  return splits;
}

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

template <typename T, int W, int E, bool SplitKV>
__global__ void flash_attn_with_kvcache_kernel_decode(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_seqlens_q,
    const std::int32_t* __restrict__ cu_seqlens_k,
    const std::int32_t* __restrict__ block_table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* __restrict__ partials,
    int num_splits, int split_tokens) {
  constexpr int kThreads = W * 32;

  constexpr int query_local_idx = 0;
  const std::size_t query_head_idx = blockIdx.y;
  const std::size_t batch = blockIdx.z;

  const int query_start = cu_seqlens_q[batch];
  const int query_length = cu_seqlens_q[batch + 1] - query_start;
  const int key_length = cu_seqlens_k[batch + 1] - cu_seqlens_k[batch];
  const int cached_length = key_length - query_length;

  if (query_local_idx >= query_length) {
    return;
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int head_dim = shape.head_dim;
  const int query_idx = query_start + query_local_idx;
  const std::size_t group_size = shape.query_heads / shape.key_heads;
  const std::size_t key_head = query_head_idx / group_size;

  const std::size_t query_offset =
      (query_idx * shape.query_heads + query_head_idx) * head_dim;
  const int key_end =
      causal ? min(key_length,
                   static_cast<int>(cached_length + query_local_idx + 1))
             : key_length;
  const float scale = rsqrtf(head_dim);

  extern __shared__ float s_mem[];
  float* s_out = s_mem;
  float* s_m = s_out + head_dim;
  float* s_l = s_m + W;

  for (int k = threadIdx.x; k < head_dim; k += kThreads) {
    s_out[k] = 0.0f;
  }

  float q[E] = {};
  float acc[E] = {};
#pragma unroll
  for (int c = 0; c < E; ++c) {
    const int d = lane + c * 32;
    if (d < head_dim) {
      q[c] = detail::to_float(query[query_offset + d]);
    }
  }

  float m = -CUDART_INF_F;
  float l = 0.0f;

  const int key_begin = SplitKV ? blockIdx.x * split_tokens : 0;
  const int partition_end =
      SplitKV ? static_cast<int>(
                    min(static_cast<std::size_t>(key_end),
                        static_cast<std::size_t>(key_begin) + split_tokens))
              : key_end;
  for (int j = key_begin + warp; j < partition_end; j += W) {
    const auto page =
        block_table[batch * shape.max_blocks + j / shape.block_tokens];
    const auto cache_token = page * shape.block_tokens + j % shape.block_tokens;
    const auto cache_offset =
        (cache_token * shape.key_heads + key_head) * head_dim;
    float v[E] = {};
    float dot = 0.0f;

#pragma unroll
    for (int c = 0; c < E; ++c) {
      const int d = lane + c * 32;
      if (d < head_dim) {
        T kj, vj;
        if (j < cached_length) {
          kj = key_cache[cache_offset + d];
          vj = value_cache[cache_offset + d];
        } else {
          const int new_pos = j - cached_length;
          const auto input_offset =
              ((query_start + new_pos) * shape.key_heads + key_head) * head_dim
              + d;

          kj = key[input_offset];
          vj = value[input_offset];

          if (query_head_idx % group_size == 0 && new_pos == query_local_idx) {
            key_cache[cache_offset + d] = kj;
            value_cache[cache_offset + d] = vj;
          }
        }
        dot += q[c] * detail::to_float(kj);
        v[c] = detail::to_float(vj);
      }
    }

    const float score = detail::warp_sum_reduce(dot) * scale;
    const float next_m = fmaxf(m, score);
    const float alpha = __expf(m - next_m);
    const float beta = __expf(score - next_m);
    l = l * alpha + beta;
    m = next_m;
#pragma unroll
    for (int c = 0; c < E; ++c) {
      acc[c] = acc[c] * alpha + beta * v[c];
    }
  }

  if (lane == 0) {
    s_m[warp] = m;
    s_l[warp] = l;
  }
  __syncthreads();

  float g_m = -CUDART_INF_F;
  float g_l = 0.0f;
#pragma unroll
  for (int w = 0; w < W; ++w) {
    g_m = fmaxf(g_m, s_m[w]);
  }
#pragma unroll
  for (int w = 0; w < W; ++w) {
    if (s_l[w] > 0.0f) {
      g_l += s_l[w] * expf(s_m[w] - g_m);
    }
  }

  const float warp_scale = l > 0.0f ? expf(m - g_m) : 0.0f;
#pragma unroll
  for (int c = 0; c < E; ++c) {
    const int d = lane + c * 32;
    if (d < head_dim) {
      atomicAdd(&s_out[d], acc[c] * warp_scale);
    }
  }
  __syncthreads();

  if constexpr (SplitKV) {
    const auto row =
        (batch * shape.query_heads + query_head_idx) * num_splits + blockIdx.x;
    const auto rows = shape.batch * shape.query_heads * num_splits;
    for (int d = threadIdx.x; d < head_dim; d += kThreads) {
      partials[row * head_dim + d] = s_out[d];
    }
    if (threadIdx.x == 0) {
      // Empty partitions explicitly overwrite stale workspace with (0,-inf,0).
      partials[rows * head_dim + row] = g_m;
      partials[rows * (head_dim + 1) + row] = g_l;
    }
  } else {
    const float inv_l = g_l > 0.0f ? 1.0f / g_l : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += kThreads) {
      output[query_offset + d] = detail::from_float<T>(s_out[d] * inv_l);
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
  __shared__ float weights[kMaxDecodeSplits];
  __shared__ float inv_sum;

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

std::size_t flash_attn_decode_workspace_elements(std::size_t max_batch,
                                                 std::size_t query_heads,
                                                 std::size_t head_dim,
                                                 std::size_t max_seqlen_k) {
  if (query_heads == 0 || head_dim == 0 || head_dim > 256
      || max_seqlen_k > INT_MAX) {
    throw std::invalid_argument("invalid decode workspace dimensions");
  }
  // The split count falls as batch grows. Reserving only for max_batch could
  // underallocate for a smaller batch; bound all potentially split batches.
  const auto limit = std::min(max_batch, kDecodeTargetBlocks / query_heads);
  std::size_t elements = 0;
  for (std::size_t batch = 1; batch <= limit; ++batch) {
    const int splits = decode_splits(batch, query_heads, max_seqlen_k);
    if (splits > 1) {
      elements =
          std::max(elements, batch * query_heads * splits * (head_dim + 2));
    }
  }
  return elements;
}

void flash_attn_with_kvcache(
    ConstTensorView query, TensorView key_cache, TensorView value_cache,
    ConstTensorView key, ConstTensorView value, ConstTensorView cu_seqlens_q,
    ConstTensorView cu_seqlens_k, std::size_t max_seqlen_q,
    std::size_t max_seqlen_k, ConstTensorView block_table, TensorView output,
    bool causal, cudaStream_t stream, TensorView workspace) {
  const auto shape =
      attn_shape(query, key_cache, value_cache, key, value, cu_seqlens_q,
                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table, output);

  if (shape.query_tokens == 0) {
    return;
  }

  const int splits =
      max_seqlen_q == 1 && workspace.data() != nullptr
          ? decode_splits(shape.batch, shape.query_heads, max_seqlen_k)
          : 1;
  if (splits > 1
      && (workspace.dtype() != DataType::Float32
          || !same_device(query, workspace) || !workspace.is_contiguous()
          || workspace.numel() < shape.batch * shape.query_heads * splits
                                     * (shape.head_dim + 2))) {
    throw std::invalid_argument(
        "Split-KV requires a sufficiently large contiguous FP32 workspace "
        "on the query device");
  }

  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    if (max_seqlen_q == 1) {
      constexpr int warps = 4;
      const dim3 grid(splits, shape.query_heads, shape.batch);
      const auto shared_bytes = (shape.head_dim + 2 * warps) * sizeof(float);
      if (splits > 1) {
        const int split_tokens = (max_seqlen_k + splits - 1) / splits;
        flash_attn_with_kvcache_kernel_decode<T, warps, kHeadElements, true>
            <<<grid, warps * 32, shared_bytes, stream>>>(
                query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
                key.data<T>(), value.data<T>(),
                cu_seqlens_q.data<std::int32_t>(),
                cu_seqlens_k.data<std::int32_t>(),
                block_table.data<std::int32_t>(), output.data<T>(), shape,
                causal, workspace.data<float>(), splits, split_tokens);
        check_cuda(cudaGetLastError(), "launch Split-KV decode partitions");
        const dim3 merge_grid(shape.query_heads, shape.batch);
        merge_decode_splits_kernel<T><<<merge_grid, 128, 0, stream>>>(
            workspace.data<float>(), cu_seqlens_q.data<std::int32_t>(),
            output.data<T>(), shape, splits);
      } else {
        flash_attn_with_kvcache_kernel_decode<T, warps, kHeadElements, false>
            <<<grid, warps * 32, shared_bytes, stream>>>(
                query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
                key.data<T>(), value.data<T>(),
                cu_seqlens_q.data<std::int32_t>(),
                cu_seqlens_k.data<std::int32_t>(),
                block_table.data<std::int32_t>(), output.data<T>(), shape,
                causal, nullptr, 1, 0);
      }
    } else {
      const dim3 grid((max_seqlen_q + kQueryTile - 1) / kQueryTile,
                      shape.query_heads, shape.batch);
      const auto shared_bytes = 2 * kKeyTile * shape.head_dim * sizeof(float);
      flash_attn_with_kvcache_kernel<T>
          <<<grid, kThreads, shared_bytes, stream>>>(
              query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
              key.data<T>(), value.data<T>(), cu_seqlens_q.data<std::int32_t>(),
              cu_seqlens_k.data<std::int32_t>(),
              block_table.data<std::int32_t>(), output.data<T>(), shape,
              causal);
    }
  });
  check_cuda(cudaGetLastError(), "launch flash_attn_with_kvcache");
}

} // namespace nanovllm::kernels::cuda
