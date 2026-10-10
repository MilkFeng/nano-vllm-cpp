#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/attention.hh"
#include "kernels/common.cuh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kValuesPerLane = 256 / kWarpSize;
constexpr unsigned kFullWarp = 0xffffffffu;

void assert_heads(const ConstTensorView& tensor) {
  detail::assert_tensor(tensor);
  assert(tensor.rank() == 3 && tensor.shape()[1] > 0);
  assert(tensor.shape()[2] > 0 && tensor.shape()[2] <= 256);
}

template <bool IsConst>
void assert_cache(const BasicTensorView<IsConst>& tensor) {
  detail::assert_tensor(tensor);
  assert(tensor.rank() == 4 && tensor.shape()[1] > 0 && tensor.shape()[2] > 0);
  assert(tensor.shape()[3] > 0 && tensor.shape()[3] <= 256);
}

void assert_indices(const ConstTensorView& tensor, std::size_t rank) {
  detail::assert_tensor(tensor);
  assert(tensor.dtype() == DataType::Int32 && tensor.rank() == rank);
}

void assert_output(const ConstTensorView& query, const TensorView& output,
                   float scale) {
  detail::assert_tensor(output);
  assert(query.dtype() == output.dtype() && detail::same_shape(query, output));
  assert(std::isfinite(scale) && scale > 0);
}

void assert_gqa(const ConstTensorView& query, const ConstTensorView& key,
                bool paged) {
  assert(query.dtype() == key.dtype());
  const auto kv_heads = key.shape()[paged ? 2 : 1];
  const auto width = key.shape()[paged ? 3 : 2];
  assert(query.shape()[2] == width && query.shape()[1] % kv_heads == 0);
}

template <typename T>
__global__ void store_kernel(const T* key, const T* value, T* key_cache,
                             T* value_cache, const std::int32_t* slots,
                             std::size_t tokens, std::size_t token_width,
                             std::size_t num_slots) {
  for (std::size_t token = blockIdx.x; token < tokens; token += gridDim.x) {
    const auto slot = slots[token];
    if (slot == -1) {
      continue;
    }
    assert(slot >= 0 && static_cast<std::size_t>(slot) < num_slots);
    const auto cache_offset = static_cast<std::size_t>(slot) * token_width;
    for (std::size_t d = threadIdx.x; d < token_width; d += blockDim.x) {
      key_cache[cache_offset + d] = key[token * token_width + d];
      value_cache[cache_offset + d] = value[token * token_width + d];
    }
  }
}

struct AttentionShape {
  std::size_t tokens;
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t width;
  std::size_t kv_tokens;
  std::size_t batch;
  std::size_t blocks;
  std::size_t block_size;
  std::size_t max_blocks;
  std::size_t context_capacity;
};

// A single warp owns a query head. Its lanes keep up to eight Q/output values
// each, and share a dot product with shuffle reductions. No temporary scores
// or device allocations are needed, including when sequences cross pages.
template <typename T, bool Paged, bool Decode>
__global__ void attention_kernel(const T* query, const T* key, const T* value,
                                 const std::int32_t* cu_query,
                                 const std::int32_t* cu_key,
                                 const std::int32_t* context_lens,
                                 const std::int32_t* block_tables, T* output,
                                 AttentionShape shape, float scale,
                                 std::size_t rows) {
  const auto lane = threadIdx.x;
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto token = row / shape.query_heads;
    const auto query_head = row % shape.query_heads;
    const auto kv_head = query_head / (shape.query_heads / shape.kv_heads);
    std::size_t sequence = token;
    std::int64_t key_begin = 0;
    std::int64_t visible = 0;
    if constexpr (Decode) {
      visible = context_lens[sequence];
      assert(visible >= 0
             && static_cast<std::size_t>(visible) <= shape.context_capacity);
    } else {
      assert(cu_query[0] == 0 && cu_key[0] == 0
             && cu_query[shape.batch]
                    == static_cast<std::int64_t>(shape.tokens));
      if constexpr (!Paged) {
        assert(cu_key[shape.batch]
               == static_cast<std::int64_t>(shape.kv_tokens));
      }
      // upper_bound handles empty sequences (repeated cumulative lengths).
      std::size_t low = 0, high = shape.batch;
      while (low < high) {
        const auto middle = low + (high - low) / 2;
        if (cu_query[middle + 1] <= static_cast<std::int64_t>(token)) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      sequence = low;
      assert(sequence < shape.batch);
      const std::int64_t query_begin = cu_query[sequence];
      const std::int64_t query_end = cu_query[sequence + 1];
      key_begin = cu_key[sequence];
      const std::int64_t key_end = cu_key[sequence + 1];
      const auto query_length = query_end - query_begin;
      const auto key_length = key_end - key_begin;
      assert(query_begin >= 0 && query_begin <= static_cast<std::int64_t>(token)
             && static_cast<std::int64_t>(token) < query_end
             && query_end <= static_cast<std::int64_t>(shape.tokens));
      assert(key_begin >= 0 && key_length >= 0);
      if constexpr (Paged) {
        assert(static_cast<std::size_t>(key_length) <= shape.context_capacity);
      } else {
        assert(key_end <= static_cast<std::int64_t>(shape.kv_tokens));
      }
      const auto limit = key_length - query_length
                         + static_cast<std::int64_t>(token) - query_begin + 1;
      visible = limit < 0 ? 0 : (limit < key_length ? limit : key_length);
    }

    float q[kValuesPerLane] = {};
    float accumulator[kValuesPerLane] = {};
    const auto query_offset = row * shape.width;
    if (visible != 0) {
#pragma unroll
      for (unsigned part = 0; part < kValuesPerLane; ++part) {
        const auto d = lane + part * kWarpSize;
        if (d < shape.width)
          q[part] = detail::to_float(query[query_offset + d]);
      }
    }
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    for (std::int64_t j = 0; j < visible; ++j) {
      std::size_t slot;
      if constexpr (Paged) {
        const auto block_column =
            static_cast<std::size_t>(j) / shape.block_size;
        const auto block =
            block_tables[sequence * shape.max_blocks + block_column];
        assert(block >= 0 && static_cast<std::size_t>(block) < shape.blocks);
        slot = static_cast<std::size_t>(block) * shape.block_size
               + static_cast<std::size_t>(j) % shape.block_size;
      } else {
        slot = static_cast<std::size_t>(key_begin + j);
      }
      const auto kv_offset = (slot * shape.kv_heads + kv_head) * shape.width;
      float dot = 0.0f;
#pragma unroll
      for (unsigned part = 0; part < kValuesPerLane; ++part) {
        const auto d = lane + part * kWarpSize;
        if (d < shape.width)
          dot += q[part] * detail::to_float(key[kv_offset + d]);
      }
      for (unsigned offset = kWarpSize / 2; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(kFullWarp, dot, offset);
      }
      float alpha = 0.0f, beta = 0.0f;
      if (lane == 0) {
        const float score = dot * scale;
        const float next_maximum = fmaxf(maximum, score);
        alpha = expf(maximum - next_maximum);
        beta = expf(score - next_maximum);
        denominator = denominator * alpha + beta;
        maximum = next_maximum;
      }
      alpha = __shfl_sync(kFullWarp, alpha, 0);
      beta = __shfl_sync(kFullWarp, beta, 0);
#pragma unroll
      for (unsigned part = 0; part < kValuesPerLane; ++part) {
        const auto d = lane + part * kWarpSize;
        if (d < shape.width) {
          accumulator[part] = accumulator[part] * alpha
                              + detail::to_float(value[kv_offset + d]) * beta;
        }
      }
    }
    denominator = __shfl_sync(kFullWarp, denominator, 0);
#pragma unroll
    for (unsigned part = 0; part < kValuesPerLane; ++part) {
      const auto d = lane + part * kWarpSize;
      if (d < shape.width) {
        const float result =
            visible > 0 ? accumulator[part] / denominator : 0.0f;
        output[row * shape.width + d] = detail::from_float<T>(result);
      }
    }
  }
}

AttentionShape make_shape(const ConstTensorView& query,
                          const ConstTensorView& key, std::size_t batch,
                          const ConstTensorView& block_tables, bool paged) {
  AttentionShape shape{};
  shape.tokens = query.shape()[0];
  shape.query_heads = query.shape()[1];
  shape.kv_heads = key.shape()[paged ? 2 : 1];
  shape.width = query.shape()[2];
  shape.batch = batch;
  if (paged) {
    shape.blocks = key.shape()[0];
    shape.block_size = key.shape()[1];
    shape.max_blocks = block_tables.shape()[1];
    shape.context_capacity = shape.max_blocks * shape.block_size;
  } else {
    shape.kv_tokens = key.shape()[0];
  }
  return shape;
}

} // namespace

void store_kvcache(ConstTensorView key, ConstTensorView value,
                   TensorView key_cache, TensorView value_cache,
                   ConstTensorView slot_mapping, cudaStream_t stream) {
  assert_heads(key);
  assert_heads(value);
  assert_cache(key_cache);
  assert_cache(value_cache);
  assert(key.dtype() == value.dtype() && key.dtype() == key_cache.dtype()
         && key.dtype() == value_cache.dtype());
  assert(detail::same_shape(key, value)
         && detail::same_shape(key_cache, value_cache));
  assert(key.shape()[1] == key_cache.shape()[2]
         && key.shape()[2] == key_cache.shape()[3]);
  assert_indices(slot_mapping, 1);
  assert(slot_mapping.shape()[0] == key.shape()[0]);
  const auto tokens = key.shape()[0];
  if (tokens == 0) return;
  const auto token_width = key.shape()[1] * key.shape()[2];
  const auto slots = key_cache.shape()[0] * key_cache.shape()[1];
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>(tokens, 65535));
  detail::dispatch_float(key.dtype(), [&]<typename T>() {
    store_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(key.data()), static_cast<const T*>(value.data()),
        static_cast<T*>(key_cache.data()), static_cast<T*>(value_cache.data()),
        static_cast<const std::int32_t*>(slot_mapping.data()), tokens,
        token_width, slots);
  });
  check_cuda(cudaGetLastError(), "store_kvcache launch");
}

void prefill_attention(ConstTensorView query, ConstTensorView key,
                       ConstTensorView value, ConstTensorView cu_seqlens_q,
                       ConstTensorView cu_seqlens_k,
                       ConstTensorView block_tables, TensorView output,
                       float scale, cudaStream_t stream) {
  assert_heads(query);
  assert_output(query, output, scale);
  const bool paged = block_tables.numel() != 0;
  if (paged) {
    assert_cache(key);
    assert_cache(value);
  } else {
    assert_heads(key);
    assert_heads(value);
  }
  assert(key.dtype() == value.dtype() && detail::same_shape(key, value));
  assert_gqa(query, key, paged);
  assert_indices(cu_seqlens_q, 1);
  assert_indices(cu_seqlens_k, 1);
  assert(cu_seqlens_q.shape()[0] >= 1
         && detail::same_shape(cu_seqlens_q, cu_seqlens_k));
  const auto batch = cu_seqlens_q.shape()[0] - 1;
  assert(query.shape()[0] <= INT_MAX && (paged || key.shape()[0] <= INT_MAX));
  assert(query.shape()[0] == 0 || batch > 0);
  if (paged) {
    assert_indices(block_tables, 2);
    assert(block_tables.shape()[0] == batch);
  }
  const auto rows = query.shape()[0] * query.shape()[1];
  if (rows == 0) return;
  const auto shape = make_shape(query, key, batch, block_tables, paged);
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    if (paged) {
      attention_kernel<T, true, false><<<blocks, kWarpSize, 0, stream>>>(
          static_cast<const T*>(query.data()),
          static_cast<const T*>(key.data()),
          static_cast<const T*>(value.data()),
          static_cast<const std::int32_t*>(cu_seqlens_q.data()),
          static_cast<const std::int32_t*>(cu_seqlens_k.data()), nullptr,
          static_cast<const std::int32_t*>(block_tables.data()),
          static_cast<T*>(output.data()), shape, scale, rows);
    } else {
      attention_kernel<T, false, false><<<blocks, kWarpSize, 0, stream>>>(
          static_cast<const T*>(query.data()),
          static_cast<const T*>(key.data()),
          static_cast<const T*>(value.data()),
          static_cast<const std::int32_t*>(cu_seqlens_q.data()),
          static_cast<const std::int32_t*>(cu_seqlens_k.data()), nullptr,
          nullptr, static_cast<T*>(output.data()), shape, scale, rows);
    }
  });
  check_cuda(cudaGetLastError(), "prefill_attention launch");
}

void paged_attention_decode(ConstTensorView query, ConstTensorView key_cache,
                            ConstTensorView value_cache,
                            ConstTensorView context_lens,
                            ConstTensorView block_tables, TensorView output,
                            float scale, cudaStream_t stream) {
  assert_heads(query);
  assert_output(query, output, scale);
  assert_cache(key_cache);
  assert_cache(value_cache);
  assert(key_cache.dtype() == value_cache.dtype()
         && detail::same_shape(key_cache, value_cache));
  assert_gqa(query, key_cache, true);
  assert_indices(context_lens, 1);
  assert_indices(block_tables, 2);
  const auto batch = query.shape()[0];
  assert(context_lens.shape()[0] == batch && block_tables.shape()[0] == batch);
  const auto rows = batch * query.shape()[1];
  if (rows == 0) return;
  const auto shape = make_shape(query, key_cache, batch, block_tables, true);
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    attention_kernel<T, true, true><<<blocks, kWarpSize, 0, stream>>>(
        static_cast<const T*>(query.data()),
        static_cast<const T*>(key_cache.data()),
        static_cast<const T*>(value_cache.data()), nullptr, nullptr,
        static_cast<const std::int32_t*>(context_lens.data()),
        static_cast<const std::int32_t*>(block_tables.data()),
        static_cast<T*>(output.data()), shape, scale, rows);
  });
  check_cuda(cudaGetLastError(), "paged_attention_decode launch");
}

} // namespace nanovllm::kernels
