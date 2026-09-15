#include <math_constants.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "kernels/attention.hh"
#include "kernels/common.cuh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kValuesPerLane = 256 / kWarpSize;
constexpr unsigned kFullWarp = 0xffffffffu;

void require_heads(ConstTensorView tensor) {
  detail::require_cuda(tensor);
  detail::require_float(tensor);
  detail::require(tensor.rank() == 3,
                  "attention Q/K/V must have shape [tokens,heads,head_dim]");
  const auto heads = tensor.shape()[1];
  const auto width = tensor.shape()[2];
  detail::require(heads > 0 && width > 0 && width <= 256,
                  "attention requires positive heads and head_dim in [1,256]");
  const auto token_width = detail::checked_mul(heads, width);
  detail::require(
      tensor.strides()[2] == 1 && tensor.strides()[1] == width
          && tensor.strides()[0] >= token_width,
      "attention requires contiguous heads and nonoverlapping token rows");
  (void)detail::storage_span(tensor);
}

void require_cache(ConstTensorView tensor) {
  detail::require_cuda(tensor);
  detail::require_float(tensor);
  detail::require(
      tensor.rank() == 4 && tensor.is_contiguous(),
      "attention cache must be contiguous [blocks,block_size,heads,head_dim]");
  detail::require(tensor.shape()[1] > 0 && tensor.shape()[2] > 0
                      && tensor.shape()[3] > 0 && tensor.shape()[3] <= 256,
                  "attention cache requires positive block size/heads and "
                  "head_dim in [1,256]");
  (void)detail::storage_span(tensor);
}

void require_indices(ConstTensorView tensor, ConstTensorView query,
                     std::size_t rank) {
  detail::require_cuda(tensor);
  detail::require(tensor.dtype() == DataType::Int32 && tensor.rank() == rank
                      && tensor.is_contiguous(),
                  "attention indices must be contiguous Int32 tensors of the "
                  "expected rank");
  detail::require(tensor.device() == query.device(),
                  "attention tensors must share one CUDA device");
  (void)detail::storage_span(tensor);
}

void require_output(ConstTensorView query, ConstTensorView output,
                    float scale) {
  detail::require_cuda(output);
  detail::require_same_device_dtype(query, output);
  detail::require(detail::same_shape(query, output) && output.is_contiguous(),
                  "attention output must be contiguous and match query shape");
  detail::require(std::isfinite(scale) && scale > 0,
                  "attention scale must be finite and positive");
}

void require_disjoint(ConstTensorView input, ConstTensorView output) {
  detail::require(!detail::overlaps(input, output),
                  "attention writable tensors must not overlap other tensors");
}

void require_pair(ConstTensorView key, ConstTensorView value) {
  detail::require_same_device_dtype(key, value);
  detail::require(detail::same_shape(key, value),
                  "attention key and value shapes must match");
}

void require_gqa(ConstTensorView query, ConstTensorView key, bool paged) {
  detail::require_same_device_dtype(query, key);
  const auto kv_heads = key.shape()[paged ? 2 : 1];
  const auto width = key.shape()[paged ? 3 : 2];
  detail::require(query.shape()[2] == width && query.shape()[1] % kv_heads == 0,
                  "attention requires equal head dimensions and query heads "
                  "divisible by KV heads");
}

unsigned blocks_for(std::size_t rows) {
  return static_cast<unsigned>(std::min(rows, std::size_t{65535}));
}

template <typename T>
__global__ void store_kernel(const T* key, const T* value, T* key_cache,
                             T* value_cache, const std::int32_t* slots,
                             std::size_t tokens, std::size_t token_width,
                             std::size_t key_stride, std::size_t value_stride,
                             std::size_t num_slots) {
  for (std::size_t token = blockIdx.x; token < tokens; token += gridDim.x) {
    const auto slot = slots[token];
    if (slot == -1) continue;
    const bool valid = slot >= 0 && static_cast<std::size_t>(slot) < num_slots;
    assert(valid && "KV cache slot is outside the cache");
    if (!valid) continue;
    const auto cache_offset = static_cast<std::size_t>(slot) * token_width;
    for (std::size_t d = threadIdx.x; d < token_width; d += blockDim.x) {
      key_cache[cache_offset + d] = key[token * key_stride + d];
      value_cache[cache_offset + d] = value[token * value_stride + d];
    }
  }
}

struct AttentionShape {
  std::size_t tokens;
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t width;
  std::size_t query_stride;
  std::size_t key_stride;
  std::size_t value_stride;
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
    bool valid = true;
    if constexpr (Decode) {
      visible = context_lens[sequence];
      valid = visible >= 0
              && static_cast<std::size_t>(visible) <= shape.context_capacity;
    } else {
      valid =
          cu_query[0] == 0 && cu_key[0] == 0
          && cu_query[shape.batch] == static_cast<std::int64_t>(shape.tokens);
      if constexpr (!Paged) {
        valid = valid
                && cu_key[shape.batch]
                       == static_cast<std::int64_t>(shape.kv_tokens);
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
      valid = valid && sequence < shape.batch;
      if (valid) {
        const std::int64_t query_begin = cu_query[sequence];
        const std::int64_t query_end = cu_query[sequence + 1];
        key_begin = cu_key[sequence];
        const std::int64_t key_end = cu_key[sequence + 1];
        const auto query_length = query_end - query_begin;
        const auto key_length = key_end - key_begin;
        valid = query_begin >= 0
                && query_begin <= static_cast<std::int64_t>(token)
                && static_cast<std::int64_t>(token) < query_end
                && query_end <= static_cast<std::int64_t>(shape.tokens)
                && key_begin >= 0 && key_length >= 0;
        if constexpr (Paged) {
          valid =
              valid
              && static_cast<std::size_t>(key_length) <= shape.context_capacity;
        } else {
          valid =
              valid && key_end <= static_cast<std::int64_t>(shape.kv_tokens);
        }
        const auto limit = key_length - query_length
                           + static_cast<std::int64_t>(token) - query_begin + 1;
        visible = limit < 0 ? 0 : (limit < key_length ? limit : key_length);
      }
    }
    assert(valid && "attention sequence lengths are invalid");

    float q[kValuesPerLane] = {};
    float accumulator[kValuesPerLane] = {};
    const auto query_offset =
        token * shape.query_stride + query_head * shape.width;
    if (valid && visible != 0) {
#pragma unroll
      for (unsigned part = 0; part < kValuesPerLane; ++part) {
        const auto d = lane + part * kWarpSize;
        if (d < shape.width)
          q[part] = detail::to_float(query[query_offset + d]);
      }
    }
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    for (std::int64_t j = 0; valid && j < visible; ++j) {
      std::size_t slot;
      if constexpr (Paged) {
        const auto block_column =
            static_cast<std::size_t>(j) / shape.block_size;
        const auto block =
            block_tables[sequence * shape.max_blocks + block_column];
        valid = block >= 0 && static_cast<std::size_t>(block) < shape.blocks;
        assert(valid && "attention block table contains an invalid block ID");
        if (!valid) break;
        slot = static_cast<std::size_t>(block) * shape.block_size
               + static_cast<std::size_t>(j) % shape.block_size;
      } else {
        slot = static_cast<std::size_t>(key_begin + j);
      }
      const auto key_offset = slot * shape.key_stride + kv_head * shape.width;
      const auto value_offset =
          slot * shape.value_stride + kv_head * shape.width;
      float dot = 0.0f;
#pragma unroll
      for (unsigned part = 0; part < kValuesPerLane; ++part) {
        const auto d = lane + part * kWarpSize;
        if (d < shape.width)
          dot += q[part] * detail::to_float(key[key_offset + d]);
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
          accumulator[part] =
              accumulator[part] * alpha
              + detail::to_float(value[value_offset + d]) * beta;
        }
      }
    }
    denominator = __shfl_sync(kFullWarp, denominator, 0);
#pragma unroll
    for (unsigned part = 0; part < kValuesPerLane; ++part) {
      const auto d = lane + part * kWarpSize;
      if (d < shape.width) {
        const float result =
            valid && visible > 0 ? accumulator[part] / denominator : 0.0f;
        output[row * shape.width + d] = detail::from_float<T>(result);
      }
    }
  }
}

AttentionShape make_shape(ConstTensorView query, ConstTensorView key,
                          ConstTensorView value, std::size_t batch,
                          ConstTensorView block_tables, bool paged) {
  AttentionShape shape{};
  shape.tokens = query.shape()[0];
  shape.query_heads = query.shape()[1];
  shape.kv_heads = key.shape()[paged ? 2 : 1];
  shape.width = query.shape()[2];
  shape.query_stride = query.strides()[0];
  shape.batch = batch;
  if (paged) {
    shape.blocks = key.shape()[0];
    shape.block_size = key.shape()[1];
    shape.max_blocks = block_tables.shape()[1];
    shape.context_capacity =
        detail::checked_mul(shape.max_blocks, shape.block_size);
    shape.key_stride = shape.value_stride =
        detail::checked_mul(shape.kv_heads, shape.width);
  } else {
    shape.kv_tokens = key.shape()[0];
    shape.key_stride = key.strides()[0];
    shape.value_stride = value.strides()[0];
  }
  return shape;
}

} // namespace

void store_kvcache(ConstTensorView key, ConstTensorView value,
                   TensorView key_cache, TensorView value_cache,
                   ConstTensorView slot_mapping, cudaStream_t stream) {
  require_heads(key);
  require_heads(value);
  require_pair(key, value);
  require_cache(key_cache.as_const());
  require_cache(value_cache.as_const());
  require_pair(key_cache.as_const(), value_cache.as_const());
  detail::require_same_device_dtype(key, key_cache.as_const());
  detail::require(
      key.shape()[1] == key_cache.shape()[2]
          && key.shape()[2] == key_cache.shape()[3],
      "KV cache heads and head dimension must match incoming keys/values");
  require_indices(slot_mapping, key, 1);
  detail::require(slot_mapping.shape()[0] == key.shape()[0],
                  "KV slot count must equal incoming token count");
  require_disjoint(key_cache.as_const(), value_cache.as_const());
  for (const auto input : {key, value, slot_mapping}) {
    require_disjoint(input, key_cache.as_const());
    require_disjoint(input, value_cache.as_const());
  }
  const auto tokens = key.shape()[0];
  if (tokens == 0) return;
  const auto token_width = detail::checked_mul(key.shape()[1], key.shape()[2]);
  const auto slots =
      detail::checked_mul(key_cache.shape()[0], key_cache.shape()[1]);
  const detail::DeviceGuard guard(key.device().index);
  detail::dispatch_float(key.dtype(), [&]<typename T>() {
    store_kernel<<<blocks_for(tokens), 256, 0, stream>>>(
        static_cast<const T*>(key.data()), static_cast<const T*>(value.data()),
        static_cast<T*>(key_cache.data()), static_cast<T*>(value_cache.data()),
        static_cast<const std::int32_t*>(slot_mapping.data()), tokens,
        token_width, key.strides()[0], value.strides()[0], slots);
  });
  detail::check_cuda(cudaGetLastError(), "store_kvcache launch");
}

void prefill_attention(ConstTensorView query, ConstTensorView key,
                       ConstTensorView value, ConstTensorView cu_seqlens_q,
                       ConstTensorView cu_seqlens_k,
                       ConstTensorView block_tables, TensorView output,
                       float scale, cudaStream_t stream) {
  require_heads(query);
  require_output(query, output.as_const(), scale);
  const bool paged = block_tables.numel() != 0;
  if (paged) {
    require_cache(key);
    require_cache(value);
  } else {
    require_heads(key);
    require_heads(value);
  }
  require_pair(key, value);
  require_gqa(query, key, paged);
  require_indices(cu_seqlens_q, query, 1);
  require_indices(cu_seqlens_k, query, 1);
  detail::require(
      cu_seqlens_q.shape()[0] >= 1
          && detail::same_shape(cu_seqlens_q, cu_seqlens_k),
      "prefill cumulative lengths must have matching shape [batch+1]");
  const auto batch = cu_seqlens_q.shape()[0] - 1;
  detail::require(
      query.shape()[0] <= std::numeric_limits<std::int32_t>::max()
          && (paged
              || key.shape()[0] <= std::numeric_limits<std::int32_t>::max())
          && (query.shape()[0] == 0 || batch > 0),
      "prefill token counts must fit Int32 sequence lengths and a non-empty "
      "batch");
  if (paged) {
    require_indices(block_tables, query, 2);
    detail::require(block_tables.shape()[0] == batch,
                    "prefill block table must have one row per sequence");
    require_disjoint(block_tables, output.as_const());
  }
  for (const auto input : {query, key, value, cu_seqlens_q, cu_seqlens_k}) {
    require_disjoint(input, output.as_const());
  }
  const auto rows = detail::checked_mul(query.shape()[0], query.shape()[1]);
  if (rows == 0) return;
  const auto shape = make_shape(query, key, value, batch, block_tables, paged);
  const detail::DeviceGuard guard(query.device().index);
  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    if (paged) {
      attention_kernel<T, true, false>
          <<<blocks_for(rows), kWarpSize, 0, stream>>>(
              static_cast<const T*>(query.data()),
              static_cast<const T*>(key.data()),
              static_cast<const T*>(value.data()),
              static_cast<const std::int32_t*>(cu_seqlens_q.data()),
              static_cast<const std::int32_t*>(cu_seqlens_k.data()), nullptr,
              static_cast<const std::int32_t*>(block_tables.data()),
              static_cast<T*>(output.data()), shape, scale, rows);
    } else {
      attention_kernel<T, false, false>
          <<<blocks_for(rows), kWarpSize, 0, stream>>>(
              static_cast<const T*>(query.data()),
              static_cast<const T*>(key.data()),
              static_cast<const T*>(value.data()),
              static_cast<const std::int32_t*>(cu_seqlens_q.data()),
              static_cast<const std::int32_t*>(cu_seqlens_k.data()), nullptr,
              nullptr, static_cast<T*>(output.data()), shape, scale, rows);
    }
  });
  detail::check_cuda(cudaGetLastError(), "prefill_attention launch");
}

void paged_attention_decode(ConstTensorView query, ConstTensorView key_cache,
                            ConstTensorView value_cache,
                            ConstTensorView context_lens,
                            ConstTensorView block_tables, TensorView output,
                            float scale, cudaStream_t stream) {
  require_heads(query);
  require_output(query, output.as_const(), scale);
  require_cache(key_cache);
  require_cache(value_cache);
  require_pair(key_cache, value_cache);
  require_gqa(query, key_cache, true);
  require_indices(context_lens, query, 1);
  require_indices(block_tables, query, 2);
  const auto batch = query.shape()[0];
  detail::require(
      context_lens.shape()[0] == batch && block_tables.shape()[0] == batch,
      "decode lengths and block table must have one row per query token");
  for (const auto input :
       {query, key_cache, value_cache, context_lens, block_tables}) {
    require_disjoint(input, output.as_const());
  }
  const auto rows = detail::checked_mul(batch, query.shape()[1]);
  if (rows == 0) return;
  const auto shape =
      make_shape(query, key_cache, value_cache, batch, block_tables, true);
  const detail::DeviceGuard guard(query.device().index);
  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    attention_kernel<T, true, true><<<blocks_for(rows), kWarpSize, 0, stream>>>(
        static_cast<const T*>(query.data()),
        static_cast<const T*>(key_cache.data()),
        static_cast<const T*>(value_cache.data()), nullptr, nullptr,
        static_cast<const std::int32_t*>(context_lens.data()),
        static_cast<const std::int32_t*>(block_tables.data()),
        static_cast<T*>(output.data()), shape, scale, rows);
  });
  detail::check_cuda(cudaGetLastError(), "paged_attention_decode launch");
}

} // namespace nanovllm::kernels
