#pragma once

#include <climits>

#include "kernels/cuda/common.cuh"

namespace nanovllm::kernels::cuda::attn {

inline constexpr std::size_t MaxHeadDim = 256;

struct AttentionShape {
  std::size_t batch;
  std::size_t query_tokens;
  std::size_t max_seqlen_q, max_seqlen_k;
  std::size_t query_heads, key_heads;
  std::size_t head_dim;
  std::size_t blocks, block_tokens;
  std::size_t max_blocks;
};

inline AttentionShape attn_shape(
    ConstTensorView query, TensorView key_cache, TensorView value_cache,
    ConstTensorView key, ConstTensorView value, ConstTensorView cu_seqlens_q,
    ConstTensorView cu_seqlens_k, std::size_t max_seqlen_q,
    std::size_t max_seqlen_k, ConstTensorView block_table, TensorView output) {
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
  assert(shape.query_heads > 0 && shape.head_dim > 0
         && shape.head_dim <= MaxHeadDim);

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

} // namespace nanovllm::kernels::cuda::attn
