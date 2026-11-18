#include <algorithm>
#include <climits>
#include <type_traits>

#include "common/cuda_utils.hh"
#include "kernels/cuda/attn/attn_shape.cuh"
#include "kernels/cuda/attn/attn_simt.cuh"
#include "kernels/cuda/attn/attn_simt_decode.cuh"
#include "kernels/cuda/attn/attn_tensor.cuh"
#include "kernels/cuda/attn/attn_vector_decode.cuh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr std::size_t MaxSplits = 32;
constexpr std::size_t DecodeTargetBlocks = 1024;
constexpr std::size_t MinSplitTokens = 128;
constexpr std::size_t MaxTensorRows = 256 * 64;

int decode_splits(std::size_t batch, std::size_t heads,
                  std::size_t max_seqlen_k) {
  if (batch == 0 || heads == 0 || max_seqlen_k < 2 * MinSplitTokens
      || heads >= DecodeTargetBlocks
      || batch >= (DecodeTargetBlocks + heads - 1) / heads) {
    return 1;
  }

  const auto blocks = batch * heads;
  const auto wanted = (DecodeTargetBlocks + blocks - 1) / blocks;

  int splits = 1;
  while (splits < wanted && splits < MaxSplits
         && 2 * splits <= max_seqlen_k / MinSplitTokens) {
    splits *= 2;
  }

  return splits;
}

int tensor_splits(std::size_t batch, std::size_t heads, std::size_t hgroup,
                  std::size_t max_seqlen_q, std::size_t max_seqlen_k, int M,
                  std::size_t query_tokens) {
  const auto blocks = batch * heads * ((max_seqlen_q * hgroup + M - 1) / M);
  const auto rows = query_tokens * heads * hgroup;
  const std::size_t target_blocks = M == 128 ? 512 : 256;

  std::size_t min_split_tokens = 128;
  if (max_seqlen_q == 1) {
    min_split_tokens = 32;
  } else if (M == 128) {
    min_split_tokens = 64;
  }

  int splits = 1;
  while (splits < MaxSplits) {
    const int next_splits = splits * 2;
    const auto next_blocks = blocks * next_splits;
    const auto next_rows = rows * next_splits;
    const auto tokens_per_split = max_seqlen_k / next_splits;

    if (next_blocks > target_blocks || next_rows > MaxTensorRows
        || tokens_per_split < min_split_tokens) {
      break;
    }
    splits = next_splits;
  }

  return splits;
}

} // namespace

std::size_t flash_attn_workspace_elements(std::size_t max_batch,
                                          std::size_t query_heads,
                                          std::size_t head_dim,
                                          std::size_t max_seqlen_k) {
  // 小 batch 可能使用更多分片，容量需要覆盖所有不超过 max_batch 的 batch
  const auto limit = std::min(max_batch, DecodeTargetBlocks / query_heads);
  std::size_t elements = 0;
  for (std::size_t batch = 1; batch <= limit; ++batch) {
    const int splits = decode_splits(batch, query_heads, max_seqlen_k);
    if (splits > 1) {
      elements =
          std::max(elements, batch * query_heads * splits * (head_dim + 2));
    }
  }

  // Tensor 路径限制 packed query/head/split 行数，覆盖 prefill 和 decode
  if (max_batch > 0 && max_seqlen_k >= 256) {
    elements = std::max(elements, MaxTensorRows * (head_dim + 2));
  }
  return elements;
}

void flash_attn_with_kvcache(
    ConstTensorView query, TensorView key_cache, TensorView value_cache,
    ConstTensorView key, ConstTensorView value, ConstTensorView cu_seqlens_q,
    ConstTensorView cu_seqlens_k, std::size_t max_seqlen_q,
    std::size_t max_seqlen_k, ConstTensorView block_table, TensorView output,
    bool causal, cudaStream_t stream, TensorView workspace) {
  const auto shape = attn::attn_shape(query, key_cache, value_cache, key, value,
                                      cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                      max_seqlen_k, block_table, output);

  if (shape.query_tokens == 0) {
    return;
  }

  assert(shape.head_dim <= attn::MaxHeadDim);

  const bool decode = max_seqlen_q == 1;
  const bool has_workspace = workspace.data() != nullptr;

  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    if constexpr (!std::is_same_v<T, float>) {
      // FP16/BF16：D=256 的 decode，以及 D=128/256 的 prefill 使用 Tensor Core
      if (shape.head_dim == 256 || (shape.head_dim == 128 && !decode)) {
        const int M = decode ? 16 : shape.head_dim == 256 ? 128 : 64;
        const int ts = has_workspace && (!decode || max_seqlen_k >= 256)
                           ? tensor_splits(shape.batch, shape.key_heads,
                                           shape.query_heads / shape.key_heads,
                                           max_seqlen_q, max_seqlen_k, M,
                                           shape.query_tokens)
                           : 1;

        const auto launch = [&]<int HeadDim, int Warps, int Keys, int Cols>() {
          if (ts > 1) {
            attn::launch_attn_tensor<T, HeadDim, Warps, Keys, true, Cols>(
                query, key_cache, value_cache, key, value, cu_seqlens_q,
                cu_seqlens_k, block_table, output, shape, causal, stream,
                workspace.data<float>(), ts);
          } else {
            attn::launch_attn_tensor<T, HeadDim, Warps, Keys, false, Cols>(
                query, key_cache, value_cache, key, value, cu_seqlens_q,
                cu_seqlens_k, block_table, output, shape, causal, stream,
                nullptr, ts);
          }
        };

        if (decode) {
          launch.template operator()<256, 4, 32, 4>();
        } else if (shape.head_dim == 256) {
          launch.template operator()<256, 8, 16, 1>();
        } else {
          launch.template operator()<128, 4, 32, 1>();
        }
        return;
      }
    }

    if (decode) {
      const int s =
          decode && has_workspace
              ? decode_splits(shape.batch, shape.query_heads, max_seqlen_k)
              : 1;

      const auto launch = [&]<bool SplitKV>() {
        float* partials = SplitKV ? workspace.data<float>() : nullptr;

        if constexpr (!std::is_same_v<T, float>) {
          if (shape.head_dim == 128) {
            attn::launch_attn_vector_decode<T, 128, 4, SplitKV>(
                query, key_cache, value_cache, key, value, cu_seqlens_q,
                cu_seqlens_k, block_table, output, shape, causal, stream,
                partials, s);
            return;
          }
        }

        attn::launch_attn_simt_decode<T, 4, SplitKV>(
            query, key_cache, value_cache, key, value, cu_seqlens_q,
            cu_seqlens_k, block_table, output, shape, causal, stream, partials,
            s);
      };

      if (s > 1) {
        launch.template operator()<true>();
      } else {
        launch.template operator()<false>();
      }
    } else {
      attn::launch_attn_simt<T, 4, 16>(
          query, key_cache, value_cache, key, value, cu_seqlens_q, cu_seqlens_k,
          block_table, output, shape, causal, stream);
    }
  });

  check_cuda(cudaGetLastError(), "launch flash_attn_with_kvcache");
}

} // namespace nanovllm::kernels::cuda
