#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "kernels/common.cuh"
#include "kernels/kernels.hh"

namespace nanovllm::kernels {
namespace {

constexpr std::uint32_t kThreads = 256;

std::uint32_t block_count(std::size_t count) {
  return static_cast<std::uint32_t>(
      std::min<std::size_t>(1 + (count - 1) / kThreads, 65535));
}

__global__ void build_rope_cache_kernel(float* cache, std::size_t pairs,
                                        std::size_t half_dim, float theta) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (std::size_t index = start_index; index < pairs; index += step) {
    const auto position = index / half_dim;
    const auto dimension = index % half_dim;
    const float exponent = static_cast<float>(dimension) / half_dim;
    const float inverse_frequency = 1.0f / __powf(theta, exponent);
    const float angle = static_cast<float>(position) * inverse_frequency;
    float sine;
    float cosine;
    __sincosf(angle, &sine, &cosine);
    const auto offset = position * (2 * half_dim) + dimension;
    cache[offset] = cosine;
    cache[offset + half_dim] = sine;
  }
}

template <typename T>
__global__ void rotary_embedding_kernel(
    const std::int32_t* positions, const T* query, const T* key,
    const float* cache, T* query_output, T* key_output, std::size_t pairs,
    std::size_t query_heads, std::size_t key_heads, std::size_t half_dim,
    std::size_t query_stride, std::size_t key_stride,
    std::size_t max_positions) {
  const auto head_dim = 2 * half_dim;
  const auto total_heads = query_heads + key_heads;

  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (std::size_t index = start_index; index < pairs; index += step) {
    const auto dimension = index % half_dim;
    const auto head = (index / half_dim) % total_heads;
    const auto token = index / (half_dim * total_heads);
    const auto position = positions[token];
    if (position < 0 || position >= max_positions) {
      continue;
    }

    const bool is_query = head < query_heads;
    const auto local_head = is_query ? head : head - query_heads;
    const auto heads = is_query ? query_heads : key_heads;
    const auto stride = is_query ? query_stride : key_stride;
    const T* input = is_query ? query : key;
    T* output = is_query ? query_output : key_output;
    const auto input_offset =
        token * stride + local_head * head_dim + dimension;
    const auto output_offset =
        (token * heads + local_head) * head_dim + dimension;
    const auto cache_offset =
        static_cast<std::size_t>(position) * head_dim + dimension;

    // Read both halves before writing, so contiguous inputs can be rotated in
    // place.
    const float first = detail::to_float(input[input_offset]);
    const float second = detail::to_float(input[input_offset + half_dim]);
    const float cosine = cache[cache_offset];
    const float sine = cache[cache_offset + half_dim];
    output[output_offset] =
        detail::from_float<T>(first * cosine - second * sine);
    output[output_offset + half_dim] =
        detail::from_float<T>(second * cosine + first * sine);
  }
}

void require_cache(ConstTensorView cache) {
  detail::require_cuda(cache);
  detail::require(
      cache.dtype() == DataType::Float32 && cache.rank() == 2
          && cache.is_contiguous(),
      "RoPE cache must be contiguous Float32 [positions, head_dim]");
  detail::require(cache.shape()[1] > 0 && cache.shape()[1] % 2 == 0,
                  "RoPE head_dim must be positive and even");
}

void require_heads(ConstTensorView input, ConstTensorView output) {
  detail::require_cuda(input);
  detail::require_cuda(output);
  detail::require_float(input);
  detail::require_same_device_dtype(input, output);
  detail::require(input.rank() == 3 && output.rank() == 3,
                  "RoPE Q/K tensors must have shape [tokens, heads, head_dim]");
  detail::require(std::equal(input.shape().begin(), input.shape().end(),
                             output.shape().begin())
                      && output.is_contiguous(),
                  "RoPE output must be contiguous and match its input shape");
  const auto heads = input.shape()[1];
  const auto head_dim = input.shape()[2];
  detail::require(heads > 0 && head_dim > 0 && head_dim % 2 == 0,
                  "RoPE requires positive heads and positive even head_dim");
  detail::require(heads <= std::numeric_limits<std::size_t>::max() / head_dim,
                  "RoPE head dimensions are too large");
  detail::require(
      input.strides()[2] == 1 && input.strides()[1] == head_dim
          && input.strides()[0] >= heads * head_dim,
      "RoPE input must have contiguous heads and a valid token stride");
}

} // namespace

void build_rope_cache(TensorView cos_sin_cache, float theta,
                      cudaStream_t stream) {
  require_cache(cos_sin_cache.as_const());
  detail::require(std::isfinite(theta) && theta > 0,
                  "RoPE theta must be finite and positive");
  const auto pairs = cos_sin_cache.numel() / 2;
  if (pairs == 0) {
    return;
  }

  const auto blocks = block_count(pairs);

  const detail::DeviceGuard guard(cos_sin_cache.device().index);
  build_rope_cache_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<float*>(cos_sin_cache.data()), pairs,
      cos_sin_cache.shape()[1] / 2, theta);

  detail::check_cuda(cudaGetLastError(), "build_rope_cache launch");
}

void apply_rotary_embedding(ConstTensorView positions, ConstTensorView query,
                            ConstTensorView key, ConstTensorView cos_sin_cache,
                            TensorView query_output, TensorView key_output,
                            cudaStream_t stream) {
  require_cache(cos_sin_cache);
  require_heads(query, query_output.as_const());
  require_heads(key, key_output.as_const());
  detail::require_cuda(positions);
  detail::require_same_device_dtype(query, key);
  detail::require(positions.device() == query.device()
                      && cos_sin_cache.device() == query.device(),
                  "RoPE tensors must be on the same CUDA device");
  detail::require(positions.dtype() == DataType::Int32 && positions.rank() == 1
                      && positions.is_contiguous(),
                  "RoPE positions must be contiguous Int32 [tokens]");
  detail::require(positions.shape()[0] == query.shape()[0]
                      && query.shape()[0] == key.shape()[0]
                      && query.shape()[2] == key.shape()[2]
                      && query.shape()[2] == cos_sin_cache.shape()[1],
                  "RoPE token counts and head dimensions must match");
  if (query.shape()[0] == 0) {
    return;
  }
  detail::require(cos_sin_cache.shape()[0] > 0,
                  "Non-empty RoPE input requires a non-empty cache");
  const auto q_output = query_output.as_const();
  const auto k_output = key_output.as_const();
  detail::require(
      !detail::overlaps(q_output, k_output) && !detail::overlaps(q_output, key)
          && !detail::overlaps(k_output, query),
      "RoPE Q/K outputs must not overlap each other or the other input");
  detail::require_exact_or_disjoint(
      query, q_output,
      "RoPE only supports exact contiguous query/output aliasing");
  detail::require_exact_or_disjoint(
      key, k_output, "RoPE only supports exact contiguous key/output aliasing");
  detail::require(!detail::overlaps(q_output, cos_sin_cache)
                      && !detail::overlaps(k_output, cos_sin_cache)
                      && !detail::overlaps(q_output, positions)
                      && !detail::overlaps(k_output, positions),
                  "RoPE outputs must not overlap the cache or positions");
  detail::require(query.numel() / 2 <= std::numeric_limits<std::size_t>::max()
                                           - key.numel() / 2,
                  "RoPE input is too large");

  const auto pairs = query.numel() / 2 + key.numel() / 2;
  const auto blocks = block_count(pairs);

  const detail::DeviceGuard guard(query.device().index);
  detail::dispatch_float(query.dtype(), [&]<typename T> {
    rotary_embedding_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data()),
        static_cast<const T*>(query.data()), static_cast<const T*>(key.data()),
        static_cast<const float*>(cos_sin_cache.data()),
        static_cast<T*>(query_output.data()),
        static_cast<T*>(key_output.data()), pairs, query.shape()[1],
        key.shape()[1], query.shape()[2] / 2, query.strides()[0],
        key.strides()[0], cos_sin_cache.shape()[0]);
  });

  detail::check_cuda(cudaGetLastError(), "apply_rotary_embedding launch");
}

} // namespace nanovllm::kernels
