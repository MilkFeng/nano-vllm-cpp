#include <algorithm>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/common.cuh"
#include "kernels/kernels.hh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

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
    const std::int32_t* positions, T* query, T* key, const float* cache,
    std::size_t pairs, std::size_t query_heads, std::size_t key_heads,
    std::size_t half_dim, std::size_t max_positions) {
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
    assert(position >= 0 && static_cast<std::size_t>(position) < max_positions);

    const bool is_query = head < query_heads;
    const auto local_head = is_query ? head : head - query_heads;
    const auto heads = is_query ? query_heads : key_heads;
    T* data = is_query ? query : key;
    const auto offset = (token * heads + local_head) * head_dim + dimension;
    const auto cache_offset =
        static_cast<std::size_t>(position) * head_dim + dimension;

    // Read both halves before rotating in place.
    const float first = detail::to_float(data[offset]);
    const float second = detail::to_float(data[offset + half_dim]);
    const float cosine = cache[cache_offset];
    const float sine = cache[cache_offset + half_dim];
    data[offset] = detail::from_float<T>(first * cosine - second * sine);
    data[offset + half_dim] =
        detail::from_float<T>(second * cosine + first * sine);
  }
}

template <bool IsConst>
void assert_cache(const BasicTensorView<IsConst>& cache) {
  detail::assert_tensor(cache);
  assert(cache.dtype() == DataType::Float32 && cache.rank() == 2);
  assert(cache.shape()[1] > 0 && cache.shape()[1] % 2 == 0);
}

} // namespace

void build_rope_cache(TensorView cos_sin_cache, float theta,
                      cudaStream_t stream) {
  assert_cache(cos_sin_cache);
  assert(std::isfinite(theta) && theta > 0);
  const auto pairs = cos_sin_cache.numel() / 2;
  if (pairs == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((pairs + kThreads - 1) / kThreads, 65535));
  build_rope_cache_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<float*>(cos_sin_cache.data()), pairs,
      cos_sin_cache.shape()[1] / 2, theta);

  check_cuda(cudaGetLastError(), "build_rope_cache launch");
}

void apply_rotary_embedding(ConstTensorView positions, TensorView query,
                            TensorView key, ConstTensorView cos_sin_cache,
                            cudaStream_t stream) {
  assert_cache(cos_sin_cache);
  detail::assert_tensor(positions);
  detail::assert_tensor(query);
  detail::assert_tensor(key);
  assert(positions.dtype() == DataType::Int32 && positions.rank() == 1);
  assert(query.rank() == 3 && key.rank() == 3 && same_dtype(query, key));
  assert(query.shape()[1] > 0 && key.shape()[1] > 0);
  assert(positions.shape()[0] == query.shape()[0]
         && query.shape()[0] == key.shape()[0]);
  assert(query.shape()[2] == key.shape()[2]
         && query.shape()[2] == cos_sin_cache.shape()[1]);
  if (query.shape()[0] == 0) {
    return;
  }

  const auto dtype = query.dtype();
  const auto pairs = query.numel() / 2 + key.numel() / 2;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((pairs + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(dtype, [&]<typename T> {
    rotary_embedding_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data()),
        static_cast<T*>(query.data()), static_cast<T*>(key.data()),
        static_cast<const float*>(cos_sin_cache.data()), pairs,
        query.shape()[1], key.shape()[1], query.shape()[2] / 2,
        cos_sin_cache.shape()[0]);
  });

  check_cuda(cudaGetLastError(), "apply_rotary_embedding launch");
}

} // namespace nanovllm::kernels
