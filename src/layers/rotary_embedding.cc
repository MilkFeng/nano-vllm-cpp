#include "layers/rotary_embedding.hh"

#include <cassert>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

RotaryEmbedding::RotaryEmbedding(std::size_t max_positions,
                                 std::size_t head_dim, float theta,
                                 Device device, std::size_t rotary_dim)
    : head_dim_(head_dim),
      rotary_dim_(rotary_dim == 0 ? head_dim : rotary_dim),
      theta_(theta),
      rope_cache_(Tensor::empty({max_positions, rotary_dim_}, DataType::Float32,
                                device)) {
  assert(device.type == DeviceType::Cuda);
}

void RotaryEmbedding::build_cache(cudaStream_t stream) {
  kernels::cuda::build_rope_cache(rope_cache_.view(), theta_, stream);
  cache_built_ = true;
}

void RotaryEmbedding::forward(ConstTensorView positions, TensorView query,
                              TensorView key, cudaStream_t stream) const {
  assert(cache_built_);
  kernels::cuda::apply_rotary_embedding(positions, query, key,
                                        rope_cache_.view(), stream);
}

} // namespace nanovllm
