#include "layers/rotary_embedding.hh"

#include <cassert>
#include <cmath>

#include "common/cuda_utils.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {

RotaryEmbedding::RotaryEmbedding(std::size_t max_positions,
                                 std::size_t head_dim, float theta,
                                 Device device, std::size_t rotary_dim)
    : head_dim_(head_dim),
      rotary_dim_(rotary_dim == 0 ? head_dim : rotary_dim) {
  assert(max_positions > 0);
  assert(head_dim_ > 0);
  assert(rotary_dim_ > 0 && rotary_dim_ <= head_dim_ && rotary_dim_ % 2 == 0);
  assert(std::isfinite(theta) && theta > 0);
  // CPU construction remains useful for checkpoint inspection and loading.
  if (device.type != DeviceType::Cuda) {
    return;
  }
  rope_cache_ =
      Tensor::empty({max_positions, rotary_dim_}, DataType::Float32, device);
  const CudaDeviceGuard guard(device.index);
  kernels::cuda::build_rope_cache(rope_cache_.view(), theta);
  // Publish the cache to forwards on any stream, once during construction.
  check_cuda(cudaStreamSynchronize(nullptr), "initialize RoPE cache");
}

void RotaryEmbedding::forward(ConstTensorView positions, TensorView query,
                              TensorView key, cudaStream_t stream) const {
  assert(query.rank() == 3 && key.rank() == 3);
  assert(query.shape()[2] == head_dim_ && key.shape()[2] == head_dim_);
  kernels::cuda::apply_rotary_embedding(positions, query, key,
                                        rope_cache_.view(), stream);
}

} // namespace nanovllm
