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
  assert(device.type == DeviceType::Cuda);

  rope_cache_ =
      Tensor::empty({max_positions, rotary_dim_}, DataType::Float32, device);
  const CudaDeviceGuard guard(device.index);
  kernels::cuda::build_rope_cache(rope_cache_.view(), theta);
  check_cuda(cudaStreamSynchronize(nullptr), "initialize RoPE cache");
}

void RotaryEmbedding::forward(ConstTensorView positions, TensorView query,
                              TensorView key, cudaStream_t stream) const {
  kernels::cuda::apply_rotary_embedding(positions, query, key,
                                        rope_cache_.view(), stream);
}

} // namespace nanovllm
