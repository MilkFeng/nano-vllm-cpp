#include "layers/rotary_embedding.hh"

#include <cassert>
#include <cmath>

#include "common/cuda_utils.hh"
#include "kernels/kernels.hh"

namespace nanovllm {

RotaryEmbedding::RotaryEmbedding(std::size_t max_positions,
                                 std::size_t head_dim, float theta,
                                 Device device) {
  assert(max_positions > 0 && head_dim > 0 && head_dim % 2 == 0);
  assert(std::isfinite(theta) && theta > 0);
  // CPU construction remains useful for checkpoint inspection and loading.
  if (device.type != DeviceType::Cuda) return;
  rope_cache_ =
      Tensor::empty({max_positions, head_dim}, DataType::Float32, device);
  int previous_device = 0;
  check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice");
  check_cuda(cudaSetDevice(device.index), "cudaSetDevice");
  try {
    kernels::build_rope_cache(rope_cache_.view(), theta);
    // Publish fully initialized cache data to forwards on any CUDA stream.
    // This synchronization occurs once during construction, never in forward.
    check_cuda(cudaStreamSynchronize(nullptr), "initialize RoPE cache");
  } catch (...) {
    cudaSetDevice(previous_device);
    throw;
  }
  check_cuda(cudaSetDevice(previous_device), "restore CUDA device");
}

void RotaryEmbedding::forward(ConstTensorView positions, TensorView query,
                              TensorView key, cudaStream_t stream) const {
  kernels::apply_rotary_embedding(positions, query, key, rope_cache_.view(),
                                  stream);
}

} // namespace nanovllm
