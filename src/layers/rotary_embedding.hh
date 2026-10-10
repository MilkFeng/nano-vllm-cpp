#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor.hh"

namespace nanovllm {

// Owns the constant cosine/sine cache for full-head, unscaled RoPE.
// Q/K are rotated in place using pairs (i, i + head_dim/2).
// CPU construction is allowed for model inspection; forward requires CUDA.
class RotaryEmbedding {
 public:
  RotaryEmbedding(std::size_t max_positions, std::size_t head_dim, float theta,
                  Device device);

  void forward(ConstTensorView positions, TensorView query, TensorView key,
               cudaStream_t stream = nullptr) const;

 private:
  Tensor rope_cache_;
};

} // namespace nanovllm
