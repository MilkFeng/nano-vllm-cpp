#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor.hh"

namespace nanovllm {

// Owns the constant cosine/sine cache for unscaled, optionally partial RoPE.
// Rotates pairs (i, i + rotary_dim/2) in the first rotary_dim head columns;
// the remaining columns are untouched. rotary_dim=0 means the full head.
// Text positions only; multimodal position axes are not implemented.
// CPU construction is allowed for model inspection; forward requires CUDA.
class RotaryEmbedding {
 public:
  RotaryEmbedding(std::size_t max_positions, std::size_t head_dim, float theta,
                  Device device, std::size_t rotary_dim = 0);

  void forward(ConstTensorView positions, TensorView query, TensorView key,
               cudaStream_t stream = nullptr) const;

 private:
  std::size_t head_dim_;
  std::size_t rotary_dim_;
  Tensor rope_cache_;
};

} // namespace nanovllm
