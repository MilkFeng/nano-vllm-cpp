#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor.hh"

namespace nanovllm {

class RotaryEmbedding {
 public:
  RotaryEmbedding(std::size_t max_positions, std::size_t head_dim, float theta,
                  Device device, std::size_t rotary_dim = 0);

  void build_cache(cudaStream_t stream);

  void forward(ConstTensorView positions, TensorView query, TensorView key,
               cudaStream_t stream = nullptr) const;

 private:
  std::size_t head_dim_, rotary_dim_;
  float theta_;
  Tensor rope_cache_;
  bool cache_built_ = false;
};

} // namespace nanovllm
