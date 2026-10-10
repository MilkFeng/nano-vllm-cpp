#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"
#include "layers/linear.hh"

namespace nanovllm {

// down_proj(SiLU(gate_proj(x)) * up_proj(x)).
class SwiGLUMLP final : public Module {
 public:
  SwiGLUMLP(std::size_t hidden_size, std::size_t intermediate_size,
            DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               cudaStream_t stream = nullptr) const;

 private:
  std::size_t hidden_size_;
  std::size_t intermediate_size_;

  // gate_proj and up_proj remain separate to match checkpoint names. They can
  // be presented as one packed matrix to a fused SwiGLU kernel later.
  Linear gate_proj_;
  Linear up_proj_;
  Linear down_proj_;
};

} // namespace nanovllm
