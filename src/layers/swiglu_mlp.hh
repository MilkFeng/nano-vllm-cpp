#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"
#include "layers/linear.hh"

namespace nanovllm {

class SwiGLUMLP final : public Module {
 public:
  SwiGLUMLP(std::size_t hidden_size, std::size_t intermediate_size,
            DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               cudaStream_t stream = nullptr) const;

 private:
  std::size_t hidden_size_, intermediate_size_;
  Linear gate_proj_, up_proj_, down_proj_;
};

class FP8SwiGLUMLP final : public Module {
 public:
  FP8SwiGLUMLP(std::size_t hidden_size, std::size_t intermediate_size,
               Device device);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               cudaStream_t stream = nullptr) const;

 private:
  std::size_t intermediate_size_;
  FP8Linear gate_proj_, up_proj_, down_proj_;
};

} // namespace nanovllm
