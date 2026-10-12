#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"

namespace nanovllm {

struct RMSNormOutput {
  Tensor hidden_states;
  Tensor residual;
};

class RMSNorm final : public Module {
 public:
  RMSNorm(std::size_t normalized_size, float epsilon, DataType dtype,
          Device device, bool zero_centered = false);

  [[nodiscard]] Tensor forward(ConstTensorView input,
                               cudaStream_t stream = nullptr) const;
  [[nodiscard]] RMSNormOutput forward(ConstTensorView input,
                                      ConstTensorView residual,
                                      cudaStream_t stream = nullptr) const;

  [[nodiscard]] std::size_t normalized_size() const noexcept;
  [[nodiscard]] float epsilon() const noexcept;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t normalized_size_;
  float epsilon_;
  bool zero_centered_;
  Parameter weight_;
};

} // namespace nanovllm
