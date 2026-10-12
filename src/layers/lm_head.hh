#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"

namespace nanovllm {

class LMHead final : public Module {
 public:
  LMHead(std::size_t vocab_size, std::size_t hidden_size, DataType dtype,
         Device device, const Tensor* tied_weight = nullptr);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               cudaStream_t stream = nullptr) const;

  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t vocab_size_;
  std::size_t hidden_size_;
  Parameter weight_;
};

} // namespace nanovllm
