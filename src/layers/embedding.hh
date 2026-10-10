#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"

namespace nanovllm {

class Embedding final : public Module {
 public:
  Embedding(std::size_t vocab_size, std::size_t hidden_size, DataType dtype,
            Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               cudaStream_t stream = nullptr) const;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t vocab_size_;
  std::size_t hidden_size_;
  Parameter weight_;
};

} // namespace nanovllm
