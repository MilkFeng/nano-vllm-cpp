#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>

#include "core/module.hh"

namespace nanovllm {

class Linear final : public Module {
 public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias,
         DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input,
                               cudaStream_t stream = nullptr) const;

  [[nodiscard]] std::size_t in_features() const noexcept;
  [[nodiscard]] std::size_t out_features() const noexcept;

  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

  [[nodiscard]] Parameter* bias() noexcept;
  [[nodiscard]] const Parameter* bias() const noexcept;

  void use_fp8_weight();

 private:
  std::size_t in_features_, out_features_;

  // weight_: [out_features, in_features]
  Parameter weight_;

  std::optional<Parameter> bias_;
  std::optional<Parameter> weight_scale_inv_;
};

} // namespace nanovllm
