#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"

namespace nanovllm {

// GatedDeltaNet: RMSNorm(x) * weight * SiLU(gate), normalized over each value
// head. Weight is used directly, unlike Qwen3.5's zero-centered RMSNorm.
// This learning implementation computes in FP32; forward requires CUDA.
class GatedRMSNorm final : public Module {
 public:
  GatedRMSNorm(std::size_t normalized_size, float epsilon, Device device);
  [[nodiscard]] Tensor forward(ConstTensorView input, ConstTensorView gate,
                               cudaStream_t stream = nullptr) const;
  [[nodiscard]] Parameter& weight() noexcept { return weight_; }
  [[nodiscard]] const Parameter& weight() const noexcept { return weight_; }

 private:
  std::size_t normalized_size_;
  float epsilon_;
  Parameter weight_;
};

} // namespace nanovllm
