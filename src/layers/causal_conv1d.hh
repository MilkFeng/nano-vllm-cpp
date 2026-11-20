#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"

namespace nanovllm {

// Causal Depthwise Convolution
//
// 对于一个 token 的一个 channel，从 state_indices 中拿到槽位 id，然后从 conv
// 中取出历史的 kernel_size-1 个原始输入，然后一共 kernel_size
// 个输入做卷积，然后应用 SiLU
class CausalConv1d final : public Module {
 public:
  CausalConv1d(std::size_t channels, std::size_t kernel_size, DataType dtype,
               Device device);

  // input/output:  [tokens, channels]
  // state:         [slots, channels, kernel_size-1]
  //
  // Empty state_indices maps sequence i to state slot i.
  [[nodiscard]] Tensor forward(ConstTensorView input, TensorView state,
                               ConstTensorView cu_seqlens = {},
                               ConstTensorView state_indices = {},
                               cudaStream_t stream = nullptr) const;

  [[nodiscard]] std::size_t channels() const noexcept;
  [[nodiscard]] std::size_t kernel_size() const noexcept;

  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t channels_, kernel_size_;

  // weight_: [channels, 1, kernel_size]
  Parameter weight_;
};

} // namespace nanovllm
