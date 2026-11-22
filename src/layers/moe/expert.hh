#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "core/module.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/linear.hh"
#include "layers/moe/workspace.hh"

namespace nanovllm {

class FP8Expert final : public Module {
 public:
  FP8Expert(std::size_t hidden_size, std::size_t intermediate_size,
            Device device = Device::cpu());

  void forward_into(ConstTensorView input, TensorView output) const;
  void forward_into(ConstTensorView input, TensorView output,
                    FP8ExpertWorkspace& workspace,
                    cudaStream_t stream = nullptr) const;

  [[nodiscard]] kernels::cuda::FP8ExpertWeights device_weights() const;

 private:
  FP8Linear gate_, up_, down_;
};

class ExpertStore final : public Module {
 public:
  ExpertStore(std::size_t hidden_size, std::size_t intermediate_size,
              std::size_t num_experts);

  [[nodiscard]] const FP8Expert& expert_at(std::size_t expert_id) const;

  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t intermediate_size() const noexcept;
  [[nodiscard]] std::size_t num_experts() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;

 private:
  std::size_t hidden_size_, intermediate_size_;
  std::vector<std::unique_ptr<FP8Expert>> experts_;
};

} // namespace nanovllm
