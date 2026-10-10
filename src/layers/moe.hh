#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"
#include "layers/linear.hh"

namespace nanovllm {

struct MoEConfig {
  std::size_t hidden_size;
  std::size_t intermediate_size;
  std::size_t num_experts;
  std::size_t top_k;

  void validate() const;
};

// Keep intermediate results visible while learning and checking the math.
// Expert rows follow router rank, not expert ID. All outputs live on CUDA.
struct MoEOutput {
  Tensor hidden_states;  // Float32 [1, H], weighted sum of expert outputs.
  Tensor router_logits;  // Float32 [1, E].
  Tensor expert_ids;     // Int32 [1, K], descending score, lowest ID on ties.
  Tensor expert_weights; // Float32 [1, K], selected probabilities sum to 1.
  Tensor expert_outputs; // Float32 [K, H], before applying router weights.
};

// A small routed MoE: FP32, one token at a time, no shared expert yet.
// CPU construction is useful for initializing/reference-checking weights.
// forward requires CUDA device 0 and synchronizes stream before returning.
class MoE final : public Module {
 public:
  explicit MoE(MoEConfig config, Device device = Device::cuda(0));

  [[nodiscard]] MoEOutput forward(ConstTensorView hidden_states,
                                  cudaStream_t stream = nullptr) const;
  [[nodiscard]] const MoEConfig& config() const noexcept;

 private:
  MoEConfig config_;
  Linear gate_;
  ModuleList experts_;
};

} // namespace nanovllm
