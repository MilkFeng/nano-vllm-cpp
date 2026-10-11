#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/tensor_view.hh"
#include "kernels/cpu/expert.hh"

namespace nanovllm::kernels::cpu {

struct SharedExpertWeights {
  ExpertWeights expert;        // SwiGLU with its own intermediate dimension.
  ConstTensorView gate_weight; // Float32 [1,H], no bias.
};

struct MoEReferenceOutput {
  std::vector<float> hidden_states;
  std::vector<float> router_logits;
  std::vector<std::int32_t> expert_ids;
  std::vector<float> expert_weights;
  std::vector<float> expert_outputs; // [K,H], in router rank order.
  // Shared intermediates are empty when the branch is disabled.
  std::vector<float> shared_expert_output; // Raw [1,H].
  std::vector<float> shared_gate_logits;   // [1,1], before sigmoid.
};

// Independent, scalar reference for one token. All views are contiguous CPU
// Float32 tensors: input [1,H], router_weight [E,H], expert matrices above.
// Dot products accumulate in double, with FP32 intermediate/output storage.
// The optional shared branch adds sigmoid(gate_weight * input) * shared SwiGLU.
[[nodiscard]] MoEReferenceOutput moe_reference(
    ConstTensorView input, ConstTensorView router_weight,
    std::span<const ExpertWeights> experts, std::size_t top_k,
    std::optional<SharedExpertWeights> shared_expert = std::nullopt);

} // namespace nanovllm::kernels::cpu
