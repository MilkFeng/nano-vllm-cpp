#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cpu {

struct ExpertWeights {
  ConstTensorView gate; // Float32 [I,H].
  ConstTensorView up;   // Float32 [I,H].
  ConstTensorView down; // Float32 [H,I].
};

struct MoEReferenceOutput {
  std::vector<float> hidden_states;
  std::vector<float> router_logits;
  std::vector<std::int32_t> expert_ids;
  std::vector<float> expert_weights;
  std::vector<float> expert_outputs; // [K,H], in router rank order.
};

// Independent, scalar reference for one token. All views are contiguous CPU
// Float32 tensors: input [1,H], router_weight [E,H], expert matrices above.
// Dot products accumulate in double, with FP32 intermediate/output storage.
[[nodiscard]] MoEReferenceOutput moe_reference(
    ConstTensorView input, ConstTensorView router_weight,
    std::span<const ExpertWeights> experts, std::size_t top_k);

} // namespace nanovllm::kernels::cpu
