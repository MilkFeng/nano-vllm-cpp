#pragma once

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cpu {

struct ExpertWeights {
  ConstTensorView gate; // Float32 [I,H].
  ConstTensorView up;   // Float32 [I,H].
  ConstTensorView down; // Float32 [H,I].
};

// One CPU SwiGLU expert: down(silu(gate(x)) * up(x)). No router weighting.
// input/output: contiguous CPU Float32 [1,H]; weights have the shapes above.
// Scalar FP32 dot products, with temporary gate/up vectors allocated here.
void expert_forward(ConstTensorView input, const ExpertWeights& weights,
                    TensorView output);

} // namespace nanovllm::kernels::cpu
