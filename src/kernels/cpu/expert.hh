#pragma once

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cpu {

struct ExpertWeights {
  ConstTensorView gate;
  ConstTensorView up;
  ConstTensorView down;
};

void expert_forward(ConstTensorView input, const ExpertWeights& weights,
                    TensorView output);

} // namespace nanovllm::kernels::cpu
