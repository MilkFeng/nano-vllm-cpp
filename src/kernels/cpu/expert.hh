#pragma once

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cpu {

struct ExpertWeights {
  ConstTensorView gate;
  ConstTensorView up;
  ConstTensorView down;
  ConstTensorView gate_scale;
  ConstTensorView up_scale;
  ConstTensorView down_scale;
};

// BF16 or FP32 input [rows, H], FP32 output, with one expert's FP8 weights.
void expert_forward(ConstTensorView input, const ExpertWeights& weights,
                    TensorView output, std::size_t block_size);

} // namespace nanovllm::kernels::cpu
