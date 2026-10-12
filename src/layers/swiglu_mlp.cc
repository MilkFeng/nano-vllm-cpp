#include "layers/swiglu_mlp.hh"

#include <cassert>

#include "kernels/cuda/inference.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {

SwiGLUMLP::SwiGLUMLP(std::size_t hidden_size, std::size_t intermediate_size,
                     DataType dtype, Device device)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      gate_proj_(hidden_size_, intermediate_size_, false, dtype, device),
      up_proj_(hidden_size_, intermediate_size_, false, dtype, device),
      down_proj_(intermediate_size_, hidden_size_, false, dtype, device) {
  register_module("gate_proj", gate_proj_);
  register_module("up_proj", up_proj_);
  register_module("down_proj", down_proj_);
}

Tensor SwiGLUMLP::forward(ConstTensorView hidden_states,
                          cudaStream_t stream) const {
  auto gate = gate_proj_.forward(hidden_states, stream);
  auto up = up_proj_.forward(hidden_states, stream);
  auto activated = Tensor::empty({hidden_states.shape()[0], intermediate_size_},
                                 hidden_states.dtype(), hidden_states.device());
  kernels::cuda::silu_and_mul(gate.view().as_const(), up.view().as_const(),
                              activated.view(), stream);
  return down_proj_.forward(activated.view().as_const(), stream);
}

void SwiGLUMLP::forward_into(ConstTensorView hidden_states, TensorView output,
                             SwiGLUWorkspace& workspace,
                             cudaStream_t stream) const {
  const auto weight = gate_proj_.weight().view();

  auto gate = workspace.gate.view();
  auto up = workspace.up.view();
  auto activated = workspace.activated.view();

  kernels::cuda::linear(hidden_states, weight, gate, std::nullopt, stream);
  kernels::cuda::linear(hidden_states, up_proj_.weight().view(), up,
                        std::nullopt, stream);
  kernels::cuda::silu_and_mul(gate.as_const(), up.as_const(), activated,
                              stream);
  kernels::cuda::linear(activated.as_const(), down_proj_.weight().view(),
                        output, std::nullopt, stream);
}

} // namespace nanovllm
