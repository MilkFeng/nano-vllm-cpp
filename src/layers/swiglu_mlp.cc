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
  assert(hidden_states.rank() == 2);
  assert(hidden_states.shape()[0] > 0);
  assert(hidden_states.shape()[1] == hidden_size_);
  assert(hidden_states.device().type == DeviceType::Cuda);
  assert(same_device(hidden_states, weight));
  assert(same_dtype(hidden_states, weight));
  assert(hidden_states.is_contiguous());
  const auto validate = [&]([[maybe_unused]] TensorView buffer,
                            [[maybe_unused]] std::size_t width) {
    assert(buffer.rank() == 2);
    assert(buffer.shape()[0] == hidden_states.shape()[0]);
    assert(buffer.shape()[1] == width);
    assert(buffer.is_contiguous());
    assert(same_device(buffer, hidden_states));
    assert(same_dtype(buffer, hidden_states));
  };
  auto gate = workspace.gate.view();
  auto up = workspace.up.view();
  auto activated = workspace.activated.view();
  validate(output, hidden_size_);
  validate(gate, intermediate_size_);
  validate(up, intermediate_size_);
  validate(activated, intermediate_size_);

  kernels::cuda::linear(hidden_states, weight, gate, std::nullopt, stream);
  kernels::cuda::linear(hidden_states, up_proj_.weight().view(), up,
                        std::nullopt, stream);
  kernels::cuda::silu_and_mul(gate.as_const(), up.as_const(), activated,
                              stream);
  kernels::cuda::linear(activated.as_const(), down_proj_.weight().view(),
                        output, std::nullopt, stream);
}

} // namespace nanovllm
