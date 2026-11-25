#include "layers/lm_head.hh"

#include <cassert>
#include <optional>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

LMHead::LMHead(std::size_t vocab_size, std::size_t hidden_size, DataType dtype,
               Device device, const Tensor* tied_weight)
    : vocab_size_(vocab_size),
      hidden_size_(hidden_size),
      weight_(tied_weight == nullptr
                  ? Tensor::empty({vocab_size, hidden_size}, dtype, device)
                  : *tied_weight) {
  if (tied_weight == nullptr) {
    register_parameter("weight", weight_);
  }
}

Tensor LMHead::forward(ConstTensorView hidden_states, cudaStream_t stream,
                       std::optional<DataType> output_dtype) const {
  auto logits = Tensor::empty({hidden_states.shape()[0], vocab_size_},
                              output_dtype.value_or(hidden_states.dtype()),
                              hidden_states.device());
  kernels::cuda::linear(hidden_states, weight_.view(), logits.view(),
                        std::nullopt, stream);
  return logits;
}

Parameter& LMHead::weight() noexcept {
  return weight_;
}

const Parameter& LMHead::weight() const noexcept {
  return weight_;
}

} // namespace nanovllm
