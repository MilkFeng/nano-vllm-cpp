#include "layers/lm_head.hh"

#include <cassert>
#include <optional>

#include "kernels/cuda/inference.hh"

namespace nanovllm {

namespace {

Tensor make_lm_head_weight(std::size_t vocab_size, std::size_t hidden_size,
                           DataType dtype, Device device,
                           const Tensor* tied_weight) {
  if (tied_weight == nullptr) {
    return Tensor::empty({vocab_size, hidden_size}, dtype, device);
  }

  assert(tied_weight->shape().size() == 2);
  assert(tied_weight->shape()[0] == vocab_size
         && tied_weight->shape()[1] == hidden_size);
  assert(tied_weight->dtype() == dtype && tied_weight->device() == device);

  // Tensor copies share Storage, matching tie_word_embeddings semantics.
  return *tied_weight;
}

} // namespace

LMHead::LMHead(std::size_t vocab_size, std::size_t hidden_size, DataType dtype,
               Device device, const Tensor* tied_weight)
    : vocab_size_(vocab_size),
      hidden_size_(hidden_size),
      weight_(make_lm_head_weight(vocab_size_, hidden_size_, dtype, device,
                                  tied_weight)) {
  assert(vocab_size_ > 0 && hidden_size_ > 0);
  register_parameter("weight", weight_);
}

Tensor LMHead::forward(ConstTensorView hidden_states,
                       cudaStream_t stream) const {
  const auto weight = weight_.view();
  assert(hidden_states.rank() == 2 && hidden_states.shape()[1] == hidden_size_);
  assert(hidden_states.device().type == DeviceType::Cuda);
  assert(same_device(hidden_states, weight));
  assert(same_dtype(hidden_states, weight));
  auto logits = Tensor::empty({hidden_states.shape()[0], vocab_size_},
                              hidden_states.dtype(), hidden_states.device());
  kernels::cuda::linear(hidden_states, weight, logits.view(), std::nullopt,
                        stream);
  return logits;
}

Parameter& LMHead::weight() noexcept {
  return weight_;
}

const Parameter& LMHead::weight() const noexcept {
  return weight_;
}

} // namespace nanovllm
