#include "layers/embedding.hh"

#include <cassert>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

Embedding::Embedding(std::size_t vocab_size, std::size_t hidden_size,
                     DataType dtype, Device device)
    : vocab_size_(vocab_size),
      hidden_size_(hidden_size),
      weight_(Tensor::empty({vocab_size_, hidden_size_}, dtype, device)) {
  register_parameter("weight", weight_);
}

Tensor Embedding::forward(ConstTensorView input_ids,
                          cudaStream_t stream) const {
  const auto weight = weight_.view();
  auto output = Tensor::empty({input_ids.shape()[0], hidden_size_},
                              weight.dtype(), weight.device());
  kernels::cuda::embedding(input_ids, weight, output.view(), stream);
  return output;
}

Parameter& Embedding::weight() noexcept {
  return weight_;
}

const Parameter& Embedding::weight() const noexcept {
  return weight_;
}

} // namespace nanovllm
