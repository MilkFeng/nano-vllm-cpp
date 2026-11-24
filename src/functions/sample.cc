#include "functions/sample.hh"

#include "kernels/cuda/kernels.hh"

namespace nanovllm {

Tensor sample(ConstTensorView logits, ConstTensorView temperatures,
              std::uint64_t seed, std::uint64_t step, cudaStream_t stream) {
  auto output =
      Tensor::empty({logits.shape()[0]}, DataType::Int32, logits.device());
  kernels::cuda::sample(logits, temperatures, output.view(), seed, step,
                        stream);
  return output;
}

} // namespace nanovllm
