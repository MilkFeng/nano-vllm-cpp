#include "functions/functions.hh"

#include <cassert>
#include <cstring>

#include "common/cuda_utils.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::functions {

void zero_tensor(TensorView tensor, cudaStream_t stream) {
  const auto bytes = tensor.size_bytes();
  if (bytes == 0) {
    return;
  }

  assert(tensor.is_contiguous());

  const auto device = tensor.device();
  if (device.type == DeviceType::Cpu) {
    std::memset(tensor.data(), 0, bytes);
  } else {
    const CudaDeviceGuard guard(device.index);
    check_cuda(cudaMemsetAsync(tensor.data(), 0, bytes, stream), "zero tensor");
  }
}

Tensor cast_tensor(ConstTensorView input, DataType dtype, cudaStream_t stream) {
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              dtype, input.device());
  kernels::cuda::cast(input, output.view(), stream);
  return output;
}

Tensor sample(ConstTensorView logits, ConstTensorView temperatures,
              std::uint64_t seed, std::uint64_t step, cudaStream_t stream) {
  auto output =
      Tensor::empty({logits.shape()[0]}, DataType::Int32, logits.device());
  kernels::cuda::sample(logits, temperatures, output.view(), seed, step,
                        stream);
  return output;
}

} // namespace nanovllm::functions
