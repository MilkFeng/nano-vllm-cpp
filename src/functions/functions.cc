#include "functions/functions.hh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "common/cuda_utils.hh"
#include "core/device.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::functions {
namespace {

template <typename Function>
void dispatch_cast_type(DataType dtype, const Function& function) {
  switch (dtype) {
    case DataType::Float32:
      function.template operator()<float>();
      return;
    case DataType::Float16:
      function.template operator()<__half>();
      return;
    case DataType::BFloat16:
      function.template operator()<__nv_bfloat16>();
      return;
    default:
      throw std::invalid_argument("unsupported CPU cast dtype");
  }
}

void cast_cpu(ConstTensorView input, TensorView output) {
  if (!input.is_contiguous()) {
    throw std::invalid_argument("CPU cast requires contiguous input");
  }
  const auto count = input.numel();
  if (count == 0) {
    return;
  }

  dispatch_cast_type(input.dtype(), [&]<typename Input>() {
    dispatch_cast_type(output.dtype(), [&]<typename Output>() {
      const auto* source = input.data<Input>();
      auto* destination = output.data<Output>();
      for (std::size_t i = 0; i < count; ++i) {
        destination[i] = static_cast<Output>(static_cast<float>(source[i]));
      }
    });
  });
}

} // namespace

void zero_tensor(TensorView tensor, cudaStream_t stream) {
  const auto bytes = tensor.size_bytes();
  if (bytes == 0) {
    return;
  }

  assert(tensor.is_contiguous());

  if (tensor.device().type == DeviceType::Cpu) {
    std::memset(tensor.data(), 0, bytes);
  } else {
    check_cuda(cudaMemsetAsync(tensor.data(), 0, bytes, stream), "zero tensor");
  }
}

Tensor cast_tensor(ConstTensorView input, DataType dtype, cudaStream_t stream) {
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              dtype, input.device());
  if (input.device().type == DeviceType::Cpu) {
    cast_cpu(input, output.view());
  } else {
    kernels::cuda::cast(input, output.view(), stream);
  }
  return output;
}

Tensor sample(ConstTensorView logits, ConstTensorView temperatures,
              std::uint64_t seed, std::uint64_t step, cudaStream_t stream) {
  assert(same_device(logits, temperatures));
  assert(logits.device().type == DeviceType::Cuda);

  auto output =
      Tensor::empty({logits.shape()[0]}, DataType::Int32, logits.device());
  kernels::cuda::sample(logits, temperatures, output.view(), seed, step,
                        stream);
  return output;
}

} // namespace nanovllm::functions
