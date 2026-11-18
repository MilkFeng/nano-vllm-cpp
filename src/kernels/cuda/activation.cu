#include <algorithm>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

template <typename T, typename U>
__global__ void cast_kernel(const T* input, U* output, std::size_t count) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    output[i] = detail::from_float<U>(detail::to_float(input[i]));
  }
}

template <typename T>
__global__ void silu_and_mul_kernel(const T* __restrict__ gate,
                                    const T* __restrict__ up,
                                    T* __restrict__ output, std::size_t count) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::size_t index = start_index; index < count; index += step) {
    const float g = detail::to_float(gate[index]);
    const float u = detail::to_float(up[index]);
    output[index] = detail::from_float<T>((g / (1.0f + __expf(-g))) * u);
  }
}

} // namespace

void cast(ConstTensorView input, TensorView output, cudaStream_t stream) {
  detail::assert_tensor(input);
  detail::assert_tensor(output);
  assert(same_shape(input, output) && same_device(input, output));
  if (input.numel() == 0) {
    return;
  }
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((input.numel() + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    detail::dispatch_float(output.dtype(), [&]<typename U>() {
      cast_kernel<<<blocks, kThreads, 0, stream>>>(
          input.data<T>(), output.data<U>(), input.numel());
    });
  });
  check_cuda(cudaGetLastError(), "cast kernel launch");
}

void silu_and_mul(ConstTensorView gate, ConstTensorView up, TensorView output,
                  cudaStream_t stream) {
  detail::assert_tensor(gate);
  detail::assert_tensor(up);
  detail::assert_tensor(output);
  assert(gate.rank() == 2 && gate.shape()[1] > 0);
  assert(same_shape(gate, up, output));
  assert(same_dtype(gate, up, output));
  const std::size_t count = output.numel();
  if (count == 0) {
    return;
  }

  const auto dtype = gate.dtype();
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(dtype, [&]<typename T>() {
    silu_and_mul_kernel<<<blocks, kThreads, 0, stream>>>(
        gate.data<T>(), up.data<T>(), output.data<T>(), count);
  });

  check_cuda(cudaGetLastError(), "silu_and_mul kernel launch");
}

} // namespace nanovllm::kernels::cuda
