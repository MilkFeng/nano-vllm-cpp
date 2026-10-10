#include <algorithm>

#include "common/cuda_utils.hh"
#include "kernels/common.cuh"
#include "kernels/kernels.hh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

template <typename T>
__global__ void silu_and_mul_kernel(const T* gate, const T* up, T* output,
                                    std::size_t count) {
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

void silu_and_mul(ConstTensorView gate, ConstTensorView up, TensorView output,
                  cudaStream_t stream) {
  detail::assert_tensor(gate);
  detail::assert_tensor(up);
  detail::assert_tensor(output);
  assert(gate.rank() == 2 && gate.shape()[1] > 0);
  assert(detail::same_shape(gate, up) && detail::same_shape(gate, output));
  assert(gate.dtype() == up.dtype() && gate.dtype() == output.dtype());
  const std::size_t count = output.numel();
  if (count == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(gate.dtype(), [&]<typename T>() {
    silu_and_mul_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(gate.data()), static_cast<const T*>(up.data()),
        static_cast<T*>(output.data()), count);
  });

  check_cuda(cudaGetLastError(), "silu_and_mul kernel launch");
}

} // namespace nanovllm::kernels
