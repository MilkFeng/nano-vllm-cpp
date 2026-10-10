#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels/common.cuh"
#include "kernels/kernels.hh"

namespace nanovllm::kernels {
namespace {

template <typename T>
__global__ void silu_and_mul_kernel(const T* gate, const T* up, T* output,
                                    std::size_t count, std::size_t width,
                                    std::size_t gate_stride,
                                    std::size_t up_stride) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::size_t index = start_index; index < count; index += step) {
    const std::size_t row = index / width;
    const std::size_t column = index % width;
    const float g = detail::to_float(gate[row * gate_stride + column]);
    const float u = detail::to_float(up[row * up_stride + column]);
    output[index] = detail::from_float<T>((g / (1.0f + __expf(-g))) * u);
  }
}

} // namespace

void silu_and_mul(ConstTensorView gate, ConstTensorView up, TensorView output,
                  cudaStream_t stream) {
  detail::require_cuda(gate);
  detail::require_cuda(up);
  detail::require_cuda(output.as_const());
  detail::require_float(gate);
  detail::require_same_device_dtype(gate, up);
  detail::require_same_device_dtype(gate, output.as_const());
  detail::require(gate.rank() == 2 && up.rank() == 2 && output.rank() == 2,
                  "silu_and_mul expects rank-2 gate, up, and output tensors");
  detail::require(gate.shape()[0] == up.shape()[0]
                      && gate.shape()[1] == up.shape()[1]
                      && gate.shape()[0] == output.shape()[0]
                      && gate.shape()[1] == output.shape()[1],
                  "silu_and_mul requires matching gate, up, and output shapes");
  const std::size_t width = gate.shape()[1];
  detail::require(width > 0, "silu_and_mul requires a positive feature width");
  detail::require(
      gate.strides()[1] == 1 && up.strides()[1] == 1
          && gate.strides()[0] >= width && up.strides()[0] >= width,
      "silu_and_mul requires unit feature strides and nonoverlapping rows");
  detail::require(output.is_contiguous(),
                  "silu_and_mul requires contiguous output");
  detail::require_exact_or_disjoint(
      gate, output.as_const(),
      "silu_and_mul permits only exact contiguous gate/output aliases");
  detail::require_exact_or_disjoint(
      up, output.as_const(),
      "silu_and_mul permits only exact contiguous up/output aliases");
  const std::size_t count = output.numel();
  if (count == 0) {
    return;
  }

  const detail::DeviceGuard guard(gate.device().index);
  constexpr std::uint32_t threads = 256;
  const auto blocks = static_cast<std::uint32_t>(
      std::min<std::size_t>((count - 1) / threads + 1, 65535));

  detail::dispatch_float(gate.dtype(), [&]<typename T>() {
    silu_and_mul_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(gate.data()), static_cast<const T*>(up.data()),
        static_cast<T*>(output.data()), count, width, gate.strides()[0],
        up.strides()[0]);
  });

  detail::check_cuda(cudaGetLastError(), "silu_and_mul kernel launch");
}

} // namespace nanovllm::kernels
