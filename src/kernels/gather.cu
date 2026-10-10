#include <algorithm>
#include <cassert>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/common.cuh"
#include "kernels/inference.hh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

template <typename T>
__global__ void gather_kernel(const T* hidden, const std::int32_t* boundaries,
                              T* output, std::size_t count, std::size_t tokens,
                              std::size_t batch, std::size_t width) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (std::size_t index = start_index; index < count; index += step) {
    const auto row = index / width;
    const auto end = boundaries[row + 1];
    assert(boundaries[row] >= 0 && end > boundaries[row]
           && static_cast<std::size_t>(end) <= tokens);
    assert((row != 0 || boundaries[row] == 0)
           && (row + 1 != batch || end == tokens));

    output[index] =
        hidden[(static_cast<std::size_t>(end) - 1) * width + index % width];
  }
}

} // namespace

void gather_last_hidden(ConstTensorView hidden, ConstTensorView cu_seqlens_q,
                        TensorView output, cudaStream_t stream) {
  detail::assert_tensor(hidden);
  detail::assert_tensor(cu_seqlens_q);
  detail::assert_tensor(output);
  assert(same_dtype(hidden, output));
  assert(cu_seqlens_q.dtype() == DataType::Int32);
  assert(hidden.rank() == 2 && output.rank() == 2 && cu_seqlens_q.rank() == 1);
  const auto tokens = hidden.shape()[0], width = hidden.shape()[1];
  const auto batch = output.shape()[0];
  assert(width > 0 && output.shape()[1] == width);
  assert(cu_seqlens_q.shape()[0] == batch + 1 && batch <= tokens);
  assert(batch != 0 || tokens == 0);
  if (batch == 0) {
    return;
  }

  const auto dtype = hidden.dtype();
  const auto count = batch * width;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(dtype, [&]<typename T> {
    gather_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(hidden.data()),
        static_cast<const std::int32_t*>(cu_seqlens_q.data()),
        static_cast<T*>(output.data()), count, tokens, batch, width);
  });

  check_cuda(cudaGetLastError(), "gather_last_hidden launch");
}

} // namespace nanovllm::kernels
