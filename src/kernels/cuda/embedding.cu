#include <algorithm>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

template <typename T>
__global__ void embedding_kernel(const std::int32_t* __restrict__ input_ids,
                                 const T* __restrict__ weight,
                                 T* __restrict__ output, std::size_t count,
                                 std::size_t vocab_size,
                                 std::size_t hidden_size) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::size_t index = start_index; index < count; index += step) {
    const std::int32_t token = input_ids[index / hidden_size];
    assert(token >= 0 && static_cast<std::size_t>(token) < vocab_size);
    output[index] = weight[static_cast<std::size_t>(token) * hidden_size
                           + index % hidden_size];
  }
}

} // namespace

void embedding(ConstTensorView input_ids, ConstTensorView weight,
               TensorView output, cudaStream_t stream) {
  detail::assert_tensor(input_ids);
  detail::assert_tensor(weight);
  detail::assert_tensor(output);
  assert(input_ids.dtype() == DataType::Int32);
  assert(same_dtype(weight, output));
  assert(input_ids.rank() == 1 && weight.rank() == 2 && output.rank() == 2);
  assert(weight.shape()[0] > 0 && weight.shape()[1] > 0);
  assert(output.shape()[0] == input_ids.shape()[0]
         && output.shape()[1] == weight.shape()[1]);
  const std::size_t count = output.numel();
  if (count == 0) {
    return;
  }

  const auto dtype = weight.dtype();
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(dtype, [&]<typename T>() {
    embedding_kernel<<<blocks, kThreads, 0, stream>>>(
        input_ids.data<std::int32_t>(), weight.data<T>(), output.data<T>(),
        count, weight.shape()[0], weight.shape()[1]);
  });

  check_cuda(cudaGetLastError(), "embedding kernel launch");
}

} // namespace nanovllm::kernels::cuda
