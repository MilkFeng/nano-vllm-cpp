#include "kernels/kernels.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "kernels/common.cuh"

namespace nanovllm::kernels {
namespace {

template <typename T>
__global__ void embedding_kernel(const std::int32_t* input_ids, const T* weight,
                                 T* output, std::size_t count,
                                 std::size_t vocab_size,
                                 std::size_t hidden_size) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
       index < count; index += step) {
    const std::int32_t token = input_ids[index / hidden_size];
    const bool valid = token >= 0 && static_cast<std::size_t>(token) < vocab_size;
    assert(valid && "embedding token ID is outside the vocabulary");
    // Keep the bounds check in release builds, where assert may be disabled.
    if (!valid) continue;
    output[index] = weight[static_cast<std::size_t>(token) * hidden_size
                           + index % hidden_size];
  }
}

} // namespace

void embedding(ConstTensorView input_ids, ConstTensorView weight,
               TensorView output, cudaStream_t stream) {
  detail::require_cuda(input_ids);
  detail::require_cuda(weight);
  detail::require_cuda(output.as_const());
  detail::require(input_ids.dtype() == DataType::Int32,
                  "embedding requires Int32 token IDs");
  detail::require_float(weight);
  detail::require_same_device_dtype(weight, output.as_const());
  detail::require(input_ids.device() == weight.device(),
                  "embedding requires token IDs and weights on the same CUDA device");
  detail::require(input_ids.rank() == 1 && weight.rank() == 2 && output.rank() == 2,
                  "embedding expects rank-1 IDs and rank-2 weights and output");
  detail::require(input_ids.is_contiguous() && weight.is_contiguous()
                      && output.is_contiguous(),
                  "embedding requires contiguous IDs, weights, and output");
  detail::require(weight.shape()[0] > 0 && weight.shape()[1] > 0,
                  "embedding requires positive vocabulary and hidden sizes");
  detail::require(output.shape()[0] == input_ids.shape()[0]
                      && output.shape()[1] == weight.shape()[1],
                  "embedding output shape must be [num_tokens, hidden_size]");
  detail::require(!detail::overlaps(input_ids, output.as_const())
                      && !detail::overlaps(weight, output.as_const()),
                  "embedding output must not overlap token IDs or weights");
  const std::size_t count = output.numel();
  if (count == 0) return;

  const detail::DeviceGuard guard(weight.device().index);
  constexpr unsigned int threads = 256;
  const auto blocks = static_cast<unsigned int>(
      std::min<std::size_t>((count - 1) / threads + 1, 65535));
  detail::dispatch_float(weight.dtype(), [&]<typename T>() {
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const std::int32_t*>(input_ids.data()),
        static_cast<const T*>(weight.data()), static_cast<T*>(output.data()),
        count, weight.shape()[0], weight.shape()[1]);
  });
  detail::check_cuda(cudaGetLastError(), "embedding kernel launch");
}

} // namespace nanovllm::kernels
