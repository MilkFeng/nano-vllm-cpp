#include <algorithm>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

template <typename T>
__global__ void split_kernel(const T* packed, T* query, T* gate,
                             std::size_t count, std::size_t head_dim) {
  const auto start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (auto index = start_index; index < count; index += step) {
    const auto offset = (index / head_dim) * (2 * head_dim) + index % head_dim;
    query[index] = packed[offset];
    gate[index] = packed[offset + head_dim];
  }
}

template <typename T>
__global__ void gate_kernel(T* output, const T* gate, std::size_t count) {
  const auto start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (auto index = start_index; index < count; index += step) {
    output[index] = detail::from_float<T>(
        detail::to_float(output[index])
        / (1.0f + __expf(-detail::to_float(gate[index]))));
  }
}

} // namespace

void split_attention_gate(ConstTensorView packed, TensorView query,
                          TensorView gate, cudaStream_t stream) {
  detail::assert_tensor(packed);
  detail::assert_tensor(query);
  detail::assert_tensor(gate);

  assert(packed.rank() == 2 && query.rank() == 3 && same_shape(query, gate));
  assert(packed.shape()[0] == query.shape()[0]
         && packed.shape()[1] == 2 * query.shape()[1] * query.shape()[2]);
  assert(same_dtype(packed, query, gate));
  assert(same_device(packed, query, gate));

  const auto count = query.numel();
  if (count == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(packed.dtype(), [&]<typename T>() {
    split_kernel<<<blocks, kThreads, 0, stream>>>(
        packed.data<T>(), query.data<T>(), gate.data<T>(), count,
        query.shape()[2]);
  });

  check_cuda(cudaGetLastError(), "split_attention_gate launch");
}

void apply_sigmoid_gate(TensorView output, ConstTensorView gate,
                        cudaStream_t stream) {
  detail::assert_tensor(output);
  detail::assert_tensor(gate);

  assert(same_dtype(output, gate));
  assert(same_shape(output, gate) && same_device(output, gate));

  const auto count = output.numel();
  if (count == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  detail::dispatch_float(output.dtype(), [&]<typename T>() {
    gate_kernel<<<blocks, kThreads, 0, stream>>>(output.data<T>(),
                                                 gate.data<T>(), count);
  });

  check_cuda(cudaGetLastError(), "apply_sigmoid_gate launch");
}
} // namespace nanovllm::kernels::cuda
