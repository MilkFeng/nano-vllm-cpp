#include <algorithm>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

__global__ void split_kernel(const float* packed, float* query, float* gate,
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

__global__ void gate_kernel(float* output, const float* gate,
                            std::size_t count) {
  const auto start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (auto index = start_index; index < count; index += step) {
    output[index] *= 1.0f / (1.0f + __expf(-gate[index]));
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
  assert(packed.dtype() == DataType::Float32
         && same_dtype(packed, query, gate));
  assert(same_device(packed, query, gate));

  const auto count = query.numel();
  if (count == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  split_kernel<<<blocks, kThreads, 0, stream>>>(
      packed.data<float>(), query.data<float>(), gate.data<float>(), count,
      query.shape()[2]);

  check_cuda(cudaGetLastError(), "split_attention_gate launch");
}

void apply_sigmoid_gate(TensorView output, ConstTensorView gate,
                        cudaStream_t stream) {
  detail::assert_tensor(output);
  detail::assert_tensor(gate);

  assert(output.dtype() == DataType::Float32 && same_dtype(output, gate));
  assert(same_shape(output, gate) && same_device(output, gate));

  const auto count = output.numel();
  if (count == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
  gate_kernel<<<blocks, kThreads, 0, stream>>>(output.data<float>(),
                                               gate.data<float>(), count);

  check_cuda(cudaGetLastError(), "apply_sigmoid_gate launch");
}
} // namespace nanovllm::kernels::cuda
