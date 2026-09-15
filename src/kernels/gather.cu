#include "kernels/inference.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

#include "kernels/common.cuh"

namespace nanovllm::kernels {
namespace {

template <typename T>
__global__ void gather_kernel(const T* hidden, const std::int32_t* boundaries,
                              T* output, std::size_t count, std::size_t tokens,
                              std::size_t batch, std::size_t width,
                              std::size_t stride) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += std::size_t{blockDim.x} * gridDim.x) {
    const auto row = i / width;
    const auto begin = boundaries[row], end = boundaries[row + 1];
    const bool valid = begin >= 0 && end > begin
        && static_cast<std::size_t>(end) <= tokens
        && (row != 0 || begin == 0)
        && (row + 1 != batch || static_cast<std::size_t>(end) == tokens);
    assert(valid && "gather boundaries must partition nonempty query sequences");
    if (!valid) continue;
    output[i] = hidden[(static_cast<std::size_t>(end) - 1) * stride + i % width];
  }
}

} // namespace

void gather_last_hidden(ConstTensorView hidden, ConstTensorView cu_seqlens_q,
                        TensorView output, cudaStream_t stream) {
  detail::require_cuda(hidden);
  detail::require_cuda(cu_seqlens_q);
  detail::require_cuda(output.as_const());
  detail::require_float(hidden);
  detail::require_same_device_dtype(hidden, output.as_const());
  detail::require(cu_seqlens_q.device() == hidden.device()
                      && cu_seqlens_q.dtype() == DataType::Int32,
                  "gather boundaries must be Int32 on the hidden-state device");
  detail::require(hidden.rank() == 2 && output.rank() == 2
                      && cu_seqlens_q.rank() == 1,
                  "gather expects hidden[T,H], boundaries[B+1], output[B,H]");
  const auto tokens = hidden.shape()[0], width = hidden.shape()[1];
  const auto batch = output.shape()[0];
  detail::require(width > 0 && output.shape()[1] == width
                      && cu_seqlens_q.shape()[0] == detail::checked_add(batch, 1)
                      && (batch != 0 || tokens == 0) && batch <= tokens,
                  "gather dimensions must describe nonempty sequences partitioning hidden");
  detail::require(tokens <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
                  "gather token count must fit Int32 sequence boundaries");
  detail::require(hidden.strides().size() == 2 && hidden.strides()[1] == 1
                      && hidden.strides()[0] >= width
                      && cu_seqlens_q.is_contiguous() && output.is_contiguous(),
                  "gather requires contiguous hidden columns, boundaries, and output");
  detail::require(!detail::overlaps(hidden, output.as_const())
                      && !detail::overlaps(cu_seqlens_q, output.as_const()),
                  "gather output must not overlap inputs");
  if (batch == 0) return;
  const detail::DeviceGuard guard(hidden.device().index);
  const auto count = detail::checked_mul(batch, width);
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count - 1) / 256 + 1, 65535));
  detail::dispatch_float(hidden.dtype(), [&]<typename T> {
    gather_kernel<<<blocks, 256, 0, stream>>>(static_cast<const T*>(hidden.data()),
        static_cast<const std::int32_t*>(cu_seqlens_q.data()),
        static_cast<T*>(output.data()), count, tokens, batch, width, hidden.strides()[0]);
  });
  detail::check_cuda(cudaGetLastError(), "gather_last_hidden launch");
}

} // namespace nanovllm::kernels
