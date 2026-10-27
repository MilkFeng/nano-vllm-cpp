#include "common/pinned_buffer.hh"

#include "common/cuda_utils.hh"

namespace nanovllm {

PinnedHostBuffer::~PinnedHostBuffer() noexcept {
  if (data_) cudaFreeHost(data_);
}

void PinnedHostBuffer::reserve(std::size_t bytes) {
  if (bytes <= capacity_) return;
  void* allocation = nullptr;
  check_cuda(cudaMallocHost(&allocation, bytes), "allocate pinned host staging");
  if (data_) cudaFreeHost(data_);
  data_ = allocation;
  capacity_ = bytes;
}

} // namespace nanovllm
