#include "common/pinned_buffer.hh"

#include "common/cuda_utils.hh"

namespace nanovllm {

PinnedHostBuffer::~PinnedHostBuffer() noexcept {
  if (data_) cudaFreeHost(data_);
}

void PinnedHostBuffer::reserve(std::size_t bytes) {
  if (bytes <= capacity_) return;
  void* allocation = nullptr;
  check_cuda(cudaHostAlloc(&allocation, bytes,
                            mapped_ ? cudaHostAllocMapped : cudaHostAllocDefault),
             "allocate pinned host staging");
  void* device_allocation = nullptr;
  if (mapped_) {
    const auto status = cudaHostGetDevicePointer(&device_allocation, allocation, 0);
    if (status != cudaSuccess) {
      cudaFreeHost(allocation);
      check_cuda(status, "map pinned host staging");
    }
  }
  if (data_) cudaFreeHost(data_);
  data_ = allocation;
  capacity_ = bytes;
  device_data_ = device_allocation;
}

} // namespace nanovllm
