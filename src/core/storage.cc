#include "core/storage.hh"

#include <format>
#include <stdexcept>

#include "common/cuda_utils.hh"

namespace nanovllm {

Storage::Storage(std::size_t size_bytes, Device device)
    : size_bytes_(size_bytes),
      device_(device) {
  switch (device_.type) {
    case DeviceType::Cpu:
      if (size_bytes_ != 0) {
        data_ = ::operator new(size_bytes_);
      }
      return;

    case DeviceType::Cuda: {
      if (size_bytes_ == 0) {
        return;
      }

      check_cuda(cudaMalloc(&data_, size_bytes_),
                 std::format("failed to allocate {} bytes on CUDA device 0",
                             size_bytes_));
      return;
    }
  }

  throw std::invalid_argument("unsupported tensor device type");
}

Storage::~Storage() noexcept {
  if (data_ == nullptr) {
    return;
  }

  switch (device_.type) {
    case DeviceType::Cpu:
      ::operator delete(data_);
      return;
    case DeviceType::Cuda:
      cudaFree(data_);
      return;
  }
}

void* Storage::data() noexcept {
  return data_;
}

const void* Storage::data() const noexcept {
  return data_;
}

std::size_t Storage::size_bytes() const noexcept {
  return size_bytes_;
}

Device Storage::device() const noexcept {
  return device_;
}

} // namespace nanovllm
