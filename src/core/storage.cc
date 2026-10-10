#include "core/storage.hh"

#include <cuda_runtime_api.h>

#include <format>
#include <stdexcept>

namespace nanovllm {

namespace {
class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int index) {
    const auto get_status = cudaGetDevice(&previous_device_);
    if (get_status != cudaSuccess) {
      throw std::runtime_error(
          std::format("failed to get the current CUDA device: {}",
                      cudaGetErrorString(get_status)));
    }
    if (previous_device_ == index) return;

    const auto set_status = cudaSetDevice(index);
    if (set_status != cudaSuccess) {
      throw std::runtime_error(
          std::format("failed to select CUDA device {}: {}", index,
                      cudaGetErrorString(set_status)));
    }
    restore_ = true;
  }

  ~CudaDeviceGuard() {
    if (restore_) {
      cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_ = false;
};
} // namespace

Storage::Storage(std::size_t size_bytes, Device device)
    : size_bytes_(size_bytes),
      device_(device) {
  if (!device_.is_valid()) {
    throw std::invalid_argument("tensor device is invalid");
  }

  switch (device_.type) {
    case DeviceType::Cpu:
      if (size_bytes_ != 0) data_ = ::operator new(size_bytes_);
      return;

    case DeviceType::Cuda: {
      if (size_bytes_ == 0) return;

      CudaDeviceGuard guard(device_.index);
      const auto status = cudaMalloc(&data_, size_bytes_);
      if (status != cudaSuccess) {
        throw std::runtime_error(
            std::format("failed to allocate {} bytes on CUDA device {}",
                        size_bytes_, device_.index));
      }
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
    case DeviceType::Cuda: {
      int previous_device = 0;
      const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
      const bool restore = have_previous && previous_device != device_.index
                           && cudaSetDevice(device_.index) == cudaSuccess;
      cudaFree(data_);
      if (restore) {
        cudaSetDevice(previous_device);
      }
      return;
    }
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
