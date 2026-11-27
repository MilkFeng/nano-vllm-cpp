#include "core/device.hh"

#include <cassert>
#include <stdexcept>

#include "common/cuda_utils.hh"

namespace nanovllm {
std::size_t data_type_size(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
    case DataType::Int32:
      return 4;
    case DataType::Float16:
    case DataType::BFloat16:
      return 2;
    case DataType::Float8E4M3FN:
      return 1;
  }
  throw std::invalid_argument("unknown tensor data type");
}

Device::Device(DeviceType type)
    : type(type) {
  assert(is_valid());
}

Device Device::cpu() noexcept {
  return Device(DeviceType::Cpu);
}

Device Device::cuda() {
  return Device(DeviceType::Cuda);
}

bool Device::is_valid() const {
  switch (type) {
    case DeviceType::Cpu:
      return true;

    case DeviceType::Cuda: {
      int device_count = 0;
      check_cuda(cudaGetDeviceCount(&device_count),
                 "failed to query CUDA devices");

      return device_count > 0;
    }
  }

  return false;
}

} // namespace nanovllm
