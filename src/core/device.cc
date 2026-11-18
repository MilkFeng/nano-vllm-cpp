#include "core/device.hh"

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

Device::Device(DeviceType type, int index)
    : type(type),
      index(index) {
  if (!is_valid()) {
    throw std::invalid_argument("invalid tensor device");
  }
}

Device Device::cpu() noexcept {
  return {};
}

Device Device::cuda(int index) {
  return Device(DeviceType::Cuda, index);
}

bool Device::is_valid() const {
  switch (type) {
    case DeviceType::Cpu:
      return index == 0;

    case DeviceType::Cuda: {
      if (index < 0) {
        return false;
      }

      int device_count = 0;
      check_cuda(cudaGetDeviceCount(&device_count),
                 "failed to query CUDA devices");

      return index < device_count;
    }
  }

  return false;
}

} // namespace nanovllm
