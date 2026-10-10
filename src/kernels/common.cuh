#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "model/tensor_view.hh"

namespace nanovllm::kernels::detail {

inline void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) {
      check_cuda(cudaSetDevice(device), "cudaSetDevice");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      cudaSetDevice(previous_);
    }
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
  bool restore_ = false;
};

inline void require_cuda(ConstTensorView tensor) {
  require(
      tensor.device().type == DeviceType::Cuda && tensor.device().index >= 0,
      "kernel tensors must be on a CUDA device");
  require(tensor.numel() == 0 || tensor.data() != nullptr,
          "a non-empty kernel tensor requires data");
}

inline void require_float(ConstTensorView tensor) {
  require(tensor.dtype() == DataType::Float32
              || tensor.dtype() == DataType::Float16
              || tensor.dtype() == DataType::BFloat16,
          "kernel supports only Float32, Float16 and BFloat16");
}

inline void require_same_device_dtype(ConstTensorView lhs,
                                      ConstTensorView rhs) {
  require(lhs.device() == rhs.device(), "kernel tensor devices must match");
  require(lhs.dtype() == rhs.dtype(), "kernel tensor dtypes must match");
}

inline std::size_t checked_add(std::size_t a, std::size_t b) {
  require(b <= std::numeric_limits<std::size_t>::max() - a,
          "kernel tensor size or address range overflows");
  return a + b;
}

inline std::size_t checked_mul(std::size_t a, std::size_t b) {
  require(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a,
          "kernel tensor size or address range overflows");
  return a * b;
}

// A conservative storage span includes holes introduced by strided slices.
// Read-only tensors may overlap; each kernel checks only writable aliases.
inline std::size_t storage_span(ConstTensorView tensor) {
  require(tensor.strides().size() == tensor.rank(),
          "kernel tensor shape and strides must have equal rank");
  if (std::ranges::find(tensor.shape(), std::size_t{0})
      != tensor.shape().end()) {
    return 0;
  }
  std::size_t elements = 1;
  for (std::size_t axis = 0; axis < tensor.rank(); ++axis) {
    elements = checked_add(elements, checked_mul(tensor.shape()[axis] - 1,
                                                 tensor.strides()[axis]));
  }
  return checked_mul(elements, data_type_size(tensor.dtype()));
}

struct AddressRange {
  std::uintptr_t begin;
  std::uintptr_t end;
};

inline AddressRange address_range(ConstTensorView tensor) {
  const auto bytes = storage_span(tensor);
  if (bytes == 0) {
    return {0, 0};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data());
  require(bytes <= std::numeric_limits<std::uintptr_t>::max() - begin,
          "kernel tensor address range overflows");
  return {begin, begin + bytes};
}

inline bool overlaps(ConstTensorView lhs, ConstTensorView rhs) {
  const auto left = address_range(lhs);
  const auto right = address_range(rhs);
  return left.begin < right.end && right.begin < left.end;
}

inline bool same_shape(ConstTensorView lhs, ConstTensorView rhs) {
  return std::ranges::equal(lhs.shape(), rhs.shape());
}

inline void require_exact_or_disjoint(ConstTensorView input,
                                      ConstTensorView output,
                                      const char* message) {
  if (!overlaps(input, output)) {
    return;
  }
  require(input.data() == output.data() && input.is_contiguous()
              && output.is_contiguous() && same_shape(input, output)
              && input.dtype() == output.dtype(),
          message);
}

template <typename Function>
void dispatch_float(DataType dtype, Function&& function) {
  switch (dtype) {
    case DataType::Float32:
      std::forward<Function>(function).template operator()<float>();
      return;
    case DataType::Float16:
      std::forward<Function>(function).template operator()<__half>();
      return;
    case DataType::BFloat16:
      std::forward<Function>(function).template operator()<__nv_bfloat16>();
      return;
    default:
      throw std::invalid_argument("unsupported kernel dtype");
  }
}

template <typename T>
__device__ inline float to_float(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ inline T from_float(float value) {
  return static_cast<T>(value);
}

} // namespace nanovllm::kernels::detail
