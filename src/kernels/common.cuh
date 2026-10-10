#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::detail {

template <bool IsConst>
void assert_tensor(const BasicTensorView<IsConst>& tensor) {
  assert(tensor.device().type == DeviceType::Cuda
         && tensor.device().index == 0);
  assert(tensor.is_contiguous());
}

template <bool LeftConst, bool RightConst>
bool same_shape(const BasicTensorView<LeftConst>& lhs,
                const BasicTensorView<RightConst>& rhs) {
  return std::ranges::equal(lhs.shape(), rhs.shape());
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
