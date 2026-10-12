#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nanovllm {

inline void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

inline void check_blas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status "
                             + std::to_string(static_cast<int>(status)));
  }
}

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int index) {
    check_cuda(cudaGetDevice(&previous_device_), "get the current CUDA device");
    if (previous_device_ == index) {
      return;
    }
    check_cuda(cudaSetDevice(index),
               std::format("select CUDA device {}", index));
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

} // namespace nanovllm
