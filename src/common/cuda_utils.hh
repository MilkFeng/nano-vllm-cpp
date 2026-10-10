#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

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

} // namespace nanovllm
