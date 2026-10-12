#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <stdexcept>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/inference.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

class BlasHandle {
 public:
  BlasHandle() {
    check_blas(cublasCreate(&handle_), "cublasCreate");
    const auto status = cublasSetMathMode(
        handle_, static_cast<cublasMath_t>(
                     CUBLAS_DEFAULT_MATH
                     | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      check_blas(status, "cublasSetMathMode");
    }
  }

  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;

  ~BlasHandle() { cublasDestroy(handle_); }

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

cublasHandle_t blas_handle() {
  thread_local BlasHandle handle;
  return handle.get();
}

cudaDataType_t cuda_dtype(DataType type) {
  switch (type) {
    case DataType::Float32:
      return CUDA_R_32F;
    case DataType::Float16:
      return CUDA_R_16F;
    case DataType::BFloat16:
      return CUDA_R_16BF;
    default:
      throw std::invalid_argument("linear requires floating point tensors");
  }
}

template <typename T>
__global__ void broadcast_bias(const T* __restrict__ bias,
                               T* __restrict__ output, std::size_t count,
                               std::size_t width) {
  const std::size_t start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t index = start_index; index < count; index += step) {
    output[index] = bias[index % width];
  }
}

} // namespace

void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias, cudaStream_t stream) {
  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(output);
  assert(same_dtype(input, weight, output));
  assert(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2);
  const auto m = input.shape()[0], k = input.shape()[1], n = weight.shape()[0];
  assert(k > 0 && n > 0 && weight.shape()[1] == k);
  assert(output.shape()[0] == m && output.shape()[1] == n);
  assert(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX);
  if (bias) {
    detail::assert_tensor(*bias);
    assert(same_dtype(*bias, input));
    assert(bias->rank() == 1 && bias->shape()[0] == n);
  }
  if (m == 0) {
    return;
  }

  const auto handle = blas_handle();
  check_blas(cublasSetStream(handle, stream), "cublasSetStream");

  if (bias) {
    const auto count = m * n;
    const auto blocks = static_cast<unsigned>(
        std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
    detail::dispatch_float(input.dtype(), [&]<typename T> {
      broadcast_bias<<<blocks, kThreads, 0, stream>>>(
          bias->data<T>(), output.data<T>(), count, n);
    });

    check_cuda(cudaGetLastError(), "linear bias launch");
  }

  const float alpha = 1.0f, beta = bias ? 1.0f : 0.0f;
  const auto type = cuda_dtype(input.dtype());
  check_blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n),
                          static_cast<int>(m), static_cast<int>(k), &alpha,
                          weight.data(), type, static_cast<int>(k),
                          input.data(), type, static_cast<int>(k), &beta,
                          output.data(), type, static_cast<int>(n),
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
             "cublasGemmEx");
}

} // namespace nanovllm::kernels::cuda
