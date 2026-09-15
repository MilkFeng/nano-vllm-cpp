#include <cublas_v2.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "kernels/common.cuh"
#include "kernels/inference.hh"

namespace nanovllm::kernels {
namespace {

void check_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status "
                             + std::to_string(static_cast<int>(status)));
  }
}

class BlasHandle {
 public:
  explicit BlasHandle(int device)
      : device_(device) {
    check_blas(cublasCreate(&handle_), "cublasCreate");
    // Keep split-K reductions in the FP32 accumulator type even when the
    // output uses FP16/BF16. The compute type alone permits reduced reductions.
    const auto status = cublasSetMathMode(
        handle_, static_cast<cublasMath_t>(
                     CUBLAS_DEFAULT_MATH
                     | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (status != CUBLAS_STATUS_SUCCESS) {
      (void)cublasDestroy(handle_);
      handle_ = nullptr;
      check_blas(status, "cublasSetMathMode");
    }
  }
  ~BlasHandle() {
    int previous = device_;
    const bool have_device = cudaGetDevice(&previous) == cudaSuccess;
    if (!have_device || previous != device_) (void)cudaSetDevice(device_);
    (void)cublasDestroy(handle_);
    if (have_device && previous != device_) (void)cudaSetDevice(previous);
  }
  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
  int device_;
};

cublasHandle_t blas_handle(int device) {
  thread_local std::unordered_map<int, std::unique_ptr<BlasHandle>> handles;
  auto& handle = handles[device];
  if (!handle) handle = std::make_unique<BlasHandle>(device);
  return handle->get();
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
__global__ void broadcast_bias(const T* bias, T* output, std::size_t count,
                               std::size_t width) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += std::size_t{blockDim.x} * gridDim.x) {
    output[i] = bias[i % width];
  }
}

} // namespace

void linear(ConstTensorView input, ConstTensorView weight, TensorView output,
            std::optional<ConstTensorView> bias, cudaStream_t stream) {
  detail::require_cuda(input);
  detail::require_cuda(weight);
  detail::require_cuda(output.as_const());
  detail::require_float(input);
  detail::require_same_device_dtype(input, weight);
  detail::require_same_device_dtype(input, output.as_const());
  detail::require(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2,
                  "linear expects rank-2 input, weight, and output");
  const auto m = input.shape()[0], k = input.shape()[1], n = weight.shape()[0];
  detail::require(
      k > 0 && n > 0 && weight.shape()[1] == k && output.shape()[0] == m
          && output.shape()[1] == n,
      "linear requires X[M,K], W[N,K], and Y[M,N] with positive K,N");
  detail::require(input.strides().size() == 2 && input.strides()[1] == 1
                      && input.strides()[0] >= k && weight.is_contiguous()
                      && output.is_contiguous(),
                  "linear requires contiguous columns and weights/output");
  constexpr auto maximum =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  detail::require(
      m <= maximum && n <= maximum && k <= maximum
          && input.strides()[0] <= maximum,
      "linear dimensions and input row stride must fit cuBLAS int parameters");
  detail::require(!detail::overlaps(input, output.as_const())
                      && !detail::overlaps(weight, output.as_const()),
                  "linear output must not overlap input or weight");
  if (bias) {
    detail::require_cuda(*bias);
    detail::require_same_device_dtype(input, *bias);
    detail::require(
        bias->rank() == 1 && bias->shape()[0] == n && bias->is_contiguous(),
        "linear bias must be a contiguous vector of output features");
    detail::require(!detail::overlaps(*bias, output.as_const()),
                    "linear output must not overlap bias");
  }
  if (m == 0) return;

  const detail::DeviceGuard guard(input.device().index);
  const auto handle = blas_handle(input.device().index);
  check_blas(cublasSetStream(handle, stream), "cublasSetStream");
  if (bias) {
    const auto count = detail::checked_mul(m, n);
    const auto blocks = static_cast<unsigned>(
        std::min<std::size_t>((count - 1) / 256 + 1, 65535));
    detail::dispatch_float(input.dtype(), [&]<typename T> {
      broadcast_bias<<<blocks, 256, 0, stream>>>(
          static_cast<const T*>(bias->data()), static_cast<T*>(output.data()),
          count, n);
    });
    detail::check_cuda(cudaGetLastError(), "linear bias launch");
  }
  // Row-major Y = X * W^T is column-major Y^T = W * X^T. Initializing C
  // with bias and beta=1 adds it inside GEMM before the output dtype
  // conversion.
  const float alpha = 1.0f, beta = bias ? 1.0f : 0.0f;
  const auto type = cuda_dtype(input.dtype());
  check_blas(
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n),
                   static_cast<int>(m), static_cast<int>(k), &alpha,
                   weight.data(), type, static_cast<int>(k), input.data(), type,
                   static_cast<int>(input.strides()[0]), &beta, output.data(),
                   type, static_cast<int>(n), CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT),
      "cublasGemmEx");
}

} // namespace nanovllm::kernels
