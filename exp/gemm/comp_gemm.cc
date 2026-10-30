#include "comp_gemm.hh"

#include <cublasLt.h>

#include <cassert>
#include <climits>
#include <memory>
#include <stdexcept>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"

namespace nanovllm::exp {
namespace {

using BlasHandle = std::unique_ptr<cublasLtContext, decltype(&cublasLtDestroy)>;

BlasHandle make_blas_handle() {
  cublasLtHandle_t raw = nullptr;
  check_blas(cublasLtCreate(&raw), "cublasLtCreate");

  return BlasHandle(raw, cublasLtDestroy);
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

} // namespace

void cublas_linear(ConstTensorView input, ConstTensorView weight,
                   TensorView output, std::optional<ConstTensorView> bias,
                   cudaStream_t stream) {
  kernels::cuda::detail::assert_tensor(input);
  kernels::cuda::detail::assert_tensor(weight);
  kernels::cuda::detail::assert_tensor(output);

  assert(same_dtype(input, weight, output));
  assert(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2);

  const auto m = input.shape()[0];
  const auto k = input.shape()[1];
  const auto n = weight.shape()[0];

  assert(k > 0 && n > 0 && weight.shape()[1] == k);
  assert(output.shape()[0] == m && output.shape()[1] == n);
  assert(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX);

  if (bias) {
    kernels::cuda::detail::assert_tensor(*bias);
    assert(same_dtype(*bias, input));
    assert(bias->rank() == 1 && bias->shape()[0] == n);
  }

  if (m == 0) {
    return;
  }

  thread_local const auto handle = make_blas_handle();
  const auto type = cuda_dtype(input.dtype());

  cublasLtMatrixLayoutOpaque_t weight_layout{};
  cublasLtMatrixLayoutOpaque_t input_layout{};
  cublasLtMatrixLayoutOpaque_t output_layout{};

  check_blas(cublasLtMatrixLayoutInit(&weight_layout, type, k, n, k),
             "initialize linear weight layout");
  check_blas(cublasLtMatrixLayoutInit(&input_layout, type, k, m, k),
             "initialize linear input layout");
  check_blas(cublasLtMatrixLayoutInit(&output_layout, type, n, m, n),
             "initialize linear output layout");

  cublasLtMatmulDescOpaque_t operation{};
  check_blas(cublasLtMatmulDescInit(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
             "initialize linear operation");

  const cublasOperation_t transpose = CUBLAS_OP_T;
  check_blas(
      cublasLtMatmulDescSetAttribute(&operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &transpose, sizeof(transpose)),
      "transpose linear weight");

  if (bias) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    const void* pointer = bias->data();

    check_blas(cublasLtMatmulDescSetAttribute(&operation,
                                              CUBLASLT_MATMUL_DESC_EPILOGUE,
                                              &epilogue, sizeof(epilogue)),
               "enable linear bias epilogue");
    check_blas(cublasLtMatmulDescSetAttribute(&operation,
                                              CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                              &pointer, sizeof(pointer)),
               "set linear bias pointer");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  check_blas(cublasLtMatmul(handle.get(), &operation, &alpha, weight.data(),
                            &weight_layout, input.data(), &input_layout, &beta,
                            output.data(), &output_layout, output.data(),
                            &output_layout, nullptr, nullptr, 0, stream),
             "cublasLtMatmul linear");
}

} // namespace nanovllm::exp
