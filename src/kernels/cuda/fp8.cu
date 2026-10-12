#include <cublasLt.h>
#include <cuda_fp8.h>

#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/fp8.hh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kBlockSize = 128;
constexpr std::size_t kTokenAlignment = 16;
constexpr std::size_t kBlasWorkspaceBytes = 1 << 20;

std::size_t round_up(std::size_t value, std::size_t alignment) {
  return ((value - 1) / alignment + 1) * alignment;
}

class LtHandle {
 public:
  LtHandle() { check_blas(cublasLtCreate(&handle_), "create cuBLASLt handle"); }
  ~LtHandle() { cublasLtDestroy(handle_); }
  cublasLtHandle_t get() const { return handle_; }

 private:
  cublasLtHandle_t handle_ = nullptr;
};

cublasLtHandle_t lt_handle() {
  thread_local LtHandle handle;
  return handle.get();
}

class MatmulPlan {
 public:
  MatmulPlan(std::size_t weight_stride, std::size_t buffer_stride,
             std::size_t tokens, const float* weight_scale,
             const float* input_scale, std::size_t workspace_bytes) {
    try {
      check_blas(
          cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "create FP8 matmul descriptor");
      const cublasOperation_t transpose = CUBLAS_OP_T;
      check_blas(cublasLtMatmulDescSetAttribute(operation_,
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transpose, sizeof(transpose)),
                 "set FP8 weight transpose");
      const std::int8_t fast_accum = 0;
      check_blas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum,
                     sizeof(fast_accum)),
                 "disable FP8 fast accumulation");
      set_scales(weight_scale, input_scale);
      // Row-major W/X/Y are column-major W^T/X^T/Y^T: TN GEMM.
      check_blas(
          cublasLtMatrixLayoutCreate(&weight_, CUDA_R_8F_E4M3, kBlockSize,
                                     kBlockSize, weight_stride),
          "create FP8 weight layout");
      check_blas(cublasLtMatrixLayoutCreate(&input_, CUDA_R_8F_E4M3, kBlockSize,
                                            tokens, buffer_stride),
                 "create FP8 input layout");
      check_blas(cublasLtMatrixLayoutCreate(&output_, CUDA_R_32F, kBlockSize,
                                            tokens, buffer_stride),
                 "create FP8 output layout");
      check_blas(cublasLtMatmulPreferenceCreate(&preference_),
                 "create FP8 matmul preference");
      check_blas(cublasLtMatmulPreferenceSetAttribute(
                     preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_bytes, sizeof(workspace_bytes)),
                 "set FP8 matmul workspace limit");
      int count = 0;
      check_blas(cublasLtMatmulAlgoGetHeuristic(
                     lt_handle(), operation_, weight_, input_, output_, output_,
                     preference_, 1, &algorithm_, &count),
                 "select native FP8 matmul algorithm");
      if (count == 0 || algorithm_.state != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(
            "cuBLASLt has no native FP8 algorithm for this GPU/layout");
    } catch (...) {
      close();
      throw;
    }
  }
  ~MatmulPlan() { close(); }

  MatmulPlan(const MatmulPlan&) = delete;
  MatmulPlan& operator=(const MatmulPlan&) = delete;

  void run(const std::uint8_t* weight, const std::uint8_t* input, float* output,
           const float* weight_scale, const float* input_scale, bool first,
           Tensor& workspace, cudaStream_t stream) {
    set_scales(weight_scale, input_scale);
    const float alpha = 1.0f, beta = first ? 0.0f : 1.0f;
    check_blas(cublasLtMatmul(lt_handle(), operation_, &alpha, weight, weight_,
                              input, input_, &beta, output, output_, output,
                              output_, &algorithm_.algo, workspace.data(),
                              workspace.size_bytes(), stream),
               "native FP8 Tensor Core matmul");
  }

 private:
  void set_scales(const float* weight_scale, const float* input_scale) {
    check_blas(cublasLtMatmulDescSetAttribute(
                   operation_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                   &weight_scale, sizeof(weight_scale)),
               "set FP8 weight scale");
    check_blas(cublasLtMatmulDescSetAttribute(
                   operation_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                   &input_scale, sizeof(input_scale)),
               "set FP8 input scale");
  }

  void close() noexcept {
    if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
    if (output_) cublasLtMatrixLayoutDestroy(output_);
    if (input_) cublasLtMatrixLayoutDestroy(input_);
    if (weight_) cublasLtMatrixLayoutDestroy(weight_);
    if (operation_) cublasLtMatmulDescDestroy(operation_);
  }

  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t weight_ = nullptr, input_ = nullptr, output_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  cublasLtMatmulHeuristicResult_t algorithm_{};
};

__global__ void quantize_input_kernel(const float* __restrict__ input,
                                      std::uint8_t* __restrict__ quantized,
                                      float* __restrict__ scales,
                                      std::size_t tokens, std::size_t columns,
                                      std::size_t padded_tokens,
                                      std::size_t buffer_stride) {
  __shared__ float maxima[kThreads];
  const std::size_t column_begin = blockIdx.x * kBlockSize;
  float maximum = 0.0f;
  for (std::size_t index = threadIdx.x; index < tokens * kBlockSize;
       index += kThreads) {
    const auto token = index / kBlockSize;
    const auto column = column_begin + index % kBlockSize;
    if (column < columns)
      maximum = fmaxf(maximum, fabsf(input[token * columns + column]));
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
    __syncthreads();
  }
  const float scale =
      maxima[0] == 0.0f ? 1.0f : fmaxf(maxima[0] / 448.0f, FLT_MIN);
  if (threadIdx.x == 0) scales[blockIdx.x] = scale;
  for (std::size_t index = threadIdx.x; index < padded_tokens * kBlockSize;
       index += kThreads) {
    const auto token = index / kBlockSize;
    const auto column = column_begin + index % kBlockSize;
    const float value = token < tokens && column < columns
                            ? input[token * columns + column] / scale
                            : 0.0f;
    quantized[token * buffer_stride + column] = __nv_fp8_e4m3(value).__x;
  }
}

} // namespace

FP8LinearWorkspace::FP8LinearWorkspace(std::size_t max_tokens,
                                       std::size_t max_features) {
  assert(max_tokens > 0 && max_features > 0);
  const auto tokens = round_up(max_tokens, kTokenAlignment);
  const auto features = round_up(max_features, kBlockSize);
  quantized_input = Tensor::empty({tokens, features}, DataType::Float8E4M3FN,
                                  Device::cuda(0));
  input_scale = Tensor::empty({features / kBlockSize}, DataType::Float32,
                              Device::cuda(0));
  accumulator =
      Tensor::empty({tokens, features}, DataType::Float32, Device::cuda(0));
  blas = Tensor::empty({kBlasWorkspaceBytes}, DataType::Float8E4M3FN,
                       Device::cuda(0));
}

void linear_fp8(ConstTensorView input, ConstTensorView weight,
                ConstTensorView weight_scale_inv, TensorView output,
                FP8LinearWorkspace& workspace, cudaStream_t stream) {
  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(weight_scale_inv);
  detail::assert_tensor(output);
  assert(input.dtype() == DataType::Float32
         && output.dtype() == DataType::Float32);
  assert(weight.dtype() == DataType::Float8E4M3FN);
  assert(weight_scale_inv.dtype() == DataType::Float32);
  assert(same_device(input, weight, weight_scale_inv, output));
  assert(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2);
  assert(weight_scale_inv.rank() == 2);
  const auto tokens = input.shape()[0];
  const auto columns = input.shape()[1];
  const auto rows = output.shape()[1];
  assert(rows > 0 && columns > 0 && output.shape()[0] == tokens);
  const auto padded_rows = round_up(rows, kBlockSize);
  const auto padded_columns = round_up(columns, kBlockSize);
  assert(weight.shape()[0] == padded_rows
         && weight.shape()[1] == padded_columns);
  assert(weight_scale_inv.shape()[0] == padded_rows / kBlockSize);
  assert(weight_scale_inv.shape()[1] == padded_columns / kBlockSize);
  if (tokens == 0) return;

  const auto padded_tokens = round_up(tokens, kTokenAlignment);
  const auto buffer_stride = workspace.quantized_input.shape()[1];
  assert(workspace.quantized_input.shape()[0] >= padded_tokens);
  assert(buffer_stride >= padded_columns && buffer_stride >= padded_rows);
  assert(workspace.accumulator.shape()[0] >= padded_tokens);
  assert(workspace.accumulator.shape()[1] == buffer_stride);
  assert(workspace.input_scale.numel() >= padded_columns / kBlockSize);
  const auto column_blocks = padded_columns / kBlockSize;
  quantize_input_kernel<<<static_cast<unsigned>(column_blocks), kThreads, 0,
                          stream>>>(
      input.data<float>(),
      workspace.quantized_input.view().data<std::uint8_t>(),
      workspace.input_scale.view().data<float>(), tokens, columns,
      padded_tokens, buffer_stride);
  check_cuda(cudaGetLastError(), "quantize FP8 GEMM input");

  MatmulPlan plan(padded_columns, buffer_stride, padded_tokens,
                  weight_scale_inv.data<float>(),
                  workspace.input_scale.view().data<float>(),
                  workspace.blas.size_bytes());
  for (std::size_t row = 0; row < padded_rows; row += kBlockSize) {
    for (std::size_t column = 0; column < padded_columns;
         column += kBlockSize) {
      const auto block = column / kBlockSize;
      plan.run(weight.data<std::uint8_t>() + row * padded_columns + column,
               workspace.quantized_input.view().data<std::uint8_t>() + column,
               workspace.accumulator.view().data<float>() + row,
               weight_scale_inv.data<float>()
                   + (row / kBlockSize) * column_blocks + block,
               workspace.input_scale.view().data<float>() + block, column == 0,
               workspace.blas, stream);
    }
  }
  check_cuda(
      cudaMemcpy2DAsync(output.data(), rows * sizeof(float),
                        workspace.accumulator.data(),
                        buffer_stride * sizeof(float), rows * sizeof(float),
                        tokens, cudaMemcpyDeviceToDevice, stream),
      "copy native FP8 matmul output");
}

} // namespace nanovllm::kernels::cuda
