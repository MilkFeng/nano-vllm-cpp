#include <algorithm>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr unsigned kThreads = 256;

void assert_norm(const ConstTensorView& input, const ConstTensorView& weight,
                 const TensorView& output, float epsilon) {
  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(output);
  assert(input.rank() >= 1 && input.rank() <= 3 && input.shape().back() > 0);
  assert(weight.rank() == 1 && weight.shape()[0] == input.shape().back());
  assert(same_shape(input, output));
  assert(same_device(input, weight, output));
  assert(same_dtype(input, weight, output));
  assert(std::isfinite(epsilon) && epsilon > 0.0f);
}

template <typename T, bool Fused, bool ZeroCentered>
__device__ float input_value(const T* input, const T* residual, std::size_t n) {
  float value = detail::to_float(input[n]);
  if constexpr (Fused) {
    value += detail::to_float(residual[n]);
    // Qwen3.5 normalizes the residual sum after rounding to its tensor dtype.
    if constexpr (ZeroCentered)
      value = detail::to_float(detail::from_float<T>(value));
  }
  return value;
}

template <typename T, bool Fused, bool ZeroCentered>
__global__ void normalization_kernel(const T* input, const T* residual,
                                     const T* weight, T* output,
                                     T* residual_output, std::size_t rows,
                                     std::size_t width, float epsilon) {
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto offset = row * width;
    float sum = 0.0f;
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      const float value =
          input_value<T, Fused, ZeroCentered>(input, residual, offset + col);
      sum += value * value;
    }
    sum = detail::block_sum_reduce<float, kThreads>(sum);
    const float inv_rms = rsqrtf(sum / static_cast<float>(width) + epsilon);
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      const float value =
          input_value<T, Fused, ZeroCentered>(input, residual, offset + col);
      if constexpr (Fused)
        residual_output[offset + col] = detail::from_float<T>(value);
      if constexpr (ZeroCentered) {
        // Qwen3.5 applies (1 + weight) in FP32 before the output cast.
        output[offset + col] = detail::from_float<T>(
            value * inv_rms * (1.0f + detail::to_float(weight[col])));
      } else {
        // Preserve nano-vllm/Qwen3's intermediate rounding semantics.
        const T normalized = detail::from_float<T>(value * inv_rms);
        output[offset + col] = detail::from_float<T>(
            detail::to_float(normalized) * detail::to_float(weight[col]));
      }
    }
  }
}

template <typename T, bool Fused>
void launch_norm(ConstTensorView input, ConstTensorView residual,
                 ConstTensorView weight, TensorView output,
                 TensorView residual_output, float epsilon, cudaStream_t stream,
                 bool zero_centered) {
  const auto width = input.shape().back();
  const auto rows = input.numel() / width;
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
  const T* residual_data = nullptr;
  T* residual_output_data = nullptr;
  if constexpr (Fused) {
    residual_data = residual.data<T>();
    residual_output_data = residual_output.data<T>();
  }
  if (zero_centered)
    normalization_kernel<T, Fused, true><<<blocks, kThreads, 0, stream>>>(
        input.data<T>(), residual_data, weight.data<T>(), output.data<T>(),
        residual_output_data, rows, width, epsilon);
  else
    normalization_kernel<T, Fused, false><<<blocks, kThreads, 0, stream>>>(
        input.data<T>(), residual_data, weight.data<T>(), output.data<T>(),
        residual_output_data, rows, width, epsilon);
}

__global__ void gated_norm_kernel(const float* input, const float* gate,
                                  const float* weight, float* output,
                                  std::size_t rows, std::size_t width,
                                  float epsilon) {
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto offset = row * width;
    float sum = 0;
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      const float value = input[offset + col];
      sum += value * value;
    }
    sum = detail::block_sum_reduce<float, kThreads>(sum);
    const float inv_rms = rsqrtf(sum / static_cast<float>(width) + epsilon);
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      const auto n = offset + col;
      const float z = gate[n];
      output[n] = input[n] * inv_rms * weight[col] * (z / (1.0f + expf(-z)));
    }
  }
}

} // namespace

void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream, bool zero_centered) {
  assert_norm(input, weight, output, epsilon);
  if (input.numel() == 0) return;
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    launch_norm<T, false>(input, {}, weight, output, {}, epsilon, stream,
                          zero_centered);
  });
  check_cuda(cudaGetLastError(), "launch rms_norm");
}

void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream, bool zero_centered) {
  assert_norm(input, weight, output, epsilon);
  detail::assert_tensor(residual);
  detail::assert_tensor(residual_output);
  assert(same_shape(input, residual, residual_output));
  assert(same_device(input, residual, residual_output));
  assert(same_dtype(input, residual, residual_output));
  if (input.numel() == 0) return;
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    launch_norm<T, true>(input, residual, weight, output, residual_output,
                         epsilon, stream, zero_centered);
  });
  check_cuda(cudaGetLastError(), "launch fused_add_rms_norm");
}

void gated_rms_norm(ConstTensorView input, ConstTensorView gate,
                    ConstTensorView weight, TensorView output, float epsilon,
                    cudaStream_t stream) {
  assert_norm(input, weight, output, epsilon);
  detail::assert_tensor(gate);
  assert(input.dtype() == DataType::Float32);
  assert(same_shape(input, gate));
  assert(same_device(input, gate));
  assert(same_dtype(input, gate));
  if (input.numel() == 0) return;
  const auto width = input.shape().back();
  const auto rows = input.numel() / width;
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
  gated_norm_kernel<<<blocks, kThreads, 0, stream>>>(
      input.data<float>(), gate.data<float>(), weight.data<float>(),
      output.data<float>(), rows, width, epsilon);
  check_cuda(cudaGetLastError(), "launch gated_rms_norm");
}

} // namespace nanovllm::kernels::cuda
