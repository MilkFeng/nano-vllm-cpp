#include <algorithm>
#include <cmath>

#include "kernels/common.cuh"
#include "kernels/kernels.hh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

void assert_norm(const ConstTensorView& input, const ConstTensorView& weight,
                 const TensorView& output, float epsilon) {
  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(output);
  assert(input.rank() >= 1 && input.rank() <= 3 && input.shape().back() > 0);
  assert(weight.rank() == 1 && weight.shape()[0] == input.shape().back());
  assert(detail::same_shape(input, output));
  assert(input.dtype() == weight.dtype() && input.dtype() == output.dtype());
  assert(std::isfinite(epsilon) && epsilon > 0.0f);
}

template <typename T, bool Fused>
__global__ void normalization_kernel(const T* input, const T* residual,
                                    const T* weight, T* output,
                                    T* residual_output, std::size_t rows,
                                    std::size_t width, float epsilon) {
  __shared__ float sums[kThreads];
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto offset = row * width;
    float sum = 0.0f;
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      float value = detail::to_float(input[offset + col]);
      if constexpr (Fused) {
        value += detail::to_float(residual[offset + col]);
      }
      sum += value * value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
      __syncthreads();
    }
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(width) + epsilon);
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      float value = detail::to_float(input[offset + col]);
      if constexpr (Fused) {
        value += detail::to_float(residual[offset + col]);
        residual_output[offset + col] = detail::from_float<T>(value);
      }
      // The reference rounds the normalized value before multiplying weight.
      const T normalized = detail::from_float<T>(value * inv_rms);
      output[offset + col] = detail::from_float<T>(
          detail::to_float(normalized) * detail::to_float(weight[col]));
    }
    // Keep a fast warp from overwriting sums for the next grid-stride row.
    __syncthreads();
  }
}

} // namespace

void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream) {
  assert_norm(input, weight, output, epsilon);
  const auto width = input.shape().back();
  const auto rows = input.numel() / width;
  if (rows == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    normalization_kernel<T, false><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(input.data()), nullptr,
        static_cast<const T*>(weight.data()), static_cast<T*>(output.data()),
        nullptr, rows, width, epsilon);
  });

  detail::check_cuda(cudaGetLastError(), "launch rms_norm");
}

void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream) {
  assert_norm(input, weight, output, epsilon);
  detail::assert_tensor(residual);
  detail::assert_tensor(residual_output);
  assert(detail::same_shape(input, residual)
         && detail::same_shape(input, residual_output));
  assert(input.dtype() == residual.dtype()
         && input.dtype() == residual_output.dtype());
  const auto width = input.shape().back();
  const auto rows = input.numel() / width;
  if (rows == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    normalization_kernel<T, true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(input.data()),
        static_cast<const T*>(residual.data()),
        static_cast<const T*>(weight.data()), static_cast<T*>(output.data()),
        static_cast<T*>(residual_output.data()), rows, width, epsilon);
  });
  detail::check_cuda(cudaGetLastError(), "launch fused_add_rms_norm");
}

} // namespace nanovllm::kernels
