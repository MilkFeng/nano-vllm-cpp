#include "kernels/kernels.hh"

#include <algorithm>
#include <cmath>

#include "kernels/common.cuh"

namespace nanovllm::kernels {
namespace {

constexpr unsigned kThreads = 256;

struct Rows {
  std::size_t count;
  std::size_t width;
  std::size_t heads;
  std::size_t token_stride;
};

Rows validate_input(ConstTensorView input) {
  detail::require_cuda(input);
  detail::require_float(input);
  detail::require(input.rank() >= 1 && input.rank() <= 3,
                  "normalization input must have rank 1, 2, or 3");
  detail::require(input.strides().size() == input.rank(),
                  "normalization input shape and strides must have equal rank");
  const auto width = input.shape().back();
  detail::require(width > 0, "normalization width must be positive");
  detail::require(input.strides().back() == 1,
                  "normalization input last dimension must be contiguous");
  if (input.rank() == 1) return {1, width, 1, width};

  const auto heads = input.rank() == 3 ? input.shape()[1] : 1;
  if (input.rank() == 3) {
    detail::require(input.strides()[1] == width,
                    "normalization input heads must be contiguous");
  }
  const auto token_width = detail::checked_mul(heads, width);
  if (input.shape()[0] > 1 && heads != 0) {
    detail::require(input.strides()[0] >= token_width,
                    "normalization input token rows must not overlap");
  }
  return {detail::checked_mul(input.shape()[0], heads), width, heads,
          input.strides()[0]};
}

void validate_output(ConstTensorView input, ConstTensorView output) {
  detail::require_same_device_dtype(input, output);
  detail::require(detail::same_shape(input, output),
                  "normalization output shape must match input");
  detail::require(output.is_contiguous(),
                  "normalization output must be contiguous");
}

void validate_alias(ConstTensorView input, ConstTensorView output) {
  detail::require_exact_or_disjoint(input, output,
      "normalization permits only exact contiguous input/output aliases");
}

void validate_weight(ConstTensorView input, ConstTensorView weight,
                     ConstTensorView output, float epsilon) {
  detail::require(std::isfinite(epsilon) && epsilon > 0.0f,
                  "normalization epsilon must be finite and positive");
  detail::require_same_device_dtype(input, weight);
  detail::require(weight.rank() == 1 && weight.is_contiguous()
                      && weight.shape()[0] == input.shape().back(),
                  "normalization weight must be a contiguous vector matching width");
  detail::require(!detail::overlaps(weight, output),
                  "normalization weight must not overlap output");
}

template <typename T, bool Fused>
__global__ void normalization_kernel(
    const T* input, const T* residual, const T* weight, T* output,
    T* residual_output, Rows rows, std::size_t residual_token_stride,
    float epsilon) {
  __shared__ float sums[kThreads];
  for (std::size_t row = blockIdx.x; row < rows.count; row += gridDim.x) {
    const auto token = row / rows.heads;
    const auto head = row % rows.heads;
    const auto input_offset = token * rows.token_stride + head * rows.width;
    const auto residual_offset = token * residual_token_stride
                                 + head * rows.width;
    const auto output_offset = row * rows.width;
    float sum = 0.0f;
    for (std::size_t col = threadIdx.x; col < rows.width; col += blockDim.x) {
      float value = detail::to_float(input[input_offset + col]);
      if constexpr (Fused) {
        value += detail::to_float(residual[residual_offset + col]);
      }
      sum += value * value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = kThreads / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
      __syncthreads();
    }
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(rows.width)
                                + epsilon);
    for (std::size_t col = threadIdx.x; col < rows.width; col += blockDim.x) {
      float value = detail::to_float(input[input_offset + col]);
      if constexpr (Fused) {
        value += detail::to_float(residual[residual_offset + col]);
        residual_output[output_offset + col] = detail::from_float<T>(value);
      }
      // The reference rounds the normalized value before multiplying weight.
      const T normalized = detail::from_float<T>(value * inv_rms);
      output[output_offset + col] = detail::from_float<T>(
          detail::to_float(normalized) * detail::to_float(weight[col]));
    }
    // Keep a fast warp from overwriting sums for the next grid-stride row.
    __syncthreads();
  }
}

} // namespace

void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream) {
  const auto rows = validate_input(input);
  validate_output(input, output.as_const());
  validate_weight(input, weight, output.as_const(), epsilon);
  validate_alias(input, output.as_const());
  if (rows.count == 0) return;

  detail::DeviceGuard guard(input.device().index);
  const auto blocks = static_cast<unsigned>(std::min(rows.count, std::size_t{65535}));
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    normalization_kernel<T, false><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(input.data()), nullptr,
        static_cast<const T*>(weight.data()), static_cast<T*>(output.data()),
        nullptr, rows, 0, epsilon);
  });
  detail::check_cuda(cudaGetLastError(), "launch rms_norm");
}

void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream) {
  const auto rows = validate_input(input);
  const auto residual_rows = validate_input(residual);
  detail::require_same_device_dtype(input, residual);
  detail::require(detail::same_shape(input, residual),
                  "normalization residual shape must match input");
  validate_output(input, output.as_const());
  validate_output(input, residual_output.as_const());
  validate_weight(input, weight, output.as_const(), epsilon);
  detail::require(!detail::overlaps(weight, residual_output.as_const()),
                  "normalization weight must not overlap residual output");
  detail::require(!detail::overlaps(output.as_const(), residual_output.as_const()),
                  "normalization outputs must not overlap");
  validate_alias(input, output.as_const());
  validate_alias(input, residual_output.as_const());
  validate_alias(residual, output.as_const());
  validate_alias(residual, residual_output.as_const());
  if (rows.count == 0) return;

  detail::DeviceGuard guard(input.device().index);
  const auto blocks = static_cast<unsigned>(std::min(rows.count, std::size_t{65535}));
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    normalization_kernel<T, true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(input.data()), static_cast<const T*>(residual.data()),
        static_cast<const T*>(weight.data()), static_cast<T*>(output.data()),
        static_cast<T*>(residual_output.data()), rows, residual_rows.token_stride,
        epsilon);
  });
  detail::check_cuda(cudaGetLastError(), "launch fused_add_rms_norm");
}

} // namespace nanovllm::kernels
