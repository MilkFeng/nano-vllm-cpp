#include <algorithm>
#include <optional>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {
namespace {

constexpr std::size_t kWarps = 8;
constexpr std::size_t kThreads = kWarps << 5;

template <typename T, bool Fused, bool Gated, bool ZeroCentered>
__global__ void rms_norm_kernel(const T* __restrict__ input,
                                const T* __restrict__ residual_or_gate,
                                const T* __restrict__ weight,
                                T* __restrict__ output,
                                T* __restrict__ residual_output,
                                std::size_t rows, std::size_t width,
                                float epsilon) {
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto offset = row * width;

    float sum = 0.0f;
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      const std::size_t index = offset + col;

      float value = detail::to_float(input[index]);
      if constexpr (Fused) {
        value += detail::to_float(residual_or_gate[index]);
      }

      sum += value * value;
    }

    sum = detail::block_sum_reduce<float, kThreads>(sum);
    const float inv_rms = rsqrtf(sum / static_cast<float>(width) + epsilon);

    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
      const std::size_t index = offset + col;

      float value = detail::to_float(input[index]);
      if constexpr (Fused) {
        value += detail::to_float(residual_or_gate[index]);
        residual_output[offset + col] = detail::from_float<T>(value);
      }

      float w = detail::to_float(weight[col]);
      if constexpr (ZeroCentered) {
        w += 1.0f;
      }

      if constexpr (Gated) {
        const float z = residual_or_gate[index];
        output[index] = detail::from_float<T>(value * inv_rms * w
                                              * (z / (1.0f + __expf(-z))));

      } else {
        output[index] = detail::from_float<T>(value * inv_rms * w);
      }
    }
  }
}

template <typename T, bool Fused, bool Gated>
void launch_norm(ConstTensorView input,
                 std::optional<ConstTensorView> residual_or_gate,
                 ConstTensorView weight, TensorView output,
                 std::optional<TensorView> residual_output, float epsilon,
                 cudaStream_t stream, bool zero_centered) {
  if (input.numel() == 0) {
    return;
  }

  detail::assert_tensor(input);
  detail::assert_tensor(weight);
  detail::assert_tensor(output);

  assert(input.rank() >= 1 && input.rank() <= 3 && input.shape().back() > 0);
  assert(weight.rank() == 1 && weight.shape()[0] == input.shape().back());
  assert(same_shape(input, output));
  assert(same_device(input, weight, output));
  assert(same_dtype(input, weight, output));
  assert(std::isfinite(epsilon) && epsilon > 0.0f);

  if constexpr (Fused) {
    detail::assert_tensor(*residual_or_gate);
    detail::assert_tensor(*residual_output);

    assert(same_shape(input, *residual_or_gate, *residual_output));
    assert(same_device(input, *residual_or_gate, *residual_output));
    assert(same_dtype(input, *residual_or_gate, *residual_output));
  } else if constexpr (Gated) {
    detail::assert_tensor(*residual_or_gate);

    assert(input.dtype() == DataType::Float32);
    assert(same_shape(input, *residual_or_gate));
    assert(same_device(input, *residual_or_gate));
    assert(same_dtype(input, *residual_or_gate));
  }

  const auto width = input.shape().back();
  const auto rows = input.numel() / width;
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));

  const T* residual_or_gate_data = nullptr;
  T* residual_output_data = nullptr;

  if constexpr (Fused) {
    residual_or_gate_data = (*residual_or_gate).data<T>();
    residual_output_data = (*residual_output).data<T>();
  } else if constexpr (Gated) {
    residual_or_gate_data = (*residual_or_gate).data<T>();
  }

  if (zero_centered) {
    rms_norm_kernel<T, Fused, Gated, true><<<blocks, kThreads, 0, stream>>>(
        input.data<T>(), residual_or_gate_data, weight.data<T>(),
        output.data<T>(), residual_output_data, rows, width, epsilon);
  } else {
    rms_norm_kernel<T, Fused, Gated, false><<<blocks, kThreads, 0, stream>>>(
        input.data<T>(), residual_or_gate_data, weight.data<T>(),
        output.data<T>(), residual_output_data, rows, width, epsilon);
  }
}

} // namespace

void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream, bool zero_centered) {
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    launch_norm<T, false, false>(input, std::nullopt, weight, output,
                                 std::nullopt, epsilon, stream, zero_centered);
  });
  check_cuda(cudaGetLastError(), "launch rms_norm");
}

void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream, bool zero_centered) {
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    launch_norm<T, true, false>(input, residual, weight, output,
                                residual_output, epsilon, stream,
                                zero_centered);
  });
  check_cuda(cudaGetLastError(), "launch fused_add_rms_norm");
}

void gated_rms_norm(ConstTensorView input, ConstTensorView gate,
                    ConstTensorView weight, TensorView output, float epsilon,
                    cudaStream_t stream) {
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    launch_norm<T, false, true>(input, gate, weight, output, std::nullopt,
                                epsilon, stream, false);
  });
  check_cuda(cudaGetLastError(), "launch gated_rms_norm");
}

} // namespace nanovllm::kernels::cuda
