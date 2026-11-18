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

template <typename T, typename R, typename W, bool Fused, bool Gated,
          bool ZeroCentered>
__global__ void rms_norm_kernel(const T* __restrict__ input,
                                const R* __restrict__ residual_or_gate,
                                const W* __restrict__ weight,
                                T* __restrict__ output,
                                R* __restrict__ residual_output,
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
        residual_output[offset + col] = detail::from_float<R>(value);
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

template <typename T, typename R, typename W, bool Fused, bool Gated>
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
  assert(same_dtype(input, output));
  assert(std::isfinite(epsilon) && epsilon > 0.0f);

  if constexpr (Fused) {
    detail::assert_tensor(*residual_or_gate);
    detail::assert_tensor(*residual_output);

    assert(same_shape(input, *residual_or_gate, *residual_output));
    assert(same_device(input, *residual_or_gate, *residual_output));
    assert(same_dtype(*residual_or_gate, *residual_output));
  } else if constexpr (Gated) {
    detail::assert_tensor(*residual_or_gate);

    assert(same_shape(input, *residual_or_gate));
    assert(same_device(input, *residual_or_gate));
    assert(same_dtype(input, *residual_or_gate));
  }

  const auto width = input.shape().back();
  const auto rows = input.numel() / width;
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));

  const R* residual_or_gate_data = nullptr;
  R* residual_output_data = nullptr;

  if constexpr (Fused) {
    residual_or_gate_data = (*residual_or_gate).data<R>();
    residual_output_data = (*residual_output).data<R>();
  } else if constexpr (Gated) {
    residual_or_gate_data = (*residual_or_gate).data<R>();
  }

  if (zero_centered) {
    rms_norm_kernel<T, R, W, Fused, Gated, true>
        <<<blocks, kThreads, 0, stream>>>(
            input.data<T>(), residual_or_gate_data, weight.data<W>(),
            output.data<T>(), residual_output_data, rows, width, epsilon);
  } else {
    rms_norm_kernel<T, R, W, Fused, Gated, false>
        <<<blocks, kThreads, 0, stream>>>(
            input.data<T>(), residual_or_gate_data, weight.data<W>(),
            output.data<T>(), residual_output_data, rows, width, epsilon);
  }
}

template <bool Fused, bool Gated>
void dispatch_norm(ConstTensorView input,
                   std::optional<ConstTensorView> residual_or_gate,
                   ConstTensorView weight, TensorView output,
                   std::optional<TensorView> residual_output, float epsilon,
                   cudaStream_t stream, bool zero_centered) {
  assert(weight.dtype() == input.dtype()
         || weight.dtype() == DataType::Float32);
  detail::dispatch_float(input.dtype(), [&]<typename T>() {
    const auto with_residual = [&]<typename R>() {
      if (weight.dtype() == DataType::Float32) {
        launch_norm<T, R, float, Fused, Gated>(input, residual_or_gate, weight,
                                               output, residual_output, epsilon,
                                               stream, zero_centered);
      } else {
        launch_norm<T, R, T, Fused, Gated>(input, residual_or_gate, weight,
                                           output, residual_output, epsilon,
                                           stream, zero_centered);
      }
    };
    if constexpr (Fused) {
      assert(residual_or_gate->dtype() == input.dtype()
             || residual_or_gate->dtype() == DataType::Float32);
      if (residual_or_gate->dtype() == DataType::Float32) {
        with_residual.template operator()<float>();
      } else {
        with_residual.template operator()<T>();
      }
    } else {
      with_residual.template operator()<T>();
    }
  });
}

} // namespace

void rms_norm(ConstTensorView input, ConstTensorView weight, TensorView output,
              float epsilon, cudaStream_t stream, bool zero_centered) {
  dispatch_norm<false, false>(input, std::nullopt, weight, output, std::nullopt,
                              epsilon, stream, zero_centered);
  check_cuda(cudaGetLastError(), "launch rms_norm");
}

void fused_add_rms_norm(ConstTensorView input, ConstTensorView residual,
                        ConstTensorView weight, TensorView output,
                        TensorView residual_output, float epsilon,
                        cudaStream_t stream, bool zero_centered) {
  dispatch_norm<true, false>(input, residual, weight, output, residual_output,
                             epsilon, stream, zero_centered);
  check_cuda(cudaGetLastError(), "launch fused_add_rms_norm");
}

void gated_rms_norm(ConstTensorView input, ConstTensorView gate,
                    ConstTensorView weight, TensorView output, float epsilon,
                    cudaStream_t stream) {
  dispatch_norm<false, true>(input, gate, weight, output, std::nullopt, epsilon,
                             stream, false);
  check_cuda(cudaGetLastError(), "launch gated_rms_norm");
}

} // namespace nanovllm::kernels::cuda
