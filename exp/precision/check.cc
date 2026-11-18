#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/cuda_utils.hh"
#include "functions/cast.hh"
#include "kernels/cpu/fp8_dequant.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/lm_head.hh"
#include "models/qwen35.hh"

using namespace nanovllm;

namespace {
const auto gpu = Device::cuda(0);

void require(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

float bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  return std::bit_cast<float>((bits + 0x7fffU + ((bits >> 16) & 1U))
                              & 0xffff0000U);
}

void upload(TensorView target, const void* source) {
  check_cuda(cudaMemcpy(target.data(), source, target.size_bytes(),
                        cudaMemcpyHostToDevice),
             "upload precision fixture");
}

void upload_float(TensorView target, const std::vector<float>& values) {
  require(target.numel() == values.size(), "fixture size");
  if (target.dtype() == DataType::Float32) {
    upload(target, values.data());
  } else {
    require(target.dtype() == DataType::BFloat16, "fixture dtype");
    std::vector<std::uint16_t> packed(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      packed[i] = std::bit_cast<std::uint32_t>(bf16(values[i])) >> 16;
    }
    upload(target, packed.data());
  }
}

template <bool C>
std::vector<float> read(BasicTensorView<C> source) {
  ConstTensorView view(source);
  Tensor expanded;
  if (view.dtype() == DataType::BFloat16) {
    expanded = cast_tensor(view, DataType::Float32, nullptr);
    view = expanded.view().as_const();
  }
  require(view.dtype() == DataType::Float32, "read dtype");
  std::vector<float> result(view.numel());
  check_cuda(cudaMemcpy(result.data(), view.data(), view.size_bytes(),
                        cudaMemcpyDeviceToHost),
             "read precision result");
  return result;
}

void compare(const std::vector<float>& actual,
             const std::vector<float>& expected, float atol, float rtol) {
  require(actual.size() == expected.size(), "comparison size");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])
        || std::abs(actual[i] - expected[i])
               > atol + rtol * std::abs(expected[i])) {
      throw std::runtime_error("precision mismatch at " + std::to_string(i)
                               + ": " + std::to_string(actual[i]) + " / "
                               + std::to_string(expected[i]));
    }
  }
}

void check_linear(std::size_t m, std::size_t n, std::size_t k) {
  auto x = Tensor::empty({m, k}, DataType::BFloat16, gpu);
  std::vector<float> input(m * k);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = bf16(std::sin(float(i) * 0.123f));
  }
  upload_float(x.view(), input);
  auto weight_cpu = Tensor::empty({n, k}, DataType::Float8E4M3FN);
  auto scale_cpu =
      Tensor::empty({(n + 127) / 128, (k + 127) / 128}, DataType::Float32);
  for (std::size_t i = 0; i < weight_cpu.numel(); ++i) {
    weight_cpu.view().data<std::uint8_t>()[i] =
        (i * 37 % 120) | ((i % 3 == 0) ? 128 : 0);
  }
  for (std::size_t i = 0; i < scale_cpu.numel(); ++i) {
    scale_cpu.view().data<float>()[i] = 0.0012345f * (i + 1);
  }
  auto expanded = Tensor::empty({n, k}, DataType::Float32);
  kernels::cpu::dequantize_fp8(weight_cpu.view().as_const(),
                               scale_cpu.view().as_const(), expanded.view());
  auto w = Tensor::empty({n, k}, DataType::Float8E4M3FN, gpu);
  auto scale =
      Tensor::empty({(n + 127) / 128, (k + 127) / 128}, DataType::Float32, gpu);
  upload(w.view(), weight_cpu.data());
  upload(scale.view(), scale_cpu.data());
  for (const auto dtype : {DataType::Float32, DataType::BFloat16}) {
    auto output = Tensor::empty({m, n}, dtype, gpu);
    kernels::cuda::linear_fp8(x.view().as_const(), w.view().as_const(),
                              scale.view().as_const(), output.view());
    std::vector<float> expected(m * n);
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t col = 0; col < n; ++col) {
        double sum = 0;
        for (std::size_t j = 0; j < k; ++j) {
          sum += double(input[row * k + j])
                 * expanded.view().data<float>()[col * k + j];
        }
        expected[row * n + col] =
            dtype == DataType::BFloat16 ? bf16(float(sum)) : float(sum);
      }
    }
    compare(read(output.view()), expected, 2e-5f,
            dtype == DataType::BFloat16 ? 0.008f : 2e-5f);
  }
  // LM head: BF16 operands, FP32 logits. Include ragged GEMM/GEMV shapes.
  LMHead head(n, k, DataType::BFloat16, gpu);
  std::vector<float> weights(n * k);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = bf16(0.03f * std::cos(float(i) * 0.031f));
  }
  upload_float(head.weight().view(), weights);
  const auto output =
      head.forward(x.view().as_const(), nullptr, DataType::Float32);
  require(output.dtype() == DataType::Float32,
          "LM head must produce FP32 logits");
  std::vector<float> expected(m * n);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double sum = 0;
      for (std::size_t j = 0; j < k; ++j) {
        sum += double(input[row * k + j]) * weights[col * k + j];
      }
      expected[row * n + col] = float(sum);
    }
  }
  compare(read(output.view()), expected, 2e-5f, 2e-5f);
}

void check_norm() {
  constexpr std::size_t width = 257;
  auto input = Tensor::empty({2, width}, DataType::BFloat16, gpu);
  auto residual = Tensor::empty({2, width}, DataType::Float32, gpu);
  RMSNorm norm(width, 1e-6f, DataType::BFloat16, gpu, true);
  std::vector<float> x(2 * width), r(2 * width), w(width), expected(x.size()),
      sums(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = bf16(1.f + float(i % 5) * 0.125f);
    r[i] = 256.f + float(i % 3) * 0.03125f;
    sums[i] = x[i] + r[i];
  }
  for (std::size_t i = 0; i < width; ++i) {
    w[i] = bf16(0.02f * std::sin(float(i)));
  }
  upload_float(input.view(), x);
  upload_float(residual.view(), r);
  upload_float(norm.weight().view(), w);
  auto output =
      norm.forward(input.view().as_const(), residual.view().as_const());
  require(output.hidden_states.dtype() == DataType::BFloat16,
          "norm activation dtype");
  require(output.residual.dtype() == DataType::Float32,
          "residual must stay FP32");
  compare(read(output.residual.view()), sums, 0, 0);
  for (std::size_t row = 0; row < 2; ++row) {
    double square = 0;
    for (std::size_t j = 0; j < width; ++j) {
      square += double(sums[row * width + j]) * sums[row * width + j];
    }
    const auto inv = float(1.0 / std::sqrt(square / width + 1e-6));
    for (std::size_t j = 0; j < width; ++j) {
      expected[row * width + j] =
          bf16(sums[row * width + j] * inv * (1.f + w[j]));
    }
  }
  compare(read(output.hidden_states.view()), expected, 1e-6f, 0.008f);
}

void check_delta() {
  constexpr std::size_t tokens = 513, kd = 8, vd = 8, heads = 2;
  constexpr std::size_t channels = 2 * kd + heads * vd;
  auto qkv = Tensor::empty({tokens, channels}, DataType::BFloat16, gpu);
  auto weight = Tensor::empty({channels, 1, 4}, DataType::BFloat16, gpu);
  auto history = Tensor::empty({2, channels, 3}, DataType::BFloat16, gpu);
  auto state = Tensor::empty({2, heads, kd, vd}, DataType::Float32, gpu);
  auto conv = Tensor::empty({tokens, channels}, DataType::BFloat16, gpu);
  auto output = Tensor::empty({tokens, heads, vd}, DataType::BFloat16, gpu);
  auto a = Tensor::empty({tokens, heads}, DataType::Float32, gpu);
  auto b = Tensor::empty({tokens, heads}, DataType::Float32, gpu);
  auto log_a = Tensor::empty({heads}, DataType::Float32, gpu);
  auto bias = Tensor::empty({heads}, DataType::Float32, gpu);
  auto slot = Tensor::empty({1}, DataType::Int32, gpu);
  const std::int32_t one = 1;
  upload(slot.view(), &one);
  std::vector<float> values(qkv.numel()), weights(weight.numel());
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = bf16(0.3f * std::sin(float(i) * 0.071f));
  }
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = bf16(0.25f * std::cos(float(i)));
  }
  upload_float(qkv.view(), values);
  upload_float(weight.view(), weights);
  upload_float(a.view(), std::vector<float>(a.numel(), -2.f));
  upload_float(b.view(), std::vector<float>(b.numel(), 0.1f));
  upload_float(log_a.view(), std::vector<float>(heads, -3.f));
  upload_float(bias.view(), std::vector<float>(heads, 0.f));
  const auto reset = [&] {
    check_cuda(cudaMemset(history.data(), 0, history.size_bytes()),
               "reset conv state");
    check_cuda(cudaMemset(state.data(), 0, state.size_bytes()),
               "reset recurrent state");
  };
  reset();
  for (std::size_t t = 0; t < tokens; ++t) {
    kernels::cuda::causal_conv1d(
        qkv.view().as_const().slice(0, t, 1), weight.view().as_const(),
        history.view(), {}, slot.view().as_const(), conv.view().slice(0, t, 1));
    kernels::cuda::gated_delta_rule(
        conv.view().as_const().slice(0, t, 1),
        a.view().as_const().slice(0, t, 1), b.view().as_const().slice(0, t, 1),
        log_a.view().as_const(), bias.view().as_const(), state.view(), {},
        slot.view().as_const(), 1, output.view().slice(0, t, 1));
  }
  const auto expected_output = read(output.view());
  const auto expected_state = read(state.view());
  const auto expected_history = read(history.view());
  for (std::size_t i = 0; i < heads * kd * vd; ++i) {
    require(expected_state[i] == 0, "inactive state slot changed");
  }
  // A separate double-precision recurrence verifies the FP32 state update.
  const auto convolved = read(conv.view());
  std::vector<float> conv_reference(convolved.size());
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t c = 0; c < channels; ++c) {
      double sum = 0;
      for (std::size_t j = 0; j < 4; ++j) {
        if (t + j >= 3) {
          sum +=
              double(values[(t + j - 3) * channels + c]) * weights[c * 4 + j];
        }
      }
      conv_reference[t * channels + c] =
          bf16(float(sum / (1. + std::exp(-sum))));
    }
  }
  compare(convolved, conv_reference, 2e-5f, 0.008f);
  std::vector<double> reference(heads * kd * vd, 0);
  const double decay = std::exp(-std::exp(-3.) * std::log1p(std::exp(-2.)));
  const double beta = 1. / (1. + std::exp(-0.1));
  for (std::size_t t = 0; t < tokens; ++t) {
    double qnorm = 1e-6, knorm = 1e-6;
    for (std::size_t j = 0; j < kd; ++j) {
      qnorm +=
          double(convolved[t * channels + j]) * convolved[t * channels + j];
      knorm += double(convolved[t * channels + kd + j])
               * convolved[t * channels + kd + j];
    }
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t v = 0; v < vd; ++v) {
        double prediction = 0;
        for (std::size_t j = 0; j < kd; ++j) {
          auto& memory = reference[(h * kd + j) * vd + v];
          memory *= decay;
          prediction +=
              convolved[t * channels + kd + j] / std::sqrt(knorm) * memory;
        }
        const double delta =
            (convolved[t * channels + 2 * kd + h * vd + v] - prediction) * beta;
        double result = 0;
        for (std::size_t j = 0; j < kd; ++j) {
          auto& memory = reference[(h * kd + j) * vd + v];
          memory += convolved[t * channels + kd + j] / std::sqrt(knorm) * delta;
          result +=
              convolved[t * channels + j] / std::sqrt(qnorm * kd) * memory;
        }
        const auto actual = expected_output[(t * heads + h) * vd + v];
        require(std::abs(actual - result) < 2e-5 + 0.008 * std::abs(result),
                "double recurrence reference");
      }
    }
  }
  std::vector<float> reference_state(reference.begin(), reference.end());
  compare({expected_state.begin() + heads * kd * vd, expected_state.end()},
          reference_state, 2e-6f, 2e-4f);
  // Chunk boundaries cross the four-tap convolution history and recurrent
  // state.
  reset();
  for (std::size_t begin = 0; begin < tokens; begin += 37) {
    const auto count = std::min<std::size_t>(37, tokens - begin);
    auto cu = Tensor::empty({2}, DataType::Int32, gpu);
    const std::int32_t boundaries[]{0, static_cast<std::int32_t>(count)};
    upload(cu.view(), boundaries);
    kernels::cuda::causal_conv1d(qkv.view().as_const().slice(0, begin, count),
                                 weight.view().as_const(), history.view(),
                                 cu.view().as_const(), slot.view().as_const(),
                                 conv.view().slice(0, begin, count));
    kernels::cuda::gated_delta_rule(
        conv.view().as_const().slice(0, begin, count),
        a.view().as_const().slice(0, begin, count),
        b.view().as_const().slice(0, begin, count), log_a.view().as_const(),
        bias.view().as_const(), state.view(), cu.view().as_const(),
        slot.view().as_const(), 1, output.view().slice(0, begin, count));
  }
  compare(read(output.view()), expected_output, 0, 0);
  compare(read(state.view()), expected_state, 0, 0);
  compare(read(history.view()), expected_history, 0, 0);
}

void check_cache() {
  ModelConfig config;
  config.num_hidden_layers = 4;
  config.layer_types = {AttentionType::Linear, AttentionType::Linear,
                        AttentionType::Linear, AttentionType::Full};
  config.num_key_value_heads = 2;
  config.head_dim = 16;
  config.linear_num_key_heads = 1;
  config.linear_num_value_heads = 2;
  config.linear_key_head_dim = 8;
  config.linear_value_head_dim = 8;
  config.linear_conv_kernel_dim = 4;
  Qwen35Cache cache(config, 4, 16, 2, gpu);
  require(cache.key_cache(0).dtype() == DataType::BFloat16, "key cache dtype");
  require(cache.value_cache(0).dtype() == DataType::BFloat16,
          "value cache dtype");
  require(cache.conv_state(0).dtype() == DataType::BFloat16,
          "conv history dtype");
  require(cache.recurrent_state(0).dtype() == DataType::Float32,
          "recurrent cache dtype");
  require(cache.kv_bytes() == 2 * 4 * 16 * 2 * 16 * 2, "BF16 KV byte count");
}
} // namespace

int main() {
  try {
    for (auto m : {1, 3, 17, 129}) {
      check_linear(m, 33, 130);
      check_linear(m, 128, 2048);
    }
    check_linear(2, 3, 5);
    check_norm();
    check_delta();
    check_cache();
    std::cout
        << "Mixed precision checks passed: BF16/FP8 linear and FP32 logits, "
           "FP32 residual, 513-token FP64 recurrence reference, chunk/decode "
           "parity, cache dtypes\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
