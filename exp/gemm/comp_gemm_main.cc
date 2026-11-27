#include <CLI/CLI.hpp>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "comp_gemm.hh"
#include "core/tensor.hh"
#include "gemm_benchmark.hh"
#include "kernels/cuda/kernels.hh"

namespace {

using nanovllm::check_cuda;
using nanovllm::ConstTensorView;
using nanovllm::DataType;
using nanovllm::Device;
using nanovllm::Tensor;
using nanovllm::TensorLayout;
using nanovllm::TensorView;

namespace kernels = nanovllm::kernels;
namespace baseline = nanovllm::exp;

struct Shape {
  std::size_t m;
  std::size_t n;
  std::size_t k;
};

struct Stream {
  cudaStream_t value = nullptr;

  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
               "create comparison stream");
  }

  ~Stream() { cudaStreamDestroy(value); }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

template <typename T>
std::vector<T> random_values(std::size_t count, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<T> values(count);

  for (T& value : values) {
    value = static_cast<T>(distribution(generator));
  }

  return values;
}

template <typename T>
void upload(TensorView destination, const std::vector<T>& source,
            cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                             source.size() * sizeof(T), cudaMemcpyHostToDevice,
                             stream),
             "upload comparison input");
}

template <typename T>
std::vector<T> download(const Tensor& source, cudaStream_t stream) {
  std::vector<T> values(source.numel());

  check_cuda(cudaMemcpyAsync(values.data(), source.data(), source.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "download comparison output");
  check_cuda(cudaStreamSynchronize(stream), "finish comparison");

  return values;
}

void require_close(float actual, double expected, DataType dtype) {
  const double tolerance = dtype == DataType::Float32   ? 2e-4
                           : dtype == DataType::Float16 ? 2e-3
                                                        : 1.6e-2;

  if (!std::isfinite(actual)
      || std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected))) {
    throw std::runtime_error("linear mismatch: actual=" + std::to_string(actual)
                             + ", expected=" + std::to_string(expected));
  }
}

template <typename T>
void check_case(Shape shape, DataType dtype, bool with_bias, bool unaligned,
                cudaStream_t stream) {
  const auto [m, n, k] = shape;
  const std::size_t offset = unaligned ? 1 : 4;

  auto input_storage =
      Tensor::empty({m * k + offset + 1}, dtype, Device::cuda());
  auto weight_storage =
      Tensor::empty({n * k + offset + 1}, dtype, Device::cuda());
  auto output_storage =
      Tensor::empty({m * n + offset + 1}, dtype, Device::cuda());
  auto bias_storage = Tensor::empty({n + offset + 1}, dtype, Device::cuda());

  const auto view = [&](Tensor& storage, std::vector<std::size_t> dimensions) {
    return TensorView(static_cast<T*>(storage.data()) + offset, dtype,
                      TensorLayout(std::move(dimensions)), Device::cuda());
  };

  auto input = view(input_storage, {m, k});
  auto weight = view(weight_storage, {n, k});
  auto output = view(output_storage, {m, n});
  auto bias = view(bias_storage, {n});

  const auto x = random_values<T>(m * k, 17);
  const auto w = random_values<T>(n * k, 29);
  const auto b = random_values<T>(n, 43);
  const std::vector<T> sentinel(output_storage.numel(), static_cast<T>(123.0f));

  upload(input, x, stream);
  upload(weight, w, stream);
  upload(bias, b, stream);
  upload(output_storage.view(), sentinel, stream);

  const auto optional_bias =
      with_bias ? std::optional<ConstTensorView>(bias.as_const())
                : std::nullopt;

  kernels::cuda::linear(input.as_const(), weight.as_const(), output,
                        optional_bias, stream);

  const auto result = download<T>(output_storage, stream);

  for (std::size_t i = 0; i < offset; ++i) {
    if (static_cast<float>(result[i]) != 123.0f) {
      throw std::runtime_error("linear overwrote the output prefix guard");
    }
  }

  if (static_cast<float>(result.back()) != 123.0f) {
    throw std::runtime_error("linear overwrote the output suffix guard");
  }

  for (std::size_t i = 0; i < m * n; ++i) {
    if (!std::isfinite(static_cast<float>(result[offset + i]))) {
      throw std::runtime_error("linear produced a non-finite output");
    }
  }

  if (dtype == DataType::Float32 && !with_bias) {
    baseline::tiled_linear(input.as_const(), weight.as_const(), output, stream);
    const auto tiled = download<T>(output_storage, stream);

    for (std::size_t i = 0; i < result.size(); ++i) {
      require_close(static_cast<float>(tiled[i]), static_cast<float>(result[i]), dtype);
    }
  }

  // A separate FP64 dot product checks layout, tails and fused bias.
  // Sample across the entire matrix, including its first and last elements.
  const std::size_t samples = std::min<std::size_t>(m * n, 1024);

  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t index =
        samples == 1 ? 0 : sample * (m * n - 1) / (samples - 1);
    const std::size_t row = index / n;
    const std::size_t col = index % n;
    double expected = with_bias ? static_cast<float>(b[col]) : 0.0;

    for (std::size_t t = 0; t < k; ++t) {
      expected += static_cast<double>(static_cast<float>(x[row * k + t]))
                  * static_cast<float>(w[col * k + t]);
    }

    require_close(static_cast<float>(result[offset + index]), expected, dtype);
  }

  // cuBLASLt's half/bfloat16 path has stricter alignment requirements.
  const bool compare_cublas =
      dtype == DataType::Float32 || (!unaligned && n % 8 == 0 && k % 8 == 0);

  if (compare_cublas) {
    auto reference = Tensor::empty({m, n}, dtype, Device::cuda());

    baseline::cublas_linear(input.as_const(), weight.as_const(),
                            reference.view(), optional_bias, stream);

    const auto expected = download<T>(reference, stream);

    for (std::size_t i = 0; i < m * n; ++i) {
      require_close(static_cast<float>(result[offset + i]),
                    static_cast<float>(expected[i]), dtype);
    }
  }
}

template <typename T>
void check_dtype(DataType dtype, cudaStream_t stream) {
  constexpr std::array shapes = {
      Shape{0, 17, 13},     Shape{1, 7, 3},       Shape{1, 129, 257},
      Shape{2, 65, 17},     Shape{7, 127, 31},    Shape{8, 128, 64},
      Shape{9, 131, 33},    Shape{31, 128, 129},  Shape{33, 129, 15},
      Shape{127, 129, 257}, Shape{128, 128, 16},  Shape{128, 256, 64},
      Shape{129, 257, 131}, Shape{256, 512, 256}, Shape{128, 128, 1024}};

  for (Shape shape : shapes) {
    for (bool with_bias : {false, true}) {
      for (bool unaligned : {false, true}) {
        check_case<T>(shape, dtype, with_bias, unaligned, stream);
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  baseline::GemmBenchmarkOptions options;
  bool check_only = false;
  CLI::App app{"FP32 X @ W^T benchmark with explicit baselines and raw samples"};

  app.add_flag("--check-only", check_only, "Run correctness checks without timing");
  app.add_option("--suite", options.suite, "Predeclared shape suite")
      ->check(CLI::IsMember({"quick", "full"}))->capture_default_str();
  app.add_option("--cache", options.cache, "Reuse one set or rotate addresses")
      ->check(CLI::IsMember({"hot", "rotating", "both"}))->capture_default_str();
  app.add_option("--output", options.output, "Raw JSON result path")->capture_default_str();
  app.add_option("--seed", options.seed, "Input and measurement-order seed")
      ->capture_default_str();
  app.add_option("--samples", options.samples, "Event batches per implementation")
      ->check(CLI::Range(3, 1000))->capture_default_str();
  app.add_option("--warmup", options.warmup, "Warmup calls, at least one rotation")
      ->check(CLI::Range(1, 10000))->capture_default_str();
  app.add_option("--target-ms", options.target_ms, "Target duration of each event batch")
      ->check(CLI::Range(1.0, 1000.0))->capture_default_str();
  app.add_option("--max-calls", options.max_calls, "Maximum GEMMs per captured batch")
      ->check(CLI::Range(1, 4096))->capture_default_str();
  app.add_option("--workspace-mib", options.workspace_mib, "Workspace limit for each BLAS plan")
      ->check(CLI::Range(0, 1024))->capture_default_str();
  app.add_option("--heuristic-candidates", options.heuristic_candidates,
                 "Maximum cuBLASLt heuristic candidates to time")
      ->check(CLI::Range(1, 64))->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  try {
    Stream stream;
    cudaDeviceProp properties{};

    check_cuda(cudaGetDeviceProperties(&properties, 0), "read CUDA device");
    std::cout << "GPU: " << properties.name << '\n';

    check_dtype<float>(DataType::Float32, stream.value);
    check_dtype<__half>(DataType::Float16, stream.value);
    check_dtype<__nv_bfloat16>(DataType::BFloat16, stream.value);

    std::cout << "PASS: 180 production + 30 tiled-baseline correctness cases.\n"
              << std::flush;

    if (!check_only) {
      baseline::run_gemm_benchmark(options, stream.value);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
