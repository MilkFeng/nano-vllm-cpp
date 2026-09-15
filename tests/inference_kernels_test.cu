#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kernels/inference.hh"

namespace {
using namespace nanovllm;
namespace kernels = nanovllm::kernels;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
constexpr DataType dtype() {
  if constexpr (std::is_same_v<T, float>) return DataType::Float32;
  if constexpr (std::is_same_v<T, __half>) return DataType::Float16;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) return DataType::BFloat16;
  return DataType::Int32;
}

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t count) : count_(count) {
    if (count) check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
  }
  ~Buffer() { if (data_) (void)cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* data() { return data_; }
  TensorView view(std::vector<std::size_t> shape) {
    return {data_, dtype<T>(), TensorLayout(std::move(shape)), Device::cuda(0)};
  }
  TensorView view(std::vector<std::size_t> shape, std::vector<std::size_t> strides) {
    return {data_, dtype<T>(), TensorLayout(std::move(shape), std::move(strides)), Device::cuda(0)};
  }
  void upload(const std::vector<T>& host, cudaStream_t stream) {
    if (host.size() != count_) throw std::runtime_error("upload size mismatch");
    if (count_) check(cudaMemcpyAsync(data_, host.data(), count_ * sizeof(T),
                                      cudaMemcpyHostToDevice, stream), "upload");
  }
  std::vector<T> download(cudaStream_t stream) {
    std::vector<T> result(count_);
    if (count_) check(cudaMemcpyAsync(result.data(), data_, count_ * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream), "download");
    check(cudaStreamSynchronize(stream), "synchronize");
    return result;
  }
 private:
  T* data_ = nullptr;
  std::size_t count_;
};

template <typename T>
void expect_close(const std::vector<T>& actual, const std::vector<float>& expected,
                   const char* label, bool exact = false) {
  if (actual.size() != expected.size()) throw std::runtime_error(std::string(label) + " size");
  const float tolerance = exact ? 0 : std::is_same_v<T, float> ? 1e-4f
      : std::is_same_v<T, __half> ? .002f : .012f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float got = static_cast<float>(actual[i]);
    if (!std::isfinite(got)
        || std::abs(got - expected[i]) > tolerance * (1 + std::abs(expected[i]))) {
      throw std::runtime_error(std::string(label) + " index " + std::to_string(i)
          + ": got " + std::to_string(got) + ", expected " + std::to_string(expected[i]));
    }
  }
}

template <typename F>
void expect_invalid(F&& call, const char* label) {
  try { call(); }
  catch (const std::invalid_argument&) { return; }
  throw std::runtime_error(std::string(label) + " accepted invalid metadata");
}

template <typename T>
void test_linear(std::size_t m, std::size_t n, std::size_t k, bool biased,
                  bool strided, cudaStream_t stream) {
  const auto stride = k + (strided ? 7 : 0);
  std::vector<T> x(m * stride), w(n * k), b(n);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = T(std::sin(i * .37) * .8);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = T(std::cos(i * .19) * .6);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = T(std::sin(i * .13) * .7);
  std::vector<float> expected(m * n);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double sum = biased ? static_cast<float>(b[col]) : 0;
      for (std::size_t d = 0; d < k; ++d) {
        sum += static_cast<double>(static_cast<float>(x[row * stride + d]))
               * static_cast<float>(w[col * k + d]);
      }
      expected[row * n + col] = static_cast<float>(T(static_cast<float>(sum)));
    }
  }
  Buffer<T> bx(x.size()), bw(w.size()), bb(b.size()), bo(expected.size());
  bx.upload(x, stream); bw.upload(w, stream); bb.upload(b, stream);
  const auto bias = biased ? std::optional<ConstTensorView>(bb.view({n}).as_const()) : std::nullopt;
  kernels::linear(bx.view({m, k}, {stride, 1}).as_const(), bw.view({n, k}).as_const(),
                   bo.view({m, n}), bias, stream);
  expect_close(bo.download(stream), expected, "linear");
}

template <typename T>
void test_linear_bias_precision(cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    const bool half = std::is_same_v<T, __half>;
    const std::vector<T> x = {T(half ? 1.822265625f : 1.5234375f)};
    const std::vector<T> w = {T(half ? .391357421875f : .98828125f)};
    const std::vector<T> b = {T(half ? 1.2197265625f : .404296875f)};
    Buffer<T> bx(1), bw(1), bb(1), bo(1);
    bx.upload(x, stream); bw.upload(w, stream); bb.upload(b, stream);
    kernels::linear(bx.view({1, 1}).as_const(), bw.view({1, 1}).as_const(), bo.view({1, 1}),
                    bb.view({1}).as_const(), stream);
    // Adding bias after a low-precision GEMM output would change these values.
    expect_close(bo.download(stream), {half ? 1.9326171875f : 1.90625f}, "linear bias rounding", true);
  }
}

template <typename T>
void test_gather(cudaStream_t stream) {
  constexpr std::size_t tokens = 11, width = 257, stride = width + 9;
  const std::vector<std::int32_t> boundaries = {0, 1, 4, 6, 11};
  constexpr std::size_t batch = 4;
  std::vector<T> hidden(tokens * stride);
  for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = T(std::sin(i * .07));
  std::vector<float> expected(batch * width);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t d = 0; d < width; ++d) {
      expected[b * width + d] = static_cast<float>(hidden[(boundaries[b + 1] - 1) * stride + d]);
    }
  }
  Buffer<T> bh(hidden.size()), bo(expected.size());
  Buffer<std::int32_t> bc(boundaries.size());
  bh.upload(hidden, stream); bc.upload(boundaries, stream);
  kernels::gather_last_hidden(bh.view({tokens, width}, {stride, 1}).as_const(),
                              bc.view({batch + 1}).as_const(), bo.view({batch, width}), stream);
  expect_close(bo.download(stream), expected, "gather", true);
}

template <typename T>
void test_sampling(cudaStream_t stream) {
  constexpr std::size_t batch = 64, vocab = 257, stride = vocab + 5;
  std::vector<T> logits(batch * stride);
  std::vector<float> temperatures(batch);
  std::vector<std::int32_t> greedy(batch);
  for (std::size_t b = 0; b < batch; ++b) {
    temperatures[b] = b % 3 == 0 ? 0 : b % 3 == 1 ? .5f : 1.5f;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t v = 0; v < vocab; ++v) {
      logits[b * stride + v] = T(v % 11 == 0 ? -std::numeric_limits<float>::infinity()
                                             : static_cast<float>(std::sin((b + 1) * v * .17)));
      if (v == 2 || v == 256) logits[b * stride + v] = T(1.25f);
      const float value = static_cast<float>(logits[b * stride + v]);
      if (value > maximum) { maximum = value; greedy[b] = static_cast<std::int32_t>(v); }
    }
  }
  Buffer<T> bl(logits.size());
  Buffer<float> bt(batch);
  Buffer<std::int32_t> bo(batch);
  bl.upload(logits, stream); bt.upload(temperatures, stream);
  auto run = [&](std::uint64_t step) {
    kernels::sample(bl.view({batch, vocab}, {stride, 1}).as_const(), bt.view({batch}).as_const(),
                     bo.view({batch}), 12345, step, stream);
    return bo.download(stream);
  };
  const auto first = run(7), repeat = run(7), next = run(8);
  if (first != repeat) throw std::runtime_error("sampling not deterministic for equal seed/step");
  bool changed = false;
  for (std::size_t b = 0; b < batch; ++b) {
    for (const auto* draw : {&first, &next}) {
      const auto token = (*draw)[b];
      if (token < 0 || token >= vocab || !std::isfinite(static_cast<float>(logits[b * stride + token]))) {
        throw std::runtime_error("sampling selected invalid or masked token");
      }
      if (temperatures[b] == 0 && token != greedy[b]) throw std::runtime_error("greedy/tie-break mismatch");
    }
    if (temperatures[b] != 0 && first[b] != next[b]) changed = true;
  }
  if (!changed) throw std::runtime_error("sampling step did not change any random draw");
}

void test_sampling_distribution(cudaStream_t stream) {
  constexpr std::size_t batch = 32766, vocab = 4, groups = 3;
  const float limit = std::numeric_limits<float>::max();
  const float masked = -std::numeric_limits<float>::infinity();
  const std::array<float, groups> temperature = {.75f, 1.5f, limit};
  const std::array<std::array<float, vocab>, groups> rows = {{
      {0, std::log(2.0f), std::log(4.0f), masked},
      {0, std::log(2.0f), std::log(4.0f), masked},
      {-limit, 0, limit, masked}}};
  std::vector<float> logits(batch * vocab), temperatures(batch);
  for (std::size_t b = 0; b < batch; ++b) {
    const auto& row = rows[b % groups];
    std::copy(row.begin(), row.end(), logits.begin() + b * vocab);
    temperatures[b] = temperature[b % groups];
  }
  Buffer<float> bl(logits.size()), bt(batch);
  Buffer<std::int32_t> bo(batch);
  bl.upload(logits, stream); bt.upload(temperatures, stream);
  kernels::sample(bl.view({batch, vocab}).as_const(), bt.view({batch}).as_const(),
                   bo.view({batch}), 90210, 13, stream);
  const auto output = bo.download(stream);
  std::array<std::array<std::size_t, vocab>, groups> counts{};
  for (std::size_t b = 0; b < batch; ++b) {
    if (output[b] < 0 || output[b] >= 3) throw std::runtime_error("distribution selected invalid token");
    ++counts[b % groups][output[b]];
  }
  for (std::size_t t = 0; t < groups; ++t) {
    double denominator = 0;
    for (std::size_t v = 0; v < 3; ++v) denominator += std::exp(rows[t][v] / temperature[t]);
    for (std::size_t v = 0; v < 3; ++v) {
      const double expected = std::exp(rows[t][v] / temperature[t]) / denominator;
      const double actual = static_cast<double>(counts[t][v]) / (batch / groups);
      if (std::abs(actual - expected) > .025) throw std::runtime_error("sampling frequency differs from softmax");
    }
  }
}

void test_sampling_grid_stride(cudaStream_t stream) {
  constexpr std::size_t batch = 65537, vocab = 3;
  std::vector<float> logits(batch * vocab, -std::numeric_limits<float>::infinity());
  std::vector<float> temperatures(batch);
  for (std::size_t b = 0; b < batch; ++b) {
    logits[b * vocab + b % vocab] = 2;
    temperatures[b] = b % 2 == 0 ? 0 : 1;
  }
  Buffer<float> bl(logits.size()), bt(batch);
  Buffer<std::int32_t> bo(batch);
  bl.upload(logits, stream); bt.upload(temperatures, stream);
  kernels::sample(bl.view({batch, vocab}).as_const(), bt.view({batch}).as_const(),
                   bo.view({batch}), 100, 0, stream);
  const auto output = bo.download(stream);
  for (std::size_t b = 0; b < batch; ++b) {
    if (output[b] != b % vocab) throw std::runtime_error("sampling grid-stride row mismatch");
  }
}

template <typename T>
void test_empty(cudaStream_t stream) {
  Buffer<T> empty(0), weight(12);
  Buffer<float> temperatures(0);
  Buffer<std::int32_t> boundaries(1), output(0);
  const std::vector<std::int32_t> empty_boundaries = {0};
  boundaries.upload(empty_boundaries, stream);
  kernels::linear(empty.view({0, 4}).as_const(), weight.view({3, 4}).as_const(),
                   empty.view({0, 3}), std::nullopt, stream);
  kernels::gather_last_hidden(empty.view({0, 4}).as_const(), boundaries.view({1}).as_const(),
                              empty.view({0, 4}), stream);
  kernels::sample(empty.view({0, 4}).as_const(), temperatures.view({0}).as_const(),
                   output.view({0}), 0, 0, stream);
}

void test_invalid(cudaStream_t stream) {
  Buffer<float> x(16), w(16), output(16), temperatures(2);
  Buffer<std::int32_t> ids(3);
  auto xv = x.view({2, 4}), wv = w.view({3, 4}), ov = output.view({2, 3});
  expect_invalid([&] { kernels::linear(xv.as_const(), w.view({3, 3}).as_const(), ov, std::nullopt, stream); }, "linear dimensions");
  expect_invalid([&] { kernels::linear(x.view({2, 4}, {3, 1}).as_const(), wv.as_const(), ov, std::nullopt, stream); }, "linear stride");
  expect_invalid([&] { kernels::linear(xv.as_const(), wv.as_const(), ov, w.view({2}).as_const(), stream); }, "linear bias");
  expect_invalid([&] { kernels::linear(xv.as_const(), wv.as_const(), x.view({2, 3}), std::nullopt, stream); }, "linear alias");
  const auto cpu = ConstTensorView(x.data(), DataType::Float32, TensorLayout({2, 4}), Device::cpu());
  expect_invalid([&] { kernels::linear(cpu, wv.as_const(), ov, std::nullopt, stream); }, "linear device");
  expect_invalid([&] { kernels::gather_last_hidden(xv.as_const(), ids.view({2}).as_const(), output.view({2, 4}), stream); }, "gather boundaries shape");
  expect_invalid([&] { kernels::gather_last_hidden(xv.as_const(), temperatures.view({3}).as_const(), output.view({2, 4}), stream); }, "gather dtype");
  expect_invalid([&] { kernels::gather_last_hidden(xv.as_const(), ids.view({3}).as_const(), xv, stream); }, "gather alias");
  expect_invalid([&] { kernels::gather_last_hidden(xv.as_const(), ids.view({1}).as_const(), output.view({0, 4}), stream); }, "gather empty mismatch");
  expect_invalid([&] { kernels::sample(xv.as_const(), temperatures.view({1}).as_const(), ids.view({2}), 0, 0, stream); }, "sampling batch");
  expect_invalid([&] { kernels::sample(xv.as_const(), temperatures.view({2}).as_const(), output.view({2}), 0, 0, stream); }, "sampling output dtype");
  expect_invalid([&] { kernels::sample(x.view({2, 0}).as_const(), temperatures.view({2}).as_const(), ids.view({2}), 0, 0, stream); }, "sampling vocabulary");
  const auto alias = TensorView(x.data(), DataType::Int32, TensorLayout({2}), Device::cuda(0));
  expect_invalid([&] { kernels::sample(xv.as_const(), temperatures.view({2}).as_const(), alias, 0, 0, stream); }, "sampling alias");
  check(cudaStreamSynchronize(stream), "validation synchronize");
}

template <typename T>
void run_dtype(cudaStream_t stream, const char* name) {
  test_linear<T>(3, 5, 7, false, false, stream);
  test_linear<T>(3, 5, 7, true, true, stream);
  test_linear<T>(1, 11, 257, true, true, stream);
  test_linear<T>(7, 129, 64, false, false, stream);
  test_linear_bias_precision<T>(stream);
  test_gather<T>(stream);
  test_sampling<T>(stream);
  test_empty<T>(stream);
  std::cout << "PASS " << name << " linear, gather, sampling, and empty batches\n";
}
} // namespace

int main() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && count == 0)) {
    std::cout << "SKIP: CUDA GPU unavailable\n";
    return 77;
  }
  cudaStream_t stream = nullptr;
  try {
    check(status, "cudaGetDeviceCount");
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    run_dtype<float>(stream, "Float32");
    run_dtype<__half>(stream, "Float16");
    run_dtype<__nv_bfloat16>(stream, "BFloat16");
    test_sampling_distribution(stream);
    test_sampling_grid_stride(stream);
    test_invalid(stream);
    check(cudaStreamDestroy(stream), "destroy stream");
    std::cout << "PASS sampling distribution, grid-stride, and host validation\n";
    return 0;
  } catch (const std::exception& error) {
    if (stream) (void)cudaStreamDestroy(stream);
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
