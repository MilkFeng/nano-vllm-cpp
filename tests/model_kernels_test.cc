#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/tensor_layout.hh"
#include "models/qwen3.hh"

namespace {

using nanovllm::DataType;
using nanovllm::Device;
using nanovllm::Tensor;
using nanovllm::TensorLayout;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void test_tensor_layout_validation() {
  expect_invalid_argument([] { (void)TensorLayout({2, 3}, {3}); },
                          "TensorLayout accepted fewer strides than dimensions");
  expect_invalid_argument([] { (void)TensorLayout({2}, {2, 1}); },
                          "TensorLayout accepted more strides than dimensions");
  expect_invalid_argument([] { (void)TensorLayout({2, 3}, {0, 1}); },
                          "TensorLayout accepted a zero row stride");
  expect_invalid_argument([] { (void)TensorLayout({2, 3}, {3, 0}); },
                          "TensorLayout accepted a zero column stride");
  const TensorLayout singleton({1, 3}, {0, 1});
  require(singleton.is_contiguous(), "Singleton dimensions may have zero stride");
  const TensorLayout empty({0, 3}, {0, 0});
  require(empty.numel() == 0 && empty.is_contiguous(),
          "Empty layouts should permit zero strides");
}

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  ~Stream() { (void)cudaStreamDestroy(stream_); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  [[nodiscard]] cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

template <typename T, std::size_t Size>
void upload(Tensor& destination, const std::array<T, Size>& source,
             cudaStream_t stream) {
  require(destination.size_bytes() == sizeof(source), "Upload size mismatch");
  check_cuda(cudaMemcpyAsync(destination.data(), source.data(), sizeof(source),
                              cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync upload");
}

std::vector<float> download(const Tensor& tensor, cudaStream_t stream) {
  require(tensor.dtype() == DataType::Float32 && tensor.is_contiguous(),
          "Download requires a contiguous Float32 tensor");
  std::vector<float> result(tensor.numel());
  check_cuda(cudaMemcpyAsync(result.data(), tensor.data(), tensor.size_bytes(),
                              cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync download");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return result;
}

void expect_shape(const Tensor& tensor,
                   std::initializer_list<std::size_t> shape) {
  require(std::ranges::equal(tensor.shape(), shape), "Unexpected output shape");
  require(tensor.is_contiguous(), "Output must be contiguous");
  require(tensor.dtype() == DataType::Float32 &&
              tensor.device().type == nanovllm::DeviceType::Cuda,
          "Output must preserve its Float32 CUDA dtype/device");
}

template <typename Expected>
void expect_values(const Tensor& tensor, const Expected& expected,
                    cudaStream_t stream, const char* label) {
  const auto actual = download(tensor, stream);
  require(actual.size() == expected.size(), std::string(label) + " size mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = 2e-6f * std::max(1.0f, std::abs(expected[index]));
    require(std::isfinite(actual[index]) &&
                std::abs(actual[index] - expected[index]) <= tolerance,
            std::string(label) + " mismatch at index " + std::to_string(index)
                + ": expected " + std::to_string(expected[index])
                + ", got " + std::to_string(actual[index]));
  }
}

void test_rms_norm(Device device, cudaStream_t stream) {
  // Both rows have mean square 3; epsilon 1 makes their denominator exactly 2.
  const std::array weights{2.0f, -1.0f, 0.5f};
  const std::array inputs{1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 0.0f};
  const std::array residuals{1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 2.0f};
  nanovllm::Qwen3RMSNorm norm(3, 1.0f, DataType::Float32, device);
  auto input = Tensor::empty({2, 3}, DataType::Float32, device);
  auto residual = Tensor::empty({2, 3}, DataType::Float32, device);
  upload(norm.weight().tensor(), weights, stream);
  upload(input, inputs, stream);
  upload(residual, residuals, stream);

  auto normalized = norm.forward(input.view().as_const(), stream);
  expect_shape(normalized, {2, 3});
  expect_values(normalized, std::array{1.0f, -1.0f, 0.5f, 0.0f, -1.5f, 0.0f},
                  stream, "RMSNorm");

  auto fused = norm.forward(input.view().as_const(), residual.view().as_const(),
                            stream);
  expect_shape(fused.hidden_states, {2, 3});
  expect_shape(fused.residual, {2, 3});
  const float first_denominator = std::sqrt(5.0f);
  const float second_denominator = std::sqrt(11.0f / 3.0f);
  const std::array expected_fused{
      4.0f / first_denominator, -2.0f / first_denominator,
      1.0f / first_denominator, 0.0f, -2.0f / second_denominator,
      1.0f / second_denominator};
  expect_values(fused.hidden_states, expected_fused, stream, "Fused RMSNorm");
  expect_values(fused.residual,
                  std::array{2.0f, 2.0f, 2.0f, 0.0f, 2.0f, 2.0f}, stream,
                  "Fused residual");
  expect_values(input, inputs, stream, "Input preservation");
  expect_values(residual, residuals, stream, "Residual input preservation");
  require(fused.hidden_states.data() != fused.residual.data() &&
              fused.residual.data() != residual.data() &&
              fused.hidden_states.data() != input.data(),
          "Model RMSNorm outputs should own separate storage");

  // Simulate Q/K heads sliced from a wider packed projection. The token stride
  // is 9, while each selected token contains two heads of width 3.
  const std::array packed_values{
      99.0f, 99.0f, 99.0f, 1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 0.0f,
      99.0f, 99.0f, 99.0f, -2.0f, -2.0f, -2.0f, 0.0f, 0.0f, 0.0f};
  auto packed = Tensor::empty({2, 3, 3}, DataType::Float32, device);
  upload(packed, packed_values, stream);
  auto heads = packed.slice(1, 1, 2);
  auto normalized_heads = norm.forward(heads.view().as_const(), stream);
  expect_shape(normalized_heads, {2, 2, 3});
  const std::array expected_heads{
      1.0f, -1.0f, 0.5f, 0.0f, -1.5f, 0.0f,
      -4.0f / first_denominator, 2.0f / first_denominator,
      -1.0f / first_denominator, 0.0f, 0.0f, 0.0f};
  expect_values(normalized_heads, expected_heads, stream, "Per-head RMSNorm");
}

void test_embedding(Device device, cudaStream_t stream) {
  nanovllm::Qwen3Embedding embedding(4, 3, DataType::Float32, device);
  const std::array weights{
      0.5f, 1.0f, -1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, -8.0f, -9.0f, -10.0f};
  const std::array<std::int32_t, 4> ids{3, 1, 0, 3};
  auto input_ids = Tensor::empty({ids.size()}, DataType::Int32, device);
  upload(embedding.weight().tensor(), weights, stream);
  upload(input_ids, ids, stream);
  auto output = embedding.forward(input_ids.view().as_const(), stream);
  expect_shape(output, {4, 3});
  const std::array expected{
      -8.0f, -9.0f, -10.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.0f, -1.0f, -8.0f, -9.0f, -10.0f};
  expect_values(output, expected, stream, "Embedding");
  expect_values(embedding.weight().tensor(), weights, stream,
                  "Embedding weight preservation");
}

} // namespace

int main() {
  try {
    test_tensor_layout_validation();
    std::cout << "TensorLayout validation passed\n";
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
      std::cout << "Skipping CUDA model tests: "
                << (status == cudaSuccess ? "no CUDA devices"
                                           : cudaGetErrorString(status))
                << '\n';
      return 77;
    }
    const auto device = Device::cuda(0);
    check_cuda(cudaSetDevice(device.index), "cudaSetDevice");
    const Stream stream;
    test_rms_norm(device, stream.get());
    test_embedding(device, stream.get());
    std::cout << "Qwen3 RMSNorm and embedding integration tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Model kernel test failed: " << error.what() << '\n';
    return 1;
  }
}
