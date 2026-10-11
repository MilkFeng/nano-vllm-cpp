#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cpu/moe.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/moe.hh"

namespace {

using namespace nanovllm;
using kernels::cpu::ExpertWeights;
using kernels::cpu::MoEReferenceOutput;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

float compare(std::span<const float> actual, std::span<const float> expected,
              const std::string& label, float atol = 2e-5f,
              float rtol = 2e-4f) {
  require(actual.size() == expected.size(), label + ": size mismatch");
  float maximum = 0.0f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(actual[i] - expected[i]);
    require(std::isfinite(actual[i]) && std::isfinite(expected[i])
                && error <= atol + rtol * std::abs(expected[i]),
            label + ": mismatch at element " + std::to_string(i));
    maximum = std::max(maximum, error);
  }
  return maximum;
}

template <typename Function>
void expect_invalid(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("expected std::invalid_argument");
}

Tensor cpu_tensor(std::vector<std::size_t> shape,
                  std::span<const float> values) {
  auto tensor = Tensor::empty(std::move(shape), DataType::Float32);
  require(tensor.numel() == values.size(), "CPU fixture size mismatch");
  std::memcpy(tensor.data(), values.data(), tensor.size_bytes());
  return tensor;
}

void set_parameter(MoE& moe, const std::string& name,
                   std::span<const float> values) {
  auto view = moe.parameter(name).view();
  require(view.device() == Device::cpu() && view.numel() == values.size(),
          "CPU parameter fixture mismatch");
  std::memcpy(view.data(), values.data(), view.size_bytes());
}

std::vector<ExpertWeights> expert_views(const MoE& moe) {
  std::vector<ExpertWeights> views;
  for (std::size_t id = 0; id < moe.config().num_experts; ++id) {
    const auto prefix = "experts." + std::to_string(id) + ".";
    views.push_back({moe.parameter(prefix + "gate_proj.weight").view(),
                     moe.parameter(prefix + "up_proj.weight").view(),
                     moe.parameter(prefix + "down_proj.weight").view()});
  }
  return views;
}

MoEReferenceOutput reference(const MoE& moe, const Tensor& input) {
  const auto experts = expert_views(moe);
  return kernels::cpu::moe_reference(input.view(),
                                     moe.parameter("gate.weight").view(),
                                     experts, moe.config().top_k);
}

// A fixed integer generator avoids dependency on standard-library float RNGs.
float next_value(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return static_cast<float>(state >> 8) / 16777216.0f * 2.0f - 1.0f;
}

void initialize(MoE& moe, std::uint32_t seed, bool tied_router = false) {
  for (const auto& [name, parameter] : moe.named_parameters()) {
    auto view = parameter->view();
    const auto width = view.shape()[1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(width));
    for (std::size_t i = 0; i < view.numel(); ++i) {
      view.data<float>()[i] = tied_router && name == "gate.weight"
                                  ? 0.0f
                                  : next_value(seed) * scale;
    }
  }
}

Tensor random_input(std::size_t hidden_size, std::uint32_t seed) {
  std::vector<float> values(hidden_size);
  for (auto& value : values) value = next_value(seed);
  return cpu_tensor({1, hidden_size}, values);
}

void check_reference(const MoEReferenceOutput& result,
                     const MoEConfig& config) {
  require(
      result.hidden_states.size() == config.hidden_size
          && result.router_logits.size() == config.num_experts
          && result.expert_ids.size() == config.top_k
          && result.expert_outputs.size() == config.top_k * config.hidden_size,
      "reference output shape mismatch");
  auto sorted_ids = result.expert_ids;
  std::ranges::sort(sorted_ids);
  require(std::adjacent_find(sorted_ids.begin(), sorted_ids.end())
              == sorted_ids.end(),
          "router selected an expert twice");
  const float sum = std::accumulate(result.expert_weights.begin(),
                                    result.expert_weights.end(), 0.0f);
  require(std::abs(sum - 1.0f) < 2e-6f, "router weights do not sum to one");
  for (const auto value : result.hidden_states) {
    require(std::isfinite(value), "reference output is not finite");
  }
}

void initialize_analytic(MoE& moe) {
  for (const auto& [name, parameter] : moe.named_parameters()) {
    auto view = parameter->view();
    std::fill_n(view.data<float>(), view.numel(), 0.0f);
  }
  const std::vector<float> gate{1.0f, 0.0f}, up{0.0f, 1.0f};
  const std::vector<float> down0{3.0f, 5.0f}, down1{-1.0f, 2.0f};
  const std::vector<float> unused{1000.0f, -1000.0f};
  for (int id = 0; id < 3; ++id) {
    const auto prefix = "experts." + std::to_string(id) + ".";
    set_parameter(moe, prefix + "gate_proj.weight", gate);
    set_parameter(moe, prefix + "up_proj.weight", up);
    set_parameter(moe, prefix + "down_proj.weight",
                  id == 0   ? down0
                  : id == 1 ? down1
                            : unused);
  }
}

void test_cpu() {
  MoE analytic({2, 1, 3, 2}, Device::cpu());
  initialize_analytic(analytic);
  const std::vector<float> values{1.0f, 2.0f};
  const auto input = cpu_tensor({1, 2}, values);
  const auto result = reference(analytic, input);
  require(result.expert_ids == std::vector<std::int32_t>{0, 1},
          "equal router scores should choose experts 0 and 1");
  const float silu_one = 1.0f / (1.0f + std::exp(-1.0f));
  const std::vector<float> expected{2.0f * silu_one, 7.0f * silu_one};
  compare(result.hidden_states, expected, "hand-computed MoE", 1e-6f, 1e-6f);
  check_reference(result, analytic.config());

  for (const auto config : {MoEConfig{128, 64, 16, 2}, MoEConfig{7, 5, 5, 1},
                            MoEConfig{9, 7, 5, 5}, MoEConfig{3, 2, 1, 1}}) {
    MoE moe(config, Device::cpu());
    initialize(moe, 42);
    check_reference(reference(moe, random_input(config.hidden_size, 7)),
                    config);
  }
  expect_invalid([] { MoE moe({0, 64, 16, 2}, Device::cpu()); });
  expect_invalid([] { MoE moe({128, 0, 16, 2}, Device::cpu()); });
  expect_invalid([] { MoE moe({128, 64, 0, 1}, Device::cpu()); });
  expect_invalid([] { MoE moe({128, 64, 1025, 2}, Device::cpu()); });
  expect_invalid([] { MoE moe({128, 64, 16, 0}, Device::cpu()); });
  expect_invalid([] { MoE moe({128, 64, 16, 17}, Device::cpu()); });
  const auto experts = expert_views(analytic);
  expect_invalid([&] {
    (void)kernels::cpu::moe_reference(
        input.view(), analytic.parameter("gate.weight").view().as_const(),
        experts, 4);
  });
  std::cout
      << "CPU reference: analytical result, top-k limits, shapes passed\n";
}

struct Stream {
  cudaStream_t handle = nullptr;
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking),
               "create test stream");
  }
  ~Stream() { cudaStreamDestroy(handle); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

Tensor upload(const Tensor& cpu, cudaStream_t stream) {
  auto gpu = Tensor::empty({cpu.shape().begin(), cpu.shape().end()},
                           cpu.dtype(), Device::cuda(0));
  check_cuda(cudaMemcpyAsync(gpu.data(), cpu.data(), cpu.size_bytes(),
                             cudaMemcpyHostToDevice, stream),
             "upload fixture");
  check_cuda(cudaStreamSynchronize(stream), "complete fixture upload");
  return gpu;
}

template <typename T>
std::vector<T> download(const Tensor& gpu, cudaStream_t stream) {
  std::vector<T> values(gpu.numel());
  check_cuda(cudaMemcpyAsync(values.data(), gpu.data(), gpu.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "download result");
  check_cuda(cudaStreamSynchronize(stream), "complete result download");
  return values;
}

void test_route(const std::vector<float>& scores, std::size_t top_k,
                const std::vector<std::int32_t>& expected_ids,
                cudaStream_t stream) {
  const auto host_logits = cpu_tensor({1, scores.size()}, scores);
  const auto logits = upload(host_logits, stream);
  auto ids = Tensor::empty({1, top_k}, DataType::Int32, Device::cuda(0));
  auto weights = Tensor::empty({1, top_k}, DataType::Float32, Device::cuda(0));
  kernels::cuda::moe_route(logits.view(), ids.view(), weights.view(), stream);
  require(download<std::int32_t>(ids, stream) == expected_ids,
          "GPU route picked wrong experts");
  const auto maximum = scores[expected_ids[0]];
  std::vector<float> expected_weights;
  double sum = 0.0;
  for (const auto id : expected_ids)
    sum += std::exp(double(scores[id]) - maximum);
  for (const auto id : expected_ids) {
    expected_weights.push_back(
        static_cast<float>(std::exp(double(scores[id]) - maximum) / sum));
  }
  compare(download<float>(weights, stream), expected_weights, "route weights",
          2e-7f, 2e-6f);
}

void test_combine(cudaStream_t stream) {
  constexpr std::size_t hidden_size = 513;
  std::vector<float> parts(3 * hidden_size), expected(hidden_size);
  const std::vector<float> weights{0.2f, 0.3f, 0.5f};
  for (std::size_t d = 0; d < hidden_size; ++d) {
    parts[d] = static_cast<float>(d % 7) - 3.0f;
    parts[hidden_size + d] = -2.0f;
    parts[2 * hidden_size + d] = 10.0f;
    expected[d] = weights[0] * parts[d] - 0.6f + 5.0f;
  }
  const auto host_parts = cpu_tensor({3, hidden_size}, parts);
  const auto host_weights = cpu_tensor({1, 3}, weights);
  const auto gpu_parts = upload(host_parts, stream);
  const auto gpu_weights = upload(host_weights, stream);
  auto output =
      Tensor::empty({1, hidden_size}, DataType::Float32, Device::cuda(0));
  kernels::cuda::moe_combine(gpu_parts.view(), gpu_weights.view(),
                             output.view(), stream);
  compare(download<float>(output, stream), expected, "combine", 2e-6f, 2e-6f);
}

void check_gpu_moe(const MoE& cpu, const Tensor& input, cudaStream_t stream,
                   const std::string& label) {
  const auto expected = reference(cpu, input);
  for (const auto mode :
       {ExpertExecutionMode::Synchronous, ExpertExecutionMode::Overlap}) {
    MoE gpu(cpu.config(), Device::cuda(0), std::nullopt, mode);
    for (const auto& [name, parameter] : cpu.named_parameters()) {
      const auto source = parameter->view();
      auto destination = gpu.parameter(name).view();
      if (destination.device() == Device::cpu()) {
        std::memcpy(destination.data(), source.data(), source.size_bytes());
      } else {
        check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                   source.size_bytes(), cudaMemcpyHostToDevice,
                                   stream),
                   "upload MoE parameter");
      }
    }
    gpu.populate_expert_cache(stream);
    const auto gpu_input = upload(input, stream);
    const auto actual = gpu.forward(gpu_input.view(), stream);
    require(
        actual.gpu_experts == cpu.config().top_k && actual.cpu_experts == 0,
        label + ": default cache should execute all selected experts on GPU");
    require(download<std::int32_t>(actual.expert_ids, stream)
                == expected.expert_ids,
            label + ": expert ID mismatch");
    compare(download<float>(actual.router_logits, stream),
            expected.router_logits, label + " logits");
    compare(download<float>(actual.expert_weights, stream),
            expected.expert_weights, label + " weights", 2e-7f, 2e-6f);
    compare(download<float>(actual.expert_outputs, stream),
            expected.expert_outputs, label + " expert outputs");
    const auto maximum = compare(download<float>(actual.hidden_states, stream),
                                 expected.hidden_states, label + " output");
    std::cout << (mode == ExpertExecutionMode::Overlap ? "overlap " : "sync ")
              << label << ": max output error = " << maximum << '\n';

    // Repeating forward also exercises temporary-buffer lifetimes and reuse.
    const auto repeated = gpu.forward(gpu_input.view(), stream);
    compare(download<float>(repeated.hidden_states, stream),
            download<float>(actual.hidden_states, stream), label + " repeated",
            0.0f, 0.0f);
  }
}

int test_gpu() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && count == 0)) {
    std::cout << "CUDA tests skipped: no usable CUDA device\n";
    return 77;
  }
  check_cuda(status, "enumerate test GPUs");
  check_cuda(cudaSetDevice(0), "select test GPU");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0),
             "read test GPU properties");
  std::cout << "GPU: " << properties.name << '\n';
  Stream stream;
  test_route({0, 0, 0, 0, 0}, 3, {0, 1, 2}, stream.handle);
  test_route({-10000, 10000, 9999, 10000}, 2, {1, 3}, stream.handle);
  test_route({-10000, -10001, -10002}, 3, {0, 1, 2}, stream.handle);
  test_route({-1e-8f, 0.0f, -1.0f}, 2, {1, 0}, stream.handle);
  for (const std::size_t experts : {1u, 17u, 257u, 513u, 1024u}) {
    std::vector<float> scores(experts);
    std::iota(scores.begin(), scores.end(), 0.0f);
    test_route(scores, 1, {static_cast<std::int32_t>(experts - 1)},
               stream.handle);
  }
  test_combine(stream.handle);
  std::cout << "CUDA route/combine: ties, extreme scores, multi-block widths "
               "passed\n";

  MoE analytic({2, 1, 3, 2}, Device::cpu());
  initialize_analytic(analytic);
  const std::vector<float> values{1.0f, 2.0f};
  check_gpu_moe(analytic, cpu_tensor({1, 2}, values), stream.handle,
                "analytic");
  for (const auto config : {MoEConfig{128, 64, 16, 2}, MoEConfig{7, 5, 5, 1},
                            MoEConfig{9, 7, 5, 5}, MoEConfig{3, 2, 1, 1}}) {
    for (const auto seed : {7u, 42u}) {
      MoE cpu(config, Device::cpu());
      initialize(cpu, seed);
      check_gpu_moe(
          cpu, random_input(config.hidden_size, seed + 1), stream.handle,
          "E=" + std::to_string(config.num_experts) + " K="
              + std::to_string(config.top_k) + " seed=" + std::to_string(seed));
    }
  }
  MoE tied({128, 64, 16, 2}, Device::cpu());
  initialize(tied, 11, true);
  check_gpu_moe(tied, random_input(128, 12), stream.handle, "tied router");
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--cpu") {
      test_cpu();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--gpu") return test_gpu();
    std::cerr << "Usage: nano-vllm-cpp-moe-test --cpu|--gpu\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "MoE test failed: " << error.what() << '\n';
    return 1;
  }
}
