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
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cpu/expert.hh"
#include "kernels/cpu/moe.hh"
#include "layers/moe.hh"

namespace {

using namespace nanovllm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error("expected exception was not thrown");
}

float compare(std::span<const float> actual, std::span<const float> expected,
              const std::string& label) {
  require(actual.size() == expected.size(), label + ": size mismatch");
  float maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto error = std::abs(actual[i] - expected[i]);
    require(std::isfinite(actual[i]) && std::isfinite(expected[i])
                && error <= 2e-5f + 2e-4f * std::abs(expected[i]),
            label + ": mismatch at " + std::to_string(i));
    maximum = std::max(maximum, error);
  }
  return maximum;
}

void set_values(Module& module, const std::string& name,
                const std::vector<float>& values) {
  auto destination = module.parameter(name).view();
  require(destination.device() == Device::cpu()
              && destination.numel() == values.size(),
          "CPU parameter fixture mismatch");
  std::memcpy(destination.data(), values.data(), destination.size_bytes());
}

struct Fixture {
  MoE cpu{{128, 64, 16, 2}, Device::cpu()};
  Tensor input = Tensor::empty({1, 128}, DataType::Float32);
  Tensor other_input = Tensor::empty({1, 128}, DataType::Float32);

  Fixture() {
    std::uint32_t state = 123;
    for (const auto& [name, parameter] : cpu.named_parameters()) {
      auto view = parameter->view();
      const auto scale = 1.0f / std::sqrt(static_cast<float>(view.shape()[1]));
      for (std::size_t i = 0; i < view.numel(); ++i) {
        state = state * 1664525u + 1013904223u;
        view.data<float>()[i] =
            (float(state >> 8) / 16777216.0f * 2 - 1) * scale;
      }
    }
    auto router = cpu.parameter("gate.weight").view();
    std::fill_n(router.data<float>(), router.numel(), 0.0f);
    router.data<float>()[7 * 128] = 5;
    router.data<float>()[2 * 128] = 3;
    router.data<float>()[9 * 128 + 1] = 6;
    router.data<float>()[4 * 128 + 1] = 2;
    for (std::size_t d = 0; d < 128; ++d) {
      input.view().data<float>()[d] = (static_cast<int>(d % 9) - 4) * 0.1f;
    }
    input.view().data<float>()[0] = 1;
    input.view().data<float>()[1] = 0;
    std::memcpy(other_input.data(), input.data(), input.size_bytes());
    other_input.view().data<float>()[0] = 0;
    other_input.view().data<float>()[1] = 1;
  }
};

kernels::cpu::MoEReferenceOutput reference(const MoE& cpu, const Tensor& input,
                                           std::size_t top_k = 2) {
  std::vector<kernels::cpu::ExpertWeights> experts;
  for (std::size_t id = 0; id < cpu.config().num_experts; ++id) {
    experts.push_back(cpu.expert_store().weights(id));
  }
  return kernels::cpu::moe_reference(
      input.view(), cpu.parameter("gate.weight").view(), experts, top_k);
}

void test_cpu() {
  ExpertStore analytic(2, 1, 1);
  set_values(analytic, "0.gate_proj.weight", {1, 0});
  set_values(analytic, "0.up_proj.weight", {0, 1});
  set_values(analytic, "0.down_proj.weight", {3, 5});
  auto x = Tensor::empty({1, 2}, DataType::Float32);
  x.view().data<float>()[0] = 1;
  x.view().data<float>()[1] = 2;
  auto output = Tensor::empty({1, 2}, DataType::Float32);
  kernels::cpu::expert_forward(x.view().as_const(), analytic.weights(0),
                               output.view());
  const float s = 1.0f / (1.0f + std::exp(-1.0f));
  const std::vector<float> hand_computed{6 * s, 10 * s};
  compare({output.view().data<float>(), 2}, hand_computed,
          "analytical CPU expert");

  Fixture fixture;
  const auto& store = fixture.cpu.expert_store();
  require(store.size_bytes() == 16 * 3 * 128 * 64 * sizeof(float),
          "RAM weight byte count");
  require(store.named_parameters().size() == 16 * 3, "RAM parameter count");
  const auto expected = reference(fixture.cpu, fixture.input, 16);
  auto result = Tensor::empty({1, 128}, DataType::Float32);
  float maximum = 0;
  for (std::size_t rank = 0; rank < 16; ++rank) {
    kernels::cpu::expert_forward(fixture.input.view().as_const(),
                                 store.weights(expected.expert_ids[rank]),
                                 result.view());
    maximum = std::max(
        maximum,
        compare({result.view().data<float>(), 128},
                std::span(expected.expert_outputs).subspan(rank * 128, 128),
                "FP32 CPU expert"));
  }
  expect_throw<std::invalid_argument>([] { ExpertStore invalid(0, 1, 1); });
  expect_throw<std::out_of_range>([&] { (void)store.weights(16); });
  for (const auto& ids : {std::vector<std::int32_t>{-1}, {16}, {2, 2}}) {
    expect_throw<std::invalid_argument>(
        [&] { ExpertCache invalid(store, ids); });
  }
  ExpertCache empty(store, {});
  require(empty.ready() && empty.num_slots() == 0 && empty.size_bytes() == 0,
          "empty cache should allocate no GPU expert weights");
  require(empty.slot_of(7) == -1, "empty cache lookup");
  ExpertStore other(128, 64, 16);
  expect_throw<std::invalid_argument>(
      [&] { ExpertExecutor invalid(other, empty); });
  auto wrong_shape = Tensor::empty({2, 2}, DataType::Float32);
  expect_throw<std::invalid_argument>([&] {
    kernels::cpu::expert_forward(x.view().as_const(), analytic.weights(0),
                                 wrong_shape.view());
  });
  std::cout << "CPU experts/store/cache validation passed; max expert error="
            << maximum << '\n';
}

struct Stream {
  cudaStream_t handle = nullptr;
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking),
               "create offload test stream");
  }
  ~Stream() { cudaStreamDestroy(handle); }
};

Tensor upload(const Tensor& cpu, cudaStream_t stream) {
  auto gpu =
      Tensor::empty({1, cpu.shape()[1]}, DataType::Float32, Device::cuda(0));
  check_cuda(cudaMemcpyAsync(gpu.data(), cpu.data(), cpu.size_bytes(),
                             cudaMemcpyHostToDevice, stream),
             "upload input");
  check_cuda(cudaStreamSynchronize(stream), "complete input upload");
  return gpu;
}

template <typename T>
std::vector<T> download(const Tensor& gpu, cudaStream_t stream) {
  std::vector<T> values(gpu.numel());
  check_cuda(cudaMemcpyAsync(values.data(), gpu.data(), gpu.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "download output");
  check_cuda(cudaStreamSynchronize(stream), "complete output download");
  return values;
}

void copy_parameters(const MoE& cpu, MoE& gpu, cudaStream_t stream) {
  require(gpu.named_parameters().size() == 1 + 3 * cpu.config().num_experts,
          "cache copies must not be registered as model parameters");
  for (const auto& [name, parameter] : cpu.named_parameters()) {
    const auto source = parameter->view();
    auto destination = gpu.parameter(name).view();
    const auto expected_device =
        name == "gate.weight" ? Device::cuda(0) : Device::cpu();
    require(destination.device() == expected_device,
            "wrong parameter residency: " + name);
    if (destination.device() == Device::cpu()) {
      std::memcpy(destination.data(), source.data(), source.size_bytes());
    } else {
      check_cuda(
          cudaMemcpyAsync(destination.data(), source.data(),
                          source.size_bytes(), cudaMemcpyHostToDevice, stream),
          "upload router");
    }
  }
}

void check_forward(const MoE& cpu, const MoE& gpu, const Tensor& input,
                   cudaStream_t stream, const std::string& label) {
  const auto expected = reference(cpu, input);
  const auto gpu_input = upload(input, stream);
  const auto actual = gpu.forward(gpu_input.view(), stream);
  require(
      download<std::int32_t>(actual.expert_ids, stream) == expected.expert_ids,
      label + ": wrong expert IDs");
  compare(download<float>(actual.router_logits, stream), expected.router_logits,
          label + " logits");
  compare(download<float>(actual.expert_weights, stream),
          expected.expert_weights, label + " weights");
  compare(download<float>(actual.expert_outputs, stream),
          expected.expert_outputs, label + " unweighted rows");
  const float maximum =
      compare(download<float>(actual.hidden_states, stream),
              expected.hidden_states, label + " combined output");
  std::size_t expected_hits = 0;
  for (const auto id : expected.expert_ids) {
    if (std::ranges::find(gpu.expert_cache().resident_ids(), id)
        != gpu.expert_cache().resident_ids().end())
      ++expected_hits;
  }
  require(actual.gpu_experts == expected_hits
              && actual.cpu_experts == 2 - expected_hits,
          label + ": wrong CPU/GPU execution counts");
  const auto repeated = gpu.forward(gpu_input.view(), stream);
  require(download<float>(repeated.hidden_states, stream)
              == download<float>(actual.hidden_states, stream),
          label + ": repeat changed output");
  std::cout << label << ": selected=[" << expected.expert_ids[0] << ','
            << expected.expert_ids[1] << "] GPU=" << actual.gpu_experts
            << " CPU=" << actual.cpu_experts
            << " slots=" << gpu.expert_cache().num_slots()
            << " GPU weight bytes=" << gpu.expert_cache().size_bytes()
            << " max output error=" << maximum << '\n';
}

void check_layout(const Fixture& fixture,
                  std::vector<std::int32_t> resident_ids, cudaStream_t stream,
                  const std::string& label) {
  MoE gpu(fixture.cpu.config(), Device::cuda(0), resident_ids);
  copy_parameters(fixture.cpu, gpu, stream);
  require(gpu.expert_cache().num_slots() == resident_ids.size(),
          "wrong slot allocation");
  require(gpu.expert_cache().size_bytes()
              == resident_ids.size() * 3 * 128 * 64 * sizeof(float),
          "GPU allocation exceeds configured expert slots");
  if (!resident_ids.empty()) {
    require(!gpu.expert_cache().ready(), "cache should start unpublished");
    const auto x = upload(fixture.input, stream);
    expect_throw<std::logic_error>(
        [&] { (void)gpu.forward(x.view(), stream); });
    for (const auto id : resident_ids)
      require(gpu.expert_cache().slot_of(id) == -1,
              "uninitialized slot was published");
  }
  gpu.populate_expert_cache(stream);
  for (std::size_t id = 0; id < 16; ++id) {
    const auto found =
        std::ranges::find(resident_ids, static_cast<std::int32_t>(id));
    const int expected_slot =
        found == resident_ids.end()
            ? -1
            : static_cast<int>(found - resident_ids.begin());
    require(gpu.expert_cache().slot_of(id) == expected_slot,
            "wrong expert-to-slot mapping");
  }
  check_forward(fixture.cpu, gpu, fixture.input, stream, label + " first");
  check_forward(fixture.cpu, gpu, fixture.other_input, stream,
                label + " different route");
  check_forward(fixture.cpu, gpu, fixture.input, stream,
                label + " first again");
  for (const auto id : {2u, 7u}) {
    const bool resident =
        std::ranges::find(resident_ids, id) != resident_ids.end();
    require((gpu.expert_cache().slot_of(id) >= 0) == resident,
            "miss should not change fixed residency");
  }
}

void test_refresh(Fixture& fixture, cudaStream_t stream) {
  MoE gpu(fixture.cpu.config(), Device::cuda(0),
          std::vector<std::int32_t>{2, 7});
  copy_parameters(fixture.cpu, gpu, stream);
  gpu.populate_expert_cache(stream);
  const auto& cached = gpu.expert_cache()
                           .expert_at_slot(0)
                           .parameter("down_proj.weight")
                           .tensor();
  const auto before = download<float>(cached, stream);
  auto cpu_weight = fixture.cpu.parameter("experts.2.down_proj.weight").view();
  auto ram_weight = gpu.parameter("experts.2.down_proj.weight").view();
  for (std::size_t i = 0; i < cpu_weight.numel(); ++i) {
    cpu_weight.data<float>()[i] *= 2;
    ram_weight.data<float>()[i] = cpu_weight.data<float>()[i];
  }
  require(download<float>(cached, stream) == before,
          "GPU cache must own an independent copy");
  gpu.populate_expert_cache(stream);
  compare(download<float>(cached, stream),
          {ram_weight.data<float>(), ram_weight.numel()}, "refreshed weights");
  check_forward(fixture.cpu, gpu, fixture.input, stream,
                "explicit cache refresh");
}

int test_gpu() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && count == 0))
    return 77;
  check_cuda(status, "enumerate GPUs");
  check_cuda(cudaSetDevice(0), "select GPU");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "read GPU properties");
  std::cout << "GPU: " << properties.name << '\n';
  Stream stream;
  Fixture fixture;
  require(reference(fixture.cpu, fixture.input).expert_ids
              == std::vector<std::int32_t>{7, 2},
          "fixture first route");
  require(reference(fixture.cpu, fixture.other_input).expert_ids
              == std::vector<std::int32_t>{9, 4},
          "fixture second route");
  std::vector<std::int32_t> all(16);
  std::iota(all.rbegin(), all.rend(), 0);
  check_layout(fixture, all, stream.handle, "all GPU");
  check_layout(fixture, {}, stream.handle, "all CPU");
  check_layout(fixture, {4, 2, 9, 0}, stream.handle, "mixed four slots");
  check_layout(fixture, {0, 1, 3, 5}, stream.handle, "all miss four slots");
  check_layout(fixture, {2, 5, 7, 11}, stream.handle, "all hit four slots");
  test_refresh(fixture, stream.handle);
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
    std::cerr << "Usage: nano-vllm-cpp-moe-offload-test --cpu|--gpu\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "MoE offload test failed: " << error.what() << '\n';
    return 1;
  }
}
