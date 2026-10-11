#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
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
  MoE cpu;
  Tensor input = Tensor::empty({1, 128}, DataType::Float32);
  Tensor other_input = Tensor::empty({1, 128}, DataType::Float32);

  explicit Fixture(std::size_t shared_intermediate_size = 0)
      : cpu({128, 64, 16, 2, shared_intermediate_size}, Device::cpu()) {
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

std::optional<kernels::cpu::SharedExpertWeights> shared_weights(
    const MoE& moe) {
  if (moe.config().shared_expert_intermediate_size == 0) return std::nullopt;
  return kernels::cpu::SharedExpertWeights{
      {moe.parameter("shared_expert.gate_proj.weight").view(),
       moe.parameter("shared_expert.up_proj.weight").view(),
       moe.parameter("shared_expert.down_proj.weight").view()},
      moe.parameter("shared_expert_gate.weight").view()};
}

kernels::cpu::MoEReferenceOutput reference(const MoE& cpu, const Tensor& input,
                                           std::size_t top_k = 2) {
  std::vector<kernels::cpu::ExpertWeights> experts;
  for (std::size_t id = 0; id < cpu.config().num_experts; ++id) {
    experts.push_back(cpu.expert_store().weights(id));
  }
  return kernels::cpu::moe_reference(input.view(),
                                     cpu.parameter("gate.weight").view(),
                                     experts, top_k, shared_weights(cpu));
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
  expect_throw<std::invalid_argument>([&] {
    ExpertExecutor invalid(store, empty, static_cast<ExpertExecutionMode>(42));
  });
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
  const auto shared_size = cpu.config().shared_expert_intermediate_size;
  require(gpu.named_parameters().size()
              == 1 + 3 * cpu.config().num_experts + (shared_size ? 4 : 0),
          "cache copies must not be registered as model parameters");
  for (const auto& [name, parameter] : cpu.named_parameters()) {
    const auto source = parameter->view();
    auto destination = gpu.parameter(name).view();
    const auto expected_device =
        name.starts_with("experts.") ? Device::cpu() : Device::cuda(0);
    require(destination.device() == expected_device,
            "wrong parameter residency: " + name);
    if (destination.device() == Device::cpu()) {
      std::memcpy(destination.data(), source.data(), source.size_bytes());
    } else {
      check_cuda(
          cudaMemcpyAsync(destination.data(), source.data(),
                          source.size_bytes(), cudaMemcpyHostToDevice, stream),
          "upload GPU MoE parameter");
    }
  }
  std::size_t gpu_parameter_bytes = 0;
  for (const auto& [name, parameter] : gpu.named_parameters()) {
    if (parameter->view().device() == Device::cuda(0))
      gpu_parameter_bytes += parameter->view().size_bytes();
  }
  const auto hidden_size = cpu.config().hidden_size;
  const auto expected_bytes =
      (hidden_size * cpu.config().num_experts
       + (shared_size ? 3 * hidden_size * shared_size + hidden_size : 0))
      * sizeof(float);
  require(gpu_parameter_bytes == expected_bytes,
          "shared expert/gate must be separate from routed cache copies");
}

void check_forward(const MoE& cpu, const MoE& gpu, const Tensor& input,
                   cudaStream_t stream, const std::string& label,
                   const MoE* synchronous = nullptr) {
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
  if (cpu.config().shared_expert_intermediate_size != 0) {
    compare(download<float>(actual.shared_expert_output, stream),
            expected.shared_expert_output, label + " raw shared output");
    compare(download<float>(actual.shared_gate_logits, stream),
            expected.shared_gate_logits, label + " shared gate logits");
  } else {
    require(actual.shared_expert_output.numel() == 0
                && actual.shared_gate_logits.numel() == 0,
            label + ": disabled shared outputs should be empty");
  }
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
              && actual.cpu_experts == cpu.config().top_k - expected_hits,
          label + ": wrong CPU/GPU execution counts");
  if (synchronous) {
    const auto baseline = synchronous->forward(gpu_input.view(), stream);
    if (cpu.config().shared_expert_intermediate_size != 0) {
      require(
          download<float>(actual.shared_expert_output, stream)
                  == download<float>(baseline.shared_expert_output, stream)
              && download<float>(actual.shared_gate_logits, stream)
                     == download<float>(baseline.shared_gate_logits, stream),
          label + ": shared branch differs between execution modes");
    }
    require(download<float>(actual.router_logits, stream)
                    == download<float>(baseline.router_logits, stream)
                && download<std::int32_t>(actual.expert_ids, stream)
                       == download<std::int32_t>(baseline.expert_ids, stream)
                && download<float>(actual.expert_weights, stream)
                       == download<float>(baseline.expert_weights, stream)
                && download<float>(actual.expert_outputs, stream)
                       == download<float>(baseline.expert_outputs, stream)
                && download<float>(actual.hidden_states, stream)
                       == download<float>(baseline.hidden_states, stream)
                && actual.gpu_experts == baseline.gpu_experts
                && actual.cpu_experts == baseline.cpu_experts,
            label + ": overlap differs from synchronous execution");
  }
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
  MoE overlapping(fixture.cpu.config(), Device::cuda(0), resident_ids,
                  ExpertExecutionMode::Overlap);
  copy_parameters(fixture.cpu, overlapping, stream);
  if (!resident_ids.empty()) {
    const auto x = upload(fixture.input, stream);
    expect_throw<std::logic_error>(
        [&] { (void)overlapping.forward(x.view(), stream); });
  }
  overlapping.populate_expert_cache(stream);
  for (const auto* input :
       {&fixture.input, &fixture.other_input, &fixture.input}) {
    check_forward(fixture.cpu, overlapping, *input, stream, label + " overlap",
                  &gpu);
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

void test_refresh(Fixture& fixture, cudaStream_t stream,
                  ExpertExecutionMode mode) {
  MoE gpu(fixture.cpu.config(), Device::cuda(0),
          std::vector<std::int32_t>{2, 7}, mode);
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
                mode == ExpertExecutionMode::Overlap ? "overlap cache refresh"
                                                     : "sync cache refresh");
}

// Reuse one executor across changing K, route order, inputs and CUDA streams.
void test_workspace_reuse(const Fixture& fixture, cudaStream_t stream) {
  ExpertCache cache(fixture.cpu.expert_store(), {4, 2, 9, 0});
  cache.populate(stream);
  ExpertExecutor synchronous(fixture.cpu.expert_store(), cache);
  ExpertExecutor overlapping(fixture.cpu.expert_store(), cache,
                             ExpertExecutionMode::Overlap);
  const auto x = upload(fixture.input, stream);
  const auto other = upload(fixture.other_input, stream);
  const std::vector<std::int32_t> initial{7, 2};
  const auto retained = overlapping.forward(x.view(), initial, stream);
  const auto retained_values = download<float>(retained.expert_outputs, stream);
  std::vector<std::int32_t> all(16);
  std::iota(all.rbegin(), all.rend(), 0);
  const std::vector<std::vector<std::int32_t>> routes{
      {2}, {7}, {2, 7}, {7, 2}, {9, 7, 2, 4}, all, {7, 2}};
  Stream another;
  for (std::size_t run = 0; run < routes.size(); ++run) {
    const auto active_stream = run % 2 ? another.handle : stream;
    const auto& input = run % 2 ? other : x;
    const auto expected =
        synchronous.forward(input.view(), routes[run], active_stream);
    const auto actual =
        overlapping.forward(input.view(), routes[run], active_stream);
    require(download<float>(actual.expert_outputs, active_stream)
                    == download<float>(expected.expert_outputs, active_stream)
                && actual.gpu_experts == expected.gpu_experts
                && actual.cpu_experts == expected.cpu_experts,
            "workspace reuse changed output order or values");
  }
  require(download<float>(retained.expert_outputs, stream) == retained_values,
          "later calls overwrote a retained output");
  const std::vector<std::int32_t> bad{-1};
  expect_throw<std::invalid_argument>(
      [&] { (void)overlapping.forward(x.view(), bad, stream); });

  std::cout << "overlap workspace: changing K/order/input/stream, retained "
               "outputs and validation passed\n";
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
  Fixture shared(37);
  check_layout(shared, all, stream.handle, "shared all GPU");
  check_layout(shared, {}, stream.handle, "shared all CPU routed");
  check_layout(shared, {4, 2, 9, 0}, stream.handle, "shared mixed four slots");
  test_workspace_reuse(fixture, stream.handle);
  test_refresh(fixture, stream.handle, ExpertExecutionMode::Synchronous);
  test_refresh(fixture, stream.handle, ExpertExecutionMode::Overlap);
  return 0;
}

// Manual synthetic workload; no model downloads. Not part of timing assertions.
int benchmark(const std::string& mode) {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && count == 0))
    return 77;
  check_cuda(status, "enumerate benchmark GPUs");
  check_cuda(cudaSetDevice(0), "select benchmark GPU");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0),
             "benchmark GPU properties");
  Stream stream;
  constexpr std::size_t H = 4096, I = 2048, E = 8;
  ExpertStore store(H, I, E);
  // Bounded values keep FP32 results finite at this larger shape.
  std::uint32_t state = 456;
  for (const auto& [name, parameter] : store.named_parameters()) {
    auto weight = parameter->view();
    const float scale = 1.0f / std::sqrt(static_cast<float>(weight.shape()[1]));
    for (std::size_t i = 0; i < weight.numel(); ++i) {
      state = state * 1664525u + 1013904223u;
      weight.data<float>()[i] =
          (float(state >> 8) / 16777216.0f * 2 - 1) * scale;
    }
  }
  auto input = Tensor::empty({1, H}, DataType::Float32);
  for (std::size_t i = 0; i < H; ++i)
    input.view().data<float>()[i] = (int(i % 11) - 5) * 0.1f;
  const auto x = upload(input, stream.handle);
  ExpertCache cache(store, {1, 3, 5, 7});
  cache.populate(stream.handle);
  ExpertExecutor synchronous(store, cache);
  ExpertExecutor overlapping(store, cache, ExpertExecutionMode::Overlap);
  const std::vector<std::int32_t> ids{0, 1, 2, 3, 4, 5, 6, 7};
  for (int warmup = 0; warmup < 2; ++warmup) {
    const auto expected = synchronous.forward(x.view(), ids, stream.handle);
    const auto actual = overlapping.forward(x.view(), ids, stream.handle);
    require(download<float>(actual.expert_outputs, stream.handle)
                == download<float>(expected.expert_outputs, stream.handle),
            "benchmark modes differ");
  }
  std::cout << "GPU=" << properties.name << " H=" << H << " I=" << I
            << " E=8 K=8 GPU=4 CPU=4 GPU weight bytes=" << cache.size_bytes()
            << '\n';
  if (mode != "--benchmark") {
    const auto& executor =
        mode == "--profile-overlap" ? overlapping : synchronous;
    check_cuda(cudaProfilerStart(), "start overlap profile");
    const auto output = executor.forward(x.view(), ids, stream.handle);
    check_cuda(cudaProfilerStop(), "stop overlap profile");
    std::cout << mode << ": captured one warmed-up dispatch\n";
    return 0;
  }
  std::vector<double> sync_ms, overlap_ms;
  const auto measure = [&](const ExpertExecutor& executor) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = executor.forward(x.view(), ids, stream.handle);
    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
  };
  for (int run = 0; run < 7; ++run) {
    if (run % 2 == 0) {
      sync_ms.push_back(measure(synchronous));
      overlap_ms.push_back(measure(overlapping));
    } else {
      overlap_ms.push_back(measure(overlapping));
      sync_ms.push_back(measure(synchronous));
    }
  }
  std::ranges::sort(sync_ms);
  std::ranges::sort(overlap_ms);
  std::cout << "7 interleaved pairs; median dispatch ms: synchronous="
            << sync_ms[3] << " overlap=" << overlap_ms[3] << '\n';
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
    if (argc == 2
        && (std::string(argv[1]) == "--benchmark"
            || std::string(argv[1]) == "--profile-overlap"
            || std::string(argv[1]) == "--profile-sync"))
      return benchmark(argv[1]);
    std::cerr << "Usage: nano-vllm-cpp-moe-offload-test "
                 "--cpu|--gpu|--benchmark|--profile-overlap|--profile-sync\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "MoE offload test failed: " << error.what() << '\n';
    return 1;
  }
}
