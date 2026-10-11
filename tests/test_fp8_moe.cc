#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/model.hh"
#include "core/safetensors.hh"
#include "kernels/cpu/moe.hh"
#include "layers/moe.hh"

namespace {
using namespace nanovllm;
using nlohmann::json;

constexpr std::size_t H = 129, I = 130, E = 6;
constexpr std::size_t kExpandedBytes = 3 * H * I * sizeof(float);
constexpr std::size_t kPackedBytes =
    E * (3 * H * I + 3 * 2 * 2 * sizeof(float));
constexpr std::array<const char*, 3> kProjections{"gate_proj", "up_proj",
                                                  "down_proj"};
// Independent hand-written values, used instead of the decoder under test.
constexpr std::array<std::pair<std::uint8_t, float>, 10> kCodes{{
    {0x00, 0.0f},
    {0x80, -0.0f},
    {0x30, 0.5f},
    {0xb0, -0.5f},
    {0x38, 1.0f},
    {0xb8, -1.0f},
    {0x3c, 1.5f},
    {0xbc, -1.5f},
    {0x40, 2.0f},
    {0xc0, -2.0f},
}};

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
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const auto error = std::abs(actual[index] - expected[index]);
    require(std::isfinite(actual[index]) && std::isfinite(expected[index])
                && error <= 2e-5f + 2e-4f * std::abs(expected[index]),
            label + ": value mismatch at " + std::to_string(index));
    maximum = std::max(maximum, error);
  }
  return maximum;
}

MoEConfig config(DataType dtype, std::size_t shared_size = 0) {
  return {H, I, E, 2, shared_size, dtype};
}

void expand_reference(const ExpertStore& packed, MoE& expanded,
                      std::size_t id) {
  for (const auto* projection : kProjections) {
    const auto prefix = std::to_string(id) + "." + projection;
    const auto weight = packed.parameter(prefix + ".weight").view();
    const auto scale = packed.parameter(prefix + ".weight_scale_inv").view();
    auto output = expanded.parameter("experts." + prefix + ".weight").view();
    const auto columns = weight.shape()[1];
    for (std::size_t row = 0; row < weight.shape()[0]; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        const auto index = row * columns + column;
        const auto bits = weight.data<std::uint8_t>()[index];
        const auto found = std::ranges::find(
            kCodes, bits, &std::pair<std::uint8_t, float>::first);
        require(found != kCodes.end(), "unknown code in independent fixture");
        output.data<float>()[index] =
            found->second
            * scale
                  .data<float>()[(row / 128) * scale.shape()[1] + column / 128];
      }
    }
  }
}

struct Fixture {
  MoE packed;
  MoE expanded;
  Tensor input = Tensor::empty({5, H}, DataType::Float32);

  explicit Fixture(std::size_t shared_size = 0)
      : packed(config(DataType::Float8E4M3FN, shared_size), Device::cpu()),
        expanded(config(DataType::Float32, shared_size), Device::cpu()) {
    for (std::size_t id = 0; id < E; ++id) {
      for (std::size_t projection = 0; projection < kProjections.size();
           ++projection) {
        const auto prefix =
            "experts." + std::to_string(id) + "." + kProjections[projection];
        auto weight = packed.parameter(prefix + ".weight").view();
        auto scale = packed.parameter(prefix + ".weight_scale_inv").view();
        for (std::size_t index = 0; index < weight.numel(); ++index)
          weight.data<std::uint8_t>()[index] =
              kCodes[(index + 3 * id + projection) % kCodes.size()].first;
        for (std::size_t index = 0; index < scale.numel(); ++index)
          scale.data<float>()[index] =
              float(1 + id + projection + index) / 256.0f;
      }
      expand_reference(packed.expert_store(), expanded, id);
    }
    for (const auto& [name, parameter] : packed.named_parameters()) {
      if (name.starts_with("experts.")) continue;
      auto destination = parameter->view();
      for (std::size_t index = 0; index < destination.numel(); ++index)
        destination.data<float>()[index] = float(int(index % 17) - 8) * 0.02f;
      if (name == "gate.weight") {
        std::fill_n(destination.data<float>(), destination.numel(), 0.0f);
        destination.data<float>()[5 * H] = 5;
        destination.data<float>()[2 * H] = 3;
        destination.data<float>()[1 * H + 1] = 6;
        destination.data<float>()[4 * H + 1] = 4;
      }
      std::memcpy(expanded.parameter(name).view().data(), destination.data(),
                  destination.size_bytes());
    }
    for (std::size_t token = 0; token < 5; ++token) {
      auto* row = input.view().data<float>() + token * H;
      for (std::size_t column = 0; column < H; ++column)
        row[column] = float(int((column + token) % 9) - 4) * 0.01f;
      row[0] = token % 2 == 0 ? 1.0f : 0.0f;
      row[1] = token % 2 == 0 ? 0.0f : 1.0f;
    }
  }
};

kernels::cpu::MoEReferenceOutput reference(const Fixture& fixture,
                                           ConstTensorView input) {
  std::vector<kernels::cpu::ExpertWeights> weights;
  for (std::size_t id = 0; id < E; ++id)
    weights.push_back(fixture.expanded.expert_store().weights(id));
  std::optional<kernels::cpu::SharedExpertWeights> shared;
  if (fixture.expanded.config().shared_expert_intermediate_size != 0) {
    shared = kernels::cpu::SharedExpertWeights{
        {fixture.expanded.parameter("shared_expert.gate_proj.weight").view(),
         fixture.expanded.parameter("shared_expert.up_proj.weight").view(),
         fixture.expanded.parameter("shared_expert.down_proj.weight").view()},
        fixture.expanded.parameter("shared_expert_gate.weight").view()};
  }
  return kernels::cpu::moe_reference(
      input, fixture.expanded.parameter("gate.weight").view(), weights, 2,
      shared);
}

struct TestDirectory {
  std::filesystem::path path;
  TestDirectory() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "nanovllm-fp8-experts-XXXXXX")
            .string();
    const auto* directory = ::mkdtemp(pattern.data());
    if (!directory)
      throw std::runtime_error("failed to create fixture directory");
    path = directory;
  }
  ~TestDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

class StoreModel final : public Model {
 public:
  ExpertStore experts{H, I, E, DataType::Float8E4M3FN};
  StoreModel() { register_module("experts", experts); }
};

void write_shard(const std::filesystem::path& path, const json& header,
                 const std::vector<std::byte>& data) {
  auto text = header.dump();
  text.append((8 - text.size() % 8) % 8, ' ');
  std::ofstream file(path, std::ios::binary);
  const auto size = static_cast<std::uint64_t>(text.size());
  for (unsigned int byte = 0; byte < 8; ++byte)
    file.put(static_cast<char>((size >> (8 * byte)) & 0xff));
  file.write(text.data(), text.size());
  file.write(reinterpret_cast<const char*>(data.data()), data.size());
  require(static_cast<bool>(file), "failed to write expert shard");
}

void test_checkpoint(const Fixture& fixture) {
  TestDirectory directory;
  std::array<json, 2> headers{json::object(), json::object()};
  std::array<std::vector<std::byte>, 2> data;
  json mapping = json::object();
  const std::array<std::string, 2> filenames{"weights.safetensors",
                                             "scales.safetensors"};
  for (const auto& [name, parameter] :
       fixture.packed.expert_store().named_parameters()) {
    const auto view = parameter->view();
    const auto shard = view.dtype() == DataType::Float32 ? 1u : 0u;
    const auto full_name = "experts." + name;
    const auto begin = data[shard].size();
    const auto* bytes = view.data<std::byte>();
    data[shard].insert(data[shard].end(), bytes, bytes + view.size_bytes());
    headers[shard][full_name] = {
        {"dtype", view.dtype()},
        {"shape",
         std::vector<std::size_t>(view.shape().begin(), view.shape().end())},
        {"data_offsets", {begin, data[shard].size()}}};
    mapping[full_name] = filenames[shard];
  }
  for (std::size_t shard = 0; shard < 2; ++shard)
    write_shard(directory.path / filenames[shard], headers[shard], data[shard]);
  {
    std::ofstream index(directory.path / "model.safetensors.index.json");
    index << json{{"weight_map", mapping}};
    require(static_cast<bool>(index), "failed to write expert index");
  }
  StoreModel restored;
  const auto result = restored.load_ckpt(directory.path);
  require(result.num_tensors == 6 * E && result.num_bytes == kPackedBytes,
          "checkpoint loaded expanded or incomplete parameters");
  for (const auto& [name, source] :
       fixture.packed.expert_store().named_parameters()) {
    const auto destination = restored.experts.parameter(name).view();
    require(destination.dtype() == source->view().dtype()
                && std::memcmp(destination.data(), source->view().data(),
                               destination.size_bytes())
                       == 0,
            "packed checkpoint parameter mismatch");
  }
  ExpertWeightWorkspace workspace;
  (void)restored.experts.prepare_weights(5, workspace);
  require(restored.experts.size_bytes() == kPackedBytes
              && workspace.size_bytes() == kExpandedBytes,
          "checkpoint load retained all experts as FP32");
}

void test_cpu() {
  Fixture fixture;
  const auto& store = fixture.packed.expert_store();
  require(store.dtype() == DataType::Float8E4M3FN
              && store.named_parameters().size() == 6 * E
              && store.size_bytes() == kPackedBytes,
          "FP8 storage dtype/count/bytes mismatch");
  for (const auto& [name, parameter] : store.named_parameters()) {
    const auto view = parameter->view();
    require(view.device() == Device::cpu(), "packed expert is not in RAM");
    require(
        view.dtype()
            == (name.ends_with("weight_scale_inv") ? DataType::Float32
                                                   : DataType::Float8E4M3FN),
        "incorrect weight or scale dtype");
  }
  ExpertWeightWorkspace workspace;
  require(workspace.size_bytes() == 0, "workspace should start empty");
  expect_throw<std::logic_error>([&] { (void)store.weights(0); });
  expect_throw<std::out_of_range>(
      [&] { (void)store.prepare_weights(E, workspace); });
  require(workspace.size_bytes() == 0, "invalid ID allocated workspace");
  const auto first = store.prepare_weights(0, workspace);
  const std::array<const void*, 3> pointers{first.gate.data(), first.up.data(),
                                            first.down.data()};
  const auto expected = reference(fixture, fixture.input.view().as_const());
  auto output = Tensor::empty({1, H}, DataType::Float32);
  for (const std::size_t id : {0u, 5u, 2u, 1u, 4u, 0u}) {
    const auto actual = store.prepare_weights(id, workspace);
    const auto gold = fixture.expanded.expert_store().weights(id);
    require(actual.gate.data() == pointers[0] && actual.up.data() == pointers[1]
                && actual.down.data() == pointers[2],
            "expert ID changes allocated another workspace");
    for (const auto& [a, b] : std::array{std::pair{actual.gate, gold.gate},
                                         std::pair{actual.up, gold.up},
                                         std::pair{actual.down, gold.down}}) {
      require(std::memcmp(a.data(), b.data(), a.size_bytes()) == 0,
              "expanded weights differ from independent values");
    }
    require(workspace.size_bytes() == kExpandedBytes
                && store.size_bytes() == kPackedBytes,
            "RAM expansion grows with expert count");
  }
  for (std::size_t token = 0; token < 5; ++token) {
    for (std::size_t rank = 0; rank < 2; ++rank) {
      const auto id = expected.expert_ids[token * 2 + rank];
      const auto weights = store.prepare_weights(id, workspace);
      kernels::cpu::expert_forward(
          fixture.input.view().as_const().slice(0, token, 1), weights,
          output.view());
      compare(
          {output.view().data<float>(), H},
          std::span(expected.expert_outputs).subspan((token * 2 + rank) * H, H),
          "FP8 CPU expert");
    }
  }
  ExpertWeightWorkspace untouched;
  const auto direct =
      fixture.expanded.expert_store().prepare_weights(0, untouched);
  require(untouched.size_bytes() == 0
              && direct.gate.data()
                     == fixture.expanded.expert_store().weights(0).gate.data(),
          "FP32 prepare should borrow weights without allocating");
  expect_throw<std::invalid_argument>(
      [] { ExpertStore invalid(H, I, E, DataType::Float16); });
  auto invalid = config(DataType::BFloat16);
  expect_throw<std::invalid_argument>([&] { invalid.validate(); });
  ExpertCache empty(store, {});
  require(empty.size_bytes() == 0 && empty.ready(),
          "zero-slot FP8 cache allocated GPU weights");
  test_checkpoint(fixture);
  std::cout << "FP8 CPU/store/checkpoint: RAM=" << store.size_bytes()
            << " bytes, one reused expert=" << workspace.size_bytes()
            << " bytes; passed\n";
}

struct Stream {
  cudaStream_t handle = nullptr;
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking),
               "create FP8 test stream");
  }
  ~Stream() { cudaStreamDestroy(handle); }
};

Tensor upload(ConstTensorView input, cudaStream_t stream) {
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              input.dtype(), Device::cuda(0));
  check_cuda(cudaMemcpyAsync(output.data(), input.data(), input.size_bytes(),
                             cudaMemcpyHostToDevice, stream),
             "upload FP8 test input");
  check_cuda(cudaStreamSynchronize(stream), "finish FP8 input upload");
  return output;
}

template <typename T>
std::vector<T> download(ConstTensorView input, cudaStream_t stream) {
  std::vector<T> values(input.numel());
  check_cuda(cudaMemcpyAsync(values.data(), input.data(), input.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "download FP8 test output");
  check_cuda(cudaStreamSynchronize(stream), "finish FP8 output download");
  return values;
}

void copy_parameters(const Fixture& fixture, MoE& destination,
                     cudaStream_t stream) {
  for (const auto& [name, parameter] : fixture.packed.named_parameters()) {
    const auto source = parameter->view();
    auto target = destination.parameter(name).view();
    require(same_dtype(source, target) && same_shape(source, target),
            "copy parameter mismatch");
    if (target.device() == Device::cpu())
      std::memcpy(target.data(), source.data(), source.size_bytes());
    else
      check_cuda(
          cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(),
                          cudaMemcpyHostToDevice, stream),
          "upload FP8 MoE parameter");
  }
  check_cuda(cudaStreamSynchronize(stream), "finish MoE parameter upload");
}

void check_forward(const Fixture& fixture, const MoE& sync, const MoE& overlap,
                   ConstTensorView input, cudaStream_t stream) {
  const auto expected = reference(fixture, input);
  const auto x = upload(input, stream);
  const auto actual = overlap.forward(x.view(), stream);
  const auto baseline = sync.forward(x.view(), stream);
  const auto ids = download<std::int32_t>(actual.expert_ids.view(), stream);
  require(ids == expected.expert_ids, "FP8 router selection mismatch");
  compare(download<float>(actual.router_logits.view(), stream),
          expected.router_logits, "FP8 router logits");
  compare(download<float>(actual.expert_weights.view(), stream),
          expected.expert_weights, "FP8 router weights");
  compare(download<float>(actual.expert_outputs.view(), stream),
          expected.expert_outputs, "FP8 raw expert rows");
  compare(download<float>(actual.hidden_states.view(), stream),
          expected.hidden_states, "FP8 combined output");
  std::size_t hits = 0;
  for (const auto id : ids) hits += overlap.expert_cache().slot_of(id) >= 0;
  require(actual.gpu_experts == hits && actual.cpu_experts == ids.size() - hits,
          "FP8 dispatch count mismatch");
  for (const auto& [a, b] :
       std::array{std::pair{actual.router_logits.view(),
                            baseline.router_logits.view()},
                  std::pair{actual.expert_weights.view(),
                            baseline.expert_weights.view()},
                  std::pair{actual.expert_outputs.view(),
                            baseline.expert_outputs.view()},
                  std::pair{actual.hidden_states.view(),
                            baseline.hidden_states.view()}}) {
    require(download<float>(a, stream) == download<float>(b, stream),
            "FP8 overlap differs from synchronous execution");
  }
  if (fixture.expanded.config().shared_expert_intermediate_size != 0) {
    compare(download<float>(actual.shared_expert_output.view(), stream),
            expected.shared_expert_output, "FP8 raw shared output");
    compare(download<float>(actual.shared_gate_logits.view(), stream),
            expected.shared_gate_logits, "FP8 shared gate");
    require(
        download<float>(actual.shared_expert_output.view(), stream)
            == download<float>(baseline.shared_expert_output.view(), stream),
        "shared output differs between modes");
  }
}

void check_cache_weights(const Fixture& fixture, const ExpertCache& cache,
                         cudaStream_t stream) {
  require(cache.size_bytes() == cache.num_slots() * kExpandedBytes,
          "FP8 GPU cache size mismatch");
  for (std::size_t slot = 0; slot < cache.num_slots(); ++slot) {
    const auto id = cache.resident_ids()[slot];
    for (const auto* projection : kProjections) {
      const auto name = std::string(projection) + ".weight";
      const auto expected =
          fixture.expanded
              .parameter("experts." + std::to_string(id) + "." + name)
              .view();
      const auto actual = cache.expert_at_slot(slot).parameter(name).view();
      require(actual.dtype() == DataType::Float32
                  && download<float>(actual, stream)
                         == std::vector<float>(
                             expected.data<float>(),
                             expected.data<float>() + expected.numel()),
              "FP8 resident weights corrupted by staging reuse");
    }
  }
}

void test_layout(const Fixture& fixture,
                 const std::vector<std::int32_t>& residents,
                 cudaStream_t stream) {
  MoE sync(fixture.packed.config(), Device::cuda(0), residents);
  MoE overlap(fixture.packed.config(), Device::cuda(0), residents,
              ExpertExecutionMode::Overlap);
  copy_parameters(fixture, sync, stream);
  copy_parameters(fixture, overlap, stream);
  require(sync.expert_store().size_bytes() == kPackedBytes
              && sync.named_parameters().size()
                     == fixture.packed.named_parameters().size(),
          "cache or workspaces registered as FP8 model parameters");
  if (!residents.empty()) {
    const auto x = upload(fixture.input.view().slice(0, 0, 1), stream);
    expect_throw<std::logic_error>(
        [&] { (void)overlap.forward(x.view(), stream); });
  }
  sync.populate_expert_cache(stream);
  overlap.populate_expert_cache(stream);
  check_cache_weights(fixture, overlap.expert_cache(), stream);
  for (const auto tokens : {1u, 5u, 2u})
    check_forward(fixture, sync, overlap,
                  fixture.input.view().slice(0, 0, tokens), stream);
  const auto x = upload(fixture.input.view(), stream);
  const auto retained = overlap.forward(x.view(), stream);
  const auto previous = download<float>(retained.expert_outputs.view(), stream);
  Stream another;
  check_forward(fixture, sync, overlap, fixture.input.view().slice(0, 1, 3),
                another.handle);
  require(download<float>(retained.expert_outputs.view(), stream) == previous,
          "later FP8 call overwrote retained output");
  require(overlap.expert_store().size_bytes() == kPackedBytes,
          "dispatch expanded persistent store weights");
}

void test_executor(const Fixture& fixture, cudaStream_t stream) {
  ExpertCache cache(fixture.packed.expert_store(), {2, 4, 0});
  cache.populate(stream);
  ExpertExecutor sync(fixture.packed.expert_store(), cache);
  ExpertExecutor overlap(fixture.packed.expert_store(), cache,
                         ExpertExecutionMode::Overlap);
  const auto x = upload(fixture.input.view().slice(0, 0, 1), stream);
  require(overlap.cpu_weight_workspace_bytes() == 0,
          "executor eagerly expanded FP8 weights");
  const std::vector<std::int32_t> hit{2};
  const auto retained = overlap.forward(x.view(), hit, stream);
  const auto previous = download<float>(retained.expert_outputs.view(), stream);
  require(overlap.cpu_weight_workspace_bytes() == 0,
          "GPU hit allocated CPU expanded weights");
  Stream another;
  const std::vector<std::vector<std::int32_t>> routes{
      {5}, {2, 5}, {5, 2}, {1, 4, 5}, {5, 4, 3, 2, 1, 0}, {2}};
  for (std::size_t index = 0; index < routes.size(); ++index) {
    const auto active_stream = index % 2 ? another.handle : stream;
    const auto actual = overlap.forward(x.view(), routes[index], active_stream);
    const auto expected = sync.forward(x.view(), routes[index], active_stream);
    require(
        download<float>(actual.expert_outputs.view(), active_stream)
            == download<float>(expected.expert_outputs.view(), active_stream),
        "FP8 executor order/reuse mismatch");
    require(overlap.cpu_weight_workspace_bytes() == kExpandedBytes
                && sync.cpu_weight_workspace_bytes() == kExpandedBytes,
            "FP8 CPU expanded memory grows with K");
  }
  require(download<float>(retained.expert_outputs.view(), stream) == previous,
          "executor reused returned output storage");
  std::cout << "FP8 executor: lazy workspace, changing IDs/K/streams and "
               "retained outputs passed\n";
}

void test_refresh(cudaStream_t stream, ExpertExecutionMode mode) {
  Fixture fixture;
  MoE gpu(fixture.packed.config(), Device::cuda(0),
          std::vector<std::int32_t>{2, 4}, mode);
  copy_parameters(fixture, gpu, stream);
  gpu.populate_expert_cache(stream);
  const auto cached =
      gpu.expert_cache().expert_at_slot(0).parameter("gate_proj.weight").view();
  const auto before = download<float>(cached, stream);
  for (auto* module : {&fixture.packed, &gpu}) {
    module->parameter("experts.2.gate_proj.weight_scale_inv")
        .view()
        .data<float>()[0] *= 2;
    module->parameter("experts.2.gate_proj.weight")
        .view()
        .data<std::uint8_t>()[0] ^= 0x80;
  }
  expand_reference(fixture.packed.expert_store(), fixture.expanded, 2);
  require(download<float>(cached, stream) == before,
          "packed changes implicitly changed GPU copy");
  gpu.populate_expert_cache(stream);
  check_cache_weights(fixture, gpu.expert_cache(), stream);
  check_forward(fixture, gpu, gpu, fixture.input.view().as_const(), stream);
}

int test_gpu() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && count == 0))
    return 77;
  check_cuda(status, "enumerate FP8 test GPUs");
  check_cuda(cudaSetDevice(0), "select FP8 test GPU");
  Stream stream;
  Fixture fixture;
  for (const auto& residents : std::vector<std::vector<std::int32_t>>{
           {5, 4, 3, 2, 1, 0}, {}, {2, 4, 0}, {0, 3}, {1, 2, 4, 5}})
    test_layout(fixture, residents, stream.handle);
  Fixture shared(37);
  test_layout(shared, {2, 4, 0}, stream.handle);
  test_executor(fixture, stream.handle);
  test_refresh(stream.handle, ExpertExecutionMode::Synchronous);
  test_refresh(stream.handle, ExpertExecutionMode::Overlap);
  std::cout
      << "FP8 MoE GPU: cache layouts, batch/shared/modes and refresh passed\n";
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
    std::cerr << "usage: nano-vllm-cpp-fp8-moe-test --cpu|--gpu\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "FP8 MoE test failed: " << error.what() << '\n';
    return 1;
  }
}
