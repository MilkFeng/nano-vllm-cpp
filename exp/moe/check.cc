#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_utils.hh"
#include "common/thread_pool.hh"
#include "core/expert_policy.hh"
#include "core/tensor_pool.hh"
#include "functions/functions.hh"
#include "kernels/cpu/fp8_dequant.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/moe/moe.hh"
#include "models/qwen35.hh"

using namespace nanovllm;

namespace {

constexpr std::size_t kHidden = 128;
constexpr std::size_t kExperts = 16;
constexpr std::size_t kTopK = 8;
using DenseWeights = std::map<std::string, std::vector<float>>;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void compare(std::span<const float> actual, std::span<const float> expected,
             float atol = 2e-5f, float rtol = 2e-4f) {
  require(actual.size() == expected.size(), "output size");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])
        || std::abs(actual[i] - expected[i])
               > atol + rtol * std::abs(expected[i])) {
      throw std::runtime_error("output mismatch at " + std::to_string(i));
    }
  }
}

float round_bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
  return std::bit_cast<float>(bits);
}

void store(TensorView tensor, const void* data) {
  std::vector<std::uint16_t> packed;
  if (tensor.dtype() == DataType::BFloat16) {
    const auto* values = static_cast<const float*>(data);
    packed.resize(tensor.numel());
    for (std::size_t i = 0; i < packed.size(); ++i) {
      packed[i] = std::bit_cast<std::uint32_t>(round_bf16(values[i])) >> 16;
    }
    data = packed.data();
  }
  if (tensor.device().type == DeviceType::Cuda) {
    check_cuda(cudaMemcpy(tensor.data(), data, tensor.size_bytes(),
                          cudaMemcpyHostToDevice),
               "initialize test weights");
  } else {
    std::memcpy(tensor.data(), data, tensor.size_bytes());
  }
}

DenseWeights initialize(Module& module) {
  DenseWeights dense;
  std::size_t seed = 1;
  for (auto& [name, parameter] : module.named_parameters()) {
    const auto view = parameter->view();
    if (view.dtype() == DataType::Float8E4M3FN) {
      std::vector<std::uint8_t> values(view.numel());
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<std::uint8_t>(0x30 + (i + seed) % 16
                                              + ((i / 7 + seed) % 2) * 128);
      }
      store(view, values.data());
    } else {
      std::vector<float> values(view.numel());
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = name.ends_with("weight_scale_inv")
                        ? 0.06f
                        : 0.04f * std::sin(float(i + seed));
      }
      if (name == "gate.weight") {
        std::fill(values.begin(), values.end(), 0.f);
        for (std::size_t e = 0; e < kExperts; ++e) {
          values[e * kHidden] = float(e + 1) / 8;
        }
      }
      store(view, values.data());
      if (view.dtype() == DataType::BFloat16) {
        for (auto& value : values) {
          value = round_bf16(value);
        }
      }
      if (name == "gate.weight" || name.starts_with("shared_expert")) {
        dense.emplace(name, std::move(values));
      }
    }
    seed += 17;
  }
  // Build the shared expert reference from the initialized FP8 bytes/scales.
  for (const auto* projection : {"gate_proj", "up_proj", "down_proj"}) {
    const auto name = std::string("shared_expert.") + projection + ".weight";
    if (!module.has_parameter(name)) {
      continue;
    }
    const auto weight_view = module.parameter(name).view();
    const auto scale_view = module.parameter(name + "_scale_inv").view();
    auto weight = Tensor::empty(
        {weight_view.shape().begin(), weight_view.shape().end()},
        DataType::Float8E4M3FN);
    auto scale = Tensor::empty(
        {scale_view.shape().begin(), scale_view.shape().end()},
        DataType::Float32);
    for (const auto pair : {std::pair{weight_view, weight.view()},
                            std::pair{scale_view, scale.view()}}) {
      if (pair.first.device().type == DeviceType::Cuda) {
        check_cuda(cudaMemcpy(pair.second.data(), pair.first.data(),
                              pair.first.size_bytes(), cudaMemcpyDeviceToHost),
                   "read shared expert reference weights");
      } else {
        std::memcpy(pair.second.data(), pair.first.data(),
                    pair.first.size_bytes());
      }
    }
    auto expanded = Tensor::empty(
        {weight_view.shape().begin(), weight_view.shape().end()},
        DataType::Float32);
    kernels::cpu::dequantize_fp8(weight.view().as_const(),
                                 scale.view().as_const(), expanded.view());
    const auto* values = expanded.view().data<float>();
    dense[name] = std::vector<float>(values, values + expanded.numel());
  }
  return dense;
}

std::vector<float> input_values(std::size_t rows, std::size_t step,
                                DataType dtype = DataType::Float32) {
  std::vector<float> values(rows * kHidden);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 0.3f * std::cos(float(i + 3 * step));
  }
  for (std::size_t t = 0; t < rows; ++t) {
    values[t * kHidden] = (t + step) % 2 ? -0.75f : 0.75f;
  }
  if (dtype == DataType::BFloat16) {
    for (auto& value : values) {
      value = round_bf16(value);
    }
  }
  return values;
}

std::vector<float> linear(std::span<const float> x,
                          const std::vector<float>& weights, std::size_t rows) {
  std::vector<float> result(rows);
  for (std::size_t n = 0; n < rows; ++n) {
    for (std::size_t k = 0; k < x.size(); ++k) {
      result[n] += x[k] * weights[n * x.size() + k];
    }
  }
  return result;
}

std::vector<float> reference(const MoE& moe, const DenseWeights& dense,
                             const std::vector<float>& input, std::size_t rows,
                             bool shared) {
  std::vector<float> result(input.size());
  auto part = Tensor::empty({1, kHidden}, DataType::Float32);
  for (std::size_t t = 0; t < rows; ++t) {
    const auto x = std::span(input).subspan(t * kHidden, kHidden);
    const auto logits = linear(x, dense.at("gate.weight"), kExperts);
    std::vector<std::size_t> ids(kExperts);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(),
              [&](auto a, auto b) { return logits[a] > logits[b]; });
    float sum = 0.f;
    for (std::size_t r = 0; r < kTopK; ++r) {
      sum += std::exp(logits[ids[r]] - logits[ids[0]]);
    }
    const auto view = ConstTensorView(
        x.data(), DataType::Float32, TensorLayout({1, kHidden}), Device::cpu());
    for (std::size_t r = 0; r < kTopK; ++r) {
      moe.expert_store().expert_at(ids[r]).forward_into(view, part.view());
      const auto weight = std::exp(logits[ids[r]] - logits[ids[0]]) / sum;
      for (std::size_t d = 0; d < kHidden; ++d) {
        result[t * kHidden + d] += weight * part.view().data<float>()[d];
      }
    }
    if (shared) {
      auto gate =
          linear(x, dense.at("shared_expert.gate_proj.weight"), kHidden);
      const auto up =
          linear(x, dense.at("shared_expert.up_proj.weight"), kHidden);
      for (std::size_t d = 0; d < kHidden; ++d) {
        gate[d] = gate[d] / (1.f + std::exp(-gate[d])) * up[d];
      }
      const auto y =
          linear(gate, dense.at("shared_expert.down_proj.weight"), kHidden);
      const auto logit = linear(x, dense.at("shared_expert_gate.weight"), 1)[0];
      const auto weight = 1.f / (1.f + std::exp(-logit));
      for (std::size_t d = 0; d < kHidden; ++d) {
        result[t * kHidden + d] += weight * y[d];
      }
    }
  }
  return result;
}

std::vector<float> download(ConstTensorView view, cudaStream_t stream) {
  Tensor expanded;
  if (view.dtype() == DataType::BFloat16) {
    expanded = functions::cast_tensor(view, DataType::Float32, stream);
    view = expanded.view().as_const();
  }
  std::vector<float> values(view.numel());
  check_cuda(cudaMemcpyAsync(values.data(), view.data(), view.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "read test output");
  check_cuda(cudaStreamSynchronize(stream), "complete test output");
  return values;
}

void check_combine_rounding(cudaStream_t stream) {
  const auto device = Device::cuda();
  auto parts = Tensor::empty({2, 1}, DataType::Float32, device);
  auto weights = Tensor::empty({1, 2}, DataType::Float32, device);
  auto shared = Tensor::empty({1, 1}, DataType::BFloat16, device);
  auto gate = Tensor::empty({1, 1}, DataType::Float32, device);
  auto output = Tensor::empty({1, 1}, DataType::BFloat16, device);
  const float routed_parts[] = {1.0f, 1.0078125f};
  const float routed_weights[] = {0.5f, 0.5f};
  const float shared_part = 0.00390625f;
  const float gate_logit = 0.0f;
  store(parts.view(), routed_parts);
  store(weights.view(), routed_weights);
  store(shared.view(), &shared_part);
  store(gate.view(), &gate_logit);

  const auto expert_outputs = parts.view().as_const();
  const auto expert_weights = weights.view().as_const();
  kernels::cuda::moe_combine(expert_outputs, expert_weights, output.view(), {},
                            {}, stream);
  require(download(output.view().as_const(), stream).front() == 1.0f,
          "routed sum rounds to even");

  // The shared contribution moves the FP32 sum above the BF16 midpoint.
  kernels::cuda::moe_combine(expert_outputs, expert_weights, output.view(),
                            shared.view().as_const(), gate.view().as_const(),
                            stream);
  require(download(output.view().as_const(), stream).front() == 1.0078125f,
          "round after adding the shared expert");
}

void check_moe(const std::vector<std::int32_t>& resident, bool shared,
               cudaStream_t stream) {
  constexpr auto dtype = DataType::BFloat16;
  ThreadPool workers(4);
  std::optional<MoEExecutionWorkspace> workspace(std::in_place);
  const MoEConfig config{kHidden, kHidden, kExperts, kTopK,
                         shared ? kHidden : 0};
  MoE a(config, workers, Device::cuda(), resident, &*workspace);
  MoE b(config, workers, Device::cuda(), resident, &*workspace);
  const auto dense = initialize(a);
  initialize(b);
  a.populate_expert_cache(stream);
  b.populate_expert_cache(stream);

  const auto profile_values = input_values(1, 2, dtype);
  auto profile_input = Tensor::empty({1, kHidden}, dtype, Device::cuda());
  store(profile_input.view(), profile_values.data());
  MoEPhaseProfile profile;
  ExpertUsage usage;
  usage.counts.resize(kExperts);
  const auto profiled =
      a.forward(profile_input.view().as_const(), stream, &profile, &usage);
  require(profile.calls == 1 && profile.input_tokens == 1
              && profile.gpu_assignments + profile.cpu_assignments == kTopK
              && profile.seconds > 0,
          "profile accounting");
  require(usage.gpu_assignments + usage.cpu_assignments == kTopK,
          "adaptation accounting");
  compare(download(profiled.hidden_states.view(), stream),
          reference(a, dense, profile_values, 1, shared), 3e-4f, 2e-2f);

  TensorPool pool(Device::cuda(), stream);
  struct Result {
    Tensor output;
    std::vector<float> expected;
  };
  std::vector<Result> results;
  const std::vector<std::size_t> lengths{1, 1, 3, 1, 7, 1, 2, 1};
  std::vector<Tensor> inputs;
  std::vector<std::vector<float>> expected;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    const auto values = input_values(lengths[i], i, dtype);
    auto input = Tensor::empty({lengths[i], kHidden}, dtype, Device::cuda());
    store(input.view(), values.data());
    inputs.push_back(std::move(input));
    expected.push_back(reference(a, dense, values, lengths[i], shared));
  }
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    auto& moe = i % 2 ? b : a;
    const auto rows = lengths[i];
    TensorPool::Scope scope(pool, stream);
    auto output = moe.forward(inputs[i].view().as_const(), stream);
    require(output.hidden_states.dtype() == dtype, "MoE output dtype");
    require(output.gpu_experts + output.cpu_experts == rows * kTopK,
            "assignment count");
    results.push_back(
        {std::move(output.hidden_states), std::move(expected[i])});
    // Other returned owners and internal temporaries can now be recycled on
    // this stream, while the final GPU work is still queued.
  }
  // The final forward may still be using pinned staging and GPU scratch.
  workspace.reset();
  for (const auto& result : results) {
    compare(download(result.output.view(), stream), result.expected,
            3e-4f, 2e-2f);
  }
}

void check_decode_prefetch(cudaStream_t stream) {
  std::vector<std::int32_t> resident(kExperts);
  std::iota(resident.begin(), resident.end(), 0);
  resident.back() = resident.front();

  ThreadPool workers(4);
  MoE moe({kHidden, kHidden, kExperts, kTopK}, workers, Device::cuda(),
          resident);
  initialize(moe);
  moe.populate_expert_cache(stream);

  constexpr auto dtype = DataType::BFloat16;
  const auto values = input_values(1, 0, dtype);
  auto input = Tensor::empty({1, kHidden}, dtype, Device::cuda());
  store(input.view(), values.data());
  const auto output = moe.forward(input.view().as_const(), stream);
  require(output.cpu_experts == 1, "full slot count still has a CPU miss");

  // The positive first feature routes to the missing expert first.
  const ConstTensorView cpu_input(values.data(), DataType::Float32,
                                  TensorLayout({1, kHidden}), Device::cpu());
  auto expected = Tensor::empty({1, kHidden}, DataType::Float32);
  moe.expert_store().expert_at(kExperts - 1).forward_into(cpu_input,
                                                        expected.view());
  compare(download(output.expert_outputs.view().slice(0, 0, 1), stream),
          {expected.view().data<float>(), kHidden}, 0.0f, 0.0f);
}

void check_graphs(const std::vector<std::int32_t>& resident, bool shared,
                  cudaStream_t first, cudaStream_t second) {
  constexpr auto dtype = DataType::BFloat16;
  ThreadPool workers(4);
  MoE moe({kHidden, kHidden, kExperts, kTopK, shared ? kHidden : 0},
          workers, Device::cuda(), resident);
  const auto dense = initialize(moe);
  moe.populate_expert_cache(first);
  auto input = Tensor::empty({1, kHidden}, dtype, Device::cuda());
  ExpertUsage usage{std::vector<double>(kExperts)};
  for (std::size_t step = 0; step < 32; ++step) {
    const auto stream = step < 16 ? first : second;
    if (step == 8 && !resident.empty() && moe.expert_cache().slot_of(1) < 0) {
      const ExpertCacheReplacement swap{0, 1};
      require(moe.replace_experts({&swap, 1}, stream) > 0,
              "asynchronous expert replacement");
    }
    const auto values = input_values(1, step, dtype);
    store(input.view(), values.data());
    auto output =
        moe.forward(input.view().as_const(), stream, nullptr, &usage, true);
    require(output.hidden_states.dtype() == dtype, "hybrid MoE output dtype");
    compare(download(output.hidden_states.view().as_const(), stream),
            reference(moe, dense, values, 1, shared), 3e-4f, 2e-2f);
    moe.check_async_error();
    require(output.gpu_experts + output.cpu_experts == kTopK,
            "hybrid assignment counters");
  }
  require(usage.gpu_assignments + usage.cpu_assignments == 32 * kTopK,
          "hybrid usage counters");
}

void check_state_graph(cudaStream_t stream, bool whole) {
  ModelConfig config;
  config.hidden_size = kHidden;
  config.moe_intermediate_size = kHidden;
  config.num_experts = kExperts;
  config.num_experts_per_tok = kTopK;
  config.shared_expert_intermediate_size = kHidden;
  config.quantization_config = FP8QuantizationConfig{};
  config.quantization_config->weight_block_size = {128, 128};
  config.layer_types = {AttentionType::Linear};
  config.linear_num_key_heads = 1;
  config.linear_num_value_heads = 2;
  config.linear_key_head_dim = 16;
  config.linear_value_head_dim = 32;
  config.linear_conv_kernel_dim = 3;
  config.rms_norm_eps = 1e-6f;
  ThreadPool workers(4);
  MoEExecutionWorkspace work_a, work_b;
  Qwen35DecoderLayer plain(config, 0, Device::cuda(), {0, 2, 4}, workers,
                           work_a);
  Qwen35DecoderLayer graphed(config, 0, Device::cuda(), {0, 2, 4}, workers,
                             work_b);
  initialize(plain);
  initialize(graphed);
  plain.mlp().populate_expert_cache(stream);
  graphed.mlp().populate_expert_cache(stream);
  auto conv_a = Tensor::empty({1, 96, 2}, DataType::BFloat16, Device::cuda());
  auto conv_b = Tensor::empty({1, 96, 2}, DataType::BFloat16, Device::cuda());
  auto state_a =
      Tensor::empty({1, 2, 16, 32}, DataType::Float32, Device::cuda());
  auto state_b =
      Tensor::empty({1, 2, 16, 32}, DataType::Float32, Device::cuda());
  for (auto* tensor : {&conv_a, &conv_b, &state_a, &state_b}) {
    functions::zero_tensor(tensor->view(), stream);
  }
  plain.linear_attention()->set_state_cache(conv_a.view(), state_a.view());
  graphed.linear_attention()->set_state_cache(conv_b.view(), state_b.view());
  auto input = Tensor::empty({1, kHidden}, DataType::BFloat16, Device::cuda());
  auto residual =
      Tensor::empty({1, kHidden}, DataType::Float32, Device::cuda());
  auto slots = Tensor::empty({1}, DataType::Int32, Device::cuda());
  const std::int32_t zero = 0;
  store(slots.view(), &zero);
  ForwardContext context;
  context.state_indices = slots.view().as_const();
  context.stream = stream;
  WholeDecodeGraph whole_graph;
  const std::array<TensorView, 2> states{conv_b.view(), state_b.view()};
  for (std::size_t step = 0; step < 32; ++step) {
    const auto values = input_values(1, step);
    store(input.view(), values.data());
    store(residual.view(), values.data());
    context.decode_graphs = false;
    auto a = plain.forward({}, input.view().as_const(),
                           residual.view().as_const(), context);
    context.decode_graphs = true;
    Qwen35DecoderLayerOutput b;
    if (whole) {
      context.decode_graphs = false;
      context.whole_graph = true;
      auto result = whole_graph.run(
          stream, {input.data(), slots.data()}, states,
          [&](bool capturing) -> std::vector<Tensor> {
            auto captured = context;
            captured.graph_warmup = !capturing;
            auto out = graphed.forward({}, input.view().as_const(),
                                       residual.view().as_const(), captured);
            return {std::move(out.hidden_states), std::move(out.residual)};
          },
          [&] { graphed.mlp().reset_hybrid(); },
          [&] { graphed.mlp().complete_hybrid_cpu(); });
      b = {std::move(result[0]), std::move(result[1])};
    } else {
      b = graphed.forward({}, input.view().as_const(),
                          residual.view().as_const(), context);
    }
    for (const auto pair :
         {std::pair{a.hidden_states.view().as_const(),
                    b.hidden_states.view().as_const()},
          std::pair{a.residual.view().as_const(), b.residual.view().as_const()},
          std::pair{conv_a.view().as_const(), conv_b.view().as_const()},
          std::pair{state_a.view().as_const(), state_b.view().as_const()}}) {
      const auto x = download(pair.first, stream),
                 y = download(pair.second, stream);
      require(
          x.size() == y.size()
              && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0,
          "graph changed decoder output or recurrent state");
    }
    graphed.mlp().check_async_error();
    context.whole_graph = false;
  }
}

} // namespace

int main() {
  try {
    cudaStream_t first, second;
    check_cuda(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking),
               "create first test stream");
    check_cuda(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking),
               "create second test stream");
    check_combine_rounding(first);
    std::vector<std::int32_t> all(kExperts);
    std::iota(all.begin(), all.end(), 0);
    for (bool shared : {false, true}) {
      for (const auto& ids : std::vector<std::vector<std::int32_t>>{
               {},
               {0, 2, 4, 6, 8, 10, 12, 14},
               {8, 9, 10, 11, 12, 13, 14, 15},
               all}) {
        check_moe(ids, shared, first);
        check_graphs(ids, shared, first, second);
      }
    }
    check_decode_prefetch(first);
    check_state_graph(first, false);
    check_state_graph(first, true);
    check_cuda(cudaStreamDestroy(second), "destroy second test stream");
    check_cuda(cudaStreamDestroy(first), "destroy first test stream");
    std::cout
        << "MoE async checks passed: BF16, CPU/mixed/GPU, shared experts, "
           "batch growth, shared workspace, profiling and scratch lifetime\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
