#include "models/qwen35.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cpu/fp8.hh"

namespace nanovllm {
namespace {

bool is_float_source(DataType dtype) {
  return dtype == DataType::BFloat16 || dtype == DataType::Float32;
}

// mmap byte spans need not have float/uint16_t alignment. BF16 -> FP32 is
// exact.
void copy_float(const Checkpoint& checkpoint, std::string_view name,
                TensorView output) {
  const auto& source = checkpoint.tensor(name);
  const auto bytes = checkpoint.raw_data(name);
  if (source.dtype == DataType::Float32) {
    std::memcpy(output.data(), bytes.data(), bytes.size());
    return;
  }
  auto* result = output.data<float>();
  for (std::size_t i = 0; i < output.numel(); ++i) {
    std::uint16_t bits;
    std::memcpy(&bits, bytes.data() + i * sizeof(bits), sizeof(bits));
    result[i] = std::bit_cast<float>(std::uint32_t{bits} << 16);
  }
}

Tensor float_tensor(const Checkpoint& checkpoint, std::string_view name) {
  auto result = Tensor::empty(checkpoint.tensor(name).shape, DataType::Float32);
  copy_float(checkpoint, name, result.view());
  return result;
}

struct LoadEntry {
  std::string name;
  TensorView destination;
  std::string scale_name; // Nonempty for FP8 -> FP32 expansion.
};

struct LoadPlan {
  std::vector<LoadEntry> entries;
  std::set<std::string, std::less<>> sources;
  ModelLoadResult result;
  void count(const Checkpoint& checkpoint, const std::string& name) {
    if (!sources.insert(name).second) return;
    const auto bytes = checkpoint.tensor(name).byte_size();
    if (bytes > std::numeric_limits<std::uint64_t>::max() - result.num_bytes)
      throw std::overflow_error("loaded Qwen3.5 byte count is too large");
    ++result.num_tensors;
    result.num_bytes += bytes;
  }
};

LoadPlan plan_load(Module& module, const ModelConfig& config,
                   const Checkpoint& checkpoint, std::string_view prefix,
                   bool allow_packed_experts) {
  std::string stem(prefix);
  if (!stem.empty() && !stem.ends_with('.')) stem += '.';
  LoadPlan plan;
  for (const auto& [name, parameter] : module.named_parameters()) {
    LoadEntry entry{stem + name, parameter->view(), {}};
    const auto destination = entry.destination;
    if (!destination.is_contiguous()
        || (destination.dtype() != DataType::Float32
            && !(allow_packed_experts
                 && destination.dtype() == DataType::Float8E4M3FN
                 && destination.device().type == DeviceType::Cpu)))
      throw std::invalid_argument(
          "Qwen3.5 non-expert destinations must be contiguous FP32 tensors");
    const auto& source = checkpoint.tensor(entry.name);
    if (!std::ranges::equal(source.shape, destination.shape()))
      throw std::runtime_error(
          std::format("tensor '{}' has an unexpected shape", entry.name));
    if (destination.dtype() == DataType::Float8E4M3FN) {
      if (source.dtype != DataType::Float8E4M3FN)
        throw std::runtime_error(
            std::format("tensor '{}' must be F8_E4M3", entry.name));
    } else if (source.dtype == DataType::Float8E4M3FN) {
      if (!config.quantization_config || source.shape.size() != 2
          || !entry.name.ends_with(".weight"))
        throw std::runtime_error(
            "FP8 expansion needs block quantization and a 2D .weight tensor");
      entry.scale_name =
          entry.name.substr(0, entry.name.size() - 6) + "weight_scale_inv";
      const auto& scale = checkpoint.tensor(entry.scale_name);
      const auto blocks = config.quantization_config->weight_block_size;
      const std::array<std::size_t, 2> shape{
          source.shape[0] / blocks[0] + (source.shape[0] % blocks[0] != 0),
          source.shape[1] / blocks[1] + (source.shape[1] % blocks[1] != 0)};
      if (!is_float_source(scale.dtype)
          || !std::ranges::equal(scale.shape, shape))
        throw std::runtime_error(
            std::format("tensor '{}' must be a matching BF16/F32 scale grid",
                        entry.scale_name));
      plan.count(checkpoint, entry.scale_name);
    } else if (!is_float_source(source.dtype)) {
      throw std::runtime_error(std::format(
          "tensor '{}' must be BF16, F32 or block F8_E4M3", entry.name));
    }
    plan.count(checkpoint, entry.name);
    plan.entries.push_back(std::move(entry));
  }
  return plan;
}

void fill_cpu(const Checkpoint& checkpoint, const LoadEntry& entry,
              TensorView output, const ModelConfig& config) {
  if (!entry.scale_name.empty()) {
    auto scale = float_tensor(checkpoint, entry.scale_name);
    const auto& info = checkpoint.tensor(entry.name);
    ConstTensorView weight(checkpoint.raw_data(entry.name).data(), info.dtype,
                           TensorLayout(info.shape), Device::cpu());
    const auto blocks = config.quantization_config->weight_block_size;
    kernels::cpu::dequantize_fp8(weight, scale.view().as_const(), output,
                                 blocks[0], blocks[1]);
  } else if (output.dtype() == DataType::Float8E4M3FN) {
    const auto bytes = checkpoint.raw_data(entry.name);
    std::memcpy(output.data(), bytes.data(), bytes.size());
  } else {
    copy_float(checkpoint, entry.name, output);
  }
}

void apply_load(const LoadPlan& plan, const ModelConfig& config,
                const Checkpoint& checkpoint) {
  for (const auto& entry : plan.entries) {
    if (entry.destination.device().type == DeviceType::Cpu) {
      fill_cpu(checkpoint, entry, entry.destination, config);
    } else {
      auto staging = Tensor::empty(
          {entry.destination.shape().begin(), entry.destination.shape().end()},
          entry.destination.dtype());
      fill_cpu(checkpoint, entry, staging.view(), config);
      const CudaDeviceGuard guard(entry.destination.device().index);
      check_cuda(cudaMemcpy(entry.destination.data(), staging.data(),
                            staging.size_bytes(), cudaMemcpyHostToDevice),
                 std::format("load Qwen3.5 tensor '{}'", entry.name));
    }
  }
}

} // namespace

MoEConfig qwen35_moe_config(const ModelConfig& config) {
  if (!config.quantization_config)
    throw std::invalid_argument(
        "Qwen3.5 expert store currently requires an FP8 checkpoint");
  return {config.hidden_size,
          config.moe_intermediate_size,
          config.num_experts,
          config.num_experts_per_tok,
          config.shared_expert_intermediate_size,
          DataType::Float8E4M3FN};
}

ModelLoadResult load_qwen35_module(Module& module, const ModelConfig& config,
                                   const Checkpoint& checkpoint,
                                   std::string_view checkpoint_prefix) {
  // Packed expert parameters must go through the MoE loader and its cache
  // invalidation, including when nested in a larger module tree.
  const auto plan =
      plan_load(module, config, checkpoint, checkpoint_prefix, false);
  apply_load(plan, config, checkpoint);
  return plan.result;
}

ModelLoadResult load_qwen35_moe_layer(MoE& layer, const ModelConfig& config,
                                      const Checkpoint& checkpoint,
                                      std::size_t layer_index) {
  const auto expected = qwen35_moe_config(config);
  const auto& actual = layer.config();
  if (layer_index >= config.num_hidden_layers)
    throw std::out_of_range("Qwen3.5 layer index is out of range");
  if (actual.hidden_size != expected.hidden_size
      || actual.intermediate_size != expected.intermediate_size
      || actual.num_experts != expected.num_experts
      || actual.top_k != expected.top_k
      || actual.shared_expert_intermediate_size
             != expected.shared_expert_intermediate_size
      || actual.expert_dtype != expected.expert_dtype)
    throw std::invalid_argument(
        "MoE does not match the Qwen3.5 FP8 configuration");
  const auto plan = plan_load(
      layer, config, checkpoint,
      std::format("model.language_model.layers.{}.mlp", layer_index), true);
  // Metadata failures preserve the old parameters and populated cache.
  layer.invalidate_expert_cache();
  apply_load(plan, config, checkpoint);
  return plan.result;
}

} // namespace nanovllm
