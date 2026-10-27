#include "core/expert_placement.hh"

#include <charconv>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace nanovllm {
namespace {

std::vector<std::int32_t> read_ids(const nlohmann::json& value,
                                   std::size_t num_experts) {
  if (!value.is_array()) {
    throw std::invalid_argument("expert IDs must be an array of integers");
  }
  std::vector<std::int32_t> ids;
  for (const auto& id : value) {
    if (!id.is_number_integer() || id < 0 || id >= num_experts) {
      throw std::invalid_argument("expert ID outside model expert range");
    }
    ids.push_back(id.get<std::int32_t>());
  }
  return ids;
}

} // namespace

ExpertPlacement resolve_expert_placement(
    const ModelConfig& config, std::span<const std::int32_t> default_ids,
    const ExpertPlacement& layers) {
  if (config.architecture != ModelArchitecture::Qwen35Moe) {
    if (!default_ids.empty() || !layers.empty()) {
      throw std::invalid_argument("expert placement requires Qwen3.5 MoE");
    }
    return {};
  }
  if (!layers.empty() && !default_ids.empty()) {
    throw std::invalid_argument("use either uniform or per-layer expert IDs");
  }
  if (!layers.empty() && layers.size() != config.num_hidden_layers) {
    throw std::invalid_argument("expert placement must contain every layer");
  }
  auto result =
      layers.empty()
          ? ExpertPlacement(config.num_hidden_layers,
                            std::vector<std::int32_t>(default_ids.begin(),
                                                      default_ids.end()))
          : layers;
  for (const auto& ids : result) {
    std::vector<bool> seen(config.num_experts, false);
    for (const auto id : ids) {
      if (id < 0 || std::size_t(id) >= seen.size()) {
        throw std::invalid_argument("expert ID outside model expert range");
      }
      if (seen[id]) {
        throw std::invalid_argument("duplicate GPU expert ID");
      }
      seen[id] = true;
    }
  }
  return result;
}

ExpertPlacement read_expert_placement(const std::filesystem::path& path,
                                      const ModelConfig& config) {
  if (config.architecture != ModelArchitecture::Qwen35Moe) {
    throw std::invalid_argument("expert placement requires Qwen3.5 MoE");
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot read expert configuration: "
                             + path.string());
  }
  const auto document = nlohmann::json::parse(input);
  if (!document.is_object() || !document.contains("schema_version")
      || !document.at("schema_version").is_number_integer()
      || document.at("schema_version") != 1) {
    throw std::invalid_argument(
        "expert configuration requires schema_version 1");
  }
  if (document.contains("num_hidden_layers")
      && document.at("num_hidden_layers") != config.num_hidden_layers) {
    throw std::invalid_argument("expert configuration layer count mismatch");
  }
  if (document.contains("num_experts")
      && document.at("num_experts") != config.num_experts) {
    throw std::invalid_argument("expert configuration expert count mismatch");
  }
  if ((document.contains("hidden_size")
       && document.at("hidden_size") != config.hidden_size)
      || (document.contains("intermediate_size")
          && document.at("intermediate_size") != config.moe_intermediate_size)
      || (document.contains("top_k")
          && document.at("top_k") != config.num_experts_per_tok)
      || (document.contains("expert_size_bytes")
          && document.at("expert_size_bytes")
                 != routed_expert_size_bytes(config))) {
    throw std::invalid_argument(
        "expert configuration model dimensions mismatch");
  }
  const auto defaults = read_ids(
      document.value("default", nlohmann::json::array()), config.num_experts);
  ExpertPlacement result(config.num_hidden_layers, defaults);
  const auto& layers = document.at("layers");

  if (!layers.is_object()) {
    throw std::invalid_argument(
        "expert configuration layers must be an object");
  }

  for (const auto& [key, value] : layers.items()) {
    std::size_t layer = 0;
    const auto [end, error] =
        std::from_chars(key.data(), key.data() + key.size(), layer);
    if (error != std::errc{} || end != key.data() + key.size()
        || layer >= result.size() || key != std::to_string(layer)) {
      throw std::invalid_argument("invalid expert configuration layer: " + key);
    }
    result[layer] = read_ids(value, config.num_experts);
  }
  return resolve_expert_placement(config, {}, result);
}

std::size_t routed_expert_size_bytes(const ModelConfig& config) {
  const auto h = config.hidden_size;
  const auto i = config.moe_intermediate_size;
  return 3 * h * i + 3 * ((h + 127) / 128) * ((i + 127) / 128) * sizeof(float);
}

EngineParams resolve_engine_params(const ModelConfig& config,
                                   EngineParams params) {
  if (!params.gpu_expert_config.empty()) {
    if (!params.gpu_expert_ids.empty()
        || !params.gpu_expert_ids_by_layer.empty()) {
      throw std::invalid_argument(
          "use either an expert configuration file or explicit expert IDs");
    }
    params.gpu_expert_ids_by_layer =
        read_expert_placement(params.gpu_expert_config, config);
    params.gpu_expert_config.clear();
  }
  (void)resolve_expert_placement(config, params.gpu_expert_ids,
                                 params.gpu_expert_ids_by_layer);
  if (params.profile_experts
      && config.architecture != ModelArchitecture::Qwen35Moe) {
    throw std::invalid_argument("expert profiling requires Qwen3.5 MoE");
  }
  return params;
}

ModelInfo inspect_model(const std::filesystem::path& model_path,
                        const EngineParams& params) {
  const auto config = ModelConfig::read_from_file(model_path / "config.json");
  const auto resolved = resolve_engine_params(config, params);
  const bool moe = config.architecture == ModelArchitecture::Qwen35Moe;
  return {
      .model_type = config.model_type,
      .max_position_embeddings = config.max_position_embeddings,
      .hidden_size = config.hidden_size,
      .intermediate_size =
          moe ? config.moe_intermediate_size : config.intermediate_size,
      .num_hidden_layers = config.num_hidden_layers,
      .num_experts = config.num_experts,
      .top_k = config.num_experts_per_tok,
      .expert_size_bytes = moe ? routed_expert_size_bytes(config) : 0,
      .gpu_expert_ids_by_layer = resolve_expert_placement(
          config, resolved.gpu_expert_ids, resolved.gpu_expert_ids_by_layer),
  };
}

} // namespace nanovllm
