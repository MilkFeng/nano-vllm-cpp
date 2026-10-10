#include "core/model_config.hh"

#include <cassert>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace nanovllm {
namespace {

DataType parse_model_dtype(std::string_view name) {
  if (name == "float16") return DataType::Float16;
  if (name == "bfloat16") return DataType::BFloat16;
  if (name == "float32") return DataType::Float32;
  throw std::invalid_argument("unsupported model dtype: " + std::string(name));
}

const char* model_dtype_name(DataType dtype) {
  switch (dtype) {
    case DataType::Float16:
      return "float16";
    case DataType::BFloat16:
      return "bfloat16";
    case DataType::Float32:
      return "float32";
    default:
      throw std::invalid_argument("unsupported model dtype");
  }
}

} // namespace

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(RoPEScalingType,
                                 {{RoPEScalingType::Linear, "linear"},
                                  {RoPEScalingType::Dynamic, "dynamic"},
                                  {RoPEScalingType::Yarn, "yarn"}})

nlohmann::json RoPEScaling::to_json() const {
  return {
      {"rope_type", rope_type},
      {"factor", factor},
      {"low_freq_factor", low_freq_factor},
      {"high_freq_factor", high_freq_factor},
      {"original_max_position_embeddings", original_max_position_embeddings},
  };
}

RoPEScaling RoPEScaling::from_json(const nlohmann::json& j) {
  return {
      .rope_type = j.at("rope_type").get<RoPEScalingType>(),
      .factor = j.at("factor").get<float>(),
      .low_freq_factor = j.at("low_freq_factor").get<float>(),
      .high_freq_factor = j.at("high_freq_factor").get<float>(),
      .original_max_position_embeddings =
          j.at("original_max_position_embeddings").get<std::size_t>(),
  };
}

VLLM_IMPL_NLOHMANN_JSON(RoPEScaling)

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(HiddenAct, {{HiddenAct::GeLU, "gelu"},
                                             {HiddenAct::SiLU, "silu"}})

nlohmann::json ModelConfig::to_json() const {
  return {
      {"model_type", model_type},
      {"architectures", architectures},

      {"vocab_size", vocab_size},
      {"hidden_size", hidden_size},
      {"head_dim", head_dim},
      {"intermediate_size", intermediate_size},
      {"num_hidden_layers", num_hidden_layers},

      {"num_attention_heads", num_attention_heads},
      {"num_key_value_heads", num_key_value_heads},
      {"attention_bias", attention_bias},

      {"max_position_embeddings", max_position_embeddings},
      {"rope_theta", rope_theta},
      {"rope_scaling", rope_scaling},

      {"rms_norm_eps", rms_norm_eps},
      {"hidden_act", hidden_act},

      {"tie_word_embeddings", tie_word_embeddings},

      {"dtype", model_dtype_name(dtype)},
  };
}

ModelConfig ModelConfig::from_json(const nlohmann::json& j) {
  const ModelConfig config = {
      .model_type = j.at("model_type").get<std::string>(),
      .architectures = j.at("architectures").get<std::vector<std::string>>(),

      .vocab_size = j.at("vocab_size").get<std::size_t>(),
      .hidden_size = j.at("hidden_size").get<std::size_t>(),
      .head_dim = j.at("head_dim").get<std::size_t>(),
      .intermediate_size = j.at("intermediate_size").get<std::size_t>(),
      .num_hidden_layers = j.at("num_hidden_layers").get<std::size_t>(),

      .num_attention_heads = j.at("num_attention_heads").get<std::size_t>(),
      .num_key_value_heads = j.at("num_key_value_heads").get<std::size_t>(),
      .attention_bias = j.at("attention_bias").get<bool>(),

      .max_position_embeddings =
          j.at("max_position_embeddings").get<std::size_t>(),
      .rope_theta = j.at("rope_theta").get<float>(),
      .rope_scaling = j.at("rope_scaling").get<std::optional<RoPEScaling>>(),

      .rms_norm_eps = j.at("rms_norm_eps").get<float>(),
      .hidden_act = j.at("hidden_act").get<HiddenAct>(),

      .tie_word_embeddings = j.at("tie_word_embeddings").get<bool>(),

      .dtype = parse_model_dtype(j.at("torch_dtype").get<std::string>()),
  };

  config.assert_valid();
  return config;
}

void ModelConfig::assert_valid() const {
  assert(vocab_size > 0);
  assert(hidden_size > 0);
  assert(intermediate_size > 0);
  assert(num_hidden_layers > 0);
  assert(num_attention_heads > 0);
  assert(num_key_value_heads > 0);
  assert(head_dim > 0);
  assert(num_attention_heads % num_key_value_heads == 0);
  assert(max_position_embeddings > 0);
  assert(std::isfinite(rope_theta) && rope_theta > 0);
  assert(std::isfinite(rms_norm_eps) && rms_norm_eps > 0);
}

ModelConfig ModelConfig::read_from_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  nlohmann::json j;
  file >> j;
  return from_json(j);
}

VLLM_IMPL_NLOHMANN_JSON(ModelConfig)

} // namespace nanovllm
