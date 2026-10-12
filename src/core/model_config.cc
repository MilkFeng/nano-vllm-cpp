#include "core/model_config.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

namespace nanovllm {
namespace {

DataType parse_model_dtype(std::string_view name) {
  if (name == "float16") {
    return DataType::Float16;
  }
  if (name == "bfloat16") {
    return DataType::BFloat16;
  }
  if (name == "float32") {
    return DataType::Float32;
  }

  throw std::invalid_argument("unsupported model dtype: " + std::string(name));
}

const std::string_view model_dtype_name(DataType dtype) {
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

ModelArchitecture architecture_for(const std::vector<std::string>& names) {
  if (names.size() != 1) {
    throw std::invalid_argument(
        "expected exactly one supported model architecture");
  }

  if (names[0] == "Qwen3ForCausalLM") {
    return ModelArchitecture::Qwen3;
  } else if (names[0] == "Qwen3_5MoeForConditionalGeneration"
             || names[0] == "Qwen3_5MoeForCausalLM") {
    return ModelArchitecture::Qwen35Moe;
  }

  throw std::invalid_argument("unsupported model architecture: " + names[0]);
}

} // namespace

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(RoPEScalingType,
                                 {{RoPEScalingType::Linear, "linear"},
                                  {RoPEScalingType::Dynamic, "dynamic"},
                                  {RoPEScalingType::Yarn, "yarn"}})

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(HiddenAct, {{HiddenAct::GeLU, "gelu"},
                                             {HiddenAct::SiLU, "silu"}})

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(ModelArchitecture,
                                 {{ModelArchitecture::Qwen3, "Qwen 3"},
                                  {ModelArchitecture::Qwen35Moe,
                                   "Qwen 3.5 Moe"}})

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(AttentionType,
                                 {{AttentionType::Full, "full_attention"},
                                  {AttentionType::Linear, "linear_attention"}})

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

nlohmann::json FP8QuantizationConfig::to_json() const {
  return {
      {"quant_method", quant_method},
      {"activation_scheme", activation_scheme},
      {"weight_block_size", weight_block_size},
      {"weight_per_tensor", weight_per_tensor},
      {"act_per_tensor", act_per_tensor},
  };
}

FP8QuantizationConfig FP8QuantizationConfig::from_json(
    const nlohmann::json& j) {
  FP8QuantizationConfig config = {};

  config.quant_method = j.at("quant_method").get<std::string>();
  config.activation_scheme = j.at("activation_scheme").get<std::string>();
  config.weight_per_tensor = j.value("weight_per_tensor", false);
  config.act_per_tensor = j.value("act_per_tensor", false);

  assert(config.quant_method == "fp8");
  assert(config.activation_scheme == "dynamic");
  assert(!config.weight_per_tensor);
  assert(!config.act_per_tensor);

  const auto& blocks = j.at("weight_block_size");
  assert(blocks.is_array() && blocks.size() == 2);

  config.weight_block_size = {
      blocks[0].get<std::size_t>(),
      blocks[1].get<std::size_t>(),
  };

  assert(config.weight_block_size[0] == 128
         && config.weight_block_size[1] == 128);

  return config;
}

VLLM_IMPL_NLOHMANN_JSON(FP8QuantizationConfig)

ModelConfig ModelConfig::from_json(const nlohmann::json& j) {
  ModelConfig config;

  config.architectures = j.at("architectures").get<std::vector<std::string>>();
  config.architecture = architecture_for(config.architectures);
  config.model_type = j.at("model_type").get<std::string>();

  if (config.architecture == ModelArchitecture::Qwen3) {
    assert(config.model_type == "qwen3");
  } else if (config.architecture == ModelArchitecture::Qwen35Moe) {
    if (config.architectures[0] == "Qwen3_5MoeForConditionalGeneration") {
      // Omini
      assert(config.model_type == "qwen3_5_moe");
    } else {
      assert(config.model_type == "qwen3_5_moe_text");
    }
  }

  const auto& text =
      config.architectures[0] == "Qwen3_5MoeForConditionalGeneration"
          ? j.at("text_config")
          : j;

  if (config.architecture == ModelArchitecture::Qwen35Moe) {
    assert(text.at("model_type").get<std::string>() == "qwen3_5_moe_text");
  }

  config.vocab_size = text.at("vocab_size").get<std::size_t>();
  config.hidden_size = text.at("hidden_size").get<std::size_t>();
  config.head_dim = text.at("head_dim").get<std::size_t>();
  config.num_hidden_layers = text.at("num_hidden_layers").get<std::size_t>();
  config.num_attention_heads =
      text.at("num_attention_heads").get<std::size_t>();
  config.num_key_value_heads =
      text.at("num_key_value_heads").get<std::size_t>();
  config.attention_bias = text.value("attention_bias", false);
  config.max_position_embeddings =
      text.at("max_position_embeddings").get<std::size_t>();
  config.rms_norm_eps = text.at("rms_norm_eps").get<float>();
  config.hidden_act = text.at("hidden_act").get<HiddenAct>();
  config.tie_word_embeddings = j.value("tie_word_embeddings", false);
  config.dtype = parse_model_dtype(
      text.at(text.contains("dtype") ? "dtype" : "torch_dtype")
          .get<std::string>());
  if (j.contains("quantization_config")
      && !j.at("quantization_config").is_null())
    config.quantization_config =
        j.at("quantization_config").get<FP8QuantizationConfig>();

  if (config.architecture != ModelArchitecture::Qwen35Moe) {
    config.intermediate_size = text.at("intermediate_size").get<std::size_t>();
    config.rope_theta = text.at("rope_theta").get<float>();
    config.rope_scaling =
        j.at("rope_scaling").get<std::optional<RoPEScaling>>();
  } else {
    assert(!text.contains("mlp_only_layers")
           || (text.at("mlp_only_layers").is_array()
               && text.at("mlp_only_layers").empty()));

    config.moe_intermediate_size =
        text.at("moe_intermediate_size").get<std::size_t>();
    config.shared_expert_intermediate_size =
        text.at("shared_expert_intermediate_size").get<std::size_t>();
    config.num_experts = text.at("num_experts").get<std::size_t>();
    config.num_experts_per_tok =
        text.at("num_experts_per_tok").get<std::size_t>();
    config.attn_output_gate = text.value("attn_output_gate", true);
    config.full_attention_interval = text.value("full_attention_interval", 4);

    if (text.contains("layer_types")) {
      config.layer_types =
          text.at("layer_types").get<std::vector<AttentionType>>();
    } else {
      for (std::size_t i = 0; i < config.num_hidden_layers; ++i)
        if ((i + 1) % config.full_attention_interval == 0) {
          config.layer_types.push_back(AttentionType::Full);
        } else {
          config.layer_types.push_back(AttentionType::Linear);
        }
    }

    config.linear_conv_kernel_dim =
        text.at("linear_conv_kernel_dim").get<std::size_t>();
    config.linear_key_head_dim =
        text.at("linear_key_head_dim").get<std::size_t>();
    config.linear_value_head_dim =
        text.at("linear_value_head_dim").get<std::size_t>();
    config.linear_num_key_heads =
        text.at("linear_num_key_heads").get<std::size_t>();
    config.linear_num_value_heads =
        text.at("linear_num_value_heads").get<std::size_t>();

    const auto& rope = text.at("rope_parameters");

    assert(rope.at("rope_type") == "default");

    config.rope_theta = rope.at("rope_theta").get<float>();
    config.partial_rotary_factor = rope.value("partial_rotary_factor", 0.25f);

    if (rope.contains("mrope_section")) {
      const auto& sections = rope.at("mrope_section");

      // mrope_section needs three dimensions
      assert(sections.is_array() && sections.size() == 3);

      for (std::size_t i = 0; i < 3; ++i) {
        config.mrope_section[i] = sections[i].get<std::size_t>();
        assert(config.mrope_section[i] != 0);
      }
    }
    config.mrope_interleaved = rope.value("mrope_interleaved", false);
  }

  config.assert_valid();

  return config;
}

nlohmann::json ModelConfig::to_json() const {
  nlohmann::json text{
      {"model_type", model_type},
      {"vocab_size", vocab_size},
      {"hidden_size", hidden_size},
      {"head_dim", head_dim},
      {"num_hidden_layers", num_hidden_layers},
      {"num_attention_heads", num_attention_heads},
      {"num_key_value_heads", num_key_value_heads},
      {"attention_bias", attention_bias},
      {"max_position_embeddings", max_position_embeddings},
      {"rms_norm_eps", rms_norm_eps},
      {"hidden_act", hidden_act},
  };
  if (architecture == ModelArchitecture::Qwen3) {
    text["intermediate_size"] = intermediate_size;
    text["rope_theta"] = rope_theta;
    text["rope_scaling"] =
        rope_scaling ? rope_scaling->to_json() : nlohmann::json(nullptr);
    text["torch_dtype"] = model_dtype_name(dtype);
  } else {
    text["model_type"] = "qwen3_5_moe_text";
    text["dtype"] = model_dtype_name(dtype);
    text["moe_intermediate_size"] = moe_intermediate_size;
    text["shared_expert_intermediate_size"] = shared_expert_intermediate_size;
    text["num_experts"] = num_experts;
    text["num_experts_per_tok"] = num_experts_per_tok;
    text["attn_output_gate"] = attn_output_gate;
    text["full_attention_interval"] = full_attention_interval;
    text["layer_types"] = layer_types;
    text["linear_conv_kernel_dim"] = linear_conv_kernel_dim;
    text["linear_key_head_dim"] = linear_key_head_dim;
    text["linear_value_head_dim"] = linear_value_head_dim;
    text["linear_num_key_heads"] = linear_num_key_heads;
    text["linear_num_value_heads"] = linear_num_value_heads;
    text["rope_parameters"] = {
        {"rope_type", "default"},
        {"rope_theta", rope_theta},
        {"partial_rotary_factor", partial_rotary_factor},
        {"mrope_interleaved", mrope_interleaved},
    };

    if (std::ranges::any_of(mrope_section, [](auto n) { return n != 0; })) {
      text["rope_parameters"]["mrope_section"] = mrope_section;
    }
  }
  nlohmann::json result;
  if (model_type == "qwen3_5_moe") {
    result = {
        {"model_type", model_type},
        {"text_config", text},
    };
  } else {
    result = std::move(text);
  }
  result["architectures"] = architectures;
  result["tie_word_embeddings"] = tie_word_embeddings;
  if (quantization_config) {
    result["quantization_config"] = quantization_config->to_json();
  }
  return result;
}

void ModelConfig::assert_valid() const {
  assert(architecture == architecture_for(architectures));
  assert(vocab_size > 0);
  assert(hidden_size > 0);
  assert(head_dim > 0);
  assert(num_hidden_layers > 0);
  assert(num_attention_heads > 0);
  assert(num_key_value_heads > 0);
  assert(num_attention_heads % num_key_value_heads == 0);
  assert(max_position_embeddings > 0);
  assert(std::isfinite(rope_theta) && rope_theta > 0);
  assert(std::isfinite(rms_norm_eps) && rms_norm_eps > 0);
  assert(std::isfinite(partial_rotary_factor) && partial_rotary_factor > 0
         && partial_rotary_factor <= 1);
  assert(rotary_dim() > 0 && rotary_dim() % 2 == 0);

  if (quantization_config) {
    assert(quantization_config->weight_block_size[0] == 128
           && quantization_config->weight_block_size[1] == 128);
  }

  if (architecture == ModelArchitecture::Qwen3) {
    assert(model_type == "qwen3");
    assert(intermediate_size > 0);
    assert(!quantization_config);
    assert(partial_rotary_factor == 1);
  } else {
    assert(hidden_act == HiddenAct::SiLU);
    assert(moe_intermediate_size > 0);
    assert(shared_expert_intermediate_size > 0);
    assert(num_experts > 0 && num_experts <= 1024);
    assert(num_experts_per_tok > 0 && num_experts_per_tok <= num_experts);
    assert(layer_types.size() == num_hidden_layers);
    assert(full_attention_interval > 0);

    assert(linear_conv_kernel_dim > 0 && linear_key_head_dim > 0
           && linear_value_head_dim > 0);
    assert(linear_num_key_heads > 0 && linear_num_value_heads > 0
           && linear_num_value_heads % linear_num_key_heads == 0);

    // Qwen3.5 requires gated attention and default RoPE
    assert(!rope_scaling && attn_output_gate);
  }
}

ModelConfig ModelConfig::read_from_file(const std::filesystem::path& path) {
  std::ifstream file(path);

  if (!file) {
    throw std::runtime_error(
        std::format("failed to open model config '{}'", path.string()));
  }
  return from_json(nlohmann::json::parse(file));
}

VLLM_IMPL_NLOHMANN_JSON(ModelConfig)
} // namespace nanovllm
