#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.hh"
#include "core/device.hh"

namespace nanovllm {
enum class RoPEScalingType { Linear, Dynamic, Yarn };

VLLM_DEFINE_NLOHMANN_JSON(RoPEScalingType)

struct RoPEScaling {
  RoPEScalingType rope_type;
  float factor;
  float low_freq_factor;
  float high_freq_factor;
  std::size_t original_max_position_embeddings;

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static RoPEScaling from_json(const nlohmann::json& j);
};

VLLM_DEFINE_NLOHMANN_JSON(RoPEScaling)

enum class HiddenAct { GeLU, SiLU };

VLLM_DEFINE_NLOHMANN_JSON(HiddenAct)

enum class ModelArchitecture { Qwen3, Qwen35Moe };

VLLM_DEFINE_NLOHMANN_JSON(ModelArchitecture)

enum class AttentionType { Full, Linear };

VLLM_DEFINE_NLOHMANN_JSON(AttentionType)

struct FP8QuantizationConfig {
  std::string quant_method;
  std::string activation_scheme;
  bool weight_per_tensor;
  bool act_per_tensor;

  std::array<std::size_t, 2> weight_block_size = {};

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static FP8QuantizationConfig from_json(const nlohmann::json& j);
};

VLLM_DEFINE_NLOHMANN_JSON(FP8QuantizationConfig)

struct ModelConfig {
  ModelArchitecture architecture;
  std::string model_type;
  std::vector<std::string> architectures;

  std::size_t vocab_size = 0;
  std::size_t hidden_size = 0;
  std::size_t head_dim = 0;
  std::size_t intermediate_size = 0; // Dense Qwen3 only.
  std::size_t num_hidden_layers = 0;
  std::size_t num_attention_heads = 0;
  std::size_t num_key_value_heads = 0;
  bool attention_bias = false;

  std::size_t max_position_embeddings = 0;
  float rope_theta = 0;
  std::optional<RoPEScaling> rope_scaling;
  float rms_norm_eps = 0;
  HiddenAct hidden_act = HiddenAct::SiLU;
  bool tie_word_embeddings = false;

  // Dtype of unquantized checkpoint parameters. Qwen3.5 runtime currently
  // expands BF16/FP8 into FP32; dtype must not be overwritten with FP8/F32.
  DataType dtype = DataType::Float32;

  std::optional<FP8QuantizationConfig> quantization_config;

  // Qwen3.5 text settings. Unused dimensions remain zero for dense Qwen3.
  std::size_t moe_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t num_experts = 0;
  std::size_t num_experts_per_tok = 0;
  bool attn_output_gate = false;
  std::vector<AttentionType> layer_types;
  std::size_t full_attention_interval = 0;
  std::size_t linear_conv_kernel_dim = 0;
  std::size_t linear_key_head_dim = 0;
  std::size_t linear_value_head_dim = 0;
  std::size_t linear_num_key_heads = 0;
  std::size_t linear_num_value_heads = 0;
  float partial_rotary_factor = 1;
  std::array<std::size_t, 3> mrope_section = {};
  bool mrope_interleaved = false;

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static ModelConfig from_json(const nlohmann::json& j);
  [[nodiscard]] static ModelConfig read_from_file(
      const std::filesystem::path& path);

  void assert_valid() const;

  [[nodiscard]] std::size_t q_size() const noexcept {
    return num_attention_heads * head_dim;
  }

  [[nodiscard]] std::size_t kv_size() const noexcept {
    return num_key_value_heads * head_dim;
  }

  [[nodiscard]] std::size_t rotary_dim() const noexcept {
    return static_cast<std::size_t>(head_dim * partial_rotary_factor);
  }
};

VLLM_DEFINE_NLOHMANN_JSON(ModelConfig)
} // namespace nanovllm
