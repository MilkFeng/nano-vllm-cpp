#pragma once

#include <cstddef>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.hh"

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

enum class TorchDataType { Float16, BFloat16, Float32 };

VLLM_DEFINE_NLOHMANN_JSON(TorchDataType)

struct ModelConfig {
  std::string model_type;
  std::vector<std::string> architectures;

  std::size_t vocab_size;
  std::size_t hidden_size;
  std::size_t head_dim;
  std::size_t intermediate_size;
  std::size_t num_hidden_layers;

  std::size_t num_attention_heads;
  std::size_t num_key_value_heads;
  bool attention_bias;

  std::size_t max_position_embeddings;
  float rope_theta;
  std::optional<RoPEScaling> rope_scaling;

  float rms_norm_eps;
  HiddenAct hidden_act;

  bool tie_word_embeddings;

  TorchDataType dtype;

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static ModelConfig from_json(const nlohmann::json& j);

  [[nodiscard]] static ModelConfig read_from_file(
      const std::filesystem::path& path);

  // Debug preconditions shared by JSON loading and direct model construction.
  void assert_valid() const;

  [[nodiscard]] std::size_t q_size() const noexcept {
    return num_attention_heads * head_dim;
  }

  [[nodiscard]] std::size_t kv_size() const noexcept {
    return num_key_value_heads * head_dim;
  }
};

VLLM_DEFINE_NLOHMANN_JSON(ModelConfig)

} // namespace nanovllm
