#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nanovllm {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct ChatTemplateOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
};

struct SamplingParams {
  float temperature = 1.0f;
  std::size_t max_tokens = 64;
  bool ignore_eos = false;
};

struct ExpertAdaptationParams {
  std::size_t every = 0; // Decode scheduling steps; 0 keeps static placement.
  std::size_t max_swaps = 8; // Across all layers per adjustment.
  double decay = 0.7;
  double min_usage = 2.0;
  double min_gain = 1.5;
};

struct EngineParams {
  std::size_t max_num_batched_tokens = 16384;
  std::size_t max_num_seqs = 512;
  std::size_t max_model_len = 4096;
  std::size_t num_kvcache_blocks = 0;
  std::size_t kvcache_block_size = 256;
  std::uint64_t seed = 0;

  std::vector<std::int32_t> gpu_expert_ids;
  std::size_t cpu_threads = 8;
  // One worker per allowed physical core. If workers leave a core free,
  // reserve that core for the GPU submission thread during model forward.
  bool cpu_affinity = false;
  bool profile_experts = false;
  bool enable_prefix_cache = true; // Qwen3.5 always disables prefix reuse.
  ExpertAdaptationParams expert_adaptation;
};

struct ModelInfo {
  std::string model_type;
  std::size_t max_position_embeddings = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t num_hidden_layers = 0;
  std::size_t num_experts = 0;
  std::size_t top_k = 0;
  std::size_t expert_size_bytes = 0;
  std::vector<std::vector<std::int32_t>> gpu_expert_ids_by_layer;
};

enum class LogLevel { Trace, Debug, Info, Warning, Error };

class Logger {
 public:
  virtual ~Logger() = default;

  virtual void log(LogLevel level, std::string_view message) = 0;

  void trace(std::string_view message) { log(LogLevel::Trace, message); }
  void debug(std::string_view message) { log(LogLevel::Debug, message); }
  void info(std::string_view message) { log(LogLevel::Info, message); }
  void warn(std::string_view message) { log(LogLevel::Warning, message); }
  void error(std::string_view message) { log(LogLevel::Error, message); }
};

struct GenerationOutput {
  std::string text;
  std::vector<std::int32_t> token_ids;
};

struct RequestMetrics {
  std::size_t input_tokens = 0;
  std::size_t output_tokens = 0;
  std::optional<double> ttft_seconds;
  std::optional<double> tpot_seconds;
  double latency_seconds = 0;
};

struct ExpertRoutingProfile {
  std::size_t tokens = 0;
  std::size_t calls = 0;
  std::map<std::size_t, std::size_t> group_sizes;
};

struct MoEPhaseProfile {
  std::size_t calls = 0;
  std::size_t input_tokens = 0;
  std::size_t gpu_assignments = 0;
  std::size_t cpu_assignments = 0;
  double seconds = 0;
  std::vector<ExpertRoutingProfile> experts;
};

struct MoELayerProfile {
  MoEPhaseProfile prefill;
  MoEPhaseProfile decode;
};

struct ExpertAdaptationStats {
  std::size_t rounds = 0;
  std::size_t swaps = 0;
  std::size_t bytes_uploaded = 0;
  std::size_t gpu_assignments = 0; // Decode only.
  std::size_t cpu_assignments = 0;
  double seconds = 0; // Selection and synchronous uploads.
  std::vector<std::vector<std::int32_t>> resident_ids_by_layer;
};

struct GenerationMetrics {
  std::size_t input_tokens = 0;
  std::size_t output_tokens = 0;

  std::size_t prefill_input_tokens = 0;
  std::size_t prefill_output_tokens = 0;
  std::size_t decode_output_tokens = 0;
  std::size_t prefill_steps = 0;
  std::size_t decode_steps = 0;

  double prefill_seconds = 0;
  double decode_seconds = 0;
  double inference_seconds = 0;
  double tokenization_seconds = 0;
  double detokenization_seconds = 0;
  double total_seconds = 0;
  std::vector<RequestMetrics> requests;
  std::vector<MoELayerProfile> expert_profile;
  ExpertAdaptationStats expert_adaptation;
};

class LLMEngine {
 public:
  LLMEngine(const std::filesystem::path& model_path, EngineParams params,
            std::shared_ptr<Logger> logger = nullptr);
  ~LLMEngine();

  LLMEngine(const LLMEngine&) = delete;
  LLMEngine& operator=(const LLMEngine&) = delete;

  LLMEngine(LLMEngine&&) noexcept;
  LLMEngine& operator=(LLMEngine&&) noexcept;

 public:
  [[nodiscard]] static ModelInfo inspect(
      const std::filesystem::path& model_path, const EngineParams& params = {});

 public:
  // Render text messages using the checkpoint's tokenizer chat template.
  [[nodiscard]] std::string apply_chat_template(
      std::span<const ChatMessage> messages,
      ChatTemplateOptions options = {}) const;

  // Encode raw text with the model tokenizer, without a chat template.
  [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;

  // Prompts are already formatted; this method performs raw text generation.
  [[nodiscard]] std::vector<GenerationOutput> generate(
      std::span<const std::string> prompts, SamplingParams sampling_params = {},
      GenerationMetrics* metrics = nullptr);

  // Exact-length token inputs, without a chat template or text tokenization.
  [[nodiscard]] std::vector<GenerationOutput> generate(
      std::span<const std::vector<std::int32_t>> prompts,
      SamplingParams sampling_params = {},
      GenerationMetrics* metrics = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};
} // namespace nanovllm
