#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
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

struct EngineParams {
  std::size_t max_num_batched_tokens = 16384;
  std::size_t max_num_seqs = 512;
  std::size_t max_model_len = 4096;
  std::size_t num_kvcache_blocks = 0;
  std::size_t kvcache_block_size = 256;
  std::uint64_t seed = 0;

  std::vector<std::int32_t> gpu_expert_ids;
  std::size_t cpu_threads = 8;
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
  // Render text messages using the checkpoint's tokenizer chat template.
  [[nodiscard]] std::string apply_chat_template(
      std::span<const ChatMessage> messages,
      ChatTemplateOptions options = {}) const;

  // Prompts are already formatted; this method performs raw text generation.
  [[nodiscard]] std::vector<GenerationOutput> generate(
      std::span<const std::string> prompts,
      SamplingParams sampling_params = {});

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};
} // namespace nanovllm
