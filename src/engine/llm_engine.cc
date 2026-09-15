#include <cmath>
#include <stdexcept>

#include "engine/model_runner.hh"
#include "engine/scheduler.hh"
#include "engine/tokenizer.hh"
#include "model/model_config.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

namespace {

class NullLogger final : public Logger {
 public:
  void log(LogLevel, std::string_view) override {}
};

std::shared_ptr<Logger> null_logger() {
  static auto instance = std::make_shared<NullLogger>();
  return instance;
}

} // namespace

class LLMPath {
 public:
  LLMPath(const std::filesystem::path& model_path)
      : model_path_(model_path) {}

  std::filesystem::path model_path() const { return model_path_; }

  std::filesystem::path model_safetensors_path() const {
    return model_path_ / kModelSafetensorsPath;
  }

  std::filesystem::path model_config_path() const {
    return model_path_ / kModelConfigPath;
  }

  std::filesystem::path tokenizer_config_path() const {
    return model_path_ / kTokenizerConfigPath;
  }

  std::filesystem::path tokenizer_path() const {
    return model_path_ / kTokenizerPath;
  }

 private:
  std::filesystem::path model_path_;

  constexpr static std::string_view kModelSafetensorsPath = "model.safetensors";
  constexpr static std::string_view kModelConfigPath = "config.json";

  constexpr static std::string_view kTokenizerConfigPath =
      "tokenizer_config.json";
  constexpr static std::string_view kTokenizerPath = "tokenizer.json";
};

class LLMEngine::Impl {
 public:
  Impl(const std::filesystem::path& model_path, EngineParams params,
       std::shared_ptr<Logger> logger)
      : model_path_(model_path),
        params_(params),
        logger_(logger ? std::move(logger) : null_logger()),
        model_config_(
            ModelConfig::read_from_file(model_path_.model_config_path())),
        tokenizer_(model_path_.tokenizer_path(),
                   model_path_.tokenizer_config_path()),
        model_runner_(params, model_config_,
                      model_path_.model_safetensors_path()),
        scheduler_(model_runner_.num_blocks(), params.kvcache_block_size,
                   params.max_num_seqs, params.max_num_batched_tokens,
                   tokenizer_.eos_token_id(), params.max_model_len) {
    logger_->info("Loaded Qwen3 with "
                  + std::to_string(model_runner_.num_blocks())
                  + " KV cache blocks");
  }

  ~Impl() = default;

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  // LLMEngine moves the owning unique_ptr; the Impl object stays in place.
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  std::string apply_chat_template(std::span<const ChatMessage> messages,
                                  ChatTemplateOptions options) const {
    return tokenizer_.apply_chat_template(messages, options);
  }

  std::vector<GenerationOutput> generate(std::span<const std::string> prompts,
                                         SamplingParams sampling_params) {
    std::vector<std::vector<std::int32_t>> encoded;
    for (const auto& prompt : prompts) {
      auto ids = tokenizer_.encode(prompt);
      if (ids.empty() || ids.size() > params_.max_model_len) {
        throw std::invalid_argument(
            "prompt token count must be between 1 and max_model_len");
      }
      encoded.push_back(std::move(ids));
    }
    scheduler_.clear();
    std::vector<std::size_t> request_ids;
    try {
      for (const auto& ids : encoded) {
        request_ids.push_back(scheduler_.append_sequence(ids, sampling_params));
      }
      while (!scheduler_.finished()) {
        const auto scheduled = scheduler_.schedule();
        if (scheduled.seq_ids.empty()) {
          throw std::runtime_error("scheduler made no progress");
        }
        std::vector<const Sequence*> batch;
        for (const auto id : scheduled.seq_ids) {
          batch.push_back(&scheduler_.sequence(id));
        }
        const auto tokens = model_runner_.run(batch, scheduled.prefill);
        scheduler_.postprocess(scheduled, tokens);
      }
      std::vector<GenerationOutput> result;
      for (const auto id : request_ids) {
        const auto tokens = scheduler_.sequence(id).generated_token_ids();
        result.push_back(
            {tokenizer_.decode(tokens), {tokens.begin(), tokens.end()}});
      }
      scheduler_.clear();
      return result;
    } catch (...) {
      scheduler_.clear();
      throw;
    }
  }

 private:
  LLMPath model_path_;
  EngineParams params_;

  std::shared_ptr<Logger> logger_;

  ModelConfig model_config_;
  Tokenizer tokenizer_;

  ModelRunner model_runner_;
  Scheduler scheduler_;
};

LLMEngine::LLMEngine(const std::filesystem::path& model_path,
                     EngineParams params, std::shared_ptr<Logger> logger)
    : pimpl_(std::make_unique<Impl>(model_path, params, logger)) {}

LLMEngine::~LLMEngine() = default;

LLMEngine::LLMEngine(LLMEngine&& other) noexcept
    : pimpl_(std::move(other.pimpl_)) {}

LLMEngine& LLMEngine::operator=(LLMEngine&& other) noexcept {
  if (this != &other) {
    pimpl_ = std::move(other.pimpl_);
  }
  return *this;
}

std::string LLMEngine::apply_chat_template(
    std::span<const ChatMessage> messages, ChatTemplateOptions options) const {
  return pimpl_->apply_chat_template(messages, options);
}

std::vector<GenerationOutput> LLMEngine::generate(
    std::span<const std::string> prompts, SamplingParams sampling_params) {
  return pimpl_->generate(prompts, sampling_params);
}

} // namespace nanovllm
