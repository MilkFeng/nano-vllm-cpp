#include "engine/model_runner.hh"
#include "engine/scheduler.hh"
#include "engine/tokenizer.hh"
#include "model_config.hh"
#include "nano-vllm/nano-vllm.hh"

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
        model_runner_(params, model_config_),
        scheduler_(model_runner_.num_blocks(), params.kvcache_block_size,
                   params.max_num_seqs, params.max_num_batched_tokens,
                   tokenizer_.eos_token_id()) {
    logger_->info("Initializing LLM engine");
  }

  ~Impl() = default;

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  Impl(Impl&&) = default;
  Impl& operator=(Impl&&) = default;

  std::vector<GenerationOutput> generate(std::span<const std::string> prompts,
                                         SamplingParams sampling_params) {
    for (const auto& prompt : prompts) {
      const auto token_ids = tokenizer_.encode(prompt);
      scheduler_.append_sequence(token_ids, sampling_params);
    }

    while (!scheduler_.finished()) {
      const auto scheduled = scheduler_.schedule();
    }
    return {};
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

std::vector<GenerationOutput> LLMEngine::generate(
    std::span<const std::string> prompts, SamplingParams sampling_params) {
  return pimpl_->generate(prompts, sampling_params);
}

} // namespace nanovllm
