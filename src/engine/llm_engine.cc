#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <magic_enum/magic_enum.hpp>
#include <stdexcept>

#include "core/model_config.hh"
#include "engine/model_runner.hh"
#include "engine/scheduler.hh"
#include "engine/tokenizer.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

class NullLogger final : public Logger {
 public:
  void log(LogLevel, std::string_view) override {}
};

std::shared_ptr<Logger> null_logger() {
  static auto instance = std::make_shared<NullLogger>();
  return instance;
}

} // namespace

class LLMEngine::Impl {
 public:
  Impl(const std::filesystem::path& model_path, EngineParams params,
       std::shared_ptr<Logger> logger)
      : model_path_(model_path),
        params_(params),
        logger_(logger ? std::move(logger) : null_logger()),
        model_config_(ModelConfig::read_from_file(model_path_ / "config.json")),
        tokenizer_(logger_, model_path_ / "tokenizer.json",
                   model_path_ / "tokenizer_config.json"),
        model_runner_(logger_, params, model_config_, model_path_),
        scheduler_(model_runner_.num_blocks(), params.kvcache_block_size,
                   params.max_num_seqs, params.max_num_batched_tokens,
                   tokenizer_.eos_token_id(), params.max_model_len,
                   model_config_.architecture == ModelArchitecture::Qwen35Moe,
                   params.enable_prefix_cache) {
    logger_->info(std::format(
        "Loaded {}", magic_enum::enum_name(model_config_.architecture)));

    if (model_config_.architecture == ModelArchitecture::Qwen35Moe) {
      logger_->info(std::format(
          "{} recurrent state slots ({} bytes for conv and "
          "recurrent state); "
          "prefix cache disabled; {} GPU routed experts per layer",
          params_.max_num_seqs, model_runner_.state_cache_size_bytes(),
          params_.gpu_expert_ids.size()));
    }
  }

  ~Impl() = default;

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  std::string apply_chat_template(std::span<const ChatMessage> messages,
                                  ChatTemplateOptions options) const {
    return tokenizer_.apply_chat_template(messages, options);
  }

  std::vector<GenerationOutput> generate(std::span<const std::string> prompts,
                                         SamplingParams sampling_params,
                                         GenerationMetrics* metrics) {
    const auto start = Clock::now();
    std::vector<std::vector<std::int32_t>> encoded;
    for (const auto& prompt : prompts) {
      auto ids = tokenizer_.encode(prompt);
      if (ids.empty() || ids.size() > params_.max_model_len) {
        throw std::invalid_argument(
            "prompt token count must be between 1 and max_model_len");
      }
      encoded.push_back(std::move(ids));
    }
    const auto encoded_at = Clock::now();
    auto result = generate(encoded, sampling_params, metrics);
    if (metrics) {
      metrics->tokenization_seconds = elapsed_seconds(start, encoded_at);
      metrics->total_seconds = elapsed_seconds(start, Clock::now());
    }
    return result;
  }

  std::vector<GenerationOutput> generate(
      std::span<const std::vector<std::int32_t>> prompts,
      SamplingParams sampling_params, GenerationMetrics* metrics) {
    const auto start = Clock::now();
    GenerationMetrics stats;
    if (metrics) stats.requests.resize(prompts.size());
    for (const auto& ids : prompts) {
      if (ids.empty() || ids.size() > params_.max_model_len
          || std::ranges::any_of(ids, [&](auto id) {
               return id < 0 || std::size_t(id) >= model_config_.vocab_size;
             })) {
        throw std::invalid_argument(
            "invalid token input length or vocabulary ID");
      }
    }
    scheduler_.clear();
    std::vector<std::size_t> request_ids;
    try {
      const auto submitted_at = Clock::now();
      for (const auto& ids : prompts) {
        request_ids.push_back(scheduler_.append_sequence(ids, sampling_params));
        if (metrics) {
          stats.requests[request_ids.back()].input_tokens = ids.size();
          stats.input_tokens += ids.size();
        }
      }
      while (!scheduler_.finished()) {
        const auto step_start = metrics ? Clock::now() : Clock::time_point{};
        const auto scheduled = scheduler_.schedule();
        if (scheduled.seq_ids.empty()) {
          throw std::runtime_error("scheduler made no progress");
        }
        std::vector<const Sequence*> batch;
        std::vector<std::size_t> previous_outputs;
        for (const auto id : scheduled.seq_ids) {
          const auto& seq = scheduler_.sequence(id);
          batch.push_back(&seq);
          if (metrics) {
            previous_outputs.push_back(seq.num_generated_tokens());
            if (scheduled.prefill) {
              stats.prefill_input_tokens += seq.num_scheduled_tokens;
            }
          }
        }
        const auto tokens = model_runner_.run(batch, scheduled.prefill);
        scheduler_.postprocess(scheduled, tokens);
        if (metrics) {
          const auto completed_at = Clock::now();
          const auto step_seconds = elapsed_seconds(step_start, completed_at);
          if (scheduled.prefill) {
            ++stats.prefill_steps;
            stats.prefill_seconds += step_seconds;
          } else {
            ++stats.decode_steps;
            stats.decode_seconds += step_seconds;
          }
          for (std::size_t i = 0; i < scheduled.seq_ids.size(); ++i) {
            const auto id = scheduled.seq_ids[i];
            const auto generated =
                scheduler_.sequence(id).num_generated_tokens();
            if (generated == previous_outputs[i]) continue; // Partial prefill.
            auto& request = stats.requests[id];
            const auto latency = elapsed_seconds(submitted_at, completed_at);
            if (!request.ttft_seconds) request.ttft_seconds = latency;
            request.output_tokens = generated;
            request.latency_seconds = latency;
            if (generated > 1) {
              request.tpot_seconds =
                  (latency - *request.ttft_seconds) / (generated - 1);
            }
            const auto emitted = generated - previous_outputs[i];
            stats.output_tokens += emitted;
            (scheduled.prefill ? stats.prefill_output_tokens
                               : stats.decode_output_tokens) += emitted;
          }
        }
      }
      const auto inference_done = Clock::now();
      if (metrics) {
        stats.inference_seconds = elapsed_seconds(submitted_at, inference_done);
      }
      std::vector<GenerationOutput> result;
      for (const auto id : request_ids) {
        const auto tokens = scheduler_.sequence(id).generated_token_ids();
        result.push_back({
            tokenizer_.decode(tokens),
            {tokens.begin(), tokens.end()},
        });
      }
      const auto decoded_at = Clock::now();
      if (metrics) {
        stats.detokenization_seconds =
            elapsed_seconds(inference_done, decoded_at);
      }
      scheduler_.clear();
      if (metrics) {
        stats.total_seconds = elapsed_seconds(start, Clock::now());
        *metrics = std::move(stats);
      }
      return result;
    } catch (...) {
      scheduler_.clear();
      throw;
    }
  }

 private:
  std::filesystem::path model_path_;
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
    std::span<const std::string> prompts, SamplingParams sampling_params,
    GenerationMetrics* metrics) {
  return pimpl_->generate(prompts, sampling_params, metrics);
}

std::vector<GenerationOutput> LLMEngine::generate(
    std::span<const std::vector<std::int32_t>> prompts,
    SamplingParams sampling_params, GenerationMetrics* metrics) {
  return pimpl_->generate(prompts, sampling_params, metrics);
}

} // namespace nanovllm
