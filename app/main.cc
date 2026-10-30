#include <array>
#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <nanovllm/nanovllm.hh>
#include <nlohmann/json.hpp>

#include "common.hh"

class ConsoleLogger final : public nanovllm::Logger {
 public:
  void log(nanovllm::LogLevel level, std::string_view message) override {
    std::lock_guard lock(mutex_);

    std::cerr << '[' << level_name(level) << "] " << message << '\n';
  }

 private:
  static std::string_view level_name(nanovllm::LogLevel level) noexcept {
    using nanovllm::LogLevel;

    switch (level) {
      case LogLevel::Trace:
        return "TRACE";
      case LogLevel::Debug:
        return "DEBUG";
      case LogLevel::Info:
        return "INFO";
      case LogLevel::Warning:
        return "WARNING";
      case LogLevel::Error:
        return "ERROR";
    }

    return "UNKNOWN";
  }

  std::mutex mutex_;
};

int main(int argc, char** argv) {
  int exit_code = 0;
  auto parsed = nanovllm::app::parse_args(
      argc, argv, nanovllm::app::Mode::Generate, exit_code);
  if (!parsed) return exit_code;
  auto& options = *parsed;

  try {
    const auto logger = std::make_shared<ConsoleLogger>();
    const auto& params = options.params;
    nanovllm::LLMEngine engine(options.model_path, params, logger);

    for (auto& prompt : options.prompts) {
      const std::array messages{nanovllm::ChatMessage{"user", prompt}};
      prompt = engine.apply_chat_template(
          messages, {.enable_thinking = options.thinking});
    }

    std::vector<nanovllm::GenerationOutput> outputs;
    for (std::size_t i = 0; i < options.repeat; ++i) {
      const auto start = std::chrono::steady_clock::now();
      nanovllm::GenerationMetrics metrics;
      outputs = engine.generate(options.prompts, options.sampling,
                                options.report_metrics ? &metrics : nullptr);
      const auto seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
      std::size_t generated_tokens = 0;
      for (const auto& output : outputs) {
        generated_tokens += output.token_ids.size();
      }
      const auto tokens_per_second =
          seconds > 0.0 ? generated_tokens / seconds : 0.0;
      logger->info(std::format(
          "Generation {}/{}: {} output tokens in {:.3f} s "
          "({:.2f} tok/s; includes prefill)",
          i + 1, options.repeat, generated_tokens, seconds, tokens_per_second));
      if (options.report_metrics) {
        if (params.expert_adaptation.every != 0) {
          const auto& adapted = metrics.expert_adaptation;
          logger->info(std::format(
              "Expert adaptation: {} rounds, {} swaps, {} bytes uploaded, "
              "{:.3f} ms; decode GPU/CPU assignments {}/{}",
              adapted.rounds, adapted.swaps, adapted.bytes_uploaded,
              adapted.seconds * 1000, adapted.gpu_assignments,
              adapted.cpu_assignments));
        }
        logger->info(std::format(
            "Prefill: {} input tokens, {} steps, {:.3f} ms; "
            "Decode: {} output tokens, {} steps, {:.3f} ms ({:.2f} tok/s)",
            metrics.prefill_input_tokens, metrics.prefill_steps,
            metrics.prefill_seconds * 1000, metrics.decode_output_tokens,
            metrics.decode_steps, metrics.decode_seconds * 1000,
            metrics.decode_seconds > 0
                ? metrics.decode_output_tokens / metrics.decode_seconds
                : 0));
        for (std::size_t r = 0; r < metrics.requests.size(); ++r) {
          const auto& request = metrics.requests[r];
          logger->info(std::format(
              "Request {}: {} input / {} output tokens; TTFT {} ms; TPOT {} ms",
              r + 1, request.input_tokens, request.output_tokens,
              request.ttft_seconds
                  ? std::format("{:.3f}", *request.ttft_seconds * 1000)
                  : "n/a",
              request.tpot_seconds
                  ? std::format("{:.3f}", *request.tpot_seconds * 1000)
                  : "n/a"));
        }
      }
    }

    if (options.json_output) {
      auto result = nlohmann::json::array();
      for (const auto& output : outputs) {
        result.push_back(
            {{"text", output.text}, {"token_ids", output.token_ids}});
      }
      std::cout << result.dump() << '\n';
    } else {
      for (const auto& output : outputs) {
        std::cout << output.text << '\n';
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
