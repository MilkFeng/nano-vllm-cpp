#include <CLI/CLI.hpp>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <mutex>
#include <nanovllm/nanovllm.hh>
#include <nlohmann/json.hpp>
#include <random>

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
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

  CLI::App app{"Single-GPU Qwen3 / Qwen3.5 MoE inference with C++ and CUDA"};
  std::string model_path;
  std::vector<std::string> prompts;
  nanovllm::EngineParams params;
  nanovllm::SamplingParams sampling;
  bool json_output = false;
  bool thinking = true;
  std::size_t repeat = 1;

  params.seed = dis(gen);

  app.add_option("-m,--model", model_path,
                 "Model directory containing config, safetensors weights or a "
                 "shard index, and tokenizer files")
      ->required();
  app.add_option("-p,--prompt", prompts,
                 "User message; the model chat template is applied by default; "
                 "may be repeated")
      ->required();
  app.add_flag(
      "--thinking", thinking,
      "Enable thinking in the model chat template (enabled by default)");
  app.add_option("--max-tokens", sampling.max_tokens,
                 "Maximum new tokens per prompt");
  app.add_option("--temperature", sampling.temperature,
                 "Sampling temperature; 0 selects greedy decoding");
  app.add_flag("--ignore-eos", sampling.ignore_eos,
               "Continue generation after EOS");
  app.add_option("--max-model-len", params.max_model_len,
                 "Maximum total prompt and output length");
  app.add_option("--max-seqs", params.max_num_seqs,
                 "Maximum sequences scheduled per batch; also live recurrent "
                 "state slots for Qwen3.5");
  app.add_option("--max-batched-tokens", params.max_num_batched_tokens,
                 "Per-step token budget; smaller values split prefill");
  app.add_option("--block-size", params.kvcache_block_size,
                 "Tokens per KV cache block");
  app.add_option("--num-blocks", params.num_kvcache_blocks,
                 "Total KV cache blocks shared by all sequences")
      ->required()
      ->check(CLI::PositiveNumber);
  app.add_option("--gpu-expert-ids", params.gpu_expert_ids,
                 "Qwen3.5 fixed GPU routed expert IDs for every layer; empty "
                 "by default")
      ->delimiter(',');
  app.add_option("--cpu-threads", params.cpu_threads,
                 "Qwen3.5 CPU expert workers shared by all layers (default: 8)")
      ->check(CLI::PositiveNumber);
  app.add_option("--seed", params.seed, "Sampling random seed");
  app.add_option(
      "--repeat", repeat,
      "Repeat requests on the same engine, printing the last result");
  app.add_flag("--json", json_output,
               "Print generated text and token IDs as JSON");

  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  try {
    if (repeat == 0) {
      throw std::invalid_argument("repeat must be positive");
    }

    const auto logger = std::make_shared<ConsoleLogger>();
    nanovllm::LLMEngine engine(model_path, params, logger);

    for (auto& prompt : prompts) {
      const std::array messages{nanovllm::ChatMessage{"user", prompt}};
      prompt =
          engine.apply_chat_template(messages, {.enable_thinking = thinking});
    }

    std::vector<nanovllm::GenerationOutput> outputs;
    for (std::size_t i = 0; i < repeat; ++i) {
      const auto start = std::chrono::steady_clock::now();
      outputs = engine.generate(prompts, sampling);
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
          i + 1, repeat, generated_tokens, seconds, tokens_per_second));
    }

    if (json_output) {
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
