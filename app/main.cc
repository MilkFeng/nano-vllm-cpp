#include <CLI/CLI.hpp>
#include <array>
#include <iostream>
#include <mutex>
#include <nanovllm/nanovllm.hh>
#include <nlohmann/json.hpp>

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
  CLI::App app{"Single-GPU Qwen3 inference with C++ and CUDA"};
  std::string model_path;
  std::vector<std::string> prompts;
  nanovllm::EngineParams params;
  nanovllm::SamplingParams sampling;
  bool json_output = false;
  bool thinking = true;
  std::size_t repeat = 1;

  app.add_option(
         "-m,--model", model_path,
         "Directory containing Qwen3 model.safetensors and tokenizer files")
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
                 "Maximum sequences scheduled per batch");
  app.add_option("--max-batched-tokens", params.max_num_batched_tokens,
                 "Per-step token budget; smaller values split prefill");
  app.add_option("--block-size", params.kvcache_block_size,
                 "Tokens per KV cache block");
  app.add_option("--gpu-memory-utilization", params.gpu_memory_utilization,
                 "Fraction of total GPU memory available to this engine");
  app.add_option("--seed", params.seed, "Sampling random seed");
  app.add_option(
      "--repeat", repeat,
      "Repeat requests on the same engine, printing the last result");
  app.add_flag("--json", json_output,
               "Print generated text and token IDs as JSON");

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
      outputs = engine.generate(prompts, sampling);
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
