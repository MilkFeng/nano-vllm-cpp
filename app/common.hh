#pragma once

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "nanovllm/nanovllm.hh"

namespace nanovllm::app {

enum class Mode { Generate, Benchmark };

struct Options {
  std::string model_path;
  EngineParams params;
  SamplingParams sampling;
  std::size_t repeat = 1;

  std::vector<std::string> prompts;
  bool thinking = true;
  bool json_output = false;
  bool report_metrics = false;

  std::vector<std::size_t> input_lengths{128};
  std::vector<std::size_t> output_lengths{128};
  std::vector<std::size_t> batch_sizes{1};
  std::size_t warmup = 1;
  std::string output_path = "benchmark.json";
  std::string prompts_path;
  std::string seed_text =
      "The quick brown fox jumps over the lazy dog. "
      "Explain how computers process information. ";
};

// Help and parse errors return an exit status without starting the engine.
inline std::optional<Options> parse_args(int argc, char** argv, Mode mode,
                                         int& exit_code) {
  const bool benchmark = mode == Mode::Benchmark;
  Options options;
  auto& params = options.params;
  if (benchmark) {
    params.seed = 42;
    params.enable_prefix_cache = false;
    params.max_num_batched_tokens = 1024;
    options.repeat = 3;
  } else {
    std::mt19937_64 random(std::random_device{}());
    params.seed = random();
  }

  CLI::App app{
      benchmark
          ? "Prefill/decode benchmark with exact-length synthetic token inputs"
          : "Single-GPU Qwen3 / Qwen3.5 MoE inference with C++ and CUDA"};
  app.add_option(
         "-m,--model", options.model_path,
         "Model directory containing weights, config and tokenizer files")
      ->required();
  app.add_option("--num-blocks", params.num_kvcache_blocks,
                 "Total KV cache blocks shared by all sequences")
      ->required()
      ->check(CLI::PositiveNumber);
  app.add_option("--block-size", params.kvcache_block_size,
                 "Tokens per KV cache block")
      ->check(CLI::PositiveNumber);
  app.add_option("--max-batched-tokens", params.max_num_batched_tokens,
                 "Per-step token budget; smaller values split prefill")
      ->check(CLI::PositiveNumber);
  app.add_option("--cpu-threads", params.cpu_threads,
                 "Qwen3.5 CPU expert workers shared by all layers (default: 8)")
      ->check(CLI::PositiveNumber);
  app.add_flag("--cpu-affinity", params.cpu_affinity,
               "Bind workers to physical cores; reserve caller core when possible");
  app.add_option("--gpu-expert-ids", params.gpu_expert_ids,
                 "Initial GPU expert IDs for every Qwen3.5 layer; the list "
                 "length sets each layer's cache capacity (default: empty)")
      ->delimiter(',');
  app.add_option("--expert-adapt-every", params.expert_adaptation.every,
                 "Adjust GPU expert identities every N decode scheduling "
                 "steps; 0 keeps static placement (default: 0)")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--expert-adapt-swaps", params.expert_adaptation.max_swaps,
                 "Maximum replacements across all layers per adjustment "
                 "without changing slot counts (default: 8)")
      ->check(CLI::PositiveNumber);
  app.add_option("--expert-adapt-decay", params.expert_adaptation.decay,
                 "Usage multiplier after each adjustment (default: 0.7)")
      ->check(CLI::Range(0.0, 1.0));
  app.add_option("--seed", params.seed, "Sampling random seed");
  app.add_option(
         "--repeat", options.repeat,
         benchmark
             ? "Measured generations per case (default: 3)"
             : "Repeat requests on the same engine, printing the last result")
      ->check(CLI::PositiveNumber);

  bool disable_prefix_cache = false;
  if (benchmark) {
    app.add_option("--input-tokens", options.input_lengths,
                   "Exact input lengths, comma-separated")
        ->delimiter(',')
        ->check(CLI::PositiveNumber);
    app.add_option("--output-tokens", options.output_lengths,
                   "Output lengths, comma-separated")
        ->delimiter(',')
        ->check(CLI::PositiveNumber);
    app.add_option("--batch-sizes", options.batch_sizes,
                   "Request counts, comma-separated")
        ->delimiter(',')
        ->check(CLI::PositiveNumber);
    app.add_option("--warmup", options.warmup,
                   "Unmeasured generations per case (default: 1)")
        ->check(CLI::NonNegativeNumber);
    auto* prompt =
        app.add_option("--prompt", options.seed_text,
                       "Seed text whose token IDs are cycled to fill inputs");
    app.add_option(
           "--prompts-file", options.prompts_path,
           "JSON array of seed texts, rotated across requests and repeats")
        ->check(CLI::ExistingFile)
        ->excludes(prompt);
    app.add_flag("--profile-experts", params.profile_experts,
                 "Collect per-layer prefill/decode routing counts and group "
                 "sizes; adds overhead");
    app.add_option("-o,--output", options.output_path,
                   "JSON report (default: benchmark.json)");
  } else {
    app.add_option("-p,--prompt", options.prompts,
                   "User message; the model chat template is applied by "
                   "default; may be repeated")
        ->required();
    app.add_flag(
        "--thinking", options.thinking,
        "Enable thinking in the model chat template (enabled by default)");
    app.add_option("--max-tokens", options.sampling.max_tokens,
                   "Maximum new tokens per prompt");
    app.add_option("--temperature", options.sampling.temperature,
                   "Sampling temperature; 0 selects greedy decoding");
    app.add_flag("--ignore-eos", options.sampling.ignore_eos,
                 "Continue generation after EOS");
    app.add_option("--max-model-len", params.max_model_len,
                   "Maximum total prompt and output length")
        ->check(CLI::PositiveNumber);
    app.add_option("--max-seqs", params.max_num_seqs,
                   "Maximum sequences per batch; also live recurrent state "
                   "slots for Qwen3.5")
        ->check(CLI::PositiveNumber);
    app.add_flag("--json", options.json_output,
                 "Print generated text and token IDs as JSON");
    app.add_flag(
        "--metrics", options.report_metrics,
        "Log separate prefill/decode timings and per-request TTFT/TPOT");
    app.add_flag("--no-prefix-cache", disable_prefix_cache,
                 "Disable prefix reuse, including across repeated generations");
  }

  try {
    app.parse(argc, app.ensure_utf8(argv));
    if (benchmark) {
      const auto max_input = *std::max_element(options.input_lengths.begin(),
                                               options.input_lengths.end());
      const auto max_output = *std::max_element(options.output_lengths.begin(),
                                                options.output_lengths.end());
      if (max_input > std::numeric_limits<std::size_t>::max() - max_output) {
        throw CLI::ValidationError("--input-tokens,--output-tokens",
                                   "input + output length overflows size_t");
      }
      params.max_model_len = max_input + max_output;
      params.max_num_seqs = *std::max_element(options.batch_sizes.begin(),
                                              options.batch_sizes.end());
    }
  } catch (const CLI::ParseError& error) {
    exit_code = app.exit(error);
    return std::nullopt;
  }
  params.enable_prefix_cache = !benchmark && !disable_prefix_cache;
  exit_code = 0;
  return options;
}

} // namespace nanovllm::app
