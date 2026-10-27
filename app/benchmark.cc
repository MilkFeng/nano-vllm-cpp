#include <cuda_runtime_api.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/model_config.hh"
#include "engine/tokenizer.hh"
#include "nanovllm/nanovllm.hh"

namespace {

using Json = nlohmann::json;

class SilentLogger final : public nanovllm::Logger {
 public:
  void log(nanovllm::LogLevel, std::string_view) override {}
};

Json ratio(double numerator, double denominator) {
  return denominator > 0 ? Json(numerator / denominator) : Json(nullptr);
}

Json milliseconds(const std::optional<double>& seconds) {
  return seconds ? Json(*seconds * 1000) : Json(nullptr);
}

Json describe(std::vector<double> values) {
  if (values.empty())
    return {{"count", 0},     {"mean", nullptr}, {"stddev", nullptr},
            {"min", nullptr}, {"p50", nullptr},  {"p95", nullptr},
            {"max", nullptr}};
  std::sort(values.begin(), values.end());
  const auto mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0;
  for (const auto value : values) variance += (value - mean) * (value - mean);
  const auto percentile = [&](double p) {
    const double index = p * (values.size() - 1);
    const auto lo = static_cast<std::size_t>(index);
    const auto hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (index - lo);
  };
  return {{"count", values.size()},
          {"mean", mean},
          {"stddev", std::sqrt(variance / values.size())},
          {"min", values.front()},
          {"p50", percentile(0.5)},
          {"p95", percentile(0.95)},
          {"max", values.back()}};
}

Json serialize(const nanovllm::GenerationMetrics& metrics) {
  Json requests = Json::array();
  for (const auto& request : metrics.requests) {
    requests.push_back({{"input_tokens", request.input_tokens},
                        {"output_tokens", request.output_tokens},
                        {"ttft_ms", milliseconds(request.ttft_seconds)},
                        {"tpot_ms", milliseconds(request.tpot_seconds)},
                        {"latency_ms", request.latency_seconds * 1000}});
  }
  return {
      {"input_tokens", metrics.input_tokens},
      {"output_tokens", metrics.output_tokens},
      {"prefill",
       {{"steps", metrics.prefill_steps},
        {"input_tokens", metrics.prefill_input_tokens},
        {"output_tokens", metrics.prefill_output_tokens},
        {"ms", metrics.prefill_seconds * 1000},
        {"input_tokens_per_second",
         ratio(metrics.prefill_input_tokens, metrics.prefill_seconds)}}},
      {"decode",
       {{"steps", metrics.decode_steps},
        {"output_tokens", metrics.decode_output_tokens},
        {"ms", metrics.decode_seconds * 1000},
        {"ms_per_step",
         ratio(metrics.decode_seconds * 1000, metrics.decode_steps)},
        {"output_tokens_per_second",
         ratio(metrics.decode_output_tokens, metrics.decode_seconds)}}},
      {"inference_ms", metrics.inference_seconds * 1000},
      {"output_tokens_per_second",
       ratio(metrics.output_tokens, metrics.inference_seconds)},
      {"total_ms", metrics.total_seconds * 1000},
      {"detokenization_ms", metrics.detokenization_seconds * 1000},
      {"requests", std::move(requests)},
  };
}

Json summarize(const Json& runs) {
  std::map<std::string, std::vector<double>> samples;
  const auto add = [&](const std::string& name, const Json& value) {
    auto& values = samples[name];
    if (value.is_number()) values.push_back(value.get<double>());
  };
  for (const auto& run : runs) {
    add("prefill_ms", run["prefill"]["ms"]);
    add("prefill_input_tokens_per_second",
        run["prefill"]["input_tokens_per_second"]);
    add("decode_ms", run["decode"]["ms"]);
    add("decode_ms_per_step", run["decode"]["ms_per_step"]);
    add("decode_output_tokens_per_second",
        run["decode"]["output_tokens_per_second"]);
    add("inference_ms", run["inference_ms"]);
    add("output_tokens_per_second", run["output_tokens_per_second"]);
    for (const auto& request : run["requests"]) {
      add("ttft_ms", request["ttft_ms"]);
      add("tpot_ms", request["tpot_ms"]);
      add("request_latency_ms", request["latency_ms"]);
    }
  }
  Json summary = Json::object();
  for (auto& [name, values] : samples)
    summary[name] = describe(std::move(values));
  return summary;
}

std::string display(const Json& value) {
  if (!value.is_number()) return "n/a";
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value.get<double>();
  return out.str();
}

void save(const std::filesystem::path& path, const Json& report) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("cannot write benchmark report: " + path.string());
  out << report.dump(2) << '\n';
  out.close();
  if (!out)
    throw std::runtime_error("failed to write benchmark report: "
                             + path.string());
}

} // namespace

int main(int argc, char** argv) {
  CLI::App app{
      "Prefill/decode benchmark with exact-length synthetic token inputs"};
  std::string model_path, output_path = "benchmark.json";
  std::string seed_text =
      "The quick brown fox jumps over the lazy dog. "
      "Explain how computers process information. ";
  std::vector<std::size_t> input_lengths{128}, output_lengths{128},
      batch_sizes{1};
  std::size_t warmup = 1, repeat = 3;
  nanovllm::EngineParams params;
  params.seed = 42;
  params.enable_prefix_cache = false;
  params.max_num_batched_tokens = 1024;
  app.add_option("-m,--model", model_path, "Model directory")->required();
  app.add_option("--input-tokens", input_lengths,
                 "Exact input lengths, comma-separated")
      ->delimiter(',')
      ->check(CLI::PositiveNumber);
  app.add_option("--output-tokens", output_lengths,
                 "Output lengths, comma-separated")
      ->delimiter(',')
      ->check(CLI::PositiveNumber);
  app.add_option("--batch-sizes", batch_sizes,
                 "Request counts, comma-separated")
      ->delimiter(',')
      ->check(CLI::PositiveNumber);
  app.add_option("--warmup", warmup,
                 "Unmeasured generations per case (default: 1)")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--repeat", repeat,
                 "Measured generations per case (default: 3)")
      ->check(CLI::PositiveNumber);
  app.add_option("--prompt", seed_text,
                 "Seed text whose token IDs are cycled to fill inputs");
  app.add_option("--num-blocks", params.num_kvcache_blocks,
                 "Shared KV cache block count")
      ->required()
      ->check(CLI::PositiveNumber);
  app.add_option("--block-size", params.kvcache_block_size,
                 "Tokens per KV cache block")
      ->check(CLI::PositiveNumber);
  app.add_option(
         "--max-batched-tokens", params.max_num_batched_tokens,
         "Per-step token budget, including chunked prefill (default: 1024)")
      ->check(CLI::PositiveNumber);
  app.add_option("--cpu-threads", params.cpu_threads,
                 "Qwen3.5 CPU expert workers")
      ->check(CLI::PositiveNumber);
  app.add_option("--gpu-expert-ids", params.gpu_expert_ids,
                 "Qwen3.5 fixed GPU expert IDs, comma-separated")
      ->delimiter(',');
  app.add_option("--seed", params.seed,
                 "Random seed recorded with the run (default: 42)");
  app.add_option("-o,--output", output_path,
                 "JSON report (default: benchmark.json)");
  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  try {
    const auto max_input =
        *std::max_element(input_lengths.begin(), input_lengths.end());
    const auto max_output =
        *std::max_element(output_lengths.begin(), output_lengths.end());
    if (max_input > std::numeric_limits<std::size_t>::max() - max_output) {
      throw std::invalid_argument("input + output length overflows size_t");
    }
    params.max_model_len = max_input + max_output;
    params.max_num_seqs =
        *std::max_element(batch_sizes.begin(), batch_sizes.end());
    const auto config = nanovllm::ModelConfig::read_from_file(
        std::filesystem::path(model_path) / "config.json");
    // Validate before allocating the model or constructing large input vectors.
    if (params.max_model_len > config.max_position_embeddings) {
      throw std::invalid_argument(
          "input + output length exceeds model context limit");
    }
    nanovllm::Tokenizer tokenizer(
        std::make_shared<SilentLogger>(),
        std::filesystem::path(model_path) / "tokenizer.json",
        std::filesystem::path(model_path) / "tokenizer_config.json");
    const auto seed_ids = tokenizer.encode(seed_text);
    if (seed_ids.empty())
      throw std::invalid_argument("seed prompt encodes to no tokens");

    nanovllm::check_cuda(cudaSetDevice(0), "select benchmark GPU");
    cudaDeviceProp properties{};
    nanovllm::check_cuda(cudaGetDeviceProperties(&properties, 0),
                         "read GPU properties");
    int runtime = 0, driver = 0;
    nanovllm::check_cuda(cudaRuntimeGetVersion(&runtime),
                         "read CUDA runtime version");
    nanovllm::check_cuda(cudaDriverGetVersion(&driver),
                         "read CUDA driver version");
    const auto now = std::time(nullptr);
    std::ostringstream timestamp;
    timestamp << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
    Json report = {
        {"schema_version", 1},
        {"completed", false},
        {"started_at_utc", timestamp.str()},
        {"model", std::filesystem::absolute(model_path).string()},
        {"model_type", config.model_type},
        {"build_type", NANOVLLM_BENCH_BUILD_TYPE},
        {"gpu",
         {{"name", properties.name},
          {"total_memory_bytes", properties.totalGlobalMem},
          {"compute_capability", {properties.major, properties.minor}},
          {"cuda_runtime_version", runtime},
          {"cuda_driver_version", driver}}},
        {"cpu_logical_threads", std::thread::hardware_concurrency()},
        {"settings",
         {{"input_tokens", input_lengths},
          {"output_tokens", output_lengths},
          {"batch_sizes", batch_sizes},
          {"warmup", warmup},
          {"repeat", repeat},
          {"prefix_cache", false},
          {"temperature", 0},
          {"ignore_eos", true},
          {"seed", params.seed},
          {"seed_text", seed_text},
          {"max_model_len", params.max_model_len},
          {"max_seqs", params.max_num_seqs},
          {"max_batched_tokens", params.max_num_batched_tokens},
          {"num_blocks", params.num_kvcache_blocks},
          {"block_size", params.kvcache_block_size},
          {"cpu_threads", params.cpu_threads},
          {"gpu_expert_ids", params.gpu_expert_ids}}},
        {"cases", Json::array()},
    };
    save(output_path, report);
    std::cout << "Loading " << model_path << " on " << properties.name
              << "...\n"
              << "Synthetic token inputs; prefix cache disabled; times exclude "
                 "model load, "
                 "chat templates and tokenization.\n"
              << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    nanovllm::LLMEngine engine(model_path, params);
    report["model_load_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now()
                                      - load_start)
            .count();
    std::cout << std::setw(7) << "Input" << std::setw(8) << "Output"
              << std::setw(7) << "Batch" << std::setw(13) << "Prefill ms"
              << std::setw(14) << "Prefill tok/s" << std::setw(13)
              << "TTFT p50 ms" << std::setw(13) << "TTFT p95 ms"
              << std::setw(14) << "Decode tok/s" << std::setw(13)
              << "TPOT p50 ms" << '\n';
    for (const auto input_length : input_lengths) {
      for (const auto output_length : output_lengths) {
        for (const auto batch_size : batch_sizes) {
          std::vector<std::vector<std::int32_t>> prompts(
              batch_size, std::vector<std::int32_t>(input_length));
          for (std::size_t request = 0; request < batch_size; ++request) {
            for (std::size_t token = 0; token < input_length; ++token) {
              prompts[request][token] =
                  seed_ids[(token + request) % seed_ids.size()];
            }
          }
          const nanovllm::SamplingParams sampling{.temperature = 0,
                                                  .max_tokens = output_length,
                                                  .ignore_eos = true};
          const auto run = [&](nanovllm::GenerationMetrics* metrics) {
            const auto outputs = engine.generate(prompts, sampling, metrics);
            for (const auto& output : outputs) {
              if (output.token_ids.size() != output_length) {
                throw std::runtime_error(
                    "generation did not reach requested output length");
              }
            }
          };
          for (std::size_t i = 0; i < warmup; ++i) run(nullptr);
          Json runs = Json::array();
          for (std::size_t i = 0; i < repeat; ++i) {
            nanovllm::GenerationMetrics metrics;
            run(&metrics);
            runs.push_back(serialize(metrics));
          }
          auto summary = summarize(runs);
          std::cout
              << std::setw(7) << input_length << std::setw(8) << output_length
              << std::setw(7) << batch_size << std::setw(13)
              << display(summary["prefill_ms"]["mean"]) << std::setw(14)
              << display(summary["prefill_input_tokens_per_second"]["mean"])
              << std::setw(13) << display(summary["ttft_ms"]["p50"])
              << std::setw(13) << display(summary["ttft_ms"]["p95"])
              << std::setw(14)
              << display(summary["decode_output_tokens_per_second"]["mean"])
              << std::setw(13) << display(summary["tpot_ms"]["p50"]) << '\n'
              << std::flush;
          report["cases"].push_back({{"input_tokens", input_length},
                                     {"output_tokens", output_length},
                                     {"batch_size", batch_size},
                                     {"runs", std::move(runs)},
                                     {"summary", std::move(summary)}});
          save(output_path, report);
        }
      }
    }
    report["completed"] = true;
    save(output_path, report);
    std::cout << "Report: " << output_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
