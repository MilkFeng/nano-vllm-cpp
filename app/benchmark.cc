#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common.hh"
#include "nanovllm/nanovllm.hh"

namespace {

using Json = nlohmann::json;

struct GPUInfo {
  std::string name;
  std::size_t total_memory_bytes = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int cuda_driver_version = 0;
  int cuda_runtime_version = 0;
};

struct GPUMemoryInfo {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

[[nodiscard]] GPUInfo gpu_info() {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "read GPU properties");
  GPUInfo info{.name = properties.name,
               .total_memory_bytes = properties.totalGlobalMem,
               .compute_capability_major = properties.major,
               .compute_capability_minor = properties.minor};
  check_cuda(cudaRuntimeGetVersion(&info.cuda_runtime_version),
             "read CUDA runtime version");
  check_cuda(cudaDriverGetVersion(&info.cuda_driver_version),
             "read CUDA driver version");
  return info;
}

[[nodiscard]] GPUMemoryInfo gpu_memory_info() {
  GPUMemoryInfo info;
  check_cuda(cudaMemGetInfo(&info.free_bytes, &info.total_bytes),
             "read GPU memory");
  return info;
}

Json ratio(double numerator, double denominator) {
  return denominator > 0 ? Json(numerator / denominator) : Json(nullptr);
}

Json milliseconds(const std::optional<double>& seconds) {
  return seconds ? Json(*seconds * 1000) : Json(nullptr);
}

Json describe(std::vector<double> values) {
  if (values.empty()) {
    return {{"count", 0},     {"mean", nullptr}, {"stddev", nullptr},
            {"min", nullptr}, {"p50", nullptr},  {"p95", nullptr},
            {"max", nullptr}};
  }
  std::sort(values.begin(), values.end());
  const auto mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0;
  for (const auto value : values) {
    variance += (value - mean) * (value - mean);
  }
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

Json serialize(const nanovllm::MoEPhaseProfile& phase) {
  Json experts = Json::array();
  for (std::size_t id = 0; id < phase.experts.size(); ++id) {
    const auto& expert = phase.experts[id];
    if (expert.calls == 0) {
      continue;
    }
    Json groups = Json::object();
    for (const auto& [rows, calls] : expert.group_sizes) {
      groups[std::to_string(rows)] = calls;
    }
    experts.push_back({{"id", id},
                       {"tokens", expert.tokens},
                       {"calls", expert.calls},
                       {"group_sizes", std::move(groups)}});
  }
  return {{"calls", phase.calls},
          {"input_tokens", phase.input_tokens},
          {"gpu_assignments", phase.gpu_assignments},
          {"cpu_assignments", phase.cpu_assignments},
          {"gpu_assignment_ratio",
           ratio(phase.gpu_assignments,
                 phase.gpu_assignments + phase.cpu_assignments)},
          {"moe_ms", phase.seconds * 1000},
          {"experts", std::move(experts)}};
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
  Json result = {
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
  if (!metrics.expert_profile.empty()) {
    auto& profile = result["expert_profile"] = Json::array();
    for (std::size_t layer = 0; layer < metrics.expert_profile.size();
         ++layer) {
      const auto& entry = metrics.expert_profile[layer];
      profile.push_back({{"layer", layer},
                         {"prefill", serialize(entry.prefill)},
                         {"decode", serialize(entry.decode)}});
    }
  }
  if (!metrics.expert_adaptation.resident_ids_by_layer.empty()) {
    const auto& adapted = metrics.expert_adaptation;
    result["expert_adaptation"] = {
        {"rounds", adapted.rounds},
        {"swaps", adapted.swaps},
        {"bytes_uploaded", adapted.bytes_uploaded},
        {"ms", adapted.seconds * 1000},
        {"gpu_assignments", adapted.gpu_assignments},
        {"cpu_assignments", adapted.cpu_assignments},
        {"gpu_assignment_ratio",
         ratio(adapted.gpu_assignments,
               adapted.gpu_assignments + adapted.cpu_assignments)},
        {"resident_ids_by_layer", adapted.resident_ids_by_layer}};
  }
  return result;
}

Json summarize(const Json& runs) {
  std::map<std::string, std::vector<double>> samples;
  const auto add = [&](const std::string& name, const Json& value) {
    auto& values = samples[name];
    if (value.is_number()) {
      values.push_back(value.get<double>());
    }
  };
  for (const auto& run : runs) {
    if (run.contains("expert_adaptation")) {
      for (const auto* key :
           {"swaps", "bytes_uploaded", "ms", "gpu_assignment_ratio"}) {
        add(std::string("expert_adaptation_") + key,
            run["expert_adaptation"][key]);
      }
    }
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
  for (auto& [name, values] : samples) {
    summary[name] = describe(std::move(values));
  }
  return summary;
}

std::string display(const Json& value) {
  if (!value.is_number()) {
    return "n/a";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value.get<double>();
  return out.str();
}

void save(const std::filesystem::path& path, const Json& report) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write benchmark report: " + path.string());
  }
  out << report.dump(2) << '\n';
  out.close();
  if (!out) {
    throw std::runtime_error("failed to write benchmark report: "
                             + path.string());
  }
}

} // namespace

int main(int argc, char** argv) {
  int exit_code = 0;
  auto parsed = nanovllm::app::parse_args(
      argc, argv, nanovllm::app::Mode::Benchmark, exit_code);
  if (!parsed) {
    return exit_code;
  }
  auto& options = *parsed;
  auto& params = options.params;

  try {
    const auto config =
        nanovllm::LLMEngine::inspect(options.model_path, params);
    const auto& placement = config.gpu_expert_ids_by_layer;
    // Validate before allocating the model or constructing large input vectors.
    if (params.max_model_len > config.max_position_embeddings) {
      throw std::invalid_argument(
          "input + output length exceeds model context limit");
    }
    std::vector<std::string> seed_texts{options.seed_text};
    if (!options.prompts_path.empty()) {
      std::ifstream input(options.prompts_path);
      const auto document = Json::parse(input);
      if (!document.is_array() || document.empty()) {
        throw std::invalid_argument(
            "prompts file must be a nonempty JSON string array");
      }
      seed_texts = document.get<std::vector<std::string>>();
    }
    const auto properties = gpu_info();
    const auto now = std::time(nullptr);
    std::ostringstream timestamp;
    timestamp << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
    Json report = {
        {"schema_version", 2},
        {"completed", false},
        {"started_at_utc", timestamp.str()},
        {"model", std::filesystem::absolute(options.model_path).string()},
        {"model_type", config.model_type},
        {"build_type", NANOVLLM_BENCH_BUILD_TYPE},
        {"gpu",
         {{"name", properties.name},
          {"total_memory_bytes", properties.total_memory_bytes},
          {"compute_capability",
           {properties.compute_capability_major,
            properties.compute_capability_minor}},
          {"cuda_runtime_version", properties.cuda_runtime_version},
          {"cuda_driver_version", properties.cuda_driver_version}}},
        {"cpu_logical_threads", std::thread::hardware_concurrency()},
        {"settings",
         {{"input_tokens", options.input_lengths},
          {"output_tokens", options.output_lengths},
          {"batch_sizes", options.batch_sizes},
          {"warmup", options.warmup},
          {"repeat", options.repeat},
          {"prefix_cache", false},
          {"temperature", 0},
          {"ignore_eos", true},
          {"seed", params.seed},
          {"seed_text", options.prompts_path.empty() ? Json(options.seed_text)
                                                     : Json(nullptr)},
          {"prompt_texts", seed_texts},
          {"profile_experts", params.profile_experts},
          {"expert_adaptation",
           {{"every", params.expert_adaptation.every},
            {"max_swaps", params.expert_adaptation.max_swaps},
            {"decay", params.expert_adaptation.decay},
            {"min_usage", params.expert_adaptation.min_usage},
            {"min_gain", params.expert_adaptation.min_gain}}},
          {"max_model_len", params.max_model_len},
          {"max_seqs", params.max_num_seqs},
          {"max_batched_tokens", params.max_num_batched_tokens},
          {"num_blocks", params.num_kvcache_blocks},
          {"block_size", params.kvcache_block_size},
          {"cpu_threads", params.cpu_threads},
          {"cpu_affinity", params.cpu_affinity},
          {"gpu_expert_ids", params.gpu_expert_ids}}},
        {"cases", Json::array()},
    };
    if (!placement.empty()) {
      Json layers = Json::object();
      std::size_t slots = 0;
      for (std::size_t i = 0; i < placement.size(); ++i) {
        layers[std::to_string(i)] = placement[i];
        slots += placement[i].size();
      }
      report["expert_placement"] = {
          {"num_hidden_layers", config.num_hidden_layers},
          {"num_experts", config.num_experts},
          {"top_k", config.top_k},
          {"hidden_size", config.hidden_size},
          {"intermediate_size", config.intermediate_size},
          {"expert_size_bytes", config.expert_size_bytes},
          {"gpu_slots", slots},
          {"cache_bytes", slots * config.expert_size_bytes},
          {"layers", std::move(layers)}};
    }
    save(options.output_path, report);
    std::cout << "Loading " << options.model_path << " on " << properties.name
              << "...\n"
              << "Synthetic token inputs; prefix cache disabled; times exclude "
                 "model load, "
                 "chat templates and tokenization.\n"
              << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    nanovllm::LLMEngine engine(options.model_path, params);
    report["model_load_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now()
                                      - load_start)
            .count();
    const auto memory = gpu_memory_info();
    report["memory_after_load"] = {{"free_bytes", memory.free_bytes},
                                   {"total_bytes", memory.total_bytes}};
    std::vector<std::vector<std::int32_t>> seed_ids;
    for (const auto& text : seed_texts) {
      seed_ids.push_back(engine.encode(text));
      if (seed_ids.back().empty()) {
        throw std::invalid_argument("seed prompt encodes to no tokens");
      }
    }
    if (params.profile_experts) {
      std::cout
          << "Expert profiling enabled: timings include profiling overhead.\n";
    }
    std::cout << std::setw(7) << "Input" << std::setw(8) << "Output"
              << std::setw(7) << "Batch" << std::setw(13) << "Prefill ms"
              << std::setw(14) << "Prefill tok/s" << std::setw(13)
              << "TTFT p50 ms" << std::setw(13) << "TTFT p95 ms"
              << std::setw(14) << "Decode tok/s" << std::setw(13)
              << "TPOT p50 ms" << '\n';
    for (const auto input_length : options.input_lengths) {
      for (const auto output_length : options.output_lengths) {
        for (const auto batch_size : options.batch_sizes) {
          const nanovllm::SamplingParams sampling{.temperature = 0,
                                                  .max_tokens = output_length,
                                                  .ignore_eos = true};
          const auto run = [&](std::size_t round,
                               nanovllm::GenerationMetrics* metrics) {
            std::vector<std::vector<std::int32_t>> prompts(
                batch_size, std::vector<std::int32_t>(input_length));
            for (std::size_t request = 0; request < batch_size; ++request) {
              const auto& seed =
                  seed_ids[(round * batch_size + request) % seed_ids.size()];
              for (std::size_t token = 0; token < input_length; ++token) {
                prompts[request][token] = seed[(token + request) % seed.size()];
              }
            }
            const auto outputs = engine.generate(prompts, sampling, metrics);
            for (const auto& output : outputs) {
              if (output.token_ids.size() != output_length) {
                throw std::runtime_error(
                    "generation did not reach requested output length");
              }
            }
            return outputs;
          };
          for (std::size_t i = 0; i < options.warmup; ++i) {
            run(i, nullptr);
          }
          Json runs = Json::array();
          for (std::size_t i = 0; i < options.repeat; ++i) {
            nanovllm::GenerationMetrics metrics;
            const auto outputs = run(i, &metrics);
            auto record = serialize(metrics);
            auto& token_ids = record["token_ids"] = Json::array();
            for (std::size_t request = 0; request < outputs.size(); ++request) {
              const auto& output = outputs[request];
              token_ids.push_back(output.token_ids);
              record["requests"][request]["token_ids"] = output.token_ids;
              record["requests"][request]["text"] = output.text;
            }
            runs.push_back(std::move(record));
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
          save(options.output_path, report);
        }
      }
    }
    report["completed"] = true;
    save(options.output_path, report);
    std::cout << "Report: " << options.output_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
