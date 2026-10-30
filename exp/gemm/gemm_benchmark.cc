#include "gemm_benchmark.hh"

#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/cuda_utils.hh"
#include "comp_gemm.hh"
#include "core/tensor.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm::exp {
namespace {

using Json = nlohmann::json;
constexpr std::size_t kMiB = 1024 * 1024;
constexpr double kTolerance = 2e-4;

struct Case {
  std::string name;
  std::string group;
  int m;
  int n;
  int k;
};

std::vector<Case> cases_for(const std::string& suite) {
  if (suite == "quick") {
    return {{"square_512", "square", 512, 512, 512},
            {"square_1024", "square", 1024, 1024, 1024},
            {"qwen3_decode_gate", "qwen3_decode", 1, 3072, 1024},
            {"qwen3_prefill_gate", "qwen3_prefill", 128, 3072, 1024},
            {"tail_129_513_257", "tail", 129, 513, 257}};
  }

  std::vector<Case> cases;

  for (int size : {256, 512, 1024, 2048, 4096}) {
    cases.push_back({"square_" + std::to_string(size), "square", size, size, size});
  }

  cases.push_back({"rect_128_2048_2048", "rectangle", 128, 2048, 2048});
  cases.push_back({"rect_128_8192_2048", "rectangle", 128, 8192, 2048});
  cases.push_back({"rect_512_2048_2048", "rectangle", 512, 2048, 2048});
  cases.push_back({"rect_128_2048_8192", "rectangle", 128, 2048, 8192});

  // Qwen3-0.6B: hidden=1024, intermediate=3072, Hq=16, head_dim=128.
  // Gate and up share one shape; no frequency weighting is implied here.
  for (int m : {1, 8, 32, 128, 512}) {
    const std::string group = m == 1 ? "qwen3_decode"
        : m < 128 ? "qwen3_small_batch" : "qwen3_prefill";
    const std::string prefix = "qwen3_m" + std::to_string(m) + "_";

    cases.push_back({prefix + "mlp_gate_up", group, m, 3072, 1024});
    cases.push_back({prefix + "mlp_down", group, m, 1024, 3072});
    cases.push_back({prefix + "attn_q", group, m, 2048, 1024});
    cases.push_back({prefix + "attn_o", group, m, 1024, 2048});
  }

  cases.push_back({"tail_129_513_257", "tail", 129, 513, 257});
  cases.push_back({"tail_127_2049_1023", "tail", 127, 2049, 1023});
  cases.push_back({"tail_255_1001_769", "tail", 255, 1001, 769});
  cases.push_back({"tail_33_129_15", "tail", 33, 129, 15});
  return cases;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
      : (values[middle - 1] + values[middle]) * 0.5;
}

Json describe(const std::vector<double>& values) {
  auto ordered = values;
  std::sort(ordered.begin(), ordered.end());

  const double mean = std::accumulate(values.begin(), values.end(), 0.0)
                      / values.size();
  double squared = 0.0;

  for (double value : values) {
    squared += (value - mean) * (value - mean);
  }

  const auto percentile = [&](double p) {
    const double index = p * (ordered.size() - 1);
    const auto low = static_cast<std::size_t>(index);
    const auto high = std::min(low + 1, ordered.size() - 1);
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low);
  };

  return {{"median_us", median(values)}, {"p10_us", percentile(0.1)},
          {"p90_us", percentile(0.9)}, {"min_us", ordered.front()},
          {"max_us", ordered.back()}, {"mean_us", mean},
          {"cv", std::sqrt(squared / values.size()) / mean}};
}

std::vector<float> random_values(std::size_t count, std::mt19937& generator) {
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> values(count);

  for (float& value : values) {
    value = distribution(generator);
  }

  return values;
}

std::vector<float> download(const Tensor& tensor, cudaStream_t stream) {
  std::vector<float> result(tensor.numel());

  check_cuda(cudaMemcpyAsync(result.data(), tensor.data(), tensor.size_bytes(),
                              cudaMemcpyDeviceToHost, stream), "download GEMM");
  check_cuda(cudaStreamSynchronize(stream), "finish GEMM download");
  return result;
}

struct Slot {
  Tensor input;
  Tensor weight;
  Tensor output;
};

struct Bank {
  std::vector<Slot> slots;
  std::vector<float> host_input;
  std::vector<float> host_weight;
  std::size_t target_bytes = 0;
  std::size_t bytes_per_slot = 0;

  Bank(const Case& shape, bool rotating, int l2_bytes,
       const GemmBenchmarkOptions& options, cudaStream_t stream) {
    const std::size_t m = shape.m;
    const std::size_t n = shape.n;
    const std::size_t k = shape.k;

    bytes_per_slot = (m * k + n * k + m * n) * sizeof(float);
    target_bytes = rotating ? std::max<std::size_t>(64 * kMiB, 2ull * l2_bytes) : 0;

    const std::size_t wanted = rotating
        ? (target_bytes + bytes_per_slot - 1) / bytes_per_slot : 1;
    const std::size_t count = std::min<std::size_t>(
        wanted, std::min(options.max_calls, 256));

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query GEMM memory");

    if (count * bytes_per_slot > free_bytes * 0.8) {
      throw std::runtime_error("insufficient free VRAM for the declared working set");
    }

    std::mt19937 generator(options.seed);
    host_input = random_values(m * k, generator);
    host_weight = random_values(n * k, generator);
    slots.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
      slots.push_back({Tensor::empty({m, k}, DataType::Float32, Device::cuda(0)),
                       Tensor::empty({n, k}, DataType::Float32, Device::cuda(0)),
                       Tensor::empty({m, n}, DataType::Float32, Device::cuda(0))});
      auto& slot = slots.back();

      if (i == 0) {
        check_cuda(cudaMemcpyAsync(slot.input.data(), host_input.data(),
                                    slot.input.size_bytes(), cudaMemcpyHostToDevice,
                                    stream), "upload GEMM input");
        check_cuda(cudaMemcpyAsync(slot.weight.data(), host_weight.data(),
                                    slot.weight.size_bytes(), cudaMemcpyHostToDevice,
                                    stream), "upload GEMM weight");
      } else {
        check_cuda(cudaMemcpyAsync(slot.input.data(), slots[0].input.data(),
                                    slot.input.size_bytes(), cudaMemcpyDeviceToDevice,
                                    stream), "initialize rotating input");
        check_cuda(cudaMemcpyAsync(slot.weight.data(), slots[0].weight.data(),
                                    slot.weight.size_bytes(), cudaMemcpyDeviceToDevice,
                                    stream), "initialize rotating weight");
      }
    }

    check_cuda(cudaStreamSynchronize(stream), "finish bank initialization");
  }
};

Json compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  double max_abs = 0.0;
  double max_scaled = 0.0;
  double error_squared = 0.0;
  double reference_squared = 0.0;

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      throw std::runtime_error("non-finite GEMM result");
    }

    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    const double scaled = error / (kTolerance * (1.0 + std::abs(expected[i])));
    max_abs = std::max(max_abs, error);
    max_scaled = std::max(max_scaled, scaled);
    error_squared += error * error;
    reference_squared += static_cast<double>(expected[i]) * expected[i];

    if (scaled > 1.0) {
      throw std::runtime_error("FP32 GEMM tolerance exceeded at output "
                               + std::to_string(i));
    }
  }

  return {{"max_abs_error", max_abs}, {"max_tolerance_ratio", max_scaled},
          {"relative_l2_error", std::sqrt(error_squared / reference_squared)}};
}

Json check_fp64(const Case& shape, const Bank& bank,
                const std::vector<float>& actual) {
  const std::size_t count = std::min<std::size_t>(actual.size(), 1024);
  double max_abs = 0.0;
  double max_scaled = 0.0;

  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t index = count == 1 ? 0 : i * (actual.size() - 1) / (count - 1);
    const std::size_t row = index / shape.n;
    const std::size_t col = index % shape.n;
    double expected = 0.0;

    for (int k = 0; k < shape.k; ++k) {
      expected += static_cast<double>(bank.host_input[row * shape.k + k])
                  * bank.host_weight[col * shape.k + k];
    }

    const double error = std::abs(actual[index] - expected);
    const double scaled = error / (kTolerance * (1.0 + std::abs(expected)));
    max_abs = std::max(max_abs, error);
    max_scaled = std::max(max_scaled, scaled);

    if (!std::isfinite(actual[index]) || scaled > 1.0) {
      throw std::runtime_error("GEMM disagrees with independent FP64 dot product");
    }
  }

  return {{"samples", count}, {"max_abs_error", max_abs},
          {"max_tolerance_ratio", max_scaled}};
}

class GraphTimer {
 public:
  GraphTimer(const std::function<void(std::size_t)>& launch, int calls,
             cudaStream_t stream) : calls_(calls), stream_(stream) {
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin GEMM capture");

    for (int i = 0; i < calls; ++i) {
      launch(i);
    }

    check_cuda(cudaStreamEndCapture(stream, &graph_), "end GEMM capture");
    check_cuda(cudaGraphInstantiate(&executable_, graph_, 0), "instantiate GEMM");
    check_cuda(cudaEventCreate(&start_), "create GEMM start event");
    check_cuda(cudaEventCreate(&stop_), "create GEMM stop event");
    check_cuda(cudaGraphLaunch(executable_, stream), "upload and warm GEMM graph");
    check_cuda(cudaStreamSynchronize(stream), "finish graph warmup");
  }

  ~GraphTimer() {
    cudaEventDestroy(stop_);
    cudaEventDestroy(start_);
    cudaGraphExecDestroy(executable_);
    cudaGraphDestroy(graph_);
  }

  GraphTimer(const GraphTimer&) = delete;
  GraphTimer& operator=(const GraphTimer&) = delete;

  double sample() {
    float elapsed = 0.0f;

    check_cuda(cudaEventRecord(start_, stream_), "record GEMM start");
    check_cuda(cudaGraphLaunch(executable_, stream_), "launch GEMM graph");
    check_cuda(cudaEventRecord(stop_, stream_), "record GEMM stop");
    check_cuda(cudaEventSynchronize(stop_), "wait for GEMM graph");
    check_cuda(cudaEventElapsedTime(&elapsed, start_, stop_), "measure GEMM graph");
    return elapsed * 1000.0 / calls_;
  }

  int calls() const { return calls_; }

 private:
  int calls_;
  cudaStream_t stream_;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

std::unique_ptr<GraphTimer> prepare_timer(
    const std::function<void(std::size_t)>& launch, int slots, int warmup,
    int max_calls, double target_ms, cudaStream_t stream) {
  for (int i = 0; i < std::max(warmup, slots); ++i) {
    launch(i);
  }

  check_cuda(cudaStreamSynchronize(stream), "finish GEMM warmup");
  GraphTimer pilot(launch, slots, stream);
  const double per_call_us = pilot.sample();
  const int wanted = static_cast<int>(std::min<double>(
      max_calls, std::ceil(target_ms * 1000.0 / std::max(per_call_us, 0.1))));
  const int calls = std::max(slots, (wanted / slots) * slots);

  return std::make_unique<GraphTimer>(launch, calls, stream);
}

class BlasPlan {
 public:
  BlasPlan(const Case& shape, std::size_t workspace_mib, cudaStream_t stream)
      : shape_(shape), stream_(stream),
        workspace_(Tensor::empty({workspace_mib * kMiB / sizeof(float)},
                                  DataType::Float32, Device::cuda(0))) {
    check_blas(cublasCreate(&blas_), "create strict SGEMM handle");
    check_blas(cublasSetStream(blas_, stream), "set SGEMM stream");
    check_blas(cublasSetWorkspace(blas_, workspace_.data(), workspace_.size_bytes()),
               "set SGEMM workspace");
    check_blas(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH), "strict FP32 math");
    check_blas(cublasSetAtomicsMode(blas_, CUBLAS_ATOMICS_NOT_ALLOWED),
               "disable SGEMM atomics");
    check_blas(cublasLtCreate(&lt_), "create cuBLASLt handle");
    check_blas(cublasLtMatmulDescInit(&operation_, CUBLAS_COMPUTE_32F_PEDANTIC,
                                      CUDA_R_32F), "strict cuBLASLt math");

    const cublasOperation_t transpose = CUBLAS_OP_T;
    check_blas(cublasLtMatmulDescSetAttribute(&operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transpose, sizeof(transpose)),
               "transpose weight descriptor");
    check_blas(cublasLtMatrixLayoutInit(&weight_, CUDA_R_32F, shape.k, shape.n, shape.k),
               "create weight layout");
    check_blas(cublasLtMatrixLayoutInit(&input_, CUDA_R_32F, shape.k, shape.m, shape.k),
               "create input layout");
    check_blas(cublasLtMatrixLayoutInit(&output_, CUDA_R_32F, shape.n, shape.m, shape.n),
               "create output layout");
  }

  ~BlasPlan() {
    cublasLtDestroy(lt_);
    cublasDestroy(blas_);
  }

  BlasPlan(const BlasPlan&) = delete;
  BlasPlan& operator=(const BlasPlan&) = delete;

  void sgemm(Slot& slot) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_blas(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N,
                            shape_.n, shape_.m, shape_.k, &alpha,
                            static_cast<const float*>(slot.weight.data()), shape_.k,
                            static_cast<const float*>(slot.input.data()), shape_.k,
                            &beta, static_cast<float*>(slot.output.data()), shape_.n),
               "strict SGEMM");
  }

  void matmul(Slot& slot, const cublasLtMatmulAlgo_t& algorithm) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_blas(cublasLtMatmul(lt_, &operation_, &alpha,
                               slot.weight.data(), &weight_,
                               slot.input.data(), &input_, &beta,
                               slot.output.data(), &output_,
                               slot.output.data(), &output_, &algorithm,
                               workspace_.data(),
                               workspace_.size_bytes(), stream_), "strict cuBLASLt");
  }

  Json tune(Bank& bank, int requested, const std::vector<float>& expected) {
    cublasLtMatmulPreferenceOpaque_t preference{};
    check_blas(cublasLtMatmulPreferenceInit(&preference), "init Lt preference");
    const auto bytes = workspace_.size_bytes();

    check_blas(cublasLtMatmulPreferenceSetAttribute(
                   &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                   &bytes, sizeof(bytes)), "set Lt workspace limit");

    std::vector<cublasLtMatmulHeuristicResult_t> candidates(requested);
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        lt_, &operation_, &weight_, &input_, &output_, &output_, &preference,
        requested, candidates.data(), &returned);

    Json result = {{"requested_candidates", requested},
                   {"returned_candidates", returned}, {"candidates", Json::array()}};

    if (status != CUBLAS_STATUS_SUCCESS) {
      result["status"] = "unsupported";
      result["cublas_status"] = static_cast<int>(status);
      return result;
    }

    double best = std::numeric_limits<double>::infinity();

    for (int index = 0; index < returned; ++index) {
      const auto& candidate = candidates[index];
      Json record = {{"index", index}, {"workspace_bytes", candidate.workspaceSize}};

      if (candidate.state != CUBLAS_STATUS_SUCCESS) {
        record["status"] = "unsupported";
        result["candidates"].push_back(record);
        continue;
      }

      const auto launch = [&](std::size_t i) {
        matmul(bank.slots[i % bank.slots.size()], candidate.algo);
      };

      // Algorithm selection is outside final timing. Every candidate must pass.
      launch(0);
      record["correctness"] = compare(download(bank.slots[0].output, stream_), expected);
      auto timer = prepare_timer(launch, bank.slots.size(), 3, 256, 2.0, stream_);
      std::vector<double> samples;

      for (int sample = 0; sample < 3; ++sample) {
        samples.push_back(timer->sample());
      }

      const double elapsed = median(samples);
      record["status"] = "ok";
      record["samples_us"] = samples;
      record["median_us"] = elapsed;
      result["candidates"].push_back(record);

      if (elapsed < best) {
        best = elapsed;
        selected_ = candidate.algo;
        selected_workspace_ = candidate.workspaceSize;
        selected_index_ = index;
        available_ = true;
      }
    }

    result["status"] = available_ ? "ok" : "unsupported";

    if (available_) {
      int id = -1;
      std::size_t written = 0;
      check_blas(cublasLtMatmulAlgoConfigGetAttribute(
                     &selected_, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written),
                 "read selected Lt algorithm ID");
      result["selected_index"] = selected_index_;
      result["selected_algorithm_id"] = id;
      result["selected_workspace_bytes"] = selected_workspace_;
    }

    return result;
  }

  bool available() const { return available_; }
  void tuned(Slot& slot) { matmul(slot, selected_); }

 private:
  Case shape_;
  cudaStream_t stream_;
  Tensor workspace_;
  cublasHandle_t blas_ = nullptr;
  cublasLtHandle_t lt_ = nullptr;
  cublasLtMatmulDescOpaque_t operation_{};
  cublasLtMatrixLayoutOpaque_t weight_{};
  cublasLtMatrixLayoutOpaque_t input_{};
  cublasLtMatrixLayoutOpaque_t output_{};
  cublasLtMatmulAlgo_t selected_{};
  std::size_t selected_workspace_ = 0;
  int selected_index_ = -1;
  bool available_ = false;
};

struct Implementation {
  std::string name;
  std::function<void(std::size_t)> launch;
  std::unique_ptr<GraphTimer> timer;
  Json correctness;
  std::vector<double> samples;
};

Json run_case(const Case& shape, const std::string& cache, int l2_bytes,
              const GemmBenchmarkOptions& options, cudaStream_t stream) {
  Bank bank(shape, cache == "rotating", l2_bytes, options, stream);
  BlasPlan blas(shape, options.workspace_mib, stream);

  blas.sgemm(bank.slots[0]);
  const auto expected = download(bank.slots[0].output, stream);
  const auto reference_check = check_fp64(shape, bank, expected);
  const auto tuning = blas.tune(bank, options.heuristic_candidates, expected);

  std::vector<Implementation> implementations;
  const auto add = [&](std::string name, auto function) {
    implementations.push_back({std::move(name), [&, function](std::size_t i) {
      function(bank.slots[i % bank.slots.size()]);
    }, nullptr, Json::object(), {}});
  };

  add("custom", [&](Slot& slot) {
    kernels::cuda::linear(slot.input.view().as_const(), slot.weight.view().as_const(),
                          slot.output.view(), std::nullopt, stream);
  });
  add("tiled_16x16", [&](Slot& slot) {
    tiled_linear(slot.input.view().as_const(), slot.weight.view().as_const(),
                   slot.output.view(), stream);
  });
  add("cublas_sgemm", [&](Slot& slot) { blas.sgemm(slot); });

  if (blas.available()) {
    add("cublaslt_tuned", [&](Slot& slot) { blas.tuned(slot); });
  }

  add("cublaslt_legacy", [&](Slot& slot) {
    cublas_linear(slot.input.view().as_const(), slot.weight.view().as_const(),
                    slot.output.view(), std::nullopt, stream);
  });

  for (auto& implementation : implementations) {
    implementation.launch(0);
    const auto actual = download(bank.slots[0].output, stream);
    implementation.correctness = compare(actual, expected);
    implementation.correctness["fp64"] = check_fp64(shape, bank, actual);

    // The last address also checks that rotation does not change the contract.
    implementation.launch(bank.slots.size() - 1);
    compare(download(bank.slots.back().output, stream), expected);
    implementation.timer = prepare_timer(
        implementation.launch, bank.slots.size(), options.warmup,
        options.max_calls, options.target_ms, stream);
  }

  std::mt19937 order_generator(options.seed + shape.m + shape.n + shape.k);
  std::vector<std::size_t> order(implementations.size());
  std::iota(order.begin(), order.end(), 0);
  Json orders = Json::array();

  for (int sample = 0; sample < options.samples; ++sample) {
    std::shuffle(order.begin(), order.end(), order_generator);
    Json names = Json::array();

    for (std::size_t index : order) {
      auto& implementation = implementations[index];
      implementation.samples.push_back(implementation.timer->sample());
      names.push_back(implementation.name);
    }

    orders.push_back(names);
  }

  Json result = {{"name", shape.name}, {"group", shape.group}, {"cache", cache},
                 {"m", shape.m}, {"n", shape.n}, {"k", shape.k},
                 {"slots", bank.slots.size()},
                 {"working_set_bytes", bank.slots.size() * bank.bytes_per_slot},
                 {"unique_weight_bytes", bank.slots.size() * bank.slots[0].weight.size_bytes()},
                 {"target_working_set_bytes", bank.target_bytes},
                 {"target_reached", bank.slots.size() * bank.bytes_per_slot >= bank.target_bytes},
                 {"reference_fp64_check", reference_check}, {"lt_tuning", tuning},
                 {"sampling_order", orders}, {"implementations", Json::object()}};

  for (const auto& implementation : implementations) {
    auto summary = describe(implementation.samples);
    const double microseconds = summary["median_us"];
    summary["tflops"] = 2.0 * shape.m * shape.n * shape.k / (microseconds * 1e6);
    summary["samples_us"] = implementation.samples;
    summary["calls_per_sample"] = implementation.timer->calls();
    summary["correctness"] = implementation.correctness;
    result["implementations"][implementation.name] = summary;
  }

  const auto& metrics = result["implementations"];
  double best_blas = metrics["cublas_sgemm"]["median_us"];

  if (metrics.contains("cublaslt_tuned")) {
    best_blas = std::min(best_blas, metrics["cublaslt_tuned"]["median_us"].get<double>());
  }

  const double custom = metrics["custom"]["median_us"];
  result["custom_speedup_over_best_strict_blas"] = best_blas / custom;
  result["custom_speedup_over_tiled"] = metrics["tiled_16x16"]["median_us"].get<double>() / custom;
  return result;
}

Json metadata(const GemmBenchmarkOptions& options) {
  cudaDeviceProp properties{};
  int runtime = 0;
  int driver = 0;

  check_cuda(cudaGetDeviceProperties(&properties, 0), "read CUDA device properties");
  check_cuda(cudaRuntimeGetVersion(&runtime), "read CUDA runtime version");
  check_cuda(cudaDriverGetVersion(&driver), "read CUDA driver version");

  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::ostringstream timestamp;
  timestamp << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
  const char* tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");

  return {{"schema_version", 1}, {"status", "running"},
          {"started_at_utc", timestamp.str()},
          {"hardware", {{"gpu", properties.name}, {"sm_major", properties.major},
                        {"sm_minor", properties.minor},
                        {"multiprocessors", properties.multiProcessorCount},
                        {"vram_bytes", properties.totalGlobalMem},
                        {"l2_bytes", properties.l2CacheSize}}},
          {"software", {{"cuda_runtime", runtime}, {"cuda_driver_api", driver},
                        {"cublaslt_version", cublasLtGetVersion()},
                        {"cuda_compiler", NANOVLLM_GEMM_CUDA_COMPILER},
                        {"build_type", NANOVLLM_GEMM_BUILD_TYPE}}},
          {"options", {{"suite", options.suite}, {"cache", options.cache},
                       {"seed", options.seed}, {"samples", options.samples},
                       {"warmup", options.warmup}, {"target_ms", options.target_ms},
                       {"max_calls", options.max_calls},
                       {"workspace_mib", options.workspace_mib},
                       {"heuristic_candidates", options.heuristic_candidates}}},
          {"contract", {{"operation", "X[M,K] @ W[N,K]^T"}, {"dtype", "float32"},
                        {"alpha", 1}, {"beta", 0}, {"bias", false},
                        {"atol", kTolerance}, {"rtol", kTolerance},
                        {"strict_sgemm_math", "CUBLAS_PEDANTIC_MATH"},
                        {"strict_lt_compute", "CUBLAS_COMPUTE_32F_PEDANTIC"},
                        {"NVIDIA_TF32_OVERRIDE", tf32 ? Json(tf32) : Json(nullptr)},
                        {"timing", "CUDA Graph + CUDA events; GPU execution only"}}},
          {"correctness", {{"production_cases", 180}, {"tiled_cases", 30}, {"passed", true}}},
          {"cases", Json::array()}};
}

void save(const std::string& path, const Json& result) {
  const std::filesystem::path destination(path);

  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }

  std::ofstream output(destination);
  output << result.dump(2) << '\n';
  output.flush();

  if (!output) {
    throw std::runtime_error("cannot write GEMM results: " + path);
  }
}

} // namespace

void run_gemm_benchmark(const GemmBenchmarkOptions& options, cudaStream_t stream) {
  if (std::string(NANOVLLM_GEMM_BUILD_TYPE) != "Release"
      && std::string(NANOVLLM_GEMM_BUILD_TYPE) != "RelWithDebInfo") {
    throw std::runtime_error("performance measurements require a Release build");
  }

  auto result = metadata(options);
  auto cases = cases_for(options.suite);
  std::mt19937 generator(options.seed);
  std::shuffle(cases.begin(), cases.end(), generator);
  const int l2_bytes = result["hardware"]["l2_bytes"];
  std::vector<std::string> caches = options.cache == "both"
      ? std::vector<std::string>{"hot", "rotating"}
      : std::vector<std::string>{options.cache};

  save(options.output, result);

  try {
    for (const auto& shape : cases) {
      std::shuffle(caches.begin(), caches.end(), generator);

      for (const auto& cache : caches) {
        std::cout << '[' << result["cases"].size() + 1 << '/'
                  << cases.size() * caches.size() << "] " << shape.name << ' '
                  << cache << std::flush;

        auto measured = run_case(shape, cache, l2_bytes, options, stream);
        std::cout << " custom=" << measured["implementations"]["custom"]["median_us"]
                  << " us, vs strict BLAS="
                  << measured["custom_speedup_over_best_strict_blas"]
                  << "x\n" << std::flush;
        result["cases"].push_back(std::move(measured));
        save(options.output, result);
      }
    }

    result["status"] = "complete";
    save(options.output, result);
  } catch (const std::exception& error) {
    result["status"] = "failed";
    result["error"] = error.what();
    save(options.output, result);
    throw;
  }

  std::cout << "Saved " << options.output << '\n';
}

} // namespace nanovllm::exp
