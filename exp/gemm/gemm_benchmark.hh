#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

#include "core/tensor_view.hh"

namespace nanovllm::exp {

struct GemmBenchmarkOptions {
  std::string suite = "quick";
  std::string cache = "both";
  std::string output = "gemm-results.json";
  unsigned seed = 42;
  int samples = 15;
  int warmup = 20;
  int max_calls = 1024;
  int heuristic_candidates = 8;
  double target_ms = 10.0;
  std::size_t workspace_mib = 32;
};

// Simple 16x16 shared-memory tiling, with one FP32 accumulator per thread.
void tiled_linear(ConstTensorView input, ConstTensorView weight,
                   TensorView output, cudaStream_t stream);

void run_gemm_benchmark(const GemmBenchmarkOptions& options, cudaStream_t stream);

} // namespace nanovllm::exp
