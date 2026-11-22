#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "common/thread_pool.hh"
#include "core/tensor_view.hh"
#include "layers/moe/expert.hh"
#include "layers/moe/expert_cache.hh"
#include "layers/moe/workspace.hh"

namespace nanovllm {

struct ExpertExecutionStats {
  std::size_t gpu_experts = 0;
  std::size_t cpu_experts = 0;
};

class ExpertExecutor final {
 public:
  ExpertExecutor(const ExpertStore& store, const ExpertCache& cache,
                 ThreadPool& cpu_pool);

  ExpertExecutor(const ExpertExecutor&) = delete;
  ExpertExecutor& operator=(const ExpertExecutor&) = delete;
  ExpertExecutor(ExpertExecutor&&) = delete;
  ExpertExecutor& operator=(ExpertExecutor&&) = delete;

  [[nodiscard]] ExpertExecutionStats forward_into(
      ConstTensorView gpu_input, std::span<const std::int32_t> expert_ids,
      std::size_t top_k, TensorView expert_outputs,
      MoEExecutionWorkspace& workspace, cudaStream_t stream) const;

 private:
  [[nodiscard]] ExpertExecutionStats forward_single_into(
      ConstTensorView gpu_input, std::span<const std::int32_t> expert_ids,
      TensorView expert_outputs, MoEExecutionWorkspace& workspace,
      cudaStream_t stream) const;

  const ExpertStore& store_;
  const ExpertCache& cache_;

  ThreadPool& cpu_pool_;
};

namespace detail {

struct CpuExpertJob {
  const FP8Expert* expert;
  ConstTensorView input;
  TensorView output;

  static void run(void* argument);
};

} // namespace detail

} // namespace nanovllm
