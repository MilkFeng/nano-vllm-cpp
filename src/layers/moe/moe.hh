#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "common/pinned_buffer.hh"
#include "common/thread_pool.hh"
#include "core/decode_graph.hh"
#include "core/expert_policy.hh"
#include "core/module.hh"
#include "layers/linear.hh"
#include "layers/moe/expert.hh"
#include "layers/moe/expert_cache.hh"
#include "layers/moe/expert_executor.hh"
#include "layers/moe/workspace.hh"
#include "layers/swiglu_mlp.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

struct MoEConfig {
  std::size_t hidden_size;
  std::size_t intermediate_size;
  std::size_t num_experts;
  std::size_t top_k;
  std::size_t shared_expert_intermediate_size = 0;
};

struct MoEOutput {
  Tensor hidden_states;
  Tensor router_logits;
  Tensor expert_ids;
  Tensor expert_weights;
  Tensor expert_outputs;
  std::size_t gpu_experts = 0;
  std::size_t cpu_experts = 0;
  Tensor shared_expert_output;
  Tensor shared_gate_logits;
};

class MoE final : public Module {
 public:
  ~MoE() override;
  explicit MoE(
      MoEConfig config, ThreadPool& cpu_pool, Device device = Device::cuda(0),
      std::optional<std::vector<std::int32_t>> gpu_expert_ids = std::nullopt,
      MoEExecutionWorkspace* workspace = nullptr);

  void populate_expert_cache(cudaStream_t stream = nullptr);
  void invalidate_expert_cache() noexcept;

  [[nodiscard]] MoEOutput forward(ConstTensorView hidden_states,
                                  cudaStream_t stream = nullptr,
                                  MoEPhaseProfile* profile = nullptr,
                                  ExpertUsage* usage = nullptr,
                                  bool decode_graphs = false,
                                  bool defer_cpu = false,
                                  bool graph_warmup = false) const;

  [[nodiscard]] std::size_t replace_experts(
      std::span<const ExpertCacheReplacement> replacements,
      cudaStream_t stream) const;

  [[nodiscard]] const MoEConfig& config() const noexcept;

  [[nodiscard]] const ExpertStore& expert_store() const noexcept;
  [[nodiscard]] const ExpertCache& expert_cache() const;

  void check_async_error() const;
  void reset_hybrid() const;
  void complete_hybrid_cpu() const;
  void abort_hybrid() const noexcept;

 private:
  struct HybridWorkspace {
    HybridWorkspace(const ExpertStore& s, const ExpertCache& c, ThreadPool& p,
                    std::size_t k, DataType dtype);
    ~HybridWorkspace();

    void reset();
    void abort() noexcept;
    void complete_cpu();
    void check_async_error() const;

    const ExpertStore& store;
    const ExpertCache& cache;

    ThreadPool& pool;

    std::size_t top_k;

    PinnedHostBuffer ids{true};
    PinnedHostBuffer input{true};
    PinnedHostBuffer parts{true};
    PinnedHostBuffer flags{true};

    Tensor activated;

    cudaEvent_t ready = nullptr;
    ExpertUsage* usage = nullptr;
    std::exception_ptr error;

    std::size_t gpu_count = 0, cpu_count = 0;

   private:
    void compute() noexcept;
  };

  [[nodiscard]] MoEOutput forward_hybrid(ConstTensorView input,
                                         cudaStream_t stream,
                                         ExpertUsage* usage, bool defer_cpu,
                                         bool graph_warmup) const;
  MoEConfig config_;
  ThreadPool& cpu_pool_;

  Linear gate_;
  ExpertStore experts_;
  std::unique_ptr<FP8SwiGLUMLP> shared_expert_;
  std::unique_ptr<Linear> shared_expert_gate_;
  std::unique_ptr<MoEExecutionWorkspace> owned_workspace_;

  MoEExecutionWorkspace* workspace_ = nullptr;

  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<ExpertExecutor> executor_;

  mutable std::unique_ptr<HybridWorkspace> hybrid_;
  mutable DecodeGraph decode_prefix_;
};

} // namespace nanovllm
