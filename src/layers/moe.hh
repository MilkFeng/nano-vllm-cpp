#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "common/pinned_buffer.hh"
#include "common/thread_pool.hh"
#include "core/decode_graph.hh"
#include "core/module.hh"
#include "layers/linear.hh"
#include "layers/swiglu_mlp.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

namespace kernels::cuda {
struct FP8ExpertWeights;
}

struct ExpertUsage;

struct ExpertCacheReplacement {
  std::size_t slot;
  std::int32_t expert_id;
};

struct FP8ExpertWorkspace {
  Tensor gate;
  Tensor up;
  Tensor activated;

  FP8ExpertWorkspace() = default;
  explicit FP8ExpertWorkspace(std::size_t intermediate_size,
                              std::size_t rows = 1);
  void reserve(std::size_t rows, std::size_t intermediate_size);
};

struct MoEExecutionWorkspace {
  MoEExecutionWorkspace() = default;
  ~MoEExecutionWorkspace() noexcept;
  MoEExecutionWorkspace(const MoEExecutionWorkspace&) = delete;
  MoEExecutionWorkspace& operator=(const MoEExecutionWorkspace&) = delete;

  // Complete queued uses before host staging is overwritten or destroyed.
  void synchronize();

  PinnedHostBuffer host_ids;
  PinnedHostBuffer host_order;
  PinnedHostBuffer host_input;
  PinnedHostBuffer host_parts;
  Tensor host_grouped_input;
  Tensor order;
  Tensor grouped_input;
  Tensor grouped_parts;
  FP8ExpertWorkspace expert;

 private:
  friend class MoE;
  friend class ExpertExecutor;
  void prepare(cudaStream_t stream);
  void record_host_ready(cudaStream_t stream);
  void wait_host_ready();
  void record_completion(cudaStream_t stream);

  cudaEvent_t host_ready_ = nullptr;
  cudaEvent_t completion_ = nullptr;
  cudaStream_t pending_stream_ = nullptr;
  bool pending_ = false;
};

class FP8Expert final : public Module {
 public:
  FP8Expert(std::size_t hidden_size, std::size_t intermediate_size,
            Device device = Device::cpu());

  void forward_into(ConstTensorView input, TensorView output) const;

  void forward_into(ConstTensorView input, TensorView output,
                    FP8ExpertWorkspace& workspace,
                    cudaStream_t stream = nullptr) const;
  [[nodiscard]] kernels::cuda::FP8ExpertWeights device_weights() const;

 private:
  Linear gate_, up_, down_;
  Parameter gate_scale_, up_scale_, down_scale_;
};

class ExpertStore final : public Module {
 public:
  ExpertStore(std::size_t hidden_size, std::size_t intermediate_size,
              std::size_t num_experts);

  [[nodiscard]] const FP8Expert& expert_at(std::size_t expert_id) const;

  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t intermediate_size() const noexcept;
  [[nodiscard]] std::size_t num_experts() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;

 private:
  std::size_t hidden_size_;
  std::size_t intermediate_size_;
  std::vector<std::unique_ptr<FP8Expert>> experts_;
};

class ExpertCache final {
 public:
  ExpertCache(const ExpertStore& source,
              std::vector<std::int32_t> resident_ids);
  ~ExpertCache();

  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;
  ExpertCache(ExpertCache&&) = delete;
  ExpertCache& operator=(ExpertCache&&) = delete;

  void populate(cudaStream_t stream = nullptr);
  // Call only between forwards. Default behavior completes uploads before
  // returning. With synchronize=false, subsequent GPU uses must follow on the
  // same stream; immutable RAM weights and routing staging stay alive.
  [[nodiscard]] std::size_t replace(
      std::span<const ExpertCacheReplacement> replacements,
      cudaStream_t stream = nullptr, bool synchronize = true);
  void invalidate() noexcept;

  [[nodiscard]] int slot_of(std::size_t expert_id) const;
  [[nodiscard]] const FP8Expert& expert_at_slot(std::size_t slot) const;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::size_t num_slots() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] std::span<const std::int32_t> resident_ids() const noexcept;
  [[nodiscard]] const ExpertStore& source() const noexcept;
  [[nodiscard]] ConstTensorView device_routes() const;

 private:
  const ExpertStore& source_;
  std::vector<std::int32_t> resident_ids_;
  std::vector<int> expert_to_slot_;
  std::vector<std::unique_ptr<FP8Expert>> slots_;
  Tensor routes_;
  PinnedHostBuffer host_routes_;
  cudaEvent_t routes_ready_ = nullptr;
  void publish_routes(cudaStream_t stream);
  bool ready_ = false;
};

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
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      std::size_t top_k, TensorView expert_outputs,
      MoEExecutionWorkspace& workspace, cudaStream_t stream = nullptr) const;

 private:
  friend class MoE;
  // The caller has completed previous workspace uses. A supplied host input
  // has already been downloaded into workspace.host_input.
  [[nodiscard]] ExpertExecutionStats forward_prepared(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      std::size_t top_k, TensorView expert_outputs,
      MoEExecutionWorkspace& workspace, cudaStream_t stream,
      ConstTensorView host_input) const;

  [[nodiscard]] ExpertExecutionStats forward_single_into(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      TensorView expert_outputs, MoEExecutionWorkspace& workspace,
      cudaStream_t stream, ConstTensorView host_input) const;

  const ExpertStore& store_;
  const ExpertCache& cache_;
  ThreadPool& cpu_pool_;
};

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

  // GPU work is asynchronous on stream; profiling waits for accurate wall time.
  // Keep the module and returned tensors alive until queued uses complete.
  [[nodiscard]] MoEOutput forward(ConstTensorView hidden_states,
                                  cudaStream_t stream = nullptr,
                                  MoEPhaseProfile* profile = nullptr,
                                  ExpertUsage* usage = nullptr,
                                  bool decode_graphs = false,
                                  bool defer_cpu = false,
                                  bool graph_warmup = false) const;

  [[nodiscard]] std::size_t replace_experts(
      std::span<const ExpertCacheReplacement> replacements,
      cudaStream_t stream = nullptr, bool synchronize = true) const;

  [[nodiscard]] const MoEConfig& config() const noexcept;

  [[nodiscard]] const ExpertStore& expert_store() const noexcept;
  [[nodiscard]] const ExpertCache& expert_cache() const;
  void check_async_error() const;
  void reset_hybrid() const;
  void complete_hybrid_cpu(cudaStream_t stream) const;
  void abort_hybrid() const noexcept;

 private:
  struct HybridWorkspace;
  [[nodiscard]] MoEOutput forward_hybrid(ConstTensorView input,
                                         cudaStream_t stream,
                                         ExpertUsage* usage, bool defer_cpu,
                                         bool graph_warmup) const;
  MoEConfig config_;
  Linear gate_;
  ExpertStore experts_;
  std::unique_ptr<SwiGLUMLP> shared_expert_;
  std::unique_ptr<Linear> shared_expert_gate_;
  std::unique_ptr<MoEExecutionWorkspace> owned_workspace_;
  MoEExecutionWorkspace* workspace_ = nullptr;
  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<ExpertExecutor> executor_;
  mutable std::unique_ptr<HybridWorkspace> hybrid_;
  mutable DecodeGraph decode_prefix_;
};

} // namespace nanovllm
