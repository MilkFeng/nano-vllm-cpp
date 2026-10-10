#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/module.hh"
#include "kernels/cpu/expert.hh"
#include "layers/linear.hh"
#include "layers/swiglu_mlp.hh"

namespace nanovllm {

// Owns all expert parameters in ordinary system RAM. Registered directly as
// experts.0.gate_proj.weight, etc. GPU cache copies are separate runtime data.
class ExpertStore final : public Module {
 public:
  ExpertStore(std::size_t hidden_size, std::size_t intermediate_size,
              std::size_t num_experts);

  [[nodiscard]] kernels::cpu::ExpertWeights weights(
      std::size_t expert_id) const;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t intermediate_size() const noexcept;
  [[nodiscard]] std::size_t num_experts() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;

 private:
  std::size_t hidden_size_;
  std::size_t intermediate_size_;
  std::vector<std::unique_ptr<SwiGLUMLP>> experts_;
};

// Fixed resident list on CUDA device 0. Allocates only resident_ids.size()
// expert copies. The RAM source must outlive this cache. Calls are serialized.
class ExpertCache final {
 public:
  ExpertCache(const ExpertStore& source,
              std::vector<std::int32_t> resident_ids);

  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;
  ExpertCache(ExpertCache&&) = delete;
  ExpertCache& operator=(ExpertCache&&) = delete;

  // Call after initializing RAM weights; call again after changing them.
  // Publishes the mapping after all uploads complete. Synchronizes stream.
  void populate(cudaStream_t stream = nullptr);
  [[nodiscard]] int slot_of(std::size_t expert_id) const;
  [[nodiscard]] const SwiGLUMLP& expert_at_slot(std::size_t slot) const;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::size_t num_slots() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] std::span<const std::int32_t> resident_ids() const noexcept;
  [[nodiscard]] const ExpertStore& source() const noexcept;

 private:
  const ExpertStore& source_;
  std::vector<std::int32_t> resident_ids_;
  std::vector<int> expert_to_slot_;
  std::vector<std::unique_ptr<SwiGLUMLP>> slots_;
  bool ready_ = false;
};

struct ExpertExecutionOutput {
  Tensor expert_outputs; // CUDA Float32 [K,H], unweighted, in router order.
  std::size_t gpu_experts = 0;
  std::size_t cpu_experts = 0;
};

// Synchronous dispatcher for one token. Store/cache must outlive the executor.
// Cache hits use GPU slots; misses execute from RAM and upload only outputs.
class ExpertExecutor final {
 public:
  ExpertExecutor(const ExpertStore& store, const ExpertCache& cache);

  [[nodiscard]] ExpertExecutionOutput forward(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      cudaStream_t stream = nullptr) const;

 private:
  const ExpertStore& store_;
  const ExpertCache& cache_;
};

struct MoEConfig {
  std::size_t hidden_size;
  std::size_t intermediate_size;
  std::size_t num_experts;
  std::size_t top_k;

  void validate() const;
};

// All tensor outputs live on CUDA. Expert rows follow router rank, not ID or
// cache slot. Weights are applied once when combining the unweighted rows.
struct MoEOutput {
  Tensor hidden_states;  // Float32 [1,H].
  Tensor router_logits;  // Float32 [1,E].
  Tensor expert_ids;     // Int32 [1,K].
  Tensor expert_weights; // Float32 [1,K], sum to 1.
  Tensor expert_outputs; // Float32 [K,H].
  std::size_t gpu_experts = 0;
  std::size_t cpu_experts = 0;
};

// Single-token FP32 routed MoE. All expert parameters are in RAM; only the
// fixed resident list is copied to GPU. nullopt caches all experts; an empty
// vector caches none. CPU construction creates reference weights without CUDA
// allocations. Initialize parameters, then populate_expert_cache(), then
// forward(). Calls are serialized; populate/forward synchronize the supplied
// CUDA stream.
class MoE final : public Module {
 public:
  explicit MoE(
      MoEConfig config, Device device = Device::cuda(0),
      std::optional<std::vector<std::int32_t>> gpu_expert_ids = std::nullopt);

  void populate_expert_cache(cudaStream_t stream = nullptr);
  [[nodiscard]] MoEOutput forward(ConstTensorView hidden_states,
                                  cudaStream_t stream = nullptr) const;
  [[nodiscard]] const MoEConfig& config() const noexcept;
  [[nodiscard]] const ExpertStore& expert_store() const noexcept;
  [[nodiscard]] const ExpertCache& expert_cache() const;

 private:
  MoEConfig config_;
  Linear gate_;
  ExpertStore experts_;
  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<ExpertExecutor> executor_;
};

} // namespace nanovllm
