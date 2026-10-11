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

enum class ExpertExecutionMode {
  Synchronous, // Original router-order dispatcher, used as the comparison path.
  Overlap, // Submit every GPU hit, then compute CPU misses on the host thread.
};

// One-token dispatcher. Store/cache must outlive it. Calls must be serialized.
// Both modes synchronize stream before returning. Overlap reuses private
// scratch. Input shape/device/dtype and selection size use Debug assertions.
class ExpertExecutor final {
 public:
  ExpertExecutor(const ExpertStore& store, const ExpertCache& cache,
                 ExpertExecutionMode mode = ExpertExecutionMode::Synchronous);

  ExpertExecutor(const ExpertExecutor&) = delete;
  ExpertExecutor& operator=(const ExpertExecutor&) = delete;
  ExpertExecutor(ExpertExecutor&&) = delete;
  ExpertExecutor& operator=(ExpertExecutor&&) = delete;

  [[nodiscard]] ExpertExecutionOutput forward(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      cudaStream_t stream = nullptr) const;

 private:
  [[nodiscard]] ExpertExecutionOutput forward_synchronous(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      cudaStream_t stream) const;
  [[nodiscard]] ExpertExecutionOutput forward_overlap(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      cudaStream_t stream) const;

  const ExpertStore& store_;
  const ExpertCache& cache_;
  ExpertExecutionMode mode_;
  mutable SwiGLUWorkspace gpu_workspace_;
  mutable Tensor host_input_;
  mutable Tensor host_parts_;
};

struct MoEConfig {
  std::size_t hidden_size;
  std::size_t intermediate_size;
  std::size_t num_experts;
  std::size_t top_k;
  // 0 disables the shared branch.
  std::size_t shared_expert_intermediate_size = 0;

  void validate() const;
};

// Nonempty outputs live on CUDA, in token order. Expert row t*K+rank belongs
// to token t and its router rank, not expert ID/cache slot. Weights are applied
// once in combine. T=1 keeps the original [K,H] expert output layout.
struct MoEOutput {
  Tensor hidden_states;  // Float32 [T,H].
  Tensor router_logits;  // Float32 [T,E].
  Tensor expert_ids;     // Int32 [T,K].
  Tensor expert_weights; // Float32 [T,K], each row sums to 1.
  Tensor expert_outputs; // Float32 [T*K,H], unweighted.
  // Counts cover all T*K routed selections, excluding the shared branch.
  std::size_t gpu_experts = 0;
  std::size_t cpu_experts = 0;
  // Shared intermediates are empty when the branch is disabled.
  Tensor shared_expert_output; // Raw Float32 [T,H].
  Tensor shared_gate_logits;   // Float32 [T,1], before sigmoid.
};

// FP32 MoE for [T,H], T>0, with an optional Qwen3.5-style shared branch. Routed
// expert parameters are in RAM; only the fixed resident list is copied to GPU.
// The shared expert and its H->1 sigmoid gate stay on the selected device,
// outside the routed cache. nullopt caches all routed experts; an empty
// vector caches none. CPU construction creates reference weights without CUDA
// allocations. Initialize parameters, then populate_expert_cache(), then
// forward(). Input shape/device/dtype use Debug assertions.
// Router and shared branch process T tokens together. Routed experts are
// dispatched one token at a time with the single-token ExpertExecutor.
// Calls are serialized; populate/forward synchronize the supplied CUDA stream.
class MoE final : public Module {
 public:
  explicit MoE(
      MoEConfig config, Device device = Device::cuda(0),
      std::optional<std::vector<std::int32_t>> gpu_expert_ids = std::nullopt,
      ExpertExecutionMode mode = ExpertExecutionMode::Synchronous);

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
  std::unique_ptr<SwiGLUMLP> shared_expert_;
  std::unique_ptr<Linear> shared_expert_gate_;
  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<ExpertExecutor> executor_;
};

} // namespace nanovllm
