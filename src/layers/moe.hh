#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "common/thread_pool.hh"
#include "core/module.hh"
#include "layers/linear.hh"
#include "layers/swiglu_mlp.hh"

namespace nanovllm {

struct FP8ExpertWorkspace {
  Tensor gate;
  Tensor up;
  Tensor activated;

  FP8ExpertWorkspace() = default;
  explicit FP8ExpertWorkspace(std::size_t intermediate_size);
};

class FP8Expert final : public Module {
 public:
  FP8Expert(std::size_t hidden_size, std::size_t intermediate_size,
            Device device = Device::cpu());

  void forward_into(ConstTensorView input, TensorView output) const;

  void forward_into(ConstTensorView input, TensorView output,
                    FP8ExpertWorkspace& workspace,
                    cudaStream_t stream = nullptr) const;

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

  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;
  ExpertCache(ExpertCache&&) = delete;
  ExpertCache& operator=(ExpertCache&&) = delete;

  void populate(cudaStream_t stream = nullptr);
  void invalidate() noexcept;

  [[nodiscard]] int slot_of(std::size_t expert_id) const;
  [[nodiscard]] const FP8Expert& expert_at_slot(std::size_t slot) const;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::size_t num_slots() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] std::span<const std::int32_t> resident_ids() const noexcept;
  [[nodiscard]] const ExpertStore& source() const noexcept;

 private:
  const ExpertStore& source_;
  std::vector<std::int32_t> resident_ids_;
  std::vector<int> expert_to_slot_;
  std::vector<std::unique_ptr<FP8Expert>> slots_;
  bool ready_ = false;
};

struct ExpertExecutionOutput {
  Tensor expert_outputs;
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

  [[nodiscard]] ExpertExecutionOutput forward(
      ConstTensorView input, std::span<const std::int32_t> expert_ids,
      cudaStream_t stream = nullptr) const;

 private:
  const ExpertStore& store_;
  const ExpertCache& cache_;
  ThreadPool& cpu_pool_;
  mutable FP8ExpertWorkspace workspace_;
  mutable Tensor host_input_;
  mutable Tensor host_parts_;
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
  explicit MoE(
      MoEConfig config, ThreadPool& cpu_pool, Device device = Device::cuda(0),
      std::optional<std::vector<std::int32_t>> gpu_expert_ids = std::nullopt);

  void populate_expert_cache(cudaStream_t stream = nullptr);
  void invalidate_expert_cache() noexcept;

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
