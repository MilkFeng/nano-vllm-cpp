#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "common/pinned_buffer.hh"
#include "core/tensor.hh"
#include "layers/moe/expert.hh"

namespace nanovllm {

struct ExpertCacheReplacement {
  std::size_t slot;
  std::int32_t expert_id;
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

  [[nodiscard]] std::size_t replace(
      std::span<const ExpertCacheReplacement> replacements,
      cudaStream_t stream);

  void invalidate() noexcept;

  [[nodiscard]] int slot_of(std::size_t expert_id) const;
  [[nodiscard]] const FP8Expert& expert_at_slot(std::size_t slot) const;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool all_experts_resident() const noexcept;

  [[nodiscard]] std::size_t num_slots() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] std::span<const std::int32_t> resident_ids() const noexcept;
  [[nodiscard]] const ExpertStore& source() const noexcept;

  [[nodiscard]] ConstTensorView device_routes() const;

 private:
  void publish_routes(cudaStream_t stream);

 private:
  const ExpertStore& source_;

  std::vector<std::int32_t> resident_ids_;
  std::vector<int> expert_to_slot_;

  std::vector<std::unique_ptr<FP8Expert>> slots_;

  Tensor routes_;
  PinnedHostBuffer host_routes_;

  cudaEvent_t routes_ready_ = nullptr;
  bool ready_ = false;
};

} // namespace nanovllm
