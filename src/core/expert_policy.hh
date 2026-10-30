#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/model_config.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

using ExpertPlacement = std::vector<std::vector<std::int32_t>>;

[[nodiscard]] ExpertPlacement resolve_expert_placement(
    const ModelConfig& config, std::span<const std::int32_t> default_ids);

[[nodiscard]] std::size_t routed_expert_size_bytes(const ModelConfig& config);

struct ExpertUsage {
  std::vector<double> counts;
  std::size_t gpu_assignments = 0;
  std::size_t cpu_assignments = 0;
};

struct ExpertSwap {
  std::size_t layer;
  std::size_t slot;
  std::int32_t old_id;
  std::int32_t new_id;
  double gain;
};

// Host-only policy. Slot ownership stays with each layer; ties are
// deterministic.
class ExpertAdaptationPolicy {
 public:
  ExpertAdaptationPolicy(std::size_t layers, std::size_t experts,
                         ExpertAdaptationParams params = {});

  [[nodiscard]] bool enabled() const noexcept { return params_.every != 0; }
  [[nodiscard]] ExpertUsage* usage(std::size_t layer);
  [[nodiscard]] bool advance() noexcept;
  [[nodiscard]] std::vector<ExpertSwap> plan(
      const ExpertPlacement& placement) const;
  void decay() noexcept;
  void reset_assignment_stats() noexcept;
  void add_assignment_stats(ExpertAdaptationStats& stats) const noexcept;

 private:
  ExpertAdaptationParams params_;
  std::size_t experts_;
  std::size_t steps_ = 0;
  std::vector<ExpertUsage> usage_;
};

} // namespace nanovllm
