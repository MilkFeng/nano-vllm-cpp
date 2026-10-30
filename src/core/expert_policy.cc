#include "core/expert_policy.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace nanovllm {

ExpertPlacement resolve_expert_placement(
    const ModelConfig& config, std::span<const std::int32_t> default_ids) {
  return ExpertPlacement(
      config.num_hidden_layers,
      std::vector<std::int32_t>(default_ids.begin(), default_ids.end()));
}

std::size_t routed_expert_size_bytes(const ModelConfig& config) {
  const auto h = config.hidden_size;
  const auto i = config.moe_intermediate_size;
  return 3 * h * i + 3 * ((h + 127) / 128) * ((i + 127) / 128) * sizeof(float);
}

ExpertAdaptationPolicy::ExpertAdaptationPolicy(std::size_t layers,
                                               std::size_t experts,
                                               ExpertAdaptationParams params)
    : params_(params),
      experts_(experts) {
  if (params.max_swaps == 0 || !std::isfinite(params.decay) || params.decay < 0
      || params.decay > 1 || !std::isfinite(params.min_usage)
      || params.min_usage < 0 || !std::isfinite(params.min_gain)
      || params.min_gain <= 0) {
    throw std::invalid_argument(
        "expert adaptation needs positive max_swaps, decay in [0,1], "
        "nonnegative min_usage and positive min_gain (all finite)");
  }

  if (enabled()) {
    usage_.assign(layers, ExpertUsage{std::vector<double>(experts)});
  }
}

ExpertUsage* ExpertAdaptationPolicy::usage(std::size_t layer) {
  return enabled() ? &usage_.at(layer) : nullptr;
}

bool ExpertAdaptationPolicy::advance() noexcept {
  if (!enabled()) {
    return false;
  }
  if (++steps_ < params_.every) {
    return false;
  }
  steps_ = 0;
  return true;
}

std::vector<ExpertSwap> ExpertAdaptationPolicy::plan(
    const ExpertPlacement& placement) const {
  if (!enabled()) {
    return {};
  }
  if (placement.size() != usage_.size()) {
    throw std::invalid_argument("adaptive placement must contain every layer");
  }
  std::vector<ExpertSwap> swaps;
  for (std::size_t layer = 0; layer < placement.size(); ++layer) {
    const auto& ids = placement[layer];
    const auto& counts = usage_[layer].counts;
    std::vector<bool> resident(experts_, false);
    std::vector<std::pair<double, std::size_t>> victims;
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
      const auto id = ids[slot];
      if (id < 0 || std::size_t(id) >= experts_ || resident[id]) {
        throw std::invalid_argument("invalid adaptive resident expert ID");
      }
      resident[id] = true;
      victims.emplace_back(counts[id], slot);
    }
    std::vector<std::pair<double, std::int32_t>> candidates;
    for (std::size_t id = 0; id < experts_; ++id) {
      if (!resident[id] && counts[id] >= params_.min_usage) {
        candidates.emplace_back(counts[id], static_cast<std::int32_t>(id));
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::sort(victims.begin(), victims.end());
    const auto pairs = std::min(candidates.size(), victims.size());
    for (std::size_t i = 0; i < pairs; ++i) {
      const auto gain = candidates[i].first - victims[i].first;
      if (gain < params_.min_gain) {
        break;
      }
      const auto slot = victims[i].second;
      swaps.push_back({layer, slot, ids[slot], candidates[i].second, gain});
    }
  }
  std::sort(swaps.begin(), swaps.end(), [](const auto& a, const auto& b) {
    if (a.gain != b.gain) {
      return a.gain > b.gain;
    }
    if (a.layer != b.layer) {
      return a.layer < b.layer;
    }
    if (a.new_id != b.new_id) {
      return a.new_id < b.new_id;
    }
    return a.slot < b.slot;
  });
  if (swaps.size() > params_.max_swaps) {
    swaps.resize(params_.max_swaps);
  }
  return swaps;
}

void ExpertAdaptationPolicy::decay() noexcept {
  for (auto& layer : usage_) {
    for (auto& count : layer.counts) {
      count *= params_.decay;
    }
  }
}

void ExpertAdaptationPolicy::reset_assignment_stats() noexcept {
  for (auto& layer : usage_) {
    layer.gpu_assignments = 0;
    layer.cpu_assignments = 0;
  }
}

void ExpertAdaptationPolicy::add_assignment_stats(
    ExpertAdaptationStats& stats) const noexcept {
  for (const auto& layer : usage_) {
    stats.gpu_assignments += layer.gpu_assignments;
    stats.cpu_assignments += layer.cpu_assignments;
  }
}

} // namespace nanovllm
