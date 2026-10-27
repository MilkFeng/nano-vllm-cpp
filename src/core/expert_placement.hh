#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "core/model_config.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

using ExpertPlacement = std::vector<std::vector<std::int32_t>>;

[[nodiscard]] ExpertPlacement resolve_expert_placement(
    const ModelConfig& config, std::span<const std::int32_t> default_ids,
    const ExpertPlacement& layers);

[[nodiscard]] ExpertPlacement read_expert_placement(
    const std::filesystem::path& path, const ModelConfig& config);

[[nodiscard]] std::size_t routed_expert_size_bytes(const ModelConfig& config);

[[nodiscard]] EngineParams resolve_engine_params(const ModelConfig& config,
                                                 EngineParams params);

} // namespace nanovllm
