#pragma once

#include <cstddef>
#include <string_view>

#include "core/checkpoint.hh"
#include "core/model.hh"
#include "core/model_config.hh"
#include "layers/moe.hh"

namespace nanovllm {

// Model-specific construction stays here; core config has no layer dependency.
[[nodiscard]] MoEConfig qwen35_moe_config(const ModelConfig& config);

// Loads registered, non-expert FP32 parameters by prefix + parameter name.
// BF16/F32 sources may have any shape. F8_E4M3 matrices are expanded with the
// sibling weight_scale_inv (BF16/F32, 128x128 blocks). Unregistered checkpoint
// tensors are ignored. All metadata is checked before writing parameters.
// Copies are synchronous; loading must be serialized with forward. Result
// counts unique source tensors/bytes, including scales used for expansion.
[[nodiscard]] ModelLoadResult load_qwen35_module(
    Module& module, const ModelConfig& config, const Checkpoint& checkpoint,
    std::string_view checkpoint_prefix = {});

// Loads model.language_model.layers.L.mlp.*. Routed FP8 bytes stay on CPU;
// router, shared expert and all scales are converted to FP32. Validates before
// invalidating the old cache; populate_expert_cache() after a successful load,
// including zero-slot GPU caches. Copy failures may leave partial parameters,
// but cannot leave an executable stale GPU cache. Serialize with forward/cache
// population. The generic non-expert loader must not be used on a MoE.
[[nodiscard]] ModelLoadResult load_qwen35_moe_layer(
    MoE& layer, const ModelConfig& config, const Checkpoint& checkpoint,
    std::size_t layer_index);

} // namespace nanovllm
