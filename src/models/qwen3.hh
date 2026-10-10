#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "core/forward_context.hh"
#include "core/model.hh"
#include "core/model_config.hh"
#include "layers/embedding.hh"
#include "layers/linear.hh"
#include "layers/lm_head.hh"
#include "layers/rms_norm.hh"
#include "layers/rotary_embedding.hh"
#include "layers/swiglu_mlp.hh"

namespace nanovllm {

// Tensor shapes, dtypes and execution metadata are internal preconditions,
// checked with debug assertions. CUDA/cuBLAS failures throw in all builds.

class Qwen3Attention final : public Module {
 public:
  Qwen3Attention(const ModelConfig& config, DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const ForwardContext& context = {}) const;

  // KV-cache views are non-owning. ModelRunner must keep the backing storage
  // alive for as long as the model can execute.
  void set_kv_cache(TensorView key_cache, TensorView value_cache);
  void clear_kv_cache() noexcept;
  [[nodiscard]] ConstTensorView key_cache() const;
  [[nodiscard]] ConstTensorView value_cache() const;

  [[nodiscard]] std::size_t num_heads() const noexcept;
  [[nodiscard]] std::size_t num_kv_heads() const noexcept;
  [[nodiscard]] std::size_t head_dim() const noexcept;
  [[nodiscard]] std::size_t q_size() const noexcept;
  [[nodiscard]] std::size_t kv_size() const noexcept;
  [[nodiscard]] float scaling() const noexcept;

 private:
  std::size_t hidden_size_;
  std::size_t num_heads_;
  std::size_t num_kv_heads_;
  std::size_t head_dim_;
  std::size_t q_size_;
  std::size_t kv_size_;
  float scaling_;
  bool qkv_bias_;

  // The checkpoint stores Q/K/V separately. Keeping these parameters unpacked
  // makes strict loading work with the generic Model loader. A future fused
  // GEMM may pack or concatenate them when kernels are initialized.
  Linear q_proj_;
  Linear k_proj_;
  Linear v_proj_;
  Linear o_proj_;
  std::unique_ptr<RMSNorm> q_norm_;
  std::unique_ptr<RMSNorm> k_norm_;
  RotaryEmbedding rotary_emb_;

  TensorView key_cache_;
  TensorView value_cache_;
};

struct Qwen3DecoderLayerOutput {
  Tensor hidden_states;
  Tensor residual;
};

class Qwen3DecoderLayer final : public Module {
 public:
  Qwen3DecoderLayer(const ModelConfig& config, DataType dtype, Device device);

  [[nodiscard]] Qwen3DecoderLayerOutput forward(
      ConstTensorView positions, ConstTensorView hidden_states,
      std::optional<ConstTensorView> residual = std::nullopt,
      const ForwardContext& context = {}) const;

  [[nodiscard]] Qwen3Attention& self_attention() noexcept;
  [[nodiscard]] const Qwen3Attention& self_attention() const noexcept;

 private:
  Qwen3Attention self_attn_;
  SwiGLUMLP mlp_;
  RMSNorm input_layernorm_;
  RMSNorm post_attention_layernorm_;
};

class Qwen3Model final : public Module {
 public:
  Qwen3Model(const ModelConfig& config, DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const ForwardContext& context = {}) const;

  [[nodiscard]] Embedding& embed_tokens() noexcept;
  [[nodiscard]] const Embedding& embed_tokens() const noexcept;
  [[nodiscard]] std::size_t num_layers() const noexcept;
  [[nodiscard]] Qwen3DecoderLayer& layer(std::size_t index);
  [[nodiscard]] const Qwen3DecoderLayer& layer(std::size_t index) const;

 private:
  std::size_t hidden_size_;
  Embedding embed_tokens_;
  ModuleList layers_;
  RMSNorm norm_;
};

class Qwen3ForCausalLM final : public Model {
 public:
  explicit Qwen3ForCausalLM(const ModelConfig& config,
                            Device device = Device::cpu());

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const ForwardContext& context = {}) const;
  [[nodiscard]] Tensor compute_logits(ConstTensorView hidden_states,
                                      const ForwardContext& context = {}) const;

  [[nodiscard]] const ModelConfig& config() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] Qwen3Model& transformer() noexcept;
  [[nodiscard]] const Qwen3Model& transformer() const noexcept;
  [[nodiscard]] LMHead& lm_head() noexcept;
  [[nodiscard]] const LMHead& lm_head() const noexcept;

 private:
  ModelConfig config_;
  DataType dtype_;
  Device device_;
  Qwen3Model model_;
  LMHead lm_head_;
};

} // namespace nanovllm
