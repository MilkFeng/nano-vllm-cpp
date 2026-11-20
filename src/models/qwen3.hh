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

class Qwen3Attention final : public Module {
 public:
  Qwen3Attention(const ModelConfig& config, DataType dtype, Device device);

  void build_rope_cache(cudaStream_t stream);

  [[nodiscard]] Tensor forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const ForwardContext& context = {}) const;

  void set_kv_cache(TensorView key_cache, TensorView value_cache);
  void clear_kv_cache() noexcept;

  [[nodiscard]] ConstTensorView key_cache() const;
  [[nodiscard]] ConstTensorView value_cache() const;

  [[nodiscard]] std::size_t num_heads() const noexcept;
  [[nodiscard]] std::size_t num_kv_heads() const noexcept;
  [[nodiscard]] std::size_t head_dim() const noexcept;
  [[nodiscard]] std::size_t q_size() const noexcept;
  [[nodiscard]] std::size_t kv_size() const noexcept;

 private:
  std::size_t hidden_size_;
  std::size_t num_heads_;
  std::size_t num_kv_heads_;
  std::size_t head_dim_;
  std::size_t q_size_;
  std::size_t kv_size_;
  bool qkv_bias_;

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

class Qwen3Cache final {
 public:
  Qwen3Cache(const ModelConfig& config, std::size_t num_blocks,
             std::size_t block_size, Device device = Device::cuda(0));

  Qwen3Cache(const Qwen3Cache&) = delete;
  Qwen3Cache& operator=(const Qwen3Cache&) = delete;

  [[nodiscard]] TensorView key_cache(std::size_t layer);
  [[nodiscard]] TensorView value_cache(std::size_t layer);

  [[nodiscard]] std::size_t kv_bytes() const;

 private:
  Tensor key_, value_;
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

  void build_rope_cache(cudaStream_t stream);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const ForwardContext& context = {}) const;

  void set_cache(Qwen3Cache& cache);
  void clear_cache() noexcept;

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

  [[nodiscard]] Tensor forward(
      ConstTensorView input_ids, ConstTensorView positions,
      const ForwardContext& context = {}) const override;

  [[nodiscard]] Tensor compute_logits(
      ConstTensorView hidden_states,
      const ForwardContext& context = {}) const override;

  [[nodiscard]] const ModelConfig& config() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept override;
  [[nodiscard]] Device device() const noexcept override;
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
