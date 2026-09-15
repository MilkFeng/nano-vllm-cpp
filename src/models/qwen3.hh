#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>

#include "model/model.hh"
#include "model/model_config.hh"

namespace nanovllm {

// Non-owning execution metadata prepared by ModelRunner. Empty tensor views
// mean that the corresponding metadata is not used by the current execution
// mode.
struct Qwen3ForwardContext {
  bool is_prefill = false;
  ConstTensorView cu_seqlens_q;
  ConstTensorView cu_seqlens_k;
  std::size_t max_seqlen_q = 0;
  std::size_t max_seqlen_k = 0;
  ConstTensorView slot_mapping;
  ConstTensorView context_lens;
  ConstTensorView block_tables;
  cudaStream_t stream = nullptr;
};

class Qwen3Linear final : public Module {
 public:
  Qwen3Linear(std::size_t in_features, std::size_t out_features, bool bias,
              DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input,
                               cudaStream_t stream = nullptr) const;

  [[nodiscard]] std::size_t in_features() const noexcept;
  [[nodiscard]] std::size_t out_features() const noexcept;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;
  [[nodiscard]] Parameter* bias() noexcept;
  [[nodiscard]] const Parameter* bias() const noexcept;

 private:
  std::size_t in_features_;
  std::size_t out_features_;
  Parameter weight_;
  std::unique_ptr<Parameter> bias_;
};

struct Qwen3RMSNormOutput {
  Tensor hidden_states;
  Tensor residual;
};

class Qwen3RMSNorm final : public Module {
 public:
  Qwen3RMSNorm(std::size_t normalized_size, float epsilon, DataType dtype,
               Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input,
                               cudaStream_t stream = nullptr) const;
  [[nodiscard]] Qwen3RMSNormOutput forward(
      ConstTensorView input, ConstTensorView residual,
      cudaStream_t stream = nullptr) const;
  [[nodiscard]] std::size_t normalized_size() const noexcept;
  [[nodiscard]] float epsilon() const noexcept;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t normalized_size_;
  float epsilon_;
  Parameter weight_;
};

class Qwen3Embedding final : public Module {
 public:
  Qwen3Embedding(std::size_t vocab_size, std::size_t hidden_size,
                 DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               cudaStream_t stream = nullptr) const;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t vocab_size_;
  std::size_t hidden_size_;
  Parameter weight_;
};

class Qwen3LMHead final : public Module {
 public:
  Qwen3LMHead(std::size_t vocab_size, std::size_t hidden_size, DataType dtype,
              Device device, const Tensor* tied_weight = nullptr);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               const Qwen3ForwardContext& context = {}) const;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

 private:
  std::size_t vocab_size_;
  std::size_t hidden_size_;
  Parameter weight_;
};

class Qwen3Attention final : public Module {
 public:
  Qwen3Attention(const ModelConfig& config, DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const Qwen3ForwardContext& context = {}) const;

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
  std::size_t max_position_embeddings_;
  float rope_theta_;
  float scaling_;
  bool qkv_bias_;
  std::optional<RoPEScaling> rope_scaling_;

  // The checkpoint stores Q/K/V separately. Keeping these parameters unpacked
  // makes strict loading work with the generic Model loader. A future fused
  // GEMM may pack or concatenate them when kernels are initialized.
  Qwen3Linear q_proj_;
  Qwen3Linear k_proj_;
  Qwen3Linear v_proj_;
  Qwen3Linear o_proj_;
  std::unique_ptr<Qwen3RMSNorm> q_norm_;
  std::unique_ptr<Qwen3RMSNorm> k_norm_;
  Tensor rope_cache_;

  TensorView key_cache_;
  TensorView value_cache_;
};

class Qwen3MLP final : public Module {
 public:
  Qwen3MLP(const ModelConfig& config, DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               cudaStream_t stream = nullptr) const;

 private:
  std::size_t hidden_size_;
  std::size_t intermediate_size_;

  // gate_proj and up_proj remain separate to match checkpoint names. They can
  // be presented as one packed matrix to a fused SwiGLU kernel later.
  Qwen3Linear gate_proj_;
  Qwen3Linear up_proj_;
  Qwen3Linear down_proj_;
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
      const Qwen3ForwardContext& context = {}) const;

  [[nodiscard]] Qwen3Attention& self_attention() noexcept;
  [[nodiscard]] const Qwen3Attention& self_attention() const noexcept;

 private:
  Qwen3Attention self_attn_;
  Qwen3MLP mlp_;
  Qwen3RMSNorm input_layernorm_;
  Qwen3RMSNorm post_attention_layernorm_;
};

class Qwen3Model final : public Module {
 public:
  Qwen3Model(const ModelConfig& config, DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const Qwen3ForwardContext& context = {}) const;

  [[nodiscard]] Qwen3Embedding& embed_tokens() noexcept;
  [[nodiscard]] const Qwen3Embedding& embed_tokens() const noexcept;
  [[nodiscard]] std::size_t num_layers() const noexcept;
  [[nodiscard]] Qwen3DecoderLayer& layer(std::size_t index);
  [[nodiscard]] const Qwen3DecoderLayer& layer(std::size_t index) const;

 private:
  std::size_t hidden_size_;
  Qwen3Embedding embed_tokens_;
  ModuleList layers_;
  Qwen3RMSNorm norm_;
};

class Qwen3ForCausalLM final : public Model {
 public:
  explicit Qwen3ForCausalLM(const ModelConfig& config,
                            Device device = Device::cpu());

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const Qwen3ForwardContext& context = {}) const;
  [[nodiscard]] Tensor compute_logits(
      ConstTensorView hidden_states,
      const Qwen3ForwardContext& context = {}) const;

  [[nodiscard]] const ModelConfig& config() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] Qwen3Model& transformer() noexcept;
  [[nodiscard]] const Qwen3Model& transformer() const noexcept;
  [[nodiscard]] Qwen3LMHead& lm_head() noexcept;
  [[nodiscard]] const Qwen3LMHead& lm_head() const noexcept;

 private:
  ModelConfig config_;
  DataType dtype_;
  Device device_;
  Qwen3Model model_;
  Qwen3LMHead lm_head_;
};

} // namespace nanovllm
