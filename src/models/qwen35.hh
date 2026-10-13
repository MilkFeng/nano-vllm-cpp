#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/model.hh"
#include "core/model_config.hh"
#include "layers/embedding.hh"
#include "layers/gated_delta_net.hh"
#include "layers/lm_head.hh"
#include "layers/moe.hh"
#include "layers/rms_norm.hh"
#include "layers/rotary_embedding.hh"

namespace nanovllm {

class Qwen35Attention final : public Module {
 public:
  Qwen35Attention(const ModelConfig& config, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const ForwardContext& context) const;

  void set_kv_cache(TensorView key, TensorView value);
  void clear_kv_cache() noexcept;

  [[nodiscard]] ConstTensorView key_cache() const;
  [[nodiscard]] ConstTensorView value_cache() const;

 private:
  std::size_t hidden_size_, num_heads_, num_kv_heads_, head_dim_, q_size_;
  float scaling_;
  Linear q_proj_, k_proj_, v_proj_, o_proj_;
  RMSNorm q_norm_, k_norm_;
  RotaryEmbedding rotary_emb_;
  TensorView key_cache_, value_cache_;
};

class Qwen35Cache final {
 public:
  Qwen35Cache(const ModelConfig& config, std::size_t num_blocks,
              std::size_t block_size, std::size_t state_slots,
              Device device = Device::cuda(0));

  Qwen35Cache(const Qwen35Cache&) = delete;
  Qwen35Cache& operator=(const Qwen35Cache&) = delete;

  void reset_state(std::size_t slot, cudaStream_t stream = nullptr);

  [[nodiscard]] TensorView key_cache(std::size_t full_layer);
  [[nodiscard]] TensorView value_cache(std::size_t full_layer);
  [[nodiscard]] TensorView conv_state(std::size_t linear_layer);
  [[nodiscard]] TensorView recurrent_state(std::size_t linear_layer);
  [[nodiscard]] std::size_t kv_bytes() const;
  [[nodiscard]] std::size_t state_bytes() const;

 private:
  Tensor key_, value_, conv_, recurrent_;
};

struct Qwen35DecoderLayerOutput {
  Tensor hidden_states;
  Tensor residual;
};

class Qwen35DecoderLayer final : public Module {
 public:
  Qwen35DecoderLayer(const ModelConfig& config, std::size_t index,
                     Device device,
                     const std::vector<std::int32_t>& gpu_expert_ids,
                     ThreadPool& cpu_pool);

  [[nodiscard]] Qwen35DecoderLayerOutput forward(
      ConstTensorView positions, ConstTensorView hidden_states,
      std::optional<ConstTensorView> residual,
      const ForwardContext& context) const;

  [[nodiscard]] Qwen35Attention* self_attention() noexcept;
  [[nodiscard]] GatedDeltaNet* linear_attention() noexcept;
  [[nodiscard]] MoE& mlp() noexcept;

 private:
  std::unique_ptr<Qwen35Attention> self_attn_;
  std::unique_ptr<GatedDeltaNet> linear_attn_;
  MoE mlp_;
  RMSNorm input_layernorm_, post_attention_layernorm_;
};

class Qwen35Model final : public Module {
 public:
  Qwen35Model(const ModelConfig& config, Device device,
              const std::vector<std::int32_t>& gpu_expert_ids,
              std::size_t cpu_threads = 8);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const ForwardContext& context) const;

  void set_cache(Qwen35Cache& cache);
  void clear_cache() noexcept;

  [[nodiscard]] Embedding& embed_tokens() noexcept;
  [[nodiscard]] std::size_t num_layers() const noexcept;
  [[nodiscard]] Qwen35DecoderLayer& layer(std::size_t index);
  [[nodiscard]] const Qwen35DecoderLayer& layer(std::size_t index) const;

 private:
  // Declared before layers so the workers outlive every executor.
  ThreadPool cpu_pool_;
  Embedding embed_tokens_;
  ModuleList layers_;
  RMSNorm norm_;
};

class Qwen35ForCausalLM final : public Model {
 public:
  explicit Qwen35ForCausalLM(const ModelConfig& config,
                             Device device = Device::cpu(),
                             std::vector<std::int32_t> gpu_expert_ids = {},
                             std::size_t cpu_threads = 8);

  ModelLoadResult load_ckpt(const std::filesystem::path& path) override;

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const ForwardContext& context) const override;

  [[nodiscard]] Tensor compute_logits(
      ConstTensorView hidden_states,
      const ForwardContext& context) const override;

  [[nodiscard]] const ModelConfig& config() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept override;
  [[nodiscard]] Device device() const noexcept override;
  [[nodiscard]] Qwen35Model& transformer() noexcept;
  [[nodiscard]] LMHead& lm_head() noexcept;

 private:
  ModelConfig config_;
  Device device_;
  Qwen35Model model_;
  LMHead lm_head_;
};

} // namespace nanovllm
