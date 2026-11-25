#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "common/thread_pool.hh"
#include "core/decode_graph.hh"
#include "core/expert_policy.hh"
#include "core/model.hh"
#include "core/model_config.hh"
#include "layers/embedding.hh"
#include "layers/gated_delta_net.hh"
#include "layers/lm_head.hh"
#include "layers/moe/moe.hh"
#include "layers/rms_norm.hh"
#include "layers/rotary_embedding.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

class Qwen35Attention final : public Module {
 public:
  Qwen35Attention(const ModelConfig& config, Device device);

  void build_rope_cache(cudaStream_t stream);

  [[nodiscard]] Tensor forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const ForwardContext& context) const;

  void set_kv_cache(TensorView key, TensorView value);
  void clear_kv_cache() noexcept;

  [[nodiscard]] ConstTensorView key_cache() const;
  [[nodiscard]] ConstTensorView value_cache() const;

 private:
  std::size_t hidden_size_, num_heads_, num_kv_heads_, head_dim_, q_size_;
  FP8Linear q_proj_, k_proj_, v_proj_, o_proj_;
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
                     ThreadPool& cpu_pool, MoEExecutionWorkspace& workspace);

  [[nodiscard]] Qwen35DecoderLayerOutput forward(
      ConstTensorView positions, ConstTensorView hidden_states,
      std::optional<ConstTensorView> residual, const ForwardContext& context,
      MoEPhaseProfile* profile = nullptr, ExpertUsage* usage = nullptr) const;

  [[nodiscard]] Qwen35Attention* self_attention() noexcept;
  [[nodiscard]] GatedDeltaNet* linear_attention() noexcept;
  [[nodiscard]] MoE& mlp() noexcept;
  [[nodiscard]] const MoE& mlp() const noexcept;

 private:
  std::unique_ptr<Qwen35Attention> self_attn_;
  std::unique_ptr<GatedDeltaNet> linear_attn_;
  MoE mlp_;
  RMSNorm input_layernorm_, post_attention_layernorm_;
  mutable DecodeGraph decode_prefix_;
};

class Qwen35Model final : public Module {
 public:
  Qwen35Model(const ModelConfig& config, Device device,
              std::size_t cpu_threads = 8,
              const ExpertPlacement& gpu_expert_ids_by_layer = {},
              bool profile_experts = false,
              ExpertAdaptationParams expert_adaptation = {});

  void build_rope_cache(cudaStream_t stream);

  [[nodiscard]] Tensor forward(ConstTensorView input_ids,
                               ConstTensorView positions,
                               const ForwardContext& context) const;

  void set_cache(Qwen35Cache& cache);
  void clear_cache() noexcept;

  [[nodiscard]] Embedding& embed_tokens() noexcept;
  [[nodiscard]] std::size_t num_layers() const noexcept;
  [[nodiscard]] Qwen35DecoderLayer& layer(std::size_t index);
  [[nodiscard]] const Qwen35DecoderLayer& layer(std::size_t index) const;
  void reset_expert_profile();
  [[nodiscard]] const std::vector<MoELayerProfile>& expert_profile()
      const noexcept;
  [[nodiscard]] ExpertAdaptationStats expert_adaptation_stats() const;
  void finish_decode_step(cudaStream_t stream) const;
  void reset_hybrid_step() const;
  void run_hybrid_cpu() const;
  void abort_hybrid_step() const noexcept;

 private:
  void adapt_experts(cudaStream_t stream) const;
  // Workers and scratch storage outlive every executor.
  mutable ThreadPool cpu_pool_;
  mutable MoEExecutionWorkspace moe_workspace_;
  Embedding embed_tokens_;
  ModuleList layers_;
  RMSNorm norm_;
  mutable std::vector<MoELayerProfile> expert_profile_;
  mutable ExpertAdaptationPolicy expert_adaptation_;
  mutable ExpertAdaptationStats adaptation_stats_;
};

class Qwen35ForCausalLM final : public Model {
 public:
  explicit Qwen35ForCausalLM(
      const ModelConfig& config, Device device = Device::cpu(),
      std::size_t cpu_threads = 8,
      const ExpertPlacement& gpu_expert_ids_by_layer = {},
      bool profile_experts = false,
      ExpertAdaptationParams expert_adaptation = {});

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
