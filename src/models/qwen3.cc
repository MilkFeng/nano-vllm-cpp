#include "models/qwen3.hh"

#include <array>
#include <cassert>
#include <optional>
#include <utility>
#include <vector>

namespace nanovllm {
namespace {

TensorView cache_layer(Tensor& pool, std::size_t index) {
  assert(index < pool.shape()[0]);
  const std::array shape{pool.shape()[1], pool.shape()[2], pool.shape()[3],
                         pool.shape()[4]};
  return pool.view().slice(0, index, 1).reshape(shape);
}

} // namespace

Qwen3Cache::Qwen3Cache(const ModelConfig& config, std::size_t num_blocks,
                       std::size_t block_size, Device device) {
  const std::vector<std::size_t> shape{
      config.num_hidden_layers,   num_blocks,      block_size,
      config.num_key_value_heads, config.head_dim,
  };

  key_ = Tensor::empty(shape, config.dtype, device);
  value_ = Tensor::empty(shape, config.dtype, device);
}

TensorView Qwen3Cache::key_cache(std::size_t layer) {
  return cache_layer(key_, layer);
}

TensorView Qwen3Cache::value_cache(std::size_t layer) {
  return cache_layer(value_, layer);
}

std::size_t Qwen3Cache::kv_bytes() const {
  return key_.size_bytes() + value_.size_bytes();
}

Qwen3DecoderLayer::Qwen3DecoderLayer(const ModelConfig& config, DataType dtype,
                                     Device device)
    : self_attn_(config.hidden_size, config.num_attention_heads,
                 config.num_key_value_heads, config.head_dim,
                 config.rms_norm_eps, config.max_position_embeddings,
                 config.rope_theta, dtype, device),
      mlp_(config.hidden_size, config.intermediate_size, dtype, device),
      input_layernorm_(config.hidden_size, config.rms_norm_eps, dtype, device),
      post_attention_layernorm_(config.hidden_size, config.rms_norm_eps, dtype,
                                device) {
  register_module("self_attn", self_attn_);
  register_module("mlp", mlp_);
  register_module("input_layernorm", input_layernorm_);
  register_module("post_attention_layernorm", post_attention_layernorm_);
}

Qwen3DecoderLayerOutput Qwen3DecoderLayer::forward(
    ConstTensorView positions, ConstTensorView hidden_states,
    std::optional<ConstTensorView> residual,
    const ForwardContext& context) const {
  RMSNormOutput before;

  auto residual_view = hidden_states;
  if (residual) {
    before = input_layernorm_.forward(hidden_states, *residual, context.stream);
    residual_view = before.residual.view();
  } else {
    before.hidden_states =
        input_layernorm_.forward(hidden_states, context.stream);
  }

  auto attention =
      self_attn_.forward(positions, before.hidden_states.view(), context);

  auto after = post_attention_layernorm_.forward(attention.view(),
                                                 residual_view, context.stream);

  auto output = mlp_.forward(after.hidden_states.view(), context.stream);

  return {std::move(output), std::move(after.residual)};
}

Qwen3SelfAttention& Qwen3DecoderLayer::self_attention() noexcept {
  return self_attn_;
}

const Qwen3SelfAttention& Qwen3DecoderLayer::self_attention() const noexcept {
  return self_attn_;
}

Qwen3Model::Qwen3Model(const ModelConfig& config, DataType dtype, Device device)
    : hidden_size_(config.hidden_size),
      embed_tokens_(config.vocab_size, hidden_size_, dtype, device),
      norm_(hidden_size_, config.rms_norm_eps, dtype, device) {
  register_module("embed_tokens", embed_tokens_);
  register_module("layers", layers_);
  register_module("norm", norm_);

  const auto num_layers = config.num_hidden_layers;
  for (std::size_t index = 0; index < num_layers; ++index) {
    layers_.emplace_back<Qwen3DecoderLayer>(config, dtype, device);
  }
}

void Qwen3Model::build_rope_cache(cudaStream_t stream) {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).self_attention().build_rope_cache(stream);
  }
}

Tensor Qwen3Model::forward(ConstTensorView input_ids, ConstTensorView positions,
                           const ForwardContext& context) const {
  Tensor hidden = embed_tokens_.forward(input_ids, context.stream);

  Tensor residual;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    std::optional<ConstTensorView> residual_view = std::nullopt;
    if (i > 0) {
      residual_view = residual.view();
    }

    auto output =
        layer(i).forward(positions, hidden.view(), residual_view, context);

    hidden = std::move(output.hidden_states);
    residual = std::move(output.residual);
  }

  auto normalized =
      norm_.forward(hidden.view(), residual.view(), context.stream);
  return std::move(normalized.hidden_states);
}

void Qwen3Model::set_cache(Qwen3Cache& cache) {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).self_attention().set_kv_cache(cache.key_cache(i),
                                           cache.value_cache(i));
  }
}

void Qwen3Model::clear_cache() noexcept {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).self_attention().clear_kv_cache();
  }
}

Embedding& Qwen3Model::embed_tokens() noexcept {
  return embed_tokens_;
}

const Embedding& Qwen3Model::embed_tokens() const noexcept {
  return embed_tokens_;
}

std::size_t Qwen3Model::num_layers() const noexcept {
  return layers_.size();
}

Qwen3DecoderLayer& Qwen3Model::layer(std::size_t index) {
  return static_cast<Qwen3DecoderLayer&>(layers_.at(index));
}

const Qwen3DecoderLayer& Qwen3Model::layer(std::size_t index) const {
  return static_cast<const Qwen3DecoderLayer&>(layers_.at(index));
}

Qwen3ForCausalLM::Qwen3ForCausalLM(const ModelConfig& config, Device device)
    : config_(config),
      dtype_(config_.dtype),
      device_(device),
      model_(config_, dtype_, device_),
      lm_head_(config_.vocab_size, config_.hidden_size, dtype_, device_,
               config_.tie_word_embeddings
                   ? &model_.embed_tokens().weight().tensor()
                   : nullptr) {
  register_module("model", model_);
  register_module("lm_head", lm_head_);
}

Tensor Qwen3ForCausalLM::forward(ConstTensorView input_ids,
                                 ConstTensorView positions,
                                 const ForwardContext& context) const {
  return model_.forward(input_ids, positions, context);
}

Tensor Qwen3ForCausalLM::compute_logits(ConstTensorView hidden_states,
                                        const ForwardContext& context) const {
  if (context.is_prefill) {
    return lm_head_.forward(hidden_states, context.cu_seqlens_q,
                            context.stream);
  }
  return lm_head_.forward(hidden_states, context.stream);
}

const ModelConfig& Qwen3ForCausalLM::config() const noexcept {
  return config_;
}

DataType Qwen3ForCausalLM::dtype() const noexcept {
  return dtype_;
}

Device Qwen3ForCausalLM::device() const noexcept {
  return device_;
}

Qwen3Model& Qwen3ForCausalLM::transformer() noexcept {
  return model_;
}

const Qwen3Model& Qwen3ForCausalLM::transformer() const noexcept {
  return model_;
}

LMHead& Qwen3ForCausalLM::lm_head() noexcept {
  return lm_head_;
}

const LMHead& Qwen3ForCausalLM::lm_head() const noexcept {
  return lm_head_;
}

} // namespace nanovllm
