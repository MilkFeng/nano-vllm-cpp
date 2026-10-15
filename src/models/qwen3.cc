#include "models/qwen3.hh"

#include <array>
#include <cassert>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kernels/cuda/kernels.hh"

namespace nanovllm {
namespace {

[[maybe_unused]] bool is_index_vector(ConstTensorView tensor,
                                      ConstTensorView reference) {
  return tensor.rank() == 1 && tensor.dtype() == DataType::Int32
         && same_device(tensor, reference) && tensor.is_contiguous();
}

TensorView cache_layer(Tensor& pool, std::size_t index) {
  assert(index < pool.shape()[0]);
  const std::array shape{pool.shape()[1], pool.shape()[2], pool.shape()[3],
                         pool.shape()[4]};
  return pool.view().slice(0, index, 1).reshape(shape);
}

} // namespace

Qwen3Attention::Qwen3Attention(const ModelConfig& config, DataType dtype,
                               Device device)
    : hidden_size_(config.hidden_size),
      num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      q_size_(num_heads_ * head_dim_),
      kv_size_(num_kv_heads_ * head_dim_),
      qkv_bias_(config.attention_bias),
      q_proj_(hidden_size_, q_size_, qkv_bias_, dtype, device),
      k_proj_(hidden_size_, kv_size_, qkv_bias_, dtype, device),
      v_proj_(hidden_size_, kv_size_, qkv_bias_, dtype, device),
      o_proj_(q_size_, hidden_size_, false, dtype, device),
      rotary_emb_(config.max_position_embeddings, head_dim_, config.rope_theta,
                  device) {
  if (config.rope_scaling) {
    throw std::invalid_argument("Qwen3 RoPE scaling is not implemented");
  }

  register_module("q_proj", q_proj_);
  register_module("k_proj", k_proj_);
  register_module("v_proj", v_proj_);
  register_module("o_proj", o_proj_);

  if (!qkv_bias_) {
    q_norm_ = std::make_unique<RMSNorm>(head_dim_, config.rms_norm_eps, dtype,
                                        device);
    k_norm_ = std::make_unique<RMSNorm>(head_dim_, config.rms_norm_eps, dtype,
                                        device);

    register_module("q_norm", *q_norm_);
    register_module("k_norm", *k_norm_);
  }
}

Tensor Qwen3Attention::forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const ForwardContext& context) const {
  const auto stream = context.stream;
  const auto device = q_proj_.weight().view().device();

  const auto tokens = hidden_states.shape()[0];

  auto query_projection = q_proj_.forward(hidden_states, stream);
  auto key_projection = k_proj_.forward(hidden_states, stream);
  auto value_projection = v_proj_.forward(hidden_states, stream);

  auto query_heads =
      query_projection.reshape(std::array{tokens, num_heads_, head_dim_});
  auto key_heads =
      key_projection.reshape(std::array{tokens, num_kv_heads_, head_dim_});
  auto value =
      value_projection.reshape(std::array{tokens, num_kv_heads_, head_dim_});

  auto query = q_norm_ ? q_norm_->forward(query_heads.view().as_const(), stream)
                       : query_heads;
  auto key = k_norm_ ? k_norm_->forward(key_heads.view().as_const(), stream)
                     : key_heads;

  rotary_emb_.forward(positions, query.view(), key.view(), stream);

  auto attention_output = Tensor::empty({tokens, num_heads_, head_dim_},
                                        hidden_states.dtype(), device);
  kernels::cuda::flash_attn_with_kvcache(
      query.view().as_const(), key_cache_, value_cache_, key.view().as_const(),
      value.view().as_const(), context.cu_seqlens_q, context.cu_seqlens_k,
      context.max_seqlen_q, context.max_seqlen_k, context.block_tables,
      attention_output.view(), true, stream);

  const auto flattened = attention_output.reshape(std::array{tokens, q_size_});
  return o_proj_.forward(flattened.view(), stream);
}

void Qwen3Attention::set_kv_cache(TensorView key_cache,
                                  TensorView value_cache) {
  key_cache_ = std::move(key_cache);
  value_cache_ = std::move(value_cache);
}

void Qwen3Attention::clear_kv_cache() noexcept {
  key_cache_ = {};
  value_cache_ = {};
}

ConstTensorView Qwen3Attention::key_cache() const {
  return ConstTensorView(key_cache_);
}

ConstTensorView Qwen3Attention::value_cache() const {
  return ConstTensorView(value_cache_);
}

std::size_t Qwen3Attention::num_heads() const noexcept {
  return num_heads_;
}

std::size_t Qwen3Attention::num_kv_heads() const noexcept {
  return num_kv_heads_;
}

std::size_t Qwen3Attention::head_dim() const noexcept {
  return head_dim_;
}

std::size_t Qwen3Attention::q_size() const noexcept {
  return q_size_;
}

std::size_t Qwen3Attention::kv_size() const noexcept {
  return kv_size_;
}

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
    : self_attn_(config, dtype, device),
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
  RMSNormOutput before_attention;

  auto residual_view = hidden_states;
  if (residual) {
    before_attention =
        input_layernorm_.forward(hidden_states, *residual, context.stream);
    residual_view = before_attention.residual.view().as_const();
  } else {
    before_attention.hidden_states =
        input_layernorm_.forward(hidden_states, context.stream);
  }

  auto attention = self_attn_.forward(
      positions, before_attention.hidden_states.view().as_const(), context);

  auto after_attention = post_attention_layernorm_.forward(
      attention.view().as_const(), residual_view, context.stream);

  auto output = mlp_.forward(after_attention.hidden_states.view().as_const(),
                             context.stream);

  return {std::move(output), std::move(after_attention.residual)};
}

Qwen3Attention& Qwen3DecoderLayer::self_attention() noexcept {
  return self_attn_;
}

const Qwen3Attention& Qwen3DecoderLayer::self_attention() const noexcept {
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

Tensor Qwen3Model::forward(ConstTensorView input_ids, ConstTensorView positions,
                           const ForwardContext& context) const {
  auto hidden_states = embed_tokens_.forward(input_ids, context.stream);
  Tensor residual;
  for (std::size_t index = 0; index < layers_.size(); ++index) {
    std::optional<ConstTensorView> residual_view = std::nullopt;
    if (index > 0) {
      residual_view = residual.view().as_const();
    }

    auto output = layer(index).forward(
        positions, hidden_states.view().as_const(), residual_view, context);

    hidden_states = std::move(output.hidden_states);
    residual = std::move(output.residual);
  }

  auto normalized = norm_.forward(hidden_states.view().as_const(),
                                  residual.view().as_const(), context.stream);
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
    const auto batch = context.cu_seqlens_q.shape()[0] - 1;
    auto last_hidden =
        Tensor::empty({batch, config_.hidden_size}, hidden_states.dtype(),
                      hidden_states.device());
    kernels::cuda::gather_last_hidden(hidden_states, context.cu_seqlens_q,
                                      last_hidden.view(), context.stream);
    return lm_head_.forward(last_hidden.view().as_const(), context.stream);
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
