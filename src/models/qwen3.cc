#include "models/qwen3.hh"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "kernels/cuda/attention.hh"
#include "kernels/cuda/inference.hh"

namespace nanovllm {
namespace {

[[maybe_unused]] bool is_index_vector(ConstTensorView tensor, Device device) {
  return tensor.rank() == 1 && tensor.dtype() == DataType::Int32
         && tensor.device() == device && tensor.is_contiguous();
}

ModelConfig validate_qwen3_config(ModelConfig config) {
  config.assert_valid();
  if (config.model_type != "qwen3") {
    throw std::invalid_argument("Qwen3ForCausalLM requires model_type 'qwen3'");
  }
  if (config.head_dim == 0 || config.head_dim > 256
      || config.head_dim % 2 != 0) {
    throw std::invalid_argument("Qwen3 head_dim must be even and in [2, 256]");
  }
  if (config.hidden_act != HiddenAct::SiLU) {
    throw std::invalid_argument("Qwen3 requires hidden_act 'silu'");
  }
  if (config.rope_scaling) {
    throw std::invalid_argument("Qwen3 RoPE scaling is not implemented");
  }
  return config;
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
      scaling_(1.0f / std::sqrtf(head_dim_)),
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

  // This follows the nano-vllm Qwen3 implementation: bias-free QKV uses
  // per-head Q/K RMSNorm, while the bias-enabled variant omits them.
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
  const auto device = q_proj_.weight().view().device();
  assert(device.type == DeviceType::Cuda && hidden_states.device() == device);
  assert(hidden_states.rank() == 2 && hidden_states.shape()[1] == hidden_size_);
  assert(is_index_vector(positions, device));
  assert(positions.shape()[0] == hidden_states.shape()[0]);
  const auto tokens = hidden_states.shape()[0];
  const bool have_cache = key_cache_.numel() != 0 && value_cache_.numel() != 0;
  const bool paged_prefill = context.block_tables.numel() != 0;
  if (have_cache) {
    assert(is_index_vector(context.slot_mapping, device));
    assert(context.slot_mapping.shape()[0] == tokens);
  }
  if (context.is_prefill) {
    assert(is_index_vector(context.cu_seqlens_q, device));
    assert(is_index_vector(context.cu_seqlens_k, device));
    assert(context.cu_seqlens_q.shape()[0] > 0
           && same_shape(context.cu_seqlens_q, context.cu_seqlens_k));
    assert(!paged_prefill || have_cache);
  } else {
    assert(have_cache && is_index_vector(context.context_lens, device));
    assert(context.context_lens.shape()[0] == tokens);
    assert(context.block_tables.rank() == 2
           && context.block_tables.shape()[0] == tokens);
  }

  const auto stream = context.stream;
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
  if (have_cache) {
    kernels::cuda::store_kvcache(key.view().as_const(), value.view().as_const(),
                                 key_cache_, value_cache_, context.slot_mapping,
                                 stream);
  }
  auto attention_output = Tensor::empty({tokens, num_heads_, head_dim_},
                                        hidden_states.dtype(), device);
  if (context.is_prefill) {
    const auto attention_key =
        paged_prefill ? key_cache() : key.view().as_const();
    const auto attention_value =
        paged_prefill ? value_cache() : value.view().as_const();
    kernels::cuda::prefill_attention(query.view().as_const(), attention_key,
                                     attention_value, context.cu_seqlens_q,
                                     context.cu_seqlens_k, context.block_tables,
                                     attention_output.view(), scaling_, stream);
  } else {
    kernels::cuda::paged_attention_decode(
        query.view().as_const(), key_cache(), value_cache(),
        context.context_lens, context.block_tables, attention_output.view(),
        scaling_, stream);
  }
  const auto flattened = attention_output.reshape(std::array{tokens, q_size_});
  return o_proj_.forward(flattened.view(), stream);
}

void Qwen3Attention::set_kv_cache(TensorView key_cache,
                                  TensorView value_cache) {
  assert(key_cache.rank() == 4 && value_cache.rank() == 4);
  assert(same_shape(key_cache, value_cache));
  assert(key_cache.shape()[0] > 0 && key_cache.shape()[1] > 0
         && key_cache.shape()[2] == num_kv_heads_
         && key_cache.shape()[3] == head_dim_);
  assert(key_cache.data() != value_cache.data());
  assert(same_dtype(key_cache, value_cache, q_proj_.weight().view()));
  assert(key_cache.device() == value_cache.device()
         && key_cache.device() == q_proj_.weight().view().device());
  assert(key_cache.is_contiguous() && value_cache.is_contiguous());

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

float Qwen3Attention::scaling() const noexcept {
  return scaling_;
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
  // The initial residual borrows hidden_states only until this fused operation
  // creates the owning updated residual returned to the next decoder layer.
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
    const auto residual_view =
        index == 0 ? std::nullopt
                   : std::optional<ConstTensorView>(residual.view().as_const());
    auto output = layer(index).forward(
        positions, hidden_states.view().as_const(), residual_view, context);
    hidden_states = std::move(output.hidden_states);
    residual = std::move(output.residual);
  }
  auto normalized = norm_.forward(hidden_states.view().as_const(),
                                  residual.view().as_const(), context.stream);
  return std::move(normalized.hidden_states);
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
    : config_(validate_qwen3_config(config)),
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
  // Prefill keeps only each sequence's last row before vocabulary projection.
  if (context.is_prefill) {
    assert(hidden_states.rank() == 2
           && hidden_states.shape()[1] == config_.hidden_size);
    assert(is_index_vector(context.cu_seqlens_q, hidden_states.device()));
    assert(context.cu_seqlens_q.shape()[0] > 0);
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
