#include "models/qwen3.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace nanovllm {
namespace {

std::size_t require_positive(std::size_t value, std::string_view name) {
  if (value == 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

float require_positive_finite(float value, std::string_view name) {
  if (!(value > 0.0f) || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(name)
                                + " must be positive and finite");
  }
  return value;
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs,
                            std::string_view name) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(std::string(name) + " is too large");
  }
  return lhs * rhs;
}

std::size_t validate_num_kv_heads(std::size_t num_heads,
                                  std::size_t num_kv_heads) {
  require_positive(num_kv_heads, "num_key_value_heads");
  if (num_heads % num_kv_heads != 0) {
    throw std::invalid_argument(
        "num_attention_heads must be divisible by num_key_value_heads");
  }
  return num_kv_heads;
}

std::size_t validate_intermediate_size(const ModelConfig& config) {
  if (config.hidden_act != HiddenAct::SiLU) {
    throw std::invalid_argument("Qwen3 requires the SiLU activation");
  }
  return require_positive(config.intermediate_size, "intermediate_size");
}

DataType model_data_type(TorchDataType dtype) {
  switch (dtype) {
    case TorchDataType::Float16:
      return DataType::Float16;
    case TorchDataType::BFloat16:
      return DataType::BFloat16;
    case TorchDataType::Float32:
      return DataType::Float32;
  }
  throw std::invalid_argument("unsupported Qwen3 model dtype");
}

ModelConfig validate_config(ModelConfig config) {
  if (config.model_type != "qwen3") {
    throw std::invalid_argument("Qwen3ForCausalLM requires model_type 'qwen3'");
  }

  require_positive(config.vocab_size, "vocab_size");
  require_positive(config.hidden_size, "hidden_size");
  require_positive(config.head_dim, "head_dim");
  require_positive(config.intermediate_size, "intermediate_size");
  require_positive(config.num_hidden_layers, "num_hidden_layers");
  const auto num_heads =
      require_positive(config.num_attention_heads, "num_attention_heads");
  validate_num_kv_heads(num_heads, config.num_key_value_heads);
  require_positive(config.max_position_embeddings, "max_position_embeddings");
  require_positive_finite(config.rope_theta, "rope_theta");
  require_positive_finite(config.rms_norm_eps, "rms_norm_eps");
  if (config.hidden_act != HiddenAct::SiLU) {
    throw std::invalid_argument("Qwen3 requires hidden_act 'silu'");
  }

  // Validate derived projection sizes before any parameter storage is created.
  (void)checked_product(config.num_attention_heads, config.head_dim,
                        "query projection size");
  (void)checked_product(config.num_key_value_heads, config.head_dim,
                        "key/value projection size");
  return config;
}

Tensor make_lm_head_weight(std::size_t vocab_size, std::size_t hidden_size,
                           DataType dtype, Device device,
                           const Tensor* tied_weight) {
  if (tied_weight == nullptr) {
    return Tensor::empty({vocab_size, hidden_size}, dtype, device);
  }

  const auto shape = tied_weight->shape();
  if (shape.size() != 2 || shape[0] != vocab_size || shape[1] != hidden_size
      || tied_weight->dtype() != dtype || tied_weight->device() != device) {
    throw std::invalid_argument(
        "the tied embedding is incompatible with the Qwen3 LM head");
  }

  // Tensor copies share Storage, matching tie_word_embeddings semantics.
  return *tied_weight;
}

[[noreturn]] void throw_kernels_not_implemented(std::string_view operation) {
  throw std::logic_error(std::string(operation)
                         + " requires Qwen3 compute kernels that have not been "
                           "implemented yet");
}

} // namespace

Qwen3Linear::Qwen3Linear(std::size_t in_features, std::size_t out_features,
                         bool bias, DataType dtype, Device device)
    : in_features_(require_positive(in_features, "linear in_features")),
      out_features_(require_positive(out_features, "linear out_features")),
      weight_(Tensor::empty({out_features_, in_features_}, dtype, device)) {
  register_parameter("weight", weight_);
  if (bias) {
    bias_ = std::make_unique<Parameter>(
        Tensor::empty({out_features_}, dtype, device));
    register_parameter("bias", *bias_);
  }
}

Tensor Qwen3Linear::forward(ConstTensorView input) const {
  (void)input;
  // TODO(kernel): return linear(input, weight_.view(), bias());
  throw_kernels_not_implemented("Qwen3Linear::forward");
}

std::size_t Qwen3Linear::in_features() const noexcept {
  return in_features_;
}

std::size_t Qwen3Linear::out_features() const noexcept {
  return out_features_;
}

Parameter& Qwen3Linear::weight() noexcept {
  return weight_;
}

const Parameter& Qwen3Linear::weight() const noexcept {
  return weight_;
}

Parameter* Qwen3Linear::bias() noexcept {
  return bias_.get();
}

const Parameter* Qwen3Linear::bias() const noexcept {
  return bias_.get();
}

Qwen3RMSNorm::Qwen3RMSNorm(std::size_t normalized_size, float epsilon,
                           DataType dtype, Device device)
    : normalized_size_(
          require_positive(normalized_size, "RMSNorm normalized_size")),
      epsilon_(require_positive_finite(epsilon, "RMSNorm epsilon")),
      weight_(Tensor::empty({normalized_size_}, dtype, device)) {
  register_parameter("weight", weight_);
}

Tensor Qwen3RMSNorm::forward(ConstTensorView input) const {
  (void)input;
  // TODO(kernel): compute rms = rsqrt(mean(input * input) + epsilon_), then
  // multiply input * rms * weight_.view() in the model dtype.
  throw_kernels_not_implemented("Qwen3RMSNorm::forward");
}

std::size_t Qwen3RMSNorm::normalized_size() const noexcept {
  return normalized_size_;
}

float Qwen3RMSNorm::epsilon() const noexcept {
  return epsilon_;
}

Parameter& Qwen3RMSNorm::weight() noexcept {
  return weight_;
}

const Parameter& Qwen3RMSNorm::weight() const noexcept {
  return weight_;
}

Qwen3Embedding::Qwen3Embedding(std::size_t vocab_size, std::size_t hidden_size,
                               DataType dtype, Device device)
    : vocab_size_(require_positive(vocab_size, "embedding vocab_size")),
      hidden_size_(require_positive(hidden_size, "embedding hidden_size")),
      weight_(Tensor::empty({vocab_size_, hidden_size_}, dtype, device)) {
  register_parameter("weight", weight_);
}

Tensor Qwen3Embedding::forward(ConstTensorView input_ids) const {
  (void)input_ids;
  // TODO(kernel): gather rows from weight_.view() for every token in input_ids.
  // A tensor-parallel implementation should mask out non-local vocabulary IDs
  // and all-reduce the local embedding outputs.
  throw_kernels_not_implemented("Qwen3Embedding::forward");
}

Parameter& Qwen3Embedding::weight() noexcept {
  return weight_;
}

const Parameter& Qwen3Embedding::weight() const noexcept {
  return weight_;
}

Qwen3LMHead::Qwen3LMHead(std::size_t vocab_size, std::size_t hidden_size,
                         DataType dtype, Device device,
                         const Tensor* tied_weight)
    : vocab_size_(require_positive(vocab_size, "LM head vocab_size")),
      hidden_size_(require_positive(hidden_size, "LM head hidden_size")),
      weight_(make_lm_head_weight(vocab_size_, hidden_size_, dtype, device,
                                  tied_weight)) {
  register_parameter("weight", weight_);
}

Tensor Qwen3LMHead::forward(ConstTensorView hidden_states,
                            const Qwen3ForwardContext& context) const {
  (void)hidden_states;
  (void)context;
  // TODO(kernel): during prefill, gather only each sequence's last hidden state
  // using context.cu_seqlens_q; then compute hidden_states @ weight_.view().T.
  // A tensor-parallel implementation should gather vocabulary shards on rank 0.
  throw_kernels_not_implemented("Qwen3LMHead::forward");
}

Parameter& Qwen3LMHead::weight() noexcept {
  return weight_;
}

const Parameter& Qwen3LMHead::weight() const noexcept {
  return weight_;
}

Qwen3Attention::Qwen3Attention(const ModelConfig& config, DataType dtype,
                               Device device)
    : hidden_size_(require_positive(config.hidden_size, "hidden_size")),
      num_heads_(
          require_positive(config.num_attention_heads, "num_attention_heads")),
      num_kv_heads_(
          validate_num_kv_heads(num_heads_, config.num_key_value_heads)),
      head_dim_(require_positive(config.head_dim, "head_dim")),
      q_size_(checked_product(num_heads_, head_dim_, "query projection size")),
      kv_size_(checked_product(num_kv_heads_, head_dim_,
                               "key/value projection size")),
      max_position_embeddings_(require_positive(config.max_position_embeddings,
                                                "max_position_embeddings")),
      rope_theta_(require_positive_finite(config.rope_theta, "rope_theta")),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      qkv_bias_(config.attention_bias),
      rope_scaling_(config.rope_scaling),
      q_proj_(hidden_size_, q_size_, qkv_bias_, dtype, device),
      k_proj_(hidden_size_, kv_size_, qkv_bias_, dtype, device),
      v_proj_(hidden_size_, kv_size_, qkv_bias_, dtype, device),
      o_proj_(q_size_, hidden_size_, false, dtype, device) {
  register_module("q_proj", q_proj_);
  register_module("k_proj", k_proj_);
  register_module("v_proj", v_proj_);
  register_module("o_proj", o_proj_);

  // This follows the nano-vllm Qwen3 implementation: bias-free QKV uses
  // per-head Q/K RMSNorm, while the bias-enabled variant omits them.
  if (!qkv_bias_) {
    q_norm_ = std::make_unique<Qwen3RMSNorm>(head_dim_, config.rms_norm_eps,
                                             dtype, device);
    k_norm_ = std::make_unique<Qwen3RMSNorm>(head_dim_, config.rms_norm_eps,
                                             dtype, device);
    register_module("q_norm", *q_norm_);
    register_module("k_norm", *k_norm_);
  }
}

Tensor Qwen3Attention::forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const Qwen3ForwardContext& context) const {
  (void)positions;
  (void)hidden_states;
  (void)context;
  // TODO(kernel):
  // 1. Project hidden_states with q_proj_, k_proj_ and v_proj_. A fused GEMM
  //    may consume the three checkpoint-compatible weights as packed views.
  // 2. Reshape Q/K/V to [tokens, heads, head_dim_].
  // 3. Apply q_norm_/k_norm_ when present, then apply RoPE using positions,
  //    rope_theta_, rope_scaling_ and max_position_embeddings_.
  // 4. Store new K/V in key_cache_/value_cache_ using context.slot_mapping.
  // 5. Run causal variable-length attention for prefill or paged KV-cache
  //    attention for decode, scaled by scaling_.
  // 6. Flatten the attention heads and apply o_proj_.
  throw_kernels_not_implemented("Qwen3Attention::forward");
}

void Qwen3Attention::set_kv_cache(TensorView key_cache,
                                  TensorView value_cache) {
  const auto key_shape = key_cache.shape();
  const auto value_shape = value_cache.shape();
  const auto parameter_view = q_proj_.weight().view();
  if (key_cache.rank() != 4 || value_cache.rank() != 4
      || !std::ranges::equal(key_shape, value_shape) || key_shape[0] == 0
      || key_shape[1] == 0 || key_shape[2] != num_kv_heads_
      || key_shape[3] != head_dim_ || key_cache.data() == value_cache.data()
      || key_cache.dtype() != value_cache.dtype()
      || key_cache.dtype() != parameter_view.dtype()
      || key_cache.device() != value_cache.device()
      || key_cache.device() != parameter_view.device()
      || !key_cache.is_contiguous() || !value_cache.is_contiguous()) {
    throw std::invalid_argument(
        "Qwen3 KV caches must be matching contiguous "
        "[blocks, block_size, num_kv_heads, head_dim] tensors on the model "
        "device");
  }

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

Qwen3MLP::Qwen3MLP(const ModelConfig& config, DataType dtype, Device device)
    : hidden_size_(require_positive(config.hidden_size, "hidden_size")),
      intermediate_size_(validate_intermediate_size(config)),
      gate_proj_(hidden_size_, intermediate_size_, false, dtype, device),
      up_proj_(hidden_size_, intermediate_size_, false, dtype, device),
      down_proj_(intermediate_size_, hidden_size_, false, dtype, device) {
  register_module("gate_proj", gate_proj_);
  register_module("up_proj", up_proj_);
  register_module("down_proj", down_proj_);
}

Tensor Qwen3MLP::forward(ConstTensorView hidden_states) const {
  (void)hidden_states;
  // TODO(kernel): gate = gate_proj_(hidden_states),
  // up = up_proj_(hidden_states), then return down_proj_(SiLU(gate) * up).
  // gate_proj_ and up_proj_ can share one merged-column GEMM/SwiGLU kernel.
  throw_kernels_not_implemented("Qwen3MLP::forward");
}

Qwen3DecoderLayer::Qwen3DecoderLayer(const ModelConfig& config, DataType dtype,
                                     Device device)
    : self_attn_(config, dtype, device),
      mlp_(config, dtype, device),
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
    const Qwen3ForwardContext& context) const {
  (void)positions;
  (void)hidden_states;
  (void)residual;
  (void)context;
  // TODO(kernel): fuse residual-add + RMSNorm where possible:
  // - Normalize the input and preserve/update residual.
  // - Run self_attn_ and post_attention_layernorm_.
  // - Run mlp_; the next layer consumes the MLP result plus residual.
  throw_kernels_not_implemented("Qwen3DecoderLayer::forward");
}

Qwen3Attention& Qwen3DecoderLayer::self_attention() noexcept {
  return self_attn_;
}

const Qwen3Attention& Qwen3DecoderLayer::self_attention() const noexcept {
  return self_attn_;
}

Qwen3Model::Qwen3Model(const ModelConfig& config, DataType dtype, Device device)
    : hidden_size_(require_positive(config.hidden_size, "hidden_size")),
      embed_tokens_(config.vocab_size, hidden_size_, dtype, device),
      norm_(hidden_size_, config.rms_norm_eps, dtype, device) {
  register_module("embed_tokens", embed_tokens_);
  register_module("layers", layers_);
  register_module("norm", norm_);

  const auto num_layers =
      require_positive(config.num_hidden_layers, "num_hidden_layers");
  for (std::size_t index = 0; index < num_layers; ++index) {
    layers_.emplace_back<Qwen3DecoderLayer>(config, dtype, device);
  }
}

Tensor Qwen3Model::forward(ConstTensorView input_ids, ConstTensorView positions,
                           const Qwen3ForwardContext& context) const {
  (void)input_ids;
  (void)positions;
  (void)context;
  // TODO(kernel): gather token embeddings, execute every decoder layer while
  // carrying the fused residual stream, then apply the final residual-add and
  // norm_.
  throw_kernels_not_implemented("Qwen3Model::forward");
}

Qwen3Embedding& Qwen3Model::embed_tokens() noexcept {
  return embed_tokens_;
}

const Qwen3Embedding& Qwen3Model::embed_tokens() const noexcept {
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
    : config_(validate_config(config)),
      dtype_(model_data_type(config_.dtype)),
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
                                 const Qwen3ForwardContext& context) const {
  (void)input_ids;
  (void)positions;
  (void)context;
  // TODO(kernel): return model_.forward(input_ids, positions, context).
  throw_kernels_not_implemented("Qwen3ForCausalLM::forward");
}

Tensor Qwen3ForCausalLM::compute_logits(
    ConstTensorView hidden_states, const Qwen3ForwardContext& context) const {
  (void)hidden_states;
  (void)context;
  // TODO(kernel): return lm_head_.forward(hidden_states, context).
  throw_kernels_not_implemented("Qwen3ForCausalLM::compute_logits");
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

Qwen3LMHead& Qwen3ForCausalLM::lm_head() noexcept {
  return lm_head_;
}

const Qwen3LMHead& Qwen3ForCausalLM::lm_head() const noexcept {
  return lm_head_;
}

} // namespace nanovllm
