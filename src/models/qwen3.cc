#include "models/qwen3.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "kernels/attention.hh"
#include "kernels/inference.hh"
#include "kernels/kernels.hh"

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

std::size_t validate_head_dim(std::size_t head_dim) {
  if (head_dim == 0 || head_dim > 256 || head_dim % 2 != 0) {
    throw std::invalid_argument("Qwen3 head_dim must be even and in [2, 256]");
  }
  return head_dim;
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

Tensor make_rope_cache(std::size_t max_positions, std::size_t head_dim,
                       float theta, Device device) {
  // CPU construction remains useful for checkpoint inspection and loading.
  if (device.type != DeviceType::Cuda) return {};
  auto cache = Tensor::empty({max_positions, head_dim}, DataType::Float32, device);
  int previous_device = 0;
  check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice");
  check_cuda(cudaSetDevice(device.index), "cudaSetDevice");
  try {
    kernels::build_rope_cache(cache.view(), theta);
    // Publish fully initialized cache data to forwards on any CUDA stream.
    // This synchronization occurs once during construction, never in forward.
    check_cuda(cudaStreamSynchronize(nullptr), "initialize RoPE cache");
  } catch (...) {
    (void)cudaSetDevice(previous_device);
    throw;
  }
  check_cuda(cudaSetDevice(previous_device), "restore CUDA device");
  return cache;
}

bool is_index_vector(ConstTensorView tensor, Device device) {
  return tensor.rank() == 1 && tensor.dtype() == DataType::Int32
         && tensor.device() == device && tensor.is_contiguous();
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
  validate_head_dim(config.head_dim);
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
  if (config.rope_scaling) {
    throw std::invalid_argument("Qwen3 RoPE scaling is not implemented");
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

Tensor Qwen3Linear::forward(ConstTensorView input, cudaStream_t stream) const {
  const auto weight = weight_.view();
  if (input.rank() != 2 || input.shape()[1] != in_features_
      || input.device().type != DeviceType::Cuda
      || input.device() != weight.device() || input.dtype() != weight.dtype()) {
    throw std::invalid_argument("Qwen3Linear requires matching CUDA input "
                                "with shape [tokens, in_features]");
  }
  auto output = Tensor::empty({input.shape()[0], out_features_}, input.dtype(),
                              input.device());
  std::optional<ConstTensorView> bias;
  if (bias_) bias = std::as_const(*bias_).view();
  kernels::linear(input, weight, output.view(), bias, stream);
  return output;
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

Tensor Qwen3RMSNorm::forward(ConstTensorView input, cudaStream_t stream) const {
  const auto weight = weight_.view();
  if (input.rank() < 1 || input.rank() > 3
      || input.shape().back() != normalized_size_
      || input.device() != weight.device() || input.dtype() != weight.dtype()
      || input.device().type != DeviceType::Cuda) {
    throw std::invalid_argument("Qwen3RMSNorm requires a matching CUDA input "
                                "of rank 1 to 3 and the configured width");
  }
  auto output = Tensor::empty({input.shape().begin(), input.shape().end()},
                              input.dtype(), input.device());
  kernels::rms_norm(input, weight, output.view(), epsilon_, stream);
  return output;
}

Qwen3RMSNormOutput Qwen3RMSNorm::forward(ConstTensorView input,
                                       ConstTensorView residual,
                                       cudaStream_t stream) const {
  const auto weight = weight_.view();
  if (input.rank() < 1 || input.rank() > 3
      || input.shape().back() != normalized_size_
      || input.device() != weight.device() || input.dtype() != weight.dtype()
      || input.device().type != DeviceType::Cuda
      || !std::ranges::equal(input.shape(), residual.shape())
      || input.device() != residual.device() || input.dtype() != residual.dtype()) {
    throw std::invalid_argument("Qwen3RMSNorm requires matching CUDA input and "
                                "residual of rank 1 to 3 and the configured width");
  }
  const std::vector<std::size_t> shape(input.shape().begin(), input.shape().end());
  Qwen3RMSNormOutput result{
      Tensor::empty(shape, input.dtype(), input.device()),
      Tensor::empty(shape, input.dtype(), input.device())};
  kernels::fused_add_rms_norm(input, residual, weight, result.hidden_states.view(),
                             result.residual.view(), epsilon_, stream);
  return result;
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

Tensor Qwen3Embedding::forward(ConstTensorView input_ids,
                               cudaStream_t stream) const {
  const auto weight = weight_.view();
  if (input_ids.rank() != 1 || input_ids.dtype() != DataType::Int32
      || input_ids.device() != weight.device()
      || input_ids.device().type != DeviceType::Cuda) {
    throw std::invalid_argument("Qwen3Embedding requires rank-1 Int32 token IDs "
                                "on the weight's CUDA device");
  }
  auto output = Tensor::empty({input_ids.shape()[0], hidden_size_}, weight.dtype(),
                              weight.device());
  kernels::embedding(input_ids, weight, output.view(), stream);
  return output;
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
  const auto weight = weight_.view();
  if (hidden_states.rank() != 2 || hidden_states.shape()[1] != hidden_size_
      || hidden_states.device().type != DeviceType::Cuda
      || hidden_states.device() != weight.device()
      || hidden_states.dtype() != weight.dtype()) {
    throw std::invalid_argument("Qwen3LMHead requires matching CUDA hidden states "
                                "with shape [tokens, hidden_size]");
  }
  Tensor last_hidden;
  auto selected_hidden = hidden_states;
  if (context.is_prefill) {
    if (!is_index_vector(context.cu_seqlens_q, hidden_states.device())
        || context.cu_seqlens_q.shape()[0] == 0) {
      throw std::invalid_argument("Prefill LM head requires CUDA Int32 "
                                  "cu_seqlens_q of shape [batch + 1]");
    }
    const auto batch = context.cu_seqlens_q.shape()[0] - 1;
    last_hidden = Tensor::empty({batch, hidden_size_}, hidden_states.dtype(),
                                hidden_states.device());
    kernels::gather_last_hidden(hidden_states, context.cu_seqlens_q,
                                last_hidden.view(), context.stream);
    selected_hidden = last_hidden.view().as_const();
  }
  auto logits = Tensor::empty({selected_hidden.shape()[0], vocab_size_},
                              hidden_states.dtype(), hidden_states.device());
  kernels::linear(selected_hidden, weight, logits.view(), std::nullopt,
                   context.stream);
  return logits;
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
      head_dim_(validate_head_dim(config.head_dim)),
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
  if (rope_scaling_) {
    throw std::invalid_argument("Qwen3 RoPE scaling is not implemented");
  }
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
  rope_cache_ = make_rope_cache(max_position_embeddings_, head_dim_, rope_theta_,
                               device);
}

Tensor Qwen3Attention::forward(ConstTensorView positions,
                               ConstTensorView hidden_states,
                               const Qwen3ForwardContext& context) const {
  const auto device = q_proj_.weight().view().device();
  if (device.type != DeviceType::Cuda || hidden_states.rank() != 2
      || hidden_states.shape()[1] != hidden_size_
      || hidden_states.device() != device
      || !is_index_vector(positions, device)
      || positions.shape()[0] != hidden_states.shape()[0]) {
    throw std::invalid_argument("Qwen3Attention requires CUDA hidden states "
                                "[tokens, hidden_size] and Int32 positions [tokens]");
  }
  const auto tokens = hidden_states.shape()[0];
  const bool have_cache = key_cache_.numel() != 0 && value_cache_.numel() != 0;
  const bool paged_prefill = context.block_tables.numel() != 0;
  if (have_cache
      && (!is_index_vector(context.slot_mapping, device)
          || context.slot_mapping.shape()[0] != tokens)) {
    throw std::invalid_argument("Cached Qwen3 attention requires slot_mapping "
                                "for every input token");
  }
  if (context.is_prefill) {
    if (!is_index_vector(context.cu_seqlens_q, device)
        || !is_index_vector(context.cu_seqlens_k, device)
        || context.cu_seqlens_q.shape()[0] == 0
        || context.cu_seqlens_q.shape()[0] != context.cu_seqlens_k.shape()[0]) {
      throw std::invalid_argument("Prefill attention requires matching CUDA "
                                  "Int32 cu_seqlens_q/k [batch + 1]");
    }
    if (paged_prefill && !have_cache) {
      throw std::invalid_argument("Paged prefill requires initialized KV caches");
    }
  } else if (!have_cache || !is_index_vector(context.context_lens, device)
             || context.context_lens.shape()[0] != tokens
             || context.block_tables.rank() != 2
             || context.block_tables.shape()[0] != tokens) {
    throw std::invalid_argument("Decode attention requires initialized KV caches, "
                                "context_lens and block_tables");
  }

  const auto stream = context.stream;
  auto query_projection = q_proj_.forward(hidden_states, stream);
  auto key_projection = k_proj_.forward(hidden_states, stream);
  auto value_projection = v_proj_.forward(hidden_states, stream);
  auto query_heads = query_projection.reshape(
      std::array{tokens, num_heads_, head_dim_});
  auto key_heads = key_projection.reshape(
      std::array{tokens, num_kv_heads_, head_dim_});
  auto value = value_projection.reshape(std::array{tokens, num_kv_heads_, head_dim_});
  auto query = q_norm_ ? q_norm_->forward(query_heads.view().as_const(), stream)
                       : query_heads;
  auto key = k_norm_ ? k_norm_->forward(key_heads.view().as_const(), stream)
                     : key_heads;
  kernels::apply_rotary_embedding(positions, query.view().as_const(),
                                   key.view().as_const(), rope_cache_.view(),
                                   query.view(), key.view(), stream);
  if (have_cache) {
    kernels::store_kvcache(key.view().as_const(), value.view().as_const(),
                            key_cache_, value_cache_, context.slot_mapping, stream);
  }
  auto attention_output = Tensor::empty({tokens, num_heads_, head_dim_},
                                         hidden_states.dtype(), device);
  if (context.is_prefill) {
    const auto attention_key = paged_prefill ? key_cache() : key.view().as_const();
    const auto attention_value = paged_prefill ? value_cache()
                                               : value.view().as_const();
    kernels::prefill_attention(query.view().as_const(), attention_key,
                                attention_value, context.cu_seqlens_q,
                                context.cu_seqlens_k, context.block_tables,
                                attention_output.view(), scaling_, stream);
  } else {
    kernels::paged_attention_decode(query.view().as_const(), key_cache(),
                                     value_cache(), context.context_lens,
                                     context.block_tables, attention_output.view(),
                                     scaling_, stream);
  }
  const auto flattened = attention_output.reshape(std::array{tokens, q_size_});
  return o_proj_.forward(flattened.view(), stream);
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

Tensor Qwen3MLP::forward(ConstTensorView hidden_states, cudaStream_t stream) const {
  auto gate = gate_proj_.forward(hidden_states, stream);
  auto up = up_proj_.forward(hidden_states, stream);
  auto activated = Tensor::empty({hidden_states.shape()[0], intermediate_size_},
                                  hidden_states.dtype(), hidden_states.device());
  kernels::silu_and_mul(gate.view().as_const(), up.view().as_const(),
                        activated.view(), stream);
  return down_proj_.forward(activated.view().as_const(), stream);
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
  Qwen3RMSNormOutput before_attention;
  auto residual_view = hidden_states;
  if (residual) {
    before_attention = input_layernorm_.forward(hidden_states, *residual,
                                                 context.stream);
    residual_view = before_attention.residual.view().as_const();
  } else {
    before_attention.hidden_states = input_layernorm_.forward(hidden_states,
                                                               context.stream);
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
  auto hidden_states = embed_tokens_.forward(input_ids, context.stream);
  Tensor residual;
  for (std::size_t index = 0; index < layers_.size(); ++index) {
    const auto residual_view = index == 0
        ? std::nullopt
        : std::optional<ConstTensorView>(residual.view().as_const());
    auto output = layer(index).forward(positions, hidden_states.view().as_const(),
                                        residual_view, context);
    hidden_states = std::move(output.hidden_states);
    residual = std::move(output.residual);
  }
  auto normalized = norm_.forward(hidden_states.view().as_const(),
                                   residual.view().as_const(), context.stream);
  return std::move(normalized.hidden_states);
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
  return model_.forward(input_ids, positions, context);
}

Tensor Qwen3ForCausalLM::compute_logits(
    ConstTensorView hidden_states, const Qwen3ForwardContext& context) const {
  return lm_head_.forward(hidden_states, context);
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
