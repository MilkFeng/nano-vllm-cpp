#include "models/qwen35.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/checkpoint.hh"
#include "core/device.hh"
#include "core/load_plan.hh"
#include "functions/functions.hh"

namespace nanovllm {
namespace {

MoEConfig qwen35_moe_config(const ModelConfig& config) {
  assert(config.quantization_config);
  return {
      config.hidden_size,
      config.moe_intermediate_size,
      config.num_experts,
      config.num_experts_per_tok,
      config.shared_expert_intermediate_size,
  };
}

TensorView cache_layer(Tensor& pool, std::size_t index) {
  const std::vector<std::size_t> shape(pool.shape().begin() + 1,
                                       pool.shape().end());
  return pool.view().slice(0, index, 1).reshape(shape);
}

} // namespace

Qwen35Cache::Qwen35Cache(const ModelConfig& config, std::size_t num_blocks,
                         std::size_t block_size, std::size_t state_slots,
                         Device device) {
  const auto full = static_cast<std::size_t>(
      std::count(config.layer_types.begin(), config.layer_types.end(),
                 AttentionType::Full));
  const auto linear = config.num_hidden_layers - full;
  const auto conv_dim =
      2 * config.linear_num_key_heads * config.linear_key_head_dim
      + config.linear_num_value_heads * config.linear_value_head_dim;

  key_ = Tensor::empty({full, num_blocks, block_size,
                        config.num_key_value_heads, config.head_dim},
                       DataType::BFloat16, device);
  value_ = Tensor::empty({full, num_blocks, block_size,
                          config.num_key_value_heads, config.head_dim},
                         DataType::BFloat16, device);
  conv_ = Tensor::empty(
      {linear, state_slots, conv_dim, config.linear_conv_kernel_dim - 1},
      DataType::BFloat16, device);
  recurrent_ =
      Tensor::empty({linear, state_slots, config.linear_num_value_heads,
                     config.linear_key_head_dim, config.linear_value_head_dim},
                    DataType::Float32, device);

  functions::zero_tensor(conv_.view(), nullptr);
  functions::zero_tensor(recurrent_.view(), nullptr);

  if (device.type == DeviceType::Cuda) {
    check_cuda(cudaStreamSynchronize(nullptr), "initialize Qwen3.5 cache");
  }
}

void Qwen35Cache::reset_state(std::size_t slot, cudaStream_t stream) {
  for (std::size_t i = 0; i < recurrent_.shape()[0]; ++i) {
    functions::zero_tensor(conv_state(i).slice(0, slot, 1), stream);
    functions::zero_tensor(recurrent_state(i).slice(0, slot, 1), stream);
  }
}

TensorView Qwen35Cache::key_cache(std::size_t i) {
  return cache_layer(key_, i);
}

TensorView Qwen35Cache::value_cache(std::size_t i) {
  return cache_layer(value_, i);
}

TensorView Qwen35Cache::conv_state(std::size_t i) {
  return cache_layer(conv_, i);
}

TensorView Qwen35Cache::recurrent_state(std::size_t i) {
  return cache_layer(recurrent_, i);
}

std::size_t Qwen35Cache::kv_bytes() const {
  return key_.size_bytes() + value_.size_bytes();
}

std::size_t Qwen35Cache::state_bytes() const {
  return conv_.size_bytes() + recurrent_.size_bytes();
}

Qwen35DecoderLayer::Qwen35DecoderLayer(
    const ModelConfig& config, std::size_t index, Device device,
    const std::vector<std::int32_t>& gpu_expert_ids, ThreadPool& cpu_pool,
    MoEExecutionWorkspace& workspace)
    : mlp_(qwen35_moe_config(config), cpu_pool, device, gpu_expert_ids,
           &workspace),
      input_layernorm_(config.hidden_size, config.rms_norm_eps,
                       DataType::BFloat16, device, true),
      post_attention_layernorm_(config.hidden_size, config.rms_norm_eps,
                                DataType::BFloat16, device, true) {
  if (config.layer_types.at(index) == AttentionType::Full) {
    self_attn_ = std::make_unique<Qwen35SelfAttention>(
        config.hidden_size, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, config.rms_norm_eps,
        config.max_position_embeddings, config.rope_theta, DataType::BFloat16,
        device, config.rotary_dim(), true);
    register_module("self_attn", *self_attn_);
  } else {
    linear_attn_ = std::make_unique<GatedDeltaNet>(config, device);
    register_module("linear_attn", *linear_attn_);
  }

  register_module("mlp", mlp_);
  register_module("input_layernorm", input_layernorm_);
  register_module("post_attention_layernorm", post_attention_layernorm_);
}

Qwen35DecoderLayerOutput Qwen35DecoderLayer::forward(
    ConstTensorView positions, ConstTensorView hidden_states,
    std::optional<ConstTensorView> residual, const ForwardContext& context,
    MoEPhaseProfile* profile, ExpertUsage* usage) const {
  const auto prefix = [&]() -> std::vector<Tensor> {
    RMSNormOutput before;

    auto residual_view = hidden_states;
    auto residual_dtype = DataType::Float32;
    if (residual) {
      before =
          input_layernorm_.forward(hidden_states, *residual, context.stream);
      residual_view = before.residual.view();
      residual_dtype = residual->dtype();
    } else {
      before.hidden_states =
          input_layernorm_.forward(hidden_states, context.stream);
    }

    Tensor attention;
    if (self_attn_) {
      attention =
          self_attn_->forward(positions, before.hidden_states.view(), context);
    } else {
      attention = linear_attn_->forward(
          before.hidden_states.view(),
          context.is_prefill ? context.cu_seqlens_q : ConstTensorView{},
          context.state_indices, context.stream);
    }

    auto after = post_attention_layernorm_.forward(
        attention.view(), residual_view, residual_dtype, context.stream);

    return {std::move(after.hidden_states), std::move(after.residual)};
  };

  std::vector<Tensor> after;
  if (context.decode_graphs && linear_attn_) {
    after = decode_prefix_.run(
        context.stream,
        {hidden_states.data(), residual ? residual->data() : nullptr,
         context.state_indices.data(), linear_attn_->conv_state().data(),
         linear_attn_->recurrent_state().data()},
        prefix);
  } else {
    after = prefix();
  }

  auto output = mlp_.forward(after[0].view(), context.stream, profile, usage,
                             context.decode_graphs, context.whole_graph,
                             context.graph_warmup);

  return {std::move(output.hidden_states), std::move(after[1])};
}

Qwen35SelfAttention* Qwen35DecoderLayer::self_attention() noexcept {
  return self_attn_.get();
}

GatedDeltaNet* Qwen35DecoderLayer::linear_attention() noexcept {
  return linear_attn_.get();
}

MoE& Qwen35DecoderLayer::mlp() noexcept {
  return mlp_;
}

const MoE& Qwen35DecoderLayer::mlp() const noexcept {
  return mlp_;
}

Qwen35Model::Qwen35Model(const ModelConfig& config, Device device,
                         std::size_t cpu_threads,
                         const ExpertPlacement& expert_placement,
                         bool profile_experts,
                         ExpertAdaptationParams expert_adaptation,
                         bool cpu_affinity)
    : cpu_pool_(cpu_threads, cpu_affinity),
      embed_tokens_(config.vocab_size, config.hidden_size, DataType::BFloat16,
                    device),
      norm_(config.hidden_size, config.rms_norm_eps, DataType::BFloat16, device,
            true),
      expert_adaptation_(config.num_hidden_layers, config.num_experts,
                         expert_adaptation) {
  register_module("embed_tokens", embed_tokens_);
  register_module("layers", layers_);
  register_module("norm", norm_);

  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    layers_.emplace_back<Qwen35DecoderLayer>(
        config, i, device, expert_placement[i], cpu_pool_, moe_workspace_);
  }

  if (profile_experts) {
    expert_profile_.resize(config.num_hidden_layers);
  }
}

void Qwen35Model::build_rope_cache(cudaStream_t stream) {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    if (auto* attention = layer(i).self_attention()) {
      attention->build_rope_cache(stream);
    }
  }
}

void Qwen35Model::reset_expert_profile() {
  for (auto& layer : expert_profile_) {
    layer = {};
  }
  adaptation_stats_ = {};
  expert_adaptation_.reset_assignment_stats();
}

const std::vector<MoELayerProfile>& Qwen35Model::expert_profile()
    const noexcept {
  return expert_profile_;
}

ExpertAdaptationStats Qwen35Model::expert_adaptation_stats() const {
  auto stats = adaptation_stats_;
  if (expert_adaptation_.enabled()) {
    expert_adaptation_.add_assignment_stats(stats);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const auto ids = layer(i).mlp().expert_cache().resident_ids();
      stats.resident_ids_by_layer.emplace_back(ids.begin(), ids.end());
    }
  }
  return stats;
}

Tensor Qwen35Model::forward(ConstTensorView input_ids,
                            ConstTensorView positions,
                            const ForwardContext& context) const {
  Tensor hidden = embed_tokens_.forward(input_ids, context.stream);

  Tensor residual;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    std::optional<ConstTensorView> residual_view = std::nullopt;
    if (i > 0) {
      residual_view = residual.view();
    }

    auto* profile = expert_profile_.empty() ? nullptr
                    : context.is_prefill    ? &expert_profile_[i].prefill
                                            : &expert_profile_[i].decode;
    auto* usage = context.is_prefill ? nullptr : expert_adaptation_.usage(i);

    auto output = layer(i).forward(positions, hidden.view(), residual_view,
                                   context, profile, usage);

    hidden = std::move(output.hidden_states);
    residual = std::move(output.residual);
  }

  if (!context.is_prefill && !context.decode_graphs && !context.whole_graph) {
    adapt_experts(context.stream);
  }

  auto normalized =
      norm_.forward(hidden.view(), residual.view(), context.stream);
  return std::move(normalized.hidden_states);
}

void Qwen35Model::finish_decode_step(cudaStream_t stream) const {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).mlp().check_async_error();
  }
  adapt_experts(stream);
}

void Qwen35Model::reset_hybrid_step() const {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).mlp().reset_hybrid();
  }
}

void Qwen35Model::run_hybrid_cpu() const {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).mlp().complete_hybrid_cpu();
  }
}

void Qwen35Model::abort_hybrid_step() const noexcept {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layer(i).mlp().abort_hybrid();
  }
}

void Qwen35Model::adapt_experts(cudaStream_t stream) const {
  if (expert_adaptation_.advance()) {
    const auto started = std::chrono::steady_clock::now();
    ExpertPlacement placement;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const auto ids = layer(i).mlp().expert_cache().resident_ids();
      placement.emplace_back(ids.begin(), ids.end());
    }
    const auto swaps = expert_adaptation_.plan(placement);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      std::vector<ExpertCacheReplacement> replacements;
      for (const auto& swap : swaps) {
        if (swap.layer == i) {
          replacements.push_back({swap.slot, swap.new_id});
        }
      }
      if (!replacements.empty()) {
        adaptation_stats_.bytes_uploaded +=
            layer(i).mlp().replace_experts(replacements, stream);
        adaptation_stats_.swaps += replacements.size();
      }
    }
    expert_adaptation_.decay();
    ++adaptation_stats_.rounds;
    adaptation_stats_.seconds += std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();
  }
}

void Qwen35Model::set_cache(Qwen35Cache& cache) {
  std::size_t full = 0, linear = 0;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    if (auto* attention = layer(i).self_attention()) {
      attention->set_kv_cache(cache.key_cache(full), cache.value_cache(full));
      ++full;
    } else {
      layer(i).linear_attention()->set_state_cache(
          cache.conv_state(linear), cache.recurrent_state(linear));
      ++linear;
    }
  }
}

void Qwen35Model::clear_cache() noexcept {
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    if (auto* attention = layer(i).self_attention()) {
      attention->clear_kv_cache();
    } else {
      layer(i).linear_attention()->clear_state_cache();
    }
  }
}

Embedding& Qwen35Model::embed_tokens() noexcept {
  return embed_tokens_;
}

std::size_t Qwen35Model::num_layers() const noexcept {
  return layers_.size();
}

Qwen35DecoderLayer& Qwen35Model::layer(std::size_t i) {
  return static_cast<Qwen35DecoderLayer&>(layers_.at(i));
}

const Qwen35DecoderLayer& Qwen35Model::layer(std::size_t i) const {
  return static_cast<const Qwen35DecoderLayer&>(layers_.at(i));
}

Qwen35ForCausalLM::Qwen35ForCausalLM(
    const ModelConfig& config, Device device, std::size_t cpu_threads,
    const ExpertPlacement& gpu_expert_ids_by_layer, bool profile_experts,
    ExpertAdaptationParams expert_adaptation, bool cpu_affinity)
    : config_(config),
      device_(device),
      model_(config_, device_, cpu_threads, gpu_expert_ids_by_layer,
             profile_experts, expert_adaptation, cpu_affinity),
      lm_head_(
          config_.vocab_size, config_.hidden_size, DataType::BFloat16, device_,
          config_.tie_word_embeddings ? &model_.embed_tokens().weight().tensor()
                                      : nullptr) {
  register_module("model", model_);
  register_module("lm_head", lm_head_);
}

ModelLoadResult Qwen35ForCausalLM::load_ckpt(
    const std::filesystem::path& path) {
  Checkpoint checkpoint(path);
  LoadPlan plan(checkpoint, config_);

  plan.add_module(
      model_, config_.architectures[0] == "Qwen3_5MoeForConditionalGeneration"
                  ? "model.language_model"
                  : "model");
  if (!config_.tie_word_embeddings
      || checkpoint.weight_map().contains("lm_head.weight")) {
    plan.add_module(lm_head_, "lm_head");
  }

  // 清除两种 attention cache
  model_.clear_cache();

  // 清除每层 MoE 的 CUDA Graph，并且清除 Expert 状态
  for (std::size_t i = 0; i < model_.num_layers(); ++i) {
    model_.layer(i).mlp().invalidate_expert_cache();
  }

  // 加载权重到 CPU
  const auto result = plan.apply();

  // 把需要传递到 GPU 上的专家全部上传到 GPU 上
  if (device_.type == DeviceType::Cuda) {
    for (std::size_t i = 0; i < model_.num_layers(); ++i) {
      model_.layer(i).mlp().populate_expert_cache();
    }
  }

  return result;
}

Tensor Qwen35ForCausalLM::forward(ConstTensorView input_ids,
                                  ConstTensorView positions,
                                  const ForwardContext& context) const {
  return model_.forward(input_ids, positions, context);
}

Tensor Qwen35ForCausalLM::compute_logits(ConstTensorView hidden_states,
                                         const ForwardContext& context) const {
  if (context.is_prefill) {
    return lm_head_.forward(hidden_states, context.cu_seqlens_q, context.stream,
                            DataType::Float32);
  } else {
    return lm_head_.forward(hidden_states, context.stream, DataType::Float32);
  }
}

const ModelConfig& Qwen35ForCausalLM::config() const noexcept {
  return config_;
}

DataType Qwen35ForCausalLM::dtype() const noexcept {
  return DataType::BFloat16;
}

Device Qwen35ForCausalLM::device() const noexcept {
  return device_;
}

Qwen35Model& Qwen35ForCausalLM::transformer() noexcept {
  return model_;
}

LMHead& Qwen35ForCausalLM::lm_head() noexcept {
  return lm_head_;
}

} // namespace nanovllm
