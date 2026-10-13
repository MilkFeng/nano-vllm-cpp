#include "models/qwen35.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/checkpoint.hh"
#include "core/device.hh"
#include "kernels/cpu/fp8_dequant.hh"
#include "kernels/cuda/attention.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {
namespace {

void copy_float(const Checkpoint& checkpoint, std::string_view name,
                TensorView output) {
  const auto& source = checkpoint.tensor(name);
  const auto bytes = checkpoint.raw_data(name);
  if (source.dtype == DataType::Float32) {
    std::memcpy(output.data(), bytes.data(), bytes.size());
    return;
  }
  auto* result = output.data<float>();
  for (std::size_t i = 0; i < output.numel(); ++i) {
    std::uint16_t bits;
    std::memcpy(&bits, bytes.data() + i * sizeof(bits), sizeof(bits));
    result[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
  }
}

struct LoadEntry {
  std::string name;
  TensorView destination;
  std::string scale_name;
};

struct LoadPlan {
  std::vector<LoadEntry> entries;
  std::set<std::string, std::less<>> sources;
  ModelLoadResult result;

  void count(const Checkpoint& checkpoint, const std::string& name) {
    if (!sources.insert(name).second) {
      return;
    }
    const auto bytes = checkpoint.tensor(name).byte_size();
    ++result.num_tensors;
    result.num_bytes += bytes;
  }
};

LoadPlan plan_load(Module& module, const Checkpoint& checkpoint,
                   std::string_view prefix) {
  std::string stem(prefix);
  if (!stem.empty() && !stem.ends_with('.')) {
    stem += '.';
  }

  LoadPlan plan;
  for (const auto& [name, parameter] : module.named_parameters()) {
    LoadEntry entry{stem + name, parameter->view(), {}};

    const auto destination = entry.destination;
    const auto& source = checkpoint.tensor(entry.name);

    if (!std::ranges::equal(source.shape, destination.shape())) {
      throw std::runtime_error(
          std::format("tensor '{}' has an unexpected shape", entry.name));
    }

    // FP8 -> CPU FP8
    // FP8 - dequant -> FP32
    // BF16/FP32 -> FP32

    if (destination.dtype() == DataType::Float8E4M3FN) {
      assert(source.dtype == DataType::Float8E4M3FN);
      assert(destination.device().type == DeviceType::Cpu);
    } else if (source.dtype == DataType::Float8E4M3FN) {
      if (source.shape.size() != 2 || !entry.name.ends_with(".weight")) {
        throw std::runtime_error("FP8 expansion needs a 2D .weight tensor");
      }
      entry.scale_name = entry.name + "_scale_inv";
      plan.count(checkpoint, entry.scale_name);
    } else if (source.dtype != DataType::BFloat16
               && source.dtype != DataType::Float32) {
      throw std::runtime_error(std::format(
          "tensor '{}' must be BF16, F32 or block F8_E4M3", entry.name));
    }

    plan.count(checkpoint, entry.name);
    plan.entries.push_back(std::move(entry));
  }
  return plan;
}

void fill_cpu(const Checkpoint& checkpoint, const LoadEntry& entry,
              TensorView output, const ModelConfig& config) {
  if (!entry.scale_name.empty()) {
    auto scale = Tensor::empty(checkpoint.tensor(entry.scale_name).shape,
                               DataType::Float32);
    copy_float(checkpoint, entry.scale_name, scale.view());

    const auto blocks = config.quantization_config->weight_block_size;
    const auto& info = checkpoint.tensor(entry.name);
    ConstTensorView weight(checkpoint.raw_data(entry.name).data(), info.dtype,
                           TensorLayout(info.shape), Device::cpu());
    kernels::cpu::dequantize_fp8(weight, scale.view().as_const(), output,
                                 blocks[0], blocks[1]);
  } else if (output.dtype() == DataType::Float8E4M3FN) {
    const auto bytes = checkpoint.raw_data(entry.name);
    std::memcpy(output.data(), bytes.data(), bytes.size());
  } else {
    copy_float(checkpoint, entry.name, output);
  }
}

void apply_load(const LoadPlan& plan, const ModelConfig& config,
                const Checkpoint& checkpoint) {
  for (const auto& entry : plan.entries) {
    if (entry.destination.device().type == DeviceType::Cpu) {
      fill_cpu(checkpoint, entry, entry.destination, config);
    } else {
      auto staging = Tensor::empty(
          {entry.destination.shape().begin(), entry.destination.shape().end()},
          entry.destination.dtype());
      fill_cpu(checkpoint, entry, staging.view(), config);

      const CudaDeviceGuard guard(entry.destination.device().index);
      check_cuda(cudaMemcpy(entry.destination.data(), staging.data(),
                            staging.size_bytes(), cudaMemcpyHostToDevice),
                 std::format("load Qwen3.5 tensor '{}'", entry.name));
    }
  }
}

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

void zero_tensor(TensorView tensor, cudaStream_t stream) {
  if (tensor.size_bytes() == 0) {
    return;
  }
  if (tensor.device().type == DeviceType::Cpu) {
    std::memset(tensor.data(), 0, tensor.size_bytes());
  } else {
    const CudaDeviceGuard guard(tensor.device().index);
    check_cuda(cudaMemsetAsync(tensor.data(), 0, tensor.size_bytes(), stream),
               "reset Qwen3.5 state");
  }
}

} // namespace

Qwen35Attention::Qwen35Attention(const ModelConfig& config, Device device)
    : hidden_size_(config.hidden_size),
      num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      q_size_(config.q_size()),
      scaling_(1.0f / std::sqrt(float(head_dim_))),
      q_proj_(hidden_size_, 2 * q_size_, config.attention_bias,
              DataType::Float32, device),
      k_proj_(hidden_size_, config.kv_size(), config.attention_bias,
              DataType::Float32, device),
      v_proj_(hidden_size_, config.kv_size(), config.attention_bias,
              DataType::Float32, device),
      o_proj_(q_size_, hidden_size_, config.attention_bias, DataType::Float32,
              device),
      q_norm_(head_dim_, config.rms_norm_eps, DataType::Float32, device, true),
      k_norm_(head_dim_, config.rms_norm_eps, DataType::Float32, device, true),
      rotary_emb_(config.max_position_embeddings, head_dim_, config.rope_theta,
                  device, config.rotary_dim()) {
  register_module("q_proj", q_proj_);
  register_module("k_proj", k_proj_);
  register_module("v_proj", v_proj_);
  register_module("o_proj", o_proj_);
  register_module("q_norm", q_norm_);
  register_module("k_norm", k_norm_);
}

Tensor Qwen35Attention::forward(ConstTensorView positions,
                                ConstTensorView hidden_states,
                                const ForwardContext& context) const {
  const auto tokens = hidden_states.shape()[0];
  const auto stream = context.stream;
  const auto device = hidden_states.device();
  const bool cached = key_cache_.numel() != 0;
  const bool paged = context.block_tables.numel() != 0;

  auto packed = q_proj_.forward(hidden_states, stream);
  auto query_heads =
      Tensor::empty({tokens, num_heads_, head_dim_}, DataType::Float32, device);
  auto gate =
      Tensor::empty({tokens, num_heads_, head_dim_}, DataType::Float32, device);

  kernels::cuda::split_attention_gate(packed.view().as_const(),
                                      query_heads.view(), gate.view(), stream);

  auto k = k_proj_.forward(hidden_states, stream);
  auto v = v_proj_.forward(hidden_states, stream);

  const auto key_heads =
      k.reshape(std::array{tokens, num_kv_heads_, head_dim_});
  const auto value = v.reshape(std::array{tokens, num_kv_heads_, head_dim_});

  auto query = q_norm_.forward(query_heads.view().as_const(), stream);
  auto key = k_norm_.forward(key_heads.view(), stream);

  rotary_emb_.forward(positions, query.view(), key.view(), stream);
  if (cached) {
    kernels::cuda::store_kvcache(key.view().as_const(), value.view(),
                                 key_cache_, value_cache_, context.slot_mapping,
                                 stream);
  }
  auto output =
      Tensor::empty({tokens, num_heads_, head_dim_}, DataType::Float32, device);

  if (context.is_prefill) {
    kernels::cuda::prefill_attention(
        query.view().as_const(), paged ? key_cache() : key.view().as_const(),
        paged ? value_cache() : value.view(), context.cu_seqlens_q,
        context.cu_seqlens_k, context.block_tables, output.view(), scaling_,
        stream);
  } else {
    kernels::cuda::paged_attention_decode(query.view().as_const(), key_cache(),
                                          value_cache(), context.context_lens,
                                          context.block_tables, output.view(),
                                          scaling_, stream);
  }

  kernels::cuda::apply_sigmoid_gate(output.view(), gate.view().as_const(),
                                    stream);

  const auto flattened = output.reshape(std::array{tokens, q_size_});
  return o_proj_.forward(flattened.view(), stream);
}

void Qwen35Attention::set_kv_cache(TensorView key, TensorView value) {
  key_cache_ = key;
  value_cache_ = value;
}

void Qwen35Attention::clear_kv_cache() noexcept {
  key_cache_ = {};
  value_cache_ = {};
}

ConstTensorView Qwen35Attention::key_cache() const {
  return ConstTensorView(key_cache_);
}

ConstTensorView Qwen35Attention::value_cache() const {
  return ConstTensorView(value_cache_);
}

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
                       DataType::Float32, device);
  value_ = Tensor::empty({full, num_blocks, block_size,
                          config.num_key_value_heads, config.head_dim},
                         DataType::Float32, device);
  conv_ = Tensor::empty(
      {linear, state_slots, conv_dim, config.linear_conv_kernel_dim - 1},
      DataType::Float32, device);
  recurrent_ =
      Tensor::empty({linear, state_slots, config.linear_num_value_heads,
                     config.linear_key_head_dim, config.linear_value_head_dim},
                    DataType::Float32, device);

  zero_tensor(conv_.view(), nullptr);
  zero_tensor(recurrent_.view(), nullptr);

  if (device.type == DeviceType::Cuda) {
    const CudaDeviceGuard guard(device.index);
    check_cuda(cudaStreamSynchronize(nullptr), "initialize Qwen3.5 cache");
  }
}

void Qwen35Cache::reset_state(std::size_t slot, cudaStream_t stream) {
  for (std::size_t i = 0; i < recurrent_.shape()[0]; ++i) {
    zero_tensor(conv_state(i).slice(0, slot, 1), stream);
    zero_tensor(recurrent_state(i).slice(0, slot, 1), stream);
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
    const std::vector<std::int32_t>& gpu_expert_ids, ThreadPool& cpu_pool)
    : mlp_(qwen35_moe_config(config), cpu_pool, device, gpu_expert_ids),
      input_layernorm_(config.hidden_size, config.rms_norm_eps,
                       DataType::Float32, device, true),
      post_attention_layernorm_(config.hidden_size, config.rms_norm_eps,
                                DataType::Float32, device, true) {
  if (config.layer_types.at(index) == AttentionType::Full) {
    self_attn_ = std::make_unique<Qwen35Attention>(config, device);
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
    std::optional<ConstTensorView> residual,
    const ForwardContext& context) const {
  RMSNormOutput before;
  auto residual_view = hidden_states;
  if (residual) {
    before = input_layernorm_.forward(hidden_states, *residual, context.stream);
    residual_view = before.residual.view().as_const();
  } else {
    before.hidden_states =
        input_layernorm_.forward(hidden_states, context.stream);
  }

  auto attention =
      self_attn_
          ? self_attn_->forward(positions,
                                before.hidden_states.view().as_const(), context)
          : linear_attn_->forward(before.hidden_states.view().as_const(),
                                  context);
  auto after = post_attention_layernorm_.forward(attention.view().as_const(),
                                                 residual_view, context.stream);
  auto output =
      mlp_.forward(after.hidden_states.view().as_const(), context.stream);

  return {std::move(output.hidden_states), std::move(after.residual)};
}

Qwen35Attention* Qwen35DecoderLayer::self_attention() noexcept {
  return self_attn_.get();
}

GatedDeltaNet* Qwen35DecoderLayer::linear_attention() noexcept {
  return linear_attn_.get();
}

MoE& Qwen35DecoderLayer::mlp() noexcept {
  return mlp_;
}

Qwen35Model::Qwen35Model(const ModelConfig& config, Device device,
                         const std::vector<std::int32_t>& gpu_expert_ids,
                         std::size_t cpu_threads)
    : cpu_pool_(cpu_threads),
      embed_tokens_(config.vocab_size, config.hidden_size, DataType::Float32,
                    device),
      norm_(config.hidden_size, config.rms_norm_eps, DataType::Float32, device,
            true) {
  register_module("embed_tokens", embed_tokens_);
  register_module("layers", layers_);
  register_module("norm", norm_);
  for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
    layers_.emplace_back<Qwen35DecoderLayer>(config, i, device, gpu_expert_ids,
                                             cpu_pool_);
  }
}

Tensor Qwen35Model::forward(ConstTensorView input_ids,
                            ConstTensorView positions,
                            const ForwardContext& context) const {
  auto hidden = embed_tokens_.forward(input_ids, context.stream);
  Tensor residual;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    std::optional<ConstTensorView> residual_view = std::nullopt;
    if (i > 0) {
      residual_view = residual.view().as_const();
    }

    auto output = layer(i).forward(positions, hidden.view().as_const(),
                                   residual_view, context);

    hidden = std::move(output.hidden_states);
    residual = std::move(output.residual);
  }

  auto normalized = norm_.forward(hidden.view().as_const(),
                                  residual.view().as_const(), context.stream);
  return std::move(normalized.hidden_states);
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
    if (auto* attention = layer(i).self_attention())
      attention->clear_kv_cache();
    else
      layer(i).linear_attention()->clear_state_cache();
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

Qwen35ForCausalLM::Qwen35ForCausalLM(const ModelConfig& config, Device device,
                                     std::vector<std::int32_t> gpu_expert_ids,
                                     std::size_t cpu_threads)
    : config_(config),
      device_(device),
      model_(config_, device_, gpu_expert_ids, cpu_threads),
      lm_head_(
          config_.vocab_size, config_.hidden_size, DataType::Float32, device_,
          config_.tie_word_embeddings ? &model_.embed_tokens().weight().tensor()
                                      : nullptr) {
  register_module("model", model_);
  register_module("lm_head", lm_head_);
}

ModelLoadResult Qwen35ForCausalLM::load_ckpt(
    const std::filesystem::path& path) {
  Checkpoint checkpoint(path);
  const auto prefix =
      config_.architectures[0] == "Qwen3_5MoeForConditionalGeneration"
          ? "model.language_model"
          : "model";
  auto plan = plan_load(model_, checkpoint, prefix);
  if (!config_.tie_word_embeddings
      || checkpoint.weight_map().contains("lm_head.weight")) {
    auto head = plan_load(lm_head_, checkpoint, "lm_head");

    for (const auto& name : head.sources) {
      plan.count(checkpoint, name);
    }

    for (auto& entry : head.entries) {
      plan.entries.push_back(std::move(entry));
    }
  }

  model_.clear_cache();
  for (std::size_t i = 0; i < model_.num_layers(); ++i) {
    model_.layer(i).mlp().invalidate_expert_cache();
  }

  apply_load(plan, config_, checkpoint);

  if (device_.type == DeviceType::Cuda) {
    for (std::size_t i = 0; i < model_.num_layers(); ++i) {
      model_.layer(i).mlp().populate_expert_cache();
    }
  }

  return plan.result;
}

Tensor Qwen35ForCausalLM::forward(ConstTensorView input_ids,
                                  ConstTensorView positions,
                                  const ForwardContext& context) const {
  return model_.forward(input_ids, positions, context);
}

Tensor Qwen35ForCausalLM::compute_logits(ConstTensorView hidden_states,
                                         const ForwardContext& context) const {
  if (context.is_prefill) {
    const auto batch = context.cu_seqlens_q.shape()[0] - 1;
    auto last =
        Tensor::empty({batch, config_.hidden_size}, DataType::Float32, device_);
    kernels::cuda::gather_last_hidden(hidden_states, context.cu_seqlens_q,
                                      last.view(), context.stream);
    return lm_head_.forward(last.view().as_const(), context.stream);
  }
  return lm_head_.forward(hidden_states, context.stream);
}

const ModelConfig& Qwen35ForCausalLM::config() const noexcept {
  return config_;
}

DataType Qwen35ForCausalLM::dtype() const noexcept {
  return DataType::Float32;
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
