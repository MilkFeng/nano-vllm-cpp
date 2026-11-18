#include "engine/model_runner.hh"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "common/cuda_utils.hh"
#include "core/device.hh"
#include "core/expert_policy.hh"
#include "core/forward_context.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {

ModelRunner::ModelRunner(std::shared_ptr<Logger> logger, EngineParams params,
                         ModelConfig config,
                         const std::filesystem::path& checkpoint)
    : logger_(logger),
      params_(params),
      config_(std::move(config)),
      num_blocks_(params.num_kvcache_blocks) {
  if (params_.max_num_seqs == 0 || params_.max_num_batched_tokens == 0
      || params_.max_model_len == 0 || params_.kvcache_block_size == 0
      || num_blocks_ == 0
      || params_.max_model_len > config_.max_position_embeddings) {
    throw std::invalid_argument(
        "invalid engine limits or KV cache block count");
  }

  const auto per_sequence =
      (params_.max_model_len - 1) / params_.kvcache_block_size + 1;
  if (num_blocks_ < per_sequence) {
    throw std::invalid_argument(
        "KV cache cannot hold max_model_len; increase num_kvcache_blocks "
        "or reduce max_model_len");
  }

  // Reject malformed lists before allocating model weights or CUDA resources.
  const auto placement =
      resolve_expert_placement(config_, params_.gpu_expert_ids);
  if (params_.profile_experts
      && config_.architecture != ModelArchitecture::Qwen35Moe) {
    throw std::invalid_argument("expert profiling requires Qwen3.5 MoE");
  }

  const CudaDeviceGuard guard(0);
  check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
             "create inference stream");
  try {
    temporary_pool_ = std::make_unique<TensorPool>(Device::cuda(0), stream_);
    // load model ckpt
    if (config_.architecture == ModelArchitecture::Qwen35Moe) {
      model_ = std::make_unique<Qwen35ForCausalLM>(
          config_, Device::cuda(0), params_.cpu_threads, placement,
          params_.profile_experts, params_.expert_adaptation,
          params_.cpu_affinity);
    } else {
      model_ = std::make_unique<Qwen3ForCausalLM>(config_, Device::cuda(0));
    }

    model_->load_ckpt(checkpoint);
    check_cuda(cudaStreamSynchronize(nullptr), "complete model weight uploads");

    // K/V Cache
    if (config_.architecture == ModelArchitecture::Qwen35Moe) {
      hybrid_cache_ = std::make_unique<Qwen35Cache>(
          config_, num_blocks_, params_.kvcache_block_size,
          params_.max_num_seqs, model_->device());
      static_cast<Qwen35ForCausalLM&>(*model_).transformer().set_cache(
          *hybrid_cache_);
    } else {
      qwen3_cache_ = std::make_unique<Qwen3Cache>(
          config_, num_blocks_, params_.kvcache_block_size, model_->device());
      static_cast<Qwen3ForCausalLM&>(*model_).transformer().set_cache(
          *qwen3_cache_);
    }
    const auto workspace_elements =
        kernels::cuda::flash_attn_workspace_elements(
            params_.max_num_seqs, config_.num_attention_heads, config_.head_dim,
            params_.max_model_len);
    if (workspace_elements > 0) {
      attention_workspace_ = Tensor::empty({workspace_elements},
                                           DataType::Float32, model_->device());
    }
  } catch (...) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
    throw;
  }

  logger_->info(std::format(
      "Model Runner initialized with {} KV cache blocks ({} bytes for K and V)",
      params_.num_kvcache_blocks, kvcache_size_bytes()));
}

ModelRunner::~ModelRunner() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

std::size_t ModelRunner::kvcache_size_bytes() const {
  return hybrid_cache_ ? hybrid_cache_->kv_bytes() : qwen3_cache_->kv_bytes();
}

std::size_t ModelRunner::state_cache_size_bytes() const {
  return hybrid_cache_ ? hybrid_cache_->state_bytes() : 0;
}

void ModelRunner::reset_expert_profile() {
  if (config_.architecture == ModelArchitecture::Qwen35Moe) {
    static_cast<Qwen35ForCausalLM&>(*model_)
        .transformer()
        .reset_expert_profile();
  }
}

ExpertAdaptationStats ModelRunner::expert_adaptation_stats() const {
  return params_.expert_adaptation.every != 0
             ? static_cast<Qwen35ForCausalLM&>(*model_)
                   .transformer()
                   .expert_adaptation_stats()
             : ExpertAdaptationStats{};
}

std::vector<MoELayerProfile> ModelRunner::expert_profile() const {
  return params_.profile_experts ? static_cast<Qwen35ForCausalLM&>(*model_)
                                       .transformer()
                                       .expert_profile()
                                 : std::vector<MoELayerProfile>{};
}

std::vector<std::int32_t> ModelRunner::run(
    std::span<const Sequence* const> seqs, bool prefill) {
  if (seqs.empty()) {
    return {};
  }
  if (seqs.size() > params_.max_num_seqs) {
    throw std::invalid_argument("inference batch exceeds max_num_seqs");
  }

  const CudaDeviceGuard guard(0);

  std::vector<std::int32_t> ids, positions, cu_q{0}, cu_k{0};
  std::vector<float> temperatures;
  std::vector<std::int32_t> state_ids;
  std::size_t max_q = 0, max_k = 0, table_width = 0;

  for (const auto* seq : seqs) {
    if (seq == nullptr || seq->num_tokens() > params_.max_model_len) {
      throw std::invalid_argument("invalid scheduled sequence");
    }

    if (hybrid_cache_) {
      if (!seq->state_slot || *seq->state_slot >= params_.max_num_seqs) {
        throw std::invalid_argument(
            "Qwen3.5 batch needs distinct valid sequence state slots");
      }

      state_ids.push_back(*seq->state_slot);
    }

    const auto start = seq->num_cached_tokens;
    const auto count = seq->num_scheduled_tokens;
    const auto end = start + count;

    if (end > seq->block_ids.size() * params_.kvcache_block_size) {
      throw std::invalid_argument(
          "scheduled tokens exceed allocated KV blocks");
    }

    max_q = std::max(max_q, count);
    max_k = std::max(max_k, end);
    table_width = std::max(table_width, seq->block_ids.size());

    for (auto pos = start; pos < end; ++pos) {
      const auto token = seq->token_id(pos);
      if (token < 0 || static_cast<std::size_t>(token) >= config_.vocab_size) {
        throw std::invalid_argument("token ID is outside the model vocabulary");
      }

      ids.push_back(token);
      positions.push_back(pos);
    }

    cu_q.push_back(ids.size());
    cu_k.push_back(cu_k.back() + end);
    temperatures.push_back(seq->sampling_params.temperature);
  }

  if (ids.size() > params_.max_num_batched_tokens) {
    throw std::invalid_argument(
        "scheduled batch exceeds max_num_batched_tokens");
  }

  static const bool decode_graphs =
      std::getenv("NANOVLLM_DISABLE_DECODE_GRAPHS") == nullptr;
  static const bool whole_graph =
      std::getenv("NANOVLLM_DISABLE_WHOLE_GRAPH") == nullptr;
  const bool use_whole = decode_graphs && whole_graph && !prefill
                         && seqs.size() == 1 && ids.size() == 1
                         && config_.architecture == ModelArchitecture::Qwen35Moe
                         && config_.num_experts_per_tok <= 8
                         && !params_.profile_experts && temperatures[0] == 0.0f;
  if (use_whole) {
    table_width = (params_.max_model_len + params_.kvcache_block_size - 1)
                  / params_.kvcache_block_size;
  }
  std::vector<std::int32_t> tables(seqs.size() * table_width, -1);
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    for (std::size_t j = 0; j < seqs[i]->block_ids.size(); ++j) {
      const auto block = seqs[i]->block_ids[j];
      if (block >= num_blocks_) {
        throw std::invalid_argument("invalid KV block table");
      }
      tables[i * table_width + j] = block;
    }
  }

  // Host upload sources and the readback destination must outlive the scope's
  // error-path synchronization. Normal steps synchronize explicitly below.
  std::vector<std::int32_t> result(seqs.size());
  TensorPool::Scope temporary_scope(*temporary_pool_, stream_);

  std::size_t metadata_index = 0;
  const auto upload = [&]<typename T>(const std::vector<T>& data,
                                      std::vector<std::size_t> shape) {
    DataType dtype;
    if constexpr (std::is_same_v<T, float>) {
      dtype = DataType::Float32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      dtype = DataType::Int32;
    } else {
      static_assert(false);
    }

    Tensor tensor;
    if (use_whole) {
      auto& cached = whole_metadata_[metadata_index++];
      if (cached.numel() == 0) {
        cached = Tensor::empty(std::move(shape), dtype, model_->device());
      }
      tensor = cached;
    } else {
      tensor = Tensor::empty(std::move(shape), dtype, model_->device());
    }
    if (!data.empty()) {
      check_cuda(
          cudaMemcpyAsync(tensor.data(), data.data(), tensor.size_bytes(),
                          cudaMemcpyHostToDevice, stream_),
          "upload inference metadata");
    }
    return tensor;
  };

  auto input_ids = upload(ids, {ids.size()});
  auto input_positions = upload(positions, {positions.size()});
  auto boundaries_q = upload(cu_q, {cu_q.size()});
  auto boundaries_k = upload(cu_k, {cu_k.size()});
  auto block_tables = upload(tables, {seqs.size(), table_width});
  auto state_indices = upload(state_ids, {state_ids.size()});
  auto temperature = upload(temperatures, {seqs.size()});

  ForwardContext context;
  context.is_prefill = prefill;
  context.decode_graphs =
      decode_graphs && !prefill && ids.size() == 1 && !use_whole;
  context.whole_graph = use_whole;
  context.cu_seqlens_q = boundaries_q.view().as_const();
  context.cu_seqlens_k = boundaries_k.view().as_const();
  context.max_seqlen_q = max_q;
  context.max_seqlen_k = use_whole ? params_.max_model_len : max_k;
  context.block_tables = block_tables.view().as_const();
  context.state_indices = state_indices.view().as_const();
  context.stream = stream_;
  context.attention_workspace = attention_workspace_.view();

  if (hybrid_cache_ && prefill) {
    for (const auto* seq : seqs) {
      if (seq->num_cached_tokens == 0) {
        hybrid_cache_->reset_state(*seq->state_slot, stream_);
      }
    }
  }

  Tensor output;
  if (use_whole) {
    auto& transformer = static_cast<Qwen35ForCausalLM&>(*model_).transformer();
    if (!whole_decode_.ready()) {
      whole_states_.clear();
      std::size_t linear = 0;
      for (const auto type : config_.layer_types) {
        if (type == AttentionType::Linear) {
          whole_states_.push_back(
              hybrid_cache_->conv_state(linear).slice(0, state_ids[0], 1));
          whole_states_.push_back(
              hybrid_cache_->recurrent_state(linear++).slice(0, state_ids[0],
                                                             1));
        }
      }
      // Decode writes only the current, fresh KV slot. The actual replay
      // overwrites it; no cached prefix KV is changed by graph warmup.
    }
    const auto function = [&](bool capturing) -> std::vector<Tensor> {
      auto captured_context = context;
      captured_context.graph_warmup = !capturing;
      auto hidden =
          model_->forward(input_ids.view().as_const(),
                          input_positions.view().as_const(), captured_context);
      auto logits =
          model_->compute_logits(hidden.view().as_const(), captured_context);
      auto token = Tensor::empty({1}, DataType::Int32, model_->device());
      kernels::cuda::sample(logits.view().as_const(),
                            temperature.view().as_const(), token.view(),
                            params_.seed, 0, stream_);
      return {std::move(token)};
    };
    auto tensors = whole_decode_.run(
        stream_, {input_ids.data(), block_tables.data(), state_indices.data()},
        whole_states_, function, [&] { transformer.reset_hybrid_step(); },
        [&] { transformer.run_hybrid_cpu(stream_); },
        [&] { transformer.abort_hybrid_step(); });
    output = std::move(tensors[0]);
    ++sample_step_;
  } else {
    auto hidden = model_->forward(input_ids.view().as_const(),
                                  input_positions.view().as_const(), context);
    auto logits = model_->compute_logits(hidden.view().as_const(), context);

    output = Tensor::empty({seqs.size()}, DataType::Int32, model_->device());
    kernels::cuda::sample(logits.view().as_const(),
                          temperature.view().as_const(), output.view(),
                          params_.seed, sample_step_++, stream_);
  }

  check_cuda(cudaMemcpyAsync(result.data(), output.data(), output.size_bytes(),
                             cudaMemcpyDeviceToHost, stream_),
             "download sampled tokens");
  check_cuda(cudaStreamSynchronize(stream_), "complete inference step");
  if ((context.decode_graphs || use_whole)
      && config_.architecture == ModelArchitecture::Qwen35Moe) {
    static_cast<Qwen35ForCausalLM&>(*model_).transformer().finish_decode_step(
        stream_);
    if (params_.expert_adaptation.every != 0) {
      check_cuda(cudaStreamSynchronize(stream_),
                 "complete graph expert replacements");
    }
  }
  return result;
}

} // namespace nanovllm
