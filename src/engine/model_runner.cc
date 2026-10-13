#include "engine/model_runner.hh"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "common/cuda_utils.hh"
#include "core/device.hh"
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

  const CudaDeviceGuard guard(0);
  check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
             "create inference stream");
  try {
    // load model ckpt
    if (config_.architecture == ModelArchitecture::Qwen35Moe) {
      model_ = std::make_unique<Qwen35ForCausalLM>(config_, Device::cuda(0),
                                                   params_.gpu_expert_ids,
                                                   params_.cpu_threads);
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

std::vector<std::int32_t> ModelRunner::run(
    std::span<const Sequence* const> seqs, bool prefill) {
  if (seqs.empty()) {
    return {};
  }
  if (seqs.size() > params_.max_num_seqs) {
    throw std::invalid_argument("inference batch exceeds max_num_seqs");
  }

  const CudaDeviceGuard guard(0);

  std::vector<std::int32_t> ids, positions, slots, cu_q{0}, cu_k{0}, lengths;
  std::vector<float> temperatures;
  std::vector<std::int32_t> state_ids;
  std::size_t max_q = 0, max_k = 0, table_width = 0;

  bool needs_cached_prefill = false;
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

    if (start != 0) {
      needs_cached_prefill = true;
    }

    max_q = std::max(max_q, count);
    max_k = std::max(max_k, end);
    table_width = std::max(table_width, seq->block_ids.size());

    for (auto pos = start; pos < end; ++pos) {
      const auto token = seq->token_id(pos);
      if (token < 0 || static_cast<std::size_t>(token) >= config_.vocab_size) {
        throw std::invalid_argument("token ID is outside the model vocabulary");
      }

      const auto block = seq->block_ids.at(pos / params_.kvcache_block_size);
      if (block >= num_blocks_) {
        throw std::invalid_argument("invalid KV block ID");
      }

      ids.push_back(token);
      positions.push_back(pos);
      slots.push_back(block * params_.kvcache_block_size
                      + pos % params_.kvcache_block_size);
    }

    cu_q.push_back(ids.size());
    cu_k.push_back(cu_k.back() + end);
    lengths.push_back(end);
    temperatures.push_back(seq->sampling_params.temperature);
  }

  if (ids.size() > params_.max_num_batched_tokens) {
    throw std::invalid_argument(
        "scheduled batch exceeds max_num_batched_tokens");
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

    auto tensor = Tensor::empty(std::move(shape), dtype, model_->device());
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
  auto slot_mapping = upload(slots, {slots.size()});
  auto boundaries_q = upload(cu_q, {cu_q.size()});
  auto boundaries_k = upload(cu_k, {cu_k.size()});
  auto context_lens = upload(lengths, {lengths.size()});
  auto block_tables = upload(tables, {seqs.size(), table_width});
  auto state_indices = upload(state_ids, {state_ids.size()});
  auto temperature = upload(temperatures, {seqs.size()});

  ForwardContext context;
  context.is_prefill = prefill;
  context.cu_seqlens_q = boundaries_q.view().as_const();
  context.cu_seqlens_k = boundaries_k.view().as_const();
  context.max_seqlen_q = max_q;
  context.max_seqlen_k = max_k;
  context.slot_mapping = slot_mapping.view().as_const();
  context.context_lens = context_lens.view().as_const();
  if (!prefill || needs_cached_prefill) {
    context.block_tables = block_tables.view().as_const();
  }
  context.state_indices = state_indices.view().as_const();
  context.stream = stream_;

  if (hybrid_cache_ && prefill) {
    for (const auto* seq : seqs) {
      if (seq->num_cached_tokens == 0) {
        hybrid_cache_->reset_state(*seq->state_slot, stream_);
      }
    }
  }

  auto hidden = model_->forward(input_ids.view().as_const(),
                                input_positions.view().as_const(), context);
  auto logits = model_->compute_logits(hidden.view().as_const(), context);

  auto output = Tensor::empty({seqs.size()}, DataType::Int32, model_->device());
  kernels::cuda::sample(logits.view().as_const(), temperature.view().as_const(),
                        output.view(), params_.seed, sample_step_++, stream_);

  std::vector<std::int32_t> result(seqs.size());
  check_cuda(cudaMemcpyAsync(result.data(), output.data(), output.size_bytes(),
                             cudaMemcpyDeviceToHost, stream_),
             "download sampled tokens");
  check_cuda(cudaStreamSynchronize(stream_), "complete inference step");
  return result;
}

} // namespace nanovllm
