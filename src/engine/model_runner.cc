#include "engine/model_runner.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/cuda_utils.hh"
#include "core/forward_context.hh"
#include "kernels/cuda/inference.hh"
#include "models/qwen3.hh"

namespace nanovllm {
namespace {

class DeviceGuard {
 public:
  DeviceGuard() {
    check_cuda(cudaGetDevice(&previous_), "get CUDA device");
    if (previous_ != 0) {
      check_cuda(cudaSetDevice(0), "select CUDA device 0");
    }
  }

  ~DeviceGuard() {
    if (previous_ != 0) {
      cudaSetDevice(previous_);
    }
  }

 private:
  int previous_ = 0;
};

} // namespace

ModelRunner::ModelRunner(EngineParams params, ModelConfig config,
                         const std::filesystem::path& checkpoint)
    : params_(params),
      config_(std::move(config)) {
  if (params.max_num_seqs == 0 || params.max_num_batched_tokens == 0
      || params.max_model_len == 0 || params.kvcache_block_size == 0
      || params.max_model_len > config.max_position_embeddings
      || !std::isfinite(params.gpu_memory_utilization)
      || params.gpu_memory_utilization <= 0
      || params.gpu_memory_utilization > 1) {
    throw std::invalid_argument(
        "invalid engine limits or GPU memory utilization");
  }

  const DeviceGuard guard;
  check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
             "create inference stream");
  try {
    model_ = std::make_unique<Qwen3ForCausalLM>(config_, Device::cuda(0));
    model_->load_ckpt(checkpoint);
    // Weight uploads use the default stream; publish them before inference on
    // our nonblocking stream, which does not implicitly wait for that stream.
    check_cuda(cudaStreamSynchronize(nullptr), "complete model weight uploads");

    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query CUDA memory");
    const auto item_size = data_type_size(model_->dtype());
    const auto block_bytes =
        2 * config_.num_hidden_layers * params_.kvcache_block_size
        * config_.num_key_value_heads * config_.head_dim * item_size;

    // Reserve conservative space for live activations, logits and library
    // workspaces. This baseline estimates memory instead of profiling a pass.
    const auto tokens = std::min(params_.max_num_batched_tokens,
                                 params_.max_num_seqs * params_.max_model_len);
    const auto elements_per_token =
        8.0L * config_.hidden_size + 4.0L * config_.intermediate_size
        + 4.0L * config_.q_size() + 8.0L * config_.kv_size();
    constexpr auto fixed_workspace_bytes = 256.0L * 1024 * 1024;
    const auto activation_bytes = tokens * elements_per_token * item_size;
    const auto logits_bytes =
        params_.max_num_seqs * config_.vocab_size * item_size;
    const auto reserve =
        fixed_workspace_bytes + activation_bytes + logits_bytes;
    const auto budget = total_bytes * params_.gpu_memory_utilization
                        - (total_bytes - free_bytes) - reserve;
    if (budget < block_bytes) {
      throw std::runtime_error(
          "insufficient GPU memory for KV cache and inference workspace");
    }
    const auto per_sequence =
        (params_.max_model_len - 1) / params_.kvcache_block_size + 1;
    const auto requested = per_sequence * params_.max_num_seqs;
    const auto index_limit =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
        / params_.kvcache_block_size;
    num_blocks_ = std::min({static_cast<std::size_t>(budget / block_bytes),
                            requested, index_limit});
    if (num_blocks_ < per_sequence) {
      throw std::runtime_error(
          "KV cache cannot hold max_model_len; reduce the configured limit");
    }
    const std::vector<std::size_t> shape{
        config_.num_hidden_layers, num_blocks_, params_.kvcache_block_size,
        config_.num_key_value_heads, config_.head_dim};
    key_cache_ = Tensor::empty(shape, model_->dtype(), model_->device());
    value_cache_ = Tensor::empty(shape, model_->dtype(), model_->device());
    const std::array layer_shape{num_blocks_, params_.kvcache_block_size,
                                 config_.num_key_value_heads, config_.head_dim};
    for (std::size_t i = 0; i < config_.num_hidden_layers; ++i) {
      model_->transformer().layer(i).self_attention().set_kv_cache(
          key_cache_.view().slice(0, i, 1).reshape(layer_shape),
          value_cache_.view().slice(0, i, 1).reshape(layer_shape));
    }
  } catch (...) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
    throw;
  }
}

ModelRunner::~ModelRunner() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

std::vector<std::int32_t> ModelRunner::run(
    std::span<const Sequence* const> seqs, bool prefill) {
  if (seqs.empty()) {
    return {};
  }
  if (seqs.size() > params_.max_num_seqs) {
    throw std::invalid_argument("inference batch exceeds max_num_seqs");
  }
  const DeviceGuard guard;
  std::vector<std::int32_t> ids, positions, slots, cu_q{0}, cu_k{0}, lengths;
  std::vector<float> temperatures;
  std::size_t max_q = 0, max_k = 0, table_width = 0;
  bool needs_cached_prefill = false;
  for (const auto* seq : seqs) {
    if (seq == nullptr || seq->num_scheduled_tokens == 0
        || seq->block_size != params_.kvcache_block_size
        || seq->num_tokens() > params_.max_model_len) {
      throw std::invalid_argument("invalid scheduled sequence");
    }

    const auto start = seq->num_cached_tokens;
    const auto count = seq->num_scheduled_tokens;
    if (start > seq->num_tokens() || count > seq->num_tokens() - start
        || (!prefill && (count != 1 || start + 1 != seq->num_tokens()))) {
      throw std::invalid_argument(
          "scheduled token range does not match cached context");
    }
    const auto end = start + count;

    needs_cached_prefill |= start != 0;
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
    cu_k.push_back(static_cast<std::size_t>(cu_k.back()) + end);
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

  auto upload = [&](const std::vector<std::int32_t>& data,
                    std::vector<std::size_t> shape) {
    auto tensor =
        Tensor::empty(std::move(shape), DataType::Int32, model_->device());
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
  auto temperature =
      Tensor::empty({seqs.size()}, DataType::Float32, model_->device());

  check_cuda(cudaMemcpyAsync(temperature.data(), temperatures.data(),
                             temperature.size_bytes(), cudaMemcpyHostToDevice,
                             stream_),
             "upload temperatures");

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
  context.stream = stream_;

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
