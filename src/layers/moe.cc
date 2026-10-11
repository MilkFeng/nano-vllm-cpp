#include "layers/moe.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {
namespace {

MoEConfig validated(MoEConfig config, Device device) {
  config.validate();
  if (device != Device::cpu() && device != Device::cuda(0)) {
    throw std::invalid_argument("MoE supports CPU weights or CUDA device 0");
  }
  return config;
}

} // namespace

ExpertStore::ExpertStore(std::size_t hidden_size, std::size_t intermediate_size,
                         std::size_t num_experts)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size) {
  if (hidden_size == 0 || intermediate_size == 0 || num_experts == 0) {
    throw std::invalid_argument("ExpertStore dimensions must be positive");
  }
  experts_.reserve(num_experts);
  for (std::size_t id = 0; id < num_experts; ++id) {
    auto expert = std::make_unique<SwiGLUMLP>(hidden_size, intermediate_size,
                                              DataType::Float32, Device::cpu());
    register_module(std::to_string(id), *expert);
    experts_.push_back(std::move(expert));
  }
}

kernels::cpu::ExpertWeights ExpertStore::weights(std::size_t expert_id) const {
  if (expert_id >= experts_.size()) {
    throw std::out_of_range("ExpertStore expert ID is out of range");
  }
  const auto& expert = std::as_const(*experts_[expert_id]);
  return {expert.parameter("gate_proj.weight").view(),
          expert.parameter("up_proj.weight").view(),
          expert.parameter("down_proj.weight").view()};
}

std::size_t ExpertStore::hidden_size() const noexcept {
  return hidden_size_;
}
std::size_t ExpertStore::intermediate_size() const noexcept {
  return intermediate_size_;
}
std::size_t ExpertStore::num_experts() const noexcept {
  return experts_.size();
}

std::size_t ExpertStore::size_bytes() const {
  std::size_t total = 0;
  for (const auto& [name, parameter] : named_parameters()) {
    total += parameter->view().size_bytes();
  }
  return total;
}

ExpertCache::ExpertCache(const ExpertStore& source,
                         std::vector<std::int32_t> resident_ids)
    : source_(source),
      resident_ids_(std::move(resident_ids)),
      expert_to_slot_(source.num_experts(), -1) {
  // Validate the entire list before allocating any CUDA expert weights.
  std::vector<bool> seen(source.num_experts(), false);
  for (const auto id : resident_ids_) {
    if (id < 0 || static_cast<std::size_t>(id) >= source.num_experts()
        || seen[static_cast<std::size_t>(id)]) {
      throw std::invalid_argument(
          "cached expert IDs must be unique and in range");
    }
    seen[static_cast<std::size_t>(id)] = true;
  }
  slots_.reserve(resident_ids_.size());
  for (std::size_t slot = 0; slot < resident_ids_.size(); ++slot) {
    slots_.push_back(std::make_unique<SwiGLUMLP>(
        source.hidden_size(), source.intermediate_size(), DataType::Float32,
        Device::cuda(0)));
  }
  ready_ = slots_.empty(); // Zero slots means all selected experts run on CPU.
}

void ExpertCache::populate(cudaStream_t stream) {
  ready_ = false;
  std::ranges::fill(expert_to_slot_, -1);
  try {
    for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
      const auto weights = source_.weights(resident_ids_[slot]);
      const auto copy = [&](const char* name, ConstTensorView source) {
        auto destination = slots_[slot]->parameter(name).view();
        check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                   source.size_bytes(), cudaMemcpyHostToDevice,
                                   stream),
                   "upload fixed expert cache");
      };
      copy("gate_proj.weight", weights.gate);
      copy("up_proj.weight", weights.up);
      copy("down_proj.weight", weights.down);
    }
    check_cuda(cudaStreamSynchronize(stream), "complete expert cache uploads");
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }
  for (std::size_t slot = 0; slot < resident_ids_.size(); ++slot) {
    expert_to_slot_[resident_ids_[slot]] = static_cast<int>(slot);
  }
  ready_ = true;
}

int ExpertCache::slot_of(std::size_t expert_id) const {
  if (expert_id >= expert_to_slot_.size()) {
    throw std::out_of_range("cached expert ID is out of range");
  }
  return expert_to_slot_[expert_id];
}

const SwiGLUMLP& ExpertCache::expert_at_slot(std::size_t slot) const {
  if (!ready_) throw std::logic_error("populate expert cache before execution");
  if (slot >= slots_.size())
    throw std::out_of_range("expert slot is out of range");
  return *slots_[slot];
}

bool ExpertCache::ready() const noexcept {
  return ready_;
}
std::size_t ExpertCache::num_slots() const noexcept {
  return slots_.size();
}
std::size_t ExpertCache::size_bytes() const {
  std::size_t total = 0;
  for (const auto& expert : slots_) {
    for (const auto& [name, parameter] :
         std::as_const(*expert).named_parameters()) {
      total += parameter->view().size_bytes();
    }
  }
  return total;
}
std::span<const std::int32_t> ExpertCache::resident_ids() const noexcept {
  return resident_ids_;
}
const ExpertStore& ExpertCache::source() const noexcept {
  return source_;
}

ExpertExecutor::ExpertExecutor(const ExpertStore& store,
                               const ExpertCache& cache,
                               ExpertExecutionMode mode)
    : store_(store),
      cache_(cache),
      mode_(mode) {
  if (&cache.source() != &store) {
    throw std::invalid_argument(
        "executor and cache must use the same RAM store");
  }
  if (mode_ != ExpertExecutionMode::Synchronous
      && mode_ != ExpertExecutionMode::Overlap) {
    throw std::invalid_argument("invalid expert execution mode");
  }
  if (mode_ == ExpertExecutionMode::Overlap && cache_.num_slots() != 0) {
    // One workspace serves every GPU hit: one stream orders all reads/writes.
    const auto shape = std::vector<std::size_t>{1, store_.intermediate_size()};
    gpu_workspace_ = {Tensor::empty(shape, DataType::Float32, Device::cuda(0)),
                      Tensor::empty(shape, DataType::Float32, Device::cuda(0)),
                      Tensor::empty(shape, DataType::Float32, Device::cuda(0))};
  }
}

ExpertExecutionOutput ExpertExecutor::forward(
    ConstTensorView input, std::span<const std::int32_t> expert_ids,
    cudaStream_t stream) const {
  assert(input.device().type == DeviceType::Cuda);
  assert(input.device().index == 0);
  assert(input.dtype() == DataType::Float32);
  assert(input.is_contiguous());
  assert(input.rank() == 2);
  assert(input.shape()[0] == 1);
  assert(input.shape()[1] == store_.hidden_size());
  assert(!expert_ids.empty());
  assert(expert_ids.size() <= store_.num_experts());
  if (!cache_.ready())
    throw std::logic_error("populate expert cache before execution");
  for (const auto id : expert_ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= store_.num_experts()) {
      throw std::invalid_argument("dispatched expert ID is out of range");
    }
  }

  if (mode_ == ExpertExecutionMode::Overlap) {
    return forward_overlap(input, expert_ids, stream);
  }
  return forward_synchronous(input, expert_ids, stream);
}

ExpertExecutionOutput ExpertExecutor::forward_synchronous(
    ConstTensorView input, std::span<const std::int32_t> expert_ids,
    cudaStream_t stream) const {
  ExpertExecutionOutput result{
      Tensor::empty({expert_ids.size(), store_.hidden_size()},
                    DataType::Float32, Device::cuda(0))};
  Tensor host_input, host_parts;
  std::vector<Tensor> gpu_results;
  gpu_results.reserve(expert_ids.size());
  try {
    for (std::size_t rank = 0; rank < expert_ids.size(); ++rank) {
      const auto id = static_cast<std::size_t>(expert_ids[rank]);
      const auto slot = cache_.slot_of(id);
      auto destination = result.expert_outputs.slice(0, rank, 1);
      if (slot >= 0) {
        gpu_results.push_back(
            cache_.expert_at_slot(slot).forward(input, stream));
        check_cuda(
            cudaMemcpyAsync(destination.data(), gpu_results.back().data(),
                            destination.size_bytes(), cudaMemcpyDeviceToDevice,
                            stream),
            "copy GPU expert output");
        ++result.gpu_experts;
      } else {
        // Download x once, regardless of how many experts miss the cache.
        if (host_input.numel() == 0) {
          host_input =
              Tensor::empty({1, store_.hidden_size()}, DataType::Float32);
          host_parts = Tensor::empty({expert_ids.size(), store_.hidden_size()},
                                     DataType::Float32);
          check_cuda(cudaMemcpyAsync(host_input.data(), input.data(),
                                     input.size_bytes(), cudaMemcpyDeviceToHost,
                                     stream),
                     "download MoE input for CPU experts");
          check_cuda(cudaStreamSynchronize(stream),
                     "complete CPU expert input download");
        }
        auto host_row = host_parts.slice(0, rank, 1);
        kernels::cpu::expert_forward(host_input.view().as_const(),
                                     store_.weights(id), host_row.view());
        check_cuda(cudaMemcpyAsync(destination.data(), host_row.data(),
                                   destination.size_bytes(),
                                   cudaMemcpyHostToDevice, stream),
                   "upload CPU expert output");
        ++result.cpu_experts;
      }
    }
    check_cuda(cudaStreamSynchronize(stream),
               "complete CPU/GPU expert dispatch");
  } catch (...) {
    // CPU staging memory must also survive any already-enqueued transfers.
    cudaStreamSynchronize(stream);
    throw;
  }
  return result;
}

ExpertExecutionOutput ExpertExecutor::forward_overlap(
    ConstTensorView input, std::span<const std::int32_t> expert_ids,
    cudaStream_t stream) const {
  struct Task {
    std::size_t rank;
    std::size_t expert_id;
    int slot;
  };
  std::vector<Task> hits, misses;
  hits.reserve(expert_ids.size());
  misses.reserve(expert_ids.size());
  for (std::size_t rank = 0; rank < expert_ids.size(); ++rank) {
    const auto id = static_cast<std::size_t>(expert_ids[rank]);
    const auto slot = cache_.slot_of(id);
    (slot >= 0 ? hits : misses).push_back({rank, id, slot});
  }

  // Allocate/grow buffers before submitting GPU work. Returned outputs own
  // their storage; only internal scratch is reused, so earlier results stay
  // valid.
  ExpertExecutionOutput result{
      Tensor::empty({expert_ids.size(), store_.hidden_size()},
                    DataType::Float32, Device::cuda(0)),
      hits.size(), misses.size()};
  if (!misses.empty()) {
    if (host_input_.numel() == 0) {
      host_input_ = Tensor::empty({1, store_.hidden_size()}, DataType::Float32);
    }
    if (host_parts_.numel() / store_.hidden_size() < expert_ids.size()) {
      host_parts_ = Tensor::empty({expert_ids.size(), store_.hidden_size()},
                                  DataType::Float32);
    }
  }
  try {
    // Ordinary RAM staging is sufficient here: the one input download finishes
    // BEFORE GPU hits start, and CPU outputs are uploaded AFTER CPU work ends.
    if (!misses.empty()) {
      check_cuda(
          cudaMemcpyAsync(host_input_.data(), input.data(), input.size_bytes(),
                          cudaMemcpyDeviceToHost, stream),
          "download input for overlapping CPU experts");
      check_cuda(cudaStreamSynchronize(stream),
                 "complete overlapping input download");
    }

    // Submit every GPU hit first. No CUDA allocation/free or wait inside this
    // loop.
    for (const auto& task : hits) {
      auto row = result.expert_outputs.slice(0, task.rank, 1);
      cache_.expert_at_slot(task.slot).forward_into(input, row.view(),
                                                    gpu_workspace_, stream);
    }
    // The calling thread computes misses while the GPU executes the queued
    // hits.
    for (const auto& task : misses) {
      auto row = host_parts_.slice(0, task.rank, 1);
      kernels::cpu::expert_forward(host_input_.view().as_const(),
                                   store_.weights(task.expert_id), row.view());
    }
    // Same-stream ordering places uploads after GPU hits; original rank
    // controls the destination row regardless of which device computed that
    // expert.
    for (const auto& task : misses) {
      auto source = host_parts_.slice(0, task.rank, 1);
      auto destination = result.expert_outputs.slice(0, task.rank, 1);
      check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                 destination.size_bytes(),
                                 cudaMemcpyHostToDevice, stream),
                 "upload overlapping CPU expert output");
    }
    check_cuda(cudaStreamSynchronize(stream),
               "complete overlapping expert dispatch");
  } catch (...) {
    // Protect persistent scratch and local output from already-enqueued work.
    cudaStreamSynchronize(stream);
    throw;
  }
  return result;
}

void MoEConfig::validate() const {
  if (hidden_size == 0 || intermediate_size == 0 || num_experts == 0
      || num_experts > kernels::cuda::kMaxMoEExperts || top_k == 0
      || top_k > num_experts) {
    throw std::invalid_argument(
        "MoE needs positive dimensions, 1..1024 experts and 1 <= top_k <= E");
  }
}

MoE::MoE(MoEConfig config, Device device,
         std::optional<std::vector<std::int32_t>> gpu_expert_ids,
         ExpertExecutionMode mode)
    : config_(validated(config, device)),
      gate_(config_.hidden_size, config_.num_experts, false, DataType::Float32,
            device),
      experts_(config_.hidden_size, config_.intermediate_size,
               config_.num_experts) {
  register_module("gate", gate_);
  register_module("experts", experts_);
  if (config_.shared_expert_intermediate_size != 0) {
    shared_expert_ = std::make_unique<SwiGLUMLP>(
        config_.hidden_size, config_.shared_expert_intermediate_size,
        DataType::Float32, device);
    shared_expert_gate_ = std::make_unique<Linear>(
        config_.hidden_size, 1, false, DataType::Float32, device);
    register_module("shared_expert", *shared_expert_);
    register_module("shared_expert_gate", *shared_expert_gate_);
  }
  if (device == Device::cpu()) {
    if (gpu_expert_ids.has_value()) {
      throw std::invalid_argument(
          "CPU MoE construction has no GPU resident list");
    }
    return;
  }
  if (!gpu_expert_ids) {
    gpu_expert_ids.emplace(config_.num_experts);
    std::iota(gpu_expert_ids->begin(), gpu_expert_ids->end(), 0);
  }
  cache_ = std::make_unique<ExpertCache>(experts_, std::move(*gpu_expert_ids));
  executor_ = std::make_unique<ExpertExecutor>(experts_, *cache_, mode);
}

void MoE::populate_expert_cache(cudaStream_t stream) {
  if (!cache_) throw std::logic_error("CPU MoE has no CUDA expert cache");
  cache_->populate(stream);
}

MoEOutput MoE::forward(ConstTensorView hidden_states,
                       cudaStream_t stream) const {
  assert(hidden_states.device().type == DeviceType::Cuda);
  assert(hidden_states.device().index == 0);
  assert(same_device(hidden_states, gate_.weight().view()));
  assert(hidden_states.dtype() == DataType::Float32);
  assert(same_dtype(hidden_states, gate_.weight().view()));
  assert(hidden_states.is_contiguous());
  assert(hidden_states.rank() == 2);
  assert(hidden_states.shape()[0] > 0);
  assert(hidden_states.shape()[1] == config_.hidden_size);
  if (!cache_->ready()) {
    throw std::logic_error(
        "call populate_expert_cache after initializing MoE weights");
  }

  const auto tokens = hidden_states.shape()[0];
  const auto device = hidden_states.device();
  auto logits = gate_.forward(hidden_states, stream);
  auto ids = Tensor::empty({tokens, config_.top_k}, DataType::Int32, device);
  auto weights =
      Tensor::empty({tokens, config_.top_k}, DataType::Float32, device);
  kernels::cuda::moe_route(logits.view().as_const(), ids.view(), weights.view(),
                           stream);

  std::vector<std::int32_t> host_ids(tokens * config_.top_k);
  check_cuda(cudaMemcpyAsync(host_ids.data(), ids.data(), ids.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "copy MoE expert IDs to CPU");
  check_cuda(cudaStreamSynchronize(stream), "complete MoE routing");

  auto expert_outputs = Tensor::empty(
      {tokens * config_.top_k, config_.hidden_size}, DataType::Float32, device);
  std::size_t gpu_experts = 0, cpu_experts = 0;
  // Retain each source until queued D2D copies finish on this stream.
  std::vector<Tensor> token_outputs;
  token_outputs.reserve(tokens);
  for (std::size_t t = 0; t < tokens; ++t) {
    const auto selected = std::span<const std::int32_t>(host_ids).subspan(
        t * config_.top_k, config_.top_k);
    auto executed =
        executor_->forward(hidden_states.slice(0, t, 1), selected, stream);
    auto destination =
        expert_outputs.view().slice(0, t * config_.top_k, config_.top_k);
    check_cuda(cudaMemcpyAsync(
                   destination.data(), executed.expert_outputs.data(),
                   destination.size_bytes(), cudaMemcpyDeviceToDevice, stream),
               "copy token expert rows into batch");
    gpu_experts += executed.gpu_experts;
    cpu_experts += executed.cpu_experts;
    token_outputs.push_back(std::move(executed.expert_outputs));
  }
  auto output =
      Tensor::empty({tokens, config_.hidden_size}, DataType::Float32, device);
  kernels::cuda::moe_combine(expert_outputs.view().as_const(),
                             weights.view().as_const(), output.view(), stream);
  Tensor shared_output, shared_gate_logits;
  if (shared_expert_) {
    // Both shared layers already accept [T,H]; each token has its own gate.
    shared_output = shared_expert_->forward(hidden_states, stream);
    shared_gate_logits = shared_expert_gate_->forward(hidden_states, stream);
    kernels::cuda::moe_add_shared(shared_output.view().as_const(),
                                  shared_gate_logits.view().as_const(),
                                  output.view(), stream);
  }
  check_cuda(cudaStreamSynchronize(stream), "complete MoE forward");
  return {std::move(output),
          std::move(logits),
          std::move(ids),
          std::move(weights),
          std::move(expert_outputs),
          gpu_experts,
          cpu_experts,
          std::move(shared_output),
          std::move(shared_gate_logits)};
}

const MoEConfig& MoE::config() const noexcept {
  return config_;
}
const ExpertStore& MoE::expert_store() const noexcept {
  return experts_;
}
const ExpertCache& MoE::expert_cache() const {
  if (!cache_) throw std::logic_error("CPU MoE has no CUDA expert cache");
  return *cache_;
}

} // namespace nanovllm
