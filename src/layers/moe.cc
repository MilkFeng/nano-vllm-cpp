#include "layers/moe.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/expert_policy.hh"
#include "kernels/cpu/expert.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {
namespace {

constexpr std::size_t kExpertFP8BlockSize = 128;

std::size_t scale_dimension(std::size_t dimension) {
  return dimension / kExpertFP8BlockSize
         + (dimension % kExpertFP8BlockSize != 0);
}

void reserve_rows(Tensor& tensor, std::size_t rows, std::size_t width,
                  Device device = Device::cpu()) {
  if (tensor.rank() != 2 || tensor.shape()[1] != width
      || tensor.device() != device || tensor.shape()[0] < rows) {
    tensor = Tensor::empty({rows, width}, DataType::Float32, device);
  }
}

TensorView host_rows(PinnedHostBuffer& buffer, std::size_t rows,
                     std::size_t width) {
  assert(rows * width <= buffer.capacity_bytes() / sizeof(float));
  return TensorView(buffer.data(), DataType::Float32,
                    TensorLayout({rows, width}), Device::cpu());
}

void drain_tasks(std::vector<std::future<void>>& tasks) noexcept {
  for (auto& task : tasks) {
    if (task.valid()) task.wait();
  }
}

} // namespace

FP8ExpertWorkspace::FP8ExpertWorkspace(std::size_t intermediate_size,
                                       std::size_t rows) {
  reserve(rows, intermediate_size);
}

void FP8ExpertWorkspace::reserve(std::size_t rows,
                                 std::size_t intermediate_size) {
  reserve_rows(gate, rows, intermediate_size, Device::cuda(0));
  reserve_rows(up, rows, intermediate_size, Device::cuda(0));
  reserve_rows(activated, rows, intermediate_size, Device::cuda(0));
}

FP8Expert::FP8Expert(std::size_t hidden_size, std::size_t intermediate_size,
                     Device device)
    : gate_(hidden_size, intermediate_size, false, DataType::Float8E4M3FN,
            device),
      up_(hidden_size, intermediate_size, false, DataType::Float8E4M3FN,
          device),
      down_(intermediate_size, hidden_size, false, DataType::Float8E4M3FN,
            device),
      gate_scale_(Tensor::empty(
          {scale_dimension(intermediate_size), scale_dimension(hidden_size)},
          DataType::Float32, device)),
      up_scale_(Tensor::empty(
          {scale_dimension(intermediate_size), scale_dimension(hidden_size)},
          DataType::Float32, device)),
      down_scale_(Tensor::empty(
          {scale_dimension(hidden_size), scale_dimension(intermediate_size)},
          DataType::Float32, device)) {
  gate_.register_parameter("weight_scale_inv", gate_scale_);
  up_.register_parameter("weight_scale_inv", up_scale_);
  down_.register_parameter("weight_scale_inv", down_scale_);

  register_module("gate_proj", gate_);
  register_module("up_proj", up_);
  register_module("down_proj", down_);
}

void FP8Expert::forward_into(ConstTensorView input, TensorView output) const {
  kernels::cpu::expert_forward(
      input,
      {gate_.weight().view(), up_.weight().view(), down_.weight().view(),
       gate_scale_.view(), up_scale_.view(), down_scale_.view()},
      output, kExpertFP8BlockSize);
}

void FP8Expert::forward_into(ConstTensorView input, TensorView output,
                             FP8ExpertWorkspace& workspace,
                             cudaStream_t stream) const {
  const auto rows = input.shape()[0];
  const auto gate = workspace.gate.view().slice(0, 0, rows);
  const auto up = workspace.up.view().slice(0, 0, rows);
  const auto activated = workspace.activated.view().slice(0, 0, rows);

  kernels::cuda::linear_fp8(input, gate_.weight().view(), gate_scale_.view(),
                            gate, stream);
  kernels::cuda::linear_fp8(input, up_.weight().view(), up_scale_.view(), up,
                            stream);
  kernels::cuda::silu_and_mul(gate.as_const(), up.as_const(), activated,
                              stream);
  kernels::cuda::linear_fp8(activated.as_const(), down_.weight().view(),
                            down_scale_.view(), output, stream);
}

ExpertStore::ExpertStore(std::size_t hidden_size, std::size_t intermediate_size,
                         std::size_t num_experts)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size) {
  experts_.reserve(num_experts);
  for (std::size_t id = 0; id < num_experts; ++id) {
    auto expert = std::make_unique<FP8Expert>(hidden_size, intermediate_size);

    register_module(std::to_string(id), *expert);
    experts_.push_back(std::move(expert));
  }
}

const FP8Expert& ExpertStore::expert_at(std::size_t expert_id) const {
  return *experts_.at(expert_id);
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
  slots_.reserve(resident_ids_.size());
  for (std::size_t slot = 0; slot < resident_ids_.size(); ++slot) {
    slots_.push_back(std::make_unique<FP8Expert>(
        source.hidden_size(), source.intermediate_size(), Device::cuda(0)));
  }

  // zero slots means all selected experts run on CPU
  ready_ = slots_.empty();
}

void ExpertCache::invalidate() noexcept {
  ready_ = false;
  std::ranges::fill(expert_to_slot_, -1);
}

void ExpertCache::populate(cudaStream_t stream) {
  invalidate();
  try {
    for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
      const auto& expert = source_.expert_at(resident_ids_[slot]);
      for (const auto& [name, parameter] : expert.named_parameters()) {
        const auto source = parameter->view();
        const auto destination = slots_[slot]->parameter(name).view();
        assert(same_dtype(source, destination));
        assert(same_shape(source, destination));
        check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                   source.size_bytes(), cudaMemcpyHostToDevice,
                                   stream),
                   "upload fixed expert cache");
      }
    }
    check_cuda(cudaStreamSynchronize(stream), "complete expert cache uploads");
  } catch (...) {
    cudaStreamSynchronize(stream);
    throw;
  }

  for (std::size_t slot = 0; slot < resident_ids_.size(); ++slot) {
    expert_to_slot_[resident_ids_[slot]] = slot;
  }
  ready_ = true;
}

std::size_t ExpertCache::replace(
    std::span<const ExpertCacheReplacement> replacements, cudaStream_t stream) {
  if (!ready_) throw std::logic_error("populate expert cache before replacement");
  if (replacements.empty()) return 0;

  std::vector<bool> used_slots(slots_.size(), false);
  std::vector<bool> used_ids(source_.num_experts(), false);
  for (const auto& [slot, id] : replacements) {
    if (slot >= slots_.size() || id < 0
        || std::size_t(id) >= source_.num_experts() || used_slots[slot]
        || used_ids[id] || expert_to_slot_[id] >= 0) {
      throw std::invalid_argument("invalid or duplicate expert cache replacement");
    }
    used_slots[slot] = true;
    used_ids[id] = true;
  }

  // A partially overwritten slot must never be consumed after a failed copy.
  ready_ = false;
  std::size_t bytes = 0;
  try {
    for (const auto& [slot, id] : replacements) {
      for (const auto& [name, parameter] : source_.expert_at(id).named_parameters()) {
        const auto source = parameter->view();
        const auto destination = slots_[slot]->parameter(name).view();
        check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                   source.size_bytes(), cudaMemcpyHostToDevice,
                                   stream),
                   "replace GPU expert weights");
        bytes += source.size_bytes();
      }
    }
    check_cuda(cudaStreamSynchronize(stream), "complete expert replacements");
  } catch (...) {
    cudaStreamSynchronize(stream);
    invalidate();
    throw;
  }
  for (const auto& [slot, id] : replacements) {
    expert_to_slot_[resident_ids_[slot]] = -1;
    resident_ids_[slot] = id;
    expert_to_slot_[id] = static_cast<int>(slot);
  }
  ready_ = true;
  return bytes;
}

int ExpertCache::slot_of(std::size_t expert_id) const {
  return expert_to_slot_[expert_id];
}

const FP8Expert& ExpertCache::expert_at_slot(std::size_t slot) const {
  if (!ready_) {
    throw std::logic_error("populate expert cache before execution");
  }
  return *slots_.at(slot);
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
                               const ExpertCache& cache, ThreadPool& cpu_pool)
    : store_(store),
      cache_(cache),
      cpu_pool_(cpu_pool) {
  if (&cache.source() != &store) {
    throw std::invalid_argument(
        "executor and cache must use the same RAM store");
  }
}

ExpertExecutionStats ExpertExecutor::forward_into(
    ConstTensorView input, std::span<const std::int32_t> expert_ids,
    std::size_t top_k, TensorView expert_outputs,
    MoEExecutionWorkspace& workspace, cudaStream_t stream) const {
  const auto tokens = input.shape()[0];
  const auto hidden = store_.hidden_size();
  const auto assignments = tokens * top_k;

  if (tokens == 0) {
    return {};
  }

  if (tokens == 1) {
    return forward_single_into(input, expert_ids, expert_outputs, workspace,
                               stream);
  }

  struct Group {
    std::size_t id;
    std::size_t begin;
    std::size_t rows;
    int slot;
  };

  std::vector<std::size_t> counts(store_.num_experts(), 0);

  for (const auto id : expert_ids) {
    ++counts[id];
  }

  std::vector<Group> hits, misses;
  std::vector<std::size_t> offsets(counts.size());
  std::size_t end = 0, gpu_rows = 0, max_gpu_rows = 0;

  for (const bool gpu : {true, false}) {
    for (std::size_t id = 0; id < counts.size(); ++id) {
      if (counts[id] == 0) {
        continue;
      }

      const auto slot = cache_.slot_of(id);

      if ((slot >= 0) != gpu) {
        continue;
      }
      offsets[id] = end;

      if (gpu) {
        hits.push_back({id, end, counts[id], slot});
      } else {
        misses.push_back({id, end, counts[id], slot});
      }

      end += counts[id];

      if (gpu) {
        max_gpu_rows = std::max(max_gpu_rows, counts[id]);
      }
    }
    if (gpu) {
      gpu_rows = end;
    }
  }
  const auto cpu_rows = assignments - gpu_rows;

  workspace.host_order.reserve(assignments * sizeof(std::int32_t));

  auto* order = static_cast<std::int32_t*>(workspace.host_order.data());
  for (std::size_t rank = 0; rank < assignments; ++rank) {
    order[offsets[expert_ids[rank]]++] = static_cast<std::int32_t>(rank);
  }

  if (workspace.order.numel() < assignments) {
    workspace.order =
        Tensor::empty({assignments}, DataType::Int32, Device::cuda(0));
  }
  reserve_rows(workspace.grouped_parts, assignments, hidden, Device::cuda(0));
  if (gpu_rows != 0) {
    reserve_rows(workspace.grouped_input, gpu_rows, hidden, Device::cuda(0));
    workspace.expert.reserve(max_gpu_rows, store_.intermediate_size());
  }
  if (cpu_rows != 0) {
    workspace.host_input.reserve(input.size_bytes());
    workspace.host_parts.reserve(cpu_rows * hidden * sizeof(float));
    reserve_rows(workspace.host_grouped_input, cpu_rows, hidden);
  }

  const auto indices = workspace.order.view().slice(0, 0, assignments);
  const auto parts = workspace.grouped_parts.view().slice(0, 0, assignments);
  std::vector<std::future<void>> tasks;
  tasks.reserve(misses.size());
  try {
    check_cuda(cudaMemcpyAsync(indices.data(), order, indices.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               "upload grouped expert permutation");
    if (cpu_rows != 0) {
      check_cuda(
          cudaMemcpyAsync(workspace.host_input.data(), input.data(),
                          input.size_bytes(), cudaMemcpyDeviceToHost, stream),
          "download batch input for CPU experts");
      check_cuda(cudaStreamSynchronize(stream),
                 "complete CPU expert input download");

      const auto* source =
          static_cast<const float*>(workspace.host_input.data());
      auto* packed = static_cast<float*>(workspace.host_grouped_input.data());

      for (std::size_t row = 0; row < cpu_rows; ++row) {
        const auto token = std::size_t(order[gpu_rows + row]) / top_k;
        std::memcpy(packed + row * hidden, source + token * hidden,
                    hidden * sizeof(float));
      }

      const auto host_output =
          host_rows(workspace.host_parts, cpu_rows, hidden);

      for (const auto& group : misses) {
        const auto begin = group.begin - gpu_rows;
        const auto x = workspace.host_grouped_input.view().as_const().slice(
            0, begin, group.rows);
        const auto y = host_output.slice(0, begin, group.rows);
        const auto* expert = &store_.expert_at(group.id);
        tasks.push_back(
            cpu_pool_.submit([expert, x, y] { expert->forward_into(x, y); }));
      }
    }

    if (gpu_rows != 0) {
      const auto grouped_input =
          workspace.grouped_input.view().slice(0, 0, gpu_rows);
      kernels::cuda::moe_gather(input, indices.as_const().slice(0, 0, gpu_rows),
                                top_k, grouped_input, stream);

      for (const auto& group : hits) {
        cache_.expert_at_slot(group.slot)
            .forward_into(
                grouped_input.as_const().slice(0, group.begin, group.rows),
                parts.slice(0, group.begin, group.rows), workspace.expert,
                stream);
      }
    }

    for (auto& task : tasks) {
      task.get();
    }

    if (cpu_rows != 0) {
      const auto destination = parts.slice(0, gpu_rows, cpu_rows);
      check_cuda(cudaMemcpyAsync(
                     destination.data(), workspace.host_parts.data(),
                     destination.size_bytes(), cudaMemcpyHostToDevice, stream),
                 "upload CPU expert groups");
    }
    kernels::cuda::moe_scatter(parts.as_const(), indices.as_const(),
                               expert_outputs, stream);
  } catch (...) {
    drain_tasks(tasks);
    cudaStreamSynchronize(stream);
    throw;
  }
  return {gpu_rows, cpu_rows};
}

ExpertExecutionStats ExpertExecutor::forward_single_into(
    ConstTensorView input, std::span<const std::int32_t> expert_ids,
    TensorView expert_outputs, MoEExecutionWorkspace& workspace,
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
    const Task task = {rank, id, slot};
    if (slot >= 0) {
      hits.emplace_back(task);
    } else {
      misses.emplace_back(task);
    }
  }

  if (!hits.empty()) workspace.expert.reserve(1, store_.intermediate_size());

  if (!misses.empty()) {
    reserve_rows(workspace.single_input, 1, store_.hidden_size());
    reserve_rows(workspace.single_parts, expert_ids.size(),
                 store_.hidden_size());
  }

  std::vector<std::future<void>> cpu_tasks;
  cpu_tasks.reserve(misses.size());
  try {
    if (!misses.empty()) {
      check_cuda(
          cudaMemcpyAsync(workspace.single_input.data(), input.data(),
                          input.size_bytes(), cudaMemcpyDeviceToHost, stream),
          "download input for overlapping CPU experts");
      check_cuda(cudaStreamSynchronize(stream),
                 "complete overlapping input download");
    }

    for (const auto& task : misses) {
      const auto* expert = &store_.expert_at(task.expert_id);
      const auto x = workspace.single_input.view().as_const();
      const auto row = workspace.single_parts.view().slice(0, task.rank, 1);
      cpu_tasks.push_back(
          cpu_pool_.submit([expert, x, row] { expert->forward_into(x, row); }));
    }

    for (const auto& task : hits) {
      const auto row = expert_outputs.slice(0, task.rank, 1);
      cache_.expert_at_slot(task.slot).forward_into(input, row,
                                                    workspace.expert, stream);
    }

    for (auto& task : cpu_tasks) {
      task.get();
    }

    for (const auto& task : misses) {
      const auto source = workspace.single_parts.view().slice(0, task.rank, 1);
      const auto destination = expert_outputs.slice(0, task.rank, 1);

      check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                 destination.size_bytes(),
                                 cudaMemcpyHostToDevice, stream),
                 "upload overlapping CPU expert output");
    }
  } catch (...) {
    drain_tasks(cpu_tasks);
    cudaStreamSynchronize(stream);
    throw;
  }

  return {hits.size(), misses.size()};
}

MoE::MoE(MoEConfig config, ThreadPool& cpu_pool, Device device,
         std::optional<std::vector<std::int32_t>> gpu_expert_ids,
         MoEExecutionWorkspace* workspace)
    : config_(config),
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

  if (!workspace) {
    owned_workspace_ = std::make_unique<MoEExecutionWorkspace>();
    workspace = owned_workspace_.get();
  }
  workspace_ = workspace;

  cache_ = std::make_unique<ExpertCache>(experts_, std::move(*gpu_expert_ids));
  executor_ = std::make_unique<ExpertExecutor>(experts_, *cache_, cpu_pool);
}

void MoE::invalidate_expert_cache() noexcept {
  if (cache_) {
    cache_->invalidate();
  }
}

void MoE::populate_expert_cache(cudaStream_t stream) {
  if (!cache_) {
    throw std::logic_error("CPU MoE has no CUDA expert cache");
  }
  cache_->populate(stream);
}

MoEOutput MoE::forward(ConstTensorView hidden_states,
                       cudaStream_t stream, MoEPhaseProfile* profile,
                       ExpertUsage* usage) const {
  const auto started = profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (!cache_ || !cache_->ready()) {
    throw std::logic_error(
        "call populate_expert_cache after initializing MoE weights");
  }
  if (hidden_states.rank() != 2 || hidden_states.shape()[0] == 0
      || hidden_states.shape()[1] != config_.hidden_size
      || hidden_states.dtype() != DataType::Float32
      || hidden_states.device() != Device::cuda(0)
      || !hidden_states.is_contiguous()) {
    throw std::invalid_argument("MoE input must be nonempty CUDA FP32 [T,H]");
  }

  const auto tokens = hidden_states.shape()[0];
  if (usage && usage->counts.size() != config_.num_experts) {
    throw std::invalid_argument("expert usage must contain every expert");
  }
  const auto device = hidden_states.device();
  if (config_.top_k == 0
      || tokens > std::numeric_limits<std::int32_t>::max() / config_.top_k) {
    throw std::invalid_argument("MoE assignment count exceeds Int32 indices");
  }
  const auto assignments = tokens * config_.top_k;
  workspace_->host_ids.reserve(assignments * sizeof(std::int32_t));
  const auto host_ids = std::span(
      static_cast<std::int32_t*>(workspace_->host_ids.data()), assignments);

  // These owners must outlive the error handler's stream synchronization.
  Tensor logits, shared_output, shared_gate_logits;
  auto ids = Tensor::empty({tokens, config_.top_k}, DataType::Int32, device);
  auto weights =
      Tensor::empty({tokens, config_.top_k}, DataType::Float32, device);
  auto expert_outputs = Tensor::empty({assignments, config_.hidden_size},
                                      DataType::Float32, device);
  auto output =
      Tensor::empty({tokens, config_.hidden_size}, DataType::Float32, device);
  ExpertExecutionStats stats;
  try {
    logits = gate_.forward(hidden_states, stream);
    kernels::cuda::moe_route(logits.view().as_const(), ids.view(),
                             weights.view(), stream);
    check_cuda(cudaMemcpyAsync(host_ids.data(), ids.data(), ids.size_bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "copy MoE expert IDs to CPU");
    check_cuda(cudaStreamSynchronize(stream), "complete MoE routing");

    stats = executor_->forward_into(hidden_states, host_ids, config_.top_k,
                                    expert_outputs.view(), *workspace_, stream);
    kernels::cuda::moe_combine(expert_outputs.view().as_const(),
                               weights.view().as_const(), output.view(),
                               stream);

    if (shared_expert_) {
      shared_output = shared_expert_->forward(hidden_states, stream);
      shared_gate_logits = shared_expert_gate_->forward(hidden_states, stream);
      kernels::cuda::moe_add_shared(shared_output.view().as_const(),
                                    shared_gate_logits.view().as_const(),
                                    output.view(), stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "complete MoE forward");
  } catch (...) {
    // Executor has drained CPU tasks; keep inputs, outputs, and staging alive
    // until any queued DMA/kernels finish, including failed combine/shared
    // work.
    cudaStreamSynchronize(stream);
    throw;
  }
  if (usage) {
    for (const auto id : host_ids) ++usage->counts[id];
    usage->gpu_assignments += stats.gpu_experts;
    usage->cpu_assignments += stats.cpu_experts;
  }
  if (profile) {
    profile->seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    ++profile->calls;
    profile->input_tokens += tokens;
    profile->gpu_assignments += stats.gpu_experts;
    profile->cpu_assignments += stats.cpu_experts;
    profile->experts.resize(config_.num_experts);
    std::vector<std::size_t> counts(config_.num_experts, 0);
    for (const auto id : host_ids) ++counts[id];
    for (std::size_t id = 0; id < counts.size(); ++id) {
      if (counts[id] == 0) continue;
      auto& expert = profile->experts[id];
      expert.tokens += counts[id];
      ++expert.calls;
      ++expert.group_sizes[counts[id]];
    }
  }
  return {
      std::move(output),
      std::move(logits),
      std::move(ids),
      std::move(weights),
      std::move(expert_outputs),
      stats.gpu_experts,
      stats.cpu_experts,
      std::move(shared_output),
      std::move(shared_gate_logits),
  };
}

std::size_t MoE::replace_experts(
    std::span<const ExpertCacheReplacement> replacements,
    cudaStream_t stream) const {
  if (!cache_) throw std::logic_error("CPU MoE has no CUDA expert cache");
  return cache_->replace(replacements, stream);
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
