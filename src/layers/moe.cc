#include "layers/moe.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <future>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cpu/expert.hh"
#include "kernels/cuda/fp8.hh"
#include "kernels/cuda/kernels.hh"

namespace nanovllm {
namespace {

constexpr std::size_t kExpertFP8BlockSize = 128;

std::size_t scale_dimension(std::size_t dimension) {
  return dimension / kExpertFP8BlockSize
         + (dimension % kExpertFP8BlockSize != 0);
}

std::size_t expert_dimension(std::size_t dimension, Device device) {
  return device.type == DeviceType::Cuda
             ? scale_dimension(dimension) * kExpertFP8BlockSize
             : dimension;
}

} // namespace

FP8ExpertWorkspace::FP8ExpertWorkspace(std::size_t hidden_size,
                                       std::size_t intermediate_size)
    : gate(Tensor::empty({1, intermediate_size}, DataType::Float32,
                         Device::cuda(0))),
      up(Tensor::empty({1, intermediate_size}, DataType::Float32,
                       Device::cuda(0))),
      activated(Tensor::empty({1, intermediate_size}, DataType::Float32,
                              Device::cuda(0))),
      linear(1, std::max(hidden_size, intermediate_size)) {}

FP8Expert::FP8Expert(std::size_t hidden_size, std::size_t intermediate_size,
                     Device device)
    : gate_(expert_dimension(hidden_size, device),
            expert_dimension(intermediate_size, device), false,
            DataType::Float8E4M3FN, device),
      up_(expert_dimension(hidden_size, device),
          expert_dimension(intermediate_size, device), false,
          DataType::Float8E4M3FN, device),
      down_(expert_dimension(intermediate_size, device),
            expert_dimension(hidden_size, device), false,
            DataType::Float8E4M3FN, device),
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
  kernels::cuda::linear_fp8(input, gate_.weight().view(), gate_scale_.view(),
                            workspace.gate.view(), workspace.linear, stream);
  kernels::cuda::linear_fp8(input, up_.weight().view(), up_scale_.view(),
                            workspace.up.view(), workspace.linear, stream);
  kernels::cuda::silu_and_mul(workspace.gate.view().as_const(),
                              workspace.up.view().as_const(),
                              workspace.activated.view(), stream);
  kernels::cuda::linear_fp8(workspace.activated.view().as_const(),
                            down_.weight().view(), down_scale_.view(), output,
                            workspace.linear, stream);
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
        if (source.dtype() == DataType::Float8E4M3FN
            && !same_shape(source, destination)) {
          check_cuda(cudaMemsetAsync(destination.data(), 0,
                                     destination.size_bytes(), stream),
                     "zero FP8 expert padding");
          check_cuda(cudaMemcpy2DAsync(
                         destination.data(), destination.shape()[1],
                         source.data(), source.shape()[1], source.shape()[1],
                         source.shape()[0], cudaMemcpyHostToDevice, stream),
                     "upload padded FP8 expert weight");
        } else {
          assert(same_shape(source, destination));
          check_cuda(cudaMemcpyAsync(destination.data(), source.data(),
                                     source.size_bytes(),
                                     cudaMemcpyHostToDevice, stream),
                     "upload fixed expert cache");
        }
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

  if (cache_.num_slots() != 0) {
    workspace_ =
        FP8ExpertWorkspace(store_.hidden_size(), store_.intermediate_size());
  }
}

ExpertExecutionOutput ExpertExecutor::forward(
    ConstTensorView input, std::span<const std::int32_t> expert_ids,
    cudaStream_t stream) const {
  if (!cache_.ready()) {
    throw std::logic_error("populate expert cache before execution");
  }

  for (const auto id : expert_ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= store_.num_experts()) {
      throw std::invalid_argument("dispatched expert ID is out of range");
    }
  }

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

  std::vector<std::future<void>> cpu_tasks;
  cpu_tasks.reserve(misses.size());
  try {
    if (!misses.empty()) {
      check_cuda(
          cudaMemcpyAsync(host_input_.data(), input.data(), input.size_bytes(),
                          cudaMemcpyDeviceToHost, stream),
          "download input for overlapping CPU experts");
      check_cuda(cudaStreamSynchronize(stream),
                 "complete overlapping input download");
    }

    for (const auto& task : misses) {
      const auto* expert = &store_.expert_at(task.expert_id);
      const auto x = host_input_.view().as_const();
      auto row = host_parts_.slice(0, task.rank, 1).view();
      cpu_tasks.push_back(
          cpu_pool_.submit([expert, x, row] { expert->forward_into(x, row); }));
    }

    for (const auto& task : hits) {
      auto row = result.expert_outputs.slice(0, task.rank, 1);
      cache_.expert_at_slot(task.slot).forward_into(input, row.view(),
                                                    workspace_, stream);
    }

    for (auto& task : cpu_tasks) {
      task.get();
    }

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
    for (auto& task : cpu_tasks) {
      if (task.valid()) {
        task.wait();
      }
    }
    cudaStreamSynchronize(stream);
    throw;
  }

  return result;
}

MoE::MoE(MoEConfig config, ThreadPool& cpu_pool, Device device,
         std::optional<std::vector<std::int32_t>> gpu_expert_ids)
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
                       cudaStream_t stream) const {
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
    shared_output = shared_expert_->forward(hidden_states, stream);
    shared_gate_logits = shared_expert_gate_->forward(hidden_states, stream);
    kernels::cuda::moe_add_shared(shared_output.view().as_const(),
                                  shared_gate_logits.view().as_const(),
                                  output.view(), stream);
  }
  check_cuda(cudaStreamSynchronize(stream), "complete MoE forward");
  return {
      std::move(output),
      std::move(logits),
      std::move(ids),
      std::move(weights),
      std::move(expert_outputs),
      gpu_experts,
      cpu_experts,
      std::move(shared_output),
      std::move(shared_gate_logits),
  };
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
