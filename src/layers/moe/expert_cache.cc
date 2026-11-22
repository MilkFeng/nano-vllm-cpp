#include "layers/moe/expert_cache.hh"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

#include "common/cuda_utils.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/moe/expert.hh"

namespace nanovllm {

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

  ready_ = slots_.empty();
}

void ExpertCache::invalidate() noexcept {
  ready_ = false;
  std::ranges::fill(expert_to_slot_, -1);
}

void ExpertCache::populate(cudaStream_t stream) {
  invalidate();

  for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
    const auto& expert = source_.expert_at(resident_ids_[slot]);
    for (const auto& [name, parameter] : expert.named_parameters()) {
      const auto source = parameter->view();
      const auto destination = slots_[slot]->parameter(name).view();

      assert(same_dtype(source, destination));
      assert(same_shape(source, destination));

      check_cuda(
          cudaMemcpyAsync(destination.data(), source.data(),
                          source.size_bytes(), cudaMemcpyHostToDevice, stream),
          "upload fixed expert cache");
    }
  }

  publish_routes(stream);

  check_cuda(cudaStreamSynchronize(stream),
             "complete expert weight and routing table uploads");

  for (std::size_t slot = 0; slot < resident_ids_.size(); ++slot) {
    expert_to_slot_[resident_ids_[slot]] = slot;
  }

  ready_ = true;
}

ExpertCache::~ExpertCache() {
  if (routes_ready_) {
    cudaEventSynchronize(routes_ready_);
    cudaEventDestroy(routes_ready_);
  }
}

void ExpertCache::publish_routes(cudaStream_t stream) {
  if (routes_ready_) {
    check_cuda(cudaEventSynchronize(routes_ready_),
               "complete previous routing table DMA");
  } else {
    check_cuda(cudaEventCreateWithFlags(&routes_ready_, cudaEventDisableTiming),
               "create routing table DMA event");
  }

  const auto bytes =
      source_.num_experts() * sizeof(kernels::cuda::FP8ExpertWeights);
  if (routes_.numel() < bytes) {
    routes_ = Tensor::empty({bytes}, DataType::Float8E4M3FN, Device::cuda(0));
  }

  host_routes_.reserve(bytes);
  auto* routes =
      static_cast<kernels::cuda::FP8ExpertWeights*>(host_routes_.data());
  std::fill_n(routes, source_.num_experts(), kernels::cuda::FP8ExpertWeights{});

  for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
    routes[resident_ids_[slot]] = slots_[slot]->device_weights();
  }

  check_cuda(cudaMemcpyAsync(routes_.data(), routes, bytes,
                             cudaMemcpyHostToDevice, stream),
             "publish resident expert routing table");
  check_cuda(cudaEventRecord(routes_ready_, stream),
             "record expert routing table DMA");
}

ConstTensorView ExpertCache::device_routes() const {
  return routes_.view();
}

std::size_t ExpertCache::replace(
    std::span<const ExpertCacheReplacement> replacements, cudaStream_t stream) {
  if (!ready_) {
    throw std::logic_error("populate expert cache before replacement");
  }

  if (replacements.empty()) {
    return 0;
  }

  // A partially overwritten slot must never be consumed after a failed copy.
  ready_ = false;
  std::size_t bytes = 0;

  std::vector<void*> destinations;
  std::vector<const void*> sources;
  std::vector<std::size_t> sizes;

  for (const auto& [slot, id] : replacements) {
    for (const auto& [name, parameter] :
         source_.expert_at(id).named_parameters()) {
      const auto source = parameter->view();
      const auto destination = slots_[slot]->parameter(name).view();

      destinations.push_back(destination.data());
      sources.push_back(source.data());

      sizes.push_back(source.size_bytes());
      bytes += source.size_bytes();
    }
  }

  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
  std::size_t first = 0;
  check_cuda(
      cudaMemcpyBatchAsync(destinations.data(), sources.data(), sizes.data(),
                           sizes.size(), &attributes, &first, 1, stream),
      "batch replace GPU expert weights");

  for (const auto& [slot, id] : replacements) {
    expert_to_slot_[resident_ids_[slot]] = -1;
    resident_ids_[slot] = id;
    expert_to_slot_[id] = static_cast<int>(slot);
  }

  ready_ = true;

  publish_routes(stream);
  return bytes;
}

int ExpertCache::slot_of(std::size_t expert_id) const {
  return expert_to_slot_[expert_id];
}

const FP8Expert& ExpertCache::expert_at_slot(std::size_t slot) const {
  assert(ready());
  return *slots_.at(slot);
}

bool ExpertCache::ready() const noexcept {
  return ready_;
}

bool ExpertCache::all_experts_resident() const noexcept {
  return num_slots() >= expert_to_slot_.size()
         && std::ranges::all_of(expert_to_slot_,
                                [](int slot) { return slot >= 0; });
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

} // namespace nanovllm
