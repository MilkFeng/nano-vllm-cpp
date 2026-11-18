#include "core/tensor_pool.hh"

#include <algorithm>
#include <bit>
#include <exception>
#include <limits>
#include <stdexcept>

namespace nanovllm {

thread_local TensorPool* TensorPool::current_ = nullptr;

TensorPool::TensorPool(Device device, cudaStream_t stream)
    : device_(device),
      stream_(stream) {
  if (!device.is_valid() || device.type != DeviceType::Cuda) {
    throw std::invalid_argument("temporary tensor pool requires a CUDA device");
  }
}

TensorPool::Scope::Scope(TensorPool& pool, cudaStream_t stream)
    : pool_(pool),
      previous_(TensorPool::current_),
      exceptions_on_entry_(std::uncaught_exceptions()) {
  if (stream != pool.stream_) {
    throw std::invalid_argument("temporary tensor pool stream mismatch");
  }
  TensorPool::current_ = &pool;
}

TensorPool::Scope::~Scope() noexcept {
  if (std::uncaught_exceptions() > exceptions_on_entry_) {
    // Pool ownership keeps device buffers alive while local Tensors unwind.
    // Complete DMA before caller-owned host metadata is destroyed as well.
    cudaStreamSynchronize(pool_.stream_);
  }
  TensorPool::current_ = previous_;
}

TensorPool* TensorPool::current() noexcept {
  return current_;
}

Device TensorPool::device() const noexcept {
  return device_;
}

std::shared_ptr<Storage> TensorPool::acquire(std::size_t bytes) {
  std::shared_ptr<Storage>* best = nullptr;
  for (auto& block : blocks_) {
    if (block.use_count() == 1 && block->size_bytes() >= bytes
        && (!best || block->size_bytes() < (*best)->size_bytes())) {
      best = &block;
    }
  }
  if (best) {
    return *best;
  }

  // Size classes bound retained storage when ragged batches grow a few tokens
  // at a time, instead of retaining a separate block for every observed size.
  constexpr auto largest_power =
      std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
  const auto capacity = bytes > largest_power
                            ? bytes
                            : std::bit_ceil(std::max(bytes, std::size_t{256}));
  auto block = std::make_shared<Storage>(capacity, device_);
  blocks_.push_back(block);
  reserved_bytes_ += capacity;
  return block;
}

std::size_t TensorPool::reserved_bytes() const noexcept {
  return reserved_bytes_;
}

std::size_t TensorPool::allocation_count() const noexcept {
  return blocks_.size();
}

} // namespace nanovllm
