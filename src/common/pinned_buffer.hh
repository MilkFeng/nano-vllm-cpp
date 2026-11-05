#pragma once

#include <cstddef>

namespace nanovllm {

// Reusable host staging memory. Callers must complete DMA before reuse/destruction.
class PinnedHostBuffer final {
 public:
  explicit PinnedHostBuffer(bool mapped = false) : mapped_(mapped) {}
  ~PinnedHostBuffer() noexcept;

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  void reserve(std::size_t bytes);
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_; }
  [[nodiscard]] void* device_data() const noexcept { return device_data_; }

 private:
  void* data_ = nullptr;
  std::size_t capacity_ = 0;
  void* device_data_ = nullptr;
  bool mapped_ = false;
};

} // namespace nanovllm
