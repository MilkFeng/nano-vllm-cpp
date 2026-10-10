#pragma once

#include <cstddef>

#include "core/device.hh"

namespace nanovllm {

class Storage {
 public:
  Storage(std::size_t size_bytes, Device device);
  ~Storage() noexcept;

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const noexcept;
  [[nodiscard]] Device device() const noexcept;

 private:
  void* data_ = nullptr;
  std::size_t size_bytes_ = 0;
  Device device_;
};

} // namespace nanovllm
