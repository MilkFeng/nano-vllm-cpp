#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "core/storage.hh"

namespace nanovllm {

// Reusable temporary storage for one CUDA stream and one submitting host thread.
// A block can be reused after its last Tensor owner is released: previously
// submitted uses precede its next use on the same stream. Persistent tensors and
// Tensor slices retain ownership, so their blocks are never handed out again.
class TensorPool final {
 public:
  TensorPool(Device device, cudaStream_t stream);

  TensorPool(const TensorPool&) = delete;
  TensorPool& operator=(const TensorPool&) = delete;

  class Scope final {
   public:
    Scope(TensorPool& pool, cudaStream_t stream);
    ~Scope() noexcept;

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    TensorPool& pool_;
    TensorPool* previous_;
    int exceptions_on_entry_;
  };

  [[nodiscard]] static TensorPool* current() noexcept;
  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] std::shared_ptr<Storage> acquire(std::size_t bytes);
  [[nodiscard]] std::size_t reserved_bytes() const noexcept;
  [[nodiscard]] std::size_t allocation_count() const noexcept;

 private:
  static thread_local TensorPool* current_;

  Device device_;
  cudaStream_t stream_;
  std::vector<std::shared_ptr<Storage>> blocks_;
  std::size_t reserved_bytes_ = 0;
};

} // namespace nanovllm
