#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "core/storage.hh"
#include "core/tensor_view.hh"

namespace nanovllm {

class Tensor {
 public:
  Tensor() = default;
  ~Tensor() = default;

  [[nodiscard]] static Tensor empty(std::vector<std::size_t> shape,
                                    DataType dtype,
                                    Device device = Device::cpu());

  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] const TensorLayout& layout() const noexcept;
  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept;
  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] std::size_t numel() const;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] bool is_contiguous() const noexcept;

  [[nodiscard]] TensorView view();
  [[nodiscard]] ConstTensorView view() const;
  [[nodiscard]] ConstTensorView const_view() const;

  [[nodiscard]] Tensor reshape(std::span<const std::size_t> shape) const;
  [[nodiscard]] Tensor slice(std::size_t dimension, std::size_t begin,
                             std::size_t length) const;

 private:
  Tensor(std::shared_ptr<Storage> storage, std::size_t byte_offset,
         DataType dtype, TensorLayout layout, Device device);

  std::shared_ptr<Storage> storage_;
  std::size_t byte_offset_ = 0;
  DataType dtype_ = DataType::Float32;
  TensorLayout layout_;
  Device device_;
};

} // namespace nanovllm
