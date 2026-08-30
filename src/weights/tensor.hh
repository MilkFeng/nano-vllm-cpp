#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nanovllm {

enum class DataType : std::uint8_t { Float32, Float16, BFloat16, Int32 };

[[nodiscard]] std::size_t data_type_size(DataType dtype);

class TensorLayout {
 public:
  // A default layout represents an empty one-dimensional tensor.
  TensorLayout();

  explicit TensorLayout(std::vector<std::size_t> shape);
  TensorLayout(std::vector<std::size_t> shape,
               std::vector<std::size_t> strides);

  [[nodiscard]] static TensorLayout contiguous(
      std::span<const std::size_t> shape);

  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept;
  [[nodiscard]] std::size_t dimension(std::size_t index) const;
  [[nodiscard]] std::size_t stride(std::size_t index) const;

  [[nodiscard]] std::size_t numel() const;
  [[nodiscard]] std::size_t size_bytes(DataType dtype) const;
  [[nodiscard]] bool is_contiguous() const noexcept;

  [[nodiscard]] TensorLayout reshape(std::span<const std::size_t> shape) const;
  [[nodiscard]] TensorLayout slice(std::size_t dimension, std::size_t begin,
                                   std::size_t length) const;

 private:
  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
};

class ConstTensorView;

class TensorView {
 public:
  TensorView() = default;
  TensorView(void* data, DataType dtype, TensorLayout layout);

  [[nodiscard]] void* data() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] const TensorLayout& layout() const noexcept;
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept;
  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] std::size_t numel() const;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] bool is_contiguous() const noexcept;

  [[nodiscard]] TensorView reshape(std::span<const std::size_t> shape) const;
  [[nodiscard]] TensorView slice(std::size_t dimension, std::size_t begin,
                                 std::size_t length) const;
  [[nodiscard]] ConstTensorView as_const() const;

 private:
  std::byte* data_ = nullptr;
  DataType dtype_ = DataType::Float32;
  TensorLayout layout_;
};

class ConstTensorView {
 public:
  ConstTensorView() = default;
  ConstTensorView(const void* data, DataType dtype, TensorLayout layout);
  explicit ConstTensorView(const TensorView& view);

  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] const TensorLayout& layout() const noexcept;
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept;
  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] std::size_t numel() const;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] bool is_contiguous() const noexcept;

  [[nodiscard]] ConstTensorView reshape(
      std::span<const std::size_t> shape) const;
  [[nodiscard]] ConstTensorView slice(std::size_t dimension, std::size_t begin,
                                      std::size_t length) const;

 private:
  const std::byte* data_ = nullptr;
  DataType dtype_ = DataType::Float32;
  TensorLayout layout_;
};

} // namespace nanovllm
