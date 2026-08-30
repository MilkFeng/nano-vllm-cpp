#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "model/device.hh"

namespace nanovllm {

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

} // namespace nanovllm
