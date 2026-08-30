#include "model/tensor_layout.hh"

#include <limits>
#include <stdexcept>

namespace nanovllm {

namespace {
std::vector<std::size_t> make_contiguous_strides(
    std::span<const std::size_t> shape) {
  std::vector<std::size_t> strides(shape.size());
  std::size_t stride = 1;

  for (std::size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride = stride * shape[i - 1];
  }
  return strides;
}

bool has_zero_dimension(std::span<const std::size_t> shape) noexcept {
  return std::ranges::find(shape, std::size_t{0}) != shape.end();
}

std::size_t compute_numel(std::span<const std::size_t> shape) {
  if (has_zero_dimension(shape)) return 0;

  std::size_t count = 1;
  for (const auto dimension : shape) {
    count = count * dimension;
  }
  return count;
}

} // namespace

TensorLayout::TensorLayout()
    : shape_{0},
      strides_{1} {}

TensorLayout::TensorLayout(std::vector<std::size_t> shape)
    : shape_(std::move(shape)),
      strides_(make_contiguous_strides(shape_)) {}

TensorLayout::TensorLayout(std::vector<std::size_t> shape,
                           std::vector<std::size_t> strides)
    : shape_(std::move(shape)),
      strides_(std::move(strides)) {
  if (shape.size() != strides.size()) {
    throw std::invalid_argument(
        "tensor shape and strides must have equal rank");
  }

  if (has_zero_dimension(shape)) return;

  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 1 && strides[i] == 0) {
      throw std::invalid_argument(
          "zero strides are not supported for non-empty dimensions");
    }
  }
}

TensorLayout TensorLayout::contiguous(std::span<const std::size_t> shape) {
  return TensorLayout({shape.begin(), shape.end()});
}

std::size_t TensorLayout::rank() const noexcept {
  return shape_.size();
}

std::span<const std::size_t> TensorLayout::shape() const noexcept {
  return shape_;
}

std::span<const std::size_t> TensorLayout::strides() const noexcept {
  return strides_;
}

std::size_t TensorLayout::dimension(std::size_t index) const {
  return shape_.at(index);
}

std::size_t TensorLayout::stride(std::size_t index) const {
  return strides_.at(index);
}

std::size_t TensorLayout::numel() const {
  return compute_numel(shape_);
}

std::size_t TensorLayout::size_bytes(DataType dtype) const {
  return numel() * data_type_size(dtype);
}

bool TensorLayout::is_contiguous() const noexcept {
  if (has_zero_dimension(shape_)) return true;

  std::size_t expected_stride = 1;
  for (std::size_t i = shape_.size(); i > 0; --i) {
    const auto dimension = shape_[i - 1];
    if (dimension != 1 && strides_[i - 1] != expected_stride) return false;

    if (dimension != 0
        && expected_stride
               > std::numeric_limits<std::size_t>::max() / dimension) {
      return false;
    }
    expected_stride *= dimension;
  }
  return true;
}

TensorLayout TensorLayout::reshape(std::span<const std::size_t> shape) const {
  if (!is_contiguous()) {
    throw std::invalid_argument("cannot reshape a non-contiguous tensor");
  }

  auto result = TensorLayout::contiguous(shape);
  if (result.numel() != numel()) {
    throw std::invalid_argument(
        "tensor reshape must preserve the element count");
  }
  return result;
}

TensorLayout TensorLayout::slice(std::size_t dimension, std::size_t begin,
                                 std::size_t length) const {
  if (dimension >= rank()) {
    throw std::out_of_range("tensor slice dimension is out of range");
  }

  const auto dimension_size = this->dimension(dimension);
  if (begin > dimension_size || length > dimension_size - begin) {
    throw std::out_of_range("tensor slice range is out of bounds");
  }

  auto sliced_shape = shape_;
  sliced_shape[dimension] = length;
  return TensorLayout(std::move(sliced_shape), strides_);
}

} // namespace nanovllm
