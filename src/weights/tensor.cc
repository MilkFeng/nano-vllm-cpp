#include "weights/tensor.hh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nanovllm {
namespace {

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(message);
  }
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(message);
  }
  return left + right;
}

bool has_zero_dimension(std::span<const std::size_t> shape) noexcept {
  return std::ranges::find(shape, std::size_t{0}) != shape.end();
}

std::size_t compute_numel(std::span<const std::size_t> shape) {
  if (has_zero_dimension(shape)) return 0;

  std::size_t count = 1;
  for (const auto dimension : shape) {
    count =
        checked_multiply(count, dimension, "tensor element count is too large");
  }
  return count;
}

std::vector<std::size_t> make_contiguous_strides(
    std::span<const std::size_t> shape) {
  std::vector<std::size_t> strides(shape.size());
  std::size_t stride = 1;

  for (std::size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride =
        checked_multiply(stride, shape[i - 1], "tensor layout is too large");
  }
  return strides;
}

void validate_strided_layout(std::span<const std::size_t> shape,
                             std::span<const std::size_t> strides) {
  if (shape.size() != strides.size()) {
    throw std::invalid_argument(
        "tensor shape and strides must have equal rank");
  }

  (void)compute_numel(shape);
  if (has_zero_dimension(shape)) return;

  std::size_t maximum_offset = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 1 && strides[i] == 0) {
      throw std::invalid_argument(
          "zero strides are not supported for non-empty dimensions");
    }

    const auto dimension_offset = checked_multiply(
        shape[i] - 1, strides[i], "tensor layout offset is too large");
    maximum_offset = checked_add(maximum_offset, dimension_offset,
                                 "tensor layout offset is too large");
  }
}

void validate_slice(const TensorLayout& layout, std::size_t dimension,
                    std::size_t begin, std::size_t length) {
  if (dimension >= layout.rank()) {
    throw std::out_of_range("tensor slice dimension is out of range");
  }

  const auto dimension_size = layout.dimension(dimension);
  if (begin > dimension_size || length > dimension_size - begin) {
    throw std::out_of_range("tensor slice range is out of bounds");
  }
}

std::size_t slice_byte_offset(const TensorLayout& layout, DataType dtype,
                              std::size_t dimension, std::size_t begin) {
  const auto element_offset = checked_multiply(
      begin, layout.stride(dimension), "tensor slice offset is too large");
  return checked_multiply(element_offset, data_type_size(dtype),
                          "tensor slice byte offset is too large");
}

template <typename Byte>
Byte* offset_pointer(Byte* data, std::size_t byte_offset) {
  if (byte_offset == 0) return data;
  if (data == nullptr) {
    throw std::logic_error("cannot offset a null tensor data pointer");
  }
  return data + byte_offset;
}

void validate_view(const void* data, DataType dtype,
                   const TensorLayout& layout) {
  (void)layout.size_bytes(dtype);
  if (data == nullptr && layout.numel() != 0) {
    throw std::invalid_argument(
        "a non-empty tensor view requires a non-null data pointer");
  }
}

} // namespace

std::size_t data_type_size(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
    case DataType::Int32:
      return 4;
    case DataType::Float16:
    case DataType::BFloat16:
      return 2;
  }
  throw std::invalid_argument("unknown tensor data type");
}

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
  validate_strided_layout(shape_, strides_);
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
  return checked_multiply(numel(), data_type_size(dtype),
                          "tensor byte size is too large");
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
  validate_slice(*this, dimension, begin, length);

  auto sliced_shape = shape_;
  sliced_shape[dimension] = length;
  return TensorLayout(std::move(sliced_shape), strides_);
}

TensorView::TensorView(void* data, DataType dtype, TensorLayout layout)
    : data_(static_cast<std::byte*>(data)),
      dtype_(dtype),
      layout_(std::move(layout)) {
  validate_view(data_, dtype_, layout_);
}

void* TensorView::data() const noexcept {
  return data_;
}

DataType TensorView::dtype() const noexcept {
  return dtype_;
}

const TensorLayout& TensorView::layout() const noexcept {
  return layout_;
}

std::span<const std::size_t> TensorView::shape() const noexcept {
  return layout_.shape();
}

std::span<const std::size_t> TensorView::strides() const noexcept {
  return layout_.strides();
}

std::size_t TensorView::rank() const noexcept {
  return layout_.rank();
}

std::size_t TensorView::numel() const {
  return layout_.numel();
}

std::size_t TensorView::size_bytes() const {
  return layout_.size_bytes(dtype_);
}

bool TensorView::is_contiguous() const noexcept {
  return layout_.is_contiguous();
}

TensorView TensorView::reshape(std::span<const std::size_t> shape) const {
  return TensorView(data_, dtype_, layout_.reshape(shape));
}

TensorView TensorView::slice(std::size_t dimension, std::size_t begin,
                             std::size_t length) const {
  auto sliced_layout = layout_.slice(dimension, begin, length);
  const auto byte_offset = slice_byte_offset(layout_, dtype_, dimension, begin);
  return TensorView(offset_pointer(data_, byte_offset), dtype_,
                    std::move(sliced_layout));
}

ConstTensorView TensorView::as_const() const {
  return ConstTensorView(data_, dtype_, layout_);
}

ConstTensorView::ConstTensorView(const void* data, DataType dtype,
                                 TensorLayout layout)
    : data_(static_cast<const std::byte*>(data)),
      dtype_(dtype),
      layout_(std::move(layout)) {
  validate_view(data_, dtype_, layout_);
}

ConstTensorView::ConstTensorView(const TensorView& view)
    : ConstTensorView(view.data(), view.dtype(), view.layout()) {}

const void* ConstTensorView::data() const noexcept {
  return data_;
}

DataType ConstTensorView::dtype() const noexcept {
  return dtype_;
}

const TensorLayout& ConstTensorView::layout() const noexcept {
  return layout_;
}

std::span<const std::size_t> ConstTensorView::shape() const noexcept {
  return layout_.shape();
}

std::span<const std::size_t> ConstTensorView::strides() const noexcept {
  return layout_.strides();
}

std::size_t ConstTensorView::rank() const noexcept {
  return layout_.rank();
}

std::size_t ConstTensorView::numel() const {
  return layout_.numel();
}

std::size_t ConstTensorView::size_bytes() const {
  return layout_.size_bytes(dtype_);
}

bool ConstTensorView::is_contiguous() const noexcept {
  return layout_.is_contiguous();
}

ConstTensorView ConstTensorView::reshape(
    std::span<const std::size_t> shape) const {
  return ConstTensorView(data_, dtype_, layout_.reshape(shape));
}

ConstTensorView ConstTensorView::slice(std::size_t dimension, std::size_t begin,
                                       std::size_t length) const {
  auto sliced_layout = layout_.slice(dimension, begin, length);
  const auto byte_offset = slice_byte_offset(layout_, dtype_, dimension, begin);
  return ConstTensorView(offset_pointer(data_, byte_offset), dtype_,
                         std::move(sliced_layout));
}

} // namespace nanovllm
