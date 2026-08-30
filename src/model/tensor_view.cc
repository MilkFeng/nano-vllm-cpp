#include "model/tensor_view.hh"

#include <stdexcept>
#include <utility>

namespace nanovllm {
template <bool IsConst>
BasicTensorView<IsConst>::BasicTensorView(DataPointer data, DataType dtype,
                                          TensorLayout layout, Device device)
    : data_(static_cast<Byte*>(data)),
      dtype_(dtype),
      layout_(std::move(layout)),
      device_(device) {
  if (!device.is_valid()) {
    throw std::invalid_argument("tensor device is invalid");
  }

  if (data == nullptr && layout_.numel() != 0) {
    throw std::invalid_argument(
        "a non-empty tensor view requires a non-null data pointer");
  }
}

template <bool IsConst>
BasicTensorView<IsConst>::DataPointer BasicTensorView<IsConst>::data()
    const noexcept {
  return data_;
}

template <bool IsConst>
DataType BasicTensorView<IsConst>::dtype() const noexcept {
  return dtype_;
}

template <bool IsConst>
const TensorLayout& BasicTensorView<IsConst>::layout() const noexcept {
  return layout_;
}

template <bool IsConst>
Device BasicTensorView<IsConst>::device() const noexcept {
  return device_;
}

template <bool IsConst>
std::span<const std::size_t> BasicTensorView<IsConst>::shape() const noexcept {
  return layout_.shape();
}

template <bool IsConst>
std::span<const std::size_t> BasicTensorView<IsConst>::strides()
    const noexcept {
  return layout_.strides();
}

template <bool IsConst>
std::size_t BasicTensorView<IsConst>::rank() const noexcept {
  return layout_.rank();
}

template <bool IsConst>
std::size_t BasicTensorView<IsConst>::numel() const {
  return layout_.numel();
}

template <bool IsConst>
std::size_t BasicTensorView<IsConst>::size_bytes() const {
  return layout_.size_bytes(dtype_);
}

template <bool IsConst>
bool BasicTensorView<IsConst>::is_contiguous() const noexcept {
  return layout_.is_contiguous();
}

template <bool IsConst>
BasicTensorView<IsConst> BasicTensorView<IsConst>::reshape(
    std::span<const std::size_t> shape) const {
  return BasicTensorView(data_, dtype_, layout_.reshape(shape), device_);
}

template <bool IsConst>
BasicTensorView<IsConst> BasicTensorView<IsConst>::slice(
    std::size_t dimension, std::size_t begin, std::size_t length) const {
  auto sliced_layout = layout_.slice(dimension, begin, length);
  if (sliced_layout.numel() == 0) {
    return BasicTensorView(data_, dtype_, std::move(sliced_layout), device_);
  }

  const auto element_offset = begin * layout_.stride(dimension);
  const auto byte_offset = element_offset * data_type_size(dtype_);

  return BasicTensorView(data_ + byte_offset, dtype_, std::move(sliced_layout),
                         device_);
}

template <bool IsConst>
BasicTensorView<true> BasicTensorView<IsConst>::as_const() const
  requires(!IsConst)
{
  return BasicTensorView<true>(data_, dtype_, layout_, device_);
}

template class BasicTensorView<false>;
template class BasicTensorView<true>;

} // namespace nanovllm
