#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/tensor_layout.hh"

namespace nanovllm {

template <bool IsConst>
class BasicTensorView {
 public:
  template <typename T>
  using TypedDataPointer = std::conditional_t<IsConst, const T*, T*>;
  using DataPointer = TypedDataPointer<void>;

  BasicTensorView() = default;
  BasicTensorView(DataPointer data, DataType dtype, TensorLayout layout,
                  Device device)
      : data_(static_cast<Byte*>(data)),
        dtype_(dtype),
        layout_(std::move(layout)),
        device_(device) {
    if (data == nullptr && layout_.numel() != 0) {
      throw std::invalid_argument(
          "a non-empty tensor view requires a non-null data pointer");
    }
  }

  template <bool OtherIsConst>
    requires(IsConst && !OtherIsConst)
  BasicTensorView(const BasicTensorView<OtherIsConst>& view)
      : BasicTensorView(view.data(), view.dtype(), view.layout(),
                        view.device()) {}

  [[nodiscard]] DataPointer data() const noexcept { return data_; }

  template <typename T>
  [[nodiscard]] TypedDataPointer<T> data() const noexcept {
    return static_cast<TypedDataPointer<T>>(data());
  }

  [[nodiscard]] DataType dtype() const noexcept { return dtype_; }
  [[nodiscard]] const TensorLayout& layout() const noexcept { return layout_; }
  [[nodiscard]] Device device() const noexcept { return device_; }

  [[nodiscard]] std::span<const std::size_t> shape() const noexcept {
    return layout_.shape();
  }

  [[nodiscard]] std::span<const std::size_t> strides() const noexcept {
    return layout_.strides();
  }

  [[nodiscard]] std::size_t rank() const noexcept { return layout_.rank(); }
  [[nodiscard]] std::size_t numel() const { return layout_.numel(); }

  [[nodiscard]] std::size_t size_bytes() const {
    return layout_.size_bytes(dtype_);
  }

  [[nodiscard]] bool is_contiguous() const noexcept {
    return layout_.is_contiguous();
  }

  [[nodiscard]] BasicTensorView reshape(
      std::span<const std::size_t> shape) const {
    return BasicTensorView(data_, dtype_, layout_.reshape(shape), device_);
  }

  [[nodiscard]] BasicTensorView slice(std::size_t dimension, std::size_t begin,
                                      std::size_t length) const {
    auto sliced_layout = layout_.slice(dimension, begin, length);
    if (sliced_layout.numel() == 0) {
      return BasicTensorView(data_, dtype_, std::move(sliced_layout), device_);
    }

    const auto element_offset = begin * layout_.stride(dimension);
    const auto byte_offset = element_offset * data_type_size(dtype_);

    return BasicTensorView(data_ + byte_offset, dtype_,
                           std::move(sliced_layout), device_);
  }

  [[nodiscard]] BasicTensorView<true> as_const() const
    requires(!IsConst)
  {
    return BasicTensorView<true>(data_, dtype_, layout_, device_);
  }

 private:
  using Byte = std::conditional_t<IsConst, const std::byte, std::byte>;

  Byte* data_ = nullptr;
  DataType dtype_ = DataType::Float32;
  TensorLayout layout_;
  Device device_;
};

using TensorView = BasicTensorView<false>;
using ConstTensorView = BasicTensorView<true>;

template <bool FirstConst, bool... RestConst>
[[nodiscard]] bool same_shape(const BasicTensorView<FirstConst>& first,
                              const BasicTensorView<RestConst>&... rest) {
  return (std::ranges::equal(first.shape(), rest.shape()) && ...);
}

template <bool FirstConst, bool... RestConst>
[[nodiscard]] bool same_dtype(const BasicTensorView<FirstConst>& first,
                              const BasicTensorView<RestConst>&... rest) {
  return ((first.dtype() == rest.dtype()) && ...);
}

template <bool FirstConst, bool... RestConst>
[[nodiscard]] bool same_device(const BasicTensorView<FirstConst>& first,
                               const BasicTensorView<RestConst>&... rest) {
  return ((first.device() == rest.device()) && ...);
}

} // namespace nanovllm
