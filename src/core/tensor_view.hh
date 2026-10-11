#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>

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
                  Device device);

  template <bool OtherIsConst>
    requires(IsConst && !OtherIsConst)
  explicit BasicTensorView(const BasicTensorView<OtherIsConst>& view)
      : BasicTensorView(view.data(), view.dtype(), view.layout(),
                        view.device()) {}

  [[nodiscard]] DataPointer data() const noexcept;

  template <typename T>
  [[nodiscard]] TypedDataPointer<T> data() const noexcept {
    return static_cast<TypedDataPointer<T>>(data());
  }

  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] const TensorLayout& layout() const noexcept;
  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept;
  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] std::size_t numel() const;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] bool is_contiguous() const noexcept;

  [[nodiscard]] BasicTensorView reshape(
      std::span<const std::size_t> shape) const;
  [[nodiscard]] BasicTensorView slice(std::size_t dimension, std::size_t begin,
                                      std::size_t length) const;
  [[nodiscard]] BasicTensorView<true> as_const() const
    requires(!IsConst);

 private:
  using Byte = std::conditional_t<IsConst, const std::byte, std::byte>;

  Byte* data_ = nullptr;
  DataType dtype_ = DataType::Float32;
  TensorLayout layout_;
  Device device_;
};

extern template class BasicTensorView<false>;
extern template class BasicTensorView<true>;

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
