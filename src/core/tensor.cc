#include "core/tensor.hh"

namespace nanovllm {

namespace {
std::size_t layout_span_size_bytes(const TensorLayout& layout, DataType dtype) {
  if (layout.numel() == 0) {
    return 0;
  }

  std::size_t maximum_offset = 0;
  for (std::size_t i = 0; i < layout.rank(); ++i) {
    const auto dimension_offset = (layout.dimension(i) - 1) * layout.stride(i);
    maximum_offset = maximum_offset + dimension_offset;
  }
  const auto span_elements = maximum_offset + 1;
  return span_elements * data_type_size(dtype);
}
} // namespace

Tensor::Tensor(std::shared_ptr<Storage> storage, std::size_t byte_offset,
               DataType dtype, TensorLayout layout, Device device)
    : storage_(std::move(storage)),
      byte_offset_(byte_offset),
      dtype_(dtype),
      layout_(std::move(layout)),
      device_(device) {
  if (!device.is_valid()) {
    throw std::invalid_argument("tensor device is invalid");
  }

  if (!storage_) {
    if (byte_offset != 0 || layout_.numel() != 0) {
      throw std::invalid_argument(
          "a non-empty tensor requires allocated storage");
    }
    return;
  }

  if (storage_->device() != device) {
    throw std::invalid_argument(
        "tensor and its storage must be on the same device");
  }

  if (byte_offset > storage_->size_bytes()) {
    throw std::out_of_range("tensor byte offset is outside its storage");
  }

  const auto span_size = layout_span_size_bytes(layout_, dtype);
  if (span_size > storage_->size_bytes() - byte_offset) {
    throw std::out_of_range("tensor layout extends outside its storage");
  }
}

Tensor Tensor::empty(std::vector<std::size_t> shape, DataType dtype,
                     Device device) {
  auto layout = TensorLayout(std::move(shape));
  auto storage = std::make_shared<Storage>(layout.size_bytes(dtype), device);
  return Tensor(std::move(storage), 0, dtype, std::move(layout), device);
}

void* Tensor::data() noexcept {
  if (!storage_) {
    return nullptr;
  }
  return static_cast<std::byte*>(storage_->data()) + byte_offset_;
}

const void* Tensor::data() const noexcept {
  if (!storage_) {
    return nullptr;
  }
  return static_cast<const std::byte*>(storage_->data()) + byte_offset_;
}

DataType Tensor::dtype() const noexcept {
  return dtype_;
}

const TensorLayout& Tensor::layout() const noexcept {
  return layout_;
}

Device Tensor::device() const noexcept {
  return device_;
}

std::span<const std::size_t> Tensor::shape() const noexcept {
  return layout_.shape();
}

std::span<const std::size_t> Tensor::strides() const noexcept {
  return layout_.strides();
}

std::size_t Tensor::rank() const noexcept {
  return layout_.rank();
}

std::size_t Tensor::numel() const {
  return layout_.numel();
}

std::size_t Tensor::size_bytes() const {
  return layout_.size_bytes(dtype_);
}

bool Tensor::is_contiguous() const noexcept {
  return layout_.is_contiguous();
}

TensorView Tensor::view() {
  return TensorView(data(), dtype_, layout_, device_);
}

ConstTensorView Tensor::view() const {
  return ConstTensorView(data(), dtype_, layout_, device_);
}

Tensor Tensor::reshape(std::span<const std::size_t> shape) const {
  return Tensor(storage_, byte_offset_, dtype_, layout_.reshape(shape),
                device_);
}

Tensor Tensor::slice(std::size_t dimension, std::size_t begin,
                     std::size_t length) const {
  auto sliced_layout = layout_.slice(dimension, begin, length);
  if (sliced_layout.numel() == 0) {
    return Tensor(storage_, byte_offset_, dtype_, std::move(sliced_layout),
                  device_);
  }

  const auto relative_offset =
      begin * layout_.stride(dimension) * data_type_size(dtype_);
  const auto byte_offset = byte_offset_ + relative_offset;
  return Tensor(storage_, byte_offset, dtype_, std::move(sliced_layout),
                device_);
}
} // namespace nanovllm
