#include "kernels/cpu/fp8.hh"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nanovllm::kernels::cpu {
namespace {

template <bool IsConst>
void validate_matrix(const BasicTensorView<IsConst>& matrix, DataType dtype) {
  if (matrix.device().type != DeviceType::Cpu) {
    throw std::invalid_argument("FP8 dequantization requires CPU tensors");
  }
  if (matrix.dtype() != dtype) {
    throw std::invalid_argument(
        "incorrect tensor dtype for FP8 dequantization");
  }
  if (matrix.rank() != 2 || !matrix.is_contiguous()) {
    throw std::invalid_argument(
        "FP8 dequantization requires contiguous matrices");
  }
  if (matrix.shape()[0] == 0 || matrix.shape()[1] == 0) {
    throw std::invalid_argument(
        "FP8 dequantization requires nonempty matrices");
  }
}

} // namespace

float fp8_e4m3fn_to_float(std::uint8_t bits) {
  const auto exponent = (bits >> 3) & 0xf;
  const auto fraction = bits & 0x7;
  float value;
  if (exponent == 15 && fraction == 7) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else if (exponent == 0) {
    // Subnormals: (fraction / 8) * 2^(1 - bias).
    value = std::ldexp(static_cast<float>(fraction), -9);
  } else {
    value =
        std::ldexp(1.0f + static_cast<float>(fraction) / 8.0f, exponent - 7);
  }
  return (bits & 0x80) != 0 ? -value : value;
}

void dequantize_fp8(ConstTensorView weight, ConstTensorView scale_inv,
                    TensorView output, std::size_t block_rows,
                    std::size_t block_columns) {
  if (block_rows == 0 || block_columns == 0) {
    throw std::invalid_argument("FP8 block dimensions must be positive");
  }
  validate_matrix(weight, DataType::Float8E4M3FN);
  validate_matrix(scale_inv, DataType::Float32);
  validate_matrix(output, DataType::Float32);
  if (!same_device(weight, scale_inv, output)) {
    throw std::invalid_argument("FP8 tensors must be on the same device");
  }
  if (!same_shape(weight, output)) {
    throw std::invalid_argument("FP8 output shape must match the weight");
  }
  const auto rows = weight.shape()[0];
  const auto columns = weight.shape()[1];
  const auto scale_rows = rows / block_rows + (rows % block_rows != 0);
  const auto scale_columns =
      columns / block_columns + (columns % block_columns != 0);
  if (scale_inv.shape()[0] != scale_rows
      || scale_inv.shape()[1] != scale_columns) {
    throw std::invalid_argument(
        "FP8 scale shape does not match the weight and block dimensions");
  }

  // Decode the 256 byte values once, then use the table for every matrix.
  static const auto values = [] {
    std::array<float, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
      table[i] = fp8_e4m3fn_to_float(static_cast<std::uint8_t>(i));
    }
    return table;
  }();
  const auto* weights = weight.data<std::uint8_t>();
  const auto* scales = scale_inv.data<float>();
  auto* result = output.data<float>();
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const auto index = row * columns + column;
      const auto scale_index =
          (row / block_rows) * scale_columns + column / block_columns;
      result[index] = values[weights[index]] * scales[scale_index];
    }
  }
}

} // namespace nanovllm::kernels::cpu
