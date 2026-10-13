#include "kernels/cpu/fp8_dequant.hh"

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace nanovllm::kernels::cpu {
namespace {

template <bool IsConst>
void validate_matrix(const BasicTensorView<IsConst>& matrix, DataType dtype) {
  (void)matrix;
  (void)dtype;

  assert(matrix.device() == Device::cpu());
  assert(matrix.dtype() == dtype);
  assert(matrix.is_contiguous());
  assert(matrix.rank() == 2);
  assert(matrix.shape()[0] != 0 && matrix.shape()[1] != 0);
}

} // namespace

float fp8_e4m3fn_to_float(std::uint8_t bits) {
  const auto exponent = (bits >> 3) & 0xf;
  const auto fraction = bits & 0x7;
  float value;
  if (exponent == 15 && fraction == 7) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else if (exponent == 0) {
    value = std::ldexp(static_cast<float>(fraction), -9);
  } else {
    value =
        std::ldexp(1.0f + static_cast<float>(fraction) / 8.0f, exponent - 7);
  }
  return (bits & 0x80) != 0 ? -value : value;
}

const std::array<float, 256>& fp8_e4m3fn_table() {
  static const auto values = [] {
    std::array<float, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
      table[i] = fp8_e4m3fn_to_float(static_cast<std::uint8_t>(i));
    }
    return table;
  }();
  return values;
}

void dequantize_fp8(ConstTensorView weight, ConstTensorView scale_inv,
                    TensorView output, std::size_t block_rows,
                    std::size_t block_columns) {
  assert(block_rows != 0 && block_columns != 0);

  validate_matrix(weight, DataType::Float8E4M3FN);
  validate_matrix(scale_inv, DataType::Float32);
  validate_matrix(output, DataType::Float32);

  assert(same_device(weight, scale_inv, output));
  assert(same_shape(weight, output));

  const auto rows = weight.shape()[0];
  const auto columns = weight.shape()[1];
  const auto scale_columns =
      columns / block_columns + (columns % block_columns != 0);

  const auto& values = fp8_e4m3fn_table();

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
