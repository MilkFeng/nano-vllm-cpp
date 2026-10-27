#include "kernels/cpu/expert.hh"

#include <cassert>
#include <cmath>
#include <vector>

#include "kernels/cpu/fp8_dot.hh"

namespace nanovllm::kernels::cpu {

namespace {

std::size_t scale_dimension(std::size_t dimension, std::size_t block_size) {
  return dimension / block_size + (dimension % block_size != 0);
}

template <bool IsConst>
void validate_matrix(const BasicTensorView<IsConst>& matrix, DataType dtype,
                     std::size_t rows, std::size_t columns) {
  (void)matrix;
  (void)dtype;
  (void)rows;
  (void)columns;

  assert(matrix.dtype() == dtype);
  assert(matrix.is_contiguous());
  assert(matrix.rank() == 2);
  assert(matrix.shape()[0] == rows && matrix.shape()[1] == columns);
}

} // namespace

void expert_forward(ConstTensorView input, const ExpertWeights& weights,
                    TensorView output, std::size_t block_size) {
  assert(input.rank() == 2 && input.shape()[1] > 0);
  assert(weights.gate.rank() == 2 && weights.gate.shape()[0] > 0);

  const auto tokens = input.shape()[0];
  const auto hidden_size = input.shape()[1];
  const auto intermediate_size = weights.gate.shape()[0];
  const auto hidden_blocks = scale_dimension(hidden_size, block_size);
  const auto intermediate_blocks =
      scale_dimension(intermediate_size, block_size);

  assert(input.device().type == DeviceType::Cpu);
  assert(same_device(input, output, weights.gate, weights.up, weights.down,
                     weights.gate_scale, weights.up_scale, weights.down_scale));

  validate_matrix(input, DataType::Float32, tokens, hidden_size);
  validate_matrix(output, DataType::Float32, tokens, hidden_size);
  validate_matrix(weights.gate, DataType::Float8E4M3FN, intermediate_size,
                  hidden_size);
  validate_matrix(weights.up, DataType::Float8E4M3FN, intermediate_size,
                  hidden_size);
  validate_matrix(weights.down, DataType::Float8E4M3FN, hidden_size,
                  intermediate_size);
  validate_matrix(weights.gate_scale, DataType::Float32, intermediate_blocks,
                  hidden_blocks);
  validate_matrix(weights.up_scale, DataType::Float32, intermediate_blocks,
                  hidden_blocks);
  validate_matrix(weights.down_scale, DataType::Float32, hidden_blocks,
                  intermediate_blocks);

  thread_local std::vector<float> activated;
  activated.resize(tokens * intermediate_size);

  const auto* x = input.data<float>();
  const auto* gate = weights.gate.data<std::uint8_t>();
  const auto* up = weights.up.data<std::uint8_t>();
  const auto* gate_scale = weights.gate_scale.data<float>();
  const auto* up_scale = weights.up_scale.data<float>();
  const auto* down = weights.down.data<std::uint8_t>();
  const auto* down_scale = weights.down_scale.data<float>();
  auto* result = output.data<float>();

  // Keep the original dot products and storage path for single-token decode.
  if (tokens == 1) {
    for (std::size_t row = 0; row < intermediate_size; ++row) {
      const auto scale_offset = (row / block_size) * hidden_blocks;
      const float g =
          fp8_dot_kernel(gate + row * hidden_size, gate_scale + scale_offset, x,
                         hidden_size, block_size);
      const float u =
          fp8_dot_kernel(up + row * hidden_size, up_scale + scale_offset, x,
                         hidden_size, block_size);
      activated[row] = (g / (1.0f + std::exp(-g))) * u;
    }
    for (std::size_t row = 0; row < hidden_size; ++row) {
      result[row] =
          fp8_dot_kernel(down + row * intermediate_size,
                         down_scale + (row / block_size) * intermediate_blocks,
                         activated.data(), intermediate_size, block_size);
    }
    return;
  }

  for (std::size_t row = 0; row < intermediate_size; ++row) {
    const auto scale_offset = (row / block_size) * hidden_blocks;
    for (std::size_t token = 0; token < tokens;) {
      const auto remaining = tokens - token;
      const std::size_t rows = remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
      const auto* input_row = x + token * hidden_size;
      float g[4], u[4];
      if (rows == 1) {
        g[0] =
            fp8_dot_kernel(gate + row * hidden_size, gate_scale + scale_offset,
                           input_row, hidden_size, block_size);
        u[0] = fp8_dot_kernel(up + row * hidden_size, up_scale + scale_offset,
                              input_row, hidden_size, block_size);
      } else {
        fp8_dot_batch_kernel(gate + row * hidden_size,
                             gate_scale + scale_offset, input_row, hidden_size,
                             rows, hidden_size, block_size, g);
        fp8_dot_batch_kernel(up + row * hidden_size, up_scale + scale_offset,
                             input_row, hidden_size, rows, hidden_size,
                             block_size, u);
      }
      for (std::size_t b = 0; b < rows; ++b) {
        activated[(token + b) * intermediate_size + row] =
            (g[b] / (1.0f + std::exp(-g[b]))) * u[b];
      }
      token += rows;
    }
  }

  for (std::size_t row = 0; row < hidden_size; ++row) {
    const auto* weight_row = down + row * intermediate_size;
    const auto* scale_row =
        down_scale + (row / block_size) * intermediate_blocks;
    for (std::size_t token = 0; token < tokens;) {
      const auto remaining = tokens - token;
      const std::size_t rows = remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
      const auto* input_row = activated.data() + token * intermediate_size;
      float values[4];
      if (rows == 1) {
        values[0] = fp8_dot_kernel(weight_row, scale_row, input_row,
                                   intermediate_size, block_size);
      } else {
        fp8_dot_batch_kernel(weight_row, scale_row, input_row,
                             intermediate_size, rows, intermediate_size,
                             block_size, values);
      }
      for (std::size_t b = 0; b < rows; ++b) {
        result[(token + b) * hidden_size + row] = values[b];
      }
      token += rows;
    }
  }
}

} // namespace nanovllm::kernels::cpu
