#include "kernels/cpu/expert.hh"

#include <bit>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#include "kernels/cpu/fp8_dot.hh"
#include "kernels/cpu/fp8_matvec.hh"

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

const float* float_input(const ConstTensorView& input, std::size_t elements) {
  if (input.dtype() == DataType::Float32) {
    return input.data<float>();
  }

  // Expand once before the dot products reuse each activation across channels.
  constexpr std::size_t alignment = 64;
  const auto capacity = elements + alignment / sizeof(float) - 1;
  thread_local std::vector<float> expanded;
  if (expanded.size() < capacity) {
    expanded.resize(capacity);
  }
  void* storage = expanded.data();
  auto space = expanded.size() * sizeof(float);
  auto* output = static_cast<float*>(
      std::align(alignment, elements * sizeof(float), storage, space));
  assert(output != nullptr);

  const auto* source = input.data<std::uint16_t>();
  for (std::size_t i = 0; i < elements; ++i) {
    output[i] = std::bit_cast<float>(std::uint32_t(source[i]) << 16);
  }
  return output;
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

  assert(input.dtype() == DataType::Float32
         || input.dtype() == DataType::BFloat16);
  validate_matrix(input, input.dtype(), tokens, hidden_size);
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

  const auto* gate = weights.gate.data<std::uint8_t>();
  const auto* up = weights.up.data<std::uint8_t>();
  const auto* gate_scale = weights.gate_scale.data<float>();
  const auto* up_scale = weights.up_scale.data<float>();
  const auto* down = weights.down.data<std::uint8_t>();
  const auto* down_scale = weights.down_scale.data<float>();
  auto* result = output.data<float>();

  if (tokens == 1) {
    const bool used_bf16 =
        input.dtype() == DataType::BFloat16 && block_size == 128
        && hidden_size % 128 == 0 && intermediate_size % 4 == 0
        && try_fp8_gate_up_bf16(gate, up, gate_scale, up_scale,
                                input.data<std::uint16_t>(), intermediate_size,
                                hidden_size, activated.data());
    if (!used_bf16) {
      fp8_gate_up(gate, up, gate_scale, up_scale,
                  float_input(input, hidden_size), intermediate_size,
                  hidden_size, block_size, activated.data());
    }
    fp8_matvec(down, down_scale, activated.data(), hidden_size,
               intermediate_size, block_size, result);
    return;
  }

  const auto* x = float_input(input, tokens * hidden_size);
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
