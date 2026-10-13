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
  assert(input.rank() == 2 && input.shape()[0] == 1 && input.shape()[1] > 0);
  assert(weights.gate.rank() == 2 && weights.gate.shape()[0] > 0);

  const auto hidden_size = input.shape()[1];
  const auto intermediate_size = weights.gate.shape()[0];
  const auto hidden_blocks = scale_dimension(hidden_size, block_size);
  const auto intermediate_blocks =
      scale_dimension(intermediate_size, block_size);

  assert(input.device().type == DeviceType::Cpu);
  assert(same_device(input, output, weights.gate, weights.up, weights.down,
                     weights.gate_scale, weights.up_scale, weights.down_scale));

  validate_matrix(input, DataType::Float32, 1, hidden_size);
  validate_matrix(output, DataType::Float32, 1, hidden_size);
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
  activated.resize(intermediate_size);

  const auto* x = input.data<float>();
  const auto* gate = weights.gate.data<std::uint8_t>();
  const auto* up = weights.up.data<std::uint8_t>();
  const auto* gate_scale = weights.gate_scale.data<float>();
  const auto* up_scale = weights.up_scale.data<float>();

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

  const auto* down = weights.down.data<std::uint8_t>();
  const auto* down_scale = weights.down_scale.data<float>();

  auto* result = output.data<float>();
  for (std::size_t row = 0; row < hidden_size; ++row) {
    result[row] =
        fp8_dot_kernel(down + row * intermediate_size,
                       down_scale + (row / block_size) * intermediate_blocks,
                       activated.data(), intermediate_size, block_size);
  }
}

} // namespace nanovllm::kernels::cpu
