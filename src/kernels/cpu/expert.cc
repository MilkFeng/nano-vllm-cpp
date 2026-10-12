#include "kernels/cpu/expert.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "kernels/cpu/fp8.hh"

namespace nanovllm::kernels::cpu {
namespace {

constexpr std::size_t kBlockSize = 128;

std::size_t scale_dimension(std::size_t dimension) {
  return dimension / kBlockSize + (dimension % kBlockSize != 0);
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

void matvec(ConstTensorView matrix, ConstTensorView scale,
            std::span<const float> input, std::span<float> output) {
  const auto* weights = matrix.data<std::uint8_t>();
  const auto* scales = scale.data<float>();
  const auto& values = fp8_e4m3fn_table();
  const auto scale_columns = scale.shape()[1];

  for (std::size_t row = 0; row < output.size(); ++row) {
    float sum = 0.0f;
    for (std::size_t begin = 0; begin < input.size(); begin += kBlockSize) {
      const float block_scale =
          scales[(row / kBlockSize) * scale_columns + begin / kBlockSize];
      const auto end = std::min(begin + kBlockSize, input.size());
      for (std::size_t column = begin; column < end; ++column) {
        const float weight =
            values[weights[row * input.size() + column]] * block_scale;
        sum += weight * input[column];
      }
    }
    output[row] = sum;
  }
}

} // namespace

void expert_forward(ConstTensorView input, const ExpertWeights& weights,
                    TensorView output) {
  if (input.rank() != 2 || input.shape()[0] != 1 || input.shape()[1] == 0
      || weights.gate.rank() != 2 || weights.gate.shape()[0] == 0) {
    throw std::invalid_argument("invalid single-token CPU expert dimensions");
  }
  const auto hidden_size = input.shape()[1];
  const auto intermediate_size = weights.gate.shape()[0];
  const auto hidden_blocks = scale_dimension(hidden_size);
  const auto intermediate_blocks = scale_dimension(intermediate_size);

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

  std::vector<float> gate(intermediate_size), up(intermediate_size);
  const std::span<const float> x(input.data<float>(), hidden_size);

  matvec(weights.gate, weights.gate_scale, x, gate);
  matvec(weights.up, weights.up_scale, x, up);

  for (std::size_t i = 0; i < intermediate_size; ++i) {
    gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
  }

  matvec(weights.down, weights.down_scale, gate,
         {output.data<float>(), hidden_size});
}

} // namespace nanovllm::kernels::cpu
