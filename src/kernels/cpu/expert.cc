#include "kernels/cpu/expert.hh"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace nanovllm::kernels::cpu {
namespace {

template <bool IsConst>
void validate_matrix(const BasicTensorView<IsConst>& matrix, std::size_t rows,
                     std::size_t columns) {
  if (matrix.device() != Device::cpu() || matrix.dtype() != DataType::Float32
      || !matrix.is_contiguous() || matrix.rank() != 2
      || matrix.shape()[0] != rows || matrix.shape()[1] != columns) {
    throw std::invalid_argument(
        "CPU expert requires contiguous Float32 matrices of matching shape");
  }
}

void matvec(ConstTensorView matrix, std::span<const float> input,
            std::span<float> output) {
  const auto* weights = matrix.data<float>();
  for (std::size_t row = 0; row < output.size(); ++row) {
    float sum = 0.0f;
    for (std::size_t column = 0; column < input.size(); ++column) {
      sum += weights[row * input.size() + column] * input[column];
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
  validate_matrix(input, 1, hidden_size);
  validate_matrix(output, 1, hidden_size);
  validate_matrix(weights.gate, intermediate_size, hidden_size);
  validate_matrix(weights.up, intermediate_size, hidden_size);
  validate_matrix(weights.down, hidden_size, intermediate_size);

  std::vector<float> gate(intermediate_size), up(intermediate_size);
  const std::span<const float> x(input.data<float>(), hidden_size);
  matvec(weights.gate, x, gate);
  matvec(weights.up, x, up);
  for (std::size_t i = 0; i < intermediate_size; ++i) {
    gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
  }
  matvec(weights.down, gate, {output.data<float>(), hidden_size});
}

} // namespace nanovllm::kernels::cpu
