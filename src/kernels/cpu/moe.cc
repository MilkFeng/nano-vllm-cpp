#include "kernels/cpu/moe.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nanovllm::kernels::cpu {
namespace {

void validate_matrix(ConstTensorView matrix, std::size_t rows,
                     std::size_t columns) {
  if (matrix.device() != Device::cpu() || matrix.dtype() != DataType::Float32
      || !matrix.is_contiguous() || matrix.rank() != 2
      || matrix.shape()[0] != rows || matrix.shape()[1] != columns) {
    throw std::invalid_argument(
        "MoE reference needs contiguous CPU Float32 matrices of matching "
        "shape");
  }
}

void validate_expert(const ExpertWeights& expert, std::size_t hidden_size) {
  if (expert.gate.rank() != 2 || expert.gate.shape()[0] == 0) {
    throw std::invalid_argument("invalid MoE expert intermediate dimension");
  }
  const auto intermediate_size = expert.gate.shape()[0];
  validate_matrix(expert.gate, intermediate_size, hidden_size);
  validate_matrix(expert.up, intermediate_size, hidden_size);
  validate_matrix(expert.down, hidden_size, intermediate_size);
}

std::vector<float> matvec(ConstTensorView weight,
                          std::span<const float> input) {
  std::vector<float> result(weight.shape()[0]);
  const auto* data = weight.data<float>();
  for (std::size_t row = 0; row < result.size(); ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < input.size(); ++column) {
      sum += static_cast<double>(data[row * input.size() + column])
             * input[column];
    }
    result[row] = static_cast<float>(sum);
  }
  return result;
}

std::vector<float> expert_forward(const ExpertWeights& weights,
                                  std::span<const float> input) {
  auto gate = matvec(weights.gate, input);
  const auto up = matvec(weights.up, input);
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const double g = gate[i];
    gate[i] = static_cast<float>((g / (1.0 + std::exp(-g))) * up[i]);
  }
  return matvec(weights.down, gate);
}

} // namespace

MoEReferenceOutput moe_reference(
    ConstTensorView input, ConstTensorView router_weight,
    std::span<const ExpertWeights> experts, std::size_t top_k,
    std::optional<SharedExpertWeights> shared_expert) {
  if (input.rank() != 2 || input.shape()[0] != 1 || input.shape()[1] == 0
      || experts.empty() || top_k == 0 || top_k > experts.size()) {
    throw std::invalid_argument(
        "invalid single-token MoE reference dimensions");
  }
  const auto hidden_size = input.shape()[1];
  validate_matrix(input, 1, hidden_size);
  validate_matrix(router_weight, experts.size(), hidden_size);
  for (const auto& expert : experts) validate_expert(expert, hidden_size);
  if (shared_expert) {
    validate_expert(shared_expert->expert, hidden_size);
    validate_matrix(shared_expert->gate_weight, 1, hidden_size);
  }

  const std::span<const float> x(input.data<float>(), hidden_size);
  MoEReferenceOutput result;
  result.router_logits = matvec(router_weight, x);
  for (const auto score : result.router_logits) {
    if (!std::isfinite(score)) {
      throw std::invalid_argument("MoE reference needs finite router logits");
    }
  }

  const auto maximum = *std::ranges::max_element(result.router_logits);
  std::vector<double> probabilities(experts.size());
  double denominator = 0.0;
  for (std::size_t e = 0; e < experts.size(); ++e) {
    probabilities[e] =
        std::exp(static_cast<double>(result.router_logits[e]) - maximum);
    denominator += probabilities[e];
  }
  for (auto& probability : probabilities) probability /= denominator;

  // Sort instead of copying the GPU's repeated-scan selection algorithm.
  result.expert_ids.resize(experts.size());
  std::iota(result.expert_ids.begin(), result.expert_ids.end(), 0);
  std::ranges::sort(result.expert_ids, [&](auto a, auto b) {
    const auto left = result.router_logits[a], right = result.router_logits[b];
    return left == right ? a < b : left > right;
  });
  result.expert_ids.resize(top_k);
  double selected_sum = 0.0;
  for (const auto id : result.expert_ids) selected_sum += probabilities[id];

  result.expert_weights.resize(top_k);
  result.expert_outputs.reserve(top_k * hidden_size);
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    const auto id = result.expert_ids[rank];
    result.expert_weights[rank] =
        static_cast<float>(probabilities[id] / selected_sum);
    const auto part = expert_forward(experts[id], x);
    result.expert_outputs.insert(result.expert_outputs.end(), part.begin(),
                                 part.end());
  }

  result.hidden_states.resize(hidden_size);
  for (std::size_t d = 0; d < hidden_size; ++d) {
    double sum = 0.0;
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      sum += static_cast<double>(result.expert_weights[rank])
             * result.expert_outputs[rank * hidden_size + d];
    }
    result.hidden_states[d] = static_cast<float>(sum);
  }
  if (shared_expert) {
    result.shared_expert_output = expert_forward(shared_expert->expert, x);
    result.shared_gate_logits = matvec(shared_expert->gate_weight, x);
    const double logit = result.shared_gate_logits[0];
    if (!std::isfinite(logit)) {
      throw std::invalid_argument(
          "MoE reference needs a finite shared gate logit");
    }
    const double exponential = std::exp(-std::abs(logit));
    const double weight = logit >= 0.0 ? 1.0 / (1.0 + exponential)
                                       : exponential / (1.0 + exponential);
    for (std::size_t d = 0; d < hidden_size; ++d) {
      result.hidden_states[d] = static_cast<float>(
          result.hidden_states[d] + weight * result.shared_expert_output[d]);
    }
  }
  return result;
}

} // namespace nanovllm::kernels::cpu
