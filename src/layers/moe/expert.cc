#include "layers/moe/expert.hh"

#include <cstdint>
#include <string>
#include <utility>

#include "kernels/cpu/expert.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/moe/workspace.hh"

namespace nanovllm {

namespace {

constexpr std::size_t kExpertFP8BlockSize = 128;

} // namespace

FP8Expert::FP8Expert(std::size_t hidden_size, std::size_t intermediate_size,
                     Device device)
    : gate_(hidden_size, intermediate_size, device),
      up_(hidden_size, intermediate_size, device),
      down_(intermediate_size, hidden_size, device) {
  register_module("gate_proj", gate_);
  register_module("up_proj", up_);
  register_module("down_proj", down_);
}

void FP8Expert::forward_into(ConstTensorView input, TensorView output) const {
  kernels::cpu::expert_forward(input,
                               {
                                   gate_.weight().view(),
                                   up_.weight().view(),
                                   down_.weight().view(),
                                   gate_.weight_scale_inv().view(),
                                   up_.weight_scale_inv().view(),
                                   down_.weight_scale_inv().view(),
                               },
                               output, kExpertFP8BlockSize);
}

void FP8Expert::forward_into(ConstTensorView input, TensorView output,
                             FP8ExpertWorkspace& workspace,
                             cudaStream_t stream) const {
  const auto rows = input.shape()[0];
  workspace.reserve(rows, gate_.out_features(), input.dtype());
  const auto gate = workspace.gate.view().slice(0, 0, rows);
  const auto up = workspace.up.view().slice(0, 0, rows);
  const auto activated = workspace.activated.view().slice(0, 0, rows);

  gate_.forward_into(input, gate, stream);
  up_.forward_into(input, up, stream);
  kernels::cuda::silu_and_mul(gate, up, activated, stream);
  down_.forward_into(activated, output, stream);
}

kernels::cuda::FP8ExpertWeights FP8Expert::device_weights() const {
  return {
      gate_.weight().view().data<std::uint8_t>(),
      up_.weight().view().data<std::uint8_t>(),
      down_.weight().view().data<std::uint8_t>(),
      gate_.weight_scale_inv().view().data<float>(),
      up_.weight_scale_inv().view().data<float>(),
      down_.weight_scale_inv().view().data<float>(),
  };
}

ExpertStore::ExpertStore(std::size_t hidden_size, std::size_t intermediate_size,
                         std::size_t num_experts)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size) {
  experts_.reserve(num_experts);
  for (std::size_t id = 0; id < num_experts; ++id) {
    auto expert = std::make_unique<FP8Expert>(hidden_size, intermediate_size);

    register_module(std::to_string(id), *expert);
    experts_.push_back(std::move(expert));
  }
}

const FP8Expert& ExpertStore::expert_at(std::size_t expert_id) const {
  return *experts_.at(expert_id);
}

std::size_t ExpertStore::hidden_size() const noexcept {
  return hidden_size_;
}

std::size_t ExpertStore::intermediate_size() const noexcept {
  return intermediate_size_;
}

std::size_t ExpertStore::num_experts() const noexcept {
  return experts_.size();
}

std::size_t ExpertStore::size_bytes() const {
  std::size_t total = 0;
  for (const auto& [name, parameter] : named_parameters()) {
    total += parameter->view().size_bytes();
  }
  return total;
}

} // namespace nanovllm
