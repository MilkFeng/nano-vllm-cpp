#include "layers/moe.hh"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/swiglu_mlp.hh"

namespace nanovllm {
namespace {

MoEConfig validated(MoEConfig config, Device device) {
  config.validate();
  if (device != Device::cpu() && device != Device::cuda(0)) {
    throw std::invalid_argument("MoE supports CPU weights or CUDA device 0");
  }
  return config;
}

} // namespace

void MoEConfig::validate() const {
  if (hidden_size == 0 || intermediate_size == 0 || num_experts == 0
      || num_experts > kernels::cuda::kMaxMoEExperts || top_k == 0
      || top_k > num_experts) {
    throw std::invalid_argument(
        "MoE needs positive dimensions, 1..1024 experts and 1 <= top_k <= E");
  }
}

MoE::MoE(MoEConfig config, Device device)
    : config_(validated(config, device)),
      gate_(config_.hidden_size, config_.num_experts, false, DataType::Float32,
            device) {
  register_module("gate", gate_);
  register_module("experts", experts_);
  for (std::size_t i = 0; i < config_.num_experts; ++i) {
    experts_.emplace_back<SwiGLUMLP>(config_.hidden_size,
                                     config_.intermediate_size,
                                     DataType::Float32, device);
  }
}

MoEOutput MoE::forward(ConstTensorView hidden_states,
                       cudaStream_t stream) const {
  if (hidden_states.device() != Device::cuda(0)
      || gate_.weight().view().device() != hidden_states.device()
      || hidden_states.dtype() != DataType::Float32
      || !hidden_states.is_contiguous() || hidden_states.rank() != 2
      || hidden_states.shape()[0] != 1
      || hidden_states.shape()[1] != config_.hidden_size) {
    throw std::invalid_argument(
        "MoE forward needs a contiguous Float32 [1,H] input and CUDA weights");
  }

  const auto device = hidden_states.device();
  auto logits = gate_.forward(hidden_states, stream);
  auto ids = Tensor::empty({1, config_.top_k}, DataType::Int32, device);
  auto weights = Tensor::empty({1, config_.top_k}, DataType::Float32, device);
  kernels::cuda::moe_route(logits.view().as_const(), ids.view(), weights.view(),
                           stream);

  // The host picks which SwiGLUMLP to call. A blocking route copy keeps this
  // first implementation easy to follow; CPU/GPU overlap can be added later.
  std::vector<std::int32_t> host_ids(config_.top_k);
  check_cuda(cudaMemcpyAsync(host_ids.data(), ids.data(), ids.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "copy MoE expert IDs to CPU");
  check_cuda(cudaStreamSynchronize(stream), "complete MoE routing");

  auto parts = Tensor::empty({config_.top_k, config_.hidden_size},
                             DataType::Float32, device);
  std::vector<Tensor> expert_results;
  expert_results.reserve(config_.top_k);
  for (std::size_t rank = 0; rank < config_.top_k; ++rank) {
    const auto& expert = static_cast<const SwiGLUMLP&>(
        experts_[static_cast<std::size_t>(host_ids[rank])]);
    expert_results.push_back(expert.forward(hidden_states, stream));
    auto row = parts.slice(0, rank, 1);
    check_cuda(
        cudaMemcpyAsync(row.data(), expert_results.back().data(),
                        row.size_bytes(), cudaMemcpyDeviceToDevice, stream),
        "store MoE expert output in router order");
  }

  auto output =
      Tensor::empty({1, config_.hidden_size}, DataType::Float32, device);
  kernels::cuda::moe_combine(parts.view().as_const(), weights.view().as_const(),
                             output.view(), stream);
  // Keep all expert temporaries alive until their copies and combine finish.
  check_cuda(cudaStreamSynchronize(stream), "complete MoE forward");
  return {std::move(output), std::move(logits), std::move(ids),
          std::move(weights), std::move(parts)};
}

const MoEConfig& MoE::config() const noexcept {
  return config_;
}

} // namespace nanovllm
