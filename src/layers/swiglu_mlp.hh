#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/module.hh"
#include "layers/linear.hh"

namespace nanovllm {

// Caller-owned CUDA buffers [tokens,I], reused on one ordered stream.
// Keep all three alive until the stream finishes; buffers must be disjoint.
struct SwiGLUWorkspace {
  Tensor gate;
  Tensor up;
  Tensor activated;
};

// down_proj(SiLU(gate_proj(x)) * up_proj(x)).
class SwiGLUMLP final : public Module {
 public:
  SwiGLUMLP(std::size_t hidden_size, std::size_t intermediate_size,
            DataType dtype, Device device);

  [[nodiscard]] Tensor forward(ConstTensorView hidden_states,
                               cudaStream_t stream = nullptr) const;

  // Enqueues into output [tokens,H] without allocating/freeing CUDA tensors.
  // Shape/device/dtype preconditions are checked with Debug assertions.
  // The caller owns output/workspace and waits before reusing on another
  // stream.
  void forward_into(ConstTensorView hidden_states, TensorView output,
                    SwiGLUWorkspace& workspace,
                    cudaStream_t stream = nullptr) const;

 private:
  std::size_t hidden_size_;
  std::size_t intermediate_size_;

  // gate_proj and up_proj remain separate to match checkpoint names. They can
  // be presented as one packed matrix to a fused SwiGLU kernel later.
  Linear gate_proj_;
  Linear up_proj_;
  Linear down_proj_;
};

} // namespace nanovllm
