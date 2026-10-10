#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor_view.hh"

namespace nanovllm {

// Non-owning execution metadata prepared by ModelRunner. Empty tensor views
// mean that the corresponding metadata is not used by the current execution
// mode.
struct ForwardContext {
  bool is_prefill = false;
  ConstTensorView cu_seqlens_q;
  ConstTensorView cu_seqlens_k;
  std::size_t max_seqlen_q = 0;
  std::size_t max_seqlen_k = 0;
  ConstTensorView slot_mapping;
  ConstTensorView context_lens;
  ConstTensorView block_tables;
  cudaStream_t stream = nullptr;
};

} // namespace nanovllm
