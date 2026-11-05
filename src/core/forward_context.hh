#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor_view.hh"

namespace nanovllm {

struct ForwardContext {
  bool is_prefill = false;
  // Only the engine may opt into segments whose outputs are reused next step.
  bool decode_graphs = false;
  bool whole_graph = false;
  bool graph_warmup = false;
  ConstTensorView cu_seqlens_q;
  ConstTensorView cu_seqlens_k;
  std::size_t max_seqlen_q = 0;
  std::size_t max_seqlen_k = 0;
  ConstTensorView block_tables;
  ConstTensorView state_indices;
  cudaStream_t stream = nullptr;
  TensorView attention_workspace;
};

} // namespace nanovllm
