#pragma once

#include <cuda_runtime_api.h>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

void store_kvcache(ConstTensorView key, ConstTensorView value,
                   TensorView key_cache, TensorView value_cache,
                   ConstTensorView slot_mapping, cudaStream_t stream = nullptr);

void prefill_attention(ConstTensorView query, ConstTensorView key,
                       ConstTensorView value, ConstTensorView cu_seqlens_q,
                       ConstTensorView cu_seqlens_k,
                       ConstTensorView block_tables, TensorView output,
                       float scale, cudaStream_t stream = nullptr);

void paged_attention_decode(ConstTensorView query, ConstTensorView key_cache,
                            ConstTensorView value_cache,
                            ConstTensorView context_lens,
                            ConstTensorView block_tables, TensorView output,
                            float scale, cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
