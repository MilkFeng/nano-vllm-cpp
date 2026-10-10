#pragma once

#include <cuda_runtime_api.h>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cuda {

// These baseline kernels enqueue work without allocation or synchronization.
// All tensors are contiguous on CUDA device 0, which the caller selects.
// Floating point tensors share a Float32/Float16/BFloat16 dtype; D is in
// [1,256]. Writable tensors must not overlap other tensors. Preconditions use
// debug assertions; callers retain buffers until the stream finishes.

// key/value: [T,Hkv,D]; caches: [P,block_size,Hkv,D]; slot_mapping: Int32 [T].
// A slot of -1 skips that token. Other slots must be in [0,P*block_size) and
// unique within the call. Prefix-cache sharing changes block tables, not data.
void store_kvcache(ConstTensorView key, ConstTensorView value,
                   TensorView key_cache, TensorView value_cache,
                   ConstTensorView slot_mapping, cudaStream_t stream = nullptr);

// query/output: [T,Hq,D]. Hq must be divisible by Hkv; consecutive groups of
// Hq/Hkv query heads share one KV head. Scale must be finite and positive.
// cu_seqlens_q/k: contiguous Int32 [B+1], monotone and starting at zero.
// cu_seqlens_q[B] == T. With an empty block_tables view, key/value are packed
// [Tk,Hkv,D], and cu_seqlens_k[B] == Tk. Otherwise key/value are paged caches
// [P,block_size,Hkv,D], and block_tables is contiguous Int32 [B,max_blocks].
// Each sequence's KV length must fit its table and all used block IDs must be
// valid. Causal alignment is j <= Lk - Lq + i, including cached prefixes and
// chunks. Rows with no visible keys produce zeros.
// Uses FP32 online softmax and O(D) registers per query head, without a full
// scores matrix. This is a correctness baseline, not tiled FlashAttention.
void prefill_attention(ConstTensorView query, ConstTensorView key,
                       ConstTensorView value, ConstTensorView cu_seqlens_q,
                       ConstTensorView cu_seqlens_k,
                       ConstTensorView block_tables, TensorView output,
                       float scale, cudaStream_t stream = nullptr);

// query/output: [B,Hq,D]; caches: [P,block_size,Hkv,D]; context_lens: Int32
// [B]; block_tables: Int32 [B,max_blocks]. Context lengths include the current
// token already stored in the cache. A zero length yields zeros without reading
// a block-table entry, so CUDA Graph padding and empty caches are safe.
void paged_attention_decode(ConstTensorView query, ConstTensorView key_cache,
                            ConstTensorView value_cache,
                            ConstTensorView context_lens,
                            ConstTensorView block_tables, TensorView output,
                            float scale, cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
