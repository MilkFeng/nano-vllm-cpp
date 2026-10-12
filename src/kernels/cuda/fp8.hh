#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "core/tensor.hh"

namespace nanovllm::kernels::cuda {

struct FP8LinearWorkspace {
  Tensor quantized_input;
  Tensor input_scale;
  Tensor accumulator;
  Tensor blas;

  FP8LinearWorkspace() = default;
  FP8LinearWorkspace(std::size_t max_tokens, std::size_t max_features);
};

// X/Y: contiguous FP32 [M,K]/[M,N]. W: E4M3FN, zero-padded to 128x128 blocks.
// scale: FP32 [W.rows/128,W.columns/128]. Quantize X by 128-column blocks,
// then use native cuBLASLt FP8 GEMM per weight block and accumulate FP32.
// CUDA device 0. Keep buffers alive and serialize workspace reuse on stream.
void linear_fp8(ConstTensorView input, ConstTensorView weight,
                ConstTensorView weight_scale_inv, TensorView output,
                FP8LinearWorkspace& workspace, cudaStream_t stream = nullptr);

} // namespace nanovllm::kernels::cuda
