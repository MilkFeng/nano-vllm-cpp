#pragma once

#include <cstddef>
#include <cstdint>

#include "core/tensor_view.hh"

namespace nanovllm::kernels::cpu {

// E4M3FN: sign bit, 4 exponent bits, 3 fraction bits, exponent bias 7.
// Preserves signed zero and subnormals; 0x7f/0xff are NaNs, not infinities.
[[nodiscard]] float fp8_e4m3fn_to_float(std::uint8_t bits);

// CPU, contiguous, nonempty matrices:
// weight: E4M3FN [R,C], scale_inv: Float32 [ceil(R/Br),ceil(C/Bc)],
// output: Float32 [R,C]. Input/output buffers must be disjoint.
// output[r,c] = decode(weight[r,c]) * scale_inv[r/Br,c/Bc].
// Caller owns the output, so repeated calls can reuse the same workspace.
void dequantize_fp8(ConstTensorView weight, ConstTensorView scale_inv,
                    TensorView output, std::size_t block_rows = 128,
                    std::size_t block_columns = 128);

} // namespace nanovllm::kernels::cpu
