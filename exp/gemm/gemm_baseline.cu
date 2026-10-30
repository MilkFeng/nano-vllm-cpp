#include <cassert>
#include <cstddef>

#include "common/cuda_utils.hh"
#include "gemm_benchmark.hh"
#include "kernels/cuda/common.cuh"

namespace nanovllm::exp {
namespace {

constexpr int kTile = 16;

__global__ void tiled_gemm_kernel(const float* __restrict__ input,
                                  const float* __restrict__ weight,
                                  float* __restrict__ output,
                                  int m, int n, int k) {
  __shared__ float tile_input[kTile][kTile];
  __shared__ float tile_weight[kTile][kTile + 1];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  const int weight_row = blockIdx.x * kTile + threadIdx.y;
  float sum = 0.0f;

  for (int begin = 0; begin < k; begin += kTile) {
    const int t = begin + threadIdx.x;

    tile_input[threadIdx.y][threadIdx.x] = row < m && t < k
        ? input[static_cast<std::size_t>(row) * k + t] : 0.0f;
    tile_weight[threadIdx.y][threadIdx.x] = weight_row < n && t < k
        ? weight[static_cast<std::size_t>(weight_row) * k + t] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < kTile; ++i) {
      sum = fmaf(tile_input[threadIdx.y][i], tile_weight[threadIdx.x][i], sum);
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    output[static_cast<std::size_t>(row) * n + col] = sum;
  }
}

} // namespace

void tiled_linear(ConstTensorView input, ConstTensorView weight,
                   TensorView output, cudaStream_t stream) {
  kernels::cuda::detail::assert_tensor(input);
  kernels::cuda::detail::assert_tensor(weight);
  kernels::cuda::detail::assert_tensor(output);

  assert(input.dtype() == DataType::Float32 && same_dtype(input, weight, output));
  assert(input.rank() == 2 && weight.rank() == 2 && output.rank() == 2);

  const auto m = input.shape()[0];
  const auto n = weight.shape()[0];
  const auto k = input.shape()[1];

  assert(k > 0 && n > 0 && weight.shape()[1] == k);
  assert(output.shape()[0] == m && output.shape()[1] == n);

  if (m == 0) {
    return;
  }

  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);

  tiled_gemm_kernel<<<grid, block, 0, stream>>>(
      input.data<float>(), weight.data<float>(), output.data<float>(), m, n, k);

  check_cuda(cudaGetLastError(), "tiled GEMM kernel launch");
}

} // namespace nanovllm::exp
