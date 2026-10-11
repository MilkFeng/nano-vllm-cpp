#include <algorithm>
#include <cstdint>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/gated_delta_net.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {
namespace {
constexpr unsigned kThreads = 256;

template <bool C>
void assert_float(const BasicTensorView<C>& view) {
  detail::assert_tensor(view);
  assert(view.dtype() == DataType::Float32);
}

std::size_t assert_sequences(std::size_t tokens, std::size_t slots,
                             ConstTensorView cu, ConstTensorView ids) {
  const auto batch = cu.numel() != 0 ? cu.shape()[0] - 1 : tokens;
  if (cu.numel() != 0) {
    detail::assert_tensor(cu);
    assert(cu.dtype() == DataType::Int32 && cu.rank() == 1
           && cu.shape()[0] > 1);
  }
  if (ids.numel() != 0) {
    detail::assert_tensor(ids);
    assert(ids.dtype() == DataType::Int32 && ids.rank() == 1
           && ids.shape()[0] == batch);
  } else {
    assert(batch <= slots);
  }

  return batch;
}

__global__ void conv_kernel(const float* input, const float* weight,
                            float* state, const std::int32_t* cu,
                            const std::int32_t* ids, float* output,
                            std::size_t tokens, std::size_t batch,
                            std::size_t channels, std::size_t kernel,
                            std::size_t slots) {
  const auto start_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t index = start_index; index < batch * channels;
       index += step) {
    const auto sequence = index / channels, channel = index % channels;
    const auto slot = ids ? ids[sequence] : std::int32_t(sequence);
    assert(slot >= 0 && slot < slots);
    const auto begin = cu ? cu[sequence] : std::int32_t(sequence);
    const auto end = cu ? cu[sequence + 1] : begin + 1;
    assert(begin >= 0 && end >= begin && end <= tokens);
    // Avoid pointer arithmetic on a null zero-byte history when K=1.
    float* history =
        kernel > 1 ? state
                         + (static_cast<std::size_t>(slot) * channels + channel)
                               * (kernel - 1)
                   : nullptr;
    for (auto token = begin; token < end; ++token) {
      const auto n = static_cast<std::size_t>(token) * channels + channel;
      const float current = input[n];
      float sum = current * weight[channel * kernel + kernel - 1];
      for (std::size_t j = 0; j + 1 < kernel; ++j)
        sum += history[j] * weight[channel * kernel + j];
      for (std::size_t j = 0; j + 2 < kernel; ++j) history[j] = history[j + 1];
      if (kernel > 1) history[kernel - 2] = current;
      output[n] = sum / (1.0f + expf(-sum));
    }
  }
}

__global__ void delta_kernel(const float* qkv, const float* a, const float* b,
                             const float* A_log, const float* dt_bias,
                             float* state, const std::int32_t* cu,
                             const std::int32_t* ids, float* output,
                             std::size_t tokens, std::size_t slots,
                             std::size_t key_heads, std::size_t value_heads,
                             std::size_t key_dim, std::size_t value_dim) {
  const auto sequence = blockIdx.x / value_heads;
  const auto head = blockIdx.x % value_heads;
  const auto slot = ids ? ids[sequence] : std::int32_t(sequence);
  assert(slot >= 0 && std::size_t(slot) < slots);
  const auto begin = cu ? cu[sequence] : std::int32_t(sequence);
  const auto end = cu ? cu[sequence + 1] : begin + 1;
  assert(begin >= 0 && end >= begin && std::size_t(end) <= tokens);
  const auto key_head = head / (value_heads / key_heads);
  const auto key_size = key_heads * key_dim;
  const auto width = 2 * key_size + value_heads * value_dim;
  const auto state_offset =
      (std::size_t(slot) * value_heads + head) * key_dim * value_dim;

  __shared__ float query[256], key[256], decay, beta;

  const std::size_t lane = threadIdx.x;

  for (auto token = begin; token < end; ++token) {
    const auto row = std::size_t(token) * width;
    const float q = lane < key_dim ? qkv[row + key_head * key_dim + lane] : 0;
    const float k =
        lane < key_dim ? qkv[row + key_size + key_head * key_dim + lane] : 0;
    const float q_sum = detail::block_sum_reduce<float, kThreads>(q * q);
    const float k_sum = detail::block_sum_reduce<float, kThreads>(k * k);
    if (lane < key_dim) {
      query[lane] = q * rsqrtf(q_sum + 1e-6f) * rsqrtf(key_dim);
      key[lane] = k * rsqrtf(k_sum + 1e-6f);
    }
    if (lane == 0) {
      const auto n = static_cast<std::size_t>(token) * value_heads + head;
      const float x = a[n] + dt_bias[head];
      const float softplus = x > 20.0f ? x : log1pf(expf(x));
      decay = __expf(-__expf(A_log[head]) * softplus);
      beta = 1.0f / (1.0f + __expf(-b[n]));
    }
    __syncthreads();
    if (lane < value_dim) {
      float prediction = 0;
      for (std::size_t i = 0; i < key_dim; ++i) {
        auto& memory = state[state_offset + i * value_dim + lane];
        memory *= decay;
        prediction += key[i] * memory;
      }
      const float v = qkv[row + 2 * key_size + head * value_dim + lane];
      const float delta = (v - prediction) * beta;
      float result = 0;
      for (std::size_t i = 0; i < key_dim; ++i) {
        auto& memory = state[state_offset + i * value_dim + lane];
        memory += key[i] * delta;
        result += query[i] * memory;
      }
      output[(static_cast<std::size_t>(token) * value_heads + head) * value_dim
             + lane] = result;
    }
    // All value columns finish reading shared Q/K before the next token.
    __syncthreads();
  }
}
} // namespace

void causal_conv1d(ConstTensorView input, ConstTensorView weight,
                   TensorView state, ConstTensorView cu_seqlens,
                   ConstTensorView state_indices, TensorView output,
                   cudaStream_t stream) {
  assert_float(input);
  assert_float(weight);
  assert_float(state);
  assert_float(output);
  assert(input.rank() == 2 && weight.rank() == 3 && state.rank() == 3);
  assert(weight.shape()[0] == input.shape()[1] && weight.shape()[1] == 1
         && weight.shape()[2] > 0);
  assert(state.shape()[1] == input.shape()[1]
         && state.shape()[2] == weight.shape()[2] - 1);
  assert(same_shape(input, output));
  assert(same_device(input, weight, state, output));
  const auto batch = assert_sequences(input.shape()[0], state.shape()[0],
                                      cu_seqlens, state_indices);
  const auto work = batch * input.shape()[1];
  if (work == 0) {
    return;
  }

  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((work + kThreads - 1) / kThreads, 65535));

  conv_kernel<<<blocks, kThreads, 0, stream>>>(
      input.data<float>(), weight.data<float>(), state.data<float>(),
      cu_seqlens.numel() ? cu_seqlens.data<std::int32_t>() : nullptr,
      state_indices.numel() ? state_indices.data<std::int32_t>() : nullptr,
      output.data<float>(), input.shape()[0], batch, input.shape()[1],
      weight.shape()[2], state.shape()[0]);
  check_cuda(cudaGetLastError(), "causal_conv1d launch");
}

void gated_delta_rule(ConstTensorView qkv, ConstTensorView a, ConstTensorView b,
                      ConstTensorView A_log, ConstTensorView dt_bias,
                      TensorView state, ConstTensorView cu_seqlens,
                      ConstTensorView state_indices, std::size_t key_heads,
                      TensorView output, cudaStream_t stream) {
  assert_float(qkv);
  assert_float(a);
  assert_float(b);
  assert_float(A_log);
  assert_float(dt_bias);
  assert_float(state);
  assert_float(output);
  assert(qkv.rank() == 2 && a.rank() == 2 && state.rank() == 4
         && output.rank() == 3);
  const auto heads = state.shape()[1], kd = state.shape()[2],
             vd = state.shape()[3];
  assert(key_heads > 0 && heads > 0 && heads % key_heads == 0);
  assert(kd > 0 && kd <= 256 && vd > 0 && vd <= 256);
  assert(qkv.shape()[1] == 2 * key_heads * kd + heads * vd);
  assert(a.shape()[0] == qkv.shape()[0] && a.shape()[1] == heads
         && same_shape(a, b));
  assert(A_log.rank() == 1 && A_log.shape()[0] == heads
         && same_shape(A_log, dt_bias));
  assert(output.shape()[0] == qkv.shape()[0] && output.shape()[1] == heads
         && output.shape()[2] == vd);
  assert(same_device(qkv, a, b, A_log, dt_bias, state, output));
  const auto batch = assert_sequences(qkv.shape()[0], state.shape()[0],
                                      cu_seqlens, state_indices);
  if (batch == 0) {
    return;
  }

  delta_kernel<<<unsigned(batch * heads), kThreads, 0, stream>>>(
      qkv.data<float>(), a.data<float>(), b.data<float>(), A_log.data<float>(),
      dt_bias.data<float>(), state.data<float>(),
      cu_seqlens.numel() ? cu_seqlens.data<std::int32_t>() : nullptr,
      state_indices.numel() ? state_indices.data<std::int32_t>() : nullptr,
      output.data<float>(), qkv.shape()[0], state.shape()[0], key_heads, heads,
      kd, vd);

  check_cuda(cudaGetLastError(), "gated_delta_rule launch");
}
} // namespace nanovllm::kernels::cuda
