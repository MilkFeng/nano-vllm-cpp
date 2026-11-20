#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cpu/fp8_dequant.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/linear.hh"
#include "layers/moe.hh"

using namespace nanovllm;

namespace {
void store(TensorView target, const void* source) {
  check_cuda(cudaMemcpy(target.data(), source, target.size_bytes(),
                        cudaMemcpyHostToDevice), "upload check input");
}
void compare(ConstTensorView a, ConstTensorView b) {
  std::vector<float> x(a.numel()), y(b.numel());
  check_cuda(cudaMemcpy(x.data(), a.data(), a.size_bytes(), cudaMemcpyDeviceToHost), "read compact output");
  check_cuda(cudaMemcpy(y.data(), b.data(), b.size_bytes(), cudaMemcpyDeviceToHost), "read expanded output");
  // Same FP32 operands and accumulation order: require bitwise equality.
  if (std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) != 0) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (std::bit_cast<std::uint32_t>(x[i]) != std::bit_cast<std::uint32_t>(y[i]))
        throw std::runtime_error("compact mismatch at " + std::to_string(i)
                                 + ": " + std::to_string(x[i]) + " / " + std::to_string(y[i]));
    }
  }
}
void check(std::size_t m, std::size_t n, std::size_t k) {
  const auto device = Device::cuda(0);
  auto input = Tensor::empty({m, k}, DataType::Float32, device);
  std::vector<float> x(m * k);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 1.234f);
  store(input.view(), x.data());
  Linear dense(k, n, false, DataType::Float32, device);
  FP8Linear compact(k, n, device);
  auto weights = Tensor::empty({n, k}, DataType::Float8E4M3FN);
  auto scales = Tensor::empty({(n + 127) / 128, (k + 127) / 128}, DataType::Float32);
  auto expanded = Tensor::empty({n, k}, DataType::Float32);
  for (std::size_t i = 0; i < weights.numel(); ++i)
    weights.view().data<std::uint8_t>()[i] = (i * 37 % 126) | ((i % 3 == 0) ? 128 : 0);
  for (std::size_t i = 0; i < scales.numel(); ++i)
    scales.view().data<float>()[i] = 0.0012345f * (i + 1);
  kernels::cpu::dequantize_fp8(weights.view().as_const(), scales.view().as_const(), expanded.view(), 128, 128);
  store(dense.weight().view(), expanded.data());
  store(compact.weight().view(), weights.data());
  store(compact.weight_scale_inv().view(), scales.data());
  auto reference = dense.forward(input.view().as_const());
  auto actual = compact.forward(input.view().as_const());
  compare(actual.view().as_const(), reference.view().as_const());

  auto bf16 = Tensor::empty({n, k}, DataType::BFloat16, device);
  std::vector<std::uint16_t> bits(n * k);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    bits[i] = std::bit_cast<std::uint32_t>(std::cos(float(i) * 0.123f)) >> 16;
    expanded.view().data<float>()[i] = std::bit_cast<float>(std::uint32_t(bits[i]) << 16);
  }
  store(bf16.view(), bits.data());
  store(dense.weight().view(), expanded.data());
  reference = dense.forward(input.view().as_const());
  kernels::cuda::linear(input.view().as_const(), bf16.view().as_const(), actual.view(), std::nullopt, nullptr);
  compare(actual.view().as_const(), reference.view().as_const());
}

void check_experts(std::size_t hidden, std::size_t intermediate) {
  const auto device = Device::cuda(0);
  FP8Expert expert(hidden, intermediate, device);
  for (const auto& [name, parameter] : expert.named_parameters()) {
    const auto view = parameter->view();
    if (view.dtype() == DataType::Float8E4M3FN) {
      std::vector<std::uint8_t> w(view.numel());
      for (std::size_t i = 0; i < w.size(); ++i) w[i] = (i * 37 % 126) | ((i % 3 == 0) ? 128 : 0);
      store(view, w.data());
    } else {
      std::vector<float> scales(view.numel());
      for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = 0.0012345f * (i + 1);
      store(view, scales.data());
    }
  }
  auto input = Tensor::empty({1, hidden}, DataType::Float32, device);
  std::vector<float> x(hidden);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 1.234f);
  store(input.view(), x.data());
  auto actual = Tensor::empty({8, hidden}, DataType::Float32, device);
  auto expected = Tensor::empty({8, hidden}, DataType::Float32, device);
  auto activated = Tensor::empty({8, intermediate}, DataType::Float32, device);
  FP8ExpertWorkspace workspace(intermediate);
  for (unsigned count : {1, 3, 8}) {
    check_cuda(cudaMemset(actual.data(), 0, actual.size_bytes()), "clear batched check output");
    check_cuda(cudaMemset(expected.data(), 0, expected.size_bytes()), "clear reference output");
    kernels::cuda::FP8ExpertBatch batch;
    batch.count = count;
    for (unsigned i = 0; i < count; ++i) {
      batch.rows[i] = (3 * i + 5) % 8;
      batch.weights[i] = expert.device_weights();
      expert.forward_into(input.view().as_const(), expected.view().slice(0, batch.rows[i], 1), workspace);
    }
    kernels::cuda::fp8_experts_decode(input.view().as_const(), batch, activated.view(), actual.view());
    compare(actual.view().as_const(), expected.view().as_const());
  }
}
}

int main() {
  try {
    for (const auto m : {1, 2, 7, 16, 129})
      for (const auto k : {5, 128, 257, 2048})
        for (const auto n : {3, 128, 259}) check(m, n, k);
    check_experts(128, 128);
    check_experts(259, 257);
    check_experts(2048, 512);
    std::cout << "Compact FP8/BF16 weights: 120 bitwise comparisons passed\n";
    std::cout << "Batched FP8 experts: 9 bitwise comparisons passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
