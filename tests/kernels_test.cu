#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kernels/kernels.hh"

namespace {
using namespace nanovllm;
namespace kernels = nanovllm::kernels;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

template <typename T>
constexpr DataType dtype() {
  if constexpr (std::is_same_v<T, float>) return DataType::Float32;
  if constexpr (std::is_same_v<T, __half>) return DataType::Float16;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) return DataType::BFloat16;
  return DataType::Int32;
}

template <typename T>
float value(T x) { return static_cast<float>(x); }

template <typename T>
float rounded(float x) { return value(static_cast<T>(x)); }

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t count) : count_(count) {
    if (count != 0) check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
  }
  ~Buffer() { if (data_) (void)cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* data() { return data_; }
  TensorView view(std::vector<std::size_t> shape) {
    return {data_, dtype<T>(), TensorLayout(std::move(shape)), Device::cuda(0)};
  }
  TensorView view(std::vector<std::size_t> shape,
                  std::vector<std::size_t> strides) {
    return {data_, dtype<T>(), TensorLayout(std::move(shape), std::move(strides)),
            Device::cuda(0)};
  }
  void upload(const std::vector<T>& host, cudaStream_t stream) {
    if (host.size() != count_) throw std::runtime_error("upload size mismatch");
    if (count_ != 0) {
      check(cudaMemcpyAsync(data_, host.data(), count_ * sizeof(T),
                            cudaMemcpyHostToDevice, stream), "upload");
    }
  }
  std::vector<T> download(cudaStream_t stream) {
    std::vector<T> host(count_);
    if (count_ != 0) {
      check(cudaMemcpyAsync(host.data(), data_, count_ * sizeof(T),
                            cudaMemcpyDeviceToHost, stream), "download");
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return host;
  }

 private:
  T* data_ = nullptr;
  std::size_t count_;
};

template <typename T>
void expect_close(const std::vector<T>& actual, const std::vector<float>& expected,
                   const std::string& name, bool exact = false) {
  if (actual.size() != expected.size()) throw std::runtime_error(name + " size");
  const float tolerance = exact ? 0.0f
      : std::is_same_v<T, float> ? 4.0e-5f
      : std::is_same_v<T, __half> ? 2.0e-3f : 1.6e-2f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float got = value(actual[i]);
    if (!std::isfinite(got)
        || std::abs(got - expected[i]) > tolerance * (1.0f + std::abs(expected[i]))) {
      throw std::runtime_error(name + " index " + std::to_string(i)
                               + ": got " + std::to_string(got)
                               + ", expected " + std::to_string(expected[i]));
    }
  }
}

template <typename Function>
void expect_invalid(Function&& function, const char* name) {
  try { function(); }
  catch (const std::invalid_argument&) { return; }
  throw std::runtime_error(std::string(name) + " did not reject invalid input");
}

// Independent CPU reference: double reduction, float normalization, followed
// by the dtype conversion that nano-vllm performs before multiplying weights.
template <typename T>
std::vector<float> norm_reference(const std::vector<T>& input,
                                  const std::vector<T>& residual,
                                  const std::vector<T>& weight,
                                  std::size_t tokens, std::size_t heads,
                                  std::size_t width, std::size_t input_stride,
                                  std::size_t residual_stride, bool fused) {
  std::vector<float> expected(tokens * heads * width);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t h = 0; h < heads; ++h) {
      std::vector<float> row(width);
      double squares = 0;
      for (std::size_t d = 0; d < width; ++d) {
        row[d] = value(input[t * input_stride + h * width + d]);
        if (fused) row[d] += value(residual[t * residual_stride + h * width + d]);
        squares += static_cast<double>(row[d]) * row[d];
      }
      const float scale = 1.0f / std::sqrt(static_cast<float>(squares / width) + 1e-6f);
      for (std::size_t d = 0; d < width; ++d) {
        expected[(t * heads + h) * width + d]
            = rounded<T>(rounded<T>(row[d] * scale) * value(weight[d]));
      }
    }
  }
  return expected;
}

template <typename T>
void test_norm(std::size_t width, int rank, bool padded, bool inplace,
                cudaStream_t stream, std::size_t requested_tokens = 3) {
  const std::size_t tokens = rank == 1 ? 1 : requested_tokens;
  const std::size_t heads = rank == 3 ? 2 : 1;
  const std::size_t count = tokens * heads * width;
  const std::size_t xs = heads * width + (padded ? 7 : 0);
  const std::size_t rs = heads * width + (padded ? 11 : 0);
  std::vector<T> x(tokens * xs), r(tokens * rs), w(width);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<T>(std::sin(i * .37) * 2.1);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = static_cast<T>(std::cos(i * .23) * .71);
  for (std::size_t i = 0; i < width; ++i) w[i] = static_cast<T>(.75 + std::sin(i * .11) * .5);
  Buffer<T> bx(x.size()), br(r.size()), bw(width), bo(count), bro(count);
  bx.upload(x, stream); br.upload(r, stream); bw.upload(w, stream);
  std::vector<std::size_t> shape, xstrides, rstrides;
  if (rank == 1) {
    shape = {width}; xstrides = {1}; rstrides = {1};
  } else if (rank == 2) {
    shape = {tokens, width}; xstrides = {xs, 1}; rstrides = {rs, 1};
  } else {
    shape = {tokens, heads, width};
    xstrides = {xs, width, 1}; rstrides = {rs, width, 1};
  }
  auto xv = bx.view(shape, xstrides), rv = br.view(shape, rstrides);
  auto ov = inplace ? xv : bo.view(shape);
  auto rov = inplace ? rv : bro.view(shape);
  kernels::rms_norm(xv.as_const(), bw.view({width}).as_const(), ov, 1e-6f, stream);
  expect_close(inplace ? bx.download(stream) : bo.download(stream),
               norm_reference(x, r, w, tokens, heads, width, xs, rs, false), "RMSNorm");
  bx.upload(x, stream);
  kernels::fused_add_rms_norm(xv.as_const(), rv.as_const(), bw.view({width}).as_const(),
                             ov, rov, 1e-6f, stream);
  expect_close(inplace ? bx.download(stream) : bo.download(stream),
               norm_reference(x, r, w, tokens, heads, width, xs, rs, true), "fused RMSNorm");
  std::vector<float> expected_residual(count);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t d = 0; d < heads * width; ++d) {
      expected_residual[t * heads * width + d]
          = rounded<T>(value(x[t * xs + d]) + value(r[t * rs + d]));
    }
  }
  expect_close(inplace ? br.download(stream) : bro.download(stream),
               expected_residual, "fused residual", true);
}

template <typename T>
void test_fused_precision(cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    const bool half = std::is_same_v<T, __half>;
    const std::vector<T> x = half
        ? std::vector<T>{T(-1.6201171875f), T(-.155029296875f)}
        : std::vector<T>{T(.302734375f), T(-1.75f)};
    const std::vector<T> r = half
        ? std::vector<T>{T(-.84130859375f), T(.56298828125f)}
        : std::vector<T>{T(-.037109375f), T(.51953125f)};
    const std::vector<T> w = half
        ? std::vector<T>{T(1.306640625f), T(.65966796875f)}
        : std::vector<T>{T(1.6328125f), T(.5390625f)};
    const std::vector<float> expected = half
        ? std::vector<float>{-1.8232421875f, .1524658203125f}
        : std::vector<float>{.48828125f, -.74609375f};
    Buffer<T> bx(2), br(2), bw(2), bo(2), bro(2);
    bx.upload(x, stream); br.upload(r, stream); bw.upload(w, stream);
    kernels::fused_add_rms_norm(bx.view({1, 2}).as_const(), br.view({1, 2}).as_const(),
                               bw.view({2}).as_const(), bo.view({1, 2}),
                               bro.view({1, 2}), 1e-6f, stream);
    // These cases change one output if the sum is rounded before statistics,
    // or if normalized values are multiplied by weight before dtype rounding.
    expect_close(bo.download(stream), expected, "fused rounding regression", true);
  }
}

template <typename T>
void test_activation(cudaStream_t stream) {
  constexpr std::size_t tokens = 3, width = 257;
  std::vector<T> packed(tokens * 2 * width);
  std::vector<float> expected(tokens * width);
  constexpr float special[] = {-100, -10, -1, 0, 1, 10, 100};
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t d = 0; d < width; ++d) {
      const T g = static_cast<T>(special[d % 7]);
      const T u = static_cast<T>(std::cos((t * width + d) * .31) * 1.7);
      packed[t * 2 * width + d] = g;
      packed[t * 2 * width + width + d] = u;
      const double gf = value(g);
      expected[t * width + d] = rounded<T>(static_cast<float>(
          gf / (1.0 + std::exp(-gf)) * value(u)));
    }
  }
  Buffer<T> bp(packed.size()), bo(expected.size());
  bp.upload(packed, stream);
  auto pv = bp.view({tokens, 2 * width});
  kernels::silu_and_mul(pv.slice(1, 0, width).as_const(),
                        pv.slice(1, width, width).as_const(),
                        bo.view({tokens, width}), stream);
  expect_close(bo.download(stream), expected, "strided SiLU+Mul");

  std::vector<T> gate(tokens * width), up(tokens * width);
  for (std::size_t t = 0; t < tokens; ++t) {
    std::copy_n(packed.begin() + t * 2 * width, width, gate.begin() + t * width);
    std::copy_n(packed.begin() + t * 2 * width + width, width, up.begin() + t * width);
  }
  Buffer<T> bg(gate.size()), bu(up.size());
  bg.upload(gate, stream); bu.upload(up, stream);
  kernels::silu_and_mul(bg.view({tokens, width}).as_const(),
                        bu.view({tokens, width}).as_const(), bg.view({tokens, width}), stream);
  expect_close(bg.download(stream), expected, "in-place SiLU+Mul");
}

template <typename T>
void test_embedding(cudaStream_t stream) {
  constexpr std::size_t vocab = 17, width = 257;
  const std::vector<std::int32_t> ids = {16, 0, 7, 7, 1, 16};
  std::vector<T> weight(vocab * width);
  for (std::size_t i = 0; i < weight.size(); ++i) weight[i] = static_cast<T>(std::sin(i * .17));
  std::vector<float> expected(ids.size() * width);
  for (std::size_t t = 0; t < ids.size(); ++t) {
    for (std::size_t d = 0; d < width; ++d) expected[t * width + d] = value(weight[ids[t] * width + d]);
  }
  Buffer<std::int32_t> bi(ids.size());
  Buffer<T> bw(weight.size()), bo(expected.size());
  bi.upload(ids, stream); bw.upload(weight, stream);
  kernels::embedding(bi.view({ids.size()}).as_const(), bw.view({vocab, width}).as_const(),
                     bo.view({ids.size(), width}), stream);
  expect_close(bo.download(stream), expected, "embedding", true);
}

template <typename T>
void test_rope(std::size_t width, bool inplace, cudaStream_t stream) {
  constexpr std::size_t tokens = 4, nq = 3, nk = 1, max_positions = 64;
  const std::vector<std::int32_t> positions = {0, 31, 2, 63};
  const std::size_t qs = nq * width + (inplace ? 0 : 7);
  const std::size_t ks = nk * width + (inplace ? 0 : 11);
  std::vector<T> q(tokens * qs), k(tokens * ks);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = static_cast<T>(std::sin(i * .31) * 1.9);
  for (std::size_t i = 0; i < k.size(); ++i) k[i] = static_cast<T>(std::cos(i * .19) * .7);
  std::vector<float> cache(max_positions * width);
  for (std::size_t p = 0; p < max_positions; ++p) {
    for (std::size_t d = 0; d < width / 2; ++d) {
      const double angle = p * std::pow(10000.0, -2.0 * d / width);
      cache[p * width + d] = static_cast<float>(std::cos(angle));
      cache[p * width + d + width / 2] = static_cast<float>(std::sin(angle));
    }
  }
  auto reference = [&](const std::vector<T>& input, std::size_t heads, std::size_t stride) {
    std::vector<float> result(tokens * heads * width);
    for (std::size_t t = 0; t < tokens; ++t) {
      for (std::size_t h = 0; h < heads; ++h) {
        for (std::size_t d = 0; d < width / 2; ++d) {
          const std::size_t offset = t * stride + h * width + d;
          const float x = value(input[offset]), y = value(input[offset + width / 2]);
          const float c = cache[positions[t] * width + d];
          const float s = cache[positions[t] * width + d + width / 2];
          result[(t * heads + h) * width + d] = rounded<T>(x * c - y * s);
          result[(t * heads + h) * width + d + width / 2] = rounded<T>(y * c + x * s);
        }
      }
    }
    return result;
  };
  Buffer<std::int32_t> bp(tokens);
  Buffer<float> bc(cache.size());
  Buffer<T> bq(q.size()), bk(k.size()), bqo(tokens * nq * width), bko(tokens * nk * width);
  bp.upload(positions, stream); bq.upload(q, stream); bk.upload(k, stream);
  kernels::build_rope_cache(bc.view({max_positions, width}), 10000, stream);
  expect_close(bc.download(stream), cache, "RoPE cache");
  auto qv = bq.view({tokens, nq, width}, {qs, width, 1});
  auto kv = bk.view({tokens, nk, width}, {ks, width, 1});
  kernels::apply_rotary_embedding(bp.view({tokens}).as_const(), qv.as_const(), kv.as_const(),
                                  bc.view({max_positions, width}).as_const(),
                                  inplace ? qv : bqo.view({tokens, nq, width}),
                                  inplace ? kv : bko.view({tokens, nk, width}), stream);
  expect_close(inplace ? bq.download(stream) : bqo.download(stream), reference(q, nq, qs), "RoPE Q");
  expect_close(inplace ? bk.download(stream) : bko.download(stream), reference(k, nk, ks), "RoPE K");
}

template <typename T>
void test_empty(cudaStream_t stream) {
  Buffer<T> empty(0), weight(4);
  Buffer<std::int32_t> ids(0);
  Buffer<float> cache(16);
  auto e = empty.view({0, 4}), w = weight.view({4});
  kernels::rms_norm(e.as_const(), w.as_const(), e, 1e-6f, stream);
  kernels::fused_add_rms_norm(e.as_const(), e.as_const(), w.as_const(), e, e, 1e-6f, stream);
  kernels::silu_and_mul(e.as_const(), e.as_const(), e, stream);
  kernels::embedding(ids.view({0}).as_const(), weight.view({1, 4}).as_const(), e, stream);
  kernels::apply_rotary_embedding(ids.view({0}).as_const(), empty.view({0, 2, 4}).as_const(),
                                  empty.view({0, 1, 4}).as_const(), cache.view({4, 4}).as_const(),
                                  empty.view({0, 2, 4}), empty.view({0, 1, 4}), stream);
  Buffer<float> empty_cache(0);
  kernels::build_rope_cache(empty_cache.view({0, 4}), 10000, stream);
}

void test_invalid(cudaStream_t stream) {
  Buffer<float> x(8), r(8), w(4), out(8), cache(16);
  Buffer<std::int32_t> ids(2);
  auto xv = x.view({2, 4}), rv = r.view({2, 4}), wv = w.view({4}), ov = out.view({2, 4});
  for (const float epsilon : {-1.0f, 0.0f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
    expect_invalid([&] { kernels::rms_norm(xv.as_const(), wv.as_const(), ov, epsilon, stream); }, "RMS epsilon");
  }
  expect_invalid([&] { kernels::rms_norm(xv.as_const(), w.view({2, 2}).as_const(), ov, 1e-6f, stream); }, "RMS weight rank");
  expect_invalid([&] { kernels::rms_norm(xv.as_const(), wv.as_const(), out.view({4, 2}), 1e-6f, stream); }, "RMS shape");
  expect_invalid([&] { kernels::fused_add_rms_norm(xv.as_const(), rv.as_const(), wv.as_const(), ov, ov, 1e-6f, stream); }, "fused output alias");
  auto cpu = TensorView(x.data(), DataType::Float32, TensorLayout({2, 4}), Device::cpu());
  expect_invalid([&] { kernels::silu_and_mul(cpu.as_const(), rv.as_const(), ov, stream); }, "CPU input");
  expect_invalid([&] { kernels::silu_and_mul(xv.as_const(), r.view({4, 2}).as_const(), ov, stream); }, "SiLU shape");
  auto integer = TensorView(x.data(), DataType::Int32, TensorLayout({2, 4}), Device::cuda(0));
  expect_invalid([&] { kernels::silu_and_mul(integer.as_const(), rv.as_const(), ov, stream); }, "SiLU dtype");
  auto partial = TensorView(x.data() + 1, DataType::Float32, TensorLayout({1, 4}), Device::cuda(0));
  expect_invalid([&] { kernels::silu_and_mul(x.view({1, 4}).as_const(), r.view({1, 4}).as_const(),
                                           partial, stream); }, "SiLU partial alias");
  expect_invalid([&] { kernels::rms_norm(x.view({1, 4}).as_const(), wv.as_const(),
                                        partial, 1e-6f, stream); }, "RMS partial alias");
  expect_invalid([&] { kernels::embedding(w.view({2}).as_const(), xv.as_const(), ov, stream); }, "embedding IDs dtype");
  expect_invalid([&] { kernels::embedding(ids.view({2}).as_const(), xv.as_const(), out.view({4, 2}), stream); }, "embedding shape");
  expect_invalid([&] { kernels::embedding(ids.view({2}).as_const(), xv.as_const(), xv, stream); }, "embedding output alias");
  expect_invalid([&] { kernels::build_rope_cache(cache.view({4, 4}), 0, stream); }, "RoPE theta");
  expect_invalid([&] { kernels::build_rope_cache(cache.view({4, 3}), 10000, stream); }, "RoPE odd dimension");
  expect_invalid([&] { kernels::apply_rotary_embedding(ids.view({1}).as_const(), x.view({2, 1, 4}).as_const(),
                        r.view({2, 1, 4}).as_const(), cache.view({4, 4}).as_const(),
                        out.view({2, 1, 4}), r.view({2, 1, 4}), stream); }, "RoPE positions shape");
  expect_invalid([&] { kernels::apply_rotary_embedding(ids.view({2}).as_const(), x.view({2, 1, 4}).as_const(),
                        r.view({2, 1, 4}).as_const(), cache.view({4, 4}).as_const(),
                        out.view({2, 1, 4}), out.view({2, 1, 4}), stream); }, "RoPE output alias");
  check(cudaStreamSynchronize(stream), "invalid-input checks");
}

template <typename T>
void run_dtype(cudaStream_t stream, const char* name) {
  for (const std::size_t width : {1, 7, 128, 257, 1024}) test_norm<T>(width, 2, false, false, stream);
  test_norm<T>(7, 1, false, true, stream);
  test_norm<T>(257, 2, true, false, stream);
  test_norm<T>(257, 2, false, true, stream);
  test_norm<T>(7, 3, true, false, stream);
  test_norm<T>(128, 3, true, false, stream);
  test_fused_precision<T>(stream);
  test_activation<T>(stream);
  test_embedding<T>(stream);
  for (const std::size_t width : {2, 128}) {
    test_rope<T>(width, false, stream);
    test_rope<T>(width, true, stream);
  }
  test_empty<T>(stream);
  std::cout << "PASS " << name << " numerical, strided, in-place, and empty cases\n";
}
} // namespace

int main() {
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && count == 0)) {
    std::cout << "SKIP: CUDA GPU unavailable\n";
    return 77;
  }
  cudaStream_t stream = nullptr;
  try {
    check(status, "cudaGetDeviceCount");
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    run_dtype<float>(stream, "Float32");
    run_dtype<__half>(stream, "Float16");
    run_dtype<__nv_bfloat16>(stream, "BFloat16");
    test_norm<float>(7, 2, false, false, stream, 65537);
    std::cout << "PASS normalization grid-stride rows\n";
    test_invalid(stream);
    check(cudaStreamDestroy(stream), "destroy stream");
    std::cout << "PASS host validation\n";
    return 0;
  } catch (const std::exception& error) {
    if (stream) (void)cudaStreamDestroy(stream);
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
