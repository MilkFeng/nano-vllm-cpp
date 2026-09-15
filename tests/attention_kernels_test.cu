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

#include "kernels/attention.hh"

namespace {
using namespace nanovllm;
namespace kernels = nanovllm::kernels;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
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
double number(T value) { return static_cast<float>(value); }

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t size) : size_(size) {
    if (size) check(cudaMalloc(&data_, size * sizeof(T)), "cudaMalloc");
  }
  ~Buffer() { if (data_) (void)cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* data() { return data_; }
  TensorView view(std::vector<std::size_t> shape) {
    return {data_, dtype<T>(), TensorLayout(std::move(shape)), Device::cuda(0)};
  }
  TensorView view(std::vector<std::size_t> shape,
                  std::vector<std::size_t> strides, std::size_t offset = 0) {
    return {data_ + offset, dtype<T>(),
            TensorLayout(std::move(shape), std::move(strides)), Device::cuda(0)};
  }
  void upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() != size_) throw std::runtime_error("upload size mismatch");
    if (size_) check(cudaMemcpyAsync(data_, values.data(), size_ * sizeof(T),
                                     cudaMemcpyHostToDevice, stream), "upload");
  }
  std::vector<T> download(cudaStream_t stream) {
    std::vector<T> result(size_);
    if (size_) check(cudaMemcpyAsync(result.data(), data_, size_ * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream), "download");
    check(cudaStreamSynchronize(stream), "synchronize");
    return result;
  }

 private:
  T* data_ = nullptr;
  std::size_t size_;
};

template <typename T>
void expect_close(const std::vector<T>& actual, const std::vector<double>& expected,
                   const std::string& name, bool exact = false) {
  if (actual.size() != expected.size()) throw std::runtime_error(name + ": size mismatch");
  const double tolerance = exact ? 0.0 : std::is_same_v<T, float> ? 6e-5
      : std::is_same_v<T, __half> ? 2e-3 : 1e-2;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double value = number(actual[i]);
    if (!std::isfinite(value)
        || std::abs(value - expected[i]) > tolerance * (1 + std::abs(expected[i]))) {
      throw std::runtime_error(name + " at " + std::to_string(i) + ": got "
                               + std::to_string(value) + ", expected "
                               + std::to_string(expected[i]));
    }
  }
}

template <typename Function>
void expect_invalid(Function&& function, const char* name) {
  try { function(); }
  catch (const std::invalid_argument&) { return; }
  throw std::runtime_error(std::string(name) + " accepted invalid input");
}

std::vector<std::int32_t> cumulative(const std::vector<std::int32_t>& lengths) {
  std::vector<std::int32_t> result(1, 0);
  for (const auto length : lengths) result.push_back(result.back() + length);
  return result;
}

// Independent CPU oracle: materialize each score row in double, subtract its
// maximum, then form the weighted sum. This does not reuse online softmax.
template <typename T, typename ReadKey, typename ReadValue>
std::vector<double> reference(const std::vector<T>& query, std::size_t query_stride,
                               std::size_t query_heads, std::size_t kv_heads,
                               std::size_t width, const std::vector<std::int32_t>& cu_q,
                               const std::vector<std::int32_t>& cu_k, float scale,
                               ReadKey read_key, ReadValue read_value) {
  std::vector<double> output(static_cast<std::size_t>(cu_q.back()) * query_heads * width);
  for (std::size_t sequence = 0; sequence + 1 < cu_q.size(); ++sequence) {
    const auto lq = cu_q[sequence + 1] - cu_q[sequence];
    const auto lk = cu_k[sequence + 1] - cu_k[sequence];
    for (std::int32_t i = 0; i < lq; ++i) {
      const auto token = static_cast<std::size_t>(cu_q[sequence] + i);
      const auto visible = std::max(0, std::min(lk, lk - lq + i + 1));
      for (std::size_t head = 0; head < query_heads; ++head) {
        if (visible == 0) continue;
        const auto kv_head = head / (query_heads / kv_heads);
        std::vector<double> scores(visible);
        for (std::int32_t j = 0; j < visible; ++j) {
          double dot = 0;
          for (std::size_t d = 0; d < width; ++d) {
            dot += number(query[token * query_stride + head * width + d])
                   * read_key(sequence, j, kv_head, d);
          }
          scores[j] = dot * scale;
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double denominator = 0;
        for (auto& score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (std::size_t d = 0; d < width; ++d) {
          double sum = 0;
          for (std::int32_t j = 0; j < visible; ++j) {
            sum += scores[j] * read_value(sequence, j, kv_head, d);
          }
          output[(token * query_heads + head) * width + d] = sum / denominator;
        }
      }
    }
  }
  return output;
}

template <typename T>
void test_store(std::size_t width, cudaStream_t stream) {
  constexpr std::size_t tokens = 4, heads = 2, blocks = 3, block_size = 3;
  const auto token_width = heads * width;
  const auto stride = 2 * token_width + 5;
  std::vector<T> packed(tokens * stride, T(std::numeric_limits<float>::quiet_NaN()));
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t d = 0; d < token_width; ++d) {
      packed[t * stride + d] = T(float(t * 3) + .013f * d);
      packed[t * stride + token_width + d] = T(-float(t * 2) - .017f * d);
    }
  }
  const std::vector<std::int32_t> slots = {7, -1, 1, 5};
  const auto cache_size = blocks * block_size * token_width;
  std::vector<T> initial(cache_size, T(-9.0f));
  std::vector<double> expected_k(cache_size, -9), expected_v(cache_size, -9);
  for (std::size_t t = 0; t < tokens; ++t) {
    if (slots[t] == -1) continue;
    for (std::size_t d = 0; d < token_width; ++d) {
      expected_k[slots[t] * token_width + d] = number(packed[t * stride + d]);
      expected_v[slots[t] * token_width + d] = number(packed[t * stride + token_width + d]);
    }
  }
  Buffer<T> bp(packed.size()), bk(cache_size), bv(cache_size);
  Buffer<std::int32_t> bs(tokens);
  bp.upload(packed, stream); bk.upload(initial, stream); bv.upload(initial, stream);
  bs.upload(slots, stream);
  auto k = bp.view({tokens, heads, width}, {stride, width, 1});
  auto v = bp.view({tokens, heads, width}, {stride, width, 1}, token_width);
  kernels::store_kvcache(k.as_const(), v.as_const(),
                         bk.view({blocks, block_size, heads, width}),
                         bv.view({blocks, block_size, heads, width}),
                         bs.view({tokens}).as_const(), stream);
  expect_close(bk.download(stream), expected_k, "strided KV key scatter", true);
  expect_close(bv.download(stream), expected_v, "strided KV value scatter", true);
}

template <typename T>
void test_packed(std::size_t width, bool unequal, cudaStream_t stream) {
  constexpr std::size_t qheads = 4, kvheads = 2;
  const auto cu_q = cumulative(unequal ? std::vector<std::int32_t>{4, 2}
                                       : std::vector<std::int32_t>{2, 0, 3});
  const auto cu_k = cumulative(unequal ? std::vector<std::int32_t>{2, 5}
                                       : std::vector<std::int32_t>{2, 0, 3});
  const auto qt = static_cast<std::size_t>(cu_q.back());
  const auto kt = static_cast<std::size_t>(cu_k.back());
  const auto qs = qheads * width + 7, ks = kvheads * width + 3, vs = kvheads * width + 11;
  std::vector<T> q(qt * qs, T(std::numeric_limits<float>::quiet_NaN()));
  std::vector<T> k(kt * ks, T(std::numeric_limits<float>::quiet_NaN()));
  std::vector<T> v(kt * vs, T(std::numeric_limits<float>::quiet_NaN()));
  for (std::size_t t = 0; t < qt; ++t) {
    for (std::size_t d = 0; d < qheads * width; ++d) q[t * qs + d] = T(std::sin((t * 7 + d) * .17) * 1.3);
  }
  for (std::size_t t = 0; t < kt; ++t) {
    for (std::size_t d = 0; d < kvheads * width; ++d) {
      k[t * ks + d] = T(std::cos((t * 11 + d) * .21) * .9);
      v[t * vs + d] = T(std::sin((t * 5 + d) * .13) * 1.7);
    }
  }
  const float scale = 1.0f / std::sqrt(float(width));
  const auto expected = reference(q, qs, qheads, kvheads, width, cu_q, cu_k, scale,
      [&](std::size_t b, std::int32_t j, std::size_t h, std::size_t d) {
        return number(k[(cu_k[b] + j) * ks + h * width + d]);
      }, [&](std::size_t b, std::int32_t j, std::size_t h, std::size_t d) {
        return number(v[(cu_k[b] + j) * vs + h * width + d]);
      });
  Buffer<T> bq(q.size()), bk(k.size()), bv(v.size()), bo(expected.size());
  Buffer<std::int32_t> bcq(cu_q.size()), bck(cu_k.size());
  bq.upload(q, stream); bk.upload(k, stream); bv.upload(v, stream);
  bcq.upload(cu_q, stream); bck.upload(cu_k, stream);
  kernels::prefill_attention(bq.view({qt, qheads, width}, {qs, width, 1}).as_const(),
      bk.view({kt, kvheads, width}, {ks, width, 1}).as_const(),
      bv.view({kt, kvheads, width}, {vs, width, 1}).as_const(),
      bcq.view({cu_q.size()}).as_const(), bck.view({cu_k.size()}).as_const(), {},
      bo.view({qt, qheads, width}), scale, stream);
  expect_close(bo.download(stream), expected, unequal ? "packed unequal Q/K" : "packed varlen GQA");
}

template <typename T>
void test_paged(std::size_t width, bool decode, cudaStream_t stream) {
  constexpr std::size_t qheads = 4, kvheads = 2, blocks = 8, block_size = 3, columns = 3;
  const std::vector<std::int32_t> lengths = decode
      ? std::vector<std::int32_t>{7, 1, 0, 5} : std::vector<std::int32_t>{7, 5, 0};
  const auto cu_k = cumulative(lengths);
  const auto cu_q = cumulative(decode ? std::vector<std::int32_t>(lengths.size(), 1)
                                     : std::vector<std::int32_t>{3, 2, 0});
  const std::vector<std::int32_t> tables = decode
      ? std::vector<std::int32_t>{6, 1, 4, 0, -1, -1, -1, -1, -1, 5, 3, -1}
      : std::vector<std::int32_t>{6, 1, 4, 5, 3, -1, -1, -1, -1};
  const auto qt = static_cast<std::size_t>(cu_q.back());
  const auto qs = qheads * width + 5;
  const auto token_width = kvheads * width;
  const auto cache_size = blocks * block_size * token_width;
  const T poison = T(std::numeric_limits<float>::quiet_NaN());
  std::vector<T> q(qt * qs, poison), k(cache_size, poison), v(cache_size, poison);
  for (std::size_t t = 0; t < qt; ++t) {
    if (decode && lengths[t] == 0) continue;
    for (std::size_t d = 0; d < qheads * width; ++d) {
      // Large finite scores exercise overflow-safe softmax.
      q[t * qs + d] = T(std::sin((t * 7 + d) * .17) * (decode ? 20.0 : 1.3));
    }
  }
  const auto offset = [&](std::size_t b, std::int32_t j, std::size_t h, std::size_t d) {
    const auto physical = tables[b * columns + j / block_size];
    return (static_cast<std::size_t>(physical) * block_size + j % block_size)
           * token_width + h * width + d;
  };
  for (std::size_t b = 0; b < lengths.size(); ++b) {
    for (std::int32_t j = 0; j < lengths[b]; ++j) {
      for (std::size_t h = 0; h < kvheads; ++h) {
        for (std::size_t d = 0; d < width; ++d) {
          const auto at = offset(b, j, h, d);
          k[at] = T(std::cos((b * 19 + j * 11 + h * width + d) * .21)
                    * (decode ? 25.0 : .9));
          v[at] = T(std::sin((b * 17 + j * 5 + h * width + d) * .13) * 1.7);
        }
      }
    }
  }
  const float scale = 1.0f / std::sqrt(float(width));
  const auto expected = reference(q, qs, qheads, kvheads, width, cu_q, cu_k, scale,
      [&](std::size_t b, std::int32_t j, std::size_t h, std::size_t d) { return number(k[offset(b, j, h, d)]); },
      [&](std::size_t b, std::int32_t j, std::size_t h, std::size_t d) { return number(v[offset(b, j, h, d)]); });
  Buffer<T> bq(q.size()), bk(k.size()), bv(v.size()), bo(expected.size());
  Buffer<std::int32_t> bt(tables.size()), bqcu(cu_q.size()), bkcu(cu_k.size()), bl(lengths.size());
  bq.upload(q, stream); bk.upload(k, stream); bv.upload(v, stream);
  bt.upload(tables, stream); bqcu.upload(cu_q, stream); bkcu.upload(cu_k, stream); bl.upload(lengths, stream);
  const auto qv = bq.view({qt, qheads, width}, {qs, width, 1});
  const auto kv = bk.view({blocks, block_size, kvheads, width});
  const auto vv = bv.view({blocks, block_size, kvheads, width});
  const auto table = bt.view({lengths.size(), columns});
  if (decode) {
    kernels::paged_attention_decode(qv.as_const(), kv.as_const(), vv.as_const(),
        bl.view({lengths.size()}).as_const(), table.as_const(),
        bo.view({qt, qheads, width}), scale, stream);
  } else {
    kernels::prefill_attention(qv.as_const(), kv.as_const(), vv.as_const(),
        bqcu.view({cu_q.size()}).as_const(), bkcu.view({cu_k.size()}).as_const(),
        table.as_const(), bo.view({qt, qheads, width}), scale, stream);
  }
  expect_close(bo.download(stream), expected, decode ? "paged decode GQA" : "paged prefix/chunk prefill");
}

template <typename T>
void test_masked_poison(cudaStream_t stream) {
  const T poison = T(std::numeric_limits<float>::quiet_NaN());
  Buffer<T> q(2), k(2), v(2), out(2);
  Buffer<std::int32_t> lengths(2);
  q.upload({T(1.0f), T(1.0f)}, stream);
  k.upload({T(1.0f), poison}, stream);
  v.upload({T(3.0f), poison}, stream);
  lengths.upload({0, 2}, stream);
  kernels::prefill_attention(q.view({2, 1, 1}).as_const(), k.view({2, 1, 1}).as_const(),
      v.view({2, 1, 1}).as_const(), lengths.view({2}).as_const(),
      lengths.view({2}).as_const(), {}, out.view({2, 1, 1}), 1.0f, stream);
  const auto result = out.download(stream);
  if (number(result[0]) != 3.0) throw std::runtime_error("causal mask read future NaN values");
  // The second row legitimately sees the poisoned key; only the first must be isolated.
}

// Compare the same Q/K/V values across packed, cached chunks, shared prefixes,
// reordered batches and decode. This isolates layout/routing from GEMM rounding.
template <typename T>
void test_prefill_decode_equivalence(cudaStream_t stream) {
  constexpr std::size_t heads = 16, kvheads = 8, width = 128, block_size = 2;
  constexpr std::size_t blocks = 8, columns = 4, shared_tokens = 4;
  const std::vector<std::int32_t> lengths{7, 5, 8};
  const auto full_cu = cumulative(lengths);
  const std::size_t tokens = full_cu.back();
  const std::size_t qw = heads * width, kw = kvheads * width;
  const std::vector<std::int32_t> tables{5, 1, 6, 0, 5, 1, 4, -1, 5, 1, 7, 3};
  std::vector<T> q(tokens * qw), k(tokens * kw), v(tokens * kw);
  std::vector<std::int32_t> slots(tokens);
  for (std::size_t b = 0; b < lengths.size(); ++b) {
    for (std::int32_t j = 0; j < lengths[b]; ++j) {
      const auto token = full_cu[b] + j;
      const auto salt = j < shared_tokens ? 0 : b * 13;
      for (std::size_t d = 0; d < qw; ++d) q[token * qw + d] = T(std::sin((j * 5 + salt + d) * .113) * 1.7);
      for (std::size_t d = 0; d < kw; ++d) {
        k[token * kw + d] = T(std::cos((j * 7 + salt + d) * .137) * .9);
        v[token * kw + d] = T(std::sin((j * 11 + salt + d) * .153) * 2.1);
      }
      slots[token] = b > 0 && j < shared_tokens ? -1
          : static_cast<std::int32_t>(tables[b * columns + j / block_size]
                                      * block_size + j % block_size);
    }
  }
  Buffer<T> bq(q.size()), bk(k.size()), bv(v.size()), full_output(q.size());
  Buffer<T> kc(blocks * block_size * kw), vc(blocks * block_size * kw);
  Buffer<std::int32_t> bcu(full_cu.size()), bslots(tokens);
  bq.upload(q, stream); bk.upload(k, stream); bv.upload(v, stream);
  bcu.upload(full_cu, stream); bslots.upload(slots, stream);
  kernels::prefill_attention(bq.view({tokens, heads, width}).as_const(),
      bk.view({tokens, kvheads, width}).as_const(), bv.view({tokens, kvheads, width}).as_const(),
      bcu.view({full_cu.size()}).as_const(), bcu.view({full_cu.size()}).as_const(), {},
      full_output.view({tokens, heads, width}), .125f, stream);
  const auto full = full_output.download(stream);
  kernels::store_kvcache(bk.view({tokens, kvheads, width}).as_const(),
      bv.view({tokens, kvheads, width}).as_const(),
      kc.view({blocks, block_size, kvheads, width}), vc.view({blocks, block_size, kvheads, width}),
      bslots.view({tokens}).as_const(), stream);
  auto compare = [&](const std::vector<std::size_t>& order,
                      const std::vector<std::int32_t>& starts,
                      const std::vector<std::int32_t>& counts, bool decode) {
    const auto cu_q = cumulative(counts);
    std::vector<std::int32_t> key_lengths, selected_tables;
    std::vector<T> selected_query;
    std::vector<double> expected;
    for (std::size_t row = 0; row < order.size(); ++row) {
      const auto sequence = order[row];
      key_lengths.push_back(starts[row] + counts[row]);
      selected_tables.insert(selected_tables.end(), tables.begin() + sequence * columns,
                              tables.begin() + (sequence + 1) * columns);
      for (auto j = starts[row]; j < starts[row] + counts[row]; ++j) {
        const auto token = full_cu[sequence] + j;
        selected_query.insert(selected_query.end(), q.begin() + token * qw,
                                q.begin() + (token + 1) * qw);
        for (std::size_t d = 0; d < qw; ++d) expected.push_back(number(full[token * qw + d]));
      }
    }
    const auto cu_k = cumulative(key_lengths);
    const auto nt = static_cast<std::size_t>(cu_q.back());
    Buffer<T> query(selected_query.size()), output(selected_query.size());
    Buffer<std::int32_t> cq(cu_q.size()), ck(cu_k.size()), table(selected_tables.size()), lens(key_lengths.size());
    query.upload(selected_query, stream); cq.upload(cu_q, stream); ck.upload(cu_k, stream);
    table.upload(selected_tables, stream); lens.upload(key_lengths, stream);
    if (decode) {
      kernels::paged_attention_decode(query.view({nt, heads, width}).as_const(),
          kc.view({blocks, block_size, kvheads, width}).as_const(),
          vc.view({blocks, block_size, kvheads, width}).as_const(),
          lens.view({order.size()}).as_const(), table.view({order.size(), columns}).as_const(),
          output.view({nt, heads, width}), .125f, stream);
    } else {
      kernels::prefill_attention(query.view({nt, heads, width}).as_const(),
          kc.view({blocks, block_size, kvheads, width}).as_const(),
          vc.view({blocks, block_size, kvheads, width}).as_const(),
          cq.view({cu_q.size()}).as_const(), ck.view({cu_k.size()}).as_const(),
          table.view({order.size(), columns}).as_const(), output.view({nt, heads, width}), .125f, stream);
    }
    expect_close(output.download(stream), expected, "packed/cached/shared-prefix equivalence", true);
  };
  compare({0, 1, 2}, {0, 0, 0}, {3, 3, 3}, false);
  compare({0, 1, 2}, {3, 3, 3}, {3, 2, 3}, false);
  compare({0}, {0}, {3}, false);
  compare({0}, {3}, {3}, false);
  compare({2, 0, 1}, {6, 6, 4}, {2, 1, 1}, false);
  compare({2, 0, 1}, {7, 6, 4}, {1, 1, 1}, true);
  compare({0}, {6}, {1}, true);
  compare({0, 2}, {6, 7}, {1, 1}, true);
}

template <typename T>
void test_empty(cudaStream_t stream) {
  Buffer<T> empty(0), query(8), output(8);
  Buffer<std::int32_t> zero(1), no_indices(0), lengths(2), slots(2);
  zero.upload({0}, stream); lengths.upload({0, 0}, stream); slots.upload({-1, -1}, stream);
  query.upload(std::vector<T>(8, T(std::numeric_limits<float>::quiet_NaN())), stream);
  auto cache = empty.view({0, 3, 1, 4});
  kernels::prefill_attention(empty.view({0, 1, 4}).as_const(),
      empty.view({0, 1, 4}).as_const(), empty.view({0, 1, 4}).as_const(),
      zero.view({1}).as_const(), zero.view({1}).as_const(), {}, empty.view({0, 1, 4}), .5f, stream);
  kernels::paged_attention_decode(empty.view({0, 1, 4}).as_const(), cache.as_const(),
      cache.as_const(), no_indices.view({0}).as_const(), no_indices.view({0, 0}).as_const(),
      empty.view({0, 1, 4}), .5f, stream);
  kernels::paged_attention_decode(query.view({2, 1, 4}).as_const(), cache.as_const(),
      cache.as_const(), lengths.view({2}).as_const(), no_indices.view({2, 0}).as_const(),
      output.view({2, 1, 4}), .5f, stream);
  expect_close(output.download(stream), std::vector<double>(8, 0.0), "zero context with empty cache", true);
  kernels::store_kvcache(empty.view({0, 1, 4}).as_const(), empty.view({0, 1, 4}).as_const(),
      cache, cache, no_indices.view({0}).as_const(), stream);
  kernels::store_kvcache(query.view({2, 1, 4}).as_const(), query.view({2, 1, 4}).as_const(),
      cache, cache, slots.view({2}).as_const(), stream);
  check(cudaStreamSynchronize(stream), "empty cache skip stores");
}

void test_grid_stride(cudaStream_t stream) {
  constexpr std::size_t rows = 65537;
  Buffer<float> q(rows), k(1), v(1), output(rows);
  Buffer<std::int32_t> lengths(rows), tables(rows);
  q.upload(std::vector<float>(rows, 2.0f), stream); k.upload({.25f}, stream); v.upload({3.0f}, stream);
  lengths.upload(std::vector<std::int32_t>(rows, 1), stream);
  tables.upload(std::vector<std::int32_t>(rows, 0), stream);
  kernels::paged_attention_decode(q.view({rows, 1, 1}).as_const(), k.view({1, 1, 1, 1}).as_const(),
      v.view({1, 1, 1, 1}).as_const(), lengths.view({rows}).as_const(), tables.view({rows, 1}).as_const(),
      output.view({rows, 1, 1}), 1.0f, stream);
  expect_close(output.download(stream), std::vector<double>(rows, 3.0), "grid stride and shared cache", true);
}

void test_invalid(cudaStream_t stream) {
  Buffer<float> q(2048), k(2048), v(2048), out(2048);
  Buffer<__half> half(8);
  Buffer<std::int32_t> indices(8);
  const auto qv = q.view({2, 4, 4});
  const auto kv = k.view({2, 2, 4});
  const auto vv = v.view({2, 2, 4});
  const auto ov = out.view({2, 4, 4});
  const auto cu = indices.view({2});
  const auto kc = k.view({2, 3, 2, 4});
  const auto vc = v.view({2, 3, 2, 4});
  auto prefill = [&](ConstTensorView query, ConstTensorView key, ConstTensorView value,
                     ConstTensorView bounds, TensorView output, float scale) {
    kernels::prefill_attention(query, key, value, bounds, cu.as_const(), {}, output, scale, stream);
  };
  for (const auto scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN()}) {
    expect_invalid([&] { prefill(qv.as_const(), kv.as_const(), vv.as_const(), cu.as_const(), ov, scale); }, "attention scale");
  }
  expect_invalid([&] { prefill(q.view({2, 3, 4}).as_const(), kv.as_const(), vv.as_const(), cu.as_const(), out.view({2, 3, 4}), 1); }, "GQA heads");
  expect_invalid([&] { prefill(q.view({1, 1, 257}).as_const(), k.view({1, 1, 257}).as_const(), v.view({1, 1, 257}).as_const(), cu.as_const(), out.view({1, 1, 257}), 1); }, "head dimension limit");
  expect_invalid([&] { prefill(q.view({2, 4, 4}, {15, 4, 1}).as_const(), kv.as_const(), vv.as_const(), cu.as_const(), ov, 1); }, "overlapping query rows");
  expect_invalid([&] { prefill(qv.as_const(), kv.as_const(), vv.as_const(), cu.as_const(), qv, 1); }, "attention output alias");
  expect_invalid([&] { prefill(qv.as_const(), kv.as_const(), vv.as_const(), q.view({2}).as_const(), ov, 1); }, "boundary dtype");
  expect_invalid([&] { prefill(qv.as_const(), kv.as_const(), vv.as_const(), indices.view({0}).as_const(), ov, 1); }, "boundary shape");
  expect_invalid([&] { prefill(qv.as_const(), kv.as_const(), half.view({2, 2, 4}).as_const(), cu.as_const(), ov, 1); }, "KV dtype mismatch");
  expect_invalid([&] { kernels::prefill_attention(qv.as_const(), kc.as_const(), vc.as_const(), cu.as_const(), cu.as_const(), indices.view({2, 2}).as_const(), ov, 1, stream); }, "prefill table batch");
  expect_invalid([&] { kernels::paged_attention_decode(qv.as_const(), kc.as_const(), vc.as_const(), indices.view({1}).as_const(), indices.view({2, 2}).as_const(), ov, 1, stream); }, "decode lengths shape");
  expect_invalid([&] { kernels::paged_attention_decode(qv.as_const(), kv.as_const(), vv.as_const(), indices.view({2}).as_const(), indices.view({2, 2}).as_const(), ov, 1, stream); }, "cache rank");
  expect_invalid([&] { kernels::store_kvcache(kv.as_const(), vv.as_const(), kc, kc, indices.view({2}).as_const(), stream); }, "store cache alias");
  expect_invalid([&] { kernels::store_kvcache(kv.as_const(), vv.as_const(), kc, vc, indices.view({2}).as_const(), stream); }, "store source cache alias");
  expect_invalid([&] { kernels::store_kvcache(q.view({2, 2, 4}).as_const(), vv.as_const(), kc, out.view({2, 3, 2, 4}), indices.view({1}).as_const(), stream); }, "store slots shape");
  check(cudaStreamSynchronize(stream), "invalid host input tests");
}

template <typename T>
void run_dtype(cudaStream_t stream, const char* name) {
  for (const auto width : {std::size_t{1}, std::size_t{7}, std::size_t{128}, std::size_t{256}}) {
    test_store<T>(width, stream);
    test_packed<T>(width, false, stream);
    test_packed<T>(width, true, stream);
    test_paged<T>(width, false, stream);
    test_paged<T>(width, true, stream);
  }
  test_masked_poison<T>(stream);
  test_prefill_decode_equivalence<T>(stream);
  test_empty<T>(stream);
  std::cout << "PASS " << name << " KV store, packed/cached prefill, paged decode, GQA, masks and empty cases\n";
}
} // namespace

int main() {
  int devices = 0;
  const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
      || (status == cudaSuccess && devices == 0)) {
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
    test_grid_stride(stream);
    test_invalid(stream);
    check(cudaStreamDestroy(stream), "destroy stream");
    std::cout << "PASS attention grid-stride and host validation\n";
    return 0;
  } catch (const std::exception& error) {
    if (stream) (void)cudaStreamDestroy(stream);
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
