#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "attn_bridge.hh"
#include "common/cuda_utils.hh"
#include "kernels/cuda/kernels.hh"

namespace {

using nanovllm::check_cuda;

struct Buffer {
  void* pointer = nullptr;
  std::size_t bytes;

  explicit Buffer(std::size_t size) : bytes(std::max<std::size_t>(size, 1)) {
    check_cuda(cudaMalloc(&pointer, bytes), "allocate native check buffer");
  }

  ~Buffer() {
    cudaFree(pointer);
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

template <typename T>
void upload(Buffer& buffer, const std::vector<T>& values, cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(buffer.pointer, values.data(), values.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream), "upload native check");
  check_cuda(cudaStreamSynchronize(stream), "finish host upload before releasing temporary input");
}

template <typename T>
std::vector<T> download(const Buffer& buffer, std::size_t count, cudaStream_t stream) {
  std::vector<T> values(count);
  check_cuda(cudaMemcpyAsync(values.data(), buffer.pointer, count * sizeof(T),
                              cudaMemcpyDeviceToHost, stream), "download native check");
  check_cuda(cudaStreamSynchronize(stream), "finish native check");
  return values;
}

template <typename T>
void check_case(int type, int heads, int kv_heads, int dimension, int queries,
                 int prefix, bool causal, cudaStream_t stream, int page_size = 16) {
  constexpr int guard = 32;
  const int keys = prefix + queries;
  const int pages = (keys + page_size - 1) / page_size;
  const int cache_pages = pages + 2;
  const std::size_t channels = kv_heads * dimension;
  const std::size_t cache_elements = cache_pages * page_size * channels;
  const std::size_t output_elements = queries * heads * dimension;

  std::vector<std::int32_t> table(pages);
  std::iota(table.begin(), table.end(), 0);
  std::mt19937 generator(42);
  std::shuffle(table.begin(), table.end(), generator);
  std::vector<T> key_cache(cache_elements, static_cast<T>(0.0f));
  std::vector<T> value_cache(cache_elements, static_cast<T>(19.0f));
  std::vector<T> new_key(queries * channels, static_cast<T>(0.0f));
  std::vector<T> new_value(queries * channels);
  const auto slot = [&](int position) {
    return table[position / page_size] * page_size + position % page_size;
  };

  for (int position = 0; position < keys; ++position) {
    for (int head = 0; head < kv_heads; ++head) {
      for (int d = 0; d < dimension; ++d) {
        const T value = static_cast<T>(position + head * 64.0f + d * 0.01f);
        const auto channel = head * dimension + d;
        const auto offset = slot(position) * channels + channel;
        if (position < prefix) {
          value_cache[offset] = value;
        } else {
          new_value[(position - prefix) * channels + channel] = value;
          key_cache[offset] = static_cast<T>(13.0f);
          value_cache[offset] = static_cast<T>(-17.0f);
        }
      }
    }
  }

  auto expected_key = key_cache;
  auto expected_value = value_cache;
  for (int i = 0; i < queries; ++i) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      expected_key[slot(prefix + i) * channels + channel] = new_key[i * channels + channel];
      expected_value[slot(prefix + i) * channels + channel] = new_value[i * channels + channel];
    }
  }

  Buffer q(queries * heads * dimension * sizeof(T));
  Buffer kc(cache_elements * sizeof(T));
  Buffer vc(cache_elements * sizeof(T));
  Buffer k(new_key.size() * sizeof(T));
  Buffer v(new_value.size() * sizeof(T));
  Buffer cuq(2 * sizeof(std::int32_t));
  Buffer cuk(2 * sizeof(std::int32_t));
  Buffer blocks(table.size() * sizeof(std::int32_t));
  Buffer out((output_elements + 2 * guard) * sizeof(T));

  upload(q, std::vector<T>(queries * heads * dimension, static_cast<T>(0.0f)), stream);
  upload(kc, key_cache, stream);
  upload(vc, value_cache, stream);
  upload(k, new_key, stream);
  upload(v, new_value, stream);
  upload(cuq, std::vector<std::int32_t>{0, queries}, stream);
  upload(cuk, std::vector<std::int32_t>{0, keys}, stream);
  upload(blocks, table, stream);
  upload(out, std::vector<T>(output_elements + 2 * guard, static_cast<T>(-31.0f)), stream);

  NanoAttnArgs arguments{
      q.pointer, kc.pointer, vc.pointer, k.pointer, v.pointer, cuq.pointer,
      cuk.pointer, blocks.pointer, static_cast<T*>(out.pointer) + guard, stream,
      type, 1, queries, heads, kv_heads, dimension, cache_pages, page_size,
      pages, queries, keys, static_cast<int>(causal)};

  std::uint64_t workspace_elements = 0;
  if (nano_attn_workspace_size(&arguments, &workspace_elements) != 0) {
    throw std::runtime_error(nano_attn_error());
  }
  if (queries == 1) {
    // Existing callers of the decode-only sizing API must remain valid after
    // dispatching D=256 decode to grouped tensor-core partitions.
    workspace_elements = nanovllm::kernels::cuda::flash_attn_decode_workspace_elements(
        1, heads, dimension, keys);
  }
  Buffer workspace((workspace_elements + 2 * guard) * sizeof(float));
  upload(workspace, std::vector<float>(workspace_elements + 2 * guard, -37.0f), stream);
  arguments.workspace = static_cast<float*>(workspace.pointer) + guard;
  arguments.workspace_elements = workspace_elements;

  if (nano_attn_forward(&arguments) != 0) {
    throw std::runtime_error(nano_attn_error());
  }

  const auto actual_key = download<T>(kc, cache_elements, stream);
  const auto actual_value = download<T>(vc, cache_elements, stream);
  const auto actual = download<T>(out, output_elements + 2 * guard, stream);
  const auto scratch = download<float>(workspace, workspace_elements + 2 * guard, stream);
  for (int i = 0; i < guard; ++i) {
    if (scratch[i] != -37.0f || scratch[guard + workspace_elements + i] != -37.0f) {
      throw std::runtime_error("native check: workspace guard overwritten");
    }
  }
  for (std::size_t i = 0; i < cache_elements; ++i) {
    if (static_cast<float>(actual_key[i]) != static_cast<float>(expected_key[i])
        || static_cast<float>(actual_value[i]) != static_cast<float>(expected_value[i])) {
      throw std::runtime_error("native check: cache/padding mutation mismatch");
    }
  }
  for (int i = 0; i < guard; ++i) {
    if (static_cast<float>(actual[i]) != -31.0f
        || static_cast<float>(actual[guard + output_elements + i]) != -31.0f) {
      throw std::runtime_error("native check: output guard overwritten");
    }
  }
  const double tolerance = type == 0 ? 2e-4 : type == 1 ? 2e-3 : 1.6e-2;
  for (int query : {0, queries - 1}) {
    for (int head = 0; head < heads; ++head) {
      for (int d : {0, dimension - 1}) {
        const int visible = causal ? prefix + query + 1 : keys;
        double expected = 0.0;
        for (int position = 0; position < visible; ++position) {
          const auto channel = (head / (heads / kv_heads)) * dimension + d;
          expected += static_cast<float>(expected_value[slot(position) * channels + channel]);
        }
        expected /= visible;
        const auto index = guard + (query * heads + head) * dimension + d;
        const float value = static_cast<float>(actual[index]);
        if (!std::isfinite(value) || std::abs(value - expected) > tolerance * (1 + std::abs(expected))) {
          throw std::runtime_error("native check: mask or GQA mismatch");
        }
      }
    }
  }
}

template <typename T>
int check_type(int type, cudaStream_t stream) {
  int count = 0;
  for (int dimension : {80, 128, 256}) {
    for (int queries : {3, 17}) {
      for (bool causal : {true, false}) {
        check_case<T>(type, 16, dimension == 80 ? 1 : 2, dimension, queries, 17,
                      causal, stream);
        ++count;
      }
    }
    for (int prefix : {0, 510, 511, 512, 2048}) {
      for (bool causal : {true, false}) {
        check_case<T>(type, 16, dimension == 80 ? 1 : 2, dimension, 1,
                      prefix, causal, stream);
        ++count;
      }
    }
  }
  for (bool causal : {true, false}) {
    check_case<T>(type, 8, 1, 80, 1, 8192, causal, stream);
    ++count;
    for (int dimension : {128, 256}) {
      for (int queries : {3, 17, 65}) {
        check_case<T>(type, 16, dimension == 128 ? 8 : 2, dimension,
                      queries, 513, causal, stream, 7);
        ++count;
      }
    }
  }
  return count;
}

} // namespace

int main() {
  try {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create check stream");
    const int count = check_type<float>(0, stream) + check_type<__half>(1, stream)
                      + check_type<__nv_bfloat16>(2, stream);
    check_cuda(cudaStreamDestroy(stream), "destroy check stream");
    std::cout << "PASS: " << count << " native checks of the production attention kernel\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
