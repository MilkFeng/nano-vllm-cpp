#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_utils.hh"
#include "kernels/cuda/wmma_ext.cuh"

namespace {

using nanovllm::check_cuda;
namespace wmma = nanovllm::kernels::cuda::wmma_ext;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    check_cuda(cudaMalloc(&data_, count * sizeof(T)), "allocate WMMA check");
  }

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const { return data_; }

 private:
  T* data_ = nullptr;
};

template <typename T, wmma::Layout Layout, int Padding>
__global__ void check_kernel(const T* a, const T* b, float* output,
                             float initial, int repeats, bool in_place,
                             bool load_initial) {
  constexpr int AStride = 16 + Padding;
  constexpr int BStride = (Layout == wmma::Layout::RowMajor ? 8 : 16) + Padding;
  constexpr int BRows = Layout == wmma::Layout::RowMajor ? 16 : 8;
  constexpr int CStride = 8 + Padding;
  __shared__ __align__(16) T shared_a[16 * AStride];
  __shared__ __align__(16) T shared_b[BRows * BStride];
  __shared__ float shared_c[16 * CStride];

  for (int i = threadIdx.x; i < 16 * AStride; i += 32) {
    shared_a[i] = a[i];
  }
  for (int i = threadIdx.x; i < BRows * BStride; i += 32) {
    shared_b[i] = b[i];
  }
  if (load_initial) {
    for (int i = threadIdx.x; i < 16 * 8; i += 32) {
      shared_c[i / 8 * CStride + i % 8] = output[i / 8 * 13 + i % 8];
    }
  }
  __syncthreads();

  wmma::FragmentA<T> a_fragment;
  wmma::FragmentB<T, Layout> b_fragment;
  wmma::Accumulator acc;
  if (load_initial) {
    wmma::load_matrix_sync(acc, shared_c, CStride);
  } else {
    wmma::fill_fragment(acc, initial);
  }
  wmma::load_matrix_sync(a_fragment, shared_a, AStride);
  wmma::load_matrix_sync(b_fragment, shared_b, BStride);

  for (int i = 0; i < repeats; ++i) {
    if (in_place) {
      wmma::mma_sync(acc, a_fragment, b_fragment, acc);
    } else {
      wmma::Accumulator next;
      wmma::mma_sync(next, a_fragment, b_fragment, acc);
      acc = next;
    }
  }
  wmma::store_matrix_sync(output, acc, 13);
}

template <typename T, wmma::Layout Layout, int Padding>
void check_case(int repeats, bool in_place, bool load_initial) {
  constexpr int AStride = 16 + Padding;
  constexpr int BStride = (Layout == wmma::Layout::RowMajor ? 8 : 16) + Padding;
  constexpr int BRows = Layout == wmma::Layout::RowMajor ? 16 : 8;
  constexpr int OutStride = 13;
  constexpr int Guard = 16;
  constexpr float Initial = 0.75f;
  constexpr float Sentinel = -123.f;

  std::vector<T> a(16 * AStride, static_cast<T>(-7.f));
  std::vector<T> b(BRows * BStride, static_cast<T>(-9.f));
  const auto b_index = [](int k, int col) {
    return Layout == wmma::Layout::RowMajor ? k * BStride + col
                                           : col * BStride + k;
  };
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row * AStride + k] =
          static_cast<T>(((row * 5 + k * 3) % 23 - 11) / 16.f);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 8; ++col) {
      b[b_index(k, col)] =
          static_cast<T>(((k * 7 + col * 3) % 19 - 9) / 32.f);
    }
  }

  std::vector<float> expected(Guard + 16 * OutStride + Guard, Sentinel);
  std::vector<float> actual(expected.size(), Sentinel);
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 8; ++col) {
      double product = 0.0;
      for (int k = 0; k < 16; ++k) {
        product += static_cast<double>(static_cast<float>(a[row * AStride + k]))
                   * static_cast<float>(b[b_index(k, col)]);
      }
      const float initial = load_initial ? Initial + (row * 8 + col) / 32.f
                                         : Initial;
      if (load_initial) {
        actual[Guard + row * OutStride + col] = initial;
      }
      expected[Guard + row * OutStride + col] =
          static_cast<float>(initial + repeats * product);
    }
  }

  DeviceBuffer<T> device_a(a.size()), device_b(b.size());
  DeviceBuffer<float> device_output(expected.size());
  check_cuda(cudaMemcpy(device_a.data(), a.data(), a.size() * sizeof(T),
                        cudaMemcpyHostToDevice), "upload WMMA A");
  check_cuda(cudaMemcpy(device_b.data(), b.data(), b.size() * sizeof(T),
                        cudaMemcpyHostToDevice), "upload WMMA B");
  check_cuda(cudaMemcpy(device_output.data(), actual.data(),
                        actual.size() * sizeof(float), cudaMemcpyHostToDevice),
             "initialize WMMA output guards");

  check_kernel<T, Layout, Padding><<<1, 32>>>(
      device_a.data(), device_b.data(), device_output.data() + Guard,
      Initial, repeats, in_place, load_initial);
  check_cuda(cudaGetLastError(), "launch WMMA check");
  check_cuda(cudaMemcpy(actual.data(), device_output.data(),
                        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "download WMMA output");

  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float tolerance = expected[i] == Sentinel
                                ? 0.f
                                : 1e-5f + 1e-5f * std::abs(expected[i]);
    if (!std::isfinite(actual[i])
        || std::abs(actual[i] - expected[i]) > tolerance) {
      throw std::runtime_error("WMMA output or padding mismatch at element "
                               + std::to_string(i));
    }
  }
}

template <typename T, wmma::Layout Layout>
int check_layout() {
  int count = 0;
  for (int repeats : {1, 3}) {
    for (bool in_place : {false, true}) {
      for (bool load_initial : {false, true}) {
        check_case<T, Layout, 0>(repeats, in_place, load_initial);
        check_case<T, Layout, 8>(repeats, in_place, load_initial);
        count += 2;
      }
    }
  }
  return count;
}

struct SoftmaxState {
  float maximum = -INFINITY;
  float sum = 0.f;
  float tile_sum = 0.f;
};

__host__ __device__ bool valid_score(int row, int col, int tile, int warp,
                                    int columns) {
  // Empty rows, initially empty rows, later empty tiles, and ragged columns.
  return row != 15 && !(row == 8 && tile == 0) && !(row == 9 && tile > 0)
         && col < columns - (row + warp) % 7;
}

__host__ __device__ float tile_value(int row, int col, int tile, int warp) {
  return (row + col * 2 + tile * 3 + warp) / 32.f;
}

// Exercise row views with online softmax across three tiles. Each tile has a
// constant V per output channel, so PV is tile_sum * V and has a simple CPU
// reference independent of the fragment layout. Warps own distinct matrices.
template <typename T, int Fragments>
__global__ void check_rows_kernel(const float* input, T* probabilities,
                                  float* output, float* statistics) {
  constexpr int Columns = Fragments * 8;
  constexpr int PStride = Columns + 5;
  constexpr int OutFragments = 2;
  constexpr int OutStride = OutFragments * 8 + 3;
  constexpr int Tiles = 3;
  const int warp = threadIdx.x / 32;
  wmma::Accumulator acc[OutFragments] = {};
  wmma::RowState<SoftmaxState> state;

  for (int tile = 0; tile < Tiles; ++tile) {
    const int matrix = warp * Tiles + tile;
    wmma::Accumulator scores[Fragments];
#pragma unroll
    for (int n = 0; n < Fragments; ++n) {
      wmma::load_matrix_sync(scores[n], input + matrix * 16 * PStride + n * 8,
                             PStride);
    }

    wmma::for_each_row(scores, [&](auto row) {
      auto& stats = row.state(state);
      row.transform([&](int col, float value) {
        return valid_score(row.index(), col, tile, warp, Columns)
                   ? value * 0.5f
                   : -CUDART_INF_F;
      });
      const float maximum = fmaxf(stats.maximum, row.max());
      const float alpha = maximum == -CUDART_INF_F
                              ? 0.f : __expf(stats.maximum - maximum);
      row.transform([&](int, float value) {
        return value == -CUDART_INF_F ? 0.f : __expf(value - maximum);
      });
      stats.tile_sum = row.sum();
      stats.sum = stats.sum * alpha + stats.tile_sum;
      stats.maximum = maximum;
      row.scale(acc, alpha);
      row.store(probabilities + matrix * 16 * PStride, PStride);
    });

    // Access the same row state through a tile with a different column count.
    wmma::for_each_row(acc, [&](auto row) {
      const auto& stats = row.state(state);
      row.for_each([&](int col, float& value) {
        value += stats.tile_sum * tile_value(row.index(), col, tile, warp);
      });
      row.store(output + matrix * 16 * OutStride, OutStride);
      if (row.is_leader()) {
        statistics[matrix * 32 + row.index() * 2] = stats.maximum;
        statistics[matrix * 32 + row.index() * 2 + 1] = stats.sum;
      }
    });
  }
}

template <typename T>
void check_row_values(const std::vector<T>& actual,
                      const std::vector<T>& expected, float tolerance,
                      const std::string& name) {
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float a = static_cast<float>(actual[i]);
    const float e = static_cast<float>(expected[i]);
    if (a == e) {
      continue; // Includes correctly preserved guards and negative infinity.
    }
    if (e == -123.f || !std::isfinite(a) || !std::isfinite(e)
        || std::abs(a - e) > tolerance * (1.f + std::abs(e))) {
      throw std::runtime_error(name + " mismatch at element " + std::to_string(i));
    }
  }
}

template <typename T, int Fragments>
void check_rows() {
  constexpr int Columns = Fragments * 8;
  constexpr int PStride = Columns + 5;
  constexpr int OutColumns = 16;
  constexpr int OutStride = OutColumns + 3;
  constexpr int Tiles = 3;
  constexpr int Warps = 4;
  constexpr int Guard = 16;
  constexpr float Sentinel = -123.f;
  std::vector<float> input(Warps * Tiles * 16 * PStride, Sentinel);
  std::vector<T> probabilities(Guard + input.size() + Guard,
                              static_cast<T>(Sentinel));
  std::vector<float> output(Guard + Warps * Tiles * 16 * OutStride + Guard,
                            Sentinel);
  std::vector<float> statistics(Guard + Warps * Tiles * 32 + Guard, Sentinel);
  auto expected_probabilities = probabilities;
  auto expected_output = output;
  auto expected_statistics = statistics;

  for (int warp = 0; warp < Warps; ++warp) {
    for (int row = 0; row < 16; ++row) {
      double maximum = -INFINITY;
      // Keep all earlier scores and V values, then recompute the reference
      // directly after each tile rather than repeating the GPU recurrence.
      std::vector<double> logits;
      std::vector<int> source_tiles;
      for (int tile = 0; tile < Tiles; ++tile) {
        const int matrix = warp * Tiles + tile;
        for (int col = 0; col < Columns; ++col) {
          const float score = ((row * 7 + col * 3 + warp * 5) % 29 - 14) / 4.f
                              + (tile == 1 ? -4.f : tile == 2 ? 4.f : 0.f);
          input[(matrix * 16 + row) * PStride + col] = score;
          if (valid_score(row, col, tile, warp, Columns)) {
            logits.push_back(score * 0.5);
            source_tiles.push_back(tile);
            maximum = std::fmax(maximum, logits.back());
          }
        }
        double sum = 0.0;
        double numerator[OutColumns] = {};
        for (std::size_t k = 0; k < logits.size(); ++k) {
          const double weight = std::exp(logits[k] - maximum);
          sum += weight;
          for (int col = 0; col < OutColumns; ++col) {
            numerator[col] += weight * tile_value(row, col, source_tiles[k], warp);
          }
        }
        for (int col = 0; col < Columns; ++col) {
          const double score = input[(matrix * 16 + row) * PStride + col] * 0.5;
          const double weight = valid_score(row, col, tile, warp, Columns)
                                    ? std::exp(score - maximum) : 0.0;
          expected_probabilities[Guard + (matrix * 16 + row) * PStride + col] =
              static_cast<T>(static_cast<float>(weight));
        }
        for (int col = 0; col < OutColumns; ++col) {
          expected_output[Guard + (matrix * 16 + row) * OutStride + col] =
              static_cast<float>(numerator[col]);
        }
        expected_statistics[Guard + matrix * 32 + row * 2] =
            static_cast<float>(maximum);
        expected_statistics[Guard + matrix * 32 + row * 2 + 1] =
            static_cast<float>(sum);
      }
    }
  }

  DeviceBuffer<float> device_input(input.size()), device_output(output.size()),
      device_statistics(statistics.size());
  DeviceBuffer<T> device_probabilities(probabilities.size());
  check_cuda(cudaMemcpy(device_input.data(), input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice), "upload row check scores");
  check_cuda(cudaMemcpy(device_probabilities.data(), probabilities.data(),
                        probabilities.size() * sizeof(T), cudaMemcpyHostToDevice),
             "initialize row check probability guards");
  check_cuda(cudaMemcpy(device_output.data(), output.data(), output.size() * sizeof(float),
                        cudaMemcpyHostToDevice), "initialize row check output guards");
  check_cuda(cudaMemcpy(device_statistics.data(), statistics.data(),
                        statistics.size() * sizeof(float), cudaMemcpyHostToDevice),
             "initialize row check statistic guards");
  check_rows_kernel<T, Fragments><<<1, Warps * 32>>>(
      device_input.data(), device_probabilities.data() + Guard,
      device_output.data() + Guard, device_statistics.data() + Guard);
  check_cuda(cudaGetLastError(), "launch row check");
  check_cuda(cudaMemcpy(probabilities.data(), device_probabilities.data(),
                        probabilities.size() * sizeof(T), cudaMemcpyDeviceToHost),
             "download row check probabilities");
  check_cuda(cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost), "download row check output");
  check_cuda(cudaMemcpy(statistics.data(), device_statistics.data(),
                        statistics.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "download row check statistics");
  check_row_values(probabilities, expected_probabilities,
                    std::is_same_v<T, __half> ? 1e-3f : 8e-3f, "row probabilities");
  check_row_values(output, expected_output, 2e-5f, "row output");
  check_row_values(statistics, expected_statistics, 2e-5f, "row statistics");
}

} // namespace

int main() {
  try {
    const int count = check_layout<__half, wmma::Layout::RowMajor>()
                      + check_layout<__half, wmma::Layout::ColMajor>()
                      + check_layout<__nv_bfloat16, wmma::Layout::RowMajor>()
                      + check_layout<__nv_bfloat16, wmma::Layout::ColMajor>();
    std::cout << "PASS: " << count
              << " WMMA checks (FP16/BF16, both B layouts, padded strides, "
                 "accumulator loads, in-place/out-of-place accumulation)\n";
    check_rows<__half, 1>();
    check_rows<__half, 4>();
    check_rows<__half, 8>();
    check_rows<__nv_bfloat16, 1>();
    check_rows<__nv_bfloat16, 4>();
    check_rows<__nv_bfloat16, 8>();
    std::cout << "PASS: 6 row-view checks (8/32/64 columns, FP16/BF16 stores, "
                 "four warps, online softmax, empty rows/tiles, padded guards)\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
