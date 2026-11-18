#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <type_traits>

namespace nanovllm::kernels::cuda::wmma_ext {

enum class Layout { RowMajor, ColMajor };

template <typename T>
struct FragmentA {
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>);

  using value_type = T;
  static constexpr int num_elements = 8;
  static constexpr int num_registers = 4;
  unsigned reg[num_registers];
};

template <typename T, Layout MemoryLayout = Layout::ColMajor>
struct FragmentB {
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>);

  using value_type = T;
  static constexpr int num_elements = 4;
  static constexpr int num_registers = 2;
  unsigned reg[num_registers];
};

struct Accumulator {
  using value_type = float;
  static constexpr int num_elements = 4;
  float x[num_elements] = {};

  __device__ __forceinline__ float& operator[](int i) { return x[i]; }
  const __device__ __forceinline__ float& operator[](int i) const {
    return x[i];
  }
};

// Per-row state follows the accumulator's distributed row layout. Each value
// is replicated across the lanes owning that row; access it through
// row.state().
template <typename T>
class RowState {
 public:
  __device__ explicit RowState(T initial = T{})
      : values_{initial, initial} {}

 private:
  T values_[2];

  template <int, int>
  friend class AccumulatorRow;
};

// A logical row of horizontally concatenated 16x8 accumulator fragments.
// The view references registers; it does not gather or copy the row's values.
template <int Fragments, int RowPart>
class AccumulatorRow {
  static_assert(Fragments > 0 && RowPart >= 0 && RowPart < 2);

 public:
  __device__ explicit AccumulatorRow(Accumulator (&fragments)[Fragments])
      : fragments_(fragments) {}

  // Row index within the logical 16 x (8 * Fragments) matrix.
  __device__ __forceinline__ int index() const {
    return (threadIdx.x % 32) / 4 + RowPart * 8;
  }

  // Exactly one lane per logical row, for writing replicated row statistics.
  __device__ __forceinline__ bool is_leader() const {
    return threadIdx.x % 4 == 0;
  }

  template <typename T>
  __device__ __forceinline__ T& state(RowState<T>& state) const {
    return state.values_[RowPart];
  }

  // Visit locally owned elements as function(logical_column, float& value).
  // This callback must not contain warp collectives or block barriers.
  template <typename Function>
  __device__ __forceinline__ void for_each(Function function) const {
    const int col = (threadIdx.x % 4) * 2;
#pragma unroll
    for (int n = 0; n < Fragments; ++n) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        function(n * 8 + col + j, fragments_[n][2 * RowPart + j]);
      }
    }
  }

  template <typename Function>
  __device__ __forceinline__ void transform(Function function) const {
    for_each([&](int col, float& value) { value = function(col, value); });
  }

  // All 32 lanes must call row reductions in the same order. XOR masks 1 and
  // 2 reduce within the four lanes owning this row, never across row groups.
  __device__ __forceinline__ float max() const {
    float value = -CUDART_INF_F;
    for_each([&](int, float element) { value = fmaxf(value, element); });
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
    return value;
  }

  __device__ __forceinline__ float sum() const {
    float value = 0.f;
    for_each([&](int, float element) { value += element; });
    value += __shfl_xor_sync(0xffffffff, value, 1);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    return value;
  }

  // Scale the corresponding row of another tile, possibly of a different width.
  template <int OtherFragments>
  __device__ __forceinline__ void scale(Accumulator (&other)[OtherFragments],
                                        float factor) const {
    AccumulatorRow<OtherFragments, RowPart>(other).for_each(
        [&](int, float& value) { value *= factor; });
  }

  // Store into a row-major matrix, converting FP32 values to T. The pointer is
  // the matrix origin, not the row origin; stride is in elements. No barrier.
  template <typename T>
  __device__ __forceinline__ void store(T* dst, int stride) const {
    for_each([&](int col, float value) {
      dst[index() * stride + col] = static_cast<T>(value);
    });
  }

 private:
  Accumulator (&fragments_)[Fragments];
};

// Cooperatively visit all 16 rows of a horizontally concatenated tile. Every
// lane calls function(row) twice, once for each row it owns. All 32 lanes must
// participate; branches around row reductions must be uniform across the warp.
// RowPart is a template argument so register indices remain compile-time known.
// No shared memory, block synchronization, or data rearrangement is performed.
template <int Fragments, typename Function>
__device__ __forceinline__ void for_each_row(
    Accumulator (&fragments)[Fragments], Function function) {
  function(AccumulatorRow<Fragments, 0>(fragments));
  function(AccumulatorRow<Fragments, 1>(fragments));
}

__device__ __forceinline__ void fill_fragment(Accumulator& acc, float value) {
#pragma unroll
  for (int i = 0; i < Accumulator::num_elements; ++i) {
    acc.x[i] = value;
  }
}

// A is a row-major 16x16 tile. Strides are measured in elements.
template <typename T>
__device__ __forceinline__ void load_matrix_sync(FragmentA<T>& fragment,
                                                 const T* src, int stride) {
  const int lane = threadIdx.x % 32;
  const auto address = static_cast<unsigned>(
      __cvta_generic_to_shared(src + (lane % 16) * stride + (lane / 16) * 8));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(fragment.reg[0]), "=r"(fragment.reg[1]),
                 "=r"(fragment.reg[2]), "=r"(fragment.reg[3])
               : "r"(address));
}

// B is a logical 16x8 tile, stored in the selected memory layout.
template <typename T, Layout MemoryLayout>
__device__ __forceinline__ void load_matrix_sync(
    FragmentB<T, MemoryLayout>& fragment, const T* src, int stride) {
  const int lane = threadIdx.x % 32;
  const T* ptr = MemoryLayout == Layout::RowMajor
                     ? src + (lane % 16) * stride
                     : src + (lane % 8) * stride + ((lane / 8) % 2) * 8;
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));

  if constexpr (MemoryLayout == Layout::RowMajor) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(fragment.reg[0]), "=r"(fragment.reg[1])
                 : "r"(address));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(fragment.reg[0]), "=r"(fragment.reg[1])
                 : "r"(address));
  }
}

// In-place C = A * B + C. The four FP32 values occupy two rows, eight apart.
template <typename T, Layout MemoryLayout>
__device__ __forceinline__ void mma_sync(Accumulator& acc,
                                         const FragmentA<T>& a,
                                         const FragmentB<T, MemoryLayout>& b) {
  if constexpr (std::is_same_v<T, __half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc.x[0]), "+f"(acc.x[1]), "+f"(acc.x[2]), "+f"(acc.x[3])
        : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]),
          "r"(b.reg[0]), "r"(b.reg[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc.x[0]), "+f"(acc.x[1]), "+f"(acc.x[2]), "+f"(acc.x[3])
        : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]),
          "r"(b.reg[0]), "r"(b.reg[1]));
  }
}

// WMMA-style D = A * B + C; D and C may refer to the same accumulator.
template <typename T, Layout MemoryLayout>
__device__ __forceinline__ void mma_sync(Accumulator& dst,
                                         const FragmentA<T>& a,
                                         const FragmentB<T, MemoryLayout>& b,
                                         const Accumulator& src) {
  dst = src;
  mma_sync(dst, a, b);
}

// Load a complete 16x8 accumulator tile in row-major order.
// All lanes participate; stride is in elements. No block synchronization.
__device__ __forceinline__ void load_matrix_sync(Accumulator& acc,
                                                 const float* src, int stride) {
  const int lane = threadIdx.x % 32;
  const int row = lane / 4;
  const int col = (lane % 4) * 2;

#pragma unroll
  for (int i = 0; i < Accumulator::num_elements; ++i) {
    acc.x[i] = src[(row + i / 2 * 8) * stride + col + i % 2];
  }
}

// Store a complete 16x8 accumulator tile in row-major order.
// All lanes participate; stride is in elements. No block synchronization.
__device__ __forceinline__ void store_matrix_sync(float* dst,
                                                  const Accumulator& acc,
                                                  int stride) {
  const int lane = threadIdx.x % 32;
  const int row = lane / 4;
  const int col = (lane % 4) * 2;

#pragma unroll
  for (int i = 0; i < Accumulator::num_elements; ++i) {
    dst[(row + i / 2 * 8) * stride + col + i % 2] = acc.x[i];
  }
}

} // namespace nanovllm::kernels::cuda::wmma_ext
