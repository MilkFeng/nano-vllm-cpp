#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "common/cuda_utils.hh"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/kernels.hh"
#include "kernels/cuda/reduce.cuh"

namespace nanovllm::kernels::cuda {

namespace {

constexpr int kQueryTile = 4;
constexpr int kKeyTile = 16;
constexpr int kThreads = kQueryTile * 32;
constexpr int kHeadElements = 256 / 32;
constexpr int kMaxDecodeSplits = 32;
constexpr std::size_t kDecodeTargetBlocks = 1024;
constexpr std::size_t kMinSplitTokens = 128;

int decode_splits(std::size_t batch, std::size_t heads,
                  std::size_t max_seqlen_k) {
  if (batch == 0 || heads == 0 || max_seqlen_k < 2 * kMinSplitTokens
      || heads >= kDecodeTargetBlocks
      || batch >= (kDecodeTargetBlocks + heads - 1) / heads) {
    return 1;
  }
  const auto blocks = batch * heads;
  const auto wanted = (kDecodeTargetBlocks + blocks - 1) / blocks;
  int splits = 1;
  while (splits < wanted && splits < kMaxDecodeSplits
         && 2 * splits <= max_seqlen_k / kMinSplitTokens) {
    splits *= 2;
  }
  return splits;
}

struct AttentionShape {
  std::size_t batch;
  std::size_t query_tokens;
  std::size_t max_seqlen_q, max_seqlen_k;
  std::size_t query_heads, key_heads;
  std::size_t head_dim;
  std::size_t blocks, block_tokens;
  std::size_t max_blocks;
};

AttentionShape attn_shape(ConstTensorView query, TensorView key_cache,
                          TensorView value_cache, ConstTensorView key,
                          ConstTensorView value, ConstTensorView cu_seqlens_q,
                          ConstTensorView cu_seqlens_k,
                          std::size_t max_seqlen_q, std::size_t max_seqlen_k,
                          ConstTensorView block_table, TensorView output) {
  detail::assert_tensor(query);
  detail::assert_tensor(key_cache);
  detail::assert_tensor(value_cache);
  detail::assert_tensor(key);
  detail::assert_tensor(value);
  detail::assert_tensor(cu_seqlens_q);
  detail::assert_tensor(cu_seqlens_k);
  detail::assert_tensor(block_table);
  detail::assert_tensor(output);

  assert(same_dtype(query, key_cache, value_cache, key, value, output));
  assert(same_device(query, key_cache, value_cache, key, value, cu_seqlens_q,
                     cu_seqlens_k, block_table, output));
  assert(query.dtype() == DataType::Float32
         || query.dtype() == DataType::Float16
         || query.dtype() == DataType::BFloat16);

  AttentionShape shape = {};

  assert(same_shape(output, query));
  assert(query.rank() == 3);

  shape.query_tokens = query.shape()[0];
  shape.query_heads = query.shape()[1];
  shape.head_dim = query.shape()[2];
  assert(shape.query_heads > 0 && shape.head_dim > 0 && shape.head_dim <= 256);

  assert(same_shape(key_cache, value_cache));
  assert(key_cache.rank() == 4);

  shape.blocks = key_cache.shape()[0];
  shape.block_tokens = key_cache.shape()[1];
  shape.key_heads = key_cache.shape()[2];

  assert(shape.block_tokens > 0);
  assert(shape.key_heads > 0 && shape.query_heads % shape.key_heads == 0);
  assert(key_cache.shape()[3] == shape.head_dim);

  assert(same_shape(key, value));
  assert(key.rank() == 3);
  assert(key.shape()[0] == shape.query_tokens);
  assert(key.shape()[1] == shape.key_heads && key.shape()[2] == shape.head_dim);

  assert(same_shape(cu_seqlens_q, cu_seqlens_k));
  assert(same_dtype(cu_seqlens_q, cu_seqlens_k));
  assert(cu_seqlens_q.dtype() == DataType::Int32);
  assert(cu_seqlens_q.rank() == 1);
  assert(cu_seqlens_q.shape()[0] >= 1);

  shape.batch = cu_seqlens_q.shape()[0] - 1;

  shape.max_seqlen_q = max_seqlen_q;
  shape.max_seqlen_k = max_seqlen_k;

  assert(shape.batch <= INT_MAX && shape.query_tokens <= INT_MAX);
  assert(max_seqlen_q <= INT_MAX && max_seqlen_k <= INT_MAX);
  assert(shape.query_tokens == 0 || (shape.batch > 0 && max_seqlen_q > 0));
  assert(shape.query_tokens <= shape.batch * max_seqlen_q);

  assert(block_table.dtype() == DataType::Int32);
  assert(block_table.rank() == 2);
  assert(block_table.shape()[0] == shape.batch);

  shape.max_blocks = block_table.shape()[1];

  assert(max_seqlen_k == 0 || shape.blocks > 0);
  assert(max_seqlen_k <= shape.max_blocks * shape.block_tokens);

  return shape;
}

// PTX m16n8k16: each group of four lanes owns two rows, eight apart.
// Keep QK, softmax statistics, and the output accumulator in FP32. Only the
// probabilities passed to the PV tensor-core operation are rounded to T.
template <typename T>
__device__ __forceinline__ void attn_mma(float (&c)[4], unsigned a0,
                                       unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  // The host dispatcher retains the SIMT implementation on pre-Ampere GPUs.
  asm volatile("trap;");
#else
  if constexpr (std::is_same_v<T, __half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
#endif
}

bool tensor_attention_supported() {
  static const bool supported = [] {
    int major = 0, shared_bytes = 0;
    check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0),
                "query attention compute capability");
    check_cuda(cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0),
                "query attention shared memory");
    return major >= 8 && shared_bytes >= 96 * 1024;
  }();
  return supported;
}

template <typename T>
__device__ __forceinline__ void attn_ld_a(const T* p, int stride, unsigned (&a)[4]) {
  const int lane = threadIdx.x % 32;
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(
      p + (lane % 16) * stride + (lane / 16) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
}

template <bool Transpose, typename T>
__device__ __forceinline__ void attn_ld_b(const T* p, int stride, unsigned (&b)[2]) {
  const int lane = threadIdx.x % 32;
  const T* src = Transpose ? p + (lane % 16) * stride
                           : p + (lane % 8) * stride + ((lane / 8) % 2) * 8;
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(src));
  if constexpr (Transpose) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
  }
}

template <typename T>
__device__ __forceinline__ void attn_copy16(T* dst, const T* src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
#else
  *reinterpret_cast<uint4*>(dst) = valid ? *reinterpret_cast<const uint4*>(src) : uint4{};
#endif
}

__device__ __forceinline__ void attn_wait_copy() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
#endif
}

template <typename T, int D, int W = 4, int N = 32, bool SplitKV = false, int Cols = 1>
__global__ void flash_attn_tensor_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_q,
    const std::int32_t* __restrict__ cu_k,
    const std::int32_t* __restrict__ table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* partials, int splits, int split_tokens) {
  // Flatten (query position, GQA head) into rows sharing one KV head.
  // Cols=1 assigns 16 rows to each warp. For D=256 decode, Cols=4 shares
  // those 16 rows across warps: partition QK columns, exchange scores, then
  // accumulate disjoint 64-channel slices of PV to reduce register pressure.
  static_assert(W % Cols == 0 && N % (8 * Cols) == 0);
  constexpr bool Columns = Cols > 1;
  constexpr int M = W / Cols * 16;
  constexpr int OD = D / Cols;
  constexpr int DS = D + 8;
  constexpr int NS = N + 8;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int qwarp = warp / Cols;
  const int colwarp = warp % Cols;
  const int r = lane / 4;
  const int t = lane % 4;
  const int batch = blockIdx.z;
  const int group = shape.query_heads / shape.key_heads;
  const int head_base = blockIdx.y * group;
  const int qstart = cu_q[batch];
  const int qlen = cu_q[batch + 1] - qstart;
  const int klen = cu_k[batch + 1] - cu_k[batch];
  const int prefix = klen - qlen;
  const int page_tokens = shape.block_tokens;
  const int qt = blockIdx.x / splits;
  const int split = blockIdx.x % splits;
  const int qbase = qt * M;
  if (qbase >= qlen * group) return;
  const int kh = blockIdx.y;
  extern __shared__ __align__(16) unsigned char memory[];
  T* sq = reinterpret_cast<T*>(memory);
  T* sk = sq + M * DS;
  T* sv = sk + N * DS;
  T* sp = sv + N * DS;
  for (int i = threadIdx.x * 8; i < M * D; i += W * 32 * 8) {
    const int row = i / D;
    const int d = i % D;
    const int qp = (qbase + row) / group;
    const int head = head_base + (qbase + row) % group;
    const auto offset = ((qstart + qp) * shape.query_heads + head) * D + d;
    attn_copy16(sq + row * DS + d, qp < qlen ? query + offset : query, qp < qlen);
  }
  attn_wait_copy();
  float acc[OD / 8][4] = {};
  float maxima[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float sums[2] = {};
  const float scale = rsqrtf(float(D));
  const int visible = causal ? min(klen, prefix + (qbase + M - 1) / group + 1) : klen;
  const int begin = SplitKV ? split * split_tokens : 0;
  const int end = SplitKV ? min(visible, begin + split_tokens) : visible;
  __shared__ std::size_t offsets[N];
  __shared__ float exchanged_scores[Columns ? M * N : 1];
  for (int kb = begin; kb < end; kb += N) {
    if (threadIdx.x < N) {
      const int pos = kb + threadIdx.x;
      if (pos < klen) {
        const int page = table[batch * shape.max_blocks + pos / page_tokens];
        const auto slot = static_cast<std::size_t>(page) * page_tokens + pos % page_tokens;
        offsets[threadIdx.x] = (slot * shape.key_heads + kh) * D;
      }
    }
    __syncthreads();
    for (int i = threadIdx.x * 8; i < N * D; i += W * 32 * 8) {
      const int pos = kb + i / D;
      const int d = i % D;
      const T* ksrc = key;
      const T* vsrc = value;
      if (pos < klen) {
        const auto offset = offsets[i / D] + d;
        if (pos < prefix) {
          ksrc = key_cache + offset;
          vsrc = value_cache + offset;
        } else {
          const int np = pos - prefix;
          const auto source = ((qstart + np) * shape.key_heads + kh) * D + d;
          ksrc = key + source;
          vsrc = value + source;
          // Unique writer, even for noncausal attention. All readers of the
          // newly appended region use the input, never a concurrent cache write.
          if (np * group / M == qt) {
            *reinterpret_cast<uint4*>(key_cache + offset) = *reinterpret_cast<const uint4*>(ksrc);
            *reinterpret_cast<uint4*>(value_cache + offset) = *reinterpret_cast<const uint4*>(vsrc);
          }
        }
      }
      attn_copy16(sk + (i / D) * DS + d, ksrc, pos < klen);
      attn_copy16(sv + (i / D) * DS + d, vsrc, pos < klen);
    }
    attn_wait_copy();
    __syncthreads();
    float scores[N / 8][4] = {};
    if constexpr (Columns) {
      constexpr int Fragments = N / (8 * Cols);
      float fragment[Fragments][4] = {};
#pragma unroll
      for (int dk = 0; dk < D; dk += 16) {
        unsigned a[4], b[2];
        attn_ld_a(sq + qwarp * 16 * DS + dk, DS, a);
#pragma unroll
        for (int n = 0; n < Fragments; ++n) {
          attn_ld_b<false>(sk + (colwarp * N / Cols + n * 8) * DS + dk, DS, b);
          attn_mma<T>(fragment[n], a[0], a[1], a[2], a[3], b[0], b[1]);
        }
      }
#pragma unroll
      for (int n = 0; n < Fragments; ++n) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          exchanged_scores[(qwarp * 16 + r + j / 2 * 8) * N
              + colwarp * N / Cols + n * 8 + 2 * t + j % 2] = fragment[n][j];
      }
      __syncthreads();
#pragma unroll
      for (int n = 0; n < N / 8; ++n) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          scores[n][j] = exchanged_scores[(qwarp * 16 + r + j / 2 * 8) * N + n * 8 + 2 * t + j % 2];
      }
    } else {
#pragma unroll
    for (int dk = 0; dk < D; dk += 16) {
      unsigned a[4];
      attn_ld_a(sq + warp * 16 * DS + dk, DS, a);
#pragma unroll
      for (int n = 0; n < N / 8; ++n) {
        unsigned b[2];
        attn_ld_b<false>(sk + 8 * n * DS + dk, DS, b);
        attn_mma<T>(scores[n], a[0], a[1], a[2], a[3], b[0], b[1]);
      }
    }
    }
    float alpha[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int qr = (qbase + qwarp * 16 + r + 8 * rr) / group;
      float m = -CUDART_INF_F;
#pragma unroll
      for (int n = 0; n < N / 8; ++n) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int kp = kb + 8 * n + 2 * t + j;
          float& s = scores[n][2 * rr + j];
          s = qr < qlen && kp < end && (!causal || kp <= prefix + qr)
              ? s * scale : -CUDART_INF_F;
          m = fmaxf(m, s);
        }
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      m = fmaxf(m, maxima[rr]);
      alpha[rr] = m == -CUDART_INF_F ? 0.f : __expf(maxima[rr] - m);
      float l = 0.f;
#pragma unroll
      for (int n = 0; n < N / 8; ++n) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float s = scores[n][2 * rr + j];
          const float p = s == -CUDART_INF_F ? 0.f : __expf(s - m);
          l += p;
          if (colwarp == 0)
            sp[(qwarp * 16 + r + 8 * rr) * NS + n * 8 + 2 * t + j]
                = detail::from_float<T>(p);
        }
      }
      l += __shfl_xor_sync(0xffffffff, l, 1);
      l += __shfl_xor_sync(0xffffffff, l, 2);
      sums[rr] = sums[rr] * alpha[rr] + l;
      maxima[rr] = m;
    }
#pragma unroll
    for (int dv = 0; dv < OD / 8; ++dv) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[dv][j] *= alpha[j / 2];
    }
    __syncthreads();
#pragma unroll
    for (int nk = 0; nk < N; nk += 16) {
      unsigned a[4];
      attn_ld_a(sp + qwarp * 16 * NS + nk, NS, a);
#pragma unroll
      for (int dv = 0; dv < OD / 8; ++dv) {
        unsigned b[2];
        attn_ld_b<true>(sv + nk * DS + colwarp * OD + dv * 8, DS, b);
        attn_mma<T>(acc[dv], a[0], a[1], a[2], a[3], b[0], b[1]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int flat = qbase + qwarp * 16 + r + 8 * rr;
    const int qr = flat / group;
    const int head = head_base + flat % group;
    if (qr < qlen) {
      const float inv = sums[rr] > 0.f ? 1.f / sums[rr] : 0.f;
      const auto row = (((qstart + qr) * shape.query_heads + head) * splits + split);
      const auto rows = shape.query_tokens * shape.query_heads * splits;
      if constexpr (SplitKV) {
        if (t == 0 && colwarp == 0) {
          partials[rows * D + row] = maxima[rr];
          partials[rows * (D + 1) + row] = sums[rr];
        }
      }
#pragma unroll
      for (int dv = 0; dv < OD / 8; ++dv) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int d = colwarp * OD + dv * 8 + 2 * t + j;
          if constexpr (SplitKV) partials[row * D + d] = acc[dv][2 * rr + j];
          else output[((qstart + qr) * shape.query_heads + head) * D + d]
              = detail::from_float<T>(acc[dv][2 * rr + j] * inv);
        }
      }
    }
  }
}

template <typename T>
__global__ void merge_attn_tensor_splits(const float* partials, T* output,
                                         AttentionShape shape, int splits) {
  const auto token_head = blockIdx.x;
  const auto begin = token_head * splits;
  const auto rows = shape.query_tokens * shape.query_heads * splits;
  const float* maxima = partials + rows * shape.head_dim;
  const float* sums = maxima + rows;
  __shared__ float weights[32], inv;
  if (threadIdx.x < 32) {
    const int s = threadIdx.x;
    const float m = s < splits ? maxima[begin + s] : -CUDART_INF_F;
    const float l = s < splits ? sums[begin + s] : 0.f;
    const float gm = detail::warp_max_reduce(m);
    const float w = l > 0.f ? __expf(m - gm) : 0.f;
    const float gl = detail::warp_sum_reduce(l * w);
    if (s < splits) weights[s] = w;
    if (!s) inv = gl > 0.f ? 1.f / gl : 0.f;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    float o = 0.f;
    for (int s = 0; s < splits; ++s)
      o += partials[(begin + s) * shape.head_dim + d] * weights[s];
    output[token_head * shape.head_dim + d] = detail::from_float<T>(o * inv);
  }
}

int tensor_splits(std::size_t batch, std::size_t heads, std::size_t group,
                   std::size_t max_q, std::size_t max_k, int tile,
                   std::size_t tokens) {
  const auto blocks = batch * heads * ((max_q * group + tile - 1) / tile);
  int splits = 1;
  while (blocks * splits * 2 <= (tile == 128 ? 512 : 256) && splits < 32
          && tokens * heads * group * splits * 2 <= 256 * 64
          && max_k / (splits * 2) >= (max_q == 1 ? 32 : tile == 128 ? 64 : 128)) splits *= 2;
  return splits;
}

template <typename T, int D, bool SplitKV, int W = 4, int N = 32, int Cols = 1>
void launch_attn_tensor(ConstTensorView query, TensorView key_cache,
                        TensorView value_cache, ConstTensorView key,
                        ConstTensorView value, ConstTensorView cu_q,
                        ConstTensorView cu_k, ConstTensorView table,
                        TensorView output, AttentionShape shape, bool causal,
                        cudaStream_t stream, float* partials, int splits) {
  constexpr int M = W / Cols * 16;
  constexpr int bytes = (M * (D + 8) + 2 * N * (D + 8)
                         + M * (N + 8)) * sizeof(T);
  static const bool configured = [] {
    check_cuda(cudaFuncSetAttribute(flash_attn_tensor_kernel<T, D, W, N, SplitKV, Cols>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  bytes), "configure tensor attention shared memory");
    return true;
  }();
  (void)configured;
  const int group = shape.query_heads / shape.key_heads;
  const dim3 grid((shape.max_seqlen_q * group + M - 1) / M * splits, shape.key_heads, shape.batch);
  // Align partitions to N so every key is visited exactly once.
  const int tokens = (shape.max_seqlen_k + splits * N - 1) / (splits * N) * N;
  flash_attn_tensor_kernel<T, D, W, N, SplitKV, Cols><<<grid, W * 32, bytes, stream>>>(
      query.data<T>(), key_cache.data<T>(), value_cache.data<T>(), key.data<T>(),
      value.data<T>(), cu_q.data<std::int32_t>(), cu_k.data<std::int32_t>(),
      table.data<std::int32_t>(), output.data<T>(), shape, causal, partials, splits, tokens);
  if constexpr (SplitKV) {
    merge_attn_tensor_splits<T><<<shape.query_tokens * shape.query_heads, 128, 0, stream>>>(
        partials, output.data<T>(), shape, splits);
  }
}

template <typename T>
__device__ __forceinline__ void attn_load8(const T* p, float (&v)[8]) {
  const uint4 bits = *reinterpret_cast<const uint4*>(p);
  const T* data = reinterpret_cast<const T*>(&bits);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = detail::to_float(data[j]);
}

template <typename T, int D, bool SplitKV>
__global__ void flash_attn_vector_decode_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_q,
    const std::int32_t* __restrict__ cu_k,
    const std::int32_t* __restrict__ table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* __restrict__ partials,
    int splits, int split_tokens) {
  constexpr int G = D == 128 ? 4 : 8;
  constexpr int Groups = 128 / G;
  constexpr int E = D / G;
  const int lane = threadIdx.x % G;
  const int group = threadIdx.x / G;
  const unsigned group_mask = ((1u << G) - 1) << ((threadIdx.x % 32) / G * G);
  const int batch = blockIdx.z;
  const int head = blockIdx.y;
  const int qs = cu_q[batch];
  const int qlen = cu_q[batch + 1] - qs;
  if (!qlen) return;
  const int kl = cu_k[batch + 1] - cu_k[batch];
  const int prefix = kl - qlen;
  const int page_tokens = shape.block_tokens;
  const int hgroup = shape.query_heads / shape.key_heads;
  const int kh = head / hgroup;
  const auto qo = (qs * shape.query_heads + head) * D;
  float q[E], acc[E] = {};
#pragma unroll
  for (int c = 0; c < E / 8; ++c) {
    float v[8];
    attn_load8(query + qo + (c * G + lane) * 8, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) q[c * 8 + j] = v[j];
  }
  const int begin = SplitKV ? blockIdx.x * split_tokens : 0;
  const int end = SplitKV ? min(kl, begin + split_tokens) : kl;
  const float scale = rsqrtf(float(D));
  float m = -CUDART_INF_F, l = 0.f;
  for (int pos = begin + group; pos < end; pos += Groups) {
    const auto page = table[batch * shape.max_blocks + pos / page_tokens];
    const auto slot = static_cast<std::size_t>(page) * page_tokens + pos % page_tokens;
    const auto offset = (slot * shape.key_heads + kh) * D;
    const bool fresh = pos >= prefix;
    const auto input_offset = ((qs + pos - prefix) * shape.key_heads + kh) * D;
    float kval[E], vval[E];
#pragma unroll
    for (int c = 0; c < E / 8; ++c) {
      const int d = (c * G + lane) * 8;
      float kv[8], vv[8];
      attn_load8((fresh ? key + input_offset : key_cache + offset) + d, kv);
      attn_load8((fresh ? value + input_offset : value_cache + offset) + d, vv);
      if (fresh && head % hgroup == 0) {
        *reinterpret_cast<uint4*>(key_cache + offset + d) =
            *reinterpret_cast<const uint4*>(key + input_offset + d);
        *reinterpret_cast<uint4*>(value_cache + offset + d) =
            *reinterpret_cast<const uint4*>(value + input_offset + d);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        kval[c * 8 + j] = kv[j];
        vval[c * 8 + j] = vv[j];
      }
    }
    float dot = 0.f;
#pragma unroll
    for (int c = 0; c < E; ++c) dot += q[c] * kval[c];
#pragma unroll
    for (int mask = G / 2; mask; mask /= 2)
      dot += __shfl_xor_sync(group_mask, dot, mask, G);
    dot *= scale;
    const float nm = fmaxf(m, dot);
    const float a = __expf(m - nm), p = __expf(dot - nm);
    l = l * a + p;
    m = nm;
#pragma unroll
    for (int c = 0; c < E; ++c) acc[c] = acc[c] * a + vval[c] * p;
  }
  __shared__ float sm[Groups], sl[Groups];
  __shared__ float so[Groups][D];
  __shared__ float global_m, global_l;
  if (lane == 0) { sm[group] = m; sl[group] = l; }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int g = threadIdx.x;
    const float mm = g < Groups ? sm[g] : -CUDART_INF_F;
    const float ll = g < Groups ? sl[g] : 0.f;
    const float gm = detail::warp_max_reduce(mm);
    const float gl = detail::warp_sum_reduce(ll > 0.f ? ll * __expf(mm - gm) : 0.f);
    if (g == 0) { global_m = gm; global_l = gl; }
  }
  __syncthreads();
  const float gm = global_m, gl = global_l;
  const float weight = l > 0.f ? __expf(m - gm) : 0.f;
#pragma unroll
  for (int c = 0; c < E; ++c)
    so[group][(c / 8 * G + lane) * 8 + c % 8] = acc[c] * weight;
  __syncthreads();
  const auto row = (batch * shape.query_heads + head) * splits + blockIdx.x;
  const auto rows = shape.batch * shape.query_heads * splits;
  for (int d = threadIdx.x; d < D; d += 128) {
    float o = 0.f;
#pragma unroll
    for (int g = 0; g < Groups; ++g) o += so[g][d];
    if constexpr (SplitKV) partials[row * D + d] = o;
    else output[qo + d] = detail::from_float<T>(gl > 0.f ? o / gl : 0.f);
  }
  if constexpr (SplitKV) {
    if (threadIdx.x == 0) {
      partials[rows * D + row] = gm;
      partials[rows * (D + 1) + row] = gl;
    }
  }
}

template <typename T, int W, int E, bool SplitKV>
__global__ void flash_attn_with_kvcache_kernel_decode(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_seqlens_q,
    const std::int32_t* __restrict__ cu_seqlens_k,
    const std::int32_t* __restrict__ block_table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* __restrict__ partials,
    int num_splits, int split_tokens) {
  constexpr int kThreads = W * 32;

  constexpr int query_local_idx = 0;
  const std::size_t query_head_idx = blockIdx.y;
  const std::size_t batch = blockIdx.z;

  const int query_start = cu_seqlens_q[batch];
  const int query_length = cu_seqlens_q[batch + 1] - query_start;
  const int key_length = cu_seqlens_k[batch + 1] - cu_seqlens_k[batch];
  const int cached_length = key_length - query_length;

  if (query_local_idx >= query_length) {
    return;
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int head_dim = shape.head_dim;
  const int query_idx = query_start + query_local_idx;
  const std::size_t group_size = shape.query_heads / shape.key_heads;
  const std::size_t key_head = query_head_idx / group_size;

  const std::size_t query_offset =
      (query_idx * shape.query_heads + query_head_idx) * head_dim;
  const int key_end =
      causal ? min(key_length,
                   static_cast<int>(cached_length + query_local_idx + 1))
             : key_length;
  const float scale = rsqrtf(head_dim);

  extern __shared__ float s_mem[];
  float* s_out = s_mem;
  float* s_m = s_out + head_dim;
  float* s_l = s_m + W;

  for (int k = threadIdx.x; k < head_dim; k += kThreads) {
    s_out[k] = 0.0f;
  }

  float q[E] = {};
  float acc[E] = {};
#pragma unroll
  for (int c = 0; c < E; ++c) {
    const int d = lane + c * 32;
    if (d < head_dim) {
      q[c] = detail::to_float(query[query_offset + d]);
    }
  }

  float m = -CUDART_INF_F;
  float l = 0.0f;

  const int key_begin = SplitKV ? blockIdx.x * split_tokens : 0;
  const int partition_end =
      SplitKV ? static_cast<int>(
                    min(static_cast<std::size_t>(key_end),
                        static_cast<std::size_t>(key_begin) + split_tokens))
              : key_end;
  for (int j = key_begin + warp; j < partition_end; j += W) {
    const auto page =
        block_table[batch * shape.max_blocks + j / shape.block_tokens];
    const auto cache_token = page * shape.block_tokens + j % shape.block_tokens;
    const auto cache_offset =
        (cache_token * shape.key_heads + key_head) * head_dim;
    float v[E] = {};
    float dot = 0.0f;

#pragma unroll
    for (int c = 0; c < E; ++c) {
      const int d = lane + c * 32;
      if (d < head_dim) {
        T kj, vj;
        if (j < cached_length) {
          kj = key_cache[cache_offset + d];
          vj = value_cache[cache_offset + d];
        } else {
          const int new_pos = j - cached_length;
          const auto input_offset =
              ((query_start + new_pos) * shape.key_heads + key_head) * head_dim
              + d;

          kj = key[input_offset];
          vj = value[input_offset];

          if (query_head_idx % group_size == 0 && new_pos == query_local_idx) {
            key_cache[cache_offset + d] = kj;
            value_cache[cache_offset + d] = vj;
          }
        }
        dot += q[c] * detail::to_float(kj);
        v[c] = detail::to_float(vj);
      }
    }

    const float score = detail::warp_sum_reduce(dot) * scale;
    const float next_m = fmaxf(m, score);
    const float alpha = __expf(m - next_m);
    const float beta = __expf(score - next_m);
    l = l * alpha + beta;
    m = next_m;
#pragma unroll
    for (int c = 0; c < E; ++c) {
      acc[c] = acc[c] * alpha + beta * v[c];
    }
  }

  if (lane == 0) {
    s_m[warp] = m;
    s_l[warp] = l;
  }
  __syncthreads();

  float g_m = -CUDART_INF_F;
  float g_l = 0.0f;
#pragma unroll
  for (int w = 0; w < W; ++w) {
    g_m = fmaxf(g_m, s_m[w]);
  }
#pragma unroll
  for (int w = 0; w < W; ++w) {
    if (s_l[w] > 0.0f) {
      g_l += s_l[w] * expf(s_m[w] - g_m);
    }
  }

  const float warp_scale = l > 0.0f ? expf(m - g_m) : 0.0f;
#pragma unroll
  for (int c = 0; c < E; ++c) {
    const int d = lane + c * 32;
    if (d < head_dim) {
      atomicAdd(&s_out[d], acc[c] * warp_scale);
    }
  }
  __syncthreads();

  if constexpr (SplitKV) {
    const auto row =
        (batch * shape.query_heads + query_head_idx) * num_splits + blockIdx.x;
    const auto rows = shape.batch * shape.query_heads * num_splits;
    for (int d = threadIdx.x; d < head_dim; d += kThreads) {
      partials[row * head_dim + d] = s_out[d];
    }
    if (threadIdx.x == 0) {
      // Empty partitions explicitly overwrite stale workspace with (0,-inf,0).
      partials[rows * head_dim + row] = g_m;
      partials[rows * (head_dim + 1) + row] = g_l;
    }
  } else {
    const float inv_l = g_l > 0.0f ? 1.0f / g_l : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += kThreads) {
      output[query_offset + d] = detail::from_float<T>(s_out[d] * inv_l);
    }
  }
}

template <typename T>
__global__ void merge_decode_splits_kernel(
    const float* __restrict__ partials,
    const std::int32_t* __restrict__ cu_seqlens_q, T* __restrict__ output,
    AttentionShape shape, int num_splits) {
  const auto head = blockIdx.x;
  const auto batch = blockIdx.y;
  const int query_start = cu_seqlens_q[batch];
  if (cu_seqlens_q[batch + 1] == query_start) {
    return;
  }
  const auto rows = shape.batch * shape.query_heads * num_splits;
  const auto row_begin = (batch * shape.query_heads + head) * num_splits;
  const float* maxima = partials + rows * shape.head_dim;
  const float* sums = maxima + rows;
  __shared__ float weights[kMaxDecodeSplits];
  __shared__ float inv_sum;

  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const float m =
        lane < num_splits ? maxima[row_begin + lane] : -CUDART_INF_F;
    const float l = lane < num_splits ? sums[row_begin + lane] : 0.0f;
    const float global_m = detail::warp_max_reduce(m);
    const float weight = l > 0.0f ? __expf(m - global_m) : 0.0f;
    const float global_l = detail::warp_sum_reduce(l * weight);
    if (lane < num_splits) {
      weights[lane] = weight;
    }
    if (lane == 0) {
      inv_sum = global_l > 0.0f ? 1.0f / global_l : 0.0f;
    }
  }
  __syncthreads();

  const auto offset = (query_start * shape.query_heads + head) * shape.head_dim;
  for (int d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      acc += partials[(row_begin + s) * shape.head_dim + d] * weights[s];
    }
    output[offset + d] = detail::from_float<T>(acc * inv_sum);
  }
}

template <typename T, int D>
void launch_attn_vector_decode(ConstTensorView query, TensorView key_cache,
                        TensorView value_cache, ConstTensorView key,
                        ConstTensorView value, ConstTensorView cu_q,
                        ConstTensorView cu_k, ConstTensorView table,
                        TensorView output, AttentionShape shape, bool causal,
                        cudaStream_t stream, float* partials, int splits) {
  const dim3 grid(splits, shape.query_heads, shape.batch);
  if (splits > 1) {
    const int tokens = (shape.max_seqlen_k + splits - 1) / splits;
    flash_attn_vector_decode_kernel<T, D, true><<<grid, 128, 0, stream>>>(
        query.data<T>(), key_cache.data<T>(), value_cache.data<T>(), key.data<T>(),
        value.data<T>(), cu_q.data<std::int32_t>(), cu_k.data<std::int32_t>(),
        table.data<std::int32_t>(), output.data<T>(), shape, causal, partials, splits, tokens);
    const dim3 merge_grid(shape.query_heads, shape.batch);
    merge_decode_splits_kernel<T><<<merge_grid, 128, 0, stream>>>(
        partials, cu_q.data<std::int32_t>(), output.data<T>(), shape, splits);
  } else {
    flash_attn_vector_decode_kernel<T, D, false><<<grid, 128, 0, stream>>>(
        query.data<T>(), key_cache.data<T>(), value_cache.data<T>(), key.data<T>(),
        value.data<T>(), cu_q.data<std::int32_t>(), cu_k.data<std::int32_t>(),
        table.data<std::int32_t>(), output.data<T>(), shape, causal, nullptr, 1, 0);
  }
}

template <typename T>
__global__ void flash_attn_with_kvcache_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_seqlens_q,
    const std::int32_t* __restrict__ cu_seqlens_k,
    const std::int32_t* __restrict__ block_table, T* __restrict__ output,
    AttentionShape shape, bool causal) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int query_begin = blockIdx.x * kQueryTile;
  const std::size_t query_head = blockIdx.y;
  const std::size_t batch = blockIdx.z;

  const int query_start = cu_seqlens_q[batch];
  const int query_length = cu_seqlens_q[batch + 1] - query_start;
  const int key_length = cu_seqlens_k[batch + 1] - cu_seqlens_k[batch];
  const int cached_length = key_length - query_length;

  if (query_begin >= query_length) {
    return;
  }

  const int query_pos = query_begin + warp;
  const int head_dim = shape.head_dim;
  const auto group_size = shape.query_heads / shape.key_heads;
  const auto key_head = query_head / group_size;
  const auto query_offset =
      ((query_start + query_pos) * shape.query_heads + query_head) * head_dim;
  const float scale = rsqrtf(head_dim);

  float q[kHeadElements] = {};
  float acc[kHeadElements] = {};

#pragma unroll
  for (int i = 0; i < kHeadElements; ++i) {
    const int d = lane + i * 32;
    if (query_pos < query_length && d < head_dim) {
      q[i] = detail::to_float(query[query_offset + d]);
    }
  }

  extern __shared__ float shared[];
  float* tile_k = shared;
  float* tile_v = shared + kKeyTile * head_dim;

  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;

  int key_end = key_length;
  int row_end = key_length;
  if (causal) {
    key_end =
        min(key_length, key_length - query_length + query_begin + kQueryTile);
    row_end = min(key_length, key_length - query_length + query_pos + 1);
  }

  for (int key_begin = 0; key_begin < key_end; key_begin += kKeyTile) {
    const int tile_size = min(kKeyTile, key_end - key_begin);

    for (int i = threadIdx.x; i < tile_size * head_dim; i += kThreads) {
      const int key_pos = key_begin + i / head_dim;
      const int d = i % head_dim;
      const auto page =
          block_table[batch * shape.max_blocks + key_pos / shape.block_tokens];
      const auto cache_token =
          page * shape.block_tokens + key_pos % shape.block_tokens;
      const auto offset =
          (cache_token * shape.key_heads + key_head) * head_dim + d;

      T k;
      T v;
      if (key_pos < cached_length) {
        k = key_cache[offset];
        v = value_cache[offset];
      } else {
        const int new_pos = key_pos - cached_length;
        const auto input_offset =
            ((query_start + new_pos) * shape.key_heads + key_head) * head_dim
            + d;
        k = key[input_offset];
        v = value[input_offset];

        if (query_head % group_size == 0
            && new_pos / kQueryTile == blockIdx.x) {
          key_cache[offset] = k;
          value_cache[offset] = v;
        }
      }

      tile_k[i] = detail::to_float(k);
      tile_v[i] = detail::to_float(v);
    }
    __syncthreads();

    if (query_pos < query_length && key_begin < row_end) {
      const int key_count = min(tile_size, row_end - key_begin);
      float score = -CUDART_INF_F;

      for (int j = 0; j < key_count; ++j) {
        float dot = 0.0f;

#pragma unroll
        for (int i = 0; i < kHeadElements; ++i) {
          const int d = lane + i * 32;
          if (d < head_dim) {
            dot += q[i] * tile_k[j * head_dim + d];
          }
        }

        dot = detail::warp_sum_reduce(dot);
        if (lane == j) {
          score = dot * scale;
        }
      }

      const float tile_max = detail::warp_max_reduce(score);
      const float next_max = fmaxf(row_max, tile_max);
      const float rescale = expf(row_max - next_max);
      const float probability = expf(score - next_max);
      const float tile_sum = detail::warp_sum_reduce(probability);

      row_sum = row_sum * rescale + tile_sum;
      row_max = next_max;

#pragma unroll
      for (int i = 0; i < kHeadElements; ++i) {
        acc[i] *= rescale;
      }

      for (int j = 0; j < key_count; ++j) {
        const float weight = __shfl_sync(0xffffffffu, probability, j);

#pragma unroll
        for (int i = 0; i < kHeadElements; ++i) {
          const int d = lane + i * 32;
          if (d < head_dim) {
            acc[i] += weight * tile_v[j * head_dim + d];
          }
        }
      }
    }
    __syncthreads();
  }

  if (query_pos < query_length) {
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

#pragma unroll
    for (int i = 0; i < kHeadElements; ++i) {
      const int d = lane + i * 32;
      if (d < head_dim) {
        output[query_offset + d] = detail::from_float<T>(acc[i] * inv_sum);
      }
    }
  }
}

} // namespace

std::size_t flash_attn_workspace_elements(std::size_t max_batch,
                                          std::size_t query_heads,
                                          std::size_t head_dim,
                                          std::size_t max_seqlen_k) {
  const auto decode = flash_attn_decode_workspace_elements(
      max_batch, query_heads, head_dim, max_seqlen_k);
  // Tensor partitions explicitly cap packed query/head/split rows at 16384,
  // independently of padding, GQA grouping, and query length.
  return max_batch > 0 && max_seqlen_k >= 256
      ? std::max(decode, std::size_t{256 * 64} * (head_dim + 2)) : decode;
}

std::size_t flash_attn_decode_workspace_elements(std::size_t max_batch,
                                                 std::size_t query_heads,
                                                 std::size_t head_dim,
                                                 std::size_t max_seqlen_k) {
  if (query_heads == 0 || head_dim == 0 || head_dim > 256
      || max_seqlen_k > INT_MAX) {
    throw std::invalid_argument("invalid decode workspace dimensions");
  }
  // The split count falls as batch grows. Reserving only for max_batch could
  // underallocate for a smaller batch; bound all potentially split batches.
  const auto limit = std::min(max_batch, kDecodeTargetBlocks / query_heads);
  std::size_t elements = 0;
  for (std::size_t batch = 1; batch <= limit; ++batch) {
    const int splits = decode_splits(batch, query_heads, max_seqlen_k);
    if (splits > 1) {
      elements =
          std::max(elements, batch * query_heads * splits * (head_dim + 2));
    }
  }
  // D=256 can use grouped tensor-core decode. Without a KV-head argument,
  // reserve its worst case: at most 256 CTAs, 16 rows per CTA, 32 splits.
  if (head_dim == 256 && max_batch > 0 && max_seqlen_k >= 256) {
    const auto rows = std::min(std::size_t{4096},
        std::min(max_batch, std::size_t{128})
        * std::min(query_heads, std::size_t{128}) * 32);
    elements = std::max(elements, rows * (head_dim + 2));
  }
  return elements;
}

void flash_attn_with_kvcache(
    ConstTensorView query, TensorView key_cache, TensorView value_cache,
    ConstTensorView key, ConstTensorView value, ConstTensorView cu_seqlens_q,
    ConstTensorView cu_seqlens_k, std::size_t max_seqlen_q,
    std::size_t max_seqlen_k, ConstTensorView block_table, TensorView output,
    bool causal, cudaStream_t stream, TensorView workspace) {
  const auto shape =
      attn_shape(query, key_cache, value_cache, key, value, cu_seqlens_q,
                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table, output);

  if (shape.query_tokens == 0) {
    return;
  }

  const int splits =
      max_seqlen_q == 1 && workspace.data() != nullptr
          ? decode_splits(shape.batch, shape.query_heads, max_seqlen_k)
          : 1;
  if (splits > 1
      && (workspace.dtype() != DataType::Float32
          || !same_device(query, workspace) || !workspace.is_contiguous()
          || workspace.numel() < shape.batch * shape.query_heads * splits
                                     * (shape.head_dim + 2))) {
    throw std::invalid_argument(
        "Split-KV requires a sufficiently large contiguous FP32 workspace "
        "on the query device");
  }

  detail::dispatch_float(query.dtype(), [&]<typename T>() {
    if constexpr (!std::is_same_v<T, float>) {
      if ((shape.head_dim == 128 || shape.head_dim == 256) && tensor_attention_supported()) {
        if (max_seqlen_q == 1 && shape.head_dim == 256) {
          const int ts = workspace.data() != nullptr && max_seqlen_k >= 256
              ? tensor_splits(shape.batch, shape.key_heads, shape.query_heads / shape.key_heads,
                              max_seqlen_q, max_seqlen_k, 16, shape.query_tokens) : 1;
          if (ts > 1 && (workspace.dtype() != DataType::Float32
              || !same_device(query, workspace) || !workspace.is_contiguous()
              || workspace.numel() < shape.query_tokens * shape.query_heads * ts * (shape.head_dim + 2)))
            throw std::invalid_argument("tensor decode requires a sufficiently large FP32 workspace");
          if (ts > 1) {
            launch_attn_tensor<T, 256, true, 4, 32, 4>(query, key_cache, value_cache, key, value,
                cu_seqlens_q, cu_seqlens_k, block_table, output, shape, causal, stream,
                workspace.data<float>(), ts);
          } else {
            launch_attn_tensor<T, 256, false, 4, 32, 4>(query, key_cache, value_cache, key, value,
                cu_seqlens_q, cu_seqlens_k, block_table, output, shape, causal, stream,
                workspace.data<float>(), ts);
          }
        } else if (max_seqlen_q == 1) {
          launch_attn_vector_decode<T, 128>(query, key_cache, value_cache, key, value,
              cu_seqlens_q, cu_seqlens_k, block_table, output, shape, causal, stream,
              workspace.data<float>(), splits);
        } else {
          const int ts = workspace.data() != nullptr
              ? tensor_splits(shape.batch, shape.key_heads, shape.query_heads / shape.key_heads,
                              max_seqlen_q, max_seqlen_k, shape.head_dim == 256 ? 128 : 64,
                              shape.query_tokens) : 1;
          if (ts > 1 && (workspace.dtype() != DataType::Float32
              || !same_device(query, workspace) || !workspace.is_contiguous()
              || workspace.numel() < shape.query_tokens * shape.query_heads * ts * (shape.head_dim + 2)))
            throw std::invalid_argument("tensor Split-KV requires a sufficiently large FP32 workspace");
          const auto launch = [&]<int D, bool Split>() {
            launch_attn_tensor<T, D, Split, D == 256 ? 8 : 4, D == 256 ? 16 : 32>(query, key_cache, value_cache, key, value,
                cu_seqlens_q, cu_seqlens_k, block_table, output, shape, causal, stream,
                workspace.data<float>(), ts);
          };
          if (shape.head_dim == 128) {
            if (ts > 1) launch.template operator()<128, true>();
            else launch.template operator()<128, false>();
          } else {
            if (ts > 1) launch.template operator()<256, true>();
            else launch.template operator()<256, false>();
          }
        }
        check_cuda(cudaGetLastError(), "launch specialized attention");
        return;
      }
    }
    if (max_seqlen_q == 1) {
      constexpr int warps = 4;
      const dim3 grid(splits, shape.query_heads, shape.batch);
      const auto shared_bytes = (shape.head_dim + 2 * warps) * sizeof(float);
      if (splits > 1) {
        const int split_tokens = (max_seqlen_k + splits - 1) / splits;
        flash_attn_with_kvcache_kernel_decode<T, warps, kHeadElements, true>
            <<<grid, warps * 32, shared_bytes, stream>>>(
                query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
                key.data<T>(), value.data<T>(),
                cu_seqlens_q.data<std::int32_t>(),
                cu_seqlens_k.data<std::int32_t>(),
                block_table.data<std::int32_t>(), output.data<T>(), shape,
                causal, workspace.data<float>(), splits, split_tokens);
        check_cuda(cudaGetLastError(), "launch Split-KV decode partitions");
        const dim3 merge_grid(shape.query_heads, shape.batch);
        merge_decode_splits_kernel<T><<<merge_grid, 128, 0, stream>>>(
            workspace.data<float>(), cu_seqlens_q.data<std::int32_t>(),
            output.data<T>(), shape, splits);
      } else {
        flash_attn_with_kvcache_kernel_decode<T, warps, kHeadElements, false>
            <<<grid, warps * 32, shared_bytes, stream>>>(
                query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
                key.data<T>(), value.data<T>(),
                cu_seqlens_q.data<std::int32_t>(),
                cu_seqlens_k.data<std::int32_t>(),
                block_table.data<std::int32_t>(), output.data<T>(), shape,
                causal, nullptr, 1, 0);
      }
    } else {
      const dim3 grid((max_seqlen_q + kQueryTile - 1) / kQueryTile,
                      shape.query_heads, shape.batch);
      const auto shared_bytes = 2 * kKeyTile * shape.head_dim * sizeof(float);
      flash_attn_with_kvcache_kernel<T>
          <<<grid, kThreads, shared_bytes, stream>>>(
              query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
              key.data<T>(), value.data<T>(), cu_seqlens_q.data<std::int32_t>(),
              cu_seqlens_k.data<std::int32_t>(),
              block_table.data<std::int32_t>(), output.data<T>(), shape,
              causal);
    }
  });
  check_cuda(cudaGetLastError(), "launch flash_attn_with_kvcache");
}

} // namespace nanovllm::kernels::cuda
