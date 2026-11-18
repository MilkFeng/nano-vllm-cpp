#pragma once

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "kernels/cuda/attn/attn_shape.cuh"
#include "kernels/cuda/reduce.cuh"
#include "kernels/cuda/wmma_ext.cuh"

namespace nanovllm::kernels::cuda::attn {
namespace {

// QK -> Softmax -> PV
template <typename T, int HeadDim, int Warps, int Keys, bool SplitKV, int Cols>
__global__ void attn_tensor_kernel(
    const T* __restrict__ query, T* __restrict__ key_cache,
    T* __restrict__ value_cache, const T* __restrict__ key,
    const T* __restrict__ value, const std::int32_t* __restrict__ cu_q,
    const std::int32_t* __restrict__ cu_k,
    const std::int32_t* __restrict__ table, T* __restrict__ output,
    AttentionShape shape, bool causal, float* __restrict__ partials, int splits,
    int split_tokens, float scale) {
  // 16*16 @ 16*8 -> 16*8
  using AFragment = wmma_ext::FragmentA<T>;
  using KFragment = wmma_ext::FragmentB<T, wmma_ext::Layout::ColMajor>;
  using VFragment = wmma_ext::FragmentB<T, wmma_ext::Layout::RowMajor>;
  using Accumulator = wmma_ext::Accumulator;

  constexpr int Threads = Warps << 5;
  constexpr int WarpHeadDim = HeadDim / Cols;
  constexpr int WarpGroups = Warps / Cols;
  constexpr int M = WarpGroups * 16;

  constexpr bool SplitCols = Cols > 1;

  // 避免 Bank Conflict
  constexpr int DStride = HeadDim + 8;
  constexpr int PStride = Keys + 8;

  const int warp = threadIdx.x >> 5;

  // Batch, KV Head, Q Tiles, Split（后面两维合并到 blockIdx.x 中）
  //
  // 这里按照 KV Head 划分，一个 KV Head 可以对应多个 Query Head（GQA）
  // 这里把所有需要的 Query Head 全部放到一个 N*HeadDim 的矩阵中
  // 每个矩阵行是一个 Head，一共有 Q_tokens*hgroup 行，由一个或若干个 block 计算
  //
  // 把这个矩阵纵向拆成若干个 M*HeadDim 的小矩阵
  // 其中 M 由 Cols 和 Warps 决定。单个 MMA 操作是 16*8 的小矩阵，让每个 warp
  // 分组（一个 warp 分组包括 Cols 个 warp）负责 16 行，那总共就是 Warps/Cols 个
  // warp 分组，总共就是 Warps/Cols*16 行
  const int batch = blockIdx.z;
  const int kh = blockIdx.y;
  const int q_tile = blockIdx.x / splits; // 负责哪一块 Query 行
  const int split = blockIdx.x % splits;  // 负责哪一段 KV 序列

  const int qs = cu_q[batch];                     // 当前 Q 在 query 中的起点
  const int qlen = cu_q[batch + 1] - qs;          // Q 的 token 数量
  const int klen = cu_k[batch + 1] - cu_k[batch]; // KV 总长度
  const int prefix = klen - qlen;                 // 已缓存的 token 数量

  const int page_tokens = shape.block_tokens;
  const int hgroup = shape.query_heads / shape.key_heads;
  const int qh_base = kh * hgroup; // 当前 KV head 对应的第一个 Query head 编号
  const int q_base = q_tile * M;   // 当前 Query tile 对应的矩阵中的第一行

  const int q_warp = warp / Cols;   // 当前 warp 属于哪一组
  const int col_warp = warp % Cols; // 负责哪个列分区

  if (q_base >= qlen * hgroup) {
    // 矩阵一共就是 qlen * hgroup 行
    return;
  }

  extern __shared__ __align__(16) unsigned char memory[];
  T* s_q = reinterpret_cast<T*>(memory); // M * (D + 8)
  T* s_k = s_q + M * DStride;            // Keys * (D + 8)
  T* s_v = s_k + Keys * DStride;         // Keys * (D + 8)
  T* s_p = s_v + Keys * DStride;         // M * (K + 8)

  // 把 Q 的这个矩阵全部读取到 shared memory 中
  for (int i = threadIdx.x * 8; i < M * HeadDim; i += Threads * 8) {
    const int row = i / HeadDim;
    const int d = i % HeadDim;

    const int q = (q_base + row) / hgroup;
    const int qh = qh_base + (q_base + row) % hgroup;
    const int q_offset = ((qs + q) * shape.query_heads + qh) * HeadDim + d;

    if (q < qlen) {
      __pipeline_memcpy_async(s_q + row * DStride + d, query + q_offset, 16);
    } else {
      __pipeline_memcpy_async(s_q + row * DStride + d, query, 16, 16);
    }
  }

  __pipeline_commit();
  __pipeline_wait_prior(0);

  const int visible =
      causal ? min(klen, prefix + (q_base + M - 1) / hgroup + 1) : klen;
  const int begin = SplitKV ? split * split_tokens : 0;
  const int end = SplitKV ? min(visible, begin + split_tokens) : visible;

  Accumulator acc[WarpHeadDim / 8] = {};

  wmma_ext::RowState<float> m(-CUDART_INF_F);
  wmma_ext::RowState<float> l;

  __shared__ int offsets[Keys];

  // Q: [M, HeadDim]
  // K_t: [Keys, HeadDim]
  for (int k_base = begin; k_base < end; k_base += Keys) {
    // 前 Keys 个线程进行分页地址转换，得到每个 token 在分页 KV Cache
    // 中的物理位置
    if (threadIdx.x < Keys) {
      const int pos = k_base + threadIdx.x;
      if (pos < klen) {
        const int page = table[batch * shape.max_blocks + pos / page_tokens];
        const int slot = page * page_tokens + pos % page_tokens;
        offsets[threadIdx.x] = (slot * shape.key_heads + kh) * HeadDim;
      }
    }

    __syncthreads();

    // 将相关的 KV 写入 shared memory
    for (int i = threadIdx.x * 8; i < Keys * HeadDim; i += Threads * 8) {
      const int row = i / HeadDim;
      const int pos = k_base + row;
      const int d = i % HeadDim;

      const T* k_src = key;
      const T* v_src = value;

      if (pos < klen) {
        const auto offset = offsets[row] + d;
        if (pos < prefix) {
          k_src = key_cache + offset;
          v_src = value_cache + offset;
        } else {
          const int new_pos = pos - prefix;
          const auto input_offset =
              ((qs + new_pos) * shape.key_heads + kh) * HeadDim + d;
          k_src = key + input_offset;
          v_src = value + input_offset;

          // 只有唯一的一个线程来写入 KV Cache
          if (new_pos * hgroup / M == q_tile) {
            *reinterpret_cast<uint4*>(key_cache + offset) =
                *reinterpret_cast<const uint4*>(k_src);
            *reinterpret_cast<uint4*>(value_cache + offset) =
                *reinterpret_cast<const uint4*>(v_src);
          }
        }
      }

      const int zero_bytes = pos < klen ? 0 : 16;
      __pipeline_memcpy_async(s_k + row * DStride + d, k_src, 16, zero_bytes);
      __pipeline_memcpy_async(s_v + row * DStride + d, v_src, 16, zero_bytes);
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);

    __syncthreads();

    // QK 阶段

    Accumulator scores[Keys / 8] = {};

    if constexpr (!SplitCols) {
      // 每个 warp 负责其中的 16 行
      //
      // | q0  |     | k0      |^T
      // | q1  |     | k1      |
      // | q2  |  @  | k2      |
      // | q3  |     | k3      |
      // | ... |     | ...     |
      // | q15 |     | k_Keys  |
      //
      //   16*D   @   (Keys*D)^T    ->  (16*Keys)
      //
      // Q 每次只取 16 列，即取 16*16 的小矩阵
      // K 每次取 8 行 16 列，即 8*16 的小矩阵，转置后是 16*8
      //
      //                             | B_{0,0},    ..., B_{0, D/16}   |^T
      // [A_0, A_1, ..., A_{D/16}] @ |             ...                |
      //                             | B_{K/8, 0}, ..., B_{K/8, D/16} |
      //
      //                               | B_{0,0}^T,     ..., B_{K/8, 0}^T    |
      // = [A_0, A_1, ..., A_{D/16}] @ |                ...                  |
      //                               | B_{0, D/16}^T, ..., B_{K/8, D/16}^T |
      //
      // = [P_i]，P_i=sum_k(A_k @ B_{i, k}^T)

#pragma unroll
      for (int d = 0; d < HeadDim; d += 16) {
        // 加载 A_{d/16}
        AFragment a;
        KFragment b;

        wmma_ext::load_matrix_sync(a, s_q + warp * 16 * DStride + d, DStride);
#pragma unroll
        for (int n = 0; n < Keys / 8; ++n) {
          // 加载 B_{n, d/16}
          wmma_ext::load_matrix_sync(b, s_k + 8 * n * DStride + d, DStride);

          // 计入到 P_n 中
          wmma_ext::mma_sync(scores[n], a, b, scores[n]);
        }
      }
    } else {
      // 多个 warp 一起完成同一组 query 行的计算
      // 对于同一个 16 行 Query Tile，每个 warp 完成其中的一部分列的计算

      __shared__ float shared_scores[SplitCols ? M * Keys : 1];

      constexpr int Fragments = Keys / 8 / Cols;
      Accumulator fragment[Fragments] = {};
#pragma unroll
      for (int d = 0; d < HeadDim; d += 16) {
        // 加载 A_{d/16}
        AFragment a;
        KFragment b;

        wmma_ext::load_matrix_sync(a, s_q + q_warp * 16 * DStride + d, DStride);
#pragma unroll
        for (int n = 0; n < Fragments; ++n) {
          // 只处理一部分 token，加载 B_{n', d/16}
          wmma_ext::load_matrix_sync(
              b, s_k + (col_warp * Keys / Cols + n * 8) * DStride + d, DStride);

          // 计入到 P_n 中
          wmma_ext::mma_sync(fragment[n], a, b, fragment[n]);
        }
      }

      // 汇总分数

      float* score_rows = shared_scores + q_warp * 16 * Keys;
#pragma unroll
      for (int n = 0; n < Fragments; ++n) {
        const int col = col_warp * Keys / Cols + n * 8;
        wmma_ext::store_matrix_sync(score_rows + col, fragment[n], Keys);
      }

      __syncthreads();

#pragma unroll
      for (int n = 0; n < Keys / 8; ++n) {
        wmma_ext::load_matrix_sync(scores[n], score_rows + n * 8, Keys);
      }
    }

    // 现在有了 Keys/8 个 16*8 的小矩阵 score
    // 这些矩阵横向排列就是 P 矩阵，形状是 16*Keys
    // 对 16*Keys 的每一行，都执行 Online Softmax，得到 m 和 l

    // wmma_ext::for_each_row 把 scores 当作一个逻辑上的 16 × Keys
    // 矩阵，执行回调。row 是一个分布式逻辑行视图：一行的数据仍然分布在四个 lane
    // 的寄存器里。具体执行时，每个线程会调用两次回调，分别操作它参与持有的两行
    // 这些线程协作起来覆盖全部 16 行

    wmma_ext::for_each_row(scores, [&](auto row) {
      // 当前行对应的 query token 位置
      const int q_pos = (q_base + q_warp * 16 + row.index()) / hgroup;

      // 每一行的最大值和历史指数和
      float& row_m = row.state(m);
      float& row_l = row.state(l);

      // 对当前逻辑行的每一个 score 应用 mask 并且乘上 scale
      // 如果不合法变成负无穷
      row.transform([&](int col, float s) {
        const int k_pos = k_base + col;
        const bool valid =
            q_pos < qlen && k_pos < end && (!causal || k_pos <= prefix + q_pos);
        return valid ? s * scale : -CUDART_INF_F;
      });

      // 新的最大值
      const float nm = fmaxf(row.max(), row_m);

      // 计算修正值 e^(m_old-nm)
      const float alpha = nm == -CUDART_INF_F ? 0.0f : __expf(row_m - nm);

      // e^(s-nm)
      row.transform([&](int, float s) {
        return s == -CUDART_INF_F ? 0.0f : __expf(s - nm);
      });

      // 更新最大值和历史指数和
      row_l = row_l * alpha + row.sum();
      row_m = nm;

      // 历史结果乘上修正值
      row.scale(acc, alpha);

      // 写入 P 矩阵
      if (col_warp == 0) {
        row.store(s_p + q_warp * 16 * PStride, PStride);
      }
    });

    __syncthreads();

    // P@V 矩阵乘法，给累加器加上当前 PV 的结果

#pragma unroll
    for (int k = 0; k < Keys; k += 16) {
      AFragment a;
      wmma_ext::load_matrix_sync(a, s_p + q_warp * 16 * PStride + k, PStride);
#pragma unroll
      for (int c = 0; c < WarpHeadDim / 8; ++c) {
        VFragment b;
        wmma_ext::load_matrix_sync(
            b, s_v + k * DStride + col_warp * WarpHeadDim + c * 8, DStride);
        wmma_ext::mma_sync(acc[c], a, b, acc[c]);
      }
    }

    __syncthreads();
  }

  // 保存结果
  // 如果启用了 SplitKV，那么把 m、l 和没有除 l 的中间结果保存到 partials 中
  // 如果未启用 SplitKV，那么直接把最后的结果保存到 output 中

  wmma_ext::for_each_row(acc, [&](auto row) {
    // 当前行对应的 query token 位置和当前 query head
    const int flat = q_base + q_warp * 16 + row.index();
    const int q_pos = flat / hgroup;
    const int head = qh_base + flat % hgroup;

    if (q_pos < qlen) {
      // 当前行的最大值和历史指数和
      const float row_m = row.state(m);
      const float row_l = row.state(l);

      if constexpr (SplitKV) {
        // 当前中间结果对应的行号
        // partials 的逻辑行维度是 [全局 query token][query head][split]
        const auto split_row =
            (((qs + q_pos) * shape.query_heads + head) * splits + split);

        // 中间结果总共有多少行
        const auto rows = shape.query_tokens * shape.query_heads * splits;

        // 分别保存当前分片的最大值 m 和分母 l，这个东西是放在最后的
        if (row.is_leader() && col_warp == 0) {
          partials[rows * HeadDim + split_row] = row_m;
          partials[rows * (HeadDim + 1) + split_row] = row_l;
        }

        // 保存当前分片的未归一化的结果
        row.for_each([&](int col, float value) {
          const int d = col_warp * WarpHeadDim + col;
          partials[split_row * HeadDim + d] = value;
        });
      } else {
        // 没有启用 Split-KV 时，直接输出结果

        // softmax 分母的倒数
        const float inv_sum = row_l > 0.0f ? 1.0f / row_l : 0.0f;

        // 当前 query/head 的输出向量在 output 中的起始位置
        // 输出布局是 [全局 query token][query head][D 个通道]
        const auto q_offset =
            ((qs + q_pos) * shape.query_heads + head) * HeadDim;

        // 保存
        row.for_each([&](int col, float value) {
          const int d = col_warp * WarpHeadDim + col;
          output[q_offset + d] = detail::from_float<T>(value * inv_sum);
        });
      }
    }
  });
}

template <typename T>
__global__ void merge_attn_tensor_splits_kernel(const float* partials,
                                                T* output, AttentionShape shape,
                                                int splits) {
  const int token_head = blockIdx.x;
  const int row_begin = token_head * splits;
  const int rows = shape.query_tokens * shape.query_heads * splits;
  const float* maxima = partials + rows * shape.head_dim;
  const float* sums = maxima + rows;

  __shared__ float weights[32], inv_sum;

  if (threadIdx.x < 32) {
    const int split = threadIdx.x;

    const float m = split < splits ? maxima[row_begin + split] : -CUDART_INF_F;
    const float l = split < splits ? sums[row_begin + split] : 0.f;

    const float gm = detail::warp_max_reduce(m);
    const float weight = l > 0.f ? __expf(m - gm) : 0.f;
    const float gl = detail::warp_sum_reduce(l * weight);

    if (split < splits) {
      weights[split] = weight;
    }

    if (split == 0) {
      inv_sum = gl > 0.0f ? 1.0f / gl : 0.0f;
    }
  }

  __syncthreads();

  for (int d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    float acc = 0.f;
    for (int split = 0; split < splits; ++split) {
      acc +=
          partials[(row_begin + split) * shape.head_dim + d] * weights[split];
    }
    output[token_head * shape.head_dim + d] =
        detail::from_float<T>(acc * inv_sum);
  }
}

} // namespace

template <typename T, int HeadDim, int Warps, int Keys, bool SplitKV, int Cols>
void launch_attn_tensor(ConstTensorView query, TensorView key_cache,
                        TensorView value_cache, ConstTensorView key,
                        ConstTensorView value, ConstTensorView cu_q,
                        ConstTensorView cu_k, ConstTensorView table,
                        TensorView output, AttentionShape shape, bool causal,
                        cudaStream_t stream, float* partials, int splits) {
  constexpr int block = Warps << 5;
  constexpr int M = Warps / Cols * 16;
  constexpr int bytes =
      (M * (HeadDim + 8) + 2 * Keys * (HeadDim + 8) + M * (Keys + 8))
      * sizeof(T);

  check_cuda(cudaFuncSetAttribute(
                 attn_tensor_kernel<T, HeadDim, Warps, Keys, SplitKV, Cols>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
             "configure tensor attention shared memory");

  const int hgroup = shape.query_heads / shape.key_heads;
  const dim3 grid((shape.max_seqlen_q * hgroup + M - 1) / M * splits,
                  shape.key_heads, shape.batch);

  // Align partitions to Keys so every key is visited exactly once.
  const int split_tokens =
      (shape.max_seqlen_k + splits * Keys - 1) / (splits * Keys) * Keys;
  const float scale = rsqrtf(HeadDim);

  attn_tensor_kernel<T, HeadDim, Warps, Keys, SplitKV, Cols>
      <<<grid, block, bytes, stream>>>(
          query.data<T>(), key_cache.data<T>(), value_cache.data<T>(),
          key.data<T>(), value.data<T>(), cu_q.data<std::int32_t>(),
          cu_k.data<std::int32_t>(), table.data<std::int32_t>(),
          output.data<T>(), shape, causal, partials, splits, split_tokens,
          scale);

  if constexpr (SplitKV) {
    merge_attn_tensor_splits_kernel<T>
        <<<shape.query_tokens * shape.query_heads, 128, 0, stream>>>(
            partials, output.data<T>(), shape, splits);
  }
}

} // namespace nanovllm::kernels::cuda::attn
