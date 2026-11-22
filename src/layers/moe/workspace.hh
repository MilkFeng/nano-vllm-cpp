#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "common/pinned_buffer.hh"
#include "core/tensor.hh"

namespace nanovllm {

struct FP8ExpertWorkspace {
  Tensor gate;
  Tensor up;
  Tensor activated;

  FP8ExpertWorkspace() = default;
  explicit FP8ExpertWorkspace(std::size_t intermediate_size,
                              std::size_t rows = 1);

  void reserve(std::size_t rows, std::size_t intermediate_size,
               DataType dtype = DataType::Float32);
};

// Calls sharing a workspace are serialized and use the same CUDA stream.
struct MoEExecutionWorkspace {
  struct ExpertViews {
    FP8ExpertWorkspace& expert;
    std::int32_t* host_order = nullptr;
    TensorView indices = {};
    TensorView parts = {};
    TensorView grouped_input = {};
    TensorView host_input = {};
    TensorView host_grouped_input = {};
    TensorView host_parts = {};
    TensorView activated = {};
  };

  MoEExecutionWorkspace() = default;
  ~MoEExecutionWorkspace() noexcept;

  MoEExecutionWorkspace(const MoEExecutionWorkspace&) = delete;
  MoEExecutionWorkspace& operator=(const MoEExecutionWorkspace&) = delete;

  void ensure_events();
  void record_host_ready(cudaStream_t stream);
  void wait_host_ready();
  void record_completion(cudaStream_t stream);
  void synchronize();

  [[nodiscard]] std::span<std::int32_t> reserve_host_ids(
      std::size_t assignments);

  [[nodiscard]] TensorView reserve_host_input(std::size_t rows,
                                              std::size_t hidden);

  [[nodiscard]] ExpertViews reserve_batch(ConstTensorView gpu_input,
                                          std::size_t gpu_rows,
                                          std::size_t cpu_rows,
                                          std::size_t max_gpu_rows,
                                          std::size_t intermediate_size);

  [[nodiscard]] ExpertViews reserve_single(ConstTensorView gpu_input,
                                           TensorView expert_outputs,
                                           std::size_t gpu_rows,
                                           std::size_t cpu_rows,
                                           std::size_t intermediate_size);

 private:
  void reserve_grouped(ExpertViews& views, std::size_t assignments,
                       std::size_t hidden, Device device);

  PinnedHostBuffer host_ids_;
  PinnedHostBuffer host_order_;
  PinnedHostBuffer host_input_;
  PinnedHostBuffer host_parts_;

  Tensor host_grouped_input_;
  Tensor order_;
  Tensor grouped_input_;
  Tensor grouped_parts_;

  FP8ExpertWorkspace expert_;

  cudaEvent_t host_ready_ = nullptr;
  cudaEvent_t completion_ = nullptr;
  bool pending_ = false;
};

} // namespace nanovllm
