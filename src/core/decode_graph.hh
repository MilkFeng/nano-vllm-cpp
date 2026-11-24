#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/tensor.hh"
#include "core/tensor_pool.hh"

namespace nanovllm {

namespace {
void copy_states(std::span<void*> destinations, std::span<void*> sources,
                 std::span<std::size_t> sizes, cudaStream_t stream,
                 std::string_view operation) {
  if (sizes.empty()) {
    return;
  }

  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  std::size_t first = 0;
#if CUDART_VERSION >= 13000
  const auto status =
      cudaMemcpyBatchAsync(destinations.data(), sources.data(), sizes.data(),
                           sizes.size(), &attributes, &first, 1, stream);
#else
  std::size_t fail_index = 0;
  const auto status = cudaMemcpyBatchAsync(
      destinations.data(), sources.data(), sizes.data(), sizes.size(),
      &attributes, &first, 1, &fail_index, stream);
#endif
  check_cuda(status, operation);
}
} // namespace

class DecodeGraph final {
 public:
  [[nodiscard]] bool matches(cudaStream_t stream,
                             const std::vector<const void*>& key) const {
    return exec_ && stream_ == stream && key_ == key;
  }

  [[nodiscard]] bool ready() const noexcept { return exec_ != nullptr; }

  ~DecodeGraph() {
    clear();
    if (completed_) {
      cudaEventDestroy(completed_);
    }
  }

  void clear() noexcept {
    if (completed_) {
      cudaEventSynchronize(completed_);
    }
    if (exec_) {
      cudaGraphExecDestroy(exec_);
    }
    exec_ = nullptr;
    outputs_.clear();
    key_.clear();
    warmed_ = false;
  }

  template <typename Function>
  std::vector<Tensor> run(cudaStream_t stream, std::vector<const void*> key,
                          Function&& function) {
    if (stream_ != stream || key_ != key) {
      clear();
      if (stream_ != stream) {
        pool_.reset();
      }
      stream_ = stream;
      key_ = std::move(key);
    }

    if (!pool_) {
      pool_ = std::make_unique<TensorPool>(Device::cuda(0), stream);
    }

    if (!completed_) {
      check_cuda(cudaEventCreateWithFlags(&completed_, cudaEventDisableTiming),
                 "create decode graph completion event");
    }

    if (exec_) {
      // 如果之前实例化了 CUDA Graph Exec，直接执行即可
      check_cuda(cudaGraphLaunch(exec_, stream), "replay decode GPU segment");
      check_cuda(cudaEventRecord(completed_, stream),
                 "record decode graph use");
      return outputs_;
    }

    TensorPool::Scope scope(*pool_, stream);

    if (!warmed_) {
      const auto count = pool_->allocation_count();
      auto result = function();
      warmed_ = pool_->allocation_count() == count;
      check_cuda(cudaEventRecord(completed_, stream),
                 "record decode graph warmup");
      return result;
    }

    // 捕获 CUDA Graph，此时只捕获，并不会产生输出，只会准备输出张量
    cudaGraph_t graph = nullptr;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "capture decode GPU segment");
    outputs_ = function();
    check_cuda(cudaStreamEndCapture(stream, &graph),
               "finish decode GPU capture");

    // 实例化生成可反复提交的 CUDA Graph Exec，然后销毁刚才的 CUDA Graph
    const auto status = cudaGraphInstantiate(&exec_, graph, 0);
    check_cuda(cudaGraphDestroy(graph), "destroy decode GPU graph");
    check_cuda(status, "instantiate decode GPU segment");

    // 运行一下 CUDA Graph Exec
    check_cuda(cudaGraphLaunch(exec_, stream),
               "run captured decode GPU segment");
    check_cuda(cudaEventRecord(completed_, stream),
               "record captured decode use");

    return outputs_;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t completed_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;

  std::unique_ptr<TensorPool> pool_;
  std::vector<const void*> key_;

  std::vector<Tensor> outputs_;

  bool warmed_ = false;
};

class WholeDecodeGraph final {
 public:
  [[nodiscard]] bool ready() const noexcept { return graph_.ready(); }

  template <typename Function, typename Reset, typename Pump>
  std::vector<Tensor> run(cudaStream_t stream, std::vector<const void*> key,
                          std::span<const TensorView> states,
                          Function&& function, Reset&& reset, Pump&& pump) {
    std::vector<Tensor> saved;
    std::vector<void*> state_buffers;
    std::vector<void*> saved_buffers;
    std::vector<std::size_t> sizes;

    if (!graph_.matches(stream, key)) {
      // CUDA Graph 无法复用，由于 function 会直接修改
      // states，所以这里直接先备份一份
      for (const auto& state : states) {
        const auto bytes = state.size_bytes();
        if (bytes == 0) {
          continue;
        }

        saved.push_back(
            Tensor::empty({state.shape().begin(), state.shape().end()},
                          state.dtype(), state.device()));
        state_buffers.push_back(state.data());
        saved_buffers.push_back(saved.back().data());
        sizes.push_back(bytes);
      }
      copy_states(saved_buffers, state_buffers, sizes, stream,
                  "snapshot decode cache for graph warmup");
    }

    while (true) {
      reset();

      auto result = graph_.run(stream, key, [&] {
        cudaStreamCaptureStatus status;
        check_cuda(cudaStreamIsCapturing(stream, &status),
                   "query whole decode capture");
        return function(status == cudaStreamCaptureStatusActive);
      });

      if (graph_.ready()) {
        pump();
        return result;
      }

      // 恢复备份
      result.clear();
      copy_states(state_buffers, saved_buffers, sizes, stream,
                  "restore decode cache after graph warmup");

      check_cuda(cudaStreamSynchronize(stream),
                 "complete decode cache restore");
    }
  }

 private:
  DecodeGraph graph_;
};
} // namespace nanovllm
