#pragma once

#include <memory>
#include <vector>

#include "common/cuda_utils.hh"
#include "core/tensor.hh"
#include "core/tensor_pool.hh"

namespace nanovllm {

// Engine-only GPU segments. Returned storage is reused on the next replay;
// the engine consumes it on the same stream and synchronizes between steps.
// Dynamic CPU expert work stays outside these graphs.
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
    cudaGraph_t graph = nullptr;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "capture decode GPU segment");
    try {
      outputs_ = function();
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph) {
        cudaGraphDestroy(graph);
      }
      throw;
    }
    check_cuda(cudaStreamEndCapture(stream, &graph),
               "finish decode GPU capture");
    const auto status = cudaGraphInstantiate(&exec_, graph, 0);
    cudaGraphDestroy(graph);
    check_cuda(status, "instantiate decode GPU segment");
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

// Build an entire heterogeneous step without executing extra recurrent/KV
// updates. Eager GPU-only warmups use zero CPU rows, then restore all caches.
class WholeDecodeGraph final {
 public:
  [[nodiscard]] bool ready() const noexcept { return graph_.ready(); }

  template <typename Function, typename Reset, typename Pump, typename Abort>
  std::vector<Tensor> run(cudaStream_t stream, std::vector<const void*> key,
                          std::span<const TensorView> states,
                          Function&& function, Reset&& reset, Pump&& pump,
                          Abort&& abort) {
    std::vector<Tensor> saved;
    if (!graph_.matches(stream, key)) {
      for (const auto& state : states) {
        auto copy = Tensor::empty({state.shape().begin(), state.shape().end()},
                                  state.dtype(), state.device());
        check_cuda(
            cudaMemcpyAsync(copy.data(), state.data(), state.size_bytes(),
                            cudaMemcpyDeviceToDevice, stream),
            "snapshot decode cache for graph warmup");
        saved.push_back(std::move(copy));
      }
    }
    try {
      for (;;) {
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
        check_cuda(cudaStreamSynchronize(stream),
                   "complete whole decode warmup");
        result.clear();
        for (std::size_t i = 0; i < states.size(); ++i) {
          check_cuda(cudaMemcpyAsync(states[i].data(), saved[i].data(),
                                     states[i].size_bytes(),
                                     cudaMemcpyDeviceToDevice, stream),
                     "restore decode cache after graph warmup");
        }
        check_cuda(cudaStreamSynchronize(stream),
                   "complete decode cache restore");
      }
    } catch (...) {
      abort();
      cudaStreamSynchronize(stream);
      throw;
    }
  }

 private:
  DecodeGraph graph_;
};
} // namespace nanovllm
