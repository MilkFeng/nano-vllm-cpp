#include "layers/moe/workspace.hh"

#include <immintrin.h>

#include <chrono>

#include "common/cuda_utils.hh"

namespace nanovllm {

namespace {

void reserve_rows(Tensor& tensor, std::size_t rows, std::size_t width,
                  Device device = Device::cpu(),
                  DataType dtype = DataType::Float32) {
  if (tensor.rank() != 2 || tensor.shape()[1] != width
      || tensor.device() != device || tensor.dtype() != dtype
      || tensor.shape()[0] < rows) {
    tensor = Tensor::empty({rows, width}, dtype, device);
  }
}

TensorView reserve_host_rows(PinnedHostBuffer& buffer, std::size_t rows,
                             std::size_t width,
                             DataType dtype = DataType::Float32) {
  buffer.reserve(rows * width * data_type_size(dtype));
  return TensorView(buffer.data(), dtype, TensorLayout({rows, width}),
                    Device::cpu());
}

} // namespace

MoEExecutionWorkspace::~MoEExecutionWorkspace() noexcept {
  // In particular, an asynchronous H2D may still be reading host_parts.
  if (pending_) {
    cudaEventSynchronize(completion_);
  }
  if (completion_) {
    cudaEventDestroy(completion_);
  }
  if (host_ready_) {
    cudaEventDestroy(host_ready_);
  }
}

void MoEExecutionWorkspace::synchronize() {
  if (pending_) {
    check_cuda(cudaEventSynchronize(completion_),
               "complete MoE workspace uses");
    pending_ = false;
  }
}

std::span<std::int32_t> MoEExecutionWorkspace::reserve_host_ids(
    std::size_t assignments) {
  host_ids_.reserve(assignments * sizeof(std::int32_t));
  return {static_cast<std::int32_t*>(host_ids_.data()), assignments};
}

TensorView MoEExecutionWorkspace::reserve_host_input(std::size_t rows,
                                                     std::size_t hidden) {
  return reserve_host_rows(host_input_, rows, hidden, DataType::BFloat16);
}

void MoEExecutionWorkspace::reserve_grouped(ExpertViews& views,
                                            std::size_t assignments,
                                            std::size_t hidden, Device device) {
  host_order_.reserve(assignments * sizeof(std::int32_t));

  if (order_.numel() < assignments) {
    order_ = Tensor::empty({assignments}, DataType::Int32, device);
  }

  views.host_order = static_cast<std::int32_t*>(host_order_.data());
  views.indices = order_.view().slice(0, 0, assignments);

  reserve_rows(grouped_parts_, assignments, hidden, device);
  views.parts = grouped_parts_.view().slice(0, 0, assignments);
}

MoEExecutionWorkspace::ExpertViews MoEExecutionWorkspace::reserve_batch(
    ConstTensorView gpu_input, std::size_t gpu_rows, std::size_t cpu_rows,
    std::size_t max_gpu_rows, std::size_t intermediate_size) {
  const auto hidden = gpu_input.shape()[1];
  const auto device = gpu_input.device();

  ExpertViews views{expert_};

  reserve_grouped(views, gpu_rows + cpu_rows, hidden, device);

  if (gpu_rows != 0) {
    expert_.reserve(max_gpu_rows, intermediate_size, gpu_input.dtype());

    reserve_rows(grouped_input_, gpu_rows, hidden, device, gpu_input.dtype());
    views.grouped_input = grouped_input_.view().slice(0, 0, gpu_rows);
  }

  if (cpu_rows != 0) {
    views.host_input = reserve_host_input(gpu_input.shape()[0], hidden);
    views.host_parts = reserve_host_rows(host_parts_, cpu_rows, hidden);

    reserve_rows(host_grouped_input_, cpu_rows, hidden, Device::cpu(),
                 DataType::BFloat16);
    views.host_grouped_input = host_grouped_input_.view().slice(0, 0, cpu_rows);
  }
  return views;
}

MoEExecutionWorkspace::ExpertViews MoEExecutionWorkspace::reserve_single(
    ConstTensorView gpu_input, TensorView expert_outputs, std::size_t gpu_rows,
    std::size_t cpu_rows, std::size_t intermediate_size) {
  const auto hidden = gpu_input.shape()[1];

  ExpertViews views{expert_};

  views.parts = expert_outputs;

  if (gpu_rows != 0 && cpu_rows != 0) {
    reserve_grouped(views, gpu_rows + cpu_rows, hidden, gpu_input.device());
  }

  if (cpu_rows != 0) {
    views.host_input = reserve_host_input(1, hidden);
    views.host_parts =
        reserve_host_rows(host_parts_, gpu_rows + cpu_rows, hidden)
            .slice(0, 0, cpu_rows);
  }

  if (gpu_rows > 8) {
    expert_.reserve(1, intermediate_size, gpu_input.dtype());
  } else if (gpu_rows != 0) {
    reserve_rows(expert_.activated, gpu_rows, intermediate_size,
                 gpu_input.device(), gpu_input.dtype());
    views.activated = expert_.activated.view().slice(0, 0, gpu_rows);
  }
  return views;
}

void MoEExecutionWorkspace::ensure_events() {
  if (!host_ready_) {
    check_cuda(cudaEventCreateWithFlags(&host_ready_, cudaEventDisableTiming),
               "create MoE host-ready event");
  }

  if (!completion_) {
    check_cuda(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
               "create MoE workspace completion event");
  }
}

void MoEExecutionWorkspace::record_host_ready(cudaStream_t stream) {
  check_cuda(cudaEventRecord(host_ready_, stream), "record MoE host downloads");
}

void MoEExecutionWorkspace::wait_host_ready() {
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
  auto status = cudaEventQuery(host_ready_);
  while (status == cudaErrorNotReady) {
    for (int i = 0; i < 32; ++i) {
      _mm_pause();
    }
    if (std::chrono::steady_clock::now() >= until) {
      status = cudaEventSynchronize(host_ready_);
      break;
    }
    status = cudaEventQuery(host_ready_);
  }
  check_cuda(status, "complete MoE host downloads");
  // The host-ready marker also follows the previous completion on this stream.
  pending_ = false;
}

void MoEExecutionWorkspace::record_completion(cudaStream_t stream) {
  check_cuda(cudaEventRecord(completion_, stream), "record MoE workspace uses");
  pending_ = true;
}

FP8ExpertWorkspace::FP8ExpertWorkspace(std::size_t intermediate_size,
                                       std::size_t rows) {
  reserve(rows, intermediate_size);
}

void FP8ExpertWorkspace::reserve(std::size_t rows,
                                 std::size_t intermediate_size,
                                 DataType dtype) {
  reserve_rows(gate, rows, intermediate_size, Device::cuda(0), dtype);
  reserve_rows(up, rows, intermediate_size, Device::cuda(0), dtype);
  reserve_rows(activated, rows, intermediate_size, Device::cuda(0), dtype);
}

} // namespace nanovllm
