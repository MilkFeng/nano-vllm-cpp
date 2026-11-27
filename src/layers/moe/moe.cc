#include "layers/moe/moe.hh"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/cuda_utils.hh"
#include "common/pinned_buffer.hh"
#include "common/thread_pool.hh"
#include "core/expert_policy.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/moe/expert_executor.hh"
#include "layers/swiglu_mlp.hh"

namespace nanovllm {

MoE::HybridWorkspace::HybridWorkspace(const ExpertStore& s,
                                      const ExpertCache& c, ThreadPool& p,
                                      std::size_t k, DataType dtype)
    : store(s),
      cache(c),
      pool(p),
      top_k(k) {
  ids.reserve(k * sizeof(std::int32_t));
  input.reserve(s.hidden_size() * sizeof(std::uint16_t));
  parts.reserve(k * s.hidden_size() * sizeof(float));
  flags.reserve(128 + ((s.hidden_size() + 255) / 256) * sizeof(std::uint32_t));

  std::uninitialized_value_construct_n(static_cast<std::int32_t*>(ids.data()),
                                       k);
  std::uninitialized_value_construct_n(
      static_cast<std::uint16_t*>(input.data()), s.hidden_size());
  std::uninitialized_value_construct_n(static_cast<float*>(parts.data()),
                                       k * s.hidden_size());
  std::uninitialized_value_construct_n(
      static_cast<std::uint32_t*>(flags.data()),
      flags.capacity_bytes() / sizeof(std::uint32_t));

  activated = Tensor::empty({k, s.intermediate_size()}, dtype, Device::cuda());

  check_cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
             "create expert host fork");
}

MoE::HybridWorkspace::~HybridWorkspace() {
  if (ready) {
    cudaEventDestroy(ready);
  }
}

void MoE::HybridWorkspace::compute() noexcept {
  std::array<detail::CpuExpertJob, 8> jobs;
  std::array<ThreadPool::Task, 8> descriptors;
  ThreadPool::TaskGroup tasks;
  try {
    const auto* selected_ids = static_cast<const std::int32_t*>(ids.data());
    std::size_t count = 0;
    gpu_count = cpu_count = 0;
    const auto hidden = store.hidden_size();
    for (std::size_t rank = 0; rank < top_k; ++rank) {
      const auto id = selected_ids[rank];
      const bool gpu = cache.slot_of(id) >= 0;
      if (gpu) {
        ++gpu_count;
      } else {
        ++cpu_count;
      }
      if (usage) {
        ++usage->counts[id];
        if (gpu) {
          ++usage->gpu_assignments;
        } else {
          ++usage->cpu_assignments;
        }
      }
      if (gpu) {
        continue;
      }
      jobs[count] = {
          &store.expert_at(id),
          ConstTensorView(input.data(), DataType::BFloat16,
                          TensorLayout({1, hidden}), Device::cpu()),
          TensorView(static_cast<float*>(parts.data()) + rank * hidden,
                     DataType::Float32, TensorLayout({1, hidden}),
                     Device::cpu())};
      descriptors[count] = {detail::CpuExpertJob::run, &jobs[count]};
      ++count;
    }
    tasks = pool.submit_batch({descriptors.data(), count});
    tasks.wait();
  } catch (...) {
    tasks.wait_noexcept();
    error = std::current_exception();
    std::fill_n(static_cast<float*>(parts.data()), top_k * store.hidden_size(),
                0.0f);
  }
}

void MoE::HybridWorkspace::check_async_error() const {
  if (error) {
    std::rethrow_exception(error);
  }

  const auto* errors = reinterpret_cast<const std::uint32_t*>(
      static_cast<const char*>(flags.data()) + 128);
  const auto blocks = (store.hidden_size() + 255) / 256;
  for (std::size_t i = 0; i < blocks; ++i) {
    if (errors[i]) {
      throw std::runtime_error("GPU timed out waiting for CPU expert output");
    }
  }
}

void MoE::HybridWorkspace::reset() {
  error = nullptr;
  std::atomic_ref(*static_cast<std::uint32_t*>(flags.data()))
      .store(0, std::memory_order_release);
  std::atomic_ref(
      *reinterpret_cast<std::uint32_t*>(static_cast<char*>(flags.data()) + 64))
      .store(0, std::memory_order_release);
  std::memset(static_cast<char*>(flags.data()) + 128, 0,
              ((store.hidden_size() + 255) / 256) * sizeof(std::uint32_t));
}

void MoE::HybridWorkspace::abort() noexcept {
  std::atomic_ref(
      *reinterpret_cast<std::uint32_t*>(static_cast<char*>(flags.data()) + 64))
      .store(2, std::memory_order_release);
}

void MoE::HybridWorkspace::complete_cpu() {
  auto input_ready =
      std::atomic_ref(*static_cast<std::uint32_t*>(flags.data()));
  auto done = std::atomic_ref(
      *reinterpret_cast<std::uint32_t*>(static_cast<char*>(flags.data()) + 64));
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
  while (input_ready.load(std::memory_order_acquire) == 0) {
    for (int i = 0; i < 32; ++i) {
      _mm_pause();
    }
    if (std::chrono::steady_clock::now() >= until) {
      check_cuda(cudaEventSynchronize(ready),
                 "complete hybrid input publication");
      if (input_ready.load(std::memory_order_acquire) == 0) {
        throw std::runtime_error("hybrid route did not publish input");
      }
      break;
    }
  }
  compute();
  if (error) {
    std::rethrow_exception(error);
  }
  done.store(1, std::memory_order_release);
}

MoE::~MoE() = default;

void MoE::check_async_error() const {
  if (hybrid_) {
    hybrid_->check_async_error();
  }
}

void MoE::reset_hybrid() const {
  if (hybrid_) {
    hybrid_->reset();
  }
}

void MoE::abort_hybrid() const noexcept {
  if (hybrid_) {
    hybrid_->abort();
  }
}

void MoE::complete_hybrid_cpu() const {
  if (!hybrid_) {
    throw std::logic_error("hybrid experts have not been prepared");
  }
  hybrid_->complete_cpu();
}

MoEOutput MoE::forward_hybrid(ConstTensorView gpu_hidden, cudaStream_t stream,
                              ExpertUsage* usage, bool defer_cpu,
                              bool graph_warmup) const {
  if (!hybrid_) {
    hybrid_ = std::make_unique<HybridWorkspace>(
        experts_, *cache_, cpu_pool_, config_.top_k, gpu_hidden.dtype());
  }

  auto& work = *hybrid_;
  work.usage = graph_warmup ? nullptr : usage;
  work.error = nullptr;

  if (!defer_cpu || graph_warmup) {
    work.reset();
  }

  if (graph_warmup) {
    std::fill_n(static_cast<float*>(work.parts.data()),
                config_.top_k * config_.hidden_size, 0.0f);
    std::atomic_ref(*reinterpret_cast<std::uint32_t*>(
                        static_cast<char*>(work.flags.data()) + 64))
        .store(1, std::memory_order_release);
  }

  const auto function = [&]() -> std::vector<Tensor> {
    const auto device = gpu_hidden.device();

    Tensor shared, gate;
    Tensor ids = Tensor::empty({1, config_.top_k}, DataType::Int32, device);
    Tensor weights =
        Tensor::empty({1, config_.top_k}, DataType::Float32, device);
    Tensor logits = gate_.forward(gpu_hidden, stream);
    Tensor parts = Tensor::empty({config_.top_k, config_.hidden_size},
                                 DataType::Float32, device);
    Tensor output =
        Tensor::empty({1, config_.hidden_size}, DataType::BFloat16, device);

    kernels::cuda::moe_route(logits.view(), ids.view(), weights.view(), stream);

    check_cuda(cudaMemcpyAsync(work.ids.data(), ids.data(), ids.size_bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "download graph expert IDs");
    check_cuda(cudaMemcpyAsync(work.input.data(), gpu_hidden.data(),
                               gpu_hidden.size_bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "download graph expert input");

    kernels::cuda::moe_publish_decode(
        gpu_hidden, ids.view(),
        static_cast<std::uint16_t*>(work.input.device_data()),
        static_cast<std::int32_t*>(work.ids.device_data()),
        static_cast<volatile std::uint32_t*>(work.flags.device_data()), stream,
        true);
    cudaStreamCaptureStatus status;

    check_cuda(cudaStreamIsCapturing(stream, &status), "query hybrid capture");
    check_cuda(cudaEventRecordWithFlags(work.ready, stream,
                                        status == cudaStreamCaptureStatusActive
                                            ? cudaEventRecordExternal
                                            : 0),
               "record hybrid input publication");

    if (shared_expert_) {
      shared = shared_expert_->forward(gpu_hidden, stream);
      gate = shared_expert_gate_->forward(gpu_hidden, stream);
    }

    kernels::cuda::fp8_experts_decode_routed(
        gpu_hidden, ids.view(), cache_->device_routes(), work.activated.view(),
        parts.view(), stream);

    kernels::cuda::moe_combine_routed(
        parts.view(), static_cast<const float*>(work.parts.device_data()),
        ids.view(), cache_->device_routes(), weights.view(), output.view(),
        shared.view(), gate.view(), stream,
        reinterpret_cast<const volatile std::uint32_t*>(
            static_cast<char*>(work.flags.device_data()) + 64),
        reinterpret_cast<volatile std::uint32_t*>(
            static_cast<char*>(work.flags.device_data()) + 128));

    return {
        std::move(output),  std::move(logits), std::move(ids),
        std::move(weights), std::move(parts),  std::move(shared),
        std::move(gate),
    };
  };

  std::vector<Tensor> tensors;
  if (defer_cpu) {
    tensors = function();
  } else {
    try {
      tensors = decode_prefix_.run(
          stream, {gpu_hidden.data(), cache_->device_routes().data()},
          function);
      work.complete_cpu();
    } catch (...) {
      work.abort();
      cudaStreamSynchronize(stream);
      throw;
    }
  }
  return {
      std::move(tensors[0]),          std::move(tensors[1]),
      std::move(tensors[2]),          std::move(tensors[3]),
      std::move(tensors[4]),          defer_cpu ? 0 : work.gpu_count,
      defer_cpu ? 0 : work.cpu_count, std::move(tensors[5]),
      std::move(tensors[6]),
  };
}

MoE::MoE(MoEConfig config, ThreadPool& cpu_pool, Device device,
         std::optional<std::vector<std::int32_t>> gpu_expert_ids,
         MoEExecutionWorkspace* workspace)
    : config_(config),
      cpu_pool_(cpu_pool),
      gate_(config_.hidden_size, config_.num_experts, false, DataType::Float32,
            device),
      experts_(config_.hidden_size, config_.intermediate_size,
               config_.num_experts) {
  register_module("gate", gate_);
  register_module("experts", experts_);

  if (config_.shared_expert_intermediate_size != 0) {
    shared_expert_ = std::make_unique<FP8SwiGLUMLP>(
        config_.hidden_size, config_.shared_expert_intermediate_size, device);

    shared_expert_gate_ = std::make_unique<Linear>(
        config_.hidden_size, 1, false, DataType::Float32, device);

    register_module("shared_expert", *shared_expert_);
    register_module("shared_expert_gate", *shared_expert_gate_);
  }

  if (device == Device::cpu()) {
    if (gpu_expert_ids.has_value()) {
      throw std::invalid_argument(
          "CPU MoE construction has no GPU resident list");
    }
    return;
  }

  if (!gpu_expert_ids) {
    gpu_expert_ids.emplace(config_.num_experts);
    std::iota(gpu_expert_ids->begin(), gpu_expert_ids->end(), 0);
  }

  if (!workspace) {
    owned_workspace_ = std::make_unique<MoEExecutionWorkspace>();
    workspace = owned_workspace_.get();
  }
  workspace_ = workspace;

  cache_ = std::make_unique<ExpertCache>(experts_, std::move(*gpu_expert_ids));
  executor_ = std::make_unique<ExpertExecutor>(experts_, *cache_, cpu_pool_);
}

void MoE::invalidate_expert_cache() noexcept {
  decode_prefix_.clear();
  if (cache_) {
    cache_->invalidate();
  }
}

void MoE::populate_expert_cache(cudaStream_t stream) {
  if (!cache_) {
    throw std::logic_error("CPU MoE has no CUDA expert cache");
  }
  cache_->populate(stream);
}

MoEOutput MoE::forward(ConstTensorView gpu_hidden, cudaStream_t stream,
                       MoEPhaseProfile* profile, ExpertUsage* usage,
                       bool decode_graphs, bool defer_cpu,
                       bool graph_warmup) const {
  assert(cache_ && cache_->ready());

  const auto started = profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  const auto tokens = gpu_hidden.shape()[0];
  const auto device = gpu_hidden.device();

  assert(device == Device::cuda());
  assert(gpu_hidden.dtype() == DataType::BFloat16);

  const auto assignments = tokens * config_.top_k;

  if ((decode_graphs || defer_cpu) && tokens == 1 && config_.top_k <= 8
      && !profile && cache_->device_routes().numel() != 0) {
    return forward_hybrid(gpu_hidden, stream, usage, defer_cpu, graph_warmup);
  }

  workspace_->ensure_events();

  const auto host_ids = workspace_->reserve_host_ids(assignments);

  Tensor logits, shared_output, shared_gate_logits;
  Tensor ids = Tensor::empty({tokens, config_.top_k}, DataType::Int32, device);
  Tensor weights =
      Tensor::empty({tokens, config_.top_k}, DataType::Float32, device);
  Tensor expert_outputs = Tensor::empty({assignments, config_.hidden_size},
                                        DataType::Float32, device);
  Tensor output =
      Tensor::empty({tokens, config_.hidden_size}, DataType::BFloat16, device);

  ExpertExecutionStats stats;

  logits = gate_.forward(gpu_hidden, stream);
  kernels::cuda::moe_route(logits.view(), ids.view(), weights.view(), stream);
  check_cuda(cudaMemcpyAsync(host_ids.data(), ids.data(), ids.size_bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "copy MoE expert IDs to CPU");

  if (tokens == 1 && !cache_->all_experts_resident()) {
    const auto host_input =
        workspace_->reserve_host_input(tokens, config_.hidden_size);
    check_cuda(cudaMemcpyAsync(host_input.data(), gpu_hidden.data(),
                               host_input.size_bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "download decode input with MoE routing");
  }

  workspace_->record_host_ready(stream);

  const auto launch_shared = [&] {
    if (shared_expert_) {
      shared_output = shared_expert_->forward(gpu_hidden, stream);
      shared_gate_logits = shared_expert_gate_->forward(gpu_hidden, stream);
    }
  };

  if (tokens == 1) {
    launch_shared();
  }

  workspace_->wait_host_ready();
  stats = executor_->forward_into(gpu_hidden, host_ids, config_.top_k,
                                  expert_outputs.view(), *workspace_, stream);

  if (tokens != 1) {
    launch_shared();
  }

  kernels::cuda::moe_combine(expert_outputs.view(), weights.view(),
                             output.view(), shared_output.view(),
                             shared_gate_logits.view(), stream);
  workspace_->record_completion(stream);

  if (profile) {
    workspace_->synchronize();
  }

  if (usage) {
    for (const auto id : host_ids) {
      ++usage->counts[id];
    }
    usage->gpu_assignments += stats.gpu_experts;
    usage->cpu_assignments += stats.cpu_experts;
  }

  if (profile) {
    profile->seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started)
                            .count();
    ++profile->calls;
    profile->input_tokens += tokens;
    profile->gpu_assignments += stats.gpu_experts;
    profile->cpu_assignments += stats.cpu_experts;
    profile->experts.resize(config_.num_experts);

    std::vector<std::size_t> counts(config_.num_experts, 0);
    for (const auto id : host_ids) {
      ++counts[id];
    }

    for (std::size_t id = 0; id < counts.size(); ++id) {
      if (counts[id] == 0) {
        continue;
      }
      auto& expert = profile->experts[id];
      expert.tokens += counts[id];
      ++expert.calls;
      ++expert.group_sizes[counts[id]];
    }
  }

  return {
      std::move(output),
      std::move(logits),
      std::move(ids),
      std::move(weights),
      std::move(expert_outputs),
      stats.gpu_experts,
      stats.cpu_experts,
      std::move(shared_output),
      std::move(shared_gate_logits),
  };
}

std::size_t MoE::replace_experts(
    std::span<const ExpertCacheReplacement> replacements,
    cudaStream_t stream) const {
  if (!cache_) {
    throw std::logic_error("CPU MoE has no CUDA expert cache");
  }
  return cache_->replace(replacements, stream);
}

const MoEConfig& MoE::config() const noexcept {
  return config_;
}

const ExpertStore& MoE::expert_store() const noexcept {
  return experts_;
}

const ExpertCache& MoE::expert_cache() const {
  if (!cache_) {
    throw std::logic_error("CPU MoE has no CUDA expert cache");
  }
  return *cache_;
}

} // namespace nanovllm
