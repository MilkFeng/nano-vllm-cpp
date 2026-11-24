#include "layers/moe/expert_executor.hh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <vector>

#include "common/cuda_utils.hh"
#include "common/thread_pool.hh"
#include "kernels/cuda/kernels.hh"
#include "layers/moe/expert.hh"
#include "layers/moe/expert_cache.hh"
#include "layers/moe/workspace.hh"

namespace nanovllm {

using detail::CpuExpertJob;

void detail::CpuExpertJob::run(void* argument) {
  const auto& job = *static_cast<const CpuExpertJob*>(argument);
  job.expert->forward_into(job.input, job.output);
}

ExpertExecutor::ExpertExecutor(const ExpertStore& store,
                               const ExpertCache& cache, ThreadPool& cpu_pool)
    : store_(store),
      cache_(cache),
      cpu_pool_(cpu_pool) {
  assert(&cache.source() == &store);
}

ExpertExecutionStats ExpertExecutor::forward_into(
    ConstTensorView gpu_input, std::span<const std::int32_t> expert_ids,
    std::size_t top_k, TensorView expert_outputs,
    MoEExecutionWorkspace& workspace, cudaStream_t stream) const {
  const auto tokens = gpu_input.shape()[0];
  const auto hidden = store_.hidden_size();
  const auto assignments = tokens * top_k;

  if (tokens == 0) {
    return {};
  }

  if (tokens == 1) {
    return forward_single_into(gpu_input, expert_ids, expert_outputs, workspace,
                               stream);
  }

  struct Group {
    std::size_t id;
    std::size_t begin;
    std::size_t rows;
    int slot;
  };

  std::vector<std::size_t> counts(store_.num_experts(), 0);

  for (const auto id : expert_ids) {
    ++counts[id];
  }

  std::vector<Group> hits, misses;
  std::vector<std::size_t> offsets(counts.size());
  std::size_t cpu_rows = 0, gpu_rows = 0, max_gpu_rows = 0;

  for (std::size_t id = 0; id < counts.size(); ++id) {
    const auto rows = counts[id];
    if (rows == 0) {
      continue;
    }

    const auto slot = cache_.slot_of(id);

    if (slot >= 0) {
      offsets[id] = gpu_rows;
      hits.push_back({id, gpu_rows, rows, slot});

      gpu_rows += rows;
      max_gpu_rows = std::max(max_gpu_rows, rows);
    } else {
      misses.push_back({id, cpu_rows, rows, slot});
      cpu_rows += rows;
    }
  }

  for (auto& group : misses) {
    group.begin += gpu_rows;
    offsets[group.id] = group.begin;
  }

  const auto views = workspace.reserve_batch(
      gpu_input, gpu_rows, cpu_rows, max_gpu_rows, store_.intermediate_size());

  for (std::size_t rank = 0; rank < assignments; ++rank) {
    views.host_order[offsets[expert_ids[rank]]++] = rank;
  }

  std::vector<CpuExpertJob> jobs;
  std::vector<ThreadPool::Task> descriptors;

  jobs.reserve(misses.size());
  descriptors.reserve(misses.size());

  ThreadPool::TaskGroup tasks;

  check_cuda(cudaMemcpyAsync(views.indices.data(), views.host_order,
                             views.indices.size_bytes(), cudaMemcpyHostToDevice,
                             stream),
             "upload grouped expert permutation");

  if (cpu_rows != 0) {
    check_cuda(
        cudaMemcpyAsync(views.host_input.data(), gpu_input.data(),
                        gpu_input.size_bytes(), cudaMemcpyDeviceToHost, stream),
        "download batch input for CPU experts");
    check_cuda(cudaStreamSynchronize(stream),
               "complete CPU expert input download");

    const auto* source = views.host_input.data<std::uint16_t>();
    auto* packed = views.host_grouped_input.data<std::uint16_t>();

    for (std::size_t row = 0; row < cpu_rows; ++row) {
      const auto token = std::size_t(views.host_order[gpu_rows + row]) / top_k;
      std::memcpy(packed + row * hidden, source + token * hidden,
                  hidden * sizeof(std::uint16_t));
    }

    for (const auto& group : misses) {
      const auto begin = group.begin - gpu_rows;

      const auto x = views.host_grouped_input.slice(0, begin, group.rows);
      const auto y = views.host_parts.slice(0, begin, group.rows);

      const auto* expert = &store_.expert_at(group.id);

      jobs.push_back({expert, x, y});
      descriptors.push_back({CpuExpertJob::run, &jobs.back()});
    }

    tasks = cpu_pool_.submit_batch(descriptors);
  }

  if (gpu_rows != 0) {
    kernels::cuda::moe_gather(gpu_input, views.indices.slice(0, 0, gpu_rows),
                              top_k, views.grouped_input, stream);

    for (const auto& group : hits) {
      cache_.expert_at_slot(group.slot)
          .forward_into(views.grouped_input.slice(0, group.begin, group.rows),
                        views.parts.slice(0, group.begin, group.rows),
                        views.expert, stream);
    }
  }

  tasks.wait();

  if (cpu_rows != 0) {
    const auto destination = views.parts.slice(0, gpu_rows, cpu_rows);
    check_cuda(cudaMemcpyAsync(destination.data(), views.host_parts.data(),
                               destination.size_bytes(), cudaMemcpyHostToDevice,
                               stream),
               "upload CPU expert groups");
  }

  kernels::cuda::moe_scatter(views.parts, views.indices, expert_outputs,
                             stream);

  return {gpu_rows, cpu_rows};
}

ExpertExecutionStats ExpertExecutor::forward_single_into(
    ConstTensorView gpu_input, std::span<const std::int32_t> expert_ids,
    TensorView expert_outputs, MoEExecutionWorkspace& workspace,
    cudaStream_t stream) const {
  struct Task {
    std::size_t rank;
    std::size_t expert_id;
    int slot;
  };

  std::vector<Task> hits, misses;
  hits.reserve(expert_ids.size());
  misses.reserve(expert_ids.size());

  for (std::size_t rank = 0; rank < expert_ids.size(); ++rank) {
    const auto id = static_cast<std::size_t>(expert_ids[rank]);
    const auto slot = cache_.slot_of(id);
    const Task task = {rank, id, slot};
    if (slot >= 0) {
      hits.emplace_back(task);
    } else {
      misses.emplace_back(task);
    }
  }

  const auto views =
      workspace.reserve_single(gpu_input, expert_outputs, hits.size(),
                               misses.size(), store_.intermediate_size());

  const bool mixed = !hits.empty() && !misses.empty();
  if (mixed) {
    std::size_t row = 0;
    for (const auto& task : hits) {
      views.host_order[row++] = task.rank;
    }
    for (const auto& task : misses) {
      views.host_order[row++] = task.rank;
    }
  }

  std::array<CpuExpertJob, 8> small_jobs;
  std::array<ThreadPool::Task, 8> small_descriptors;

  std::vector<CpuExpertJob> large_jobs;
  std::vector<ThreadPool::Task> large_descriptors;

  if (misses.size() > small_jobs.size()) {
    large_jobs.resize(misses.size());
    large_descriptors.resize(misses.size());
  }

  auto* jobs = large_jobs.empty() ? small_jobs.data() : large_jobs.data();
  auto* descriptors = large_descriptors.empty() ? small_descriptors.data()
                                                : large_descriptors.data();

  ThreadPool::TaskGroup cpu_tasks;

  if (mixed) {
    check_cuda(cudaMemcpyAsync(views.indices.data(), views.host_order,
                               views.indices.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               "upload single-token expert permutation");
  }

  for (std::size_t i = 0; i < misses.size(); ++i) {
    const auto& task = misses[i];
    const auto* expert = &store_.expert_at(task.expert_id);
    const auto row = views.host_parts.slice(0, i, 1);
    jobs[i] = {expert, views.host_input, row};
    descriptors[i] = {CpuExpertJob::run, &jobs[i]};
  }

  cpu_tasks = cpu_pool_.submit_batch({descriptors, misses.size()});

  if (views.activated.data() != nullptr) {
    kernels::cuda::FP8ExpertBatch batch;
    batch.count = hits.size();
    for (std::size_t i = 0; i < hits.size(); ++i) {
      batch.weights[i] = cache_.expert_at_slot(hits[i].slot).device_weights();
      batch.rows[i] = mixed ? i : hits[i].rank;
    }

    kernels::cuda::fp8_experts_decode(gpu_input, batch, views.activated,
                                      views.parts, stream);
  } else {
    for (std::size_t i = 0; i < hits.size(); ++i) {
      const auto& task = hits[i];
      const auto row = views.parts.slice(0, mixed ? i : task.rank, 1);
      cache_.expert_at_slot(task.slot).forward_into(gpu_input, row,
                                                    views.expert, stream);
    }
  }

  cpu_tasks.wait();

  if (!misses.empty()) {
    const auto destination =
        views.parts.slice(0, mixed ? hits.size() : 0, misses.size());
    check_cuda(cudaMemcpyAsync(destination.data(), views.host_parts.data(),
                               destination.size_bytes(), cudaMemcpyHostToDevice,
                               stream),
               "upload single-token CPU expert outputs");
  }

  if (mixed) {
    kernels::cuda::moe_scatter(views.parts, views.indices, expert_outputs,
                               stream);
  }

  return {hits.size(), misses.size()};
}

} // namespace nanovllm
