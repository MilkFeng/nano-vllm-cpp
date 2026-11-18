#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <thread>
#include <vector>

#include "common/cpu_affinity.hh"

namespace nanovllm {

class ThreadPool final {
  struct BatchState;

 public:
  explicit ThreadPool(std::size_t num_threads, bool physical_affinity = false);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  struct Task {
    void (*function)(void*);
    void* argument;
  };
  class TaskGroup final {
   public:
    TaskGroup() = default;
    ~TaskGroup();
    TaskGroup(TaskGroup&&) noexcept;
    TaskGroup& operator=(TaskGroup&&) noexcept;
    TaskGroup(const TaskGroup&) = delete;
    TaskGroup& operator=(const TaskGroup&) = delete;
    void wait() const;
    void wait_noexcept() const noexcept;

   private:
    friend class ThreadPool;
    explicit TaskGroup(std::shared_ptr<BatchState> state);
    std::shared_ptr<BatchState> state_;
  };

  [[nodiscard]] std::future<void> submit(std::function<void()> task);
  // Copies descriptors, not their arguments. Keep arguments alive until wait.
  // Destruction also waits, including during exception unwinding.
  [[nodiscard]] TaskGroup submit_batch(std::span<const Task> tasks);
  [[nodiscard]] ScopedCpuAffinity bind_caller() const;

  class BusyScope final {
   public:
    explicit BusyScope(ThreadPool& pool, bool enabled = true);
    ~BusyScope();
    BusyScope(const BusyScope&) = delete;
    BusyScope& operator=(const BusyScope&) = delete;

   private:
    ThreadPool* pool_;
  };

 private:
  struct QueueItem {
    std::packaged_task<void()> single;
    std::shared_ptr<BatchState> batch;
  };
  void run();
  void stop() noexcept;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<QueueItem> tasks_;
  std::vector<std::thread> workers_;
  std::optional<unsigned> caller_cpu_;
  std::atomic<bool> work_available_{false};
  std::atomic<unsigned> active_scopes_{0};
  bool stopping_ = false;
};
} // namespace nanovllm
