#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nanovllm {

// Workers persist across submissions. Destruction drains all queued work.
class ThreadPool final {
 public:
  explicit ThreadPool(std::size_t num_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // The future carries exceptions raised by the task.
  [[nodiscard]] std::future<void> submit(std::function<void()> task);

 private:
  void run();
  void stop() noexcept;

  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<std::packaged_task<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};

} // namespace nanovllm
