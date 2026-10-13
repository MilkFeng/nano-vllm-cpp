#include "common/thread_pool.hh"

#include <stdexcept>
#include <utility>

namespace nanovllm {

ThreadPool::ThreadPool(std::size_t num_threads) {
  if (num_threads == 0) {
    throw std::invalid_argument("CPU thread count must be positive");
  }
  workers_.reserve(num_threads);
  try {
    for (std::size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { run(); });
    }
  } catch (...) {
    stop();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  stop();
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
  std::packaged_task<void()> packaged(std::move(task));
  auto result = packaged.get_future();
  {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      throw std::logic_error("cannot submit to a stopped thread pool");
    }
    tasks_.push(std::move(packaged));
  }
  ready_.notify_one();
  return result;
}

void ThreadPool::run() {
  while (true) {
    std::packaged_task<void()> task;
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
      if (tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

void ThreadPool::stop() noexcept {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  ready_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

} // namespace nanovllm
