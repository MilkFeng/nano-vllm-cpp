#include "common/thread_pool.hh"

#include <immintrin.h>

#include <chrono>
#include <latch>
#include <stdexcept>
#include <utility>

namespace nanovllm {

struct ThreadPool::BatchState {
  explicit BatchState(std::span<const Task> jobs)
      : tasks(jobs.begin(), jobs.end()),
        remaining(jobs.size()) {}
  std::vector<Task> tasks;
  std::size_t next = 0; // Protected by the pool queue mutex.
  std::latch remaining;
  std::mutex error_mutex;
  std::exception_ptr error;
};

ThreadPool::TaskGroup::TaskGroup(std::shared_ptr<BatchState> state)
    : state_(std::move(state)) {}

ThreadPool::TaskGroup::~TaskGroup() {
  wait_noexcept();
}

ThreadPool::TaskGroup& ThreadPool::TaskGroup::operator=(
    TaskGroup&& other) noexcept {
  if (this != &other) {
    wait_noexcept();
    state_ = std::move(other.state_);
  }
  return *this;
}

void ThreadPool::TaskGroup::wait_noexcept() const noexcept {
  if (!state_ || state_->remaining.try_wait()) {
    return;
  }
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::microseconds(500);
  while (!state_->remaining.try_wait()) {
    for (int i = 0; i < 32; ++i) {
      _mm_pause();
    }
    if (std::chrono::steady_clock::now() >= until) {
      state_->remaining.wait();
      break;
    }
  }
}

void ThreadPool::TaskGroup::wait() const {
  wait_noexcept();
  if (state_ && state_->error) {
    std::rethrow_exception(state_->error);
  }
}

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
      throw std::logic_error("cannot submit to a stopped pool");
    }
    tasks_.push({std::move(packaged), {}});
    work_available_.store(true, std::memory_order_release);
  }
  ready_.notify_one();
  return result;
}

ThreadPool::TaskGroup ThreadPool::submit_batch(std::span<const Task> tasks) {
  if (tasks.empty()) {
    return {};
  }

  for (const auto& task : tasks) {
    if (!task.function) {
      throw std::invalid_argument("null CPU task function");
    }
  }

  auto state = std::make_shared<BatchState>(tasks);

  {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      throw std::logic_error("cannot submit to a stopped pool");
    }
    tasks_.push({{}, state});
    work_available_.store(true, std::memory_order_release);
  }

  if (tasks.size() >= (workers_.size() + 1) / 2) {
    // One OS wake operation for the common top-8 batch. Repeated notify_one
    // calls can cost more than the queue operations, especially under WSL.
    ready_.notify_all();
  } else {
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      ready_.notify_one();
    }
  }

  return TaskGroup(std::move(state));
}

void ThreadPool::run() {
  while (true) {
    std::packaged_task<void()> single;
    std::shared_ptr<BatchState> batch;
    Task task{};

    {
      std::unique_lock lock(mutex_);
      if (tasks_.empty() && !stopping_) {
        lock.unlock();
        const auto until =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
        while (!work_available_.load(std::memory_order_acquire)) {
          for (int i = 0; i < 32; ++i) {
            _mm_pause();
          }
          if (std::chrono::steady_clock::now() >= until) {
            break;
          }
        }
        lock.lock();
      }

      ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });

      if (tasks_.empty()) {
        return;
      }

      auto& item = tasks_.front();
      if (item.batch) {
        batch = item.batch;
        task = batch->tasks[batch->next++];
        if (batch->next == batch->tasks.size()) {
          tasks_.pop();
        }
      } else {
        single = std::move(item.single);
        tasks_.pop();
      }

      work_available_.store(!tasks_.empty() || stopping_,
                            std::memory_order_release);
    }

    if (!batch) {
      single();
      continue;
    }

    try {
      task.function(task.argument);
    } catch (...) {
      std::lock_guard lock(batch->error_mutex);
      if (!batch->error) {
        batch->error = std::current_exception();
      }
    }

    batch->remaining.count_down();
  }
}

void ThreadPool::stop() noexcept {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
    work_available_.store(true, std::memory_order_release);
  }
  ready_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}
} // namespace nanovllm
