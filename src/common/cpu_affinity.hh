#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace nanovllm {
// One allowed logical processor per physical core, in topology order.
std::vector<unsigned> physical_cpu_ids();
void pin_cpu_thread(std::thread::native_handle_type thread, unsigned cpu);

class ScopedCpuAffinity final {
 public:
  explicit ScopedCpuAffinity(std::optional<unsigned> cpu);
  ~ScopedCpuAffinity();
  ScopedCpuAffinity(const ScopedCpuAffinity&) = delete;
  ScopedCpuAffinity& operator=(const ScopedCpuAffinity&) = delete;
 private:
  struct Saved;
  std::unique_ptr<Saved> saved_;
};
} // namespace nanovllm
