#include "common/cpu_affinity.hh"

#include <stdexcept>
#include <system_error>
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#endif

namespace nanovllm {
#if defined(_WIN32)
namespace {
GROUP_AFFINITY cpu_mask(unsigned cpu) {
  GROUP_AFFINITY mask{};
  mask.Group = static_cast<WORD>(cpu / 64);
  mask.Mask = KAFFINITY(1) << (cpu % 64);
  return mask;
}
void check(BOOL result) {
  if (!result) throw std::system_error(GetLastError(), std::system_category(),
                                       "set CPU affinity");
}
}
std::vector<unsigned> physical_cpu_ids() {
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (bytes == 0) throw std::runtime_error("cannot discover physical CPU cores");
  std::vector<std::byte> data(bytes);
  check(GetLogicalProcessorInformationEx(RelationProcessorCore,
      reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()),
      &bytes));
  GROUP_AFFINITY allowed{};
  check(GetThreadGroupAffinity(GetCurrentThread(), &allowed));
  DWORD_PTR process_mask = 0, system_mask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
    allowed.Mask &= process_mask;
  }
  std::vector<unsigned> cpus;
  for (std::size_t offset = 0; offset < bytes;) {
    const auto* info = reinterpret_cast<
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(data.data() + offset);
    for (WORD group = 0; group < info->Processor.GroupCount; ++group) {
      const auto& mask = info->Processor.GroupMask[group];
      if (mask.Group != allowed.Group) continue;
      const auto available = mask.Mask & allowed.Mask;
      for (unsigned bit = 0; bit < 64; ++bit) {
        if ((available & (KAFFINITY(1) << bit)) != 0) {
          cpus.push_back(mask.Group * 64 + bit);
          break;
        }
      }
    }
    offset += info->Size;
  }
  if (cpus.empty()) throw std::runtime_error("no allowed physical CPU cores");
  return cpus;
}
void pin_cpu_thread(std::thread::native_handle_type thread, unsigned cpu) {
  const auto mask = cpu_mask(cpu);
  check(SetThreadGroupAffinity(thread, &mask, nullptr));
}
struct ScopedCpuAffinity::Saved { GROUP_AFFINITY mask; };
ScopedCpuAffinity::ScopedCpuAffinity(std::optional<unsigned> cpu) {
  if (!cpu) return;
  auto saved = std::make_unique<Saved>();
  const auto mask = cpu_mask(*cpu);
  check(SetThreadGroupAffinity(GetCurrentThread(), &mask, &saved->mask));
  saved_ = std::move(saved);
}
ScopedCpuAffinity::~ScopedCpuAffinity() {
  if (saved_) SetThreadGroupAffinity(GetCurrentThread(), &saved_->mask, nullptr);
}
#else
namespace {
void check(int error) {
  if (error != 0) throw std::system_error(error, std::generic_category(),
                                          "set CPU affinity");
}
cpu_set_t cpu_mask(unsigned cpu) {
  if (cpu >= CPU_SETSIZE) throw std::out_of_range("CPU exceeds CPU_SETSIZE");
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  return mask;
}
}
std::vector<unsigned> physical_cpu_ids() {
  cpu_set_t allowed;
  check(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed));
  std::map<std::pair<int, int>, unsigned> cores;
  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;
    const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                      "/topology/";
    int package = 0, core = 0;
    if (!(std::ifstream(base + "physical_package_id") >> package) ||
        !(std::ifstream(base + "core_id") >> core)) {
      throw std::runtime_error("cannot discover physical CPU topology");
    }
    cores.try_emplace({package, core}, cpu);
  }
  std::vector<unsigned> cpus;
  for (const auto& [key, cpu] : cores) cpus.push_back(cpu);
  if (cpus.empty()) throw std::runtime_error("no allowed physical CPU cores");
  return cpus;
}
void pin_cpu_thread(std::thread::native_handle_type thread, unsigned cpu) {
  const auto mask = cpu_mask(cpu);
  check(pthread_setaffinity_np(thread, sizeof(mask), &mask));
}
struct ScopedCpuAffinity::Saved { cpu_set_t mask; };
ScopedCpuAffinity::ScopedCpuAffinity(std::optional<unsigned> cpu) {
  if (!cpu) return;
  auto saved = std::make_unique<Saved>();
  check(pthread_getaffinity_np(pthread_self(), sizeof(saved->mask), &saved->mask));
  const auto mask = cpu_mask(*cpu);
  check(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask));
  saved_ = std::move(saved);
}
ScopedCpuAffinity::~ScopedCpuAffinity() {
  if (saved_) pthread_setaffinity_np(pthread_self(), sizeof(saved_->mask),
                                     &saved_->mask);
}
#endif
} // namespace nanovllm
