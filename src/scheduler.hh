#pragma once

#include <deque>
#include <span>

#include "nano-vllm/nano-vllm.hh"
#include "sequence.hh"

namespace nanovllm {

class Scheduler {
 public:
  Scheduler(std::size_t block_size);
  ~Scheduler() = default;

  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  Scheduler(Scheduler&&) = default;
  Scheduler& operator=(Scheduler&&) = default;

  void append_sequence(std::span<const std::int32_t> token_ids,
                       const SamplingParams& sampling_params);

  [[nodiscard]] std::vector<std::size_t> schedule();

  [[nodiscard]] const Sequence& sequence(std::size_t seq_id) const {
    return sequences_[seq_id];
  }

 private:
  [[nodiscard]] Sequence spawn_sequence(std::size_t seq_id,
                                        std::span<const std::int32_t> token_ids,
                                        const SamplingParams& sampling_params);

 private:
  std::size_t block_size_;

  std::vector<Sequence> sequences_;

  std::deque<std::size_t> waitings_;
  std::deque<std::size_t> runnings_;
};
} // namespace nanovllm
