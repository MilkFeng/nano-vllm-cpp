#pragma once

#include <deque>
#include <limits>
#include <optional>
#include <span>

#include "engine/block_manager.hh"
#include "engine/sequence.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

struct Scheduled {
  std::vector<std::size_t> seq_ids;
  bool prefill;
};

class Scheduler {
 public:
  Scheduler(std::size_t num_blocks, std::size_t block_size,
            std::size_t max_num_seqs, std::size_t max_num_batched_tokens,
            std::int32_t eos_token_id,
            std::size_t max_model_len = std::numeric_limits<std::size_t>::max(),
            bool use_recurrent_state = false);
  ~Scheduler() = default;

  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  Scheduler(Scheduler&&) = default;
  Scheduler& operator=(Scheduler&&) = default;

  std::size_t append_sequence(std::span<const std::int32_t> token_ids,
                              const SamplingParams& sampling_params);

  [[nodiscard]] Scheduled schedule();
  void postprocess(const Scheduled& scheduled,
                   std::span<const std::int32_t> token_ids);
  // Release requests and state slots. Completed KV blocks are retained only
  // when recurrent state is disabled; KV alone cannot restore linear state.
  void clear();

  [[nodiscard]] const Sequence& sequence(std::size_t seq_id) const {
    return sequences_.at(seq_id);
  }

  [[nodiscard]] bool finished() const {
    return waitings_.empty() && runnings_.empty();
  }

 private:
  [[nodiscard]] Sequence spawn_sequence(std::size_t seq_id,
                                        std::span<const std::int32_t> token_ids,
                                        const SamplingParams& sampling_params);

  void preempt(std::size_t seq_id);
  void allocate_state_slot(Sequence& sequence);
  void release_state_slot(Sequence& sequence);

 private:
  std::size_t num_blocks_;
  std::size_t block_size_;
  std::size_t max_num_seqs_;
  std::size_t max_num_batched_tokens_;
  std::int32_t eos_token_id_;
  std::size_t max_model_len_;
  bool use_recurrent_state_;
  std::deque<std::size_t> free_state_slots_;

  BlockManager block_manager_;

  std::vector<Sequence> sequences_;

  std::deque<std::size_t> waitings_;
  std::deque<std::size_t> runnings_;
  // Scheduling reserves blocks; only successful model execution may publish
  // their hashes and advance the sequences through postprocess().
  std::optional<Scheduled> pending_;
};
} // namespace nanovllm
