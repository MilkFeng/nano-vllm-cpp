#include "scheduler.hh"

namespace nanovllm {

Scheduler::Scheduler(std::size_t block_size)
    : block_size_(block_size) {}

void Scheduler::append_sequence(std::span<const std::int32_t> token_ids,
                                const SamplingParams& sampling_params) {
  const std::size_t seq_id = sequences_.size();
  sequences_.emplace_back(spawn_sequence(seq_id, token_ids, sampling_params));
  waitings_.push_back(seq_id);
}

std::vector<std::size_t> Scheduler::schedule() {}

Sequence Scheduler::spawn_sequence(std::size_t seq_id,
                                   std::span<const std::int32_t> token_ids,
                                   const SamplingParams& sampling_params) {
  return {
      .id = seq_id,
      .status = SequenceStatus::Waiting,

      .token_ids = {token_ids.begin(), token_ids.end()},

      .num_prompt_tokens = token_ids.size(),
      .num_cached_tokens = 0,
      .num_scheduled_tokens = 0,

      .is_prefill = false,

      .block_ids = {},
      .block_size = block_size_,

      .sampling_params = sampling_params,
  };
}
} // namespace nanovllm
