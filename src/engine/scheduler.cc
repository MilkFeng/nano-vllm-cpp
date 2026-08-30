#include "engine/scheduler.hh"

#include "engine/block_manager.hh"

namespace nanovllm {

Scheduler::Scheduler(std::size_t num_blocks, std::size_t block_size,
                     std::size_t max_num_seqs,
                     std::size_t max_num_batched_tokens,
                     std::int32_t eos_token_id)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      max_num_seqs_(max_num_seqs),
      max_num_batched_tokens_(max_num_batched_tokens),
      eos_token_id_(eos_token_id),
      block_manager_(num_blocks, block_size) {}

void Scheduler::append_sequence(std::span<const std::int32_t> token_ids,
                                const SamplingParams& sampling_params) {
  const std::size_t seq_id = sequences_.size();
  sequences_.emplace_back(spawn_sequence(seq_id, token_ids, sampling_params));
  waitings_.push_back(seq_id);
}

Scheduled Scheduler::schedule() {
  std::vector<std::size_t> scheduled_seqs;
  std::size_t num_batched_tokens = 0;

  // Prefill
  while (!waitings_.empty() && scheduled_seqs.size() < max_num_seqs_
         && num_batched_tokens < max_num_batched_tokens_) {
    const auto seq_id = waitings_.front();
    auto& seq = sequences_[seq_id];

    std::size_t num_tokens = 0;

    BlockAllocation alloc;

    if (seq.block_ids.empty()) {
      // This sequence needs to be allocated blocks
      alloc = block_manager_.predict_prefill_allocation(seq);

      if (!block_manager_.can_allocate(alloc)) {
        // Cannot allocate blocks for this sequence
        // Skip prefill, schedule decode for other sequences
        break;
      } else {
        // Can allocate blocks for this sequence
        num_tokens = seq.num_tokens() - alloc.num_reused_blocks * block_size_;
      }
    } else {
      // This sequence is already allocated blocks, but prefill is not finished
      num_tokens = seq.num_tokens() - seq.num_cached_tokens;
    }

    const auto remaining_tokens = max_num_batched_tokens_ - num_batched_tokens;

    if (num_tokens > remaining_tokens && scheduled_seqs.size()) {
      // Only allow chunked prefill for the first sequence
      // Starting with the second sequence, it must be fully prefilled at once
      break;
    }

    if (seq.block_ids.empty()) {
      block_manager_.prefill_allocate(seq, alloc);
    }

    seq.num_scheduled_tokens = std::min(num_tokens, remaining_tokens);
    num_batched_tokens += seq.num_scheduled_tokens;

    // If the sequence is fully cached, move it to running
    if (seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens()) {
      seq.status = SequenceStatus::Running;
      runnings_.push_back(seq_id);
      waitings_.pop_front();
    }

    scheduled_seqs.push_back(seq_id);
  }

  if (!scheduled_seqs.empty()) {
    return {
        .seq_ids = {scheduled_seqs.begin(), scheduled_seqs.end()},
        .prefill = true,
    };
  }

  // Decode
  while (!runnings_.empty() && scheduled_seqs.size() < max_num_seqs_) {
    const auto seq_id = runnings_.front();
    auto& seq = sequences_[seq_id];

    bool current_seq_preempted = false;

    while (!block_manager_.can_decode_allocate(seq)) {
      if (!runnings_.empty()) {
        preempt(runnings_.back());
        runnings_.pop_back();
      } else {
        current_seq_preempted = true;
        preempt(seq_id);
        break;
      }
    }

    if (!current_seq_preempted) {
      seq.num_scheduled_tokens = 1;
      block_manager_.decode_allocate(seq);
      scheduled_seqs.push_back(seq_id);
    }
  }

  for (int i = scheduled_seqs.size() - 1; i >= 0; --i) {
    runnings_.push_front(scheduled_seqs[i]);
  }

  return {
      .seq_ids = {scheduled_seqs.begin(), scheduled_seqs.end()},
      .prefill = false,
  };
}

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

      .block_ids = {},
      .block_size = block_size_,

      .sampling_params = sampling_params,
  };
}

void Scheduler::preempt(std::size_t seq_id) {
  auto& seq = sequences_[seq_id];
  seq.status = SequenceStatus::Waiting;
  block_manager_.deallocate(seq);
  waitings_.push_front(seq_id);
}
} // namespace nanovllm
