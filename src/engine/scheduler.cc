#include "engine/scheduler.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "engine/block_manager.hh"

namespace nanovllm {

Scheduler::Scheduler(std::size_t num_blocks, std::size_t block_size,
                     std::size_t max_num_seqs,
                     std::size_t max_num_batched_tokens,
                     std::int32_t eos_token_id, std::size_t max_model_len)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      max_num_seqs_(max_num_seqs),
      max_num_batched_tokens_(max_num_batched_tokens),
      eos_token_id_(eos_token_id),
      max_model_len_(max_model_len),
      block_manager_(num_blocks, block_size) {
  if (max_num_seqs == 0 || max_num_batched_tokens == 0 || max_model_len == 0) {
    throw std::invalid_argument("scheduler limits must be positive");
  }
}

std::size_t Scheduler::append_sequence(std::span<const std::int32_t> token_ids,
                                       const SamplingParams& sampling_params) {
  if (token_ids.empty() || token_ids.size() > max_model_len_
      || !std::isfinite(sampling_params.temperature)
      || sampling_params.temperature < 0
      || std::ranges::any_of(token_ids, [](auto token) { return token < 0; })) {
    throw std::invalid_argument(
        "invalid prompt tokens, length or sampling temperature");
  }
  const std::size_t seq_id = sequences_.size();
  sequences_.emplace_back(spawn_sequence(seq_id, token_ids, sampling_params));
  if (sampling_params.max_tokens == 0 || token_ids.size() == max_model_len_) {
    sequences_.back().status = SequenceStatus::Finished;
  } else {
    waitings_.push_back(seq_id);
  }
  return seq_id;
}

Scheduled Scheduler::schedule() {
  if (pending_) {
    throw std::logic_error(
        "postprocess or clear the current batch before scheduling");
  }
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
      if (seq.num_blocks() > num_blocks_) {
        throw std::runtime_error("sequence exceeds KV cache capacity");
      }
      // This sequence needs to be allocated blocks
      alloc = block_manager_.predict_prefill_allocation(seq);

      if (!block_manager_.can_allocate(alloc)) {
        if (runnings_.empty() && scheduled_seqs.empty()) {
          throw std::runtime_error(
              "no KV cache capacity available for prefill");
        }
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
    pending_ = Scheduled{.seq_ids = std::move(scheduled_seqs), .prefill = true};
    return *pending_;
  }

  // Decode
  while (!runnings_.empty()
         && scheduled_seqs.size()
                < std::min(max_num_seqs_, max_num_batched_tokens_)) {
    const auto seq_id = runnings_.front();
    runnings_.pop_front();
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

  for (auto it = scheduled_seqs.rbegin(); it != scheduled_seqs.rend(); ++it) {
    runnings_.push_front(*it);
  }

  // A sequence preempted itself after exhausting the final block. Its released
  // blocks can now be reused for prefill; do not leave the engine spinning.
  if (scheduled_seqs.empty() && !waitings_.empty()) return schedule();

  if (scheduled_seqs.empty()) return {.seq_ids = {}, .prefill = false};
  pending_ = Scheduled{.seq_ids = std::move(scheduled_seqs), .prefill = false};
  return *pending_;
}

void Scheduler::postprocess(const Scheduled& scheduled,
                            std::span<const std::int32_t> token_ids) {
  if (scheduled.seq_ids.size() != token_ids.size()) {
    throw std::invalid_argument("model returned the wrong number of tokens");
  }
  if (scheduled.seq_ids.empty() && !pending_) return;
  if (!pending_ || pending_->prefill != scheduled.prefill
      || pending_->seq_ids != scheduled.seq_ids) {
    throw std::invalid_argument(
        "postprocess batch does not match the scheduled batch");
  }
  // Validate the entire response before publishing hashes or changing tokens.
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    const auto& seq = sequences_.at(scheduled.seq_ids[i]);
    if (token_ids[i] < 0 || seq.num_scheduled_tokens == 0
        || seq.num_cached_tokens > seq.num_tokens()
        || seq.num_scheduled_tokens
               > seq.num_tokens() - seq.num_cached_tokens) {
      throw std::invalid_argument(
          "invalid sampled token or scheduled token range");
    }
  }
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    auto& seq = sequences_.at(scheduled.seq_ids[i]);
    block_manager_.hash_completed_blocks(seq);
    seq.num_cached_tokens += seq.num_scheduled_tokens;
    seq.num_scheduled_tokens = 0;
    if (scheduled.prefill && seq.num_cached_tokens < seq.num_tokens()) continue;
    seq.append_token(token_ids[i]);
    if ((!seq.sampling_params.ignore_eos && token_ids[i] == eos_token_id_)
        || seq.num_generated_tokens() >= seq.sampling_params.max_tokens
        || seq.num_tokens() >= max_model_len_) {
      seq.status = SequenceStatus::Finished;
      block_manager_.deallocate(seq);
      std::erase(runnings_, seq.id);
    }
  }
  pending_.reset();
}

void Scheduler::clear() {
  for (auto& seq : sequences_) {
    if (!seq.block_ids.empty()) block_manager_.deallocate(seq);
  }
  sequences_.clear();
  waitings_.clear();
  runnings_.clear();
  pending_.reset();
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
