#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/block_manager.hh"
#include "engine/scheduler.hh"

namespace {

using nanovllm::BlockManager;
using nanovllm::SamplingParams;
using nanovllm::Scheduled;
using nanovllm::Scheduler;
using nanovllm::Sequence;
using nanovllm::SequenceStatus;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_throw(Function&& function, const char* message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void complete(Scheduler& scheduler, const Scheduled& batch, std::int32_t token) {
  const std::vector<std::int32_t> tokens(batch.seq_ids.size(), token);
  scheduler.postprocess(batch, tokens);
}

Sequence make_sequence(std::size_t id, std::vector<std::int32_t> tokens,
                         std::size_t block_size) {
  const auto prompt_size = tokens.size();
  return {.id = id,
          .status = SequenceStatus::Waiting,
          .token_ids = std::move(tokens),
          .num_prompt_tokens = prompt_size,
          .num_cached_tokens = 0,
          .num_scheduled_tokens = 0,
          .block_ids = {},
          .block_size = block_size,
          .sampling_params = {}};
}

void test_chunked_prefill_and_eos() {
  Scheduler scheduler(8, 2, 3, 2, 99, 32);
  const std::array<std::int32_t, 5> prompt{1, 2, 3, 4, 5};
  const auto id = scheduler.append_sequence(prompt, {.max_tokens = 3});
  for (std::size_t chunk = 0; chunk < 2; ++chunk) {
    const auto batch = scheduler.schedule();
    require(batch.prefill && batch.seq_ids == std::vector{id},
            "Long prompt must be scheduled as one chunk per batch");
    require(scheduler.sequence(id).num_scheduled_tokens == 2,
            "Chunk must respect the token budget");
    complete(scheduler, batch, 99);
    const auto& sequence = scheduler.sequence(id);
    require(sequence.num_generated_tokens() == 0 && sequence.num_tokens() == 5,
            "Intermediate prefill samples, including EOS, must be ignored");
    require(sequence.num_cached_tokens == (chunk + 1) * 2 &&
                sequence.status == SequenceStatus::Waiting,
            "Intermediate prefill must retain cached progress and waiting state");
  }
  const auto last_chunk = scheduler.schedule();
  require(last_chunk.prefill && scheduler.sequence(id).num_scheduled_tokens == 1,
          "Final prefill chunk should contain only the remaining token");
  complete(scheduler, last_chunk, 6);
  require(scheduler.sequence(id).num_generated_tokens() == 1 &&
              scheduler.sequence(id).num_cached_tokens == 5,
          "Completed prefill should produce exactly one token");
  const auto decode = scheduler.schedule();
  require(!decode.prefill && scheduler.sequence(id).num_scheduled_tokens == 1,
          "Completed prompt must transition to one-token decode");
  complete(scheduler, decode, 99);
  require(scheduler.finished() && scheduler.sequence(id).num_generated_tokens() == 2,
          "Decode EOS should finish the request");
  require(scheduler.sequence(id).status == SequenceStatus::Finished &&
              scheduler.sequence(id).block_ids.empty(),
          "Finished request must release its KV blocks");
}

void test_stop_conditions() {
  const std::array<std::int32_t, 1> prompt{1};
  Scheduler eos_ignored(4, 2, 1, 4, 9, 8);
  const auto id = eos_ignored.append_sequence(
      prompt, {.max_tokens = 2, .ignore_eos = true});
  complete(eos_ignored, eos_ignored.schedule(), 9);
  require(!eos_ignored.finished(), "ignore_eos should permit continued decoding");
  complete(eos_ignored, eos_ignored.schedule(), 9);
  require(eos_ignored.finished() && eos_ignored.sequence(id).num_generated_tokens() == 2,
          "max_tokens must stop decoding even when EOS is ignored");

  Scheduler length_limited(4, 2, 1, 4, 9, 4);
  const std::array<std::int32_t, 2> longer_prompt{1, 2};
  const auto length_id = length_limited.append_sequence(longer_prompt, {.max_tokens = 10});
  complete(length_limited, length_limited.schedule(), 3);
  complete(length_limited, length_limited.schedule(), 4);
  require(length_limited.finished() && length_limited.sequence(length_id).num_tokens() == 4,
          "max_model_len must cap prompt plus generated tokens");

  Scheduler no_work(4, 2, 2, 4, 9, 2);
  const auto zero_id = no_work.append_sequence(prompt, {.max_tokens = 0});
  const auto full_id = no_work.append_sequence(longer_prompt, {.max_tokens = 10});
  require(no_work.finished() && no_work.schedule().seq_ids.empty(),
          "Zero generation and full-length prompts must not schedule GPU work");
  require(no_work.sequence(zero_id).num_generated_tokens() == 0 &&
              no_work.sequence(full_id).num_generated_tokens() == 0,
          "Requests requiring no work must return no generated tokens");
}

void test_prefix_reuse_and_clear() {
  Scheduler scheduler(6, 2, 2, 16, 99, 32);
  const std::array<std::int32_t, 5> prompt{1, 2, 3, 4, 5};
  const auto first_id = scheduler.append_sequence(prompt, {.max_tokens = 1});
  const auto first = scheduler.schedule();
  const auto first_blocks = scheduler.sequence(first_id).block_ids;
  complete(scheduler, first, 6);
  require(scheduler.finished(), "Single-token request should finish after prefill");
  scheduler.clear();
  const auto second_id = scheduler.append_sequence(prompt, {.max_tokens = 1});
  const auto second = scheduler.schedule();
  const auto& reused = scheduler.sequence(second_id);
  require(second.prefill && reused.num_cached_tokens == 4 &&
              reused.num_scheduled_tokens == 1,
          "clear must retain completed prefix cache while recomputing the final block");
  require(reused.block_ids[0] == first_blocks[0] &&
              reused.block_ids[1] == first_blocks[1],
          "Repeated prompt must reuse the same physical prefix blocks");
  complete(scheduler, second, 6);
  require(scheduler.finished(), "Reused prompt should finish normally");

  // An abandoned scheduled batch has not produced valid K/V and must not make
  // its reserved blocks reusable, even after clear() releases the request.
  Scheduler cancelled(6, 2, 1, 16, 99, 32);
  cancelled.append_sequence(prompt, {.max_tokens = 1});
  (void)cancelled.schedule();
  cancelled.clear();
  const auto retry_id = cancelled.append_sequence(prompt, {.max_tokens = 1});
  const auto retry = cancelled.schedule();
  require(cancelled.sequence(retry_id).num_cached_tokens == 0 &&
              cancelled.sequence(retry_id).num_scheduled_tokens == prompt.size(),
          "Unexecuted batches must never publish reusable prefix hashes");
  complete(cancelled, retry, 7);
}

void test_hash_publication_and_shared_release() {
  BlockManager manager(5, 2);
  auto first = make_sequence(0, {1, 2, 3}, 2);
  auto second = make_sequence(1, {1, 2, 4}, 2);
  manager.prefill_allocate(first, manager.predict_prefill_allocation(first));
  first.num_scheduled_tokens = 3;
  require(manager.predict_prefill_allocation(second).num_reused_blocks == 0,
          "Allocation alone must not publish a hash before successful computation");
  manager.hash_completed_blocks(first);
  require(manager.predict_prefill_allocation(second).num_reused_blocks == 1,
          "Completed prefix must become reusable after hash publication");
  manager.prefill_allocate(second, manager.predict_prefill_allocation(second));
  const auto shared = first.block_ids[0];
  require(second.block_ids[0] == shared && second.num_cached_tokens == 2,
          "Live requests should share computed prefix blocks");
  manager.deallocate(first);
  auto unrelated = make_sequence(2, {8, 9, 10, 11, 12, 13}, 2);
  require(manager.can_prefill_allocate(unrelated),
          "Releasing a request should release only its unshared blocks");
  manager.prefill_allocate(unrelated, manager.predict_prefill_allocation(unrelated));
  require(std::ranges::find(unrelated.block_ids, shared) == unrelated.block_ids.end(),
          "A prefix still referenced by another request must not be recycled");
  manager.deallocate(second);
  manager.deallocate(unrelated);
  require(manager.can_allocate({0, 5, 5}), "All physical blocks should be released");

  // Two prefixes computed in one batch can have equal hashes but different
  // physical blocks. Recycling the older block must retain the newer mapping.
  BlockManager duplicates(5, 2);
  auto older = make_sequence(0, {1, 2, 3}, 2);
  auto newer = make_sequence(1, {1, 2, 4}, 2);
  duplicates.prefill_allocate(older, duplicates.predict_prefill_allocation(older));
  duplicates.prefill_allocate(newer, duplicates.predict_prefill_allocation(newer));
  older.num_scheduled_tokens = newer.num_scheduled_tokens = 3;
  duplicates.hash_completed_blocks(older);
  duplicates.hash_completed_blocks(newer);
  const auto newer_prefix = newer.block_ids[0];
  duplicates.deallocate(older);
  auto churn = make_sequence(2, {20, 21, 22, 23, 24, 25}, 2);
  duplicates.prefill_allocate(churn, duplicates.predict_prefill_allocation(churn));
  auto probe = make_sequence(3, {1, 2, 6}, 2);
  require(duplicates.predict_prefill_allocation(probe).num_reused_blocks == 1 &&
              newer.block_ids[0] == newer_prefix,
          "Recycling an older duplicate must not erase the live prefix mapping");
  duplicates.deallocate(churn);
  duplicates.deallocate(newer);
}

void test_block_size_one_and_pressure() {
  Scheduler single_tokens(8, 1, 1, 8, 99, 8);
  const std::array<std::int32_t, 1> prompt{1};
  const auto id = single_tokens.append_sequence(prompt, {.max_tokens = 3});
  complete(single_tokens, single_tokens.schedule(), 2);
  for (std::size_t step = 0; step < 2; ++step) {
    const auto batch = single_tokens.schedule();
    const auto& sequence = single_tokens.sequence(id);
    require(!batch.prefill && sequence.num_allocated_blocks() == sequence.num_tokens(),
            "Block size one must allocate one block for every decode input token");
    complete(single_tokens, batch, static_cast<std::int32_t>(3 + step));
  }
  require(single_tokens.finished(), "Block size one decode must finish");

  Scheduler pressure(3, 2, 2, 8, 99, 8);
  const std::array<std::int32_t, 2> first_prompt{1, 2};
  const std::array<std::int32_t, 2> second_prompt{10, 11};
  const auto first_id = pressure.append_sequence(first_prompt, {.max_tokens = 3});
  const auto second_id = pressure.append_sequence(second_prompt, {.max_tokens = 3});
  bool saw_preemption = false;
  bool saw_recompute = false;
  std::size_t steps = 0;
  while (!pressure.finished() && steps++ < 20) {
    const auto batch = pressure.schedule();
    require(!batch.seq_ids.empty(), "Memory pressure must not yield empty progress batches");
    for (const auto sequence_id : {first_id, second_id}) {
      const auto& sequence = pressure.sequence(sequence_id);
      saw_preemption |= sequence.status == SequenceStatus::Waiting &&
                        sequence.num_generated_tokens() != 0;
    }
    for (const auto sequence_id : batch.seq_ids) {
      saw_recompute |= batch.prefill &&
                       pressure.sequence(sequence_id).num_generated_tokens() != 0;
    }
    complete(pressure, batch, 42);
  }
  require(pressure.finished() && saw_preemption && saw_recompute,
          "Preempted requests must be recomputed and make finite progress");
  require(pressure.sequence(first_id).num_generated_tokens() == 3 &&
              pressure.sequence(second_id).num_generated_tokens() == 3,
          "Recomputation must retain generated tokens and respect max_tokens");
}

void test_shadow_cache_across_batched_chunks() {
  // These prompts share five tokens, crossing a block-size-two boundary.
  // Track each physical slot by its entire causal token prefix so a wrong
  // block, stale prefix, or write into a shared block is observable on the CPU.
  const std::vector<std::vector<std::int32_t>> prompts{
      {785, 6722, 315, 9625, 374},
      {785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374},
      {16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11}};
  const std::array<std::int32_t, 8> continuation{
      12095, 13, 576, 6722, 315, 9625, 374, 12095};
  for (const std::size_t block_size : {2, 4}) {
    for (const std::size_t token_budget : {3, 128}) {
      for (const std::size_t num_blocks : {12, 256}) {
        Scheduler scheduler(num_blocks, block_size, 4, token_budget, 99, 128);
        std::vector<std::vector<std::int32_t>> slots(num_blocks * block_size);
        bool saw_reused_prefix = false;
        bool saw_shared_prefix = false;
        for (int repeat = 0; repeat < 2; ++repeat) {
          for (const auto& prompt : prompts) {
            scheduler.append_sequence(prompt, {.max_tokens = continuation.size()});
          }
          std::size_t steps = 0;
          while (!scheduler.finished() && ++steps < 200) {
            const auto batch = scheduler.schedule();
            require(!batch.seq_ids.empty(), "Shadow cache scenario must make progress");
            std::vector<std::size_t> owners(num_blocks);
            for (std::size_t id = 0; id < prompts.size(); ++id) {
              for (const auto block : scheduler.sequence(id).block_ids) ++owners[block];
            }
            saw_shared_prefix |= std::ranges::any_of(owners, [](auto count) { return count > 1; });
            const auto check_prefix = [&](const Sequence& sequence, std::size_t end) {
              for (std::size_t pos = 0; pos < end; ++pos) {
                const auto slot = sequence.block_ids.at(pos / block_size) * block_size
                                  + pos % block_size;
                require(slots[slot].size() == pos + 1 &&
                            std::equal(slots[slot].begin(), slots[slot].end(),
                                       sequence.token_ids.begin()),
                        "Physical KV slot must match the complete causal token prefix");
              }
            };
            for (std::size_t id = 0; id < prompts.size(); ++id) {
              const auto& sequence = scheduler.sequence(id);
              check_prefix(sequence, sequence.num_cached_tokens);
              saw_reused_prefix |= batch.prefill && sequence.num_cached_tokens != 0;
            }
            std::vector<bool> written(slots.size());
            for (const auto id : batch.seq_ids) {
              const auto& sequence = scheduler.sequence(id);
              const auto end = sequence.num_cached_tokens + sequence.num_scheduled_tokens;
              for (auto pos = sequence.num_cached_tokens; pos < end; ++pos) {
                const auto block = sequence.block_ids.at(pos / block_size);
                const auto slot = block * block_size + pos % block_size;
                require(owners[block] == 1 && !written[slot],
                        "Scheduled writes must neither modify shared blocks nor alias");
                written[slot] = true;
                slots[slot].assign(sequence.token_ids.begin(),
                                     sequence.token_ids.begin() + pos + 1);
              }
            }
            std::vector<std::int32_t> samples;
            for (const auto id : batch.seq_ids) {
              const auto& sequence = scheduler.sequence(id);
              check_prefix(sequence, sequence.num_cached_tokens + sequence.num_scheduled_tokens);
              samples.push_back(continuation.at(sequence.num_generated_tokens()));
            }
            scheduler.postprocess(batch, samples);
          }
          require(scheduler.finished(), "Batched chunk and prefix cache scenario must terminate");
          scheduler.clear();
        }
        require(saw_reused_prefix, "Shadow cache scenario must exercise cached prefixes");
        if (num_blocks == 256) {
          require(saw_shared_prefix, "Ample-capacity scenario must exercise shared prefix blocks");
        }
      }
    }
  }
}

void test_invalid_inputs_and_pending_batches() {
  expect_throw<std::invalid_argument>([] { Scheduler invalid(0, 2, 1, 4, 99); },
                                      "Zero cache capacity must fail");
  expect_throw<std::invalid_argument>([] { Scheduler invalid(2, 0, 1, 4, 99); },
                                      "Zero block size must fail");
  expect_throw<std::invalid_argument>([] { Scheduler invalid(2, 2, 0, 4, 99); },
                                      "Zero sequence budget must fail");
  expect_throw<std::invalid_argument>([] { Scheduler invalid(2, 2, 1, 0, 99); },
                                      "Zero token budget must fail");
  Scheduler scheduler(4, 2, 2, 4, 99, 4);
  expect_throw<std::invalid_argument>([&] { scheduler.append_sequence({}, {}); },
                                      "Empty prompt must fail");
  const std::array<std::int32_t, 1> prompt{1};
  const std::array<std::int32_t, 1> invalid_prompt{-1};
  expect_throw<std::invalid_argument>([&] { scheduler.append_sequence(invalid_prompt, {}); },
                                      "Negative token IDs must fail");
  expect_throw<std::invalid_argument>([&] {
    scheduler.append_sequence(prompt, {.temperature = -1.0f});
  }, "Negative temperature must fail");
  expect_throw<std::invalid_argument>([&] {
    scheduler.append_sequence(prompt, {.temperature = std::numeric_limits<float>::quiet_NaN()});
  }, "NaN temperature must fail");
  const std::array<std::int32_t, 5> too_long{1, 2, 3, 4, 5};
  expect_throw<std::invalid_argument>([&] { scheduler.append_sequence(too_long, {}); },
                                      "Prompt exceeding max_model_len must fail");
  const auto id = scheduler.append_sequence(prompt, {.temperature = 0, .max_tokens = 1});
  const auto batch = scheduler.schedule();
  expect_throw<std::logic_error>([&] { (void)scheduler.schedule(); },
                                 "Scheduling over unfinished GPU work must fail");
  expect_throw<std::invalid_argument>([&] { scheduler.postprocess(batch, {}); },
                                      "Mismatched model result size must fail");
  const std::array<std::int32_t, 1> invalid_sample{-1};
  expect_throw<std::invalid_argument>([&] { scheduler.postprocess(batch, invalid_sample); },
                                      "Negative sampled token must fail");
  const Scheduled wrong_mode{batch.seq_ids, !batch.prefill};
  expect_throw<std::invalid_argument>([&] { complete(scheduler, wrong_mode, 2); },
                                      "Wrong batch mode must fail");
  require(scheduler.sequence(id).num_generated_tokens() == 0 &&
              scheduler.sequence(id).num_cached_tokens == 0,
          "Rejected model results must not advance sequence state");
  complete(scheduler, batch, 2);
  expect_throw<std::invalid_argument>([&] { complete(scheduler, batch, 2); },
                                      "The same batch must not be postprocessed twice");
  scheduler.clear();
  const auto first_id = scheduler.append_sequence(prompt, {.max_tokens = 1});
  const auto second_id = scheduler.append_sequence(prompt, {.max_tokens = 1});
  const auto two_sequences = scheduler.schedule();
  const std::array<std::int32_t, 2> partly_invalid{2, -1};
  expect_throw<std::invalid_argument>([&] {
    scheduler.postprocess(two_sequences, partly_invalid);
  }, "Every model result must be validated before advancing any request");
  require(scheduler.sequence(first_id).num_generated_tokens() == 0 &&
              scheduler.sequence(second_id).num_generated_tokens() == 0 &&
              scheduler.sequence(first_id).num_cached_tokens == 0,
          "An invalid later result must not partially advance the batch");
  complete(scheduler, two_sequences, 2);

  Scheduler too_small(1, 2, 1, 8, 99, 8);
  const std::array<std::int32_t, 3> oversized{1, 2, 3};
  too_small.append_sequence(oversized, {.max_tokens = 1});
  expect_throw<std::runtime_error>([&] { (void)too_small.schedule(); },
                                   "Oversized sequences must fail instead of spinning");
  too_small.clear();
  const std::array<std::int32_t, 2> fits_prompt{1, 2};
  too_small.append_sequence(fits_prompt, {.max_tokens = 3});
  complete(too_small, too_small.schedule(), 3);
  expect_throw<std::runtime_error>([&] { (void)too_small.schedule(); },
                                   "Decode exceeding total cache capacity must fail finitely");
  too_small.clear();
}

} // namespace

int main() {
  try {
    test_chunked_prefill_and_eos();
    test_stop_conditions();
    test_prefix_reuse_and_clear();
    test_hash_publication_and_shared_release();
    test_block_size_one_and_pressure();
    test_shadow_cache_across_batched_chunks();
    test_invalid_inputs_and_pending_batches();
    std::cout << "Scheduler and KV block manager tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Scheduler test failed: " << error.what() << '\n';
    return 1;
  }
}
