#include "engine/block_manager.hh"

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "common/hasher.hh"

namespace nanovllm {

void Block::reset() {
  token_ids.clear();
  hash = std::nullopt;
  ref_count = 1;
}

BlockManager::BlockManager(std::size_t num_blocks, std::size_t block_size)
    : blocks_(num_blocks),
      block_size_(block_size) {
  if (num_blocks == 0 || block_size == 0) {
    throw std::invalid_argument(
        "KV cache must have positive block count and size");
  }
  for (std::size_t i = 0; i < num_blocks; ++i) {
    blocks_[i].id = i;
    free_block_ids_.push_back(i);
  }
}

bool BlockManager::can_allocate(BlockAllocation allocation) const {
  return allocation.num_allocate_blocks <= free_block_ids_.size();
}

BlockAllocation BlockManager::predict_prefill_allocation(
    const Sequence& seq) const {
  std::optional<std::uint64_t> hash = std::nullopt;

  BlockAllocation allocation = {
      .num_reused_blocks = 0,
      .num_new_blocks = seq.num_blocks(),
      .num_allocate_blocks = seq.num_blocks(),
  };

  for (std::size_t i = 0; i + 1 < seq.num_blocks(); ++i) {
    const auto token_ids = seq.block_token_ids(i);
    hash = compute_hash(token_ids, hash);

    auto it = hash_to_block_id_.find(*hash);
    if (it == hash_to_block_id_.end()) {
      break;
    }

    const auto block_id = it->second;
    if (!std::ranges::equal(token_ids, blocks_[block_id].token_ids)) {
      break;
    }

    ++allocation.num_reused_blocks;
    --allocation.num_new_blocks;

    // If the block is already used, we don't need to allocate it again.
    if (used_block_ids_.contains(block_id)) {
      --allocation.num_allocate_blocks;
    }
  }

  return allocation;
}

void BlockManager::prefill_allocate(Sequence& seq, BlockAllocation allocation) {
  const auto predicted = predict_prefill_allocation(seq);
  if (!seq.block_ids.empty() || !can_allocate(allocation)
      || allocation.num_reused_blocks != predicted.num_reused_blocks
      || allocation.num_new_blocks != predicted.num_new_blocks
      || allocation.num_allocate_blocks != predicted.num_allocate_blocks) {
    throw std::logic_error("invalid or unavailable prefill block allocation");
  }
  std::optional<std::uint64_t> hash = std::nullopt;
  for (std::size_t i = 0; i < allocation.num_reused_blocks; ++i) {
    const auto token_ids = seq.block_token_ids(i);
    hash = compute_hash(token_ids, hash);

    const auto block_id = hash_to_block_id_.at(*hash);
    auto& block = blocks_[block_id];

    if (used_block_ids_.contains(block_id)) {
      ++block.ref_count;
    } else {
      // If the block is not used, reallocate it from the free block pool.
      allocate_block(block_id);
    }

    seq.block_ids.push_back(block_id);
  }

  for (std::size_t i = allocation.num_reused_blocks; i < seq.num_blocks();
       ++i) {
    const auto block_id = allocate_block();
    seq.block_ids.push_back(block_id);
  }

  seq.num_cached_tokens = allocation.num_reused_blocks * block_size_;
}

BlockAllocation BlockManager::predict_decode_allocation(
    const Sequence& seq) const {
  if (seq.num_allocated_blocks() > seq.num_blocks()) {
    throw std::logic_error("sequence owns more KV blocks than its token count");
  }
  const auto num_allocate_blocks =
      seq.num_blocks() - seq.num_allocated_blocks();
  return {
      .num_reused_blocks = 0,
      .num_new_blocks = num_allocate_blocks,
      .num_allocate_blocks = num_allocate_blocks,
  };
}

void BlockManager::decode_allocate(Sequence& seq, BlockAllocation allocation) {
  const auto predicted = predict_decode_allocation(seq);
  if (!can_allocate(allocation)
      || allocation.num_reused_blocks != predicted.num_reused_blocks
      || allocation.num_new_blocks != predicted.num_new_blocks
      || allocation.num_allocate_blocks != predicted.num_allocate_blocks) {
    throw std::logic_error("invalid or unavailable decode block allocation");
  }
  for (std::size_t i = 0; i < allocation.num_allocate_blocks; ++i) {
    const auto block_id = allocate_block();
    seq.block_ids.push_back(block_id);
  }
}

void BlockManager::decode_allocate(Sequence& seq) {
  decode_allocate(seq, predict_decode_allocation(seq));
}

void BlockManager::deallocate(Sequence& seq) {
  for (auto it = seq.block_ids.rbegin(); it != seq.block_ids.rend(); ++it) {
    deallocate_block(*it);
  }
  seq.num_cached_tokens = 0;
  seq.num_scheduled_tokens = 0;
  seq.block_ids.clear();
}

void BlockManager::hash_completed_blocks(const Sequence& seq) {
  const auto begin = seq.num_cached_tokens / block_size_;
  const auto end =
      (seq.num_cached_tokens + seq.num_scheduled_tokens) / block_size_;

  if (begin == end) {
    return;
  }

  std::optional<std::uint64_t> hash;

  if (begin != 0) {
    hash = blocks_.at(seq.block_ids.at(begin - 1)).hash;
    if (!hash) {
      throw std::logic_error("computed KV prefix is missing its hash");
    }
  }
  for (auto index = begin; index < end; ++index) {
    auto& block = blocks_.at(seq.block_ids.at(index));
    const auto tokens = seq.block_token_ids(index);
    hash = compute_hash(tokens, hash);
    block.hash = hash;
    block.token_ids.assign(tokens.begin(), tokens.end());
    hash_to_block_id_[*hash] = block.id;
  }
}

std::uint64_t BlockManager::compute_hash(
    std::span<const std::int32_t> token_ids,
    std::optional<std::uint64_t> prefix) const {
  Hasher hasher;
  if (prefix.has_value()) {
    hasher.update(*prefix);
  }
  for (const auto token_id : token_ids) {
    hasher.update(token_id);
  }
  return hasher.digest();
}

std::size_t BlockManager::allocate_block() {
  if (free_block_ids_.empty()) throw std::runtime_error("KV cache is full");
  const auto block_id = free_block_ids_.front();
  free_block_ids_.pop_front();

  auto& block = blocks_[block_id];

  if (block.hash.has_value()) {
    const auto it = hash_to_block_id_.find(*block.hash);
    if (it != hash_to_block_id_.end() && it->second == block_id) {
      hash_to_block_id_.erase(it);
    }
  }

  block.reset();

  used_block_ids_.insert(block_id);
  return block_id;
}

void BlockManager::allocate_block(std::size_t block_id) {
  blocks_[block_id].ref_count = 1;
  used_block_ids_.insert(block_id);

  const auto it =
      std::find(free_block_ids_.begin(), free_block_ids_.end(), block_id);
  if (it != free_block_ids_.end()) {
    free_block_ids_.erase(it);
  }
}

void BlockManager::deallocate_block(std::size_t block_id) {
  auto& block = blocks_.at(block_id);
  if (block.ref_count == 0 || !used_block_ids_.contains(block_id)) {
    throw std::logic_error("cannot release an unreferenced KV block");
  }
  --block.ref_count;
  if (block.ref_count == 0) {
    used_block_ids_.erase(block_id);
    free_block_ids_.push_back(block_id);
  }
}

} // namespace nanovllm
