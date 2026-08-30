#include "engine/block_manager.hh"

#include <cstdint>
#include <optional>

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

  for (std::size_t i = 0; i < seq.num_blocks() - 1; ++i) {
    const auto token_ids = seq.block_token_ids(i);
    hash = compute_hash(token_ids, hash);

    auto it = hash_to_block_id_.find(*hash);
    if (it == hash_to_block_id_.end()) {
      break;
    }

    const auto block_id = it->second;
    if (!std::equal(token_ids.begin(), token_ids.end(),
                    blocks_[block_id].token_ids.begin())) {
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
  const auto num_allocate_blocks = (seq.num_tokens() % block_size_ == 1);
  return {
      .num_reused_blocks = 0,
      .num_new_blocks = num_allocate_blocks,
      .num_allocate_blocks = num_allocate_blocks,
  };
}

void BlockManager::decode_allocate(Sequence& seq, BlockAllocation allocation) {
  if (allocation.num_allocate_blocks == 1) {
    const auto block_id = allocate_block();
    seq.block_ids.push_back(block_id);
  }
}

void BlockManager::decode_allocate(Sequence& seq) {
  const auto allocation = predict_decode_allocation(seq);
  if (allocation.num_allocate_blocks == 1) {
    const auto block_id = allocate_block();
    seq.block_ids.push_back(block_id);
  }
}

void BlockManager::deallocate(Sequence& seq) {
  for (int i = seq.block_ids.size() - 1; i >= 0; --i) {
    deallocate_block(seq.block_ids[i]);
  }
  seq.num_cached_tokens = 0;
  seq.block_ids.clear();
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
  auto& block = blocks_[block_id];
  --block.ref_count;
  if (block.ref_count == 0) {
    used_block_ids_.erase(block_id);
    free_block_ids_.push_back(block_id);
  }
}

} // namespace nanovllm
