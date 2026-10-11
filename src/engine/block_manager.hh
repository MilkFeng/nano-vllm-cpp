#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "engine/sequence.hh"

namespace nanovllm {

struct Block {
  std::size_t id;
  std::optional<std::uint64_t> hash;
  std::vector<std::int32_t> token_ids;

  std::size_t ref_count;

  void reset();
};

struct BlockAllocation {
  std::size_t num_reused_blocks;
  std::size_t num_new_blocks;
  std::size_t num_allocate_blocks;
};

class BlockManager {
 public:
  BlockManager(std::size_t num_blocks, std::size_t block_size,
               bool enable_prefix_cache = true);
  ~BlockManager() = default;

  BlockManager(const BlockManager&) = delete;
  BlockManager& operator=(const BlockManager&) = delete;

  BlockManager(BlockManager&&) = default;
  BlockManager& operator=(BlockManager&&) = default;

  [[nodiscard]] bool can_allocate(BlockAllocation allocation) const;

  [[nodiscard]] BlockAllocation predict_prefill_allocation(
      const Sequence& seq) const;

  [[nodiscard]] bool can_prefill_allocate(Sequence& seq) const {
    return can_allocate(predict_prefill_allocation(seq));
  }

  void prefill_allocate(Sequence& seq, BlockAllocation allocation);

  [[nodiscard]] BlockAllocation predict_decode_allocation(
      const Sequence& seq) const;

  [[nodiscard]] bool can_decode_allocate(Sequence& seq) const {
    return can_allocate(predict_decode_allocation(seq));
  }

  void decode_allocate(Sequence& seq, BlockAllocation allocation);
  void decode_allocate(Sequence& seq);

  void deallocate(Sequence& seq);

  // Publish only blocks whose K/V has been computed successfully in all layers.
  void hash_completed_blocks(const Sequence& seq);

 private:
  [[nodiscard]] std::uint64_t compute_hash(
      std::span<const std::int32_t> token_ids,
      std::optional<std::uint64_t> prefix) const;

  [[nodiscard]] std::size_t allocate_block();
  void allocate_block(std::size_t block_id);

  void deallocate_block(std::size_t block_id);

 private:
  std::vector<Block> blocks_;
  std::size_t block_size_;
  bool enable_prefix_cache_;

  std::unordered_map<std::uint64_t, std::size_t> hash_to_block_id_;

  std::unordered_set<std::size_t> used_block_ids_;
  std::deque<std::size_t> free_block_ids_;
};

} // namespace nanovllm
