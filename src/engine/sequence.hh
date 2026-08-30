#pragma once

#include <cstdint>
#include <vector>

#include "nano-vllm/nano-vllm.hh"

namespace nanovllm {

enum class SequenceStatus { Waiting, Running, Finished };

struct Sequence {
  std::size_t id;
  SequenceStatus status;

  std::vector<std::int32_t> token_ids;

  std::size_t num_prompt_tokens;
  std::size_t num_cached_tokens;
  std::size_t num_scheduled_tokens;

  std::vector<std::size_t> block_ids;
  std::size_t block_size;

  SamplingParams sampling_params;

  [[nodiscard]] std::size_t num_tokens() const { return token_ids.size(); }

  [[nodiscard]] std::size_t num_generated_tokens() const {
    return token_ids.size() - num_prompt_tokens;
  }

  [[nodiscard]] std::span<const std::int32_t> prompt_token_ids() const {
    return std::span<const std::int32_t>(token_ids).first(num_prompt_tokens);
  }

  [[nodiscard]] std::span<const std::int32_t> generated_token_ids() const {
    return std::span<const std::int32_t>(token_ids).subspan(num_prompt_tokens);
  }

  [[nodiscard]] std::int32_t token_id(std::size_t index) const {
    return token_ids[index];
  }

  [[nodiscard]] std::optional<std::int32_t> last_token_id() const {
    if (token_ids.empty()) {
      return std::nullopt;
    }
    return token_ids.back();
  }

  [[nodiscard]] std::size_t num_blocks() const {
    return (token_ids.size() + block_size - 1) / block_size;
  }

  [[nodiscard]] std::size_t num_allocated_blocks() const {
    return block_ids.size();
  }

  [[nodiscard]] std::size_t num_tokens_in_last_block() const {
    return token_ids.size() - block_size * (block_ids.size() - 1);
  }

  [[nodiscard]] std::optional<std::size_t> block_id(std::size_t index) const {
    if (index >= block_ids.size()) {
      return std::nullopt;
    }
    return block_ids[index];
  }

  [[nodiscard]] std::span<const std::int32_t> block_token_ids(
      std::size_t index) const {
    const auto start = index * block_size;
    const auto length = std::min(block_size, token_ids.size() - start);
    return std::span<const std::int32_t>(token_ids).subspan(start, length);
  }

  void append_token(std::int32_t token_id) { token_ids.push_back(token_id); }
};

} // namespace nanovllm
