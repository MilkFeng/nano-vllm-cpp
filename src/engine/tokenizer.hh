#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "nanovllm/nanovllm.hh"

namespace nanovllm {

class TokenizerBackend;

class Tokenizer {
 public:
  Tokenizer(std::shared_ptr<Logger> logger,
            const std::filesystem::path& tokenizer_path,
            const std::filesystem::path& tokenizer_config_path);
  ~Tokenizer();

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  [[nodiscard]] std::vector<std::int32_t> encode(
      std::string_view text, bool add_special_tokens = false) const;

  [[nodiscard]] std::string decode(std::span<const std::int32_t> token_ids,
                                   bool skip_special_tokens = false) const;

  // Render the checkpoint's Jinja template for text-only messages. The result
  // is ready for encode(..., false); no additional special tokens are appended.
  [[nodiscard]] std::string apply_chat_template(
      std::span<const ChatMessage> messages,
      ChatTemplateOptions options = {}) const;

  [[nodiscard]] std::int32_t eos_token_id() const noexcept;

 private:
  std::shared_ptr<Logger> logger_;
  std::unique_ptr<TokenizerBackend> backend_;

  std::int32_t eos_token_id_;
  std::optional<std::int32_t> bos_token_id_;
  std::optional<std::int32_t> pad_token_id_;

  std::optional<std::string> bos_token_;
  std::optional<std::string> eos_token_;
  std::optional<std::string> pad_token_;
  std::string chat_template_;
};

} // namespace nanovllm
