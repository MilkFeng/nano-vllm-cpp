#include "engine/tokenizer.hh"

#include <tokenizers_c.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <minja/minja.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nanovllm {
namespace {

class EncodeResult final {
 public:
  EncodeResult() = default;
  ~EncodeResult() { tokenizers_free_encode_results(&result_, 1); }

  EncodeResult(const EncodeResult&) = delete;
  EncodeResult& operator=(const EncodeResult&) = delete;

  TokenizerEncodeResult* output() noexcept { return &result_; }
  const TokenizerEncodeResult& get() const noexcept { return result_; }

 private:
  TokenizerEncodeResult result_{};
};

} // namespace

class TokenizerBackend {
 public:
  explicit TokenizerBackend(const std::filesystem::path& tokenizer_path) {
    std::ifstream file(tokenizer_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string tokenizer_json = buffer.str();
    handle_ =
        tokenizers_new_from_str(tokenizer_json.data(), tokenizer_json.size());
    if (handle_ == nullptr) {
      throw std::runtime_error("tokenizers-cpp failed to create tokenizer");
    }
  }

  ~TokenizerBackend() {
    if (handle_ != nullptr) {
      tokenizers_free(handle_);
    }
  }

  TokenizerBackend(const TokenizerBackend&) = delete;
  TokenizerBackend& operator=(const TokenizerBackend&) = delete;
  TokenizerBackend(TokenizerBackend&&) = delete;
  TokenizerBackend& operator=(TokenizerBackend&&) = delete;

  [[nodiscard]] std::vector<std::int32_t> encode(
      std::string_view text, bool add_special_tokens) const {
    std::lock_guard lock(mutex_);

    EncodeResult result;
    tokenizers_encode(handle_, text.empty() ? "" : text.data(), text.size(),
                      static_cast<int>(add_special_tokens), result.output());

    const auto& encoded = result.get();
    if (encoded.len != 0 && encoded.token_ids == nullptr) {
      throw std::runtime_error("tokenizers-cpp returned invalid encoded data");
    }

    std::vector<std::int32_t> token_ids;
    token_ids.reserve(encoded.len);
    for (std::size_t i = 0; i < encoded.len; ++i) {
      token_ids.push_back(static_cast<std::int32_t>(encoded.token_ids[i]));
    }
    return token_ids;
  }

  [[nodiscard]] std::string decode(std::span<const std::int32_t> token_ids,
                                   bool skip_special_tokens) const {
    std::vector<std::uint32_t> unsigned_token_ids;
    unsigned_token_ids.reserve(token_ids.size());
    for (const auto token_id : token_ids) {
      if (token_id < 0) {
        throw std::invalid_argument("token ID must be non-negative");
      }
      unsigned_token_ids.push_back(static_cast<std::uint32_t>(token_id));
    }

    std::lock_guard lock(mutex_);

    constexpr std::uint32_t kEmptyInput = 0;
    const auto* data =
        unsigned_token_ids.empty() ? &kEmptyInput : unsigned_token_ids.data();
    tokenizers_decode(handle_, data, unsigned_token_ids.size(),
                      static_cast<int>(skip_special_tokens));

    const char* decoded_data = nullptr;
    std::size_t decoded_size = 0;
    tokenizers_get_decode_str(handle_, &decoded_data, &decoded_size);
    if (decoded_size != 0 && decoded_data == nullptr) {
      throw std::runtime_error("tokenizers-cpp returned invalid decoded data");
    }
    return std::string(decoded_data == nullptr ? "" : decoded_data,
                       decoded_size);
  }

  [[nodiscard]] std::int32_t token_to_id(std::string_view token) const {
    std::lock_guard lock(mutex_);

    std::int32_t token_id = -1;
    tokenizers_token_to_id(handle_, token.empty() ? "" : token.data(),
                           token.size(), &token_id);
    return token_id;
  }

  [[nodiscard]] std::string render_chat_template(
      const std::string& source, const nlohmann::ordered_json& values) const {
    std::lock_guard lock(mutex_);
    if (!chat_template_ast_) {
      // Match Transformers' Jinja whitespace settings. Compile lazily so raw
      // tokenization still works when a checkpoint has no usable chat template.
      chat_template_ast_ =
          minja::Parser::parse(source, {.trim_blocks = true,
                                        .lstrip_blocks = true,
                                        .keep_trailing_newline = false});
    }
    // Template variables and namespace state belong to a single render.
    auto context = minja::Context::make(minja::Value(values));
    return chat_template_ast_->render(context);
  }

 private:
  TokenizerHandle handle_ = nullptr;
  mutable std::mutex mutex_;
  mutable std::shared_ptr<minja::TemplateNode> chat_template_ast_;
};

namespace {

std::optional<std::string> read_token_string(const nlohmann::json& config,
                                             std::string_view field) {
  const auto key = std::string(field);
  const auto it = config.find(key);

  if (it == config.end() || it->is_null()) {
    return std::nullopt;
  }

  std::string token;

  if (it->is_string()) {
    token = it->get<std::string>();
  } else if (it->is_object()) {
    const auto content = it->find("content");
    if (content != it->end() && content->is_string()) {
      token = content->get<std::string>();
    } else {
      throw std::runtime_error(
          std::format("tokenizer config field object '{}' must have a string "
                      "'content' field",
                      field));
    }
  } else {
    throw std::runtime_error(std::format(
        "tokenizer config field '{}' must be null, a string, or an object",
        field));
  }

  return token;
}

std::optional<std::int32_t> resolve_token_id(
    const std::optional<std::string>& token, std::string_view field,
    const TokenizerBackend& backend) {
  if (!token) return std::nullopt;
  const auto token_id = backend.token_to_id(*token);
  if (token_id < 0) {
    throw std::runtime_error(
        std::format("tokenizer config field '{}' references unknown token '{}'",
                    field, *token));
  }
  return token_id;
}

} // namespace

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_path,
                     const std::filesystem::path& tokenizer_config_path)
    : backend_(std::make_unique<TokenizerBackend>(tokenizer_path)),
      eos_token_id_(-1) {
  std::ifstream file(tokenizer_config_path);
  nlohmann::json config;
  file >> config;

  const auto eos = read_token_string(config, "eos_token");
  const auto bos = read_token_string(config, "bos_token");
  const auto pad = read_token_string(config, "pad_token");
  eos_token_id_ = resolve_token_id(eos, "eos_token", *backend_).value();
  bos_token_id_ = resolve_token_id(bos, "bos_token", *backend_);
  pad_token_id_ = resolve_token_id(pad, "pad_token", *backend_);
  eos_token_ = eos;
  bos_token_ = bos;
  pad_token_ = pad;
  const auto chat_template = config.find("chat_template");
  if (chat_template != config.end() && !chat_template->is_null()) {
    if (!chat_template->is_string()) {
      throw std::runtime_error(
          "tokenizer chat_template must be a Jinja string; "
          "named template collections are not supported");
    }
    chat_template_ = chat_template->get<std::string>();
  }
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<std::int32_t> Tokenizer::encode(std::string_view text,
                                            bool add_special_tokens) const {
  return backend_->encode(text, add_special_tokens);
}

std::string Tokenizer::decode(std::span<const std::int32_t> token_ids,
                              bool skip_special_tokens) const {
  return backend_->decode(token_ids, skip_special_tokens);
}

std::string Tokenizer::apply_chat_template(
    std::span<const ChatMessage> messages, ChatTemplateOptions options) const {
  if (messages.empty()) {
    throw std::invalid_argument("chat messages must not be empty");
  }
  if (chat_template_.empty()) {
    throw std::runtime_error(
        "tokenizer config does not define a chat_template");
  }
  auto message_values = nlohmann::ordered_json::array();

  for (const auto& message : messages) {
    if (message.role != "system" && message.role != "user"
        && message.role != "assistant") {
      throw std::invalid_argument(
          "text chat messages support system, user and "
          "assistant roles; tools are not supported");
    }
    message_values.push_back(
        {{"role", message.role}, {"content", message.content}});
  }

  nlohmann::ordered_json values{
      {"messages", std::move(message_values)},
      {"add_generation_prompt", options.add_generation_prompt},
      {"enable_thinking", options.enable_thinking}};

  // Transformers passes special_tokens_map, which omits unset token fields.
  if (bos_token_) {
    values["bos_token"] = *bos_token_;
  }
  if (eos_token_) {
    values["eos_token"] = *eos_token_;
  }
  if (pad_token_) {
    values["pad_token"] = *pad_token_;
  }
  try {
    return backend_->render_chat_template(chat_template_, values);
  } catch (const std::exception& error) {
    throw std::runtime_error(
        std::string("failed to render tokenizer chat_template: ")
        + error.what());
  }
}

std::int32_t Tokenizer::eos_token_id() const noexcept {
  return eos_token_id_;
}

} // namespace nanovllm
