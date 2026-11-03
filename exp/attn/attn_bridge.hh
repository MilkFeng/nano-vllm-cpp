#pragma once

#include <cstdint>

#ifdef _WIN32
  #define NANO_ATTN_EXPORT __declspec(dllexport)
#else
  #define NANO_ATTN_EXPORT __attribute__((visibility("default")))
#endif

// All pointers belong to the caller. Lengths and page IDs are checked once by
// the harness before capture; the bridge never copies GPU metadata to the CPU.
struct NanoAttnArgs {
  void* query;
  void* key_cache;
  void* value_cache;
  void* new_key;
  void* new_value;
  void* cu_query;
  void* cu_key;
  void* block_table;
  void* output;
  void* stream;

  std::int32_t dtype;
  std::int32_t batch;
  std::int32_t tokens;
  std::int32_t query_heads;
  std::int32_t kv_heads;
  std::int32_t head_dim;
  std::int32_t pages;
  std::int32_t page_size;
  std::int32_t table_width;
  std::int32_t max_query;
  std::int32_t max_key;
  std::int32_t causal;

  // Appended so the prefix remains compatible with pre-Split-KV libraries.
  void* workspace = nullptr;
  std::uint64_t workspace_elements = 0;
};

extern "C" {

NANO_ATTN_EXPORT int nano_attn_forward(const NanoAttnArgs* args) noexcept;
NANO_ATTN_EXPORT int nano_attn_workspace_size(const NanoAttnArgs* args,
                                              std::uint64_t* elements) noexcept;
NANO_ATTN_EXPORT const char* nano_attn_error() noexcept;
NANO_ATTN_EXPORT const char* nano_attn_build_type() noexcept;
}
