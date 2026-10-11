#pragma once

#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>

#include "core/safetensors.hh"

namespace nanovllm {

// A single safetensors file, an index file, or a model directory. For a
// directory, prefer model.safetensors.index.json over model.safetensors.
// Indexed shards are mmap'ed on first access and retained until destruction.
// Returned metadata references and byte spans borrow this object's lifetime.
// Calls must be serialized. This reader does not copy or dequantize weights.
class Checkpoint final {
 public:
  using WeightMap = std::map<std::string, std::string, std::less<>>;

  explicit Checkpoint(const std::filesystem::path& path);

  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;
  Checkpoint(Checkpoint&&) noexcept = default;
  Checkpoint& operator=(Checkpoint&&) noexcept = default;

  [[nodiscard]] const WeightMap& weight_map() const noexcept;
  [[nodiscard]] const TensorInfo& tensor(std::string_view name) const;
  [[nodiscard]] std::span<const std::byte> raw_data(
      std::string_view name) const;

 private:
  [[nodiscard]] const Safetensors& shard_for(std::string_view name) const;

  std::filesystem::path directory_;
  WeightMap weight_map_;
  mutable std::map<std::string, Safetensors, std::less<>> shards_;
};

} // namespace nanovllm
