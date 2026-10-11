#pragma once

#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>

#include "core/safetensors.hh"

namespace nanovllm {

class Checkpoint final {
 public:
  // layer -> safetensor file name
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
