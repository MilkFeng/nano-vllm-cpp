#include "core/checkpoint.hh"

#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace nanovllm {

Checkpoint::Checkpoint(const std::filesystem::path& path) {
  auto checkpoint_path = path;
  if (std::filesystem::is_directory(path)) {
    const auto index_path = path / "model.safetensors.index.json";
    if (std::filesystem::exists(index_path)) {
      checkpoint_path = index_path;
    } else {
      checkpoint_path = path / "model.safetensors";
    }
  }
  directory_ = checkpoint_path.parent_path();

  if (checkpoint_path.extension() == ".json") {
    // index json
    std::ifstream file(checkpoint_path);
    if (!file) {
      throw std::runtime_error(std::format(
          "failed to open checkpoint index '{}'", checkpoint_path.string()));
    }
    const auto index = nlohmann::json::parse(file);
    weight_map_ = index.at("weight_map").get<WeightMap>();

    if (weight_map_.empty()) {
      throw std::runtime_error("checkpoint index has an empty weight_map");
    }

    for (const auto& [name, filename] : weight_map_) {
      if (name.empty() || filename.empty()) {
        throw std::runtime_error(
            "checkpoint index has an empty tensor or shard name");
      }
    }
  } else {
    // safetensors
    Safetensors shard(checkpoint_path);
    const auto filename = checkpoint_path.filename().string();
    for (const auto& [name, info] : shard.tensor_map().tensors) {
      weight_map_.emplace(name, filename);
    }
    shards_.emplace(filename, std::move(shard));
  }
}

const Checkpoint::WeightMap& Checkpoint::weight_map() const noexcept {
  return weight_map_;
}

const Safetensors& Checkpoint::shard_for(std::string_view name) const {
  const auto weight = weight_map_.find(name);
  if (weight == weight_map_.end()) {
    throw std::out_of_range(
        std::format("tensor '{}' is not in the checkpoint", name));
  }

  const auto& filename = weight->second;
  auto shard = shards_.find(filename);
  if (shard == shards_.end()) {
    Safetensors mapped(directory_ / filename);
    shard = shards_.emplace(filename, std::move(mapped)).first;
  }
  return shard->second;
}

const TensorInfo& Checkpoint::tensor(std::string_view name) const {
  return shard_for(name).tensor(name);
}

std::span<const std::byte> Checkpoint::raw_data(std::string_view name) const {
  return shard_for(name).raw_data(name);
}

} // namespace nanovllm
