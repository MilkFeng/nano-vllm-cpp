#pragma once

#include <cstddef>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <span>
#include <string_view>

#include "common/macros.hh"
#include "weights/tensor.hh"

namespace nanovllm {

VLLM_DEFINE_NLOHMANN_JSON(DataType)

struct TensorInfo {
  DataType dtype = DataType::Float32;
  std::vector<std::size_t> shape;

  // Offsets are relative to the beginning of the Safetensors data section.
  std::uint64_t data_begin = 0;
  std::uint64_t data_end = 0;

  [[nodiscard]] std::size_t element_count() const;
  [[nodiscard]] std::uint64_t byte_size() const;

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static TensorInfo from_json(const nlohmann::json& j);
};

VLLM_DEFINE_NLOHMANN_JSON(TensorInfo)

struct TensorMap {
  using Tensors = std::map<std::string, TensorInfo, std::less<>>;

  Tensors tensors;

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static TensorMap from_json(const nlohmann::json& j);
};

VLLM_DEFINE_NLOHMANN_JSON(TensorMap)

class Safetensors {
 public:
  Safetensors();
  ~Safetensors();

  Safetensors(const Safetensors&) = delete;
  Safetensors& operator=(const Safetensors&) = delete;

  Safetensors(Safetensors&& other) noexcept;
  Safetensors& operator=(Safetensors&& other) noexcept;

  // Opens a safetensors file from the given path
  // Uses mmap to memory-map the file for efficient access
  void open(const std::filesystem::path& path);

  // Closes the file and releases the memory-mapped data
  void close() noexcept;

  [[nodiscard]] bool is_open() const noexcept;

  [[nodiscard]] const TensorMap& tensor_map() const noexcept;
  [[nodiscard]] const TensorInfo& tensor(std::string_view name) const;

  [[nodiscard]] std::span<const std::byte> raw_data(
      const TensorInfo& info) const;

  [[nodiscard]] std::span<const std::byte> raw_data(
      std::string_view name) const {
    return raw_data(tensor(name));
  }

 private:
  std::filesystem::path path_;

  int fd_ = -1;
  std::byte* mapping_ = nullptr;
  std::size_t file_size_ = 0;

  std::size_t data_offset_ = 0;
  std::size_t data_size_ = 0;

  TensorMap tensor_map_;
};

} // namespace nanovllm
