#include "safetensors.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <format>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace nanovllm {

VLLM_IMPL_NLOHMANN_JSON_FOR_ENUM(DataType,
                                 {{DataType::Float32, "F32"},
                                  {DataType::Float16, "F16"},
                                  {DataType::BFloat16, "BF16"},
                                  {DataType::Int32, "I32"},
                                  {DataType::Float8E4M3FN, "F8_E4M3"}})

std::size_t TensorInfo::element_count() const {
  std::size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension != 0
        && count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("tensor element count is too large");
    }
    count *= dimension;
  }
  return count;
}

std::uint64_t TensorInfo::byte_size() const {
  if (data_end < data_begin) {
    throw std::logic_error("tensor data range is invalid");
  }
  return data_end - data_begin;
}

nlohmann::json TensorInfo::to_json() const {
  return {
      {"dtype", dtype},
      {"shape", shape},
      {"data_offsets", {data_begin, data_end}},
  };
}

TensorInfo TensorInfo::from_json(const nlohmann::json& j) {
  const auto data_offsets =
      j.at("data_offsets").get<std::array<std::uint64_t, 2>>();
  return {
      .dtype = j.at("dtype").get<DataType>(),
      .shape = j.at("shape").get<std::vector<std::size_t>>(),
      .data_begin = data_offsets[0],
      .data_end = data_offsets[1],
  };
}

VLLM_IMPL_NLOHMANN_JSON(TensorInfo)

nlohmann::json TensorMap::to_json() const {
  auto result = nlohmann::json::object();
  for (const auto& [name, tensor] : tensors) {
    result[name] = tensor;
  }
  return result;
}

TensorMap TensorMap::from_json(const nlohmann::json& j) {
  TensorMap tensor_map;
  for (const auto& [name, value] : j.items()) {
    // Skip metadata
    if (name == "__metadata__") continue;
    const auto tensor = value.get<TensorInfo>();

    // Validate data range
    if (tensor.data_end < tensor.data_begin) {
      throw std::runtime_error(
          std::format("tensor '{}' has an invalid data range", name));
    }

    // Validate byte size
    const auto element_count = tensor.element_count();
    const auto element_size = data_type_size(tensor.dtype);
    if (element_count
        > std::numeric_limits<std::size_t>::max() / element_size) {
      throw std::runtime_error(
          std::format("tensor '{}' byte size is too large", name));
    }
    const auto expected_size = element_count * element_size;
    if (tensor.byte_size() != expected_size) {
      throw std::runtime_error(
          std::format("tensor '{}' byte size does not match its shape", name));
    }

    tensor_map.tensors.emplace(name, tensor);
  }

  // Validate all tensor offsets do not overlap
  for (const auto& [name1, tensor1] : tensor_map.tensors) {
    for (const auto& [name2, tensor2] : tensor_map.tensors) {
      if (name1 == name2) continue;

      if (tensor1.data_end > tensor2.data_begin
          && tensor2.data_end > tensor1.data_begin) {
        throw std::runtime_error(
            std::format("tensor '{}' and '{}' overlap", name1, name2));
      }
    }
  }

  return tensor_map;
}

VLLM_IMPL_NLOHMANN_JSON(TensorMap)

namespace {

constexpr std::size_t kHeaderLengthSize = sizeof(std::uint64_t);
constexpr std::uint64_t kMaxHeaderSize = 100'000'000;

// Read a 64-bit unsigned integer in little-endian format
std::uint64_t decode_u64_le(const std::byte* bytes) noexcept {
  std::uint64_t result = 0;
  for (std::size_t i = 0; i < kHeaderLengthSize; ++i) {
    result |= std::uint64_t{std::to_integer<unsigned char>(bytes[i])}
              << (i * 8);
  }
  return result;
}
} // namespace

Safetensors::Safetensors(const std::filesystem::path& path)
    : path_(path) {
  try {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ == -1) {
      throw std::system_error(
          errno, std::generic_category(),
          std::format("failed to open Safetensors file '{}'", path.string()));
    }

    struct stat file_stat{};
    if (::fstat(fd_, &file_stat) == -1) {
      throw std::system_error(
          errno, std::generic_category(),
          std::format("failed to stat Safetensors file '{}'", path.string()));
    }
    if (file_stat.st_size < static_cast<off_t>(kHeaderLengthSize)) {
      throw std::runtime_error("file is too small to contain a header length");
    }
    file_size_ = static_cast<std::size_t>(file_stat.st_size);

    void* const mapping =
        ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping == MAP_FAILED) {
      throw std::system_error(
          errno, std::generic_category(),
          std::format("failed to map Safetensors file '{}'", path.string()));
    }
    mapping_ = static_cast<std::byte*>(mapping);

    const auto header_size = decode_u64_le(mapping_);
    if (header_size > kMaxHeaderSize) {
      throw std::runtime_error("header exceeds the 100 MB size limit");
    }
    if (header_size > file_size_ - kHeaderLengthSize) {
      throw std::runtime_error("header extends past the end of the file");
    }

    data_offset_ = kHeaderLengthSize + static_cast<std::size_t>(header_size);
    data_size_ = file_size_ - data_offset_;

    const char* const header_begin =
        reinterpret_cast<const char*>(mapping_ + kHeaderLengthSize);
    const auto header =
        nlohmann::json::parse(header_begin, header_begin + header_size);
    tensor_map_ = header.get<TensorMap>();

    for (const auto& [name, tensor] : tensor_map_.tensors) {
      if (tensor.data_end > data_size_) {
        throw std::runtime_error(
            std::format("tensor '{}' is out of bounds", name));
      }
    }
  } catch (...) {
    // A failed constructor does not run the destructor.
    close();
    throw;
  }
}

Safetensors::~Safetensors() {
  close();
}

Safetensors::Safetensors(Safetensors&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      file_size_(std::exchange(other.file_size_, 0)),
      data_offset_(std::exchange(other.data_offset_, 0)),
      data_size_(std::exchange(other.data_size_, 0)),
      tensor_map_(std::move(other.tensor_map_)) {}

Safetensors& Safetensors::operator=(Safetensors&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  close();
  path_ = std::move(other.path_);
  fd_ = std::exchange(other.fd_, -1);
  mapping_ = std::exchange(other.mapping_, nullptr);
  file_size_ = std::exchange(other.file_size_, 0);
  data_offset_ = std::exchange(other.data_offset_, 0);
  data_size_ = std::exchange(other.data_size_, 0);
  tensor_map_ = std::move(other.tensor_map_);
  return *this;
}

void Safetensors::close() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(mapping_, file_size_);
    mapping_ = nullptr;
  }
  if (fd_ != -1) {
    ::close(fd_);
    fd_ = -1;
  }

  file_size_ = 0;
  data_offset_ = 0;
  data_size_ = 0;
}

bool Safetensors::is_open() const noexcept {
  return mapping_ != nullptr;
}

const TensorMap& Safetensors::tensor_map() const noexcept {
  return tensor_map_;
}

const TensorInfo& Safetensors::tensor(std::string_view name) const {
  const auto it = tensor_map_.tensors.find(name);
  if (it == tensor_map_.tensors.end()) {
    throw std::out_of_range(
        std::format("tensor '{}' was not found in '{}'", name, path_.string()));
  }
  return it->second;
}

std::span<const std::byte> Safetensors::raw_data(const TensorInfo& info) const {
  if (mapping_ == nullptr) {
    throw std::logic_error("Safetensors file is not mapped");
  }
  assert(info.data_end >= info.data_begin);
  assert(info.data_end <= data_size_);

  const auto begin = static_cast<std::size_t>(info.data_begin);
  const auto size = static_cast<std::size_t>(info.data_end - info.data_begin);
  return {mapping_ + data_offset_ + begin, size};
}

} // namespace nanovllm
