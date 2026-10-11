#pragma once

#include <cstddef>
#include <cstdint>

namespace nanovllm {

// Float8E4M3FN is a storage format; existing CUDA math uses F32/F16/BF16.
enum class DataType : std::uint8_t {
  Float32,
  Float16,
  BFloat16,
  Int32,
  Float8E4M3FN
};

[[nodiscard]] std::size_t data_type_size(DataType dtype);

enum class DeviceType : std::uint8_t { Cpu, Cuda };

struct Device {
  DeviceType type = DeviceType::Cpu;
  int index = 0;

  Device() noexcept = default;
  Device(DeviceType type, int index = 0);

  [[nodiscard]] static Device cpu() noexcept;
  [[nodiscard]] static Device cuda(int index);
  [[nodiscard]] bool is_valid() const;

  friend bool operator==(const Device&, const Device&) = default;
};

} // namespace nanovllm
