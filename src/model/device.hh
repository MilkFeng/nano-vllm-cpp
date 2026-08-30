#pragma once

#include <cstdint>

namespace nanovllm {

enum class DataType : std::uint8_t { Float32, Float16, BFloat16, Int32 };

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
