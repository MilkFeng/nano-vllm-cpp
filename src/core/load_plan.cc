#include "core/load_plan.hh"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <format>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

#include "common/cuda_utils.hh"
#include "kernels/cpu/fp8_dequant.hh"

namespace nanovllm {
namespace {

std::uint16_t bfloat16_bits(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  // Round to nearest, ties to even, preserving NaNs.
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return (bits >> 16) | 0x40U;
  }
  return (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
}

void copy_float(const Checkpoint& checkpoint, std::string_view name,
                TensorView output) {
  const auto& source = checkpoint.tensor(name);
  const auto bytes = checkpoint.raw_data(name);
  if (source.dtype == output.dtype()) {
    std::memcpy(output.data(), bytes.data(), bytes.size());
    return;
  }
  for (std::size_t i = 0; i < output.numel(); ++i) {
    if (output.dtype() == DataType::Float32) {
      std::uint16_t bits;
      std::memcpy(&bits, bytes.data() + i * sizeof(bits), sizeof(bits));
      output.data<float>()[i] =
          std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
    } else {
      assert(output.dtype() == DataType::BFloat16
             && source.dtype == DataType::Float32);
      std::uint32_t bits;
      std::memcpy(&bits, bytes.data() + i * sizeof(bits), sizeof(bits));
      output.data<std::uint16_t>()[i] =
          bfloat16_bits(std::bit_cast<float>(bits));
    }
  }
}

} // namespace

LoadPlan::LoadPlan(const Checkpoint& checkpoint, const ModelConfig& config)
    : checkpoint_(checkpoint),
      fp8_block_size_(config.quantization_config
                          ? config.quantization_config->weight_block_size
                          : std::array<std::size_t, 2>{}) {}

void LoadPlan::add_module(Module& module, std::string_view prefix) {
  std::string stem(prefix);
  if (!stem.empty() && !stem.ends_with('.')) {
    stem += '.';
  }

  const auto parameters = module.named_parameters();
  std::vector<Entry> entries;
  entries.reserve(parameters.size());
  for (const auto& [name, parameter] : parameters) {
    Entry entry{stem + name, parameter->view(), {}};
    const auto& source = checkpoint_.tensor(entry.name);
    const auto& destination = entry.destination;
    const auto dtype = destination.dtype();

    if (!std::ranges::equal(source.shape, destination.shape())) {
      throw std::runtime_error(
          std::format("tensor '{}' has an unexpected shape", entry.name));
    }
    if (!destination.is_contiguous()) {
      throw std::runtime_error(std::format(
          "destination for tensor '{}' must be contiguous", entry.name));
    }
    if (dtype != DataType::Float8E4M3FN && dtype != DataType::BFloat16
        && dtype != DataType::Float32) {
      throw std::runtime_error(std::format(
          "destination for tensor '{}' must be BF16, F32 or F8_E4M3",
          entry.name));
    }

    if (dtype == DataType::Float8E4M3FN) {
      if (source.dtype != DataType::Float8E4M3FN) {
        throw std::runtime_error(
            "compact FP8 weight requires an FP8 checkpoint");
      }
    } else if (source.dtype == DataType::Float8E4M3FN) {
      if (source.shape.size() != 2 || !entry.name.ends_with(".weight")) {
        throw std::runtime_error("FP8 expansion needs a 2D .weight tensor");
      }
      const auto [block_rows, block_columns] = fp8_block_size_;
      if (block_rows == 0 || block_columns == 0) {
        throw std::runtime_error("FP8 expansion requires a weight block size");
      }
      entry.scale_name = entry.name + "_scale_inv";
      const auto& scale = checkpoint_.tensor(entry.scale_name);
      if (scale.dtype != DataType::BFloat16
          && scale.dtype != DataType::Float32) {
        throw std::runtime_error(std::format(
            "FP8 scale '{}' must be BF16 or F32", entry.scale_name));
      }
      const auto rows = source.shape[0];
      const auto columns = source.shape[1];
      const std::array scale_shape{
          rows / block_rows + (rows % block_rows != 0),
          columns / block_columns + (columns % block_columns != 0)};
      if (!std::ranges::equal(scale.shape, scale_shape)) {
        throw std::runtime_error(std::format(
            "FP8 scale '{}' has an unexpected shape", entry.scale_name));
      }
    } else if (source.dtype != DataType::BFloat16
               && source.dtype != DataType::Float32) {
      throw std::runtime_error(std::format(
          "tensor '{}' must be BF16, F32 or block F8_E4M3", entry.name));
    }

    entries.push_back(std::move(entry));
  }

  entries_.insert(entries_.end(), std::make_move_iterator(entries.begin()),
                  std::make_move_iterator(entries.end()));
}

void LoadPlan::fill_cpu(const Entry& entry, TensorView output) const {
  if (!entry.scale_name.empty()) {
    auto scale = Tensor::empty(checkpoint_.tensor(entry.scale_name).shape,
                               DataType::Float32);
    copy_float(checkpoint_, entry.scale_name, scale.view());

    const auto [block_rows, block_columns] = fp8_block_size_;
    const auto& info = checkpoint_.tensor(entry.name);
    ConstTensorView weight(checkpoint_.raw_data(entry.name).data(), info.dtype,
                           TensorLayout(info.shape), Device::cpu());
    if (output.dtype() == DataType::Float32) {
      kernels::cpu::dequantize_fp8(weight, scale.view(), output, block_rows,
                                   block_columns);
    } else {
      auto expanded = Tensor::empty(info.shape, DataType::Float32);
      kernels::cpu::dequantize_fp8(weight, scale.view(), expanded.view(),
                                   block_rows, block_columns);
      assert(output.dtype() == DataType::BFloat16);
      for (std::size_t i = 0; i < output.numel(); ++i) {
        output.data<std::uint16_t>()[i] =
            bfloat16_bits(expanded.view().data<float>()[i]);
      }
    }
  } else if (output.dtype() == checkpoint_.tensor(entry.name).dtype) {
    const auto bytes = checkpoint_.raw_data(entry.name);
    std::memcpy(output.data(), bytes.data(), bytes.size());
  } else {
    copy_float(checkpoint_, entry.name, output);
  }
}

ModelLoadResult LoadPlan::apply() const {
  std::set<std::string_view> sources;
  ModelLoadResult result;
  const auto count = [&](std::string_view name) {
    if (sources.insert(name).second) {
      ++result.num_tensors;
      result.num_bytes += checkpoint_.tensor(name).byte_size();
    }
  };

  for (const auto& entry : entries_) {
    count(entry.name);
    if (!entry.scale_name.empty()) {
      count(entry.scale_name);
    }
  }

  for (const auto& entry : entries_) {
    const auto& destination = entry.destination;
    if (destination.device().type == DeviceType::Cpu) {
      fill_cpu(entry, destination);
    } else {
      auto staging = Tensor::empty(
          {destination.shape().begin(), destination.shape().end()},
          destination.dtype());
      fill_cpu(entry, staging.view());

      check_cuda(cudaMemcpy(destination.data(), staging.data(),
                            staging.size_bytes(), cudaMemcpyHostToDevice),
                 std::format("load checkpoint tensor '{}'", entry.name));
    }
  }
  return result;
}

} // namespace nanovllm
