#include "core/model.hh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include <stdexcept>
#include <string_view>

#include "core/safetensors.hh"

namespace nanovllm {

namespace {

void validate_parameter(std::string_view name, const TensorView& destination,
                        const TensorInfo& source) {
  if (!destination.is_contiguous()) {
    throw std::runtime_error(
        std::format("destination for tensor '{}' must be contiguous", name));
  }

  if (destination.dtype() != source.dtype) {
    throw std::runtime_error(
        std::format("tensor '{}' has dtype {}, but the model expects {}", name,
                    magic_enum::enum_name(source.dtype),
                    magic_enum::enum_name(destination.dtype())));
  }

  if (!std::ranges::equal(destination.shape(), source.shape)) {
    throw std::runtime_error(
        std::format("tensor '{}' shape does not match the model", name));
  }

  if (destination.size_bytes() != source.byte_size()) {
    throw std::runtime_error(
        std::format("tensor '{}' byte size does not match the model", name));
  }
}

void copy_parameter(std::string_view name, const TensorView& destination,
                    const Safetensors& checkpoint, const TensorInfo& source) {
  const auto bytes = checkpoint.raw_data(source);
  if (bytes.empty()) return;

  const auto device = destination.device();
  if (device.type == DeviceType::Cpu) {
    std::memcpy(destination.data(), bytes.data(), bytes.size());
    return;
  }

  int previous_device = 0;
  auto status = cudaGetDevice(&previous_device);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::format(
        "failed to get the current CUDA device while loading tensor '{}': {}",
        name, cudaGetErrorString(status)));
  }

  const bool restore_device = previous_device != device.index;
  if (restore_device) {
    status = cudaSetDevice(device.index);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::format(
          "failed to select CUDA device {} while loading tensor '{}': {}",
          device.index, name, cudaGetErrorString(status)));
    }
  }

  status = cudaMemcpy(destination.data(), bytes.data(), bytes.size(),
                      cudaMemcpyHostToDevice);
  if (restore_device) {
    cudaSetDevice(previous_device);
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::format("failed to copy tensor '{}' to CUDA device {}: {}", name,
                    device.index, cudaGetErrorString(status)));
  }
}

} // namespace

Model::CheckpointBindings Model::checkpoint_destinations() {
  CheckpointBindings result;
  for (const auto& [name, parameter] : named_parameters()) {
    result.emplace(name, parameter->view());
  }
  for (const auto& [name, destination] : checkpoint_bindings_) {
    result.insert_or_assign(name, destination);
  }
  return result;
}

ModelLoadResult Model::load_ckpt(const std::filesystem::path& ckpt_path,
                                 ModelLoadOptions options) {
  Safetensors ckpt;
  ckpt.open(ckpt_path);

  const auto destinations = checkpoint_destinations();
  const auto& checkpoint_tensors = ckpt.tensor_map().tensors;

  // Validate the complete mapping before copying anything, so a missing or
  // incompatible tensor cannot leave the model partially initialized.
  for (const auto& [name, destination] : destinations) {
    const auto source = checkpoint_tensors.find(name);
    if (source == checkpoint_tensors.end()) {
      throw std::runtime_error(
          std::format("checkpoint '{}' does not contain model parameter '{}'",
                      ckpt_path.string(), name));
    }
    validate_parameter(name, destination, source->second);
  }

  if (!options.allow_unexpected_tensors) {
    for (const auto& [name, source] : checkpoint_tensors) {
      if (!destinations.contains(name)) {
        throw std::runtime_error(
            std::format("checkpoint '{}' contains unexpected tensor '{}'",
                        ckpt_path.string(), name));
      }
    }
  }

  ModelLoadResult result;
  result.num_tensors = destinations.size();
  for (const auto& [name, destination] : destinations) {
    const auto source_size = checkpoint_tensors.at(name).byte_size();
    if (source_size
        > std::numeric_limits<std::uint64_t>::max() - result.num_bytes) {
      throw std::overflow_error("loaded model byte count is too large");
    }
    result.num_bytes += source_size;
  }

  for (const auto& [name, destination] : destinations) {
    copy_parameter(name, destination, ckpt, checkpoint_tensors.at(name));
  }

  return result;
}

void Model::register_checkpoint_binding(std::string name,
                                        TensorView destination) {
  if (name.empty()) {
    throw std::invalid_argument("checkpoint binding name must not be empty");
  }

  const auto [it, inserted] =
      checkpoint_bindings_.emplace(std::move(name), std::move(destination));
  if (!inserted) {
    throw std::invalid_argument("checkpoint binding was registered twice: "
                                + it->first);
  }
}

} // namespace nanovllm
