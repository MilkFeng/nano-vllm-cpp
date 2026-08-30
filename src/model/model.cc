#include "model/model.hh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <format>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include <stdexcept>
#include <string_view>

#include "weights/safetensors.hh"

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
  const auto status = cudaMemcpy(destination.data(), bytes.data(), bytes.size(),
                                 cudaMemcpyDefault);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::format("failed to copy tensor '{}' to its model destination: {}",
                    name, cudaGetErrorString(status)));
  }
}

} // namespace

TensorView Model::parameter(std::string_view name) const {
  const auto it = parameters_.find(name);
  if (it == parameters_.end()) {
    throw std::out_of_range("model parameter was not registered: "
                            + std::string(name));
  }
  return it->second;
}

ModelLoadResult Model::load_ckpt(const std::filesystem::path& ckpt_path,
                                 ModelLoadOptions options) const {
  Safetensors ckpt;
  ckpt.open(ckpt_path);

  const auto& parameters = this->parameters();
  const auto& checkpoint_tensors = ckpt.tensor_map().tensors;

  // Validate the complete mapping before copying anything, so a missing or
  // incompatible tensor cannot leave the model partially initialized.
  for (const auto& [name, destination] : parameters) {
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
      (void)source;
      if (!this->has_parameter(name)) {
        throw std::runtime_error(
            std::format("checkpoint '{}' contains unexpected tensor '{}'",
                        ckpt_path.string(), name));
      }
    }
  }

  ModelLoadResult result;
  for (const auto& [name, destination] : parameters) {
    const auto& source = checkpoint_tensors.at(name);
    copy_parameter(name, destination, ckpt, source);

    if (source.byte_size()
        > std::numeric_limits<std::uint64_t>::max() - result.num_bytes) {
      throw std::overflow_error("loaded model byte count is too large");
    }
    ++result.num_tensors;
    result.num_bytes += source.byte_size();
  }

  return result;
}

void Model::register_parameter(std::string name, TensorView destination) {
  if (name.empty()) {
    throw std::invalid_argument("model parameter name must not be empty");
  }

  const auto [it, inserted] =
      parameters_.emplace(std::move(name), std::move(destination));
  if (!inserted) {
    throw std::invalid_argument("model parameter was registered twice: "
                                + it->first);
  }
}

} // namespace nanovllm
