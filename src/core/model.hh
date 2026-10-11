#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "core/forward_context.hh"
#include "core/module.hh"

namespace nanovllm {

struct ModelLoadOptions {
  // Reject checkpoint tensors that were not registered by the model. Keeping
  // this strict by default catches architecture/checkpoint mismatches early.
  bool allow_unexpected_tensors = false;
};

struct ModelLoadResult {
  std::size_t num_tensors = 0;
  std::uint64_t num_bytes = 0;
};

// Base class for executable models. Module parameters provide default
// checkpoint destinations. Derived models may add explicit checkpoint bindings
// for external names or slices of packed parameters without teaching the loader
// about model architectures.
class Model : public Module {
 public:
  virtual ~Model() override = default;

  // Accepts a safetensors file, a shard index, or a model directory.
  // The default implementation copies exact dtypes. Derived models may
  // override loading to convert or dequantize their checkpoint weights.
  virtual ModelLoadResult load_ckpt(const std::filesystem::path& ckpt_path,
                                    ModelLoadOptions options = {});

  // Execution interfaces shared by the model runner. Forward returns hidden
  // states; compute_logits selects prefill's last rows and projects vocabulary.
  [[nodiscard]] virtual Tensor forward(ConstTensorView input_ids,
                                       ConstTensorView positions,
                                       const ForwardContext& context) const = 0;
  [[nodiscard]] virtual Tensor compute_logits(
      ConstTensorView hidden_states, const ForwardContext& context) const = 0;
  [[nodiscard]] virtual DataType dtype() const noexcept = 0;
  [[nodiscard]] virtual Device device() const noexcept = 0;

 protected:
  Model() = default;

  void register_checkpoint_binding(std::string name, TensorView destination);

 private:
  using CheckpointBindings = std::map<std::string, TensorView, std::less<>>;

  // Builds a fresh mapping so registered Parameters are never represented by
  // cached TensorViews. Explicit bindings override same-named defaults.
  [[nodiscard]] CheckpointBindings checkpoint_destinations();

  CheckpointBindings checkpoint_bindings_;
};

} // namespace nanovllm
