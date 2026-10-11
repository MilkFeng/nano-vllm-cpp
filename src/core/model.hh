#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "core/forward_context.hh"
#include "core/module.hh"

namespace nanovllm {

struct ModelLoadResult {
  std::size_t num_tensors = 0;
  std::uint64_t num_bytes = 0;
};

class Model : public Module {
 public:
  virtual ~Model() override = default;

  virtual ModelLoadResult load_ckpt(const std::filesystem::path& ckpt_path);

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

  [[nodiscard]] CheckpointBindings checkpoint_destinations();

  CheckpointBindings checkpoint_bindings_;
};

} // namespace nanovllm
