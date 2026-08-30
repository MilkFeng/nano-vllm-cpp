#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include "weights/tensor.hh"

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

// Base class for executable models that own their parameter storage.
//
// Derived models register checkpoint tensor names together with the destination
// views they own. A destination may be a slice of a packed parameter, allowing
// checkpoint tensors such as q_proj/k_proj/v_proj to be loaded directly into a
// single QKV allocation without teaching ModelLoader about model architectures.
class Model {
 public:
  using Parameters = std::map<std::string, TensorView, std::less<>>;

  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  [[nodiscard]] const Parameters& parameters() const noexcept {
    return parameters_;
  }

  [[nodiscard]] bool has_parameter(std::string_view name) const noexcept {
    return parameters_.contains(name);
  }

  [[nodiscard]] TensorView parameter(std::string_view name) const;

  [[nodiscard]] ModelLoadResult load_ckpt(
      const std::filesystem::path& ckpt_path,
      ModelLoadOptions options = {}) const;

 protected:
  Model() = default;

  void register_parameter(std::string name, TensorView destination);

 private:
  Parameters parameters_;
};

} // namespace nanovllm
