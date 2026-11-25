#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "core/checkpoint.hh"
#include "core/model.hh"
#include "core/model_config.hh"

namespace nanovllm {

class LoadPlan final {
 public:
  // The checkpoint and the registered parameter storage must outlive the plan.
  LoadPlan(const Checkpoint& checkpoint, const ModelConfig& config);

  void add_module(Module& module, std::string_view prefix = {});
  [[nodiscard]] ModelLoadResult apply() const;

 private:
  struct Entry {
    std::string name;
    TensorView destination;
    std::string scale_name;
  };

  void fill_cpu(const Entry& entry, TensorView output) const;

  const Checkpoint& checkpoint_;
  std::array<std::size_t, 2> fp8_block_size_;
  std::vector<Entry> entries_;
};

} // namespace nanovllm
