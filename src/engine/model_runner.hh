#pragma once

#include "engine/sequence.hh"
#include "model/model_config.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

class ModelRunner {
 public:
  ModelRunner(EngineParams params, ModelConfig config);
  ~ModelRunner() = default;

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  ModelRunner(ModelRunner&&) = default;
  ModelRunner& operator=(ModelRunner&&) = default;

  [[nodiscard]] std::size_t num_blocks() const { return num_blocks_; }

  [[nodiscard]] std::vector<std::size_t> run(std::span<const Sequence> seqs,
                                             bool prefill);

 private:
 private:
  EngineParams params_;
  ModelConfig config_;

  std::size_t num_blocks_;
};
} // namespace nanovllm
