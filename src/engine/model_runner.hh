#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>
#include <memory>

#include "core/model_config.hh"
#include "core/tensor.hh"
#include "engine/sequence.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

class Qwen3ForCausalLM;

class ModelRunner {
 public:
  ModelRunner(EngineParams params, ModelConfig config,
              const std::filesystem::path& checkpoint);
  ~ModelRunner();

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  ModelRunner(ModelRunner&&) = delete;
  ModelRunner& operator=(ModelRunner&&) = delete;

  [[nodiscard]] std::size_t num_blocks() const { return num_blocks_; }
  // Combined storage for K and V across all decoder layers.
  [[nodiscard]] std::size_t kvcache_size_bytes() const {
    return key_cache_.size_bytes() + value_cache_.size_bytes();
  }

  [[nodiscard]] std::vector<std::int32_t> run(
      std::span<const Sequence* const> seqs, bool prefill);

 private:
  EngineParams params_;
  ModelConfig config_;

  std::size_t num_blocks_ = 0;
  std::unique_ptr<Qwen3ForCausalLM> model_;
  Tensor key_cache_;
  Tensor value_cache_;
  cudaStream_t stream_ = nullptr;
  std::uint64_t sample_step_ = 0;
};
} // namespace nanovllm
