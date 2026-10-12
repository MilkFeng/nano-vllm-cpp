#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>
#include <memory>

#include "core/model_config.hh"
#include "core/tensor.hh"
#include "engine/sequence.hh"
#include "nanovllm/nanovllm.hh"

namespace nanovllm {

class Model;
class Qwen35Cache;

class ModelRunner {
 public:
  ModelRunner(std::shared_ptr<Logger> logger, EngineParams params,
              ModelConfig config, const std::filesystem::path& checkpoint);
  ~ModelRunner();

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  ModelRunner(ModelRunner&&) = delete;
  ModelRunner& operator=(ModelRunner&&) = delete;

  [[nodiscard]] std::size_t num_blocks() const { return num_blocks_; }
  // K/V for attention layers only; conv/recurrent state is separate.
  [[nodiscard]] std::size_t kvcache_size_bytes() const;
  [[nodiscard]] std::size_t state_cache_size_bytes() const;

  [[nodiscard]] std::vector<std::int32_t> run(
      std::span<const Sequence* const> seqs, bool prefill);

 private:
  std::shared_ptr<Logger> logger_;
  EngineParams params_;
  ModelConfig config_;

  std::size_t num_blocks_ = 0;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Qwen35Cache> hybrid_cache_;
  Tensor key_cache_;
  Tensor value_cache_;
  cudaStream_t stream_ = nullptr;
  std::uint64_t sample_step_ = 0;
};
} // namespace nanovllm
