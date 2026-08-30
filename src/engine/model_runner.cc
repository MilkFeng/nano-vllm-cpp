
#include "engine/model_runner.hh"

namespace nanovllm {

ModelRunner::ModelRunner(EngineParams params, ModelConfig config)
    : params_(params),
      config_(config) {}

std::vector<std::size_t> ModelRunner::run(std::span<const Sequence> seqs,
                                          bool prefill) {
  return {};
}

} // namespace nanovllm
