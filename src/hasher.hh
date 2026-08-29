
#pragma once

#include <cstdint>
#include <memory>

namespace nanovllm {

class HasherState;

class Hasher {
 public:
  Hasher();
  ~Hasher();

  Hasher(const Hasher&) = delete;
  Hasher& operator=(const Hasher&) = delete;

  Hasher(Hasher&&) = default;
  Hasher& operator=(Hasher&&) = default;

  void update(std::uint64_t value);
  [[nodiscard]] std::uint64_t digest() const;

 private:
  std::unique_ptr<HasherState> state_;
};

} // namespace nanovllm
