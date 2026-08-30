#include "common/hasher.hh"

#include <xxhash.h>

#include <bit>
#include <new>

namespace nanovllm {

class HasherState {
 public:
  HasherState()
      : state_(XXH64_createState(), XXH64_freeState) {
    if (!state_) {
      throw std::bad_alloc();
    }
  }

  ~HasherState() = default;

  XXH64_state_t& state() { return *state_; }
  const XXH64_state_t& state() const { return *state_; }

 private:
  using StatePtr = std::unique_ptr<XXH64_state_t, decltype(&XXH64_freeState)>;
  StatePtr state_;
};

Hasher::Hasher()
    : state_(std::make_unique<HasherState>()) {
  XXH64_reset(&state_->state(), 0);
}

Hasher::~Hasher() = default;

void Hasher::update(std::uint64_t value) {
  if constexpr (std::endian::native == std::endian::big) {
    value = std::byteswap(value);
  }
  XXH64_update(&state_->state(), &value, sizeof(value));
}

std::uint64_t Hasher::digest() const {
  return XXH64_digest(&state_->state());
}

} // namespace nanovllm
