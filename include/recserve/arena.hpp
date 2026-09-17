#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <new>

namespace recserve {

// Per-thread bump allocator. Reset at the start of each request.
class Arena {
 public:
  explicit Arena(std::size_t cap = 1 << 16) : buf_(cap, 0), off_(0) {}

  void reset() { off_ = 0; }

  void* alloc(std::size_t n, std::size_t align = alignof(std::max_align_t)) {
    std::size_t aligned = (off_ + (align - 1)) & ~(align - 1);
    if (aligned + n > buf_.size()) {
      std::size_t need = aligned + n;
      buf_.resize(need + need / 2, 0);
    }
    void* p = buf_.data() + aligned;
    off_ = aligned + n;
    return p;
  }

  template <class T>
  T* alloc_n(std::size_t count) {
    return static_cast<T*>(alloc(sizeof(T) * count, alignof(T)));
  }

  std::size_t used() const { return off_; }

 private:
  std::vector<unsigned char> buf_;
  std::size_t off_;
};

inline Arena& thread_arena() {
  thread_local Arena a;
  return a;
}

}  // namespace recserve
