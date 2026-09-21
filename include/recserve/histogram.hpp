#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace recserve {
// Fixed storage and one bucket increment per observation. Scrapes are not an
// atomic cross-counter snapshot; quiescent/delta campaigns avoid that ambiguity.
class DurationHistogram {
 public:
  static constexpr std::array<std::uint64_t, 18> limits{
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
      8192, 16384, 32768, 65536, 131072};
  void observe(std::uint64_t us) {
    std::size_t i = 0;
    while (i < limits.size() && us > limits[i]) ++i;
    buckets_[i].fetch_add(1, std::memory_order_relaxed);
    sum_.fetch_add(us, std::memory_order_relaxed);
  }
  void write(std::ostream& out, std::string_view name) const {
    std::uint64_t count = 0;
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
      count += buckets_[i].load(std::memory_order_relaxed);
      out << name << "_bucket{le=\"";
      if (i < limits.size()) out << limits[i]; else out << "+Inf";
      out << "\"} " << count << '\n';
    }
    out << name << "_count " << count << '\n'
        << name << "_sum " << sum_.load(std::memory_order_relaxed) << '\n';
  }
 private:
  std::array<std::atomic<std::uint64_t>, limits.size() + 1> buckets_{};
  std::atomic<std::uint64_t> sum_{0};
};
}  // namespace recserve
