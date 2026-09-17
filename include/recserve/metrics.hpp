#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cmath>
#include <numeric>

namespace recserve {

struct Histogram {
  std::vector<double> samples;

  void add(double x) { samples.push_back(x); }
  void clear() { samples.clear(); }

  double percentile(double p) const {
    if (samples.empty()) return 0;
    std::vector<double> s = samples;
    std::sort(s.begin(), s.end());
    double idx = p * static_cast<double>(s.size() - 1);
    auto lo = static_cast<std::size_t>(idx);
    auto hi = std::min(lo + 1, s.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[hi] * frac;
  }

  double mean() const {
    if (samples.empty()) return 0;
    return std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  }

  double cv() const {
    if (samples.size() < 2) return 0;
    double m = mean();
    double v = 0;
    for (double x : samples) v += (x - m) * (x - m);
    v /= static_cast<double>(samples.size() - 1);
    return m == 0 ? 0 : std::sqrt(v) / m;
  }

  std::uint64_t n() const { return samples.size(); }
};

struct ServeStats {
  Histogram latency_us;
  Histogram queue_us;
  Histogram score_us;
  Histogram feature_us;
  Histogram freshness_ms;
  std::uint64_t ok = 0;
  std::uint64_t timeout = 0;
  std::uint64_t loadshed = 0;
  std::uint64_t late = 0;  // completed after intended start+timeout in open-loop
  std::uint64_t sent = 0;
};

}  // namespace recserve
