#pragma once
#include "metrics.hpp"
#include <string>

namespace recserve {

enum class Bottleneck {
  CpuBoundScoring,
  CacheMissLikely,
  Queueing,
  FeatureStoreWait,
  CoordinatedOmissionTail,
  Unknown
};

inline const char* bottleneck_name(Bottleneck b) {
  switch (b) {
    case Bottleneck::CpuBoundScoring:
      return "cpu_bound_scoring";
    case Bottleneck::CacheMissLikely:
      return "cache_miss_likely";
    case Bottleneck::Queueing:
      return "queueing";
    case Bottleneck::FeatureStoreWait:
      return "feature_store_wait";
    case Bottleneck::CoordinatedOmissionTail:
      return "coordinated_omission_or_overload";
    default:
      return "unknown";
  }
}

struct DiagnoseInput {
  double p99_us = 0;
  double queue_p99_us = 0;
  double score_p99_us = 0;
  double feature_p99_us = 0;
  double ipc = 0;          // from perf stat if present; 0 = unknown
  double cache_miss_per_k = 0;
  double late_rate = 0;
  double loadshed_rate = 0;
};

inline Bottleneck classify(const DiagnoseInput& in) {
  if (in.loadshed_rate > 0.05 || in.late_rate > 0.10) return Bottleneck::CoordinatedOmissionTail;
  if (in.feature_p99_us > in.score_p99_us && in.feature_p99_us > 0.4 * in.p99_us)
    return Bottleneck::FeatureStoreWait;
  if (in.queue_p99_us > in.score_p99_us && in.queue_p99_us > 0.4 * in.p99_us)
    return Bottleneck::Queueing;
  if (in.ipc > 0 && in.ipc < 0.8 && in.cache_miss_per_k > 20) return Bottleneck::CacheMissLikely;
  if (in.score_p99_us > 0.5 * in.p99_us) return Bottleneck::CpuBoundScoring;
  return Bottleneck::Unknown;
}

}  // namespace recserve
