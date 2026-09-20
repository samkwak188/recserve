#pragma once
#include "types.hpp"
#include <memory>
#include <string>

namespace recserve {
struct GpuTiming {
  double h2d_ms = 0, compute_ms = 0, d2h_ms = 0, total_ms = 0;
};

// The catalog is immutable for this object's lifetime. Calls are serialized;
// use independent scorer instances to run independent execution streams.
class GpuScorer {
 public:
  virtual ~GpuScorer() = default;
  virtual void topk(const float* queries, int batch, int k, ScoredItem* output,
                    GpuTiming* timing = nullptr) = 0;
  virtual std::size_t device_bytes() const = 0;
  virtual std::string device_name() const = 0;
};

std::unique_ptr<GpuScorer> make_gpu_scorer(const float* catalog, int items, int dim,
                                         int max_batch, std::string* error = nullptr);
}  // namespace recserve
