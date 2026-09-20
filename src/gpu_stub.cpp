#include "recserve/gpu.hpp"
namespace recserve {
std::unique_ptr<GpuScorer> make_gpu_scorer(const float*, int, int, int, std::string* error) {
  if (error) *error = "CUDA is unavailable: configure with RECSERVE_WITH_CUDA=ON";
  return nullptr;
}
}  // namespace recserve
