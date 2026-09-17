#include "recserve/diagnose.hpp"
#include <iostream>
#include <cstdlib>
#include <string>

using namespace recserve;

int main(int argc, char** argv) {
  DiagnoseInput in;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--p99" && i + 1 < argc) in.p99_us = std::atof(argv[++i]);
    else if (a == "--queue-p99" && i + 1 < argc) in.queue_p99_us = std::atof(argv[++i]);
    else if (a == "--score-p99" && i + 1 < argc) in.score_p99_us = std::atof(argv[++i]);
    else if (a == "--feature-p99" && i + 1 < argc) in.feature_p99_us = std::atof(argv[++i]);
    else if (a == "--ipc" && i + 1 < argc) in.ipc = std::atof(argv[++i]);
    else if (a == "--cache-miss-k" && i + 1 < argc) in.cache_miss_per_k = std::atof(argv[++i]);
    else if (a == "--late-rate" && i + 1 < argc) in.late_rate = std::atof(argv[++i]);
    else if (a == "--loadshed-rate" && i + 1 < argc) in.loadshed_rate = std::atof(argv[++i]);
  }
  auto b = classify(in);
  std::cout << bottleneck_name(b) << "\n";
  std::cout << "observe: p99=" << in.p99_us << " queue=" << in.queue_p99_us
            << " score=" << in.score_p99_us << " feature=" << in.feature_p99_us << "\n";
  std::cout << "verify: rerun recserve_bench after changing the implied knob "
               "(workers, retrieve_k, feature delay, or arrival rate).\n";
  return 0;
}
