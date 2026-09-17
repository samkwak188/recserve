#include "recserve/engine.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <string>

using namespace recserve;

static std::size_t rss_bytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.WorkingSetSize;
  return 0;
#else
  std::ifstream in("/proc/self/statm");
  std::size_t pages = 0;
  if (in >> pages) return pages * 4096;
  return 0;
#endif
}

int main(int argc, char** argv) {
  int seconds = 3;
  int items = 2048;
  int dim = 32;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--seconds" && i + 1 < argc) seconds = std::atoi(argv[++i]);
    else if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
  }
  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.kernel = Kernel::Simd;
  e.init_random(items, 64, dim, 17);
  auto start = std::chrono::steady_clock::now();
  std::size_t max_rss = rss_bytes();
  std::size_t start_rss = max_rss;
  std::uint64_t n = 0;
  Request q;
  q.k = 10;
  q.retrieve_k = 64;
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(seconds)) {
    q.id = n++;
    q.user_id = static_cast<UserId>(n % 64);
    (void)e.recommend_sync(q);
    if ((n & 255) == 0) max_rss = std::max(max_rss, rss_bytes());
  }
  max_rss = std::max(max_rss, rss_bytes());
  std::cout << "requests=" << n << " start_rss=" << start_rss << " max_rss=" << max_rss
            << " delta=" << (max_rss > start_rss ? max_rss - start_rss : 0) << "\n";
  // 24h soak is `--seconds 86400`. CI uses a few seconds and fails only on a huge leak.
  if (max_rss > start_rss + (64ull << 20) && seconds <= 10) {
    std::cerr << "soak rss grew more than 64MiB on a short run\n";
    return 2;
  }
  return 0;
}
