#pragma once
#include <cstddef>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <fstream>
#include <unistd.h>
#endif

namespace recserve {

inline std::size_t process_rss_bytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.WorkingSetSize;
  return 0;
#else
  std::ifstream in("/proc/self/statm");
  std::size_t virtual_pages = 0, resident_pages = 0;
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (page_size > 0 && in >> virtual_pages >> resident_pages)
    return resident_pages * static_cast<std::size_t>(page_size);
  return 0;
#endif
}

}  // namespace recserve
