#include "recserve/engine.hpp"
#include "recserve/nearline.hpp"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>

using namespace recserve;

int main(int argc, char** argv) {
  std::string out = "data/events.bin";
  int n = 5000;
  int users = 64;
  int items = 256;
  int delay_us = 0;
  bool write_only = false;
  std::uint64_t replica_delay_ms = 15;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (a == "--delay-us" && i + 1 < argc) delay_us = std::atoi(argv[++i]);
    else if (a == "--write-log") write_only = true;
    else if (a == "--replica-delay-ms" && i + 1 < argc)
      replica_delay_ms = static_cast<std::uint64_t>(std::atoll(argv[++i]));
  }

  EventLog log;
  std::uint64_t t = 1'000'000;
  for (int i = 0; i < n; ++i) {
    EventRecord e;
    e.event_time_ms = t + static_cast<std::uint64_t>(i);
    e.user_id = static_cast<UserId>(i % users);
    e.item_id = static_cast<ItemId>(i % items);
    e.type = static_cast<std::uint8_t>(i % 7 == 0 ? 1 : 0);
    log.records.push_back(e);
  }
  std::filesystem::create_directories(std::filesystem::path(out).parent_path());
  log.write_file(out);
  if (write_only) {
    std::cout << "wrote " << n << " events to " << out << "\n";
    return 0;
  }

  FeatureStore a, b;
  a.init(users, items, true);
  b.init(users, items, true);
  a.set_delay_us(static_cast<std::uint32_t>(delay_us));
  auto now = t + static_cast<std::uint64_t>(n) + 50;
  auto fresh = log.apply(a, now);
  auto lag = replica_lag(log, replica_delay_ms);
  (void)b;
  std::cout << "events=" << n << " freshness_p50_ms=" << fresh.percentile(0.50)
            << " freshness_p99_ms=" << fresh.percentile(0.99)
            << " replica_lag_p50_ms=" << lag.p50_ms << " replica_lag_p99_ms=" << lag.p99_ms
            << " fault_delay_us=" << delay_us << "\n";
  return 0;
}
