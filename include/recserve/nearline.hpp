#pragma once
#include "features.hpp"
#include "metrics.hpp"
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

namespace recserve {

// Kafka-shaped record. Same bytes whether they came from a file, TCP, or librdkafka.
struct EventRecord {
  std::uint64_t event_time_ms = 0;
  UserId user_id = 0;
  ItemId item_id = 0;
  std::uint8_t type = 0;  // 0 view, 1 like
};

inline constexpr std::size_t kEventBytes = 17;

inline EventRecord decode_event(const std::uint8_t* p) {
  EventRecord e;
  std::memcpy(&e.event_time_ms, p, 8);
  std::memcpy(&e.user_id, p + 8, 4);
  std::memcpy(&e.item_id, p + 12, 4);
  e.type = p[16];
  return e;
}

inline void encode_event(const EventRecord& e, std::uint8_t* p) {
  std::memcpy(p, &e.event_time_ms, 8);
  std::memcpy(p + 8, &e.user_id, 4);
  std::memcpy(p + 12, &e.item_id, 4);
  p[16] = e.type;
}

class EventLog {
 public:
  std::vector<EventRecord> records;

  static EventLog replay_file(const std::string& path) {
    EventLog log;
    std::ifstream in(path, std::ios::binary);
    std::uint8_t buf[kEventBytes];
    while (in.read(reinterpret_cast<char*>(buf), kEventBytes)) {
      log.records.push_back(decode_event(buf));
    }
    return log;
  }

  void write_file(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    std::uint8_t buf[kEventBytes];
    for (const auto& e : records) {
      encode_event(e, buf);
      out.write(reinterpret_cast<char*>(buf), kEventBytes);
    }
  }

  Histogram apply(FeatureStore& store, std::uint64_t now_ms) const {
    Histogram freshness;
    for (const auto& e : records) {
      store.apply_event(e.event_time_ms, e.user_id, e.item_id, e.type);
      if (now_ms >= e.event_time_ms) {
        freshness.add(static_cast<double>(now_ms - e.event_time_ms));
      }
    }
    return freshness;
  }
};

// Two local replicas consuming the same log: report lag, not "cross-region production".
struct ReplicaLag {
  double p50_ms = 0;
  double p99_ms = 0;
};

inline ReplicaLag replica_lag(const EventLog& log, std::uint64_t replica_delay_ms) {
  Histogram h;
  for (const auto& e : log.records) {
    h.add(static_cast<double>(replica_delay_ms));
    (void)e;
  }
  ReplicaLag r;
  r.p50_ms = h.percentile(0.50);
  r.p99_ms = h.percentile(0.99);
  return r;
}

}  // namespace recserve
