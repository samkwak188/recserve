#pragma once
// Where nearline events come from. The 17-byte record is identical whether it
// arrived from a file or from Kafka, so the windowing, CTR and freshness logic
// never knows which one it is reading -- and CI can exercise the whole nearline
// path without a broker.
//
// Consumer lag here is the standard Kafka definition: high_watermark - the next
// offset this consumer would read, per partition, as described in
// github.com/confluentinc/librdkafka/wiki/Consumer-lag-monitoring. It is a
// different quantity from freshness (now - event_time) and both are reported,
// because a consumer can be caught up on offsets while still serving stale
// features, and vice versa.
#include "types.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace recserve {

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

struct SourceStats {
  std::uint64_t consumed = 0;
  std::uint64_t bytes = 0;
  std::int64_t lag = -1;      // -1 = not applicable (file source)
  std::int64_t offset = -1;
  std::uint64_t errors = 0;
};

class EventSource {
 public:
  virtual ~EventSource() = default;
  // Fills up to `max` records. Returns how many were written. 0 means "nothing
  // available right now", not end of stream, unless eof() is also true.
  virtual std::size_t poll(EventRecord* out, std::size_t max, int timeout_ms) = 0;
  virtual bool eof() const { return false; }
  virtual SourceStats stats() const { return stats_; }
  virtual const char* name() const = 0;

 protected:
  SourceStats stats_;
};

class FileEventSource : public EventSource {
 public:
  explicit FileEventSource(const std::string& path, bool loop = false)
      : in_(path, std::ios::binary), loop_(loop), path_(path) {}

  bool ok() const { return static_cast<bool>(in_); }

  std::size_t poll(EventRecord* out, std::size_t max, int) override {
    std::size_t n = 0;
    std::uint8_t buf[kEventBytes];
    while (n < max) {
      if (!in_.read(reinterpret_cast<char*>(buf), kEventBytes)) {
        if (!loop_) {
          eof_ = true;
          break;
        }
        in_.clear();
        in_.seekg(0);
        ++wraps_;
        if (!in_.read(reinterpret_cast<char*>(buf), kEventBytes)) {
          eof_ = true;
          break;
        }
      }
      out[n++] = decode_event(buf);
    }
    stats_.consumed += n;
    stats_.bytes += n * kEventBytes;
    stats_.offset = static_cast<std::int64_t>(stats_.consumed);
    return n;
  }

  bool eof() const override { return eof_; }
  const char* name() const override { return "file"; }
  std::uint64_t wraps() const { return wraps_; }

 private:
  std::ifstream in_;
  bool loop_ = false;
  bool eof_ = false;
  std::uint64_t wraps_ = 0;
  std::string path_;
};

// Built only with -DRECSERVE_WITH_RDKAFKA=ON and librdkafka present.
// Implementation in src/kafka_source.cpp.
struct KafkaConfig {
  std::string brokers = "127.0.0.1:19092";
  std::string topic = "interactions";
  std::string group = "recserve-nearline";
  bool from_beginning = true;
};

std::unique_ptr<EventSource> make_kafka_source(const KafkaConfig& cfg, std::string* err);

inline bool kafka_available() {
#ifdef RECSERVE_HAS_RDKAFKA
  return true;
#else
  return false;
#endif
}

}  // namespace recserve
