#pragma once
// Nearline feature pipeline: events in, published feature snapshots out.
//
// Two different staleness numbers are reported because they fail independently:
//   consumer lag  = high_watermark - next offset to read. Are we behind the
//                   log? Goes up when ingest cannot keep up with production.
//   freshness     = publish_time - event_time, measured at the moment the
//                   feature becomes readable by a query. Goes up when the
//                   publish interval is long, even at zero lag.
// A pipeline can be caught up on offsets and still serve minutes-old features.
#include "event_source.hpp"
#include "feature_snapshot.hpp"
#include "features.hpp"
#include "metrics.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace recserve {

// Fixed-width tumbling buckets read as a sliding window. O(1) per event and
// O(buckets) per read, which is what a per-item counter can afford when there
// are millions of items.
class SlidingWindowCounter {
 public:
  SlidingWindowCounter(int n_buckets = 60, std::uint64_t bucket_ms = 1000)
      : bucket_ms_(bucket_ms), views_(static_cast<std::size_t>(n_buckets), 0),
        likes_(static_cast<std::size_t>(n_buckets), 0),
        stamp_(static_cast<std::size_t>(n_buckets), 0) {}

  void add(std::uint64_t event_ms, bool like) {
    const std::uint64_t b = event_ms / bucket_ms_;
    const std::size_t i = static_cast<std::size_t>(b % views_.size());
    if (stamp_[i] != b) {  // bucket wrapped: this slot is now a new interval
      stamp_[i] = b;
      views_[i] = 0;
      likes_[i] = 0;
    }
    views_[i] += 1;
    likes_[i] += like ? 1 : 0;
  }

  // CTR over the buckets that fall inside [now - window, now].
  float ctr(std::uint64_t now_ms, std::uint64_t window_ms) const {
    const std::uint64_t newest = now_ms / bucket_ms_;
    const std::uint64_t oldest = window_ms / bucket_ms_ >= newest ? 0 : newest - window_ms / bucket_ms_;
    std::uint64_t v = 0, l = 0;
    for (std::size_t i = 0; i < views_.size(); ++i) {
      if (stamp_[i] >= oldest && stamp_[i] <= newest) {
        v += views_[i];
        l += likes_[i];
      }
    }
    return v ? static_cast<float>(l) / static_cast<float>(v) : 0.f;
  }

  std::uint64_t total_views() const {
    std::uint64_t v = 0;
    for (auto x : views_) v += x;
    return v;
  }

 private:
  std::uint64_t bucket_ms_;
  std::vector<std::uint32_t> views_, likes_;
  std::vector<std::uint64_t> stamp_;
};

// Batch file of 17-byte records. Used for fixtures, tests and the CI path that
// runs the whole pipeline without a broker.
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

struct NearlineStats {
  std::uint64_t consumed = 0;
  std::uint64_t published = 0;
  std::uint64_t publishes = 0;
  std::int64_t lag = -1;
  Histogram freshness_ms;   // publish_time - event_time, at visibility
  Histogram publish_us;     // cost of one publish
  std::uint64_t watermark_ms = 0;
};

// Drives an EventSource into a FeatureSnapshotStore on a publish interval.
// `clock_ms` is injectable so tests and replays can run on event time rather
// than wall time.
class NearlinePipeline {
 public:
  NearlinePipeline(EventSource& src, FeatureSnapshotStore& store, std::uint64_t publish_interval_ms)
      : src_(src), store_(store), interval_(publish_interval_ms) {}

  // Stage one record. `ts` is the event time freshness is measured against;
  // callers replaying a fixture at a live rate pass the wall clock instead of
  // the file's own timestamp.
  void stage(const EventRecord& e, std::uint64_t ts) {
    store_.stage(FeatureDelta{ts, e.user_id, e.item_id, e.type});
    staged_times_.push_back(ts);
    ++stats_.consumed;
  }

  // Publish if the interval has elapsed. Safe to call as often as you like.
  bool publish_if_due(std::uint64_t clock_ms) {
    if (last_publish_ms_ == 0) last_publish_ms_ = clock_ms;
    if (clock_ms - last_publish_ms_ < interval_ || store_.pending() == 0) return false;
    do_publish(clock_ms);
    stats_.lag = src_.stats().lag;
    return true;
  }

  // Poll the source, stage everything it returns, then publish if due.
  std::size_t step(std::uint64_t clock_ms, std::size_t batch = 1024) {
    buf_.resize(batch);
    const std::size_t n = src_.poll(buf_.data(), batch, 50);
    for (std::size_t i = 0; i < n; ++i) stage(buf_[i], buf_[i].event_time_ms);
    publish_if_due(clock_ms);
    stats_.lag = src_.stats().lag;
    return n;
  }

  void flush(std::uint64_t clock_ms) {
    if (store_.pending() > 0) do_publish(clock_ms);
    stats_.lag = src_.stats().lag;
  }

  const NearlineStats& stats() const { return stats_; }

 private:
  void do_publish(std::uint64_t clock_ms) {
    const auto t0 = now_us();
    const std::size_t n = store_.publish();
    stats_.publish_us.add(static_cast<double>(now_us() - t0));
    // Freshness is charged at the instant the feature becomes visible, which is
    // the publish, not the consume.
    for (std::uint64_t ts : staged_times_) {
      if (clock_ms >= ts) stats_.freshness_ms.add(static_cast<double>(clock_ms - ts));
      stats_.watermark_ms = std::max(stats_.watermark_ms, ts);
    }
    staged_times_.clear();
    stats_.published += n;
    ++stats_.publishes;
    last_publish_ms_ = clock_ms;
  }

  EventSource& src_;
  FeatureSnapshotStore& store_;
  std::uint64_t interval_;
  std::uint64_t last_publish_ms_ = 0;
  std::vector<EventRecord> buf_;
  std::vector<std::uint64_t> staged_times_;
  NearlineStats stats_;
};

// Deterministic synthetic event stream for fixtures and CI.
inline EventLog synth_events(int n, int users, int items, std::uint64_t t0 = 1'000'000,
                             std::uint64_t step_ms = 1) {
  EventLog log;
  log.records.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    EventRecord e;
    e.event_time_ms = t0 + static_cast<std::uint64_t>(i) * step_ms;
    e.user_id = static_cast<UserId>(i % users);
    e.item_id = static_cast<ItemId>((i * 7) % items);
    e.type = static_cast<std::uint8_t>(i % 7 == 0 ? 1 : 0);
    log.records.push_back(e);
  }
  return log;
}

}  // namespace recserve
