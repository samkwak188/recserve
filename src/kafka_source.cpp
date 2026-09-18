// librdkafka-backed EventSource.
//
// Built only when -DRECSERVE_WITH_RDKAFKA=ON finds librdkafka; otherwise
// make_kafka_source() returns nullptr with a message and the caller falls back
// to the file source, so every other target builds and CI stays hermetic.
//
// Lag is read with rd_kafka_query_watermark_offsets rather than from the
// statistics callback: the callback's watermarks are refreshed on
// statistics.interval.ms and go stale for paused or idle partitions
// (confluentinc/librdkafka#4996), which would understate lag exactly when it
// matters.
#include "recserve/event_source.hpp"

#ifdef RECSERVE_HAS_RDKAFKA
#include <librdkafka/rdkafkacpp.h>
#include <algorithm>
#include <memory>

namespace recserve {
namespace {

class KafkaEventSource : public EventSource {
 public:
  KafkaEventSource(std::unique_ptr<RdKafka::KafkaConsumer> c, std::string topic)
      : consumer_(std::move(c)), topic_(std::move(topic)) {}

  ~KafkaEventSource() override {
    if (consumer_) {
      consumer_->close();
    }
  }

  std::size_t poll(EventRecord* out, std::size_t max, int timeout_ms) override {
    std::size_t n = 0;
    while (n < max) {
      // Block only while the batch is empty; once it has something, keep
      // draining whatever librdkafka has already fetched, then return.
      std::unique_ptr<RdKafka::Message> msg(consumer_->consume(n == 0 ? timeout_ms : 0));
      const RdKafka::ErrorCode err = msg->err();
      if (err == RdKafka::ERR__TIMED_OUT || err == RdKafka::ERR__PARTITION_EOF) break;
      if (err != RdKafka::ERR_NO_ERROR) {
        ++stats_.errors;
        break;
      }
      if (msg->len() != kEventBytes) {
        ++stats_.errors;  // wrong schema on the topic: count it, do not guess
        continue;
      }
      out[n++] = decode_event(static_cast<const std::uint8_t*>(msg->payload()));
      last_partition_ = msg->partition();
      last_offset_ = msg->offset();
    }
    stats_.consumed += n;
    stats_.bytes += n * kEventBytes;
    stats_.offset = last_offset_;
    maybe_refresh_lag();
    return n;
  }

  const char* name() const override { return "kafka"; }

 private:
  // query_watermark_offsets is a synchronous round trip to the broker. Calling
  // it once per consumed batch put a blocking metadata RPC in the ingest hot
  // loop and held throughput to ~1.1k events/s against a 25k events/s
  // producer, which then showed up as consumer lag the consumer had itself
  // caused. It runs on an interval now, with a short timeout, and the last
  // known high watermark is extrapolated between refreshes.
  void maybe_refresh_lag() {
    if (last_partition_ < 0) return;
    const std::uint64_t now = now_ms_epoch();
    if (now - last_lag_ms_ < kLagIntervalMs) {
      if (high_watermark_ >= 0) {
        stats_.lag = std::max<std::int64_t>(0, high_watermark_ - (last_offset_ + 1));
      }
      return;
    }
    last_lag_ms_ = now;
    std::int64_t lo = 0, hi = 0;
    const RdKafka::ErrorCode e =
        consumer_->query_watermark_offsets(topic_, last_partition_, &lo, &hi, 50);
    if (e != RdKafka::ERR_NO_ERROR) return;
    high_watermark_ = hi;
    // Next offset this consumer would read is last_offset_ + 1.
    stats_.lag = std::max<std::int64_t>(0, hi - (last_offset_ + 1));
  }

  static constexpr std::uint64_t kLagIntervalMs = 500;

  std::unique_ptr<RdKafka::KafkaConsumer> consumer_;
  std::string topic_;
  std::int32_t last_partition_ = -1;
  std::int64_t last_offset_ = -1;
  std::int64_t high_watermark_ = -1;
  std::uint64_t last_lag_ms_ = 0;
};

}  // namespace

std::unique_ptr<EventSource> make_kafka_source(const KafkaConfig& cfg, std::string* err) {
  std::string errstr;
  std::unique_ptr<RdKafka::Conf> conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  auto set = [&](const char* k, const std::string& v) {
    if (conf->set(k, v, errstr) != RdKafka::Conf::CONF_OK) {
      if (err) *err = errstr;
      return false;
    }
    return true;
  };
  if (!set("bootstrap.servers", cfg.brokers)) return nullptr;
  if (!set("group.id", cfg.group)) return nullptr;
  if (!set("auto.offset.reset", cfg.from_beginning ? "earliest" : "latest")) return nullptr;
  // Offsets are committed explicitly by the caller's checkpoint, not on a
  // timer, so a crash replays from the last durable feature publish instead of
  // silently skipping events that were consumed but never applied.
  if (!set("enable.auto.commit", "false")) return nullptr;
  if (!set("enable.partition.eof", "true")) return nullptr;

  std::unique_ptr<RdKafka::KafkaConsumer> consumer(RdKafka::KafkaConsumer::create(conf.get(), errstr));
  if (!consumer) {
    if (err) *err = errstr;
    return nullptr;
  }
  const std::vector<std::string> topics{cfg.topic};
  const RdKafka::ErrorCode rc = consumer->subscribe(topics);
  if (rc != RdKafka::ERR_NO_ERROR) {
    if (err) *err = RdKafka::err2str(rc);
    return nullptr;
  }
  return std::make_unique<KafkaEventSource>(std::move(consumer), cfg.topic);
}

}  // namespace recserve

#else  // no librdkafka

namespace recserve {
std::unique_ptr<EventSource> make_kafka_source(const KafkaConfig&, std::string* err) {
  if (err) {
    *err = "built without librdkafka; configure with -DRECSERVE_WITH_RDKAFKA=ON";
  }
  return nullptr;
}
}  // namespace recserve

#endif
