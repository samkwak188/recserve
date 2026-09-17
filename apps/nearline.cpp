// Nearline ingest running against a live serving load.
//
// The question this answers: does the feature writer stall the readers?
//
//   --feature-store mutex     every request and every event update take the
//                             same std::mutex (the original design)
//   --feature-store snapshot  requests take an acquire load of a published
//                             table; the writer publishes on an interval
//
// Both run the identical serving workload at the identical event rate, so the
// difference in serving p99 is the cost of the lock and nothing else.
//
// Freshness is charged at visibility, not at consume: an event counts as fresh
// only once a query could read it. With a publish interval, that is bounded
// below by the interval, which is the price of not holding a lock.
#include "recserve/engine.hpp"
#include "recserve/nearline.hpp"
#include "recserve/event_source.hpp"
#include "recserve/feature_snapshot.hpp"
#include "recserve/rss.hpp"
#include <iostream>
#include <iomanip>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <fstream>

using namespace recserve;

static std::uint64_t wall_ms() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

int main(int argc, char** argv) {
  std::string source = "file", events_path = "data/events.bin", store_kind = "snapshot";
  std::string out = "data/events.bin";
  KafkaConfig kcfg;
  int publish_ms = 100, serve_threads = 4, seconds = 3, items = 16384, dim = 64, users = 4096;
  int write_log = 0, replay = 0;
  long long rate = 50'000;
  double zipf = 1.0;
  bool json = false;
  std::string dump_features;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--source" && i + 1 < argc) source = argv[++i];
    else if (a == "--events" && i + 1 < argc) events_path = argv[++i];
    else if (a == "--brokers" && i + 1 < argc) kcfg.brokers = argv[++i];
    else if (a == "--topic" && i + 1 < argc) kcfg.topic = argv[++i];
    else if (a == "--group" && i + 1 < argc) kcfg.group = argv[++i];
    else if (a == "--feature-store" && i + 1 < argc) store_kind = argv[++i];
    else if (a == "--publish-ms" && i + 1 < argc) publish_ms = std::atoi(argv[++i]);
    else if (a == "--serve-threads" && i + 1 < argc) serve_threads = std::atoi(argv[++i]);
    else if (a == "--seconds" && i + 1 < argc) seconds = std::atoi(argv[++i]);
    else if (a == "--rate" && i + 1 < argc) rate = std::atoll(argv[++i]);
    else if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--write-log" && i + 1 < argc) write_log = std::atoi(argv[++i]);
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--n" && i + 1 < argc) write_log = std::atoi(argv[++i]);
    else if (a == "--replay" && i + 1 < argc) replay = std::atoi(argv[++i]);
    else if (a == "--zipf" && i + 1 < argc) zipf = std::atof(argv[++i]);
    else if (a == "--dump-features" && i + 1 < argc) dump_features = argv[++i];
    else if (a == "--json") json = true;
  }

  // Fixture generation mode: write a log and exit.
  if (write_log > 0) {
    const auto p = std::filesystem::path(out).parent_path();
    if (!p.empty()) std::filesystem::create_directories(p);
    synth_events(write_log, users, items, 1'000'000, 1, zipf).write_file(out);
    std::cout << "wrote " << write_log << " events to " << out << "\n";
    return 0;
  }

  // ---------------------------------------------------------------- source --
  std::unique_ptr<EventSource> src;
  if (source == "kafka") {
    std::string err;
    src = make_kafka_source(kcfg, &err);
    if (!src) {
      std::cerr << "kafka source unavailable: " << err << "\n";
      return 3;
    }
  } else {
    if (!std::filesystem::exists(events_path)) {
      synth_events(200'000, users, items, 1'000'000, 1, zipf).write_file(events_path);
    }
    auto f = std::make_unique<FileEventSource>(events_path, /*loop=*/true);
    if (!f->ok()) {
      std::cerr << "cannot open " << events_path << "\n";
      return 1;
    }
    src = std::move(f);
  }

  // ----------------------------------------------------------------- replay --
  // Deterministic event-time replay used for the offline/online skew check: no
  // rate limiting, no wall clock, publish on an event-time interval. Two runs
  // over the same log produce byte-identical feature tables.
  if (replay > 0) {
    FeatureSnapshotStore rstore;
    rstore.init(items, users);
    NearlinePipeline rpipe(*src, rstore, static_cast<std::uint64_t>(publish_ms));
    std::vector<EventRecord> buf(1024);
    std::uint64_t consumed = 0, clock = 0;
    while (consumed < static_cast<std::uint64_t>(replay) && !src->eof()) {
      const std::size_t want =
          std::min<std::size_t>(buf.size(), static_cast<std::size_t>(replay) - consumed);
      const std::size_t n = src->poll(buf.data(), want, 10);
      if (n == 0) break;
      for (std::size_t i = 0; i < n; ++i) {
        clock = buf[i].event_time_ms;
        rpipe.stage(buf[i], buf[i].event_time_ms);
        rpipe.publish_if_due(clock);
      }
      consumed += n;
    }
    // Deliberately NOT flushing: whatever is still staged is the skew. A serving
    // host that is scraped between publishes sees exactly this.
    const auto g = rstore.read();
    if (!dump_features.empty()) {
      const auto dp = std::filesystem::path(dump_features).parent_path();
      if (!dp.empty()) std::filesystem::create_directories(dp);
      std::ofstream o(dump_features);
      o << "item,views,likes,ctr\n";
      for (std::size_t i = 0; i < g->items.size(); ++i) {
        if (g->items[i].views == 0) continue;
        o << i << "," << g->items[i].views << "," << g->items[i].likes << ","
          << std::setprecision(9) << g->items[i].ctr << "\n";
      }
    }
    const auto& rs = rpipe.stats();
    std::cout << "{\"mode\":\"replay\",\"consumed\":" << consumed
              << ",\"published\":" << rs.published
              << ",\"unpublished\":" << (consumed - rs.published)
              << ",\"publishes\":" << rs.publishes
              << ",\"publish_ms\":" << publish_ms
              << ",\"watermark_ms\":" << g->watermark_ms
              << ",\"generation\":" << g->generation << "}\n";
    return 0;
  }

  // ---------------------------------------------------------------- engine --
  Engine e;
  e.cfg.kernel = Kernel::Simd;
  e.cfg.use_hnsw = true;
  e.cfg.ef_search = 64;
  e.init_random(items, users, dim, 13, 512);

  FeatureSnapshotStore store;
  const bool use_snapshot = store_kind == "snapshot";
  if (use_snapshot) {
    store.init(items, users);
    e.snap = &store;
  }
  NearlinePipeline pipe(*src, store, static_cast<std::uint64_t>(publish_ms));

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> ingested{0};
  Histogram mutex_fresh_ms;  // written only by the single ingest thread

  // -------------------------------------------------------------- ingester --
  // Event times are rebased to the wall clock as they enter the pipeline, so
  // freshness measures this pipeline's own delay rather than the age of a
  // fixture file. A Kafka producer stamps real times and this does nothing.
  std::thread ingest([&]() {
    const auto t_start = wall_ms();
    std::vector<EventRecord> buf(512);
    std::uint64_t produced = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      const std::uint64_t now = wall_ms();
      const std::uint64_t elapsed = now - t_start;
      const std::uint64_t budget =
          static_cast<std::uint64_t>(static_cast<double>(rate) * static_cast<double>(elapsed) / 1000.0);
      if (produced >= budget) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      const std::size_t want = std::min<std::size_t>(buf.size(), budget - produced);
      const std::size_t n = src->poll(buf.data(), want, 10);
      if (n == 0) continue;
      for (std::size_t i = 0; i < n; ++i) {
        const auto& r = buf[i];
        const std::uint64_t ts = source == "kafka" ? r.event_time_ms : now;
        if (use_snapshot) {
          pipe.stage(r, ts);
        } else {
          e.features.apply_event(ts, r.user_id, r.item_id, r.type);
          mutex_fresh_ms.add(0.0);  // visible immediately: that is what the lock buys
        }
      }
      produced += n;
      ingested.fetch_add(n, std::memory_order_relaxed);
      if (use_snapshot) pipe.publish_if_due(now);
    }
    if (use_snapshot) pipe.flush(wall_ms());
  });

  // --------------------------------------------------------------- serving --
  std::vector<Histogram> per_thread(static_cast<std::size_t>(serve_threads));
  std::vector<std::uint64_t> served(static_cast<std::size_t>(serve_threads), 0);
  std::vector<std::thread> workers;
  const auto t0 = now_us();
  for (int t = 0; t < serve_threads; ++t) {
    workers.emplace_back([&, t]() {
      Request q;
      q.k = 10;
      q.retrieve_k = 64;
      q.timeout_us = 1'000'000;
      std::uint64_t i = 0;
      // Warm up before recording so the histogram is steady-state.
      for (int w = 0; w < 64; ++w) {
        q.user_id = static_cast<UserId>(w);
        (void)e.recommend_sync(q);
      }
      while (!stop.load(std::memory_order_relaxed)) {
        q.id = i;
        q.user_id = static_cast<UserId>((i * 2654435761u + static_cast<std::uint64_t>(t)) % users);
        const auto s = now_us();
        const auto r = e.recommend_sync(q);
        if (r.status == Status::Ok) {
          per_thread[static_cast<std::size_t>(t)].add(static_cast<double>(now_us() - s));
          ++served[static_cast<std::size_t>(t)];
        }
        ++i;
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  stop = true;
  for (auto& w : workers) w.join();
  ingest.join();
  const double elapsed_s = static_cast<double>(now_us() - t0) / 1e6;

  Histogram all;
  std::uint64_t total_served = 0;
  for (int t = 0; t < serve_threads; ++t) {
    for (double x : per_thread[static_cast<std::size_t>(t)].samples) all.add(x);
    total_served += served[static_cast<std::size_t>(t)];
  }

  const auto& ns = pipe.stats();
  const Histogram& fresh = use_snapshot ? ns.freshness_ms : mutex_fresh_ms;
  const auto ss = src->stats();
  const double serve_qps = elapsed_s > 0 ? static_cast<double>(total_served) / elapsed_s : 0;
  const double ingest_eps = elapsed_s > 0 ? static_cast<double>(ingested.load()) / elapsed_s : 0;

  if (json) {
    std::cout << std::setprecision(6) << "{"
              << "\"feature_store\":\"" << store_kind << "\""
              << ",\"source\":\"" << src->name() << "\""
              << ",\"serve_threads\":" << serve_threads
              << ",\"items\":" << items << ",\"dim\":" << dim
              << ",\"publish_ms\":" << publish_ms
              << ",\"target_rate_eps\":" << rate
              << ",\"seconds\":" << elapsed_s
              << ",\"serve_qps\":" << serve_qps
              << ",\"serve_p50_us\":" << all.percentile(0.50)
              << ",\"serve_p95_us\":" << all.percentile(0.95)
              << ",\"serve_p99_us\":" << all.percentile(0.99)
              << ",\"serve_p999_us\":" << all.percentile(0.999)
              << ",\"serve_max_us\":" << all.percentile(1.0)
              << ",\"served\":" << total_served
              << ",\"ingested\":" << ingested.load()
              << ",\"ingest_eps\":" << ingest_eps
              << ",\"publishes\":" << ns.publishes
              << ",\"published\":" << ns.published
              << ",\"freshness_p50_ms\":" << fresh.percentile(0.50)
              << ",\"freshness_p99_ms\":" << fresh.percentile(0.99)
              << ",\"publish_p99_us\":" << ns.publish_us.percentile(0.99)
              << ",\"consumer_lag\":" << ss.lag
              << ",\"source_errors\":" << ss.errors
              << ",\"rss_mib\":" << static_cast<double>(process_rss_bytes()) / (1024.0 * 1024.0)
              << "}\n";
  } else {
    std::cout << "feature_store=" << store_kind << " source=" << src->name()
              << " serve_qps=" << serve_qps << " serve_p50_us=" << all.percentile(0.50)
              << " serve_p99_us=" << all.percentile(0.99)
              << " serve_p999_us=" << all.percentile(0.999)
              << " ingest_eps=" << ingest_eps
              << " freshness_p50_ms=" << fresh.percentile(0.50)
              << " freshness_p99_ms=" << fresh.percentile(0.99)
              << " publishes=" << ns.publishes << " lag=" << ss.lag << "\n";
  }
  return 0;
}
