#include "recserve/engine.hpp"
#include "recserve/quality.hpp"
#include "recserve/nearline.hpp"
#include "recserve/diagnose.hpp"
#include "recserve/protocol.hpp"
#include "recserve/arena.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <random>
#include <cstdio>
#include <thread>

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond << "\n"; \
      throw std::runtime_error("check failed");                                     \
    }                                                                               \
  } while (0)

using namespace recserve;

static void test_dot_and_normalize() {
  float a[4] = {3.f, 0.f, 4.f, 0.f};
  float n = l2_normalize(a, 4);
  CHECK(std::fabs(n - 5.f) < 1e-5f);
  CHECK(std::fabs(dot_scalar(a, a, 4) - 1.f) < 1e-5f);
}

static void test_arena() {
  Arena a(64);
  auto* p = a.alloc_n<int>(4);
  p[0] = 1;
  p[3] = 4;
  CHECK(a.used() >= 16);
  a.reset();
  CHECK(a.used() == 0);
}

static void test_protocol() {
  Request q;
  q.id = 99;
  q.user_id = 7;
  q.k = 10;
  q.retrieve_k = 50;
  q.timeout_us = 20000;
  auto bytes = encode_request(q);
  Request q2;
  CHECK(decode_request(bytes.data(), bytes.size(), q2));
  CHECK(q2.id == 99 && q2.user_id == 7 && q2.k == 10);

  Response r;
  r.id = 99;
  r.status = Status::Ok;
  r.items.push_back({3, 0.5f});
  auto rb = encode_response(r);
  Response r2;
  CHECK(decode_response(rb.data(), rb.size(), r2));
  CHECK(r2.items.size() == 1 && r2.items[0].id == 3);
}

static void test_topk_and_timeout() {
  Engine e;
  e.cfg.use_hnsw = false;
  e.cfg.workers = 2;
  e.init_random(64, 16, 16, 1);
  e.start_pool();
  Request q;
  q.id = 1;
  q.user_id = 2;
  q.k = 5;
  q.retrieve_k = 16;
  q.timeout_us = 1'000'000;
  auto r = e.recommend_sync(q);
  CHECK(r.status == Status::Ok);
  CHECK(r.items.size() == 5);
  for (std::size_t i = 1; i < r.items.size(); ++i) CHECK(r.items[i - 1].score >= r.items[i].score);

  Request t = q;
  t.timeout_us = 1;
  e.max_queue = 1;
  e.cfg.feature_delay_us = 0;
  auto f1 = e.recommend_async(t);
  auto r1 = f1.get();
  CHECK(r1.status == Status::Ok || r1.status == Status::Timeout || r1.status == Status::LoadShed);
  e.stop_pool();
}

static void test_brute_vs_self() {
  Engine e;
  e.cfg.use_hnsw = false;
  e.init_random(32, 8, 8, 2);
  const float* q = e.cat.aos_item(0);
  auto a = e.index.brute(e.cat, q, nullptr, 5, Kernel::Scalar);
  auto b = e.index.brute(e.cat, q, nullptr, 5, Kernel::Simd);
  CHECK(a.size() == 5);
  CHECK(a[0].id == b[0].id);
  CHECK(a[0].id == 0);  // an item is its own nearest neighbour under inner product
}

// The SIMD int8 dot must agree exactly with the integer reference. A mismatch
// here is a silent wrong-answer bug, not a slow path.
static void test_int8_kernel_exactness() {
  std::mt19937 rng(9);
  for (int dim : {16, 32, 64, 65, 96, 127}) {
    std::vector<std::int8_t> a(static_cast<std::size_t>(dim)), b(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
      a[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127);
      b[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127);
    }
    const int ref = dot_i8_scalar(a.data(), b.data(), dim);
    CHECK(dot_i8(a.data(), b.data(), dim) == ref);
  }
}

// Quantization is lossy by design; bound the loss instead of asserting equality.
static void test_quantized_query_error() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int dim = 64;
  std::vector<float> x(dim), y(dim);
  for (int i = 0; i < dim; ++i) { x[i] = nd(rng); y[i] = nd(rng); }
  l2_normalize(x.data(), dim);
  l2_normalize(y.data(), dim);

  Catalog cat;
  cat.resize(1, dim);
  std::copy(y.begin(), y.end(), cat.aos_item(0));
  cat.quantize_i8();
  QuantizedQuery qq;
  qq.set(x.data(), dim);

  const float exact = dot_simd(x.data(), y.data(), dim);
  const float approx = cat.dot_item(x.data(), &qq, 0, Kernel::Int8);
  CHECK(std::fabs(exact - approx) < 0.01f);
}

// Blocked (AoSoA) scoring must return the same scores as the AoS SIMD kernel.
static void test_blocked_matches_simd() {
  Engine e;
  e.cfg.use_hnsw = false;
  e.cfg.kernel = Kernel::Blocked;
  e.init_random(301, 8, 64, 6);  // not a multiple of kBlock: exercises the tail
  const float* q = e.cat.aos_item(7);
  auto ref = e.index.brute(e.cat, q, nullptr, 10, Kernel::Simd);
  auto got = e.index.brute(e.cat, q, nullptr, 10, Kernel::Blocked);
  CHECK(ref.size() == got.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    CHECK(ref[i].id == got[i].id);
    CHECK(std::fabs(ref[i].score - got[i].score) < 1e-4f);
  }
}

// The strided SoA layout is slow but must still be correct.
static void test_soa_strided_matches_simd() {
  Engine e;
  e.cfg.use_hnsw = false;
  e.cfg.kernel = Kernel::SoaStrided;
  e.init_random(128, 8, 32, 8);
  const float* q = e.cat.aos_item(3);
  auto ref = e.index.brute(e.cat, q, nullptr, 5, Kernel::Simd);
  auto got = e.index.brute(e.cat, q, nullptr, 5, Kernel::SoaStrided);
  for (std::size_t i = 0; i < ref.size(); ++i) CHECK(ref[i].id == got[i].id);
}

static void test_visited_set_epochs() {
  VisitedSet v;
  v.resize(8);
  v.next_epoch();
  CHECK(!v.test_and_set(3));
  CHECK(v.test_and_set(3));
  v.next_epoch();
  CHECK(!v.test_and_set(3));  // new epoch clears without touching memory
}

// The graph must actually navigate: a random graph would not reach this recall.
static void test_nsw_recall() {
  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.hnsw_m = 16;
  e.cfg.ef_construction = 64;
  e.cfg.ef_search = 64;
  e.cfg.build_threads = 1;
  e.init_random(4000, 256, 32, 21);
  const double r = e.measure_retrieve_recall(10, 48);
  CHECK(r > 0.90);
  CHECK(e.index.avg_degree() > 2.0);
}

// A single-threaded build must be byte-identical run to run. The parallel build
// deliberately is not (insert order changes which reverse links survive
// pruning); this pins the property that reproducibility is available on demand.
static void test_single_threaded_build_is_deterministic() {
  auto fingerprint = [](int threads) {
    Engine e;
    e.cfg.use_hnsw = true;
    e.cfg.hnsw_m = 8;
    e.cfg.ef_construction = 32;
    e.cfg.ef_search = 32;
    e.cfg.build_threads = threads;
    e.init_random(3000, 128, 16, 77, 64);
    double acc = 0;
    for (int u = 0; u < 32; ++u) {
      const float* q = e.user_query(static_cast<UserId>(u));
      auto got = e.index.retrieve(e.cat, q, nullptr, 10, 32, Kernel::Simd);
      for (std::size_t i = 0; i < got.size(); ++i) {
        acc += static_cast<double>(got[i].id) * static_cast<double>(i + 1);
      }
    }
    return acc;
  };
  const double a = fingerprint(1);
  const double b = fingerprint(1);
  CHECK(a == b);
  CHECK(a != 0.0);
}

static void test_index_and_catalog_roundtrip() {
  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.build_threads = 1;
  e.init_random(512, 64, 16, 33);
  const std::string cp = "data/_test_cat.bin", ip = "data/_test_idx.bin";
  CHECK(e.cat.save(cp));
  CHECK(e.index.save(ip));

  Catalog c2;
  CHECK(c2.load(cp));
  CHECK(c2.n == e.cat.n && c2.dim == e.cat.dim);
  for (int d = 0; d < c2.dim; ++d) CHECK(std::fabs(c2.aos_item(5)[d] - e.cat.aos_item(5)[d]) < 1e-6f);

  Index i2;
  CHECK(i2.load(ip));
  CHECK(i2.n() == e.index.n());
  const float* q = e.cat.aos_item(9);
  auto a = e.index.retrieve(e.cat, q, nullptr, 5, 32, Kernel::Simd);
  auto b = i2.retrieve(c2, q, nullptr, 5, 32, Kernel::Simd);
  CHECK(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].id == b[i].id);
  std::remove(cp.c_str());
  std::remove(ip.c_str());
}

static void test_quality_gate() {
  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.hnsw_m = 8;
  e.cfg.ef_search = 16;
  e.init_random(48, 12, 8, 3);
  std::vector<Interaction> train, test;
  for (int u = 0; u < 12; ++u) {
    for (int t = 0; t < 6; ++t) {
      Interaction x;
      x.user = static_cast<UserId>(u);
      x.item = static_cast<ItemId>((u + t) % 48);
      x.ts = static_cast<std::uint64_t>(t);
      (t < 4 ? train : test).push_back(x);
    }
  }
  auto rep = e.eval_quality(train, test, 10);
  CHECK(rep.n_users > 0);
  CHECK(rep.recall50 >= 0.0);
}

// End to end: every kernel returns k items and the int8 path returns mostly the
// same ones as float32 on a catalog this small.
static void test_all_kernels_end_to_end() {
  Request q;
  q.user_id = 1;
  q.k = 3;
  std::vector<ItemId> f32_ids;
  for (Kernel kern : {Kernel::Scalar, Kernel::Simd, Kernel::SoaStrided, Kernel::Blocked,
                      Kernel::Int8}) {
    Engine e;
    e.cfg.use_hnsw = false;
    e.cfg.kernel = kern;
    e.init_random(256, 16, 64, 4);
    auto r = e.recommend_sync(q);
    CHECK(r.status == Status::Ok);
    CHECK(r.items.size() == 3);
    if (kern == Kernel::Scalar) {
      for (auto& it : r.items) f32_ids.push_back(it.id);
    } else {
      int overlap = 0;
      for (auto& it : r.items) {
        for (auto id : f32_ids) {
          if (it.id == id) ++overlap;
        }
      }
      CHECK(overlap >= 2);  // int8 may reorder the tail, not the head
    }
  }
}

static void test_nearline_freshness_and_fault() {
  FeatureStore store;
  store.init(8, 16, true);
  EventLog log;
  EventRecord e;
  e.event_time_ms = 1000;
  e.user_id = 1;
  e.item_id = 2;
  e.type = 1;
  log.records.push_back(e);
  auto h = log.apply(store, 1300);
  CHECK(store.item(2).likes == 1);
  CHECK(h.percentile(0.5) == 300);
  CHECK(store.freshness_ms(1300, 1) == 300);
  store.set_delay_us(1000);
  auto t0 = now_us();
  (void)store.user(1);
  auto dt = now_us() - t0;
  CHECK(dt >= 800);
}

// The RCU snapshot must never let a reader observe a torn or half-applied
// generation while the writer is publishing into the other buffer.
static void test_snapshot_rcu_concurrency() {
  FeatureSnapshotStore store;
  const int n_items = 512;
  store.init(n_items, 64);
  const int n_readers = 4;
  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> reads{0};
  std::atomic<std::uint64_t> torn{0};
  std::atomic<int> ready{0};

  std::vector<std::thread> readers;
  for (int t = 0; t < n_readers; ++t) {
    readers.emplace_back([&]() {
      ready.fetch_add(1, std::memory_order_release);
      // do-while, not while: the writer's 200 rounds take microseconds and can
      // finish before a reader is ever scheduled, which made this test pass or
      // fail on thread-start latency rather than on anything about the snapshot.
      do {
        const auto g = store.read();
        // Invariant that must hold in every published generation: likes never
        // exceed views. A half-applied delta would break it.
        for (int i = 0; i < n_items; ++i) {
          if (g->items[static_cast<std::size_t>(i)].likes >
              g->items[static_cast<std::size_t>(i)].views) {
            torn.fetch_add(1, std::memory_order_relaxed);
          }
        }
        reads.fetch_add(1, std::memory_order_relaxed);
      } while (!stop.load(std::memory_order_relaxed));
    });
  }

  // Publishes must actually overlap live readers, or the grace period is never
  // exercised and the test proves nothing.
  while (ready.load(std::memory_order_acquire) < n_readers) {
    std::this_thread::yield();
  }

  for (int round = 0; round < 200; ++round) {
    for (int i = 0; i < 64; ++i) {
      store.stage(FeatureDelta{1000 + static_cast<std::uint64_t>(round),
                               static_cast<UserId>(i % 64),
                               static_cast<ItemId>((round * 7 + i) % n_items),
                               static_cast<std::uint8_t>(i % 3 == 0 ? 1 : 0)});
    }
    store.publish();
  }
  stop = true;
  for (auto& t : readers) t.join();

  CHECK(torn.load() == 0);
  CHECK(reads.load() >= static_cast<std::uint64_t>(n_readers));
  CHECK(store.publishes() == 200);
  // Both buffers must agree once the writer has quiesced.
  const auto g = store.read();
  std::uint64_t views = 0;
  for (int i = 0; i < n_items; ++i) views += g->items[static_cast<std::size_t>(i)].views;
  CHECK(views == 200ull * 64ull);
}

static void test_sliding_window_ctr() {
  SlidingWindowCounter w(60, 1000);
  for (int i = 0; i < 100; ++i) w.add(10'000 + static_cast<std::uint64_t>(i) * 10, i % 4 == 0);
  const float ctr = w.ctr(11'000, 5'000);
  CHECK(ctr > 0.20f && ctr < 0.30f);   // 25% likes
  CHECK(w.total_views() == 100);
  // A bucket older than the window must not count.
  SlidingWindowCounter w2(4, 1000);
  w2.add(1'000, true);
  w2.add(9'000, false);   // wraps the ring and resets the reused slot
  CHECK(w2.total_views() == 1);
}

// The pipeline must publish on the interval and charge freshness at visibility.
static void test_nearline_pipeline_publishes() {
  const std::string path = "data/_test_events.bin";
  synth_events(500, 16, 64, 1'000'000, 1).write_file(path);
  FileEventSource src(path);
  FeatureSnapshotStore store;
  store.init(64, 16);
  NearlinePipeline pipe(src, store, /*publish_interval_ms=*/100);

  std::uint64_t clock = 1'000'000;
  while (!src.eof()) {
    pipe.step(clock, 64);
    clock += 50;
  }
  pipe.flush(clock);
  const auto& st = pipe.stats();
  CHECK(st.consumed == 500);
  CHECK(st.published == 500);
  CHECK(st.publishes >= 2);
  CHECK(st.freshness_ms.percentile(0.5) > 0);
  const auto g = store.read();
  CHECK(g->watermark_ms == 1'000'000 + 499);
  std::remove(path.c_str());
}

static void test_loadshed() {
  Engine e;
  e.cfg.use_hnsw = false;
  e.cfg.workers = 1;
  e.max_queue = 1;
  e.init_random(16, 4, 8, 5);
  e.start_pool();
  e.cfg.feature_delay_us = 50'000;
  e.features.set_delay_us(50'000);
  int shed = 0;
  std::vector<std::future<Response>> futs;
  for (int i = 0; i < 20; ++i) {
    Request q;
    q.id = static_cast<RequestId>(i);
    q.user_id = 0;
    q.k = 3;
    q.timeout_us = 10'000;
    futs.push_back(e.recommend_async(q));
  }
  for (auto& f : futs) {
    auto r = f.get();
    if (r.status == Status::LoadShed || r.status == Status::Timeout) ++shed;
  }
  CHECK(shed >= 1);
  e.features.set_delay_us(0);
  e.stop_pool();
}

static void test_diagnose() {
  DiagnoseInput in;
  in.p99_us = 1000;
  in.queue_p99_us = 800;
  in.score_p99_us = 100;
  CHECK(classify(in) == Bottleneck::Queueing);
  in.queue_p99_us = 10;
  in.feature_p99_us = 700;
  CHECK(classify(in) == Bottleneck::FeatureStoreWait);
  in.feature_p99_us = 10;
  in.score_p99_us = 800;
  CHECK(classify(in) == Bottleneck::CpuBoundScoring);
}

int main() {
  try {
    test_dot_and_normalize();
    test_arena();
    test_protocol();
    test_topk_and_timeout();
    test_brute_vs_self();
    test_quality_gate();
    test_int8_kernel_exactness();
    test_quantized_query_error();
    test_blocked_matches_simd();
    test_soa_strided_matches_simd();
    test_visited_set_epochs();
    test_nsw_recall();
    test_single_threaded_build_is_deterministic();
    test_index_and_catalog_roundtrip();
    test_all_kernels_end_to_end();
    test_nearline_freshness_and_fault();
    test_snapshot_rcu_concurrency();
    test_sliding_window_ctr();
    test_nearline_pipeline_publishes();
    test_loadshed();
    test_diagnose();
  } catch (...) {
    return 1;
  }
  std::cout << "recserve tests ok\n";
  return 0;
}
