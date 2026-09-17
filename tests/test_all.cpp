#include "recserve/engine.hpp"
#include "recserve/quality.hpp"
#include "recserve/nearline.hpp"
#include "recserve/diagnose.hpp"
#include "recserve/protocol.hpp"
#include "recserve/arena.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

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
  float* q = e.cat.aos_item(0);
  auto a = e.index.brute(e.cat, q, 5, false, false);
  auto b = e.index.brute(e.cat, q, 5, false, false);
  CHECK(a.size() == 5);
  CHECK(a[0].id == b[0].id);
}

static void test_quality_gate() {
  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.hnsw_m = 8;
  e.cfg.hnsw_ef = 16;
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

static void test_int8_and_soa() {
  Engine e;
  e.cfg.use_hnsw = false;
  e.init_random(24, 4, 16, 4);
  Request q;
  q.user_id = 1;
  q.k = 3;
  auto f32 = e.recommend_sync(q);
  e.cat.quantize_i8();
  e.cfg.dtype = DType::Int8;
  auto i8 = e.recommend_sync(q);
  CHECK(f32.items.size() == 3);
  CHECK(i8.items.size() == 3);
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
  auto lag = replica_lag(log, 25);
  CHECK(lag.p50_ms == 25);
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
    test_int8_and_soa();
    test_nearline_freshness_and_fault();
    test_loadshed();
    test_diagnose();
  } catch (...) {
    return 1;
  }
  std::cout << "recserve tests ok\n";
  return 0;
}
