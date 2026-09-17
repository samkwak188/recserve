#pragma once
#include "types.hpp"
#include "scorer.hpp"
#include "index.hpp"
#include "features.hpp"
#include "thread_pool.hpp"
#include "arena.hpp"
#include "metrics.hpp"
#include "quality.hpp"
#include <future>
#include <random>
#include <cstring>
#include <thread>
#include <algorithm>

namespace recserve {

// Per-thread scratch reused across requests. This is what "arena mode" buys:
// no malloc/free on the request path once the buffers reach steady-state size.
struct Scratch {
  std::vector<Neighbor> cand;
  std::vector<ScoredItem> scored;
  QuantizedQuery qq;
};

inline Scratch& thread_scratch() {
  thread_local Scratch s;
  return s;
}

class Engine {
 public:
  EngineConfig cfg;
  Catalog cat;
  Index index;
  FeatureStore features;
  ThreadPool pool;
  ServeStats stats;
  BuildStats build_stats;
  float rank_w[4] = {1.f, 0.15f, 0.10f, 0.05f};
  std::atomic<bool> running{false};
  std::uint32_t max_queue = 128;

  // Build every layout the configured kernel needs, and nothing else. Building
  // all four on a 1M catalog costs ~500 MiB of RSS for layouts that go unread.
  void prepare_layouts() {
    if (cfg.kernel == Kernel::SoaStrided) cat.rebuild_soa();
    if (cfg.kernel == Kernel::Blocked) cat.rebuild_blocked();
    if (cfg.kernel == Kernel::Int8) cat.quantize_i8();
  }

  void build_index() {
    if (!cfg.use_hnsw) return;
    build_stats = index.build(cat, cfg.hnsw_m, cfg.ef_construction, 1, cfg.build_threads);
  }

  // n_clusters == 0 gives i.i.d. Gaussian directions: uniform on the sphere, no
  // intrinsic structure, and the documented worst case for graph ANN (see
  // ann-benchmarks.com, where random data sits far below SIFT/GloVe at equal ef).
  // n_clusters > 0 draws items around random centroids, which is what a trained
  // item-embedding table actually looks like: low intrinsic dimension.
  void init_random(int n_items, int n_users, int dim, unsigned seed = 42, int n_clusters = 0) {
    cfg.n_items = n_items;
    cfg.n_users = n_users;
    cfg.dim = dim;
    cfg.n_clusters = n_clusters;
    cat.resize(n_items, dim);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    if (n_clusters > 0) {
      std::vector<float> cent(static_cast<std::size_t>(n_clusters) * dim);
      for (int c = 0; c < n_clusters; ++c) {
        float* p = cent.data() + static_cast<std::size_t>(c) * dim;
        for (int d = 0; d < dim; ++d) p[d] = nd(rng);
        l2_normalize(p, dim);
      }
      for (int i = 0; i < n_items; ++i) {
        const float* p = cent.data() +
                         static_cast<std::size_t>(rng() % static_cast<unsigned>(n_clusters)) * dim;
        float* v = cat.aos_item(i);
        for (int d = 0; d < dim; ++d) v[d] = p[d] + 0.25f * nd(rng);
        l2_normalize(v, dim);
      }
    } else {
      for (int i = 0; i < n_items; ++i) {
        float* v = cat.aos_item(i);
        for (int d = 0; d < dim; ++d) v[d] = nd(rng);
        l2_normalize(v, dim);
      }
    }
    prepare_layouts();
    build_index();
    features.init(n_users, n_items, cfg.use_flat_features);
    make_user_queries(n_users, seed);
  }

  // Load a prebuilt catalog (and index, when present) instead of regenerating.
  bool load_fixture(const std::string& catalog_path, const std::string& index_path,
                    int n_users, unsigned seed = 42) {
    if (!cat.load(catalog_path)) return false;
    cfg.n_items = cat.n;
    cfg.dim = cat.dim;
    cfg.n_users = n_users;
    prepare_layouts();
    if (cfg.use_hnsw) {
      if (!index_path.empty() && index.load(index_path)) {
        build_stats = BuildStats{};
      } else {
        build_index();
      }
    }
    features.init(n_users, cat.n, cfg.use_flat_features);
    make_user_queries(n_users, seed);
    return true;
  }

  const float* user_query(UserId u) const {
    if (user_queries_.empty()) return nullptr;
    return user_queries_.data() + static_cast<std::size_t>(u % n_user_queries_) * cfg.dim;
  }

  void start_pool() {
    int n = cfg.workers;
    if (n <= 0) n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    cfg.workers = n;
    pool.start(n, cfg.pin_workers);
    running = true;
  }

  void stop_pool() {
    running = false;
    pool.stop();
  }

  Response recommend_sync(const Request& req) {
    Response r;
    r.id = req.id;
    if (req.k == 0 || req.k > kMaxK) {
      r.status = Status::BadRequest;
      return r;
    }
    if (cfg.feature_timeout) {
      r.status = Status::Timeout;
      r.feature_us = cfg.feature_delay_us;
      return r;
    }

    auto tf0 = now_us();
    UserFeatures uf = features.user(req.user_id);
    r.feature_us = static_cast<std::uint32_t>(now_us() - tf0);

    const float* q = user_query(req.user_id);
    if (!q) {
      r.status = Status::Unavailable;
      return r;
    }

    Scratch& sc = thread_scratch();
    const QuantizedQuery* qq = nullptr;
    if (cfg.kernel == Kernel::Int8) {
      sc.qq.set(q, cfg.dim);  // one quantization per request, not per item
      qq = &sc.qq;
    }

    const int rk = static_cast<int>(std::max(req.k, req.retrieve_k));
    auto tr0 = now_us();
    if (cfg.use_hnsw && index.n() > 0) {
      sc.cand = index.retrieve(cat, q, qq, rk, cfg.ef_search, cfg.kernel);
    } else {
      sc.cand = index.brute(cat, q, qq, rk, cfg.kernel);
    }
    r.retrieve_us = static_cast<std::uint32_t>(now_us() - tr0);

    auto ts0 = now_us();
    auto& scored = sc.scored;
    scored.clear();
    scored.reserve(sc.cand.size());
    for (const auto& c : sc.cand) {
      ItemFeatures itf = features.item(c.id);
      float recency = 0.f;
      for (int i = 0; i < uf.n_last; ++i) {
        if (uf.last_items[i] == c.id) recency = 1.f;
      }
      scored.push_back({c.id, rank_score(c.score, uf.ctr, itf.ctr, recency, rank_w)});
    }
    const std::size_t want = std::min<std::size_t>(req.k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(want),
                      scored.end(),
                      [](const ScoredItem& a, const ScoredItem& b) { return a.score > b.score; });
    r.score_us = static_cast<std::uint32_t>(now_us() - ts0);
    r.items.assign(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(want));
    r.hops = static_cast<std::uint32_t>(Index::last_hops());
    r.status = Status::Ok;
    return r;
  }

  // Single-threaded stats accumulation, used by bench and the agent's observe
  // step. Concurrent servers use the atomic counters in ServeStats instead.
  void record(const Response& r, std::uint64_t total_us) {
    stats.latency_us.add(static_cast<double>(total_us));
    stats.queue_us.add(static_cast<double>(r.queue_wait_us));
    stats.score_us.add(static_cast<double>(r.score_us));
    stats.retrieve_us.add(static_cast<double>(r.retrieve_us));
    stats.feature_us.add(static_cast<double>(r.feature_us));
    switch (r.status) {
      case Status::Ok: ++stats.ok; break;
      case Status::Timeout: ++stats.timeout; break;
      case Status::LoadShed: ++stats.loadshed; break;
      default: break;
    }
  }

  std::future<Response> recommend_async(const Request& req) {
    auto promise = std::make_shared<std::promise<Response>>();
    auto fut = promise->get_future();
    int w = static_cast<int>(req.id % static_cast<RequestId>(std::max(1, pool.size())));
    auto t_submit = now_us();
    bool ok = pool.submit(
        w,
        [this, req, promise, t_submit]() {
          Response r;
          auto waited = now_us() - t_submit;
          if (waited > req.timeout_us) {
            r.id = req.id;
            r.status = Status::Timeout;
            r.queue_wait_us = static_cast<std::uint32_t>(waited);
            promise->set_value(std::move(r));
            return;
          }
          r = recommend_sync(req);
          r.queue_wait_us = static_cast<std::uint32_t>(waited);
          promise->set_value(std::move(r));
        },
        max_queue);
    if (!ok) {
      Response r;
      r.id = req.id;
      r.status = Status::LoadShed;
      promise->set_value(r);
    }
    return fut;
  }

  QualityReport eval_quality(const std::vector<Interaction>& train,
                             const std::vector<Interaction>& test, int k) {
    std::unordered_map<UserId, std::unordered_set<ItemId>> gold;
    for (const auto& e : test) gold[e.user].insert(e.item);
    std::unordered_map<UserId, std::vector<float>> qmap;
    for (const auto& e : train) {
      if (static_cast<int>(e.item) >= cat.n) continue;
      auto& v = qmap[e.user];
      if (v.empty()) v.assign(static_cast<std::size_t>(cat.dim), 0.f);
      const float* it = cat.aos_item(static_cast<int>(e.item));
      for (int d = 0; d < cat.dim; ++d) v[static_cast<std::size_t>(d)] += it[d];
    }
    double rec = 0, nd = 0, rr = 0;
    int n = 0;
    QuantizedQuery qq;
    for (auto& kv : gold) {
      auto qit = qmap.find(kv.first);
      if (qit == qmap.end()) continue;
      l2_normalize(qit->second.data(), cat.dim);
      const float* q = qit->second.data();
      const QuantizedQuery* qp = nullptr;
      if (cfg.kernel == Kernel::Int8) {
        qq.set(q, cat.dim);
        qp = &qq;
      }
      // Exact float32 top-k is the gold standard for retrieve-recall, so that
      // quantization error shows up as a recall loss rather than being hidden
      // by comparing an approximate list against an equally approximate oracle.
      auto exact = index.brute(cat, q, nullptr, k, Kernel::Simd);
      std::vector<ItemId> pred;
      pred.reserve(exact.size());
      for (auto& b : exact) pred.push_back(b.id);
      rec += recall_at_k(pred, kv.second, k);
      nd += ndcg_at_k(pred, kv.second, k);

      std::unordered_set<ItemId> gold_ids(pred.begin(), pred.end());
      std::vector<Neighbor> got = cfg.use_hnsw && index.n() > 0
                                      ? index.retrieve(cat, q, qp, k, cfg.ef_search, cfg.kernel)
                                      : index.brute(cat, q, qp, k, cfg.kernel);
      std::vector<ItemId> ap;
      ap.reserve(got.size());
      for (auto& a : got) ap.push_back(a.id);
      rr += recall_at_k(ap, gold_ids, k);
      ++n;
    }
    QualityReport r;
    if (n) {
      r.recall50 = rec / n;
      r.ndcg50 = nd / n;
      r.retrieve_recall = rr / n;
      r.n_users = n;
    }
    return r;
  }

  // Retrieve-recall against the exact float32 top-k, over `n_probe` random
  // queries. This is the number that says what a latency win actually cost.
  double measure_retrieve_recall(int k, int n_probe, unsigned seed = 7) {
    if (cat.n == 0) return 0;
    std::mt19937 rng(seed);
    double acc = 0;
    int n = 0;
    QuantizedQuery qq;
    for (int t = 0; t < n_probe; ++t) {
      const float* q = user_query(static_cast<UserId>(rng() % std::max(1, cfg.n_users)));
      if (!q) break;
      const QuantizedQuery* qp = nullptr;
      if (cfg.kernel == Kernel::Int8) {
        qq.set(q, cfg.dim);
        qp = &qq;
      }
      auto exact = index.brute(cat, q, nullptr, k, Kernel::Simd);
      std::unordered_set<ItemId> gold;
      for (auto& e : exact) gold.insert(e.id);
      auto got = cfg.use_hnsw && index.n() > 0
                     ? index.retrieve(cat, q, qp, k, cfg.ef_search, cfg.kernel)
                     : index.brute(cat, q, qp, k, cfg.kernel);
      std::vector<ItemId> ap;
      for (auto& a : got) ap.push_back(a.id);
      acc += recall_at_k(ap, gold, k);
      ++n;
    }
    return n ? acc / n : 0.0;
  }

  std::size_t catalog_bytes() const {
    return cat.aos.size() * sizeof(float) + cat.soa.size() * sizeof(float) +
           cat.blk.size() * sizeof(float) + cat.aos_i8.size() +
           cat.i8_scale.size() * sizeof(float);
  }

 private:
  void make_user_queries(int n_users, unsigned seed) {
    n_user_queries_ = std::max(1, std::min(n_users, 4096));
    user_queries_.assign(static_cast<std::size_t>(n_user_queries_) * cfg.dim, 0.f);
    std::mt19937 rng(seed + 1);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int u = 0; u < n_user_queries_; ++u) {
      float* dst = user_queries_.data() + static_cast<std::size_t>(u) * cfg.dim;
      const int anchor = static_cast<int>(rng() % static_cast<unsigned>(std::max(1, cat.n)));
      const float* src = cat.aos_item(anchor);
      // Query = an item plus noise, so the exact top-1 is not trivially itself
      // and recall@k has room to differ between kernels.
      for (int d = 0; d < cfg.dim; ++d) dst[d] = src[d] + 0.35f * nd(rng);
      l2_normalize(dst, cfg.dim);
    }
  }

  std::vector<float> user_queries_;  // flat n_user_queries_ * dim
  int n_user_queries_ = 0;
};

}  // namespace recserve
