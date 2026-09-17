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

class Engine {
 public:
  EngineConfig cfg;
  Catalog cat;
  Index index;
  FeatureStore features;
  ThreadPool pool;
  ServeStats stats;
  float rank_w[4] = {1.f, 0.15f, 0.10f, 0.05f};
  std::atomic<bool> running{false};
  std::uint32_t max_queue = 128;

  void init_random(int n_items, int n_users, int dim, unsigned seed = 42) {
    cfg.n_items = n_items;
    cfg.n_users = n_users;
    cfg.dim = dim;
    cat.resize(n_items, dim);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < n_items; ++i) {
      float* v = cat.aos_item(i);
      for (int d = 0; d < dim; ++d) v[d] = nd(rng);
      l2_normalize(v, dim);
    }
    cat.rebuild_soa();
    if (cfg.dtype == DType::Int8) cat.quantize_i8();
    if (cfg.use_hnsw) {
      if (n_items >= 8192)
        index.build_sampled(cat, cfg.hnsw_m, seed);
      else
        index.build(cat, cfg.hnsw_m, cfg.hnsw_ef, seed);
    }
    features.init(n_users, n_items, cfg.use_flat_features);
    user_queries_.assign(n_users, std::vector<float>(dim, 0.f));
    for (int u = 0; u < n_users; ++u) {
      int item = static_cast<int>(rng() % n_items);
      user_queries_[u] = std::vector<float>(cat.aos_item(item), cat.aos_item(item) + dim);
    }
  }

  const float* user_query(UserId u) const {
    if (user_queries_.empty()) return nullptr;
    return user_queries_[u % user_queries_.size()].data();
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
    auto t0 = now_us();
    if (req.k == 0 || req.k > kMaxK) {
      r.status = Status::BadRequest;
      return r;
    }
    if (cfg.feature_timeout) {
      r.status = Status::Timeout;
      r.feature_us = cfg.feature_delay_us;
      return r;
    }
    if (cfg.use_arena) thread_arena().reset();

    auto tf0 = now_us();
    UserFeatures uf = features.user(req.user_id);
    r.feature_us = static_cast<std::uint32_t>(now_us() - tf0);

    const float* q = user_query(req.user_id);
    if (!q) {
      r.status = Status::Unavailable;
      return r;
    }

    int rk = static_cast<int>(std::max(req.k, req.retrieve_k));
    auto tr0 = now_us();
    std::vector<Neighbor> cand;
    if (cfg.use_hnsw && index.n() > 0) {
      cand = index.retrieve(cat, q, rk, cfg.hnsw_ef, cfg.use_simd, cfg.use_soa,
                            cfg.use_arena ? &thread_arena() : nullptr);
    } else {
      cand = index.brute(cat, q, rk, cfg.use_simd, cfg.use_soa);
    }
    r.retrieve_us = static_cast<std::uint32_t>(now_us() - tr0);

    auto ts0 = now_us();
    std::vector<ScoredItem> scored;
    scored.reserve(cand.size());
    for (const auto& c : cand) {
      ItemFeatures itf = features.item(c.id);
      float recency = 0.f;
      for (int i = 0; i < uf.n_last; ++i) {
        if (uf.last_items[i] == c.id) recency = 1.f;
      }
      float s = rank_score(c.dist, uf.ctr, itf.ctr, recency, rank_w);
      scored.push_back({c.id, s});
    }
    std::partial_sort(scored.begin(),
                      scored.begin() + std::min(static_cast<int>(req.k), static_cast<int>(scored.size())),
                      scored.end(),
                      [](const ScoredItem& a, const ScoredItem& b) { return a.score > b.score; });
    if (scored.size() > req.k) scored.resize(req.k);
    r.score_us = static_cast<std::uint32_t>(now_us() - ts0);
    r.items = std::move(scored);
    r.status = Status::Ok;
    (void)t0;
    return r;
  }

  std::future<Response> recommend_async(const Request& req) {
    auto promise = std::make_shared<std::promise<Response>>();
    auto fut = promise->get_future();
    int w = static_cast<int>(req.id % std::max(1, pool.size()));
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
      if (v.empty()) v.assign(cat.dim, 0.f);
      const float* it = cat.aos_item(static_cast<int>(e.item));
      for (int d = 0; d < cat.dim; ++d) v[d] += it[d];
    }
    double rec = 0, nd = 0, rr = 0;
    int n = 0;
    for (auto& kv : gold) {
      auto qit = qmap.find(kv.first);
      if (qit == qmap.end()) continue;
      l2_normalize(qit->second.data(), cat.dim);
      auto brute = index.brute(cat, qit->second.data(), k, cfg.use_simd, cfg.use_soa);
      std::vector<ItemId> pred;
      for (auto& b : brute) pred.push_back(b.id);
      rec += recall_at_k(pred, kv.second, k);
      nd += ndcg_at_k(pred, kv.second, k);
      if (cfg.use_hnsw) {
        auto ann = index.retrieve(cat, qit->second.data(), k, cfg.hnsw_ef, cfg.use_simd, cfg.use_soa,
                                  nullptr);
        std::unordered_set<ItemId> gold_ids;
        for (auto& b : brute) gold_ids.insert(b.id);
        std::vector<ItemId> ap;
        for (auto& a : ann) ap.push_back(a.id);
        rr += recall_at_k(ap, gold_ids, k);
      } else {
        rr += 1.0;
      }
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

  std::vector<std::vector<float>> user_queries_;
};

}  // namespace recserve
