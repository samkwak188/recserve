#pragma once
#include "types.hpp"
#include "scorer.hpp"
#include "arena.hpp"
#include <random>
#include <queue>
#include <unordered_set>
#include <limits>

namespace recserve {

struct Neighbor {
  ItemId id = 0;
  float dist = 0.f;  // inner-product distance = 1 - dot for unit vectors
  bool operator<(const Neighbor& o) const { return dist < o.dist; }
};

class Index {
 public:
  void build(const Catalog& cat, int m, int ef, unsigned seed = 1) {
    n_ = cat.n;
    dim_ = cat.dim;
    m_ = std::max(2, m);
    ef_ = std::max(m_, ef);
    graph_.assign(n_, {});
    if (n_ == 0) return;

    std::mt19937 rng(seed);
    // Incremental NSW: each new node links to M nearest among a sampled beam of previous nodes.
    for (int i = 0; i < n_; ++i) {
      if (i == 0) continue;
      const float* q = cat.aos_item(i);
      int start = static_cast<int>(rng() % static_cast<unsigned>(i));
      auto cand = search_from(cat, q, start, std::min(ef_, i), i, /*limit_n=*/i);
      int links = std::min(m_, static_cast<int>(cand.size()));
      for (int k = 0; k < links; ++k) {
        ItemId nb = cand[k].id;
        graph_[i].push_back(nb);
        graph_[nb].push_back(static_cast<ItemId>(i));
        if (static_cast<int>(graph_[nb].size()) > m_ * 2) {
          graph_[nb].resize(m_ * 2);
        }
      }
    }
    entry_ = static_cast<ItemId>(n_ / 2);
  }

  // O(n * M) random+ring graph for large catalogs. Search still walks neighbors.
  void build_sampled(const Catalog& cat, int m, unsigned seed = 1) {
    n_ = cat.n;
    dim_ = cat.dim;
    m_ = std::max(2, m);
    ef_ = std::max(m_, ef_);
    graph_.assign(n_, {});
    if (n_ == 0) return;
    std::mt19937 rng(seed);
    for (int i = 1; i < n_; ++i) {
      int links = std::min(m_, i);
      graph_[i].push_back(static_cast<ItemId>(i - 1));
      graph_[i - 1].push_back(static_cast<ItemId>(i));
      for (int k = 1; k < links; ++k) {
        ItemId nb = static_cast<ItemId>(rng() % static_cast<unsigned>(i));
        graph_[i].push_back(nb);
        graph_[nb].push_back(static_cast<ItemId>(i));
      }
    }
    entry_ = static_cast<ItemId>(n_ / 2);
  }

  std::vector<Neighbor> retrieve(const Catalog& cat, const float* query, int k, int ef,
                                 bool simd, bool soa, Arena* arena) const {
    (void)arena;
    if (n_ == 0) return {};
    int beam = std::max(ef, k);
    auto cand = search_from(cat, query, static_cast<int>(entry_), beam, n_, n_);
    if (static_cast<int>(cand.size()) > k) cand.resize(k);
    for (auto& c : cand) {
      c.dist = cat.dot_item(query, static_cast<int>(c.id), simd, soa);  // reuse as score
    }
    return cand;
  }

  std::vector<Neighbor> brute(const Catalog& cat, const float* query, int k, bool simd,
                              bool soa) const {
    std::vector<Neighbor> best;
    best.reserve(k + 1);
    for (int i = 0; i < cat.n; ++i) {
      float s = cat.dot_item(query, i, simd, soa);
      Neighbor nb{static_cast<ItemId>(i), s};
      if (static_cast<int>(best.size()) < k) {
        best.push_back(nb);
        if (static_cast<int>(best.size()) == k) {
          std::make_heap(best.begin(), best.end(),
                         [](const Neighbor& a, const Neighbor& b) { return a.dist > b.dist; });
        }
      } else if (s > best.front().dist) {
        std::pop_heap(best.begin(), best.end(),
                      [](const Neighbor& a, const Neighbor& b) { return a.dist > b.dist; });
        best.back() = nb;
        std::push_heap(best.begin(), best.end(),
                       [](const Neighbor& a, const Neighbor& b) { return a.dist > b.dist; });
      }
    }
    std::sort(best.begin(), best.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.dist > b.dist; });
    return best;
  }

  int n() const { return n_; }

 private:
  std::vector<Neighbor> search_from(const Catalog& cat, const float* query, int start, int ef,
                                    int limit_n, int graph_n) const {
    std::vector<char> vis(limit_n, 0);
    auto cmp_min = [](const Neighbor& a, const Neighbor& b) { return a.dist > b.dist; };
    auto cmp_max = [](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; };
    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp_min)> cand(cmp_min);
    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp_max)> wip(cmp_max);

    start = std::min(std::max(start, 0), limit_n - 1);
    float s0 = cat.dot_item(query, start, true, false);
    vis[start] = 1;
    wip.push({static_cast<ItemId>(start), s0});
    cand.push({static_cast<ItemId>(start), s0});

    while (!wip.empty()) {
      Neighbor c = wip.top();
      wip.pop();
      if (!cand.empty() && c.dist < cand.top().dist) break;
      const auto& nbrs = (static_cast<int>(c.id) < graph_n && !graph_.empty())
                             ? graph_[c.id]
                             : empty_;
      for (ItemId nb : nbrs) {
        if (static_cast<int>(nb) >= limit_n || vis[nb]) continue;
        vis[nb] = 1;
        float s = cat.dot_item(query, static_cast<int>(nb), true, false);
        if (static_cast<int>(cand.size()) < ef || s > cand.top().dist) {
          cand.push({nb, s});
          wip.push({nb, s});
          if (static_cast<int>(cand.size()) > ef) cand.pop();
        }
      }
      if (nbrs.empty()) {
        // During build, graph may be empty: sample a few previous ids.
        for (int t = 0; t < std::min(8, limit_n); ++t) {
          int nb = (static_cast<int>(c.id) + 1 + t) % limit_n;
          if (vis[nb]) continue;
          vis[nb] = 1;
          float s = cat.dot_item(query, nb, true, false);
          cand.push({static_cast<ItemId>(nb), s});
          wip.push({static_cast<ItemId>(nb), s});
          if (static_cast<int>(cand.size()) > ef) cand.pop();
        }
      }
    }
    std::vector<Neighbor> out;
    while (!cand.empty()) {
      out.push_back(cand.top());
      cand.pop();
    }
    std::sort(out.begin(), out.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.dist > b.dist; });
    return out;
  }

  int n_ = 0;
  int dim_ = 0;
  int m_ = 16;
  int ef_ = 64;
  ItemId entry_ = 0;
  std::vector<std::vector<ItemId>> graph_;
  std::vector<ItemId> empty_;
};

}  // namespace recserve
