#pragma once
// Hierarchical Navigable Small World index.
//
// Implements Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI
// 42(4):824-836, 2020 (arXiv:1603.09320):
//   Algorithm 1 INSERT                      -> insert()
//   Algorithm 2 SEARCH-LAYER                -> search_layer()
//   Algorithm 4 SELECT-NEIGHBORS-HEURISTIC  -> select_heuristic()
//   Algorithm 5 K-NN-SEARCH                 -> retrieve()
// Level assignment uses the paper's exponentially decaying distribution with
// mL = 1/ln(M) (Section 4.1). Layer 0 keeps M_max0 = 2M links, upper layers M.
//
// Build concurrency follows hnswlib (github.com/nmslib/hnswlib): one mutex per
// node guarding its adjacency lists, plus a global mutex taken only when an
// insert raises the top level and moves the entry point.
//
// Why the hierarchy is here at all: a flat single-layer NSW with a fixed entry
// point measured recall@10 = 0.69 at ef=64 on a 16,384-item catalog, because
// greedy descent from one fixed node lands in a local optimum. The upper layers
// are what supply a good entry point.
#include "types.hpp"
#include "scorer.hpp"
#include "arena.hpp"
#include <random>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <fstream>
#include <cmath>
#include <limits>

namespace recserve {

struct Neighbor {
  ItemId id = 0;
  float score = 0.f;  // inner product; higher is better
  bool operator<(const Neighbor& o) const { return score < o.score; }
};

struct CmpBest {  // max-heap: pop highest score first
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.score < b.score; }
};
struct CmpWorst {  // min-heap: pop lowest score first
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.score > b.score; }
};

// Epoch-stamped visited set. Replaces a per-query vector<char> allocation plus
// memset with a single 32-bit compare; the buffer is allocated once per thread.
class VisitedSet {
 public:
  void resize(int n) {
    if (static_cast<int>(stamp_.size()) < n) {
      stamp_.assign(static_cast<std::size_t>(n), 0u);
      cur_ = 0;
    }
  }
  void next_epoch() {
    if (++cur_ == 0) {  // wrapped after 4e9 queries: one real clear
      std::fill(stamp_.begin(), stamp_.end(), 0u);
      cur_ = 1;
    }
  }
  bool test_and_set(std::uint32_t i) {
    if (stamp_[i] == cur_) return true;
    stamp_[i] = cur_;
    return false;
  }

 private:
  std::vector<std::uint32_t> stamp_;
  std::uint32_t cur_ = 0;
};

inline VisitedSet& thread_visited() {
  thread_local VisitedSet v;
  return v;
}

inline std::uint64_t& thread_hops() {
  thread_local std::uint64_t h = 0;
  return h;
}

struct BuildStats {
  double seconds = 0;
  std::uint64_t dist_calls = 0;
  int threads = 1;
  int max_level = 0;
};

class Index {
 public:
  // ---------------------------------------------------------------- build --

  BuildStats build(const Catalog& cat, int m, int ef_construction, unsigned seed = 1,
                   int threads = 0) {
    n_ = cat.n;
    dim_ = cat.dim;
    m_ = std::max(2, m);
    m_max0_ = std::min(m_ * 2, 128);
    m_max_ = std::min(m_, 128);
    ef_construction_ = std::max(ef_construction, m_);
    level_mult_ = 1.0 / std::log(static_cast<double>(m_));
    seed_ = seed;

    links0_.assign(static_cast<std::size_t>(n_) * m_max0_, 0u);
    deg0_.assign(static_cast<std::size_t>(n_), 0u);
    level_.assign(static_cast<std::size_t>(n_), 0);
    upper_.assign(static_cast<std::size_t>(n_), {});
    updeg_.assign(static_cast<std::size_t>(n_), {});
    entry_ = 0;
    max_level_ = 0;

    BuildStats bs;
    if (n_ <= 1) return bs;

    if (threads <= 0) {
      threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    threads = std::max(1, std::min(threads, std::max(1, n_ / 512)));
    bs.threads = threads;

    node_locks_ = std::vector<std::mutex>(static_cast<std::size_t>(n_));
    std::atomic<std::uint64_t> dist_calls{0};
    const auto t0 = now_us();

    level_[0] = assign_level(0);
    max_level_ = level_[0];
    alloc_upper(0);

    // Seed serially so every parallel insert starts from a populated graph.
    const int seed_n = std::min(n_, 512);
    for (int i = 1; i < seed_n; ++i) insert(cat, i, dist_calls);

    if (threads == 1 || n_ <= seed_n) {
      for (int i = seed_n; i < n_; ++i) insert(cat, i, dist_calls);
    } else {
      std::atomic<int> next{seed_n};
      std::vector<std::thread> pool;
      pool.reserve(static_cast<std::size_t>(threads));
      for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&]() {
          for (;;) {
            const int i = next.fetch_add(1);
            if (i >= n_) return;
            insert(cat, i, dist_calls);
          }
        });
      }
      for (auto& th : pool) th.join();
    }

    node_locks_.clear();  // serving path takes no locks
    bs.seconds = static_cast<double>(now_us() - t0) / 1e6;
    bs.dist_calls = dist_calls.load();
    bs.max_level = max_level_;
    return bs;
  }

  // --------------------------------------------------------------- search --

  // Algorithm 5: greedy descent through the upper layers, then one ef-wide
  // search at layer 0.
  std::vector<Neighbor> retrieve(const Catalog& cat, const float* query, const QuantizedQuery* qq,
                                 int k, int ef, Kernel kern, Arena* arena = nullptr) const {
    (void)arena;
    if (n_ == 0) return {};
    ItemId cur = entry_;
    float cur_s = cat.dot_item(query, qq, static_cast<int>(cur), kern);
    for (int lay = max_level_; lay >= 1; --lay) {
      greedy_descend(cat, query, qq, kern, cur, cur_s, lay, nullptr);
    }
    std::vector<Neighbor> w;
    search_layer(cat, query, qq, kern, cur, std::max(ef, k), 0, w, nullptr, n_);
    std::sort(w.begin(), w.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.score > b.score; });
    if (static_cast<int>(w.size()) > k) w.resize(static_cast<std::size_t>(k));
    return w;
  }

  // Exact scan. Chunked so the blocked kernel sweeps contiguous memory and the
  // top-k heap stays in cache regardless of catalog size.
  std::vector<Neighbor> brute(const Catalog& cat, const float* query, const QuantizedQuery* qq,
                              int k, Kernel kern) const {
    constexpr int kChunk = 1024;  // multiple of kBlock
    static_assert(kChunk % kBlock == 0, "chunk must cover whole blocks");
    std::vector<float> buf(kChunk);
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(k) + 1);
    auto worse = [](const Neighbor& a, const Neighbor& b) { return a.score > b.score; };
    for (int base = 0; base < cat.n; base += kChunk) {
      const int cnt = std::min(kChunk, cat.n - base);
      cat.score_range(query, qq, kern, base, cnt, buf.data());
      for (int j = 0; j < cnt; ++j) {
        const Neighbor nb{static_cast<ItemId>(base + j), buf[static_cast<std::size_t>(j)]};
        if (static_cast<int>(best.size()) < k) {
          best.push_back(nb);
          if (static_cast<int>(best.size()) == k) std::make_heap(best.begin(), best.end(), worse);
        } else if (nb.score > best.front().score) {
          std::pop_heap(best.begin(), best.end(), worse);
          best.back() = nb;
          std::push_heap(best.begin(), best.end(), worse);
        }
      }
    }
    std::sort(best.begin(), best.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.score > b.score; });
    return best;
  }

  int n() const { return n_; }
  int m() const { return m_; }
  int max_level() const { return max_level_; }
  static std::uint64_t last_hops() { return thread_hops(); }

  double avg_degree() const {
    if (deg0_.empty()) return 0;
    double s = 0;
    for (auto d : deg0_) s += d;
    return s / static_cast<double>(deg0_.size());
  }

  std::size_t graph_bytes() const {
    std::size_t b = links0_.size() * sizeof(ItemId) + deg0_.size() * sizeof(std::uint32_t);
    for (const auto& u : upper_) b += u.size() * sizeof(ItemId);
    return b;
  }

  // ----------------------------------------------------------------- disk --

  bool save(const std::string& path) const {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    const std::uint32_t magic = 0x48535732u;  // HSW2
    auto w32 = [&](int v) { o.write(reinterpret_cast<const char*>(&v), 4); };
    o.write(reinterpret_cast<const char*>(&magic), 4);
    w32(n_); w32(dim_); w32(m_); w32(m_max_); w32(m_max0_); w32(max_level_);
    w32(static_cast<int>(entry_));
    o.write(reinterpret_cast<const char*>(deg0_.data()),
            static_cast<std::streamsize>(deg0_.size() * sizeof(std::uint32_t)));
    o.write(reinterpret_cast<const char*>(links0_.data()),
            static_cast<std::streamsize>(links0_.size() * sizeof(ItemId)));
    o.write(reinterpret_cast<const char*>(level_.data()),
            static_cast<std::streamsize>(level_.size() * sizeof(int)));
    for (int i = 0; i < n_; ++i) {
      const int lv = level_[static_cast<std::size_t>(i)];
      if (lv <= 0) continue;
      o.write(reinterpret_cast<const char*>(updeg_[static_cast<std::size_t>(i)].data()),
              static_cast<std::streamsize>(static_cast<std::size_t>(lv) * sizeof(std::uint32_t)));
      o.write(reinterpret_cast<const char*>(upper_[static_cast<std::size_t>(i)].data()),
              static_cast<std::streamsize>(static_cast<std::size_t>(lv) * m_max_ * sizeof(ItemId)));
    }
    return static_cast<bool>(o);
  }

  bool load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::uint32_t magic = 0;
    int entry = 0;
    auto r32 = [&](int& v) { in.read(reinterpret_cast<char*>(&v), 4); };
    in.read(reinterpret_cast<char*>(&magic), 4);
    r32(n_); r32(dim_); r32(m_); r32(m_max_); r32(m_max0_); r32(max_level_); r32(entry);
    if (!in || magic != 0x48535732u || n_ <= 0 || m_max0_ <= 0) return false;
    entry_ = static_cast<ItemId>(entry);
    deg0_.assign(static_cast<std::size_t>(n_), 0u);
    links0_.assign(static_cast<std::size_t>(n_) * m_max0_, 0u);
    level_.assign(static_cast<std::size_t>(n_), 0);
    in.read(reinterpret_cast<char*>(deg0_.data()),
            static_cast<std::streamsize>(deg0_.size() * sizeof(std::uint32_t)));
    in.read(reinterpret_cast<char*>(links0_.data()),
            static_cast<std::streamsize>(links0_.size() * sizeof(ItemId)));
    in.read(reinterpret_cast<char*>(level_.data()),
            static_cast<std::streamsize>(level_.size() * sizeof(int)));
    upper_.assign(static_cast<std::size_t>(n_), {});
    updeg_.assign(static_cast<std::size_t>(n_), {});
    for (int i = 0; i < n_; ++i) {
      const int lv = level_[static_cast<std::size_t>(i)];
      if (lv <= 0) continue;
      updeg_[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(lv), 0u);
      upper_[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(lv) * m_max_, 0u);
      in.read(reinterpret_cast<char*>(updeg_[static_cast<std::size_t>(i)].data()),
              static_cast<std::streamsize>(static_cast<std::size_t>(lv) * sizeof(std::uint32_t)));
      in.read(reinterpret_cast<char*>(upper_[static_cast<std::size_t>(i)].data()),
              static_cast<std::streamsize>(static_cast<std::size_t>(lv) * m_max_ * sizeof(ItemId)));
    }
    node_locks_.clear();
    return static_cast<bool>(in);
  }

 private:
  // ------------------------------------------------------------- topology --

  int max_deg(int layer) const { return layer == 0 ? m_max0_ : m_max_; }

  ItemId* links_at(int i, int layer) {
    if (layer == 0) return links0_.data() + static_cast<std::size_t>(i) * m_max0_;
    return upper_[static_cast<std::size_t>(i)].data() + static_cast<std::size_t>(layer - 1) * m_max_;
  }
  const ItemId* links_at(int i, int layer) const {
    if (layer == 0) return links0_.data() + static_cast<std::size_t>(i) * m_max0_;
    return upper_[static_cast<std::size_t>(i)].data() + static_cast<std::size_t>(layer - 1) * m_max_;
  }
  std::uint32_t& deg_at(int i, int layer) {
    if (layer == 0) return deg0_[static_cast<std::size_t>(i)];
    return updeg_[static_cast<std::size_t>(i)][static_cast<std::size_t>(layer - 1)];
  }
  std::uint32_t deg_at(int i, int layer) const {
    if (layer == 0) return deg0_[static_cast<std::size_t>(i)];
    const auto& v = updeg_[static_cast<std::size_t>(i)];
    const auto idx = static_cast<std::size_t>(layer - 1);
    return idx < v.size() ? v[idx] : 0u;
  }

  // Section 4.1: l = floor(-ln(U(0,1)) * mL). Seeded from the node id so a
  // node's LEVEL does not depend on thread scheduling.
  //
  // The graph as a whole is still order-dependent: with several threads
  // inserting at once, the order in which reverse links arrive changes which
  // ones the pruning heuristic keeps, so two builds of the same catalog with
  // the same parameters produce different -- equally valid -- graphs. Measured
  // on 16,384 items: recall@10 spread of 0.0172 across five 12-thread builds,
  // and exactly 0.0000 across five single-threaded builds. hnswlib behaves the
  // same way for the same reason.
  //
  // This is why every published measurement loads an index snapshot from disk
  // instead of rebuilding: otherwise the build noise would sit underneath every
  // recall comparison. Use --build-threads 1 when reproducibility matters more
  // than build time.
  int assign_level(int i) const {
    std::mt19937 rng(seed_ * 2654435761u + static_cast<unsigned>(i));
    double u = (static_cast<double>(rng()) + 1.0) / (static_cast<double>(std::mt19937::max()) + 2.0);
    const int lv = static_cast<int>(-std::log(u) * level_mult_);
    return std::min(lv, 16);
  }

  void alloc_upper(int i) {
    const int lv = level_[static_cast<std::size_t>(i)];
    if (lv <= 0) return;
    upper_[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(lv) * m_max_, 0u);
    updeg_[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(lv), 0u);
  }

  // --------------------------------------------------------------- search --

  // Greedy ef=1 walk used on the upper layers to find the entry for the layer
  // below. Updates cur / cur_s in place.
  void greedy_descend(const Catalog& cat, const float* query, const QuantizedQuery* qq, Kernel kern,
                      ItemId& cur, float& cur_s, int layer,
                      std::atomic<std::uint64_t>* dist_calls) const {
    bool moved = true;
    while (moved) {
      moved = false;
      ItemId local[128];
      int deg = 0;
      {
        std::unique_lock<std::mutex> lk;
        if (!node_locks_.empty()) lk = std::unique_lock<std::mutex>(node_locks_[cur]);
        deg = static_cast<int>(deg_at(static_cast<int>(cur), layer));
        deg = std::min(deg, max_deg(layer));
        const ItemId* l = links_at(static_cast<int>(cur), layer);
        for (int j = 0; j < deg; ++j) local[j] = l[j];
      }
      for (int j = 0; j < deg; ++j) {
        const float s = cat.dot_item(query, qq, static_cast<int>(local[j]), kern);
        if (dist_calls) dist_calls->fetch_add(1, std::memory_order_relaxed);
        if (s > cur_s) {
          cur_s = s;
          cur = local[j];
          moved = true;
        }
      }
    }
  }

  // Algorithm 2: best-first traversal keeping the ef best seen so far.
  void search_layer(const Catalog& cat, const float* query, const QuantizedQuery* qq, Kernel kern,
                    ItemId ep, int ef, int layer, std::vector<Neighbor>& out,
                    std::atomic<std::uint64_t>* dist_calls, int limit_n) const {
    if (limit_n <= 0) return;
    auto& vis = thread_visited();
    vis.resize(n_);
    vis.next_epoch();

    std::priority_queue<Neighbor, std::vector<Neighbor>, CmpBest> cand;
    std::priority_queue<Neighbor, std::vector<Neighbor>, CmpWorst> top;

    if (ep >= static_cast<ItemId>(limit_n)) ep = 0;
    const float s0 = cat.dot_item(query, qq, static_cast<int>(ep), kern);
    if (dist_calls) dist_calls->fetch_add(1, std::memory_order_relaxed);
    vis.test_and_set(ep);
    cand.push({ep, s0});
    top.push({ep, s0});

    std::uint64_t hops = 0;
    while (!cand.empty()) {
      const Neighbor c = cand.top();
      if (static_cast<int>(top.size()) >= ef && c.score < top.top().score) break;
      cand.pop();
      ++hops;

      ItemId local[128];
      int deg = 0;
      {
        std::unique_lock<std::mutex> lk;
        if (!node_locks_.empty()) lk = std::unique_lock<std::mutex>(node_locks_[c.id]);
        deg = static_cast<int>(deg_at(static_cast<int>(c.id), layer));
        deg = std::min(deg, max_deg(layer));
        const ItemId* l = links_at(static_cast<int>(c.id), layer);
        for (int j = 0; j < deg; ++j) local[j] = l[j];
      }

      for (int j = 0; j < deg; ++j) {
        const ItemId e = local[j];
        if (e >= static_cast<ItemId>(limit_n)) continue;
        if (vis.test_and_set(e)) continue;
        const float s = cat.dot_item(query, qq, static_cast<int>(e), kern);
        if (dist_calls) dist_calls->fetch_add(1, std::memory_order_relaxed);
        if (static_cast<int>(top.size()) < ef || s > top.top().score) {
          cand.push({e, s});
          top.push({e, s});
          if (static_cast<int>(top.size()) > ef) top.pop();
        }
      }
    }
    thread_hops() = hops;

    out.clear();
    out.reserve(top.size());
    while (!top.empty()) {
      out.push_back(top.top());
      top.pop();
    }
  }

  // Algorithm 4: keep e only if it is closer to the query than to every already
  // selected neighbour. This is what makes the graph navigable instead of a
  // cluster of mutually redundant short links.
  void select_heuristic(const Catalog& cat, std::vector<Neighbor>& w, int m,
                        std::vector<Neighbor>& out) const {
    std::sort(w.begin(), w.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.score > b.score; });
    out.clear();
    for (const auto& e : w) {
      if (static_cast<int>(out.size()) >= m) break;
      bool keep = true;
      const float* ev = cat.aos_item(static_cast<int>(e.id));
      for (const auto& r : out) {
        if (dot_simd(ev, cat.aos_item(static_cast<int>(r.id)), cat.dim) > e.score) {
          keep = false;
          break;
        }
      }
      if (keep) out.push_back(e);
    }
  }

  // --------------------------------------------------------------- insert --

  // Algorithm 1.
  void insert(const Catalog& cat, int i, std::atomic<std::uint64_t>& dist_calls) {
    const float* q = cat.aos_item(i);
    const int l = assign_level(i);
    level_[static_cast<std::size_t>(i)] = l;
    alloc_upper(i);

    // Raising the top level moves the entry point, so it is done under a global
    // lock; every other insert only ever touches per-node locks.
    std::unique_lock<std::mutex> global_lk(global_mu_, std::defer_lock);
    if (l > max_level_) global_lk.lock();

    ItemId cur = entry_;
    const int top = max_level_;
    float cur_s = cat.dot_item(q, nullptr, static_cast<int>(cur), Kernel::Simd);
    dist_calls.fetch_add(1, std::memory_order_relaxed);

    for (int lay = top; lay >= l + 1; --lay) {
      greedy_descend(cat, q, nullptr, Kernel::Simd, cur, cur_s, lay, &dist_calls);
    }

    std::vector<Neighbor> w, picked;
    for (int lay = std::min(l, top); lay >= 0; --lay) {
      search_layer(cat, q, nullptr, Kernel::Simd, cur, ef_construction_, lay, w, &dist_calls, i);
      if (w.empty()) continue;
      select_heuristic(cat, w, m_, picked);
      connect(cat, i, picked, lay);
      cur = picked.empty() ? cur : picked.front().id;
    }

    if (l > max_level_) {
      max_level_ = l;
      entry_ = static_cast<ItemId>(i);
    }
  }

  void connect(const Catalog& cat, int i, const std::vector<Neighbor>& picked, int layer) {
    const int cap = max_deg(layer);
    {
      std::lock_guard<std::mutex> g(node_locks_[static_cast<std::size_t>(i)]);
      ItemId* l = links_at(i, layer);
      std::uint32_t d = 0;
      for (const auto& p : picked) {
        if (static_cast<int>(d) >= cap) break;
        l[d++] = p.id;
      }
      deg_at(i, layer) = d;
    }
    for (const auto& p : picked) {
      if (level_[p.id] < layer) continue;  // neighbour does not exist on this layer
      std::lock_guard<std::mutex> g(node_locks_[p.id]);
      ItemId* l = links_at(static_cast<int>(p.id), layer);
      std::uint32_t& d = deg_at(static_cast<int>(p.id), layer);
      if (static_cast<int>(d) < cap) {
        l[d++] = static_cast<ItemId>(i);
        continue;
      }
      // Full: re-select the whole neighbourhood with the same heuristic rather
      // than evicting the oldest link.
      std::vector<Neighbor> w;
      w.reserve(static_cast<std::size_t>(cap) + 1);
      const float* pv = cat.aos_item(static_cast<int>(p.id));
      for (std::uint32_t j = 0; j < d; ++j) {
        w.push_back({l[j], dot_simd(pv, cat.aos_item(static_cast<int>(l[j])), cat.dim)});
      }
      w.push_back({static_cast<ItemId>(i), dot_simd(pv, cat.aos_item(i), cat.dim)});
      std::vector<Neighbor> keep;
      select_heuristic(cat, w, cap, keep);
      d = 0;
      for (const auto& kp : keep) l[d++] = kp.id;
    }
  }

  int n_ = 0;
  int dim_ = 0;
  int m_ = 16;
  int m_max_ = 16;
  int m_max0_ = 32;
  int ef_construction_ = 64;
  int max_level_ = 0;
  unsigned seed_ = 1;
  double level_mult_ = 0.36;
  ItemId entry_ = 0;

  std::vector<ItemId> links0_;
  std::vector<std::uint32_t> deg0_;
  std::vector<int> level_;
  std::vector<std::vector<ItemId>> upper_;
  std::vector<std::vector<std::uint32_t>> updeg_;

  mutable std::vector<std::mutex> node_locks_;
  mutable std::mutex global_mu_;
};

}  // namespace recserve
