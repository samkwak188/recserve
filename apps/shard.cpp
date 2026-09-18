// Sharded retrieval, and the tail amplification it causes.
//
// A single-host index is the one structural thing this project was missing. Real
// recommendation catalogs do not fit on one machine, so the catalog is split
// across N shards, every query is scattered to all of them, and the partial
// top-K lists are merged.
//
// The point of measuring it is not that scatter-gather works -- it obviously
// does -- but what it costs. A request is not finished until its SLOWEST shard
// replies, so end-to-end latency is the maximum of N samples, not the mean.
// With per-shard p99 = p, the probability that a 10-shard request avoids every
// shard's tail is 0.99^10 = 0.904, so roughly one request in ten pays a p99
// shard. This is Dean & Barroso, "The Tail at Scale", CACM 56(2), 2013, and it
// is why adding shards can make p99 worse even as each shard gets faster.
//
// Honest scope: the shards are threads in one process, so there is no network,
// no separate failure domain and no cross-host variance. That makes this a
// LOWER BOUND on the amplification a real deployment would see -- the effect is
// structural and shows up even without a network, which is the point.
#include "recserve/engine.hpp"
#include "recserve/metrics.hpp"
#include "recserve/rss.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <unordered_set>

using namespace recserve;

namespace {

// One shard: a slice of the catalog with its own index. Item ids are global, so
// merged results need no remapping.
struct Shard {
  Engine engine;
  int base = 0;  // global id of local item 0
};

// Scatter to every shard in parallel, wait for all, merge. The fan-out uses a
// persistent pool rather than spawning threads per request: thread creation is
// tens of microseconds and would dominate a 100 us request.
class ShardPool {
 public:
  void start(int n) {
    n_ = n;
    threads_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      threads_.emplace_back([this, i]() { worker(i); });
    }
  }

  void stop() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
    threads_.clear();
  }

  // Runs fn(shard) on every shard, returns when all have finished.
  void run(const std::function<void(int)>& fn) {
    {
      std::lock_guard<std::mutex> g(mu_);
      job_ = &fn;
      pending_ = n_;
      ++generation_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return pending_ == 0; });
    job_ = nullptr;
  }

  ~ShardPool() { stop(); }

 private:
  void worker(int id) {
    std::uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* job = nullptr;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        job = job_;
      }
      if (job) (*job)(id);
      {
        std::lock_guard<std::mutex> g(mu_);
        if (--pending_ == 0) done_cv_.notify_one();
      }
    }
  }

  int n_ = 0;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  std::uint64_t generation_ = 0;
  int pending_ = 0;
  bool stop_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  std::string catalog, kernel_arg = "simd";
  int shards = 8, k = 10, ef = 64, retrieve_k = 64, n_req = 2000, warmup = 128;
  int items = 1'000'000, dim = 64, clusters = 4096, build_threads = 0;
  bool json = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--catalog" && i + 1 < argc) catalog = argv[++i];
    else if (a == "--shards" && i + 1 < argc) shards = std::atoi(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--ef" && i + 1 < argc) ef = std::atoi(argv[++i]);
    else if (a == "--retrieve-k" && i + 1 < argc) retrieve_k = std::atoi(argv[++i]);
    else if (a == "--n" && i + 1 < argc) n_req = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--clusters" && i + 1 < argc) clusters = std::atoi(argv[++i]);
    else if (a == "--build-threads" && i + 1 < argc) build_threads = std::atoi(argv[++i]);
    else if (a == "--kernel" && i + 1 < argc) kernel_arg = argv[++i];
    else if (a == "--json") json = true;
    else {
      std::cerr << "unknown flag: " << a << "\n";
      return 2;
    }
  }
  shards = std::max(1, shards);

  // Build the whole catalog once, then slice it. Slicing a single generated
  // catalog keeps the union of the shards identical to the unsharded baseline,
  // so any quality difference is caused by sharding and not by different data.
  Engine full;
  full.cfg.kernel = kernel_from_string(kernel_arg);
  full.cfg.use_hnsw = true;
  full.cfg.ef_search = ef;
  full.cfg.build_threads = build_threads;
  if (!catalog.empty()) {
    if (!full.load_fixture(catalog, "", 4096, 13)) {
      std::cerr << "failed to load " << catalog << "\n";
      return 1;
    }
  } else {
    full.init_random(items, 4096, dim, 13, clusters);
  }
  items = full.cat.n;
  dim = full.cat.dim;

  std::vector<Shard> sh(static_cast<std::size_t>(shards));
  const int per = (items + shards - 1) / shards;
  const auto t_build0 = now_us();
  for (int s = 0; s < shards; ++s) {
    const int base = s * per;
    const int cnt = std::min(per, items - base);
    Shard& x = sh[static_cast<std::size_t>(s)];
    x.base = base;
    x.engine.cfg.kernel = full.cfg.kernel;
    x.engine.cfg.use_hnsw = true;
    x.engine.cfg.ef_search = ef;
    x.engine.cfg.build_threads = build_threads;
    x.engine.cfg.dim = dim;
    x.engine.cat.resize(cnt, dim);
    std::copy(full.cat.aos.begin() + static_cast<std::ptrdiff_t>(base) * dim,
              full.cat.aos.begin() + static_cast<std::ptrdiff_t>(base + cnt) * dim,
              x.engine.cat.aos.begin());
    x.engine.prepare_layouts();
    x.engine.build_index();
  }
  const double build_s = static_cast<double>(now_us() - t_build0) / 1e6;

  ShardPool pool;
  pool.start(shards);

  Histogram e2e, slowest, fastest, merge_us;
  std::vector<Histogram> per_shard(static_cast<std::size_t>(shards));
  std::vector<std::vector<Neighbor>> results(static_cast<std::size_t>(shards));
  std::vector<double> shard_us(static_cast<std::size_t>(shards), 0.0);
  double recall_vs_single = 0;
  int measured = 0;

  const int rk = std::max(k, retrieve_k);
  for (int i = 0; i < warmup + n_req; ++i) {
    const UserId u = static_cast<UserId>(i);
    const float* q = full.user_query(u);
    if (!q) break;

    const auto t0 = now_us();
    auto fn = std::function<void(int)>([&](int s) {
      const auto s0 = now_us();
      Shard& x = sh[static_cast<std::size_t>(s)];
      // Each shard must return k so the merge can pick a true global top-k.
      results[static_cast<std::size_t>(s)] =
          x.engine.index.retrieve(x.engine.cat, q, nullptr, rk, ef, x.engine.cfg.kernel);
      for (auto& nb : results[static_cast<std::size_t>(s)]) {
        nb.id += static_cast<ItemId>(x.base);  // local id -> global id
      }
      shard_us[static_cast<std::size_t>(s)] = static_cast<double>(now_us() - s0);
    });
    pool.run(fn);

    const auto tm0 = now_us();
    std::vector<Neighbor> merged;
    merged.reserve(static_cast<std::size_t>(shards) * static_cast<std::size_t>(rk));
    for (const auto& r : results) merged.insert(merged.end(), r.begin(), r.end());
    const std::size_t want = std::min<std::size_t>(static_cast<std::size_t>(k), merged.size());
    std::partial_sort(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(want),
                      merged.end(),
                      [](const Neighbor& a, const Neighbor& b) { return a.score > b.score; });
    merged.resize(want);
    const double merge = static_cast<double>(now_us() - tm0);
    const double total = static_cast<double>(now_us() - t0);

    if (i < warmup) continue;

    const double smax = *std::max_element(shard_us.begin(), shard_us.end());
    const double smin = *std::min_element(shard_us.begin(), shard_us.end());
    e2e.add(total);
    slowest.add(smax);
    fastest.add(smin);
    merge_us.add(merge);
    for (int s = 0; s < shards; ++s) {
      per_shard[static_cast<std::size_t>(s)].add(shard_us[static_cast<std::size_t>(s)]);
    }

    // Quality is measured against the EXACT global top-k, so both the
    // approximation and any loss from partitioning show up in one number.
    auto exact = full.index.brute(full.cat, q, nullptr, k, Kernel::Simd);
    std::unordered_set<ItemId> gold;
    for (const auto& x : exact) gold.insert(x.id);
    std::vector<ItemId> got;
    for (const auto& m : merged) got.push_back(m.id);
    recall_vs_single += recall_at_k(got, gold, k);
    ++measured;
  }
  pool.stop();

  if (measured == 0) {
    std::cerr << "no requests measured\n";
    return 1;
  }
  recall_vs_single /= measured;

  // Pool the per-shard samples: this is the distribution one shard shows, which
  // is what the amplification is measured against.
  Histogram pooled;
  for (const auto& h : per_shard) {
    for (double x : h.samples) pooled.add(x);
  }
  const double shard_p99 = pooled.percentile(0.99);
  const double e2e_p99 = e2e.percentile(0.99);

  if (json) {
    std::cout << std::setprecision(6) << "{"
              << "\"shards\":" << shards << ",\"items\":" << items << ",\"dim\":" << dim
              << ",\"items_per_shard\":" << per << ",\"k\":" << k << ",\"ef\":" << ef
              << ",\"retrieve_k\":" << rk << ",\"requests\":" << measured
              << ",\"kernel\":\"" << kernel_name(full.cfg.kernel) << "\""
              << ",\"shard_p50_us\":" << pooled.percentile(0.50)
              << ",\"shard_p99_us\":" << shard_p99
              << ",\"slowest_shard_p50_us\":" << slowest.percentile(0.50)
              << ",\"slowest_shard_p99_us\":" << slowest.percentile(0.99)
              << ",\"fastest_shard_p50_us\":" << fastest.percentile(0.50)
              << ",\"e2e_p50_us\":" << e2e.percentile(0.50)
              << ",\"e2e_p99_us\":" << e2e_p99
              << ",\"e2e_p999_us\":" << e2e.percentile(0.999)
              << ",\"merge_p99_us\":" << merge_us.percentile(0.99)
              << ",\"tail_amplification\":" << (shard_p99 > 0 ? e2e_p99 / shard_p99 : 0)
              << ",\"recall_vs_exact\":" << recall_vs_single
              << ",\"build_s\":" << build_s
              << ",\"rss_mib\":" << static_cast<double>(process_rss_bytes()) / (1024.0 * 1024.0)
              << "}\n";
  } else {
    std::cout << "shards=" << shards << " items=" << items
              << " shard_p99=" << shard_p99 << "us e2e_p99=" << e2e_p99
              << "us amplification=" << (shard_p99 > 0 ? e2e_p99 / shard_p99 : 0) << "x"
              << " recall@" << k << "=" << recall_vs_single
              << " merge_p99=" << merge_us.percentile(0.99) << "us\n";
  }
  return 0;
}
