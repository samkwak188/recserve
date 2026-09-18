// Recommendation quality on real embeddings, measured through the serving path.
//
// Everything else in this repo measures throughput on synthetic vectors, where
// recall@10 says something about high-dimensional geometry and nothing about
// recommendation quality. This runs the actual request path -- feature read,
// HNSW retrieve, rank, top-K -- over ALS factors trained on MovieLens, scores
// the result against each user's held-out future, and reports both the quality
// and the latency it was produced at.
//
// Protocol, matching standard implicit-feedback evaluation:
//   - per-user temporal split; the model never saw the held-out tail
//   - items the user already interacted with in train are filtered out of the
//     returned list before scoring, the way a real ranker filters them
//   - recall@k and NDCG@k against the held-out set
//   - retrieve-recall of the ANN list against the exact float32 top-k, so the
//     cost of approximation is separated from the cost of the model
#include "recserve/engine.hpp"
#include "recserve/metrics.hpp"
#include "recserve/rss.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdlib>

using namespace recserve;

namespace {

// query_row,item
std::unordered_map<int, std::vector<ItemId>> load_pairs(const std::string& path, bool* ok) {
  std::unordered_map<int, std::vector<ItemId>> m;
  std::ifstream in(path);
  *ok = static_cast<bool>(in);
  if (!*ok) return m;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (header) {
      header = false;
      if (line.find("query_row") != std::string::npos) continue;
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    int row = 0;
    long item = 0;
    if (!(iss >> row >> item)) continue;
    m[row].push_back(static_cast<ItemId>(item));
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  std::string catalog, index_path, queries, test_csv, train_csv, kernel_arg = "simd";
  int k = 10, ef = 128, retrieve_k = 256, max_users = 0;
  bool json = false, brute = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--catalog" && i + 1 < argc) catalog = argv[++i];
    else if (a == "--index" && i + 1 < argc) index_path = argv[++i];
    else if (a == "--queries" && i + 1 < argc) queries = argv[++i];
    else if (a == "--test" && i + 1 < argc) test_csv = argv[++i];
    else if (a == "--train" && i + 1 < argc) train_csv = argv[++i];
    else if (a == "--kernel" && i + 1 < argc) kernel_arg = argv[++i];
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--ef" && i + 1 < argc) ef = std::atoi(argv[++i]);
    else if (a == "--retrieve-k" && i + 1 < argc) retrieve_k = std::atoi(argv[++i]);
    else if (a == "--max-users" && i + 1 < argc) max_users = std::atoi(argv[++i]);
    else if (a == "--brute") brute = true;
    else if (a == "--json") json = true;
    else {
      std::cerr << "unknown flag: " << a << "\n";
      return 2;
    }
  }
  if (catalog.empty() || queries.empty() || test_csv.empty()) {
    std::cerr << "usage: recserve_eval --catalog C --queries Q --test T "
                 "[--train S] [--index I] [--k K] [--ef E] [--kernel K] [--json]\n";
    return 2;
  }

  Engine e;
  e.cfg.kernel = kernel_from_string(kernel_arg);
  e.cfg.use_hnsw = !brute;
  e.cfg.ef_search = ef;
  if (!e.load_fixture(catalog, index_path, 1, 13)) {
    std::cerr << "failed to load catalog " << catalog << "\n";
    return 1;
  }
  if (!e.load_queries(queries)) {
    std::cerr << "failed to load queries " << queries << " (dim must match catalog)\n";
    return 1;
  }

  bool ok = false;
  auto gold = load_pairs(test_csv, &ok);
  if (!ok) {
    std::cerr << "failed to read " << test_csv << "\n";
    return 1;
  }
  std::unordered_map<int, std::vector<ItemId>> seen;
  if (!train_csv.empty()) {
    bool sok = false;
    seen = load_pairs(train_csv, &sok);
    if (!sok) std::cerr << "warning: could not read " << train_csv << "\n";
  }

  Histogram lat;
  double recall = 0, ndcg = 0, rrecall = 0;
  int n = 0, skipped = 0;
  std::size_t total_filtered = 0;

  for (const auto& kv : gold) {
    if (max_users > 0 && n >= max_users) break;
    const int row = kv.first;
    if (row < 0 || row >= e.n_queries()) {
      ++skipped;
      continue;
    }
    std::unordered_set<ItemId> hide;
    auto sit = seen.find(row);
    if (sit != seen.end()) hide.insert(sit->second.begin(), sit->second.end());

    // Ask for k plus the seen items so k survive the filter.
    const std::uint32_t want =
        std::min<std::uint32_t>(kMaxK, static_cast<std::uint32_t>(k + hide.size()));

    Request req;
    req.user_id = static_cast<UserId>(row);
    req.k = want;
    req.retrieve_k = static_cast<std::uint32_t>(std::max<std::size_t>(
        static_cast<std::size_t>(retrieve_k), want));
    req.timeout_us = 5'000'000;

    const auto t0 = now_us();
    const Response r = e.recommend_sync(req);
    lat.add(static_cast<double>(now_us() - t0));
    if (r.status != Status::Ok) {
      ++skipped;
      continue;
    }

    std::vector<ItemId> pred;
    pred.reserve(static_cast<std::size_t>(k));
    for (const auto& it : r.items) {
      if (hide.count(it.id)) {
        ++total_filtered;
        continue;
      }
      pred.push_back(it.id);
      if (static_cast<int>(pred.size()) >= k) break;
    }

    const std::unordered_set<ItemId> gset(kv.second.begin(), kv.second.end());
    recall += recall_at_k(pred, gset, k);
    ndcg += ndcg_at_k(pred, gset, k);

    // Approximation cost, separated from model quality: how much of the exact
    // float32 top-k did the served configuration actually find?
    const float* q = e.user_query(req.user_id);
    auto exact = e.index.brute(e.cat, q, nullptr, k, Kernel::Simd);
    std::unordered_set<ItemId> exact_ids;
    for (const auto& x : exact) exact_ids.insert(x.id);
    std::vector<ItemId> ann;
    for (const auto& it : r.items) {
      ann.push_back(it.id);
      if (static_cast<int>(ann.size()) >= k) break;
    }
    rrecall += recall_at_k(ann, exact_ids, k);
    ++n;
  }

  if (n == 0) {
    std::cerr << "no users evaluated\n";
    return 1;
  }
  recall /= n;
  ndcg /= n;
  rrecall /= n;

  if (json) {
    std::cout << std::setprecision(6) << "{"
              << "\"users\":" << n << ",\"skipped\":" << skipped << ",\"k\":" << k
              << ",\"items\":" << e.cat.n << ",\"dim\":" << e.cat.dim
              << ",\"kernel\":\"" << kernel_name(e.cfg.kernel) << "\""
              << ",\"ef\":" << ef << ",\"retrieve_k\":" << retrieve_k
              << ",\"brute\":" << (brute ? 1 : 0)
              << ",\"recall_at_k\":" << recall << ",\"ndcg_at_k\":" << ndcg
              << ",\"retrieve_recall\":" << rrecall
              << ",\"filtered_seen\":" << total_filtered
              << ",\"p50_us\":" << lat.percentile(0.50)
              << ",\"p99_us\":" << lat.percentile(0.99)
              << ",\"rss_mib\":" << static_cast<double>(process_rss_bytes()) / (1024.0 * 1024.0)
              << "}\n";
  } else {
    std::cout << "users=" << n << " items=" << e.cat.n << " k=" << k
              << " kernel=" << kernel_name(e.cfg.kernel) << " ef=" << ef
              << " recall@" << k << "=" << recall << " ndcg@" << k << "=" << ndcg
              << " retrieve_recall=" << rrecall << " p99_us=" << lat.percentile(0.99) << "\n";
  }
  return 0;
}
