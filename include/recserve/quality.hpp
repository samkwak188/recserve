#pragma once
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace recserve {

struct Interaction {
  UserId user = 0;
  ItemId item = 0;
  std::uint64_t ts = 0;
  float watch = 0.f;
};

inline std::vector<Interaction> load_interactions_csv(const std::string& path) {
  std::ifstream in(path);
  std::vector<Interaction> out;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (header) {
      header = false;
      if (line.find("user") != std::string::npos) continue;
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    Interaction r;
    if (!(iss >> r.user >> r.item >> r.ts)) continue;
    iss >> r.watch;
    out.push_back(r);
  }
  return out;
}

inline void temporal_split(const std::vector<Interaction>& all, double train_frac,
                           std::vector<Interaction>& train, std::vector<Interaction>& test) {
  auto sorted = all;
  std::sort(sorted.begin(), sorted.end(),
            [](const Interaction& a, const Interaction& b) { return a.ts < b.ts; });
  auto cut = static_cast<std::size_t>(sorted.size() * train_frac);
  train.assign(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(cut));
  test.assign(sorted.begin() + static_cast<std::ptrdiff_t>(cut), sorted.end());
}

inline double recall_at_k(const std::vector<ItemId>& pred, const std::unordered_set<ItemId>& gold,
                          int k) {
  if (gold.empty()) return 0;
  int hit = 0;
  int n = std::min(k, static_cast<int>(pred.size()));
  for (int i = 0; i < n; ++i) {
    if (gold.count(pred[i])) ++hit;
  }
  return static_cast<double>(hit) / static_cast<double>(gold.size());
}

inline double ndcg_at_k(const std::vector<ItemId>& pred, const std::unordered_set<ItemId>& gold,
                        int k) {
  double dcg = 0;
  int n = std::min(k, static_cast<int>(pred.size()));
  for (int i = 0; i < n; ++i) {
    if (gold.count(pred[i])) dcg += 1.0 / std::log2(static_cast<double>(i) + 2.0);
  }
  double idcg = 0;
  int ideal = std::min(k, static_cast<int>(gold.size()));
  for (int i = 0; i < ideal; ++i) idcg += 1.0 / std::log2(static_cast<double>(i) + 2.0);
  return idcg == 0 ? 0 : dcg / idcg;
}

struct QualityReport {
  double recall50 = 0;
  double ndcg50 = 0;
  double retrieve_recall = 0;
  int n_users = 0;
};

}  // namespace recserve
