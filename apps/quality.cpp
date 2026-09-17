#include "recserve/engine.hpp"
#include "recserve/quality.hpp"
#include <iostream>
#include <cstdlib>
#include <algorithm>

using namespace recserve;

int main(int argc, char** argv) {
  std::string csv = "data/quality_fixture.csv";
  int n_items = 64;
  int dim = 16;
  int k = 10;
  double min_recall = 0.0;
  bool json = false;
  bool use_int8 = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (a == "--items" && i + 1 < argc) n_items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--min-recall" && i + 1 < argc) min_recall = std::atof(argv[++i]);
    else if (a == "--json") json = true;
    else if (a == "--int8") use_int8 = true;
  }

  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.ef_search = 64;
  auto rows = load_interactions_csv(csv);
  if (!rows.empty()) {
    int max_item = 0, max_user = 0;
    for (auto& r : rows) {
      max_item = std::max(max_item, static_cast<int>(r.item));
      max_user = std::max(max_user, static_cast<int>(r.user));
    }
    n_items = std::max(n_items, max_item + 1);
    e.cfg.kernel = use_int8 ? Kernel::Int8 : Kernel::Simd;
    e.init_random(n_items, max_user + 1, dim, 11);
    std::vector<Interaction> train, test;
    temporal_split(rows, 0.8, train, test);
    auto rep = e.eval_quality(train, test, k);
    if (json) {
      std::cout << "{\"users\":" << rep.n_users << ",\"k\":" << k << ",\"recall\":" << rep.recall50
                << ",\"ndcg\":" << rep.ndcg50 << ",\"retrieve_recall\":" << rep.retrieve_recall
                << ",\"int8\":" << (use_int8 ? "true" : "false") << "}\n";
    } else {
      std::cout << "users=" << rep.n_users << " recall@" << k << "=" << rep.recall50
                << " ndcg=" << rep.ndcg50 << " retrieve_recall=" << rep.retrieve_recall << "\n";
    }
    if (rep.recall50 + 1e-12 < min_recall) {
      std::cerr << "quality gate failed\n";
      return 2;
    }
    return 0;
  }
  e.init_random(n_items, 16, dim, 11);
  std::vector<Interaction> train, test;
  for (int u = 0; u < 16; ++u) {
    for (int t = 0; t < 8; ++t) {
      Interaction x{static_cast<UserId>(u), static_cast<ItemId>((u * 3 + t) % n_items),
                    static_cast<std::uint64_t>(t), 1.f};
      (t < 6 ? train : test).push_back(x);
    }
  }
  auto rep = e.eval_quality(train, test, k);
  std::cout << "fixture_users=" << rep.n_users << " recall@" << k << "=" << rep.recall50
            << " ndcg=" << rep.ndcg50 << " retrieve_recall=" << rep.retrieve_recall << "\n";
  if (rep.recall50 + 1e-12 < min_recall) return 2;
  return 0;
}
