// Build a catalog + HNSW index once and write them to disk.
//
// A 1M-item index takes minutes to build and every bench mode would otherwise
// rebuild it. Serving hosts load an index snapshot rather than building one, so
// this also makes the benchmark protocol match how the thing would actually run.
#include "recserve/engine.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <string>

using namespace recserve;

int main(int argc, char** argv) {
  int items = 1'000'000, dim = 64, clusters = 1024, m = 16, ef_construction = 100;
  int build_threads = 0, n_queries = 4096;
  unsigned seed = 13;
  std::string out_catalog = "data/catalog.bin", out_index = "data/index.bin", out_queries;
  std::string in_catalog;
  bool skip_index = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--clusters" && i + 1 < argc) clusters = std::atoi(argv[++i]);
    else if (a == "--m" && i + 1 < argc) m = std::atoi(argv[++i]);
    else if (a == "--ef-construction" && i + 1 < argc) ef_construction = std::atoi(argv[++i]);
    else if (a == "--build-threads" && i + 1 < argc) build_threads = std::atoi(argv[++i]);
    else if (a == "--queries" && i + 1 < argc) n_queries = std::atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = static_cast<unsigned>(std::atoi(argv[++i]));
    else if (a == "--out-catalog" && i + 1 < argc) out_catalog = argv[++i];
    else if (a == "--out-index" && i + 1 < argc) out_index = argv[++i];
    else if (a == "--out-queries" && i + 1 < argc) out_queries = argv[++i];
    else if (a == "--in-catalog" && i + 1 < argc) in_catalog = argv[++i];
    else if (a == "--no-index") skip_index = true;
  }

  Engine e;
  e.cfg.kernel = Kernel::Simd;
  e.cfg.use_hnsw = !skip_index;
  e.cfg.hnsw_m = m;
  e.cfg.ef_construction = ef_construction;
  e.cfg.build_threads = build_threads;

  if (!in_catalog.empty()) {
    // Build an index over a catalog that already exists (trained ALS item
    // factors, say) instead of generating one. The index is written to disk so
    // every later measurement uses the same graph: the parallel build is
    // order-dependent, so rebuilding per run would put that spread underneath
    // every recall comparison.
    std::cerr << "building an index over " << in_catalog << "\n";
    if (!e.load_fixture(in_catalog, "", n_queries, seed)) {
      std::cerr << "failed to load " << in_catalog << "\n";
      return 1;
    }
    items = e.cat.n;
    dim = e.cat.dim;
    out_catalog = in_catalog;
  } else {
    std::cerr << "generating " << items << " x " << dim << " (clusters=" << clusters
              << ")\n";
    e.init_random(items, n_queries, dim, seed, clusters);
  }

  const auto p = std::filesystem::path(out_catalog).parent_path();
  if (!p.empty()) std::filesystem::create_directories(p);
  // An input catalog is already on disk; do not rewrite it.
  if (in_catalog.empty() && !e.cat.save(out_catalog)) {
    std::cerr << "catalog save failed\n";
    return 1;
  }
  if (!skip_index && !e.index.save(out_index)) {
    std::cerr << "index save failed\n";
    return 1;
  }
  if (!out_queries.empty()) {
    std::ofstream o(out_queries, std::ios::binary);
    const std::uint32_t magic = 0x51525931u;  // QRY1
    const int nq = e.n_queries();
    o.write(reinterpret_cast<const char*>(&magic), 4);
    o.write(reinterpret_cast<const char*>(&nq), 4);
    o.write(reinterpret_cast<const char*>(&dim), 4);
    for (int i = 0; i < nq; ++i) {
      o.write(reinterpret_cast<const char*>(e.user_query(static_cast<UserId>(i))),
              static_cast<std::streamsize>(static_cast<std::size_t>(dim) * sizeof(float)));
    }
  }

  std::cout << "{\"items\":" << items << ",\"dim\":" << dim << ",\"clusters\":" << clusters
            << ",\"m\":" << m << ",\"ef_construction\":" << ef_construction
            << ",\"build_s\":" << e.build_stats.seconds
            << ",\"build_threads\":" << e.build_stats.threads
            << ",\"max_level\":" << e.build_stats.max_level
            << ",\"dist_calls\":" << e.build_stats.dist_calls
            << ",\"avg_degree\":" << e.index.avg_degree()
            << ",\"graph_mib\":" << static_cast<double>(e.index.graph_bytes()) / (1024.0 * 1024.0)
            << ",\"catalog_mib\":" << static_cast<double>(e.catalog_bytes()) / (1024.0 * 1024.0)
            << "}\n";
  return 0;
}
