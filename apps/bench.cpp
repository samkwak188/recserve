// In-process latency/throughput/quality harness.
//
// The JSON emitted here is also the observation payload consumed by
// scripts/agent.py: every field the agent can react to is measured in one run,
// under one protocol (fixed warmup, N trials, CV reported), so an accepted
// change is never an artefact of comparing two different protocols.
#include "recserve/engine.hpp"
#include "recserve/metrics.hpp"
#include "recserve/rss.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iomanip>

using namespace recserve;

struct Trial {
  double p50 = 0, p95 = 0, p99 = 0, mean = 0, qps = 0;
  double retrieve_p99 = 0, score_p99 = 0, feature_p99 = 0, hops_mean = 0;
};

static Trial run_once(Engine& e, int nreq, int k, int retrieve_k, int warmup) {
  Histogram h, retrieve, score, feature, hops;
  Request q;
  q.k = static_cast<std::uint32_t>(k);
  q.retrieve_k = static_cast<std::uint32_t>(retrieve_k);
  q.timeout_us = 1'000'000;
  for (int i = 0; i < warmup; ++i) {
    q.user_id = static_cast<UserId>(i);
    (void)e.recommend_sync(q);
  }
  const auto wall0 = now_us();
  int ok = 0;
  for (int i = 0; i < nreq; ++i) {
    q.id = static_cast<RequestId>(i);
    q.user_id = static_cast<UserId>(i);
    const auto t0 = now_us();
    const auto r = e.recommend_sync(q);
    const auto dt = now_us() - t0;
    if (r.status == Status::Ok) {
      h.add(static_cast<double>(dt));
      retrieve.add(static_cast<double>(r.retrieve_us));
      score.add(static_cast<double>(r.score_us));
      feature.add(static_cast<double>(r.feature_us));
      hops.add(static_cast<double>(r.hops));
      ++ok;
    }
  }
  const auto wall = now_us() - wall0;
  Trial t;
  t.p50 = h.percentile(0.50);
  t.p95 = h.percentile(0.95);
  t.p99 = h.percentile(0.99);
  t.mean = h.mean();
  t.qps = wall > 0 ? (1e6 * static_cast<double>(ok) / static_cast<double>(wall)) : 0;
  t.retrieve_p99 = retrieve.percentile(0.99);
  t.score_p99 = score.percentile(0.99);
  t.feature_p99 = feature.percentile(0.99);
  t.hops_mean = hops.mean();
  return t;
}

static void usage() {
  std::cout << "recserve_bench [--items N] [--dim D] [--n REQ] [--k K] [--retrieve-k RK]\n"
               "               [--mode baseline|simd|soa|blocked|int8|pin] [--kernel NAME]\n"
               "               [--ef E] [--ef-construction E] [--m M] [--trials T]\n"
               "               [--warmup W] [--brute] [--pin] [--no-arena]\n"
               "               [--load-catalog P] [--load-index P] [--recall-probe N] [--json]\n";
}

int main(int argc, char** argv) {
  int items = 4096, dim = 64, nreq = 400, k = 10, retrieve_k = 64, trials = 3, warmup = 64;
  int ef = 64, ef_construction = 64, m = 16, recall_probe = 64, workers = 0, clusters = 0;
  int build_threads = 0;
  bool json = false, brute = false, pin = false, arena = true;
  std::string mode = "baseline", kernel_arg, load_catalog, load_index;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--n" && i + 1 < argc) nreq = std::atoi(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--retrieve-k" && i + 1 < argc) retrieve_k = std::atoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--kernel" && i + 1 < argc) kernel_arg = argv[++i];
    else if (a == "--ef" && i + 1 < argc) ef = std::atoi(argv[++i]);
    else if (a == "--ef-construction" && i + 1 < argc) ef_construction = std::atoi(argv[++i]);
    else if (a == "--m" && i + 1 < argc) m = std::atoi(argv[++i]);
    else if (a == "--build-threads" && i + 1 < argc) build_threads = std::atoi(argv[++i]);
    else if (a == "--workers" && i + 1 < argc) workers = std::atoi(argv[++i]);
    else if (a == "--trials" && i + 1 < argc) trials = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (a == "--recall-probe" && i + 1 < argc) recall_probe = std::atoi(argv[++i]);
    else if (a == "--clusters" && i + 1 < argc) clusters = std::atoi(argv[++i]);
    else if (a == "--load-catalog" && i + 1 < argc) load_catalog = argv[++i];
    else if (a == "--load-index" && i + 1 < argc) load_index = argv[++i];
    else if (a == "--brute") brute = true;
    else if (a == "--pin") pin = true;
    else if (a == "--no-arena") arena = false;
    else if (a == "--json") json = true;
    else if (a == "--help") { usage(); return 0; }
    else { std::cerr << "unknown flag: " << a << "\n"; usage(); return 2; }
  }

  // Mode presets keep the published board comparable across runs; explicit
  // flags win when the agent is driving.
  Kernel kern = Kernel::Scalar;
  if (mode == "simd") kern = Kernel::Simd;
  else if (mode == "soa") kern = Kernel::SoaStrided;
  else if (mode == "blocked") kern = Kernel::Blocked;
  else if (mode == "int8") kern = Kernel::Int8;
  else if (mode == "pin") { kern = Kernel::Simd; pin = true; }
  else if (mode == "arena") kern = Kernel::Simd;
  if (!kernel_arg.empty()) kern = kernel_from_string(kernel_arg);

  Engine e;
  e.cfg.kernel = kern;
  e.cfg.use_hnsw = !brute;
  e.cfg.use_arena = arena;
  e.cfg.pin_workers = pin;
  e.cfg.hnsw_m = m;
  e.cfg.ef_search = ef;
  e.cfg.ef_construction = ef_construction;
  e.cfg.workers = workers;
  e.cfg.build_threads = build_threads;

  const auto t_load0 = now_us();
  if (!load_catalog.empty()) {
    if (!e.load_fixture(load_catalog, load_index, 4096, 13)) {
      std::cerr << "failed to load catalog " << load_catalog << "\n";
      return 1;
    }
    items = e.cat.n;
    dim = e.cat.dim;
  } else {
    e.init_random(items, 4096, dim, 13, clusters);
  }
  const double load_s = static_cast<double>(now_us() - t_load0) / 1e6;

  Histogram p99s, qpss;
  Trial last{};
  for (int t = 0; t < trials; ++t) {
    last = run_once(e, nreq, k, retrieve_k, warmup);
    p99s.add(last.p99);
    qpss.add(last.qps);
    if (!json) {
      std::cout << "trial=" << (t + 1) << " mode=" << mode << " p50_us=" << last.p50
                << " p95_us=" << last.p95 << " p99_us=" << last.p99 << " qps=" << last.qps << "\n";
    }
  }

  const double recall = recall_probe > 0 ? e.measure_retrieve_recall(k, recall_probe) : -1.0;
  const double e2e_recall =
      recall_probe > 0 ? e.measure_response_recall(k, retrieve_k, recall_probe) : -1.0;
  const double rss_mib = static_cast<double>(process_rss_bytes()) / (1024.0 * 1024.0);
  const double cat_mib = static_cast<double>(e.catalog_bytes()) / (1024.0 * 1024.0);

  if (json) {
    std::cout << std::setprecision(6)
              << "{\"mode\":\"" << mode << "\",\"kernel\":\"" << kernel_name(kern)
              << "\",\"isa\":\"" << simd_isa() << "\",\"items\":" << items << ",\"dim\":" << dim
              << ",\"n\":" << nreq << ",\"k\":" << k << ",\"retrieve_k\":" << retrieve_k
              << ",\"ef\":" << ef << ",\"m\":" << m << ",\"clusters\":" << clusters << ",\"brute\":" << (brute ? 1 : 0)
              << ",\"pin\":" << (pin ? 1 : 0) << ",\"arena\":" << (arena ? 1 : 0)
              << ",\"p50_us\":" << last.p50 << ",\"p95_us\":" << last.p95
              << ",\"p99_mean_us\":" << p99s.mean() << ",\"p99_cv\":" << p99s.cv()
              << ",\"qps_mean\":" << qpss.mean()
              << ",\"retrieve_p99_us\":" << last.retrieve_p99
              << ",\"score_p99_us\":" << last.score_p99
              << ",\"feature_p99_us\":" << last.feature_p99
              << ",\"hops_mean\":" << last.hops_mean
              << ",\"retrieve_recall\":" << recall
              << ",\"response_recall\":" << e2e_recall
              << ",\"rss_mib\":" << rss_mib << ",\"catalog_mib\":" << cat_mib
              << ",\"build_s\":" << e.build_stats.seconds
              << ",\"build_threads\":" << e.build_stats.threads
              << ",\"load_s\":" << load_s
              << ",\"avg_degree\":" << e.index.avg_degree() << "}\n";
  } else {
    std::cout << "mode=" << mode << " kernel=" << kernel_name(kern) << " isa=" << simd_isa()
              << " p99_mean_us=" << p99s.mean() << " p99_cv=" << p99s.cv()
              << " qps_mean=" << qpss.mean() << " retrieve_recall=" << recall << " response_recall=" << e2e_recall
              << " rss_mib=" << rss_mib << "\n";
  }
  return 0;
}
