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
};

static Trial run_once(Engine& e, int nreq, int k, int retrieve_k) {
  Histogram h;
  Request q;
  q.k = static_cast<std::uint32_t>(k);
  q.retrieve_k = static_cast<std::uint32_t>(retrieve_k);
  q.timeout_us = 1'000'000;
  for (int i = 0; i < 64; ++i) {
    q.user_id = static_cast<UserId>(i % 64);
    (void)e.recommend_sync(q);
  }
  auto wall0 = now_us();
  int ok = 0;
  for (int i = 0; i < nreq; ++i) {
    q.id = static_cast<RequestId>(i);
    q.user_id = static_cast<UserId>(i % 64);
    auto t0 = now_us();
    auto r = e.recommend_sync(q);
    auto dt = now_us() - t0;
    if (r.status == Status::Ok) {
      h.add(static_cast<double>(dt));
      ++ok;
    }
  }
  auto wall = now_us() - wall0;
  Trial t;
  t.p50 = h.percentile(0.50);
  t.p95 = h.percentile(0.95);
  t.p99 = h.percentile(0.99);
  t.mean = h.mean();
  t.qps = wall > 0 ? (1e6 * static_cast<double>(ok) / static_cast<double>(wall)) : 0;
  return t;
}

int main(int argc, char** argv) {
  int items = 4096;
  int dim = 64;
  int nreq = 400;
  int k = 10;
  int retrieve_k = 64;
  int trials = 3;
  std::string mode = "baseline";
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--n" && i + 1 < argc) nreq = std::atoi(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--retrieve-k" && i + 1 < argc) retrieve_k = std::atoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--trials" && i + 1 < argc) trials = std::atoi(argv[++i]);
    else if (a == "--json") json = true;
  }

  Engine e;
  e.cfg.use_hnsw = true;
  e.cfg.use_arena = mode != "baseline";
  e.cfg.use_soa = mode == "soa" || mode == "simd" || mode == "int8" || mode == "pin";
  e.cfg.use_simd = mode == "simd" || mode == "int8" || mode == "pin";
  e.cfg.use_flat_features = mode != "hash";
  e.cfg.pin_workers = mode == "pin";
  e.cfg.dtype = mode == "int8" ? DType::Int8 : DType::Float32;
  e.init_random(items, 64, dim, 13);
  if (mode == "int8") e.cat.quantize_i8();

  Histogram p99s, qpss;
  Trial last{};
  for (int t = 0; t < trials; ++t) {
    last = run_once(e, nreq, k, retrieve_k);
    p99s.add(last.p99);
    qpss.add(last.qps);
    if (!json) {
      std::cout << "trial=" << (t + 1) << " mode=" << mode << " p50_us=" << last.p50
                << " p95_us=" << last.p95 << " p99_us=" << last.p99 << " qps=" << last.qps << "\n";
    }
  }
  double rss_mib = static_cast<double>(process_rss_bytes()) / (1024.0 * 1024.0);
  if (json) {
    std::cout << std::setprecision(6) << "{\"mode\":\"" << mode << "\",\"items\":" << items
              << ",\"dim\":" << dim << ",\"n\":" << nreq << ",\"k\":" << k
              << ",\"retrieve_k\":" << retrieve_k << ",\"p50_us\":" << last.p50
              << ",\"p95_us\":" << last.p95 << ",\"p99_mean_us\":" << p99s.mean()
              << ",\"p99_cv\":" << p99s.cv() << ",\"qps_mean\":" << qpss.mean()
              << ",\"rss_mib\":" << rss_mib << "}\n";
  } else {
    std::cout << "mode=" << mode << " p99_mean_us=" << p99s.mean() << " p99_cv=" << p99s.cv()
              << " qps_mean=" << qpss.mean() << " rss_mib=" << rss_mib << "\n";
  }
  return 0;
}
