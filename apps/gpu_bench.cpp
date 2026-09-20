#include "recserve/engine.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>

using namespace recserve;
int run(int argc, char** argv) {
  std::string catalog, index, backend = "cuda";
  int batch = 1, iterations = 30, warmup = 5, k = 10, ef = 128;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (i + 1 >= argc) throw std::invalid_argument("missing option value");
    std::string value = argv[++i];
    if (flag == "--catalog") catalog = value;
    else if (flag == "--index") index = value;
    else if (flag == "--backend") backend = value;
    else if (flag == "--batch") batch = std::stoi(value);
    else if (flag == "--iterations") iterations = std::stoi(value);
    else if (flag == "--warmup") warmup = std::stoi(value);
    else if (flag == "--k") k = std::stoi(value);
    else if (flag == "--ef") ef = std::stoi(value);
    else throw std::invalid_argument("unknown option " + flag);
  }
  if (batch < 1 || batch > 2048 || iterations < 1 || warmup < 0 || k < 1 || k > 512 || ef < k)
    throw std::invalid_argument("invalid benchmark parameters");
  Engine e;
  e.cfg.use_hnsw = backend == "hnsw";
  e.cfg.kernel = backend == "int8" ? Kernel::Int8 : Kernel::Simd;
  if (backend != "cuda" && backend != "simd" && backend != "int8" && backend != "hnsw")
    throw std::invalid_argument("unknown backend");
  if (catalog.empty() || (e.cfg.use_hnsw && index.empty()))
    throw std::invalid_argument("prebuilt catalog and HNSW index required");
  if (!e.load_fixture(catalog, index, 4096, 13) || k > e.cat.n)
    throw std::runtime_error("invalid catalog/index fixture");
  std::unique_ptr<GpuScorer> gpu;
  if (backend == "cuda") {
    std::string error;
    gpu = make_gpu_scorer(e.cat.aos.data(), e.cat.n, e.cat.dim, batch, &error);
    if (!gpu) throw std::runtime_error(error);
  }
  std::vector<float> queries(static_cast<std::size_t>(batch) * e.cat.dim);
  std::vector<ScoredItem> output(static_cast<std::size_t>(batch) * k);
  Histogram latency, h2d, compute, d2h;
  QuantizedQuery quant;
  double recall = 0;
  const int probes = 128;
  auto retrieve = [&](int count, GpuTiming* time) {
    if (gpu) gpu->topk(queries.data(), count, k, output.data(), time);
    else for (int q = 0; q < count; ++q) {
      const auto* query = queries.data() + static_cast<std::size_t>(q) * e.cat.dim;
      if (backend == "int8") quant.set(query, e.cat.dim);
      const auto* qq = backend == "int8" ? &quant : nullptr;
      auto result = e.cfg.use_hnsw ? e.index.retrieve(e.cat, query, qq, k, ef, e.cfg.kernel) :
                                    e.index.brute(e.cat, query, qq, k, e.cfg.kernel);
      for (int i = 0; i < k; ++i) output[static_cast<std::size_t>(q) * k + i] = {result[i].id, result[i].score};
    }
  };
  for (int iteration = -warmup; iteration < iterations; ++iteration) {
    for (int q = 0; q < batch; ++q)
      std::copy_n(e.user_query(static_cast<UserId>((iteration + warmup) * batch + q)), e.cat.dim,
                  queries.data() + static_cast<std::size_t>(q) * e.cat.dim);
    const auto start = now_ns();
    GpuTiming time;
    retrieve(batch, &time);
    if (iteration >= 0) {
      latency.add(static_cast<double>(now_ns()-start)/1000.0);
      h2d.add(time.h2d_ms * 1000);
      compute.add(time.compute_ms * 1000);
      d2h.add(time.d2h_ms * 1000);
    }
  }
  // Every backend and batch size uses the same independent quality queries.
  for (int first = 0; first < probes; first += batch) {
    const auto count = std::min(batch, probes - first);
    for (int q = 0; q < count; ++q)
      std::copy_n(e.user_query(static_cast<UserId>(3000 + first + q)), e.cat.dim,
                  queries.data() + static_cast<std::size_t>(q) * e.cat.dim);
    retrieve(count, nullptr);
    for (int q = 0; q < count; ++q) {
      auto exact = e.index.brute(e.cat, queries.data() + static_cast<std::size_t>(q) * e.cat.dim, nullptr, k, Kernel::Simd);
      for (int i = 0; i < k; ++i)
        for (const auto& item : exact) if (item.id == output[static_cast<std::size_t>(q) * k + i].id) recall += 1;
    }
  }
  std::cout << std::setprecision(9) << "{\"backend\":\"" << backend << "\",\"isa\":\"" << simd_isa()
            << "\",\"device\":\"" << (gpu ? gpu->device_name() : "CPU") << "\",\"items\":" << e.cat.n
            << ",\"dim\":" << e.cat.dim << ",\"batch\":" << batch << ",\"k\":" << k << ",\"ef\":" << ef
            << ",\"iterations\":" << iterations << ",\"warmup\":" << warmup
            << ",\"p50_batch_us\":" << latency.percentile(.5) << ",\"p95_batch_us\":" << latency.percentile(.95)
            << ",\"p99_batch_us\":" << latency.percentile(.99) << ",\"mean_batch_us\":" << latency.mean()
            << ",\"qps\":" << batch * 1e6 / latency.mean() << ",\"batch_cv\":" << latency.cv()
            << ",\"h2d_mean_us\":" << h2d.mean() << ",\"compute_mean_us\":" << compute.mean()
            << ",\"d2h_mean_us\":" << d2h.mean() << ",\"device_bytes\":" << (gpu ? gpu->device_bytes() : 0)
            << ",\"recall\":" << recall / (probes * k) << ",\"quality_probes\":" << probes << "}\n";
  return 0;
}
int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
