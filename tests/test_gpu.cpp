#include "recserve/engine.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>

using namespace recserve;
void require(bool ok, const char* reason) {
  if (!ok) throw std::runtime_error(reason);
}

int main() {
  try {
#ifndef RECSERVE_HAS_CUDA
    std::string error;
    require(!make_gpu_scorer(nullptr, 0, 0, 1, &error) && !error.empty(), "missing CUDA must report unavailable");
    std::cout << "CUDA unavailable stub verified\n";
#else
    for (const auto& [n, dim] : {std::pair{301, 17}, std::pair{65539, 65}}) {
      Engine e;
      e.cfg.use_hnsw = false;
      e.init_random(n, 16, dim, 42);
      std::string error;
      auto gpu = make_gpu_scorer(e.cat.aos.data(), n, dim, 8, &error);
      require(gpu != nullptr, error.c_str());
      const auto bytes = gpu->device_bytes();
      for (int batch : {1, 3, 8}) {
        for (int k : {1, 10, std::min(n, 512)}) {
          std::vector<float> queries(static_cast<std::size_t>(batch) * dim);
          for (int q = 0; q < batch; ++q) std::copy_n(e.user_query(q), dim, queries.data() + q * dim);
          std::vector<ScoredItem> got(static_cast<std::size_t>(batch) * k);
          gpu->topk(queries.data(), batch, k, got.data());
          for (int q = 0; q < batch; ++q) {
            auto expected = e.index.brute(e.cat, queries.data() + q * dim, nullptr, k, Kernel::Simd);
            for (int i = 0; i < k; ++i) {
              const auto& actual = got[static_cast<std::size_t>(q) * k + i];
              require(actual.id == expected[i].id, "GPU top-k IDs differ from CPU");
              require(std::fabs(actual.score - expected[i].score) < 2e-5f, "GPU score tolerance");
            }
          }
          require(gpu->device_bytes() == bytes, "GPU allocation changed during query");
        }
      }
      std::atomic<int> failed{0};
      std::vector<std::thread> threads;
      for (int t = 0; t < 4; ++t) threads.emplace_back([&, t] {
        try {
          auto expected = e.index.brute(e.cat, e.user_query(t), nullptr, 10, Kernel::Simd);
          for (int repeat = 0; repeat < 4; ++repeat) {
            ScoredItem out[10];
            gpu->topk(e.user_query(t), 1, 10, out);
            require(out[0].id == expected[0].id, "concurrent GPU response mismatch");
          }
        } catch (...) { ++failed; }
      });
      for (auto& t : threads) t.join();
      require(failed == 0, "concurrent GPU calls failed");
      ScoredItem out[10];
      bool rejected = false;
      try { gpu->topk(e.user_query(0), 9, 10, out); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "batch limit must fail");
      std::vector<float> invalid(dim, std::numeric_limits<float>::quiet_NaN());
      rejected = false;
      try { gpu->topk(invalid.data(), 1, 10, out); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "NaN query must fail");
      std::cout << "GPU oracle passed: " << gpu->device_name() << " n=" << n << " dim=" << dim << '\n';
    }
    std::vector<float> zeros(1030 * 8, 0.f), query(8, 0.f);
    std::string error;
    auto ties = make_gpu_scorer(zeros.data(), 1030, 8, 1, &error);
    require(ties != nullptr, error.c_str());
    ScoredItem out[10];
    ties->topk(query.data(), 1, 10, out);
    for (unsigned i = 0; i < 10; ++i) require(out[i].id == i && out[i].score == 0, "tie ordering");
    std::cout << "GPU correctness, tails, ties, limits and concurrency verified\n";
#endif
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
