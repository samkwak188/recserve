#include "recserve/gpu_batcher.hpp"
#include <iostream>

using namespace recserve;
void require(bool condition) { if (!condition) throw std::runtime_error("batcher assertion failed"); }
class TestScorer : public GpuScorer {
 public:
  TestScorer(Engine& engine, bool fail) : engine_(engine), fail_(fail) {}
  void topk(const float* queries, int batch, int k, ScoredItem* output, GpuTiming*) override {
    if (fail_) throw std::runtime_error("injected device failure");
    for (int i = 0; i < batch; ++i) {
      auto exact = engine_.index.brute(engine_.cat, queries + static_cast<std::size_t>(i) * engine_.cat.dim, nullptr, k, Kernel::Simd);
      for (int j = 0; j < k; ++j) output[i * k + j] = {exact[j].id, exact[j].score};
    }
  }
  std::size_t device_bytes() const override { return 0; }
  std::string device_name() const override { return "test-only CPU oracle"; }
 private:
  Engine& engine_;
  bool fail_;
};

int main() {
  try {
    Engine engine;
    engine.cfg.use_hnsw = false; engine.cfg.kernel = Kernel::Simd;
    engine.init_random(128, 16, 17, 7);
    for (bool fail : {false, true}) {
      GpuBatcher batcher(engine, std::make_unique<TestScorer>(engine, fail), 8, 10000, 64);
      std::vector<std::future<Response>> pending;
      for (int i = 0; i < 8; ++i) {
        Request r; r.id = i; r.user_id = i; r.k = 10; r.retrieve_k = 32; r.timeout_us = 1000000;
        pending.push_back(batcher.submit(r));
      }
      for (int i = 0; i < 8; ++i) {
        auto result = pending[static_cast<std::size_t>(i)].get();
        Request r; r.user_id = i; r.k = 10; r.retrieve_k = 32;
        auto expected = engine.recommend_sync(r);
        require(result.status == Status::Ok && result.id == static_cast<unsigned>(i));
        for (std::size_t j = 0; j < expected.items.size(); ++j) require(result.items[j].id == expected.items[j].id);
      }
      if (fail) require(batcher.fallback == 8 && !batcher.gpu_healthy);
      else require(batcher.max_observed_batch >= 2 && batcher.gpu_queries == 8);
      Request invalid; invalid.k = 0;
      require(batcher.submit(invalid).get().status == Status::BadRequest);
      batcher.stop();
      Request valid;
      require(batcher.submit(valid).get().status == Status::LoadShed);
    }
    GpuBatcher limited(engine, std::make_unique<TestScorer>(engine, false), 8, 10000, 1);
    std::vector<std::future<Response>> pending;
    for (int i = 0; i < 1000; ++i) {
      Request r; r.timeout_us = 1;
      pending.push_back(limited.submit(r));
    }
    int expired = 0;
    for (auto& future : pending) {
      auto result = future.get();
      require(result.status == Status::Timeout || result.status == Status::LoadShed);
      expired += result.status == Status::Timeout;
    }
    require(expired > 0 && limited.shed > 0);
    std::cout << "batch formation, exact fallback, deadline expiry, bounded queue and stop: PASS\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
