#include "recserve/gpu_batcher.hpp"
#include <iostream>
#include <sstream>

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

struct Probe {
  std::promise<void> entered;
  std::shared_future<void> release;
  std::vector<int> batches;
  std::vector<int> topks;
  std::vector<std::thread::id> threads;
};
class ProbeScorer : public TestScorer {
 public:
  ProbeScorer(Engine& engine, Probe& probe, bool fail = false) : TestScorer(engine, fail), probe_(probe) {}
  void topk(const float* queries, int batch, int k, ScoredItem* output, GpuTiming* timing) override {
    probe_.batches.push_back(batch);
    probe_.topks.push_back(k);
    probe_.threads.push_back(std::this_thread::get_id());
    if (probe_.batches.size() == 1) {
      probe_.entered.set_value();
      if (probe_.release.valid()) probe_.release.wait();
    }
    TestScorer::topk(queries, batch, k, output, timing);
  }
 private:
  Probe& probe_;
};

void test_warmup(Engine& engine) {
  Probe cold;
  {
    GpuBatcher batcher(engine, std::make_unique<ProbeScorer>(engine, cold), 4, 0, 64);
    require(cold.batches.empty() && batcher.warmup_batches == 0);
  }
  Probe warm;
  std::promise<void> release;
  warm.release = release.get_future().share();
  auto entered = warm.entered.get_future();
  auto startup = std::async(std::launch::async, [&] {
    return std::make_unique<GpuBatcher>(engine, std::make_unique<ProbeScorer>(engine, warm), 4, 0, 64, true);
  });
  const bool reached = entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  const bool held = startup.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout;
  release.set_value(); // Release before assertions so test failure cannot strand the worker.
  auto batcher = startup.get();
  require(reached && held);
  require(warm.batches == std::vector<int>({1, 2, 3, 4}));
  for (auto k : warm.topks) require(k == 128);
  require(batcher->warmup_batches == 4 && batcher->batches == 0 && batcher->gpu_queries == 0);
  require(batcher->fallback == 0 && batcher->first_batch_wall_us == 0);
  Request request; request.k = 10; request.retrieve_k = 32; request.timeout_us = 1000000;
  require(batcher->submit(request).get().status == Status::Ok);
  require(batcher->batches == 1 && batcher->gpu_queries == 1 && warm.batches.size() == 5);
  for (auto thread : warm.threads) require(thread == warm.threads[0] && thread != std::this_thread::get_id());
  batcher->stop();

  Probe failed;
  bool rejected = false;
  try {
    GpuBatcher broken(engine, std::make_unique<ProbeScorer>(engine, failed, true), 4, 0, 64, true);
  } catch (const std::runtime_error& error) { rejected = std::string(error.what()) == "injected device failure"; }
  require(rejected && failed.batches.size() == 1);
}

int main() {
  try {
    DurationHistogram histogram;
    for (auto value : {0u, 1u, 2u, 3u, 131073u}) histogram.observe(value);
    std::ostringstream rendered;
    histogram.write(rendered, "duration");
    require(rendered.str().find("duration_bucket{le=\"1\"} 2\n") != std::string::npos);
    require(rendered.str().find("duration_bucket{le=\"4\"} 4\n") != std::string::npos);
    require(rendered.str().find("duration_bucket{le=\"+Inf\"} 5\n") != std::string::npos);
    require(rendered.str().find("duration_sum 131079\n") != std::string::npos);
    Engine engine;
    engine.cfg.use_hnsw = false; engine.cfg.kernel = Kernel::Simd;
    engine.init_random(128, 16, 17, 7);
    test_warmup(engine);
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
