#pragma once
#include "engine.hpp"
#include "histogram.hpp"

namespace recserve {

// One bounded scheduler owns GPU work. CPU fallback is exact, never a hidden
// quality change to ANN. Deadlines include queueing and batch formation.
class GpuBatcher {
 public:
  GpuBatcher(Engine& engine, std::unique_ptr<GpuScorer> scorer, int max_batch = 8,
             std::uint32_t wait_us = 100, std::size_t capacity = 64)
      : engine_(engine), scorer_(std::move(scorer)), max_batch_(max_batch), wait_us_(wait_us), capacity_(capacity) {
    if (!scorer_ || max_batch < 1 || max_batch > 64 || capacity < 1 || capacity > 4096 || wait_us > 10000 ||
        engine.cfg.use_hnsw || engine.cfg.kernel != Kernel::Simd)
      throw std::invalid_argument("batcher requires a scorer and CPU exact fallback engine");
    queries_.resize(static_cast<std::size_t>(max_batch) * engine.cat.dim);
    output_.resize(static_cast<std::size_t>(max_batch) * kMaxK);
    thread_ = std::thread([this] { loop(); });
  }
  ~GpuBatcher() { stop(); }
  GpuBatcher(const GpuBatcher&) = delete;
  GpuBatcher& operator=(const GpuBatcher&) = delete;

  std::future<Response> submit(const Request& request) {
    Job job{request, now_us(), {}};
    auto future = job.promise.get_future();
    std::lock_guard lock(mutex_);
    if (!request.k || request.k > kMaxK || request.retrieve_k > kMaxK || !request.timeout_us ||
        request.user_id >= static_cast<unsigned>(engine_.n_queries())) finish(job, Status::BadRequest);
    else if (stopping_ || jobs_.size() >= capacity_) { ++shed; finish(job, Status::LoadShed); }
    else { jobs_.push_back(std::move(job)); changed_.notify_one(); }
    return future;
  }
  void stop() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    changed_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  std::atomic<std::uint64_t> batches{0}, gpu_queries{0}, fallback{0}, expired{0}, shed{0}, max_observed_batch{0};
  std::atomic<bool> gpu_healthy{true};
  DurationHistogram queue_time, gpu_wall_time, h2d_time, device_compute_time, d2h_time;
  std::atomic<std::uint64_t> first_batch_wall_us{0}, first_batch_device_us{0};

 private:
  struct Job { Request request; std::uint64_t submitted; std::promise<Response> promise; };
  void finish(Job& job, Status status) {
    Response response; response.id = job.request.id; response.status = status;
    job.promise.set_value(std::move(response));
  }
  void loop() {
    std::vector<Job> batch;
    batch.reserve(static_cast<std::size_t>(max_batch_));
    std::vector<Neighbor> candidates;
    candidates.reserve(kMaxK);
    for (;;) {
      batch.clear();
      {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
        if (stopping_) {
          for (auto& job : jobs_) finish(job, Status::Unavailable);
          jobs_.clear(); return;
        }
        auto until = jobs_.front().submitted + wait_us_;
        while (static_cast<int>(batch.size()) < max_batch_) {
          while (!jobs_.empty() && static_cast<int>(batch.size()) < max_batch_) {
            auto job = std::move(jobs_.front()); jobs_.pop_front();
            if (now_us() - job.submitted >= job.request.timeout_us) { ++expired; finish(job, Status::Timeout); continue; }
            until = std::min(until, job.submitted + job.request.timeout_us);
            batch.push_back(std::move(job));
          }
          const auto current = now_us();
          if (stopping_ || current >= until || static_cast<int>(batch.size()) == max_batch_) break;
          changed_.wait_for(lock, std::chrono::microseconds(until - current), [&] { return stopping_ || !jobs_.empty(); });
        }
      }
      if (batch.empty()) continue;
      int k = 1;
      for (std::size_t i = 0; i < batch.size(); ++i) {
        const auto& request = batch[i].request;
        k = std::max(k, static_cast<int>(std::max(request.k, request.retrieve_k)));
        std::copy_n(engine_.user_query(request.user_id), engine_.cat.dim, queries_.data() + i * engine_.cat.dim);
      }
      k = std::min(k, engine_.cat.n);
      const auto gpu_start = now_us();
      for (const auto& job : batch) queue_time.observe(gpu_start - job.submitted);
      bool used_gpu = gpu_healthy;
      if (used_gpu) try {
        GpuTiming timing;
        scorer_->topk(queries_.data(), static_cast<int>(batch.size()), k, output_.data(), &timing);
        h2d_time.observe(static_cast<std::uint64_t>(std::ceil(timing.h2d_ms * 1000)));
        device_compute_time.observe(static_cast<std::uint64_t>(std::ceil(timing.compute_ms * 1000)));
        d2h_time.observe(static_cast<std::uint64_t>(std::ceil(timing.d2h_ms * 1000)));
        gpu_wall_time.observe(now_us() - gpu_start);
        if (batches == 0) {
          first_batch_wall_us = now_us() - gpu_start;
          first_batch_device_us = static_cast<std::uint64_t>(std::ceil(
              (timing.h2d_ms + timing.compute_ms + timing.d2h_ms) * 1000));
        }
        ++batches; gpu_queries += batch.size();
        max_observed_batch = std::max(max_observed_batch.load(), static_cast<std::uint64_t>(batch.size()));
      } catch (const std::exception&) { gpu_healthy = false; used_gpu = false; }
      const auto retrieve_us = now_us() - gpu_start;
      for (std::size_t i = 0; i < batch.size(); ++i) {
        auto& job = batch[i];
        if (now_us() - job.submitted >= job.request.timeout_us) { ++expired; finish(job, Status::Timeout); continue; }
        Response response;
        try {
          if (used_gpu) {
            candidates.clear();
            const int count = std::min(k, static_cast<int>(std::max(job.request.k, job.request.retrieve_k)));
            for (int j = 0; j < count; ++j) {
              const auto& item = output_[i * k + j]; candidates.push_back({item.id, item.score});
            }
            response = engine_.recommend_sync(job.request, &candidates);
            response.retrieve_us = static_cast<std::uint32_t>(retrieve_us);
          } else { ++fallback; response = engine_.recommend_sync(job.request); }
        } catch (const std::exception&) { finish(job, Status::Unavailable); continue; }
        response.queue_wait_us = static_cast<std::uint32_t>(gpu_start - job.submitted);
        if (now_us() - job.submitted >= job.request.timeout_us) { ++expired; response.status = Status::Timeout; response.items.clear(); }
        job.promise.set_value(std::move(response));
      }
    }
  }
  Engine& engine_;
  std::unique_ptr<GpuScorer> scorer_;
  int max_batch_;
  std::uint32_t wait_us_;
  std::size_t capacity_;
  std::vector<float> queries_;
  std::vector<ScoredItem> output_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Job> jobs_;
  bool stopping_ = false;
  std::thread thread_;
};
}  // namespace recserve
