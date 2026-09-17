#pragma once
#include "types.hpp"
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace recserve {

class ThreadPool {
 public:
  using Job = std::function<void()>;

  void start(int n, bool pin) {
    stop_ = false;
    workers_.clear();
    queues_.assign(n, {});
    mutexes_ = std::vector<std::mutex>(n);
    cvs_ = std::vector<std::condition_variable>(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      workers_.emplace_back([this, i, pin, n]() {
        if (pin) pin_to(i, n);
        worker_loop(i);
      });
    }
  }

  void stop() {
    stop_ = true;
    for (auto& cv : cvs_) cv.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
    workers_.clear();
  }

  bool submit(int worker, Job job, std::size_t max_queue) {
    if (worker < 0 || worker >= static_cast<int>(queues_.size())) return false;
    std::lock_guard<std::mutex> g(mutexes_[worker]);
    if (queues_[worker].size() >= max_queue) return false;
    queues_[worker].push_back(std::move(job));
    cvs_[worker].notify_one();
    return true;
  }

  int size() const { return static_cast<int>(workers_.size()); }
  std::size_t queue_depth(int worker) const {
    std::lock_guard<std::mutex> g(const_cast<std::mutex&>(mutexes_[worker]));
    return queues_[worker].size();
  }

  ~ThreadPool() { stop(); }

 private:
  void worker_loop(int i) {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(mutexes_[i]);
        cvs_[i].wait(lk, [&] { return stop_ || !queues_[i].empty(); });
        if (stop_ && queues_[i].empty()) return;
        job = std::move(queues_[i].front());
        queues_[i].pop_front();
      }
      job();
    }
  }

  static void pin_to(int i, int n) {
#ifdef _WIN32
    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << (i % (sizeof(DWORD_PTR) * 8));
    (void)n;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    int cpu = n > 0 ? (i % n) : 0;
    CPU_SET(static_cast<unsigned>(cpu), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
  }

  std::vector<std::thread> workers_;
  std::vector<std::deque<Job>> queues_;
  std::vector<std::mutex> mutexes_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_{false};
};

}  // namespace recserve
