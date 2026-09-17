#include "recserve/engine.hpp"
#include "recserve/net.hpp"
#include "recserve/protocol.hpp"
#include <iostream>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cstring>

using namespace recserve;

// Ranker process: bounded queue, load-shed when full. Retrieve stays on the caller.
int main(int argc, char** argv) {
  int port = 9401;
  int queue = 32;
  int items = 2048;
  int dim = 32;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "--queue" && i + 1 < argc) queue = std::atoi(argv[++i]);
    else if (a == "--items" && i + 1 < argc) items = std::atoi(argv[++i]);
  }

  Engine e;
  e.cfg.use_hnsw = false;
  e.cfg.kernel = Kernel::Simd;
  e.init_random(items, 64, dim, 19);

  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::pair<socket_t, Request>> q;
  std::atomic<bool> live{true};

  std::thread worker([&]() {
    while (live) {
      std::pair<socket_t, Request> job;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !live || !q.empty(); });
        if (!live && q.empty()) return;
        job = std::move(q.front());
        q.pop_front();
      }
      Response r = e.recommend_sync(job.second);
      auto out = encode_response(r);
      send_all(job.first, out.data(), out.size());
      net_close(job.first);
    }
  });

  socket_t srv = listen_tcp(static_cast<std::uint16_t>(port));
  if (srv == net_invalid()) return 1;
  std::cerr << "recserve_rankd port=" << port << " max_queue=" << queue << "\n";
  while (live) {
    socket_t c = accept_tcp(srv);
    if (c == net_invalid()) continue;
    std::uint8_t hdr[8];
    if (!recv_all(c, hdr, 8)) {
      net_close(c);
      continue;
    }
    std::uint32_t len = 0;
    std::memcpy(&len, hdr + 4, 4);
    std::vector<std::uint8_t> buf(8 + len);
    std::memcpy(buf.data(), hdr, 8);
    if (len) recv_all(c, buf.data() + 8, len);
    Request req;
    if (!decode_request(buf.data(), buf.size(), req)) {
      net_close(c);
      continue;
    }
    bool shed = false;
    {
      std::lock_guard<std::mutex> g(mu);
      if (static_cast<int>(q.size()) >= queue) shed = true;
      else q.push_back({c, req});
    }
    if (shed) {
      Response r;
      r.id = req.id;
      r.status = Status::LoadShed;
      auto out = encode_response(r);
      send_all(c, out.data(), out.size());
      net_close(c);
    } else {
      cv.notify_one();
    }
  }
  live = false;
  cv.notify_all();
  worker.join();
  return 0;
}
