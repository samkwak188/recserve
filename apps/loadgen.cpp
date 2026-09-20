#include "recserve/engine.hpp"
#include "recserve/net.hpp"
#include "recserve/protocol.hpp"
#include "recserve/metrics.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <cmath>

using namespace recserve;
using Steady = std::chrono::steady_clock;

// Open-loop: send at a fixed arrival rate. Completions after the intended
// deadline are counted as `late` (coordinated-omission visible, not hidden).
int main(int argc, char** argv) {
  const char* host = "127.0.0.1";
  int port = 9400;
  double qps = 200;
  int nreq = 1000;
  int k = 10;
  int timeout_us = 50000;
  int users = 256;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) host = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "--qps" && i + 1 < argc) qps = std::atof(argv[++i]);
    else if (a == "--n" && i + 1 < argc) nreq = std::atoi(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoi(argv[++i]);
    else if (a == "--timeout-us" && i + 1 < argc) timeout_us = std::atoi(argv[++i]);
  }

  net_init();
  ServeStats st;
  std::mutex mu;
  auto t0 = Steady::now();
  double interval = 1.0 / std::max(1.0, qps);
  std::vector<std::thread> workers;
  workers.reserve(nreq);
  for (int i = 0; i < nreq; ++i) {
    auto intended = t0 + std::chrono::duration_cast<Steady::duration>(
                              std::chrono::duration<double>(interval * i));
    std::this_thread::sleep_until(intended);
    workers.emplace_back([&, i, intended]() {
      socket_t s = connect_tcp(host, static_cast<std::uint16_t>(port));
      if (s == net_invalid()) {
        std::lock_guard<std::mutex> g(mu);
        st.loadshed += 1;
        return;
      }
      Request q;
      q.id = static_cast<RequestId>(i + 1);
      q.user_id = static_cast<UserId>(i % users);
      q.k = static_cast<std::uint32_t>(k);
      q.timeout_us = static_cast<std::uint32_t>(timeout_us);
      auto bytes = encode_request(q);
      auto send_at = Steady::now();
      if (!send_all(s, bytes.data(), bytes.size())) {
        net_close(s);
        return;
      }
      std::uint8_t hdr[8];
      if (!recv_all(s, hdr, 8) || !valid_frame_header(hdr, true)) {
        net_close(s);
        return;
      }
      std::uint32_t len = 0;
      std::memcpy(&len, hdr + 4, 4);
      std::vector<std::uint8_t> buf(8 + len);
      std::memcpy(buf.data(), hdr, 8);
      if (len && !recv_all(s, buf.data() + 8, len)) {
        net_close(s);
        return;
      }
      Response r;
      if (!decode_response(buf.data(), buf.size(), r)) {
        net_close(s);
        return;
      }
      auto done = Steady::now();
      double us = std::chrono::duration<double, std::micro>(done - send_at).count();
      bool late = done > intended + std::chrono::microseconds(timeout_us);
      {
        std::lock_guard<std::mutex> g(mu);
        st.sent += 1;
        st.latency_us.add(us);
        if (r.status == Status::Ok) st.ok += 1;
        else if (r.status == Status::Timeout) st.timeout += 1;
        else if (r.status == Status::LoadShed) st.loadshed += 1;
        if (late) st.late += 1;
      }
      net_close(s);
    });
  }
  for (auto& t : workers) t.join();
  std::cout << "n=" << st.latency_us.n() << " ok=" << st.ok << " timeout=" << st.timeout
            << " loadshed=" << st.loadshed << " late=" << st.late
            << " p50_us=" << st.latency_us.percentile(0.50)
            << " p95_us=" << st.latency_us.percentile(0.95)
            << " p99_us=" << st.latency_us.percentile(0.99) << "\n";
  return 0;
}
