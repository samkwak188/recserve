#include "recserve/engine.hpp"
#include "recserve/net.hpp"
#include "recserve/protocol.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <cstring>

using namespace recserve;

int main(int argc, char** argv) {
  int port = 9400;
  int n_items = 4096;
  int dim = 64;
  bool pin = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "--items" && i + 1 < argc) n_items = std::atoi(argv[++i]);
    else if (a == "--dim" && i + 1 < argc) dim = std::atoi(argv[++i]);
    else if (a == "--pin") pin = true;
  }

  Engine e;
  e.cfg.pin_workers = pin;
  e.cfg.use_hnsw = true;
  e.cfg.kernel = Kernel::Simd;
  e.init_random(n_items, 4096, dim, 7);
  e.start_pool();

  socket_t srv = listen_tcp(static_cast<std::uint16_t>(port));
  if (srv == net_invalid()) {
    std::cerr << "listen failed\n";
    return 1;
  }
  std::cerr << "recserve_server port=" << port << " items=" << n_items << " dim=" << dim << "\n";
  std::atomic<bool> live{true};
  std::vector<std::thread> conns;
  while (live) {
    socket_t c = accept_tcp(srv);
    if (c == net_invalid()) continue;
    conns.emplace_back([c, &e]() {
      for (;;) {
        std::uint8_t hdr[8];
        if (!recv_all(c, hdr, 8)) break;
        std::uint32_t len = 0;
        std::memcpy(&len, hdr + 4, 4);
        std::vector<std::uint8_t> buf(8 + len);
        std::memcpy(buf.data(), hdr, 8);
        if (len && !recv_all(c, buf.data() + 8, len)) break;
        Request q;
        if (!decode_request(buf.data(), buf.size(), q)) break;
        auto fut = e.recommend_async(q);
        Response r = fut.get();
        auto out = encode_response(r);
        if (!send_all(c, out.data(), out.size())) break;
      }
      net_close(c);
    });
  }
  e.stop_pool();
  net_close(srv);
  return 0;
}
