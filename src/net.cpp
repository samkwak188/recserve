#include "recserve/net.hpp"
#include <cstring>

namespace recserve {

bool send_all(socket_t s, const std::uint8_t* p, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
#ifdef _WIN32
    int sent = send(s, reinterpret_cast<const char*>(p + off), static_cast<int>(n - off), 0);
#else
    ssize_t sent = send(s, p + off, n - off, 0);
#endif
    if (sent <= 0) return false;
    off += static_cast<std::size_t>(sent);
  }
  return true;
}

bool recv_all(socket_t s, std::uint8_t* p, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
#ifdef _WIN32
    int got = recv(s, reinterpret_cast<char*>(p + off), static_cast<int>(n - off), 0);
#else
    ssize_t got = recv(s, p + off, n - off, 0);
#endif
    if (got <= 0) return false;
    off += static_cast<std::size_t>(got);
  }
  return true;
}

socket_t listen_tcp(std::uint16_t port) {
  net_init();
  socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == net_invalid()) return net_invalid();
  int yes = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    net_close(s);
    return net_invalid();
  }
  if (listen(s, 128) != 0) {
    net_close(s);
    return net_invalid();
  }
  return s;
}

socket_t accept_tcp(socket_t server) {
  sockaddr_in addr{};
#ifdef _WIN32
  int len = sizeof(addr);
#else
  socklen_t len = sizeof(addr);
#endif
  return accept(server, reinterpret_cast<sockaddr*>(&addr), &len);
}

socket_t connect_tcp(const char* host, std::uint16_t port) {
  net_init();
  socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == net_invalid()) return net_invalid();
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host, &addr.sin_addr);
  if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    net_close(s);
    return net_invalid();
  }
  return s;
}

}  // namespace recserve
