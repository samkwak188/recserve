#include "recserve/net.hpp"
#include <cstring>
#include <cerrno>
#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#endif

namespace recserve {

bool send_all(socket_t s, const std::uint8_t* p, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
#ifdef _WIN32
    int sent = send(s, reinterpret_cast<const char*>(p + off), static_cast<int>(n - off), 0);
#else
    ssize_t sent = send(s, p + off, n - off, MSG_NOSIGNAL);
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

socket_t listen_tcp(std::uint16_t port, const char* bind_address) {
  net_init();
  socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == net_invalid()) return net_invalid();
  int yes = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  if (inet_pton(AF_INET, bind_address, &addr.sin_addr) != 1) {
    net_close(s);
    return net_invalid();
  }
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

bool net_nonblocking(socket_t s) {
#ifdef _WIN32
  u_long yes = 1;
  return ioctlsocket(s, FIONBIO, &yes) == 0;
#else
  const int flags = fcntl(s, F_GETFL, 0);
  return flags >= 0 && fcntl(s, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

bool net_ready(socket_t s, bool write, int timeout_ms) {
#ifdef _WIN32
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
  return select(0, write ? nullptr : &fds, write ? &fds : nullptr, nullptr, &timeout) > 0;
#else
  pollfd fd{s, static_cast<short>(write ? POLLOUT : POLLIN), 0};
  return poll(&fd, 1, timeout_ms) > 0;
#endif
}

bool transfer_until(socket_t s, std::uint8_t* data, std::size_t n, bool write, std::uint64_t deadline_us, bool* clean_eof) {
  if (clean_eof) *clean_eof = false;
  std::size_t offset = 0;
  while (offset < n) {
    const auto now = now_us();
    if (now >= deadline_us || !net_ready(s, write, static_cast<int>((deadline_us - now + 999) / 1000))) return false;
#ifdef _WIN32
    const auto got = write ? send(s, reinterpret_cast<char*>(data + offset), static_cast<int>(n-offset), 0) :
                             recv(s, reinterpret_cast<char*>(data + offset), static_cast<int>(n-offset), 0);
    if (got < 0 && WSAGetLastError() == WSAEWOULDBLOCK) continue;
#else
    const auto got = write ? send(s, data + offset, n-offset, MSG_NOSIGNAL) : recv(s, data + offset, n-offset, 0);
    if (got < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
#endif
    if (got <= 0) {
      if (clean_eof) *clean_eof = got == 0 && offset == 0;
      return false;
    }
    offset += static_cast<std::size_t>(got);
  }
  return true;
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
