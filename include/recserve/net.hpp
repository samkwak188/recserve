#pragma once
#include "protocol.hpp"
#include <string>
#include <vector>
#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
inline void net_init() {
  WSADATA w;
  WSAStartup(MAKEWORD(2, 2), &w);
}
inline void net_close(socket_t s) { closesocket(s); }
inline socket_t net_invalid() { return INVALID_SOCKET; }
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
using socket_t = int;
inline void net_init() {}
inline void net_close(socket_t s) { close(s); }
inline socket_t net_invalid() { return -1; }
#endif

namespace recserve {

bool send_all(socket_t s, const std::uint8_t* p, std::size_t n);
bool recv_all(socket_t s, std::uint8_t* p, std::size_t n);
socket_t listen_tcp(std::uint16_t port, const char* bind_address = "127.0.0.1");
bool net_nonblocking(socket_t s);
bool net_ready(socket_t s, bool write, int timeout_ms);
bool transfer_until(socket_t s, std::uint8_t* data, std::size_t n, bool write, std::uint64_t deadline_us, bool* clean_eof = nullptr);
socket_t accept_tcp(socket_t server);
socket_t connect_tcp(const char* host, std::uint16_t port);

}  // namespace recserve
