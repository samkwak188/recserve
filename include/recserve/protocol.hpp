#pragma once
#include "types.hpp"
#include <vector>
#include <cstdint>
#include <cstring>

namespace recserve {

// Length-prefixed binary protocol. Not protobuf/gRPC: owned, interviewable, no extra deps.
inline constexpr std::size_t kReqBytes = 24;
inline constexpr std::size_t kRespHeaderBytes = 16;

inline bool valid_frame_header(const std::uint8_t* p, bool response = false) {
  std::uint32_t magic = 0, len = 0;
  std::memcpy(&magic, p, 4);
  std::memcpy(&len, p + 4, 4);
  return magic == kProtocolMagic &&
      (response ? (len >= kRespHeaderBytes && len <= kRespHeaderBytes + kMaxK * 8 &&
                   (len - kRespHeaderBytes) % 8 == 0) : len == kReqBytes);
}

inline std::vector<std::uint8_t> encode_request(const Request& r) {
  std::vector<std::uint8_t> b(8 + kReqBytes, 0);
  std::uint32_t magic = kProtocolMagic;
  std::uint32_t len = static_cast<std::uint32_t>(kReqBytes);
  std::memcpy(b.data(), &magic, 4);
  std::memcpy(b.data() + 4, &len, 4);
  std::memcpy(b.data() + 8, &r.id, 8);
  std::memcpy(b.data() + 16, &r.user_id, 4);
  std::memcpy(b.data() + 20, &r.k, 4);
  std::memcpy(b.data() + 24, &r.retrieve_k, 4);
  std::memcpy(b.data() + 28, &r.timeout_us, 4);
  return b;
}

inline bool decode_request(const std::uint8_t* p, std::size_t n, Request& r) {
  if (n != 8 + kReqBytes) return false;
  std::uint32_t magic = 0, len = 0;
  std::memcpy(&magic, p, 4);
  std::memcpy(&len, p + 4, 4);
  if (magic != kProtocolMagic || len != kReqBytes) return false;
  std::memcpy(&r.id, p + 8, 8);
  std::memcpy(&r.user_id, p + 16, 4);
  std::memcpy(&r.k, p + 20, 4);
  std::memcpy(&r.retrieve_k, p + 24, 4);
  std::memcpy(&r.timeout_us, p + 28, 4);
  return true;
}

inline std::vector<std::uint8_t> encode_response(const Response& r) {
  std::uint32_t nitem = static_cast<std::uint32_t>(std::min<std::size_t>(r.items.size(), kMaxK));
  std::uint32_t payload = kRespHeaderBytes + nitem * 8;
  std::vector<std::uint8_t> b(8 + payload, 0);
  std::uint32_t magic = kProtocolMagic;
  std::memcpy(b.data(), &magic, 4);
  std::memcpy(b.data() + 4, &payload, 4);
  std::memcpy(b.data() + 8, &r.id, 8);
  std::uint32_t st = static_cast<std::uint32_t>(r.status);
  std::memcpy(b.data() + 16, &st, 4);
  std::memcpy(b.data() + 20, &nitem, 4);
  for (std::uint32_t i = 0; i < nitem; ++i) {
    std::memcpy(b.data() + 24 + i * 8, &r.items[i].id, 4);
    std::memcpy(b.data() + 28 + i * 8, &r.items[i].score, 4);
  }
  return b;
}

inline bool decode_response(const std::uint8_t* p, std::size_t n, Response& r) {
  if (n < 8 + kRespHeaderBytes) return false;
  std::uint32_t magic = 0, payload = 0;
  std::memcpy(&magic, p, 4);
  std::memcpy(&payload, p + 4, 4);
  if (magic != kProtocolMagic || !valid_frame_header(p, true) ||
      n != 8 + static_cast<std::size_t>(payload)) return false;
  std::memcpy(&r.id, p + 8, 8);
  std::uint32_t st = 0, nitem = 0;
  std::memcpy(&st, p + 16, 4);
  std::memcpy(&nitem, p + 20, 4);
  r.status = static_cast<Status>(st);
  r.items.clear();
  if (nitem > kMaxK || payload != kRespHeaderBytes + nitem * 8 ||
      st > static_cast<std::uint32_t>(Status::Unavailable)) return false;
  for (std::uint32_t i = 0; i < nitem; ++i) {
    ScoredItem it;
    std::memcpy(&it.id, p + 24 + i * 8, 4);
    std::memcpy(&it.score, p + 28 + i * 8, 4);
    r.items.push_back(it);
  }
  return true;
}

}  // namespace recserve
