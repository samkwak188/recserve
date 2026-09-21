#pragma once
#include "protocol.hpp"
#include <array>
#include <cmath>

namespace recserve {
inline constexpr std::uint32_t kVectorMagic = 0x52535632u;
inline constexpr std::size_t kVectorHeaderBytes = 52;
inline constexpr std::size_t kVectorMaxDim = 4096;
using ModelDigest = std::array<std::uint8_t, 32>;

struct VectorRequest {
  RequestId id = 0;
  std::uint32_t timeout_us = 0, count = 0;
  ModelDigest model{};
  std::vector<float> query;
};

inline bool vector_frame_size(const std::uint8_t* header, std::uint32_t& size) {
  std::uint32_t magic = 0;
  std::memcpy(&magic, header, 4);
  std::memcpy(&size, header + 4, 4);
  return magic == kVectorMagic && size >= kVectorHeaderBytes + 4 &&
      size <= kVectorHeaderBytes + 4 * kVectorMaxDim && (size - kVectorHeaderBytes) % 4 == 0;
}

inline bool decode_vector_request(const std::vector<std::uint8_t>& frame, VectorRequest& request) {
  std::uint32_t size = 0, dim = 0;
  if (frame.size() < 8 || !vector_frame_size(frame.data(), size) || frame.size() != size + 8) return false;
  const auto* p = frame.data() + 8;
  std::memcpy(&request.id, p, 8);
  std::memcpy(&request.timeout_us, p + 8, 4);
  std::memcpy(&dim, p + 12, 4);
  std::memcpy(&request.count, p + 16, 4);
  std::memcpy(request.model.data(), p + 20, 32);
  if (dim < 1 || dim > kVectorMaxDim || size != kVectorHeaderBytes + dim * 4 ||
      request.count < 1 || request.count > kMaxK || request.timeout_us < 1 || request.timeout_us > 1000000) return false;
  request.query.resize(dim);
  std::memcpy(request.query.data(), p + kVectorHeaderBytes, dim * 4);
  return std::all_of(request.query.begin(), request.query.end(), [](float x) { return std::isfinite(x); });
}

inline std::vector<std::uint8_t> encode_vector_response(const Response& response, const ModelDigest& model) {
  auto legacy = encode_response(response);
  std::vector<std::uint8_t> result(legacy.size() + 32);
  const auto length = static_cast<std::uint32_t>(result.size() - 8);
  std::memcpy(result.data(), &kVectorMagic, 4);
  std::memcpy(result.data() + 4, &length, 4);
  std::memcpy(result.data() + 8, legacy.data() + 8, 16);
  std::memcpy(result.data() + 24, model.data(), 32);
  std::memcpy(result.data() + 56, legacy.data() + 24, legacy.size() - 24);
  return result;
}
}  // namespace recserve
