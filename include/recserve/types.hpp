#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>

namespace recserve {

using ItemId = std::uint32_t;
using UserId = std::uint32_t;
using RequestId = std::uint64_t;

inline constexpr std::uint32_t kProtocolMagic = 0x52535631u;  // RSV1
inline constexpr std::uint32_t kMaxK = 64;
inline constexpr std::uint32_t kDefaultDim = 64;
inline constexpr std::uint32_t kDefaultRetrieveK = 200;

enum class Status : std::uint32_t {
  Ok = 0,
  Timeout = 1,
  LoadShed = 2,
  BadRequest = 3,
  Unavailable = 4
};

enum class DType { Float32, Int8 };

struct Request {
  RequestId id = 0;
  UserId user_id = 0;
  std::uint32_t k = 10;
  std::uint32_t retrieve_k = kDefaultRetrieveK;
  std::uint32_t timeout_us = 50'000;
};

struct ScoredItem {
  ItemId id = 0;
  float score = 0.f;
};

struct Response {
  RequestId id = 0;
  Status status = Status::Ok;
  std::vector<ScoredItem> items;
  std::uint32_t queue_wait_us = 0;
  std::uint32_t feature_us = 0;
  std::uint32_t retrieve_us = 0;
  std::uint32_t score_us = 0;
};

struct EngineConfig {
  int dim = static_cast<int>(kDefaultDim);
  int n_items = 0;
  int n_users = 0;
  int workers = 0;  // 0 = hardware_concurrency
  bool pin_workers = false;
  bool use_arena = true;
  bool use_soa = true;
  bool use_simd = true;
  bool use_flat_features = true;
  DType dtype = DType::Float32;
  int hnsw_m = 16;
  int hnsw_ef = 64;
  bool use_hnsw = true;
  std::uint32_t feature_delay_us = 0;  // fault injection
  bool feature_timeout = false;
};

inline std::uint64_t now_us() {
  using clock = std::chrono::steady_clock;
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch())
          .count());
}

inline std::uint64_t now_ns() {
  using clock = std::chrono::steady_clock;
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
          .count());
}

}  // namespace recserve
