#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

namespace recserve {

using ItemId = std::uint32_t;
using UserId = std::uint32_t;
using RequestId = std::uint64_t;

inline constexpr std::uint32_t kProtocolMagic = 0x52535631u;  // RSV1
// Upper bound on items in one response. Raised from 64 so evaluation can ask
// for k plus a user's already-seen items and filter them out afterwards, which
// is what a real ranker does before the list reaches a user.
inline constexpr std::uint32_t kMaxK = 512;
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

// Which scoring kernel the engine uses. SoaStrided is deliberately kept: it is
// the layout that loses, and the board reports it next to Blocked so the
// difference between "SoA" and "SoA that matches the access pattern" is visible.
enum class Kernel : std::uint8_t { Scalar, Simd, SoaStrided, Blocked, Int8, Cuda };

inline const char* kernel_name(Kernel k) {
  switch (k) {
    case Kernel::Scalar: return "scalar";
    case Kernel::Simd: return "simd";
    case Kernel::SoaStrided: return "soa_strided";
    case Kernel::Blocked: return "blocked";
    case Kernel::Int8: return "int8";
    case Kernel::Cuda: return "cuda";
  }
  return "scalar";
}

inline Kernel kernel_from_string(const std::string& s) {
  if (s == "simd") return Kernel::Simd;
  if (s == "soa_strided" || s == "soa") return Kernel::SoaStrided;
  if (s == "blocked") return Kernel::Blocked;
  if (s == "int8") return Kernel::Int8;
  if (s == "cuda") return Kernel::Cuda;
  if (s == "scalar") return Kernel::Scalar;
  throw std::invalid_argument("unknown kernel: " + s);
}


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
  std::uint32_t hops = 0;  // graph nodes expanded during retrieval
  std::uint32_t feature_generation = 0;  // snapshot version this request read
};

struct EngineConfig {
  int dim = static_cast<int>(kDefaultDim);
  int n_items = 0;
  int n_users = 0;
  int n_clusters = 0;
  int workers = 0;  // 0 = hardware_concurrency
  bool pin_workers = false;
  bool use_arena = true;
  bool use_flat_features = true;
  Kernel kernel = Kernel::Simd;
  int hnsw_m = 16;
  int ef_search = 64;
  int ef_construction = 64;
  int build_threads = 0;  // 0 = hardware_concurrency
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

// Event timestamps must share one epoch across processes, so this is
// system_clock (Unix epoch), NOT steady_clock. steady_clock's epoch is
// unspecified -- time since boot on Linux -- so comparing a consumer's
// steady_clock reading against a producer's wall-clock stamp silently produced
// a meaningless freshness number that a >= guard then discarded as zero.
// Use now_us()/now_ns() for durations; use this for anything crossing a
// process boundary.
inline std::uint64_t now_ms_epoch() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

inline std::uint64_t now_ns() {
  using clock = std::chrono::steady_clock;
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
          .count());
}

}  // namespace recserve
