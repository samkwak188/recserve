#pragma once
#include "types.hpp"
#include <unordered_map>
#include <mutex>
#include <vector>
#include <deque>
#include <thread>
#include <chrono>
#include <algorithm>

namespace recserve {

struct UserFeatures {
  float ctr = 0.f;
  float recency = 0.f;
  ItemId last_items[8]{};
  int n_last = 0;
  std::uint64_t last_event_ms = 0;
};

struct ItemFeatures {
  float ctr = 0.f;
  std::uint64_t views = 0;
  std::uint64_t likes = 0;
  std::uint64_t last_event_ms = 0;
};

// Flat open-addressing map for user features (Stage B optimization vs unordered_map).
class FlatUserMap {
 public:
  static constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;

  void clear_and_reserve(int n) {
    cap_ = 1;
    while (cap_ < n * 2 + 8) cap_ *= 2;
    keys_.assign(static_cast<std::size_t>(cap_), kEmpty);
    vals_.assign(static_cast<std::size_t>(cap_), {});
    size_ = 0;
  }

  // Grows instead of spinning. The previous version probed with `for (;;)` and
  // no load factor, so once the table filled -- which happens as soon as more
  // distinct user ids arrive than were reserved -- an insert became an infinite
  // loop inside the request path. A serving component must not hang on an
  // unexpected key.
  UserFeatures& get(UserId u) {
    if (cap_ == 0) clear_and_reserve(16);
    if (size_ * 4 >= static_cast<std::size_t>(cap_) * 3) grow();
    const std::uint32_t mask = static_cast<std::uint32_t>(cap_ - 1);
    std::uint32_t i = (u * 2654435761u) & mask;
    for (int probe = 0; probe < cap_; ++probe) {
      if (keys_[i] == kEmpty) {
        keys_[i] = u;
        vals_[i] = UserFeatures{};
        ++size_;
        return vals_[i];
      }
      if (keys_[i] == u) return vals_[i];
      i = (i + 1) & mask;
    }
    grow();  // table was full despite the load factor: rehash and retry once
    return get(u);
  }

  const UserFeatures* find(UserId u) const {
    if (cap_ == 0) return nullptr;
    std::uint32_t mask = static_cast<std::uint32_t>(cap_ - 1);
    std::uint32_t i = (u * 2654435761u) & mask;
    for (int s = 0; s < cap_; ++s) {
      if (keys_[i] == kEmpty) return nullptr;
      if (keys_[i] == u) return &vals_[i];
      i = (i + 1) & mask;
    }
    return nullptr;
  }

 private:
  void grow() {
    const std::vector<std::uint32_t> old_keys = keys_;
    const std::vector<UserFeatures> old_vals = vals_;
    const int old_cap = cap_;
    cap_ = std::max(16, old_cap * 2);
    keys_.assign(static_cast<std::size_t>(cap_), kEmpty);
    vals_.assign(static_cast<std::size_t>(cap_), {});
    size_ = 0;
    const std::uint32_t mask = static_cast<std::uint32_t>(cap_ - 1);
    for (int j = 0; j < old_cap; ++j) {
      if (old_keys[static_cast<std::size_t>(j)] == kEmpty) continue;
      std::uint32_t i = (old_keys[static_cast<std::size_t>(j)] * 2654435761u) & mask;
      while (keys_[i] != kEmpty) i = (i + 1) & mask;
      keys_[i] = old_keys[static_cast<std::size_t>(j)];
      vals_[i] = old_vals[static_cast<std::size_t>(j)];
      ++size_;
    }
  }

  int cap_ = 0;
  std::size_t size_ = 0;
  std::vector<std::uint32_t> keys_;
  std::vector<UserFeatures> vals_;
};

class FeatureStore {
 public:
  void init(int n_users, int n_items, bool flat) {
    n_items_ = n_items;
    use_flat_ = flat;
    items_.assign(n_items, {});
    hash_.clear();
    if (flat) {
      flat_.clear_and_reserve(std::max(n_users, 16));
    }
  }

  void set_delay_us(std::uint32_t us) { delay_us_ = us; }
  void set_fail(bool v) { fail_ = v; }

  UserFeatures user(UserId u) {
    maybe_delay();
    if (fail_) return {};
    std::lock_guard<std::mutex> g(mu_);
    if (use_flat_) return flat_.get(u);
    return hash_[u];
  }

  ItemFeatures item(ItemId i) {
    maybe_delay();
    if (fail_ || static_cast<std::size_t>(i) >= items_.size()) return {};
    std::lock_guard<std::mutex> g(mu_);
    return items_[i];
  }

  void apply_event(std::uint64_t event_ms, UserId u, ItemId it, int type) {
    std::lock_guard<std::mutex> g(mu_);
    UserFeatures* uf = use_flat_ ? &flat_.get(u) : &hash_[u];
    uf->last_event_ms = event_ms;
    if (uf->n_last < 8) {
      uf->last_items[uf->n_last++] = it;
    } else {
      for (int k = 0; k < 7; ++k) uf->last_items[k] = uf->last_items[k + 1];
      uf->last_items[7] = it;
    }
    if (static_cast<std::size_t>(it) < items_.size()) {
      auto& f = items_[it];
      f.last_event_ms = event_ms;
      f.views += 1;
      if (type == 1) f.likes += 1;
      f.ctr = f.views ? static_cast<float>(f.likes) / static_cast<float>(f.views) : 0.f;
    }
    uf->ctr = 0.f;
    uf->recency = 1.f;
  }

  std::uint64_t freshness_ms(std::uint64_t now_ms, UserId u) {
    std::lock_guard<std::mutex> g(mu_);
    const UserFeatures* uf = nullptr;
    UserFeatures tmp;
    if (use_flat_) {
      uf = flat_.find(u);
    } else {
      auto it = hash_.find(u);
      if (it != hash_.end()) {
        tmp = it->second;
        uf = &tmp;
      }
    }
    if (!uf || uf->last_event_ms == 0 || now_ms < uf->last_event_ms) return 0;
    return now_ms - uf->last_event_ms;
  }

 private:
  void maybe_delay() {
    if (delay_us_ == 0) return;
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us_));
  }

  std::mutex mu_;
  bool use_flat_ = true;
  bool fail_ = false;
  std::uint32_t delay_us_ = 0;
  int n_items_ = 0;
  FlatUserMap flat_;
  std::unordered_map<UserId, UserFeatures> hash_;
  std::vector<ItemFeatures> items_;
};

}  // namespace recserve
