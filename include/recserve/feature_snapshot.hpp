#pragma once
// Lock-free feature snapshot for the serving path.
//
// The problem: FeatureStore takes one std::mutex on every user() and item()
// call. With 12 request threads and a nearline ingest thread writing CTR
// updates, every request serializes against every other request AND against
// the writer. The writer stalling the readers is the classic nearline-feature
// failure mode.
//
// The fix is read-copy-update (McKenney & Slingwine, "Read-Copy Update: Using
// Execution History to Solve Concurrency Problems", PDCS 1998; the same shape
// as the Linux kernel's rcu_dereference / synchronize_rcu):
//
//   - Two tables. Readers take an acquire load of a pointer, then read freely.
//     No mutex, no atomic RMW on the read path.
//   - The ingest thread accumulates deltas and publishes on an interval: apply
//     deltas to the back table, release-store it as live, wait out the grace
//     period, then replay the same deltas into the now-quiescent old table so
//     the two stay identical.
//   - The grace period is explicit, not assumed: each reader publishes the
//     generation it is reading in a per-thread slot, and the writer waits until
//     no slot still holds the previous generation before touching it.
//
// The cost this design makes visible: a feature written at time T is not
// readable until the next publish, so end-to-end freshness is bounded below by
// the publish interval. Publishing more often is fresher and costs more writer
// CPU. That tradeoff is measured in apps/nearline.cpp rather than asserted.
#include "types.hpp"
#include "features.hpp"
#include <atomic>
#include <array>
#include <thread>
#include <vector>
#include <cstdint>

namespace recserve {

inline constexpr int kMaxReaders = 128;
inline constexpr std::uint64_t kReaderIdle = ~0ull;

// Per-thread slot publishing which generation this thread is reading.
class ReaderRegistry {
 public:
  ReaderRegistry() {
    for (auto& s : slots_) s.gen.store(kReaderIdle, std::memory_order_relaxed);
  }

  int acquire_slot() {
    const int id = next_.fetch_add(1, std::memory_order_relaxed);
    return id % kMaxReaders;
  }

  void enter(int slot, std::uint64_t gen) {
    slots_[static_cast<std::size_t>(slot)].gen.store(gen, std::memory_order_release);
  }

  void leave(int slot) {
    slots_[static_cast<std::size_t>(slot)].gen.store(kReaderIdle, std::memory_order_release);
  }

  // Spin until nobody is still inside `gen`. Readers hold a snapshot for the
  // length of one request, so this returns in microseconds; the yield keeps it
  // from burning a core if a reader is descheduled mid-request.
  void wait_for_quiescence(std::uint64_t gen) const {
    for (;;) {
      bool busy = false;
      for (const auto& s : slots_) {
        if (s.gen.load(std::memory_order_acquire) == gen) {
          busy = true;
          break;
        }
      }
      if (!busy) return;
      std::this_thread::yield();
    }
  }

 private:
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)  // padding from alignas is the entire point here
#endif
  struct alignas(64) Slot {  // one cache line each: no false sharing between readers
    std::atomic<std::uint64_t> gen{kReaderIdle};
  };
  std::array<Slot, kMaxReaders> slots_{};
  std::atomic<int> next_{0};
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

struct FeatureDelta {
  std::uint64_t event_ms = 0;
  UserId user = 0;
  ItemId item = 0;
  std::uint8_t type = 0;
};

// The published, immutable-to-readers view.
struct FeatureTable {
  std::vector<ItemFeatures> items;
  std::vector<UserFeatures> users;  // dense; user_id % users.size()
  std::uint64_t generation = 0;
  std::uint64_t watermark_ms = 0;  // newest event_time included in this table

  void init(int n_items, int n_users) {
    items.assign(static_cast<std::size_t>(n_items), {});
    users.assign(static_cast<std::size_t>(std::max(1, n_users)), {});
  }

  void apply(const FeatureDelta& d) {
    if (!users.empty()) {
      UserFeatures& uf = users[d.user % users.size()];
      uf.last_event_ms = d.event_ms;
      if (uf.n_last < 8) {
        uf.last_items[uf.n_last++] = d.item;
      } else {
        for (int k = 0; k < 7; ++k) uf.last_items[k] = uf.last_items[k + 1];
        uf.last_items[7] = d.item;
      }
      uf.recency = 1.f;
    }
    if (static_cast<std::size_t>(d.item) < items.size()) {
      ItemFeatures& f = items[d.item];
      f.last_event_ms = d.event_ms;
      f.views += 1;
      if (d.type == 1) f.likes += 1;
      f.ctr = f.views ? static_cast<float>(f.likes) / static_cast<float>(f.views) : 0.f;
    }
    if (d.event_ms > watermark_ms) watermark_ms = d.event_ms;
  }
};

class FeatureSnapshotStore {
 public:
  void init(int n_items, int n_users) {
    tables_[0].init(n_items, n_users);
    tables_[1].init(n_items, n_users);
    tables_[0].generation = 1;
    tables_[1].generation = 0;
    live_.store(&tables_[0], std::memory_order_release);
    back_ = &tables_[1];
    gen_.store(1, std::memory_order_release);
  }

  // Reader side. Held for the duration of one request.
  class Guard {
   public:
    Guard(const FeatureSnapshotStore& s, int slot) : store_(&s), slot_(slot) {
      // Publish intent, then read the pointer. The writer waits for this slot
      // to clear before it touches the table this pointer names.
      for (;;) {
        const std::uint64_t g = store_->gen_.load(std::memory_order_acquire);
        store_->readers_.enter(slot_, g);
        table_ = store_->live_.load(std::memory_order_acquire);
        if (table_->generation == g) break;  // no publish slipped in between
        store_->readers_.leave(slot_);
      }
    }
    ~Guard() { store_->readers_.leave(slot_); }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

    const FeatureTable& operator*() const { return *table_; }
    const FeatureTable* operator->() const { return table_; }

   private:
    const FeatureSnapshotStore* store_;
    int slot_;
    const FeatureTable* table_ = nullptr;
  };

  Guard read() const {
    thread_local int slot = readers_.acquire_slot();
    return Guard(*this, slot);
  }

  // Writer side. Single ingest thread.
  void stage(const FeatureDelta& d) { pending_.push_back(d); }

  std::size_t pending() const { return pending_.size(); }

  // Returns the number of deltas made visible.
  std::size_t publish() {
    if (pending_.empty()) return 0;
    FeatureTable* front = live_.load(std::memory_order_relaxed);
    FeatureTable* back = back_;

    for (const auto& d : pending_) back->apply(d);
    const std::uint64_t new_gen = front->generation + 1;
    back->generation = new_gen;

    gen_.store(new_gen, std::memory_order_release);
    live_.store(back, std::memory_order_release);

    // Grace period: nobody may still be reading the table we are about to
    // mutate back into agreement.
    readers_.wait_for_quiescence(new_gen - 1);
    for (const auto& d : pending_) front->apply(d);
    front->generation = new_gen;

    back_ = front;
    const std::size_t n = pending_.size();
    pending_.clear();
    ++publishes_;
    return n;
  }

  std::uint64_t publishes() const { return publishes_; }
  std::uint64_t generation() const { return gen_.load(std::memory_order_acquire); }

 private:
  FeatureTable tables_[2];
  std::atomic<FeatureTable*> live_{nullptr};
  FeatureTable* back_ = nullptr;
  std::atomic<std::uint64_t> gen_{0};
  mutable ReaderRegistry readers_;
  std::vector<FeatureDelta> pending_;
  std::uint64_t publishes_ = 0;
};

}  // namespace recserve
