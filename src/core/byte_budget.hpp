#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace strix::core {
// Governor for background host work that reads memory (snapshot packing, checkpoint CRC, cache hashing).
// Memory bandwidth is shared with the decode weight stream, so while any session decodes the bytes read by
// background jobs are limited to `cap` bytes per second; with no decode active the budget is unlimited.
//
// Token bucket with debt: a reservation is always granted, and the returned duration is how long the caller
// must wait before touching the bytes. Requests larger than the burst are therefore legal (they simply wait
// longer) and the long-run rate under the cap never exceeds cap + burst / elapsed.
class ByteBudget {
public:
    using Clock = std::chrono::steady_clock;
    struct Stats {
        std::uint64_t bytes_granted = 0; // every reservation
        std::uint64_t bytes_capped = 0;  // reservations made while the cap was in force
        std::uint64_t cap_hits = 0;      // reservations that had to wait
        std::uint64_t throttled_ns = 0;  // total wait handed out
    };

    // cap_bytes_per_second <= 0 means never capped.
    explicit ByteBudget(double cap_bytes_per_second = 0.0, double burst_seconds = 0.02)
        : cap_(cap_bytes_per_second), burst_(std::max(0.0, cap_bytes_per_second) * std::max(0.0, burst_seconds)),
          tokens_(burst_) {}

    void decode_begin(Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_++ == 0) {
            tokens_ = burst_;
            last_ = now;
        }
    }
    void decode_end() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_ > 0) --active_;
            if (active_ != 0) return;
        }
        released_.notify_all();
    }
    // One decode phase; background work is capped for its lifetime.
    class DecodeScope {
    public:
        explicit DecodeScope(ByteBudget& budget) : budget_(&budget) { budget_->decode_begin(); }
        DecodeScope(const DecodeScope&) = delete;
        DecodeScope& operator=(const DecodeScope&) = delete;
        ~DecodeScope() { budget_->decode_end(); }

    private:
        ByteBudget* budget_;
    };

    bool capped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_ != 0 && cap_ > 0.0;
    }
    double cap_bytes_per_second() const { return cap_; }

    // Deterministic core: charge `bytes` at time `now`; returns the wait owed before the bytes may be read.
    Clock::duration reserve(std::uint64_t bytes, Clock::time_point now) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.bytes_granted += bytes;
        if (active_ == 0 || cap_ <= 0.0) return Clock::duration::zero();
        stats_.bytes_capped += bytes;
        if (now > last_) {
            tokens_ = std::min(burst_, tokens_ + std::chrono::duration<double>(now - last_).count() * cap_);
            last_ = now;
        }
        tokens_ -= static_cast<double>(bytes);
        if (tokens_ >= 0.0) return Clock::duration::zero();
        const auto wait = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(-tokens_ / cap_));
        ++stats_.cap_hits;
        stats_.throttled_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(wait).count());
        return wait;
    }
    // Blocking form: returns when the bytes may be read. The wait ends early when the last decode ends.
    void acquire(std::uint64_t bytes) {
        const auto wait = reserve(bytes, Clock::now());
        if (wait <= Clock::duration::zero()) return;
        std::unique_lock<std::mutex> lock(mutex_);
        released_.wait_for(lock, wait, [&] { return active_ == 0; });
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    const double cap_;
    const double burst_;
    mutable std::mutex mutex_;
    std::condition_variable released_;
    double tokens_;
    Clock::time_point last_{};
    unsigned active_ = 0;
    Stats stats_;
};

// Runs `visit(offset, length)` over [0, total) in slices, drawing each slice from the budget first, so a
// long background pass reacts to a decode that starts in the middle of it.
template<class Visit>
void governed_slices(ByteBudget& budget, std::uint64_t total, std::uint64_t slice, Visit visit) {
    if (slice == 0) slice = total;
    for (std::uint64_t offset = 0; offset < total;) {
        const std::uint64_t length = std::min(slice, total - offset);
        budget.acquire(length);
        visit(offset, length);
        offset += length;
    }
}
} // namespace strix::core
