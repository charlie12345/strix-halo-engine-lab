#include "core/host_lane.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using strix::core::ByteBudget;
using strix::core::HostLane;
using strix::core::HostLaneConfig;
using strix::core::HostPriority;
using Clock = ByteBudget::Clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void budget_is_unlimited_without_decode() {
    ByteBudget budget(1000.0, 0.1); // 1000 B/s, burst 100 B
    const auto t0 = Clock::time_point{} + std::chrono::seconds(10);
    require(budget.reserve(1u << 30, t0) == Clock::duration::zero(), "idle budget made a job wait");
    require(!budget.capped(), "idle budget reports capped");
    const auto stats = budget.stats();
    require(stats.bytes_granted == (1u << 30) && stats.bytes_capped == 0 && stats.cap_hits == 0, "idle stats");
    ByteBudget never(0.0);
    never.decode_begin(t0);
    require(never.reserve(1u << 30, t0) == Clock::duration::zero() && !never.capped(), "cap 0 must never cap");
}

void budget_caps_during_decode() {
    ByteBudget budget(1000.0, 0.1);
    const auto t0 = Clock::time_point{} + std::chrono::seconds(10);
    budget.decode_begin(t0);
    require(budget.capped(), "decode did not arm the cap");
    require(budget.reserve(100, t0) == Clock::duration::zero(), "burst was not available");
    // Bucket empty: 500 bytes at 1000 B/s owe 0.5 s.
    require(duration_cast<milliseconds>(budget.reserve(500, t0)).count() == 500, "debt wait is not bytes / cap");
    // 0.25 s later half the debt is repaid; another 250 bytes owe 0.5 s again.
    require(duration_cast<milliseconds>(budget.reserve(250, t0 + milliseconds(250))).count() == 500, "refill");
    // After a long idle gap the bucket holds the burst only, never more.
    require(budget.reserve(100, t0 + std::chrono::seconds(100)) == Clock::duration::zero(), "burst after idle");
    require(budget.reserve(1, t0 + std::chrono::seconds(100)) > Clock::duration::zero(), "burst exceeded");
    // A clock reading older than the last one must not mint tokens.
    require(budget.reserve(1, t0) > Clock::duration::zero(), "stale clock minted tokens");
    const auto stats = budget.stats();
    require(stats.bytes_granted == 952 && stats.bytes_capped == 952 && stats.cap_hits == 4, "capped stats");
    require(stats.throttled_ns >= 1000000000ull, "throttle time not recorded");
    budget.decode_end();
    require(!budget.capped(), "cap outlived the decode");
    require(budget.reserve(1u << 20, t0 + std::chrono::seconds(101)) == Clock::duration::zero(), "idle after decode");
}

void budget_long_run_rate_stays_under_the_cap() {
    const double cap = 1e6, burst_seconds = 0.02;
    ByteBudget budget(cap, burst_seconds);
    auto now = Clock::time_point{} + std::chrono::seconds(1);
    const auto begin = now;
    budget.decode_begin(now);
    std::uint64_t bytes = 0;
    for (int i = 0; i < 2000; ++i) { // a caller that honours every wait
        const std::uint64_t slice = 1000 + static_cast<std::uint64_t>(i % 7) * 3000;
        now += budget.reserve(slice, now);
        bytes += slice;
        now += std::chrono::microseconds(i % 5);
    }
    const double elapsed = std::chrono::duration<double>(now - begin).count();
    require(static_cast<double>(bytes) <= cap * elapsed + cap * burst_seconds + 16.0, "rate exceeded cap + burst");
    require(static_cast<double>(bytes) >= 0.98 * cap * elapsed, "governor wasted more than 2 % of the cap");
    // Nested decodes: the cap holds until the last one ends.
    budget.decode_begin(now);
    budget.decode_end();
    require(budget.capped(), "cap dropped while a decode was still active");
    budget.decode_end();
    require(!budget.capped(), "cap held after the last decode");
}

void budget_wait_ends_with_the_decode() {
    ByteBudget budget(1000.0, 0.0);
    budget.decode_begin();
    std::thread releaser([&] {
        std::this_thread::sleep_for(milliseconds(30));
        budget.decode_end();
    });
    const auto begin = Clock::now();
    budget.acquire(3600 * 1000); // an hour of debt at the cap
    releaser.join();
    require(Clock::now() - begin < std::chrono::seconds(20), "acquire did not wake when decode ended");
    {
        ByteBudget::DecodeScope decode(budget);
        require(budget.capped(), "scope did not arm the cap");
    }
    require(!budget.capped(), "scope did not release the cap");
    std::uint64_t seen = 0, calls = 0;
    strix::core::governed_slices(budget, 10, 4, [&](std::uint64_t offset, std::uint64_t length) {
        require(offset == seen, "slices out of order");
        seen += length;
        ++calls;
    });
    require(seen == 10 && calls == 3, "slices do not cover the range");
}

void core_lists() {
    using strix::core::parse_core_list;
    require(parse_core_list("2-5") == std::vector<int>({2, 3, 4, 5}), "range");
    require(parse_core_list("0,2,4-6") == std::vector<int>({0, 2, 4, 5, 6}), "mixed list");
    require(parse_core_list("").empty(), "empty list");
    for (const char* bad : {"a", "1-", "3-2", "1,", "1;2", "-1", "99999"})
        require(throws([&] { (void)parse_core_list(bad); }), "malformed core list was accepted");
}

void lane_runs_jobs_and_reports() {
    HostLaneConfig config;
    config.critical_threads = 7; // clamped to two
    config.background_threads = 2;
    config.byte_cap_gbytes_ps = 1.0;
    std::atomic<int> sum{0};
    {
        HostLane lane(config);
        require(lane.config().critical_threads == 2, "critical threads were not clamped");
        const auto caller = std::this_thread::get_id();
        auto id = lane.submit(HostPriority::critical, [] { return std::this_thread::get_id(); });
        require(id.get() != caller, "critical job ran on the caller");
        auto value = lane.submit(HostPriority::background, [] { return std::string("done"); });
        require(value.get() == "done", "result lost");
        auto failing = lane.submit(HostPriority::background, [](int fail = 1) -> int {
            if (fail) throw std::runtime_error("job failed");
            return 0;
        });
        require(throws([&] { (void)failing.get(); }), "job exception was swallowed");
        // Both background threads blocked: a critical job still completes.
        std::promise<void> gate;
        const std::shared_future<void> opened = gate.get_future().share();
        auto blocked_a = lane.submit(HostPriority::background, [opened] { opened.wait(); });
        auto blocked_b = lane.submit(HostPriority::background, [opened] { opened.wait(); });
        auto urgent = lane.submit(HostPriority::critical, [] { return 7; });
        require(urgent.wait_for(std::chrono::seconds(20)) == std::future_status::ready && urgent.get() == 7,
                "critical job waited behind background work");
        gate.set_value();
        blocked_a.get();
        blocked_b.get();
        // Background jobs draw from the lane's budget; decode arms the cap.
        {
            ByteBudget::DecodeScope decode(lane.budget());
            lane.submit(HostPriority::background, [&lane] { (void)lane.budget().reserve(4096, Clock::now()); }).get();
        }
        const auto stats = lane.stats();
        require(stats.critical_jobs == 2 && stats.background_jobs == 5 && stats.inline_jobs == 0, "job counts");
        require(stats.budget.bytes_granted == 4096 && stats.budget.bytes_capped == 4096, "budget stats");
        const auto receipt = lane.receipt_json();
        require(receipt.find("\"host_lane_bytes_granted\":4096") != std::string::npos &&
                    receipt.find("\"host_lane_busy_ms\":") != std::string::npos &&
                    receipt.find("\"host_lane_cap_hits\":0") != std::string::npos,
                "receipt fields");
        for (int i = 1; i <= 200; ++i)
            (void)lane.submit(i % 2 ? HostPriority::critical : HostPriority::background, [&sum, i] { sum += i; });
    } // destruction completes everything that was queued
    require(sum.load() == 200 * 201 / 2, "queued jobs were dropped at shutdown");
}

void lane_disabled_runs_inline() {
    HostLaneConfig config;
    config.enabled = false;
    HostLane lane(config);
    const auto caller = std::this_thread::get_id();
    require(lane.submit(HostPriority::critical, [] { return std::this_thread::get_id(); }).get() == caller, "inline critical");
    require(lane.submit(HostPriority::background, [] { return std::this_thread::get_id(); }).get() == caller, "inline background");
    auto failing = lane.submit(HostPriority::critical, [](int fail = 1) -> int {
        if (fail) throw std::runtime_error("job failed");
        return 0;
    });
    require(throws([&] { (void)failing.get(); }), "inline exception must travel through the future");
    require(lane.stats().inline_jobs == 3, "inline count");
}

void lane_pins_to_declared_cores() {
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cores;
    for (int core = 0; core < CPU_SETSIZE && cores.size() < 3; ++core)
        if (CPU_ISSET(core, &allowed)) cores.push_back(core);
    if (cores.size() < 2) return;
    HostLaneConfig config;
    config.cores = cores;
    config.critical_threads = 1;
    config.background_threads = 1;
    HostLane lane(config);
    const auto mask = [] {
        cpu_set_t set;
        CPU_ZERO(&set);
        std::vector<int> out;
        if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0)
            for (int core = 0; core < CPU_SETSIZE; ++core)
                if (CPU_ISSET(core, &set)) out.push_back(core);
        return out;
    };
    require(lane.submit(HostPriority::critical, mask).get() == std::vector<int>({cores[0]}), "critical thread pin");
    require(lane.submit(HostPriority::background, mask).get() == std::vector<int>(cores.begin() + 1, cores.end()),
            "background threads must stay off the critical core");
#endif
}
} // namespace

int main() {
    try {
        budget_is_unlimited_without_decode();
        budget_caps_during_decode();
        budget_long_run_rate_stays_under_the_cap();
        budget_wait_ends_with_the_decode();
        core_lists();
        lane_runs_jobs_and_reports();
        lane_disabled_runs_inline();
        lane_pins_to_declared_cores();
    } catch (const std::exception& error) {
        std::cerr << "host lane tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "host lane tests passed\n";
    return 0;
}
