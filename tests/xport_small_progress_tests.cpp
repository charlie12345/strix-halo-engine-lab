#include "xport/small_progress.hpp"
#include "glm/engine/glm_kvshard_exchange.hpp"
#include "glm/xfer/glm_traits.hpp"
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("small progress check failed"); }
struct FakeClock {
    using duration = std::chrono::microseconds;
    using time_point = std::chrono::time_point<FakeClock>;
    static inline duration elapsed{};
    static time_point now() { return time_point(elapsed); }
};
void bounds() {
    using P = strix::xport::SmallProgress;
    require(!P::enabled(nullptr) && !P::enabled("") && !P::enabled("0") && !P::enabled("10") && P::enabled("1"));
    bool rejected = false;
    try { (void)P::parse("10"); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !P::parse(nullptr) && !P::parse("0") && P::parse("1"));
    require(!P::eligible(0) && P::eligible(1) && P::eligible(P::max_frame_bytes) && !P::eligible(P::max_frame_bytes + 1));
    namespace k = strix::glm::kvshard;
    constexpr auto trailer = sizeof(strix::glm::xfer::Trailer);
    require(k::prepare_bytes(8192 / 4) > 65536 && P::eligible(k::prepare_bytes(8192 / 4) + trailer));
    require(P::eligible(k::prepare_bytes(65536 / 4) + trailer) && P::eligible(k::partial_bytes + trailer));
    unsigned pauses = 0;
    FakeClock::elapsed = FakeClock::duration{0};
    const auto pause = [&] { ++pauses; FakeClock::elapsed += FakeClock::duration{1}; };
    require(!strix::xport::spin_small_mailbox<FakeClock>([] { return false; }, pause, P::wire_spin));
    require(pauses == 128U); // fixed 100 us deadline, at most one 64-pause checkpoint overshoot
    pauses = 0;
    require(strix::xport::spin_small_mailbox<FakeClock>([] { return true; }, pause, P::mailbox_spin));
    require(pauses == 0U); // includes immediate cancellation through the caller's predicate
    require(!strix::xport::spin_small_mailbox<FakeClock>([] { return false; }, pause, std::chrono::microseconds{0}));
    require(pauses == 0U);
}
void ordered_mailbox() {
    // One reusable slot: release/acquire must expose every payload and forbid reuse
    // until consumption, including repeated transitions between spin and timeout.
    std::atomic<unsigned> posted{0}, done{0};
    unsigned payload = 0, consumed = 0;
    constexpr unsigned count = 100000;
    std::thread producer([&] {
        for (unsigned ticket = 1; ticket <= count; ++ticket) {
            while (done.load(std::memory_order_acquire) != ticket - 1) std::this_thread::yield();
            payload = ticket;
            posted.store(ticket, std::memory_order_release);
        }
    });
    for (unsigned ticket = 1; ticket <= count; ++ticket) {
        while (!strix::xport::spin_small_mailbox([&] { return posted.load(std::memory_order_acquire) == ticket; },
                [] { std::this_thread::yield(); }, std::chrono::microseconds{10})) {}
        if (payload == ticket) ++consumed;
        done.store(ticket, std::memory_order_release);
    }
    producer.join();
    require(consumed == count);
}
} // namespace
int main() {
    try {
        bounds(); ordered_mailbox();
        std::puts("PASS: small route/frame bounds, fixed spin deadline, cancellation, 100000 ordered mailbox payloads");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
