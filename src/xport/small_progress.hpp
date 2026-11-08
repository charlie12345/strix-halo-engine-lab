#pragma once
#include <chrono>
#include <cstddef>
#include <string_view>
#include <stdexcept>

namespace strix::xport {
// Limits include the wire trailer. This is a scheduling policy, never a wire identity.
struct SmallProgress {
    static constexpr std::size_t max_frame_bytes = 256U * 1024U;
    static constexpr std::chrono::microseconds wire_spin{100};
    static constexpr std::chrono::microseconds mailbox_spin{3000};
    static constexpr bool enabled(const char* value) noexcept {
        return value != nullptr && std::string_view(value) == "1";
    }
    static bool parse(const char* value) {
        if (value != nullptr && *value != '\0' && std::string_view(value) != "0" && std::string_view(value) != "1")
            throw std::invalid_argument("STRIX_XPORT_SMALL_SPIN must be 0 or 1");
        return enabled(value);
    }
    static constexpr bool eligible(std::size_t bytes) noexcept {
        return bytes != 0 && bytes <= max_frame_bytes;
    }
};

// The caller's acquire load is the mailbox: no second queue or readiness flag.
// Check it on every pause, but amortize the clock over at most 64 pauses. A fixed
// deadline bounds idle CPU work even when another thread repeatedly wakes us.
template<class Clock = std::chrono::steady_clock, class Ready, class Pause>
bool spin_small_mailbox(Ready ready, Pause pause, std::chrono::microseconds budget) {
    const auto deadline = Clock::now() + budget;
    while (Clock::now() < deadline) {
        for (unsigned probe = 0; probe < 64U; ++probe) {
            if (ready()) return true;
            pause();
        }
    }
    return ready();
}
} // namespace strix::xport
