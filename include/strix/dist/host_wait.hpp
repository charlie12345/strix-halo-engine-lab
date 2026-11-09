#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::xport {
inline std::string staging_env(const char* name) {
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0) throw std::runtime_error("xport: environment read failed");
    const std::string result = value ? value : "";
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}
inline bool host_wait_enabled() { return staging_env("STRIX_XPORT_TIMED_WAIT") == "1"; }

// One writer per instance; use separate instances for caller and progress thread.
// OFF: no clock reads, sample allocation, output or added synchronization.
// This measures elapsed host blocking calls, including their call/poll overhead,
// never GPU stream dependencies or GPU latency. Peer calls include wire service.
class HostWaitMetric {
public:
    struct Sample { std::uint64_t calls = 0, ns = 0; };
    using Samples = std::map<std::uint32_t, Sample>;
    explicit HostWaitMetric(const char* owner, const char* wait, bool enabled = host_wait_enabled())
        : owner_(owner), wait_(wait), samples_(enabled ? std::make_unique<Samples>() : nullptr) {}
    ~HostWaitMetric() {
        if (!samples_) return;
        for (const auto& [kind, sample] : *samples_)
            std::fprintf(stderr, "xport host_wait owner=%s wait=%s class=%u calls=%llu ns=%llu\n",
                owner_, wait_, kind, static_cast<unsigned long long>(sample.calls),
                static_cast<unsigned long long>(sample.ns));
    }
    template<class F> decltype(auto) measure(std::uint32_t kind, F&& operation) {
        if (!samples_) return operation();
        auto& sample = (*samples_)[kind]; // allocation is outside the measured interval
        struct Timer {
            Sample& sample;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ~Timer() {
                sample.ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin).count());
                ++sample.calls;
            }
        } timer{sample};
        return operation();
    }
    template<class Predicate, class Body>
    void block_while(std::uint32_t kind, Predicate&& pending, Body&& body) {
        if (!pending()) return;
        measure(kind, [&] { do { body(); } while (pending()); });
    }
    void observe(std::uint32_t kind, std::chrono::steady_clock::time_point begin,
                 std::chrono::steady_clock::time_point end) {
        if (!samples_) return;
        auto& sample = (*samples_)[kind];
        ++sample.calls;
        sample.ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    }
    const Samples* samples() const noexcept { return samples_.get(); } // inspect only after writer joins
private:
    const char* owner_;
    const char* wait_;
    std::unique_ptr<Samples> samples_;
};
} // namespace strix::xport
