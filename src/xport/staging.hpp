#pragma once
#include "stats.hpp"
#include "strix/dist/host_wait.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::xport {
template<class T>
void rank_pair(unsigned rank, const T* local, const T* remote, const T** p0, const T** p1) {
    *p0 = rank == 0 ? local : remote;
    *p1 = rank == 0 ? remote : local;
}
// Shared ticket/lease core. Ops owns storage and implements event operations on
// its producer/copy streams; it must never synchronize the host in reuse().
inline bool staging_trace_enabled() { return staging_env("STRIX_XPORT_TRACE") == "1"; }
inline unsigned staging_slots(unsigned fallback = 2) {
    const auto setting = staging_env("STRIX_XPORT_STAGING_SLOTS");
    const char* value = setting.c_str();
    if (setting.empty()) return fallback;
    char* end = nullptr;
    const auto n = std::strtoul(value, &end, 10);
    if (!*value || *end || n < 2 || n > 256)
        throw std::invalid_argument("STRIX_XPORT_STAGING_SLOTS must be in [2,256]");
    return static_cast<unsigned>(n);
}

template<class Tag, class Ops>
class StagingChannel {
public:
    using Stream = typename Ops::Stream;
    using Event = typename Ops::Event;
    struct Ticket { Tag tag{}; std::uint64_t sequence = 0; };
    struct StagingView {
        void* local = nullptr;
        void* remote = nullptr;
        unsigned slot = 0;
        std::uint64_t lease = 0;
    };
    using Counters = Stats;
    struct SourceFence { Event event{}; };
    explicit StagingChannel(Ops& ops, unsigned depth, bool trace = false)
        : ops_(ops), slots_(depth), trace_(trace) {
        if (depth < 2) throw std::invalid_argument("xport: staging depth below two");
    }
    ~StagingChannel() {
        for (auto& slot : slots_) {
            ops_.destroy(slot.source_free);
            ops_.destroy(slot.remote_last_use);
        }
    }
    StagingChannel(const StagingChannel&) = delete;
    StagingChannel& operator=(const StagingChannel&) = delete;

    // Reserve a released slot, then enqueue its two lifetime dependencies.
    // A live ticket is never overwritten, even when completion is out of order.
    StagingView staging(void* local, unsigned offset, Stream stream, unsigned preferred) {
        const auto depth = static_cast<unsigned>(slots_.size());
        for (unsigned n = 0; n < depth; ++n) {
            const unsigned index = (preferred + n) % depth;
            auto& slot = slots_[index];
            if (slot.state != State::free) continue;
            ops_.init(slot.source_free, slot.remote_last_use, stream);
            if (slot.used) {
                ops_.reuse(slot.source_free, slot.remote_last_use, stream);
                if (trace_) counters_.reuse_dependencies += 2;
            }
            slot.view = {local, ops_.remote(index, offset), index, ++sequence_};
            slot.state = State::reserved;
            return slot.view;
        }
        throw std::logic_error("xport: all staging tickets are still held");
    }
    template<class How>
    Ticket post(const Tag& tag, StagingView view, std::size_t bytes, How how) {
        auto& slot = slots_.at(view.slot);
        if (slot.state != State::reserved || view.lease != slot.view.lease ||
            view.local != slot.view.local || view.remote != slot.view.remote)
            throw std::invalid_argument("xport: foreign staging view");
        slot.ticket = {tag, view.lease};
        slot.state = State::posted;
        const auto copied = ops_.post(tag, view, bytes, how, slot.source_free, slot.remote_last_use);
        if (trace_) { counters_.bytes += bytes; counters_.staging_bytes += copied; ++counters_.exchanges; }
        return slot.ticket;
    }
    StagingView wait(const Ticket& ticket, Stream stream) {
        auto& slot = find(ticket);
        if (slot.state != State::posted) throw std::logic_error("xport: ticket already waited");
        slot.view.remote = const_cast<void*>(peer_wait_.measure(static_cast<std::uint32_t>(ticket.tag.kind), [&] {
            return ops_.wait(slot.view, stream, slot.source_free);
        }));
        slot.state = State::held;
        return slot.view;
    }
    void release(const Ticket& ticket, Stream stream) {
        auto& slot = find(ticket);
        if (slot.state != State::held) throw std::logic_error("xport: ticket not held");
        ops_.release(slot.remote_last_use, stream);
        slot.used = true;
        slot.state = State::free;
    }
    SourceFence source_fence(const Ticket& ticket) { return {find(ticket).source_free}; }
    void wait_source(const SourceFence& fence, Stream stream) {
        // The slot owns the event until channel destruction. A later recording
        // can strengthen this dependency, but cannot retire it prematurely.
        ops_.wait_source(fence.event, stream);
    }
    const Counters& counters() const noexcept { return counters_; }
private:
    enum class State { free, reserved, posted, held };
    struct Slot {
        Event source_free{}, remote_last_use{};
        Ticket ticket{};
        StagingView view{};
        State state = State::free;
        bool used = false;
    };
    static bool same(const Tag& a, const Tag& b) {
        if constexpr (requires { a.step; })
            return a.kind == b.kind && a.layer == b.layer && a.step == b.step;
        else return a.kind == b.kind && a.layer == b.layer && a.request == b.request;
    }
    Slot& find(const Ticket& ticket) {
        for (auto& slot : slots_) {
            if (slot.ticket.sequence != ticket.sequence || !ticket.sequence) continue;
            if (!same(slot.ticket.tag, ticket.tag)) throw std::invalid_argument("xport: ticket tag differs");
            return slot;
        }
        throw std::invalid_argument("xport: unknown staging ticket");
    }
    Ops& ops_;
    std::vector<Slot> slots_;
    bool trace_;
    Counters counters_;
    HostWaitMetric peer_wait_{"staging", "peer-call"};
    std::uint64_t sequence_ = 0;
};
} // namespace strix::xport
