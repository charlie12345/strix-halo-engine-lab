#pragma once
#include "backend.hpp"
#include <utility>

namespace strix::xport {
// Adapter above Channel's framing and progress machinery. Channel remains the
// sole owner of Transport, slots, sequence numbers, flags and deferred CRCs.
// Runtime is statically bound (a Channel in production; a recorder in host tests).
template<class Runtime> class TransportBackend {
public:
    using Tag = typename Runtime::Tag;
    using Stream = decltype(std::declval<Runtime&>().options().stream);
    using Event = std::nullptr_t;
    using Buffer = BufferView<Event>;
    struct Ticket { std::uint64_t sequence; };
    struct RxView { const void* remote; };
    using ConsumerView = decltype(std::declval<Runtime&>().recv_view());

    explicit TransportBackend(Runtime& runtime)
        : runtime_(runtime), next_receive_(runtime.sent() + 1) {}
    TransportBackend(const TransportBackend&) = delete;
    TransportBackend& operator=(const TransportBackend&) = delete;

    Ticket post(Tag tag, Buffer source, std::size_t bytes, BackendPublish how) {
        // This adapter admits only the existing copy-and-publish route. Its
        // fixed stream owns the producer/consumer lifetime, without new events.
        if (source.address != AddressSpace::device || source.remote != nullptr ||
            !how.staging || !how.asynchronous)
            throw std::invalid_argument("xport: unsupported Transport buffer/publish");
        runtime_.send_copy(source.local, bytes, tag);
        return {runtime_.sent()};
    }
    Ticket post_copy(Tag tag, const void* source, std::size_t bytes) {
        return post(tag, {const_cast<void*>(source), nullptr, AddressSpace::device},
                    bytes, {true, true});
    }
    template<class Publish> Ticket post_publish(Tag tag, std::size_t bytes, Publish how) {
        // The producer already owns the TX slot; preserve its publication mode.
        runtime_.send(bytes, tag, how);
        return {runtime_.sent()};
    }
    RxView wait(Ticket ticket, Stream stream) {
        ordered(ticket, stream);
        const void* remote = runtime_.recv(); // existing stream wait packet
        ++next_receive_;
        return {remote};
    }
    ConsumerView wait_view(Ticket ticket, Stream stream) {
        ordered(ticket, stream);
        auto view = runtime_.recv_view(); // existing in-kernel wait contract
        ++next_receive_;
        return view;
    }
    void release(Ticket, Stream) noexcept {
        // No extra fence: preserve Channel's existing fixed-stream slot lifetime.
        // recv/recv_view already advances its receive state; Channel has no
        // separate release operation.
    }
    void drain() { runtime_.drain(); }

private:
    void ordered(Ticket ticket, Stream stream) const {
        if (stream != runtime_.options().stream || ticket.sequence != next_receive_ ||
            ticket.sequence > runtime_.sent())
            throw std::invalid_argument("xport: Transport wait must use posting stream/order");
    }
    Runtime& runtime_;
    std::uint64_t next_receive_;
};
} // namespace strix::xport
