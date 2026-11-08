#pragma once
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::xport {
enum class AddressSpace { host_coherent, device };
struct BackendPublish { bool staging; bool asynchronous; };
// Unframed engine payload only. The two fences belong to the staging lease.
template<class Event> struct BufferView {
    void* local = nullptr;
    void* remote = nullptr;
    AddressSpace address = AddressSpace::device;
    Event source_free{}, consumer_last_use{};
    std::uint64_t sequence = 0;
};
template<class B>
concept Backend = requires(B& b, typename B::Tag tag, typename B::Buffer view,
                           typename B::Ticket ticket, typename B::Stream stream) {
    { b.post(tag, view, std::size_t{}, BackendPublish{}) } -> std::same_as<typename B::Ticket>;
    { b.wait(ticket, stream) } -> std::same_as<typename B::RxView>;
    b.release(ticket, stream);
    b.drain();
};
// StagingChannel keeps its existing slot state machine and counters. This bridge
// only translates its lease into the backend's typed, unframed buffer contract.
template<Backend B> struct BackendStagingOps {
    using Stream = typename B::Stream;
    using Event = typename B::Event;
    using How = BackendPublish;
    B& backend;
    typename B::Ticket ticket{};
    explicit BackendStagingOps(B& b) : backend(b) {}
    void destroy(Event e) { backend.runtime().destroy(e); }
    void init(Event& a, Event& b, Stream s) { backend.runtime().init(a, b, s); }
    void reuse(Event a, Event b, Stream s) { backend.runtime().reuse(a, b, s); }
    void wait_source(Event a, Stream s) { backend.runtime().wait_source(a, s); }
    void* remote(unsigned slot, unsigned offset) { return backend.runtime().remote(slot, offset); }
    template<class View>
    std::size_t post(const typename B::Tag& tag, View view, std::size_t bytes,
                     How how, Event source, Event consumer) {
        ticket = backend.post(tag, {view.local, view.remote, AddressSpace::device,
                                   source, consumer, view.lease}, bytes, how);
        return backend.copied_bytes();
    }
    template<class View> const void* wait(View, Stream s, Event) {
        return backend.wait(ticket, s).remote;
    }
    void release(Event, Stream s) { backend.release(ticket, s); }
};
} // namespace strix::xport
