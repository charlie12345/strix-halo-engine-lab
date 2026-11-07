#pragma once
#include "backend.hpp"

namespace strix::xport {
// Communicator remains the sole owner of wire framing, CRC, sequence and progress.
// Runtime supplies typed stream/event operations, allowing host-only fixtures.
// In GLM Stream is hipStream_t; no virtual dispatch or second staging allocation.
template<class Communicator, class Runtime, class TagType, bool OrdinaryExchangeOnly = false>
class CommunicatorBackend {
public:
    using Tag = TagType;
    using Stream = typename Runtime::Stream;
    using Event = typename Runtime::Event;
    using Buffer = BufferView<Event>;
    struct Ticket { Tag tag{}; std::uint64_t sequence = 0; };
    struct RxView { void* remote; };
    static constexpr unsigned max_outstanding = 1;
    CommunicatorBackend(Communicator& comm, Runtime& runtime) : comm_(comm), runtime_(runtime) {}
    Runtime& runtime() { return runtime_; }
    std::size_t copied_bytes() const { return copied_; }
    Ticket post(const Tag& tag, Buffer view, std::size_t bytes, BackendPublish how) {
        if (state_ != State::idle) throw std::logic_error("xport: communicator exchange still held");
        if (view.address != AddressSpace::device)
            throw std::invalid_argument("xport: communicator requires device payload");
        view_ = view;
        ticket_ = {tag, ++sequence_};
        staged_ = false;
        copied_ = 0;
        if constexpr (OrdinaryExchangeOnly) {
            if (how.staging || how.asynchronous)
                throw std::invalid_argument("xport: ordinary exchange cannot stage or submit asynchronously");
            async_ = false;
            comm_.start(runtime_.stream, view.local, bytes, tag);
        } else {
            if (how.staging) {
                if (auto* tx = comm_.send_staging(bytes)) {
                    runtime_.copy(view.local, tx, bytes, view.consumer_last_use);
                    comm_.start_staged(runtime_.stream, bytes, tag);
                    runtime_.record_source(view.source_free, runtime_.stream, true);
                    staged_ = true;
                    copied_ = bytes;
                }
            }
            if (!staged_) {
                if (how.asynchronous)
                    async_ = comm_.start_async(runtime_.stream, view.local, view.remote, bytes, tag,
                                              view.consumer_last_use, view.source_free);
                else {
                    async_ = false;
                    comm_.start(runtime_.stream, view.local, bytes, tag);
                }
            }
        }
        state_ = State::posted;
        return ticket_;
    }
    RxView wait(const Ticket& ticket, Stream stream) {
        check(ticket, State::posted);
        void* remote;
        if constexpr (OrdinaryExchangeOnly) {
            comm_.finish(stream, view_.remote);
            remote = view_.remote;
        } else {
            if (staged_) remote = const_cast<void*>(static_cast<const void*>(comm_.finish_staged(stream)));
            else {
                comm_.finish(stream, view_.remote);
                if (!async_) runtime_.record_source(view_.source_free, stream, false);
                remote = view_.remote;
            }
        }
        state_ = State::held;
        return {remote};
    }
    void release(const Ticket& ticket, Stream stream) {
        check(ticket, State::held);
        if constexpr (!OrdinaryExchangeOnly) {
            if (staged_) comm_.release_staged(stream);
        }
        runtime_.record_remote(view_.consumer_last_use, stream);
        state_ = State::idle;
    }
    // No worker to drain. Live leases must be consumed/released on their owning
    // stream; drain must not invent a host synchronize or release a consumer.
    void drain() {
        if (state_ != State::idle) throw std::logic_error("xport: drain with live communicator lease");
    }
private:
    enum class State { idle, posted, held };
    void check(const Ticket& ticket, State expected) const {
        bool same;
        if constexpr (requires { ticket.tag.step; })
            same = ticket.tag.kind == ticket_.tag.kind && ticket.tag.layer == ticket_.tag.layer && ticket.tag.step == ticket_.tag.step;
        else
            same = ticket.tag.kind == ticket_.tag.kind && ticket.tag.layer == ticket_.tag.layer && ticket.tag.request == ticket_.tag.request;
        if (state_ != expected || ticket.sequence != ticket_.sequence || !same)
            throw std::invalid_argument("xport: invalid communicator ticket");
    }
    Communicator& comm_;
    Runtime& runtime_;
    Buffer view_{};
    Ticket ticket_{};
    std::uint64_t sequence_ = 0;
    State state_ = State::idle;
    bool staged_ = false, async_ = false;
    std::size_t copied_ = 0;
};
} // namespace strix::xport
