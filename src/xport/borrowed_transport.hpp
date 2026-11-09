#pragma once
#include "strix/dist/transport.hpp"

namespace strix::xport {

// The probe historically inherited Transport's no-op abort. Keep that policy
// explicit while ordinary borrowers forward the complete interface.
enum class BorrowedAbort { forward, base_default };

// Non-owning: the transport must outlive this adapter and all of its calls.
// Framing, progress, CRCs and state remain owned by the underlying transport.
template<BorrowedAbort Abort = BorrowedAbort::forward>
class BorrowedTransport final : public dist::Transport {
public:
    explicit BorrowedTransport(dist::Transport& t) : t_(t) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }
    void exchange_overlapped(std::span<const std::byte> o, std::span<std::byte> i,
                             void (*on_wire)(void*), void* context) override {
        t_.exchange_overlapped(o, i, on_wire, context);
    }
    [[nodiscard]] std::string name() const override { return t_.name(); }
    void abort(const std::string& why) override {
        if constexpr (Abort == BorrowedAbort::forward) t_.abort(why);
        else dist::Transport::abort(why);
    }
    [[nodiscard]] std::string state() const override { return t_.state(); }

private:
    dist::Transport& t_;
};

} // namespace strix::xport
