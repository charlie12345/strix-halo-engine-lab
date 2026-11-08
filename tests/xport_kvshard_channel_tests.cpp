#include "xport/kvshard_channel.hpp"
#include <cstdio>
#include <functional>

namespace kv = strix::xport::kvshard;
namespace {
unsigned checks = 0;
void require(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("fixture assertion");
}
template<class F> void rejects(F&& f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw);
}
struct Runtime {
    unsigned waits = 0;
    int stream() const { return 7; }
    int publish() const { return 0; }
    void synchronize(int s) { require(s == stream()); ++waits; }
};
struct FakeChannel {
    struct Tag { unsigned kind; std::uint16_t layer; std::uint64_t step; };
    struct Ticket { Tag tag; std::uint64_t sequence; };
    struct Staging { void* data; };
    std::array<unsigned char, 70000> tx{}, rx{};
    std::uint64_t sent = 0, received = 0;
    bool peer_failed = false, fail_on_wait = false;
    std::function<void(unsigned char*)> mutate;
    Staging staging() { return {tx.data()}; }
    void check() const { if (peer_failed) throw std::runtime_error("peer failure"); }
    Ticket post(const Tag& tag, Staging source, std::size_t bytes, int publish) {
        require(source.data == tx.data() && bytes <= tx.size() && publish == 0);
        std::memcpy(rx.data(), tx.data(), bytes);
        if (mutate) mutate(rx.data());
        return {tag, ++sent};
    }
    const void* wait(const Ticket& ticket, int stream) {
        require(stream == 7 && ticket.sequence == received + 1);
        ++received;
        peer_failed = fail_on_wait;
        return rx.data();
    }
};
using Adapter = kv::AttentionChannel<FakeChannel, Runtime>;
constexpr kv::AttentionTag qtag{11, 3, 8191, 4, kv::Phase::queries};
constexpr kv::AttentionTag ptag{11, 3, 8191, 4, kv::Phase::partials};
constexpr FakeChannel::Tag wire{5, 3, 8191};
const std::vector<float> query(Adapter::query_floats, 1.25F);
const std::vector<float> partial(Adapter::partial_floats, 2.5F);
void successful_round() {
    FakeChannel channel;
    Runtime runtime;
    Adapter adapter(channel, runtime, true);
    auto q = adapter.post(qtag, wire, query.data(), query.size());
    rejects([&] { (void)adapter.post(qtag, wire, query.data(), query.size()); });
    auto stale = q;
    ++stale.header[2];
    rejects([&] { (void)adapter.wait(stale); });
    require(adapter.wait(q) == query);
    rejects([&] { (void)adapter.wait(q); });
    rejects([&] { (void)adapter.post(qtag, wire, query.data(), query.size()); });
    auto p = adapter.post(ptag, wire, partial.data(), partial.size());
    require(adapter.wait(p) == partial);
    rejects([&] { (void)adapter.post(qtag, wire, query.data(), query.size()); });
    auto next = qtag;
    ++next.position;
    auto next_wire = wire;
    ++next_wire.step;
    require(adapter.wait(adapter.post(next, next_wire, query.data(), query.size())) == query);
    require(runtime.waits == 3);
}
void stale_peer_frames() {
    // Each protocol field independently binds the returned frame, including
    // request/layer/position/phase/ownership epoch, size and exchange sequence.
    for (std::size_t field = 0; field < 9; ++field) {
        FakeChannel channel;
        Runtime runtime;
        Adapter adapter(channel, runtime, true);
        channel.mutate = [field](unsigned char* bytes) {
            Adapter::Header h{};
            std::memcpy(h.data(), bytes, sizeof(h));
            ++h[field];
            std::memcpy(bytes, h.data(), sizeof(h));
        };
        auto ticket = adapter.post(qtag, wire, query.data(), query.size());
        rejects([&] { (void)adapter.wait(ticket); });
        rejects([&] { (void)adapter.post(qtag, wire, query.data(), query.size()); });
    }
}
void failures() {
    FakeChannel channel;
    Runtime runtime;
    Adapter disabled(channel, runtime, false);
    rejects([&] { (void)disabled.post(qtag, wire, query.data(), query.size()); });
    Adapter adapter(channel, runtime, true);
    rejects([&] { (void)adapter.post(ptag, wire, partial.data(), partial.size()); });
    rejects([&] { (void)adapter.post(qtag, wire, query.data(), query.size() - 1); });
    auto bad_wire = wire;
    ++bad_wire.step;
    rejects([&] { (void)adapter.post(qtag, bad_wire, query.data(), query.size()); });
    auto ticket = adapter.post(qtag, wire, query.data(), query.size());
    adapter.cancel();
    rejects([&] { (void)adapter.wait(ticket); });
    require(channel.received == 0); // cancellation never consumes stale RX
    FakeChannel failed;
    Adapter peer(failed, runtime, true);
    failed.fail_on_wait = true;
    auto pending = peer.post(qtag, wire, query.data(), query.size());
    rejects([&] { (void)peer.wait(pending); });
    rejects([&] { (void)peer.post(qtag, wire, query.data(), query.size()); });
}
}
int main() {
    try {
        successful_round(); stale_peer_frames(); failures();
        std::printf("PASS kvshard Channel fixture checks=%u\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
