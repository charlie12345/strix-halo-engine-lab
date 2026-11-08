#include "minimax/pfx/pf_wire.hpp"
#include "minimax/pfx/pf_route.hpp"
#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
namespace px = strix::minimax::pfx;
using Message = std::vector<std::byte>;
void require(bool value) { if (!value) throw std::runtime_error("MiniMax pfx wire/route fixture"); }
struct Options {
    std::uint64_t epoch = 7, plan = 0x5846504dULL;
    std::size_t chunk = 65536;
    std::uint32_t rank = 0;
    bool crc = true;
    int stream = 3;
};
struct FakeTransport {
    std::vector<Message> messages;
    void exchange(std::span<const std::byte> source, std::span<std::byte> destination) {
        messages.emplace_back(source.begin(), source.end());
        std::copy(source.begin(), source.end(), destination.begin());
    }
};
// Frozen parent serializer: independent of PrefillTraits, including its native
// 64-byte identity, CRC bit, chunk boundaries and trailer CRC byte ordering.
std::vector<Message> parent_messages(const Options& options, std::uint64_t sequence,
                                    const px::Tag& tag, const Message& payload,
                                    const std::vector<std::uint32_t>& crcs) {
    struct OldHeader {
        std::uint64_t magic, epoch, plan, sequence, request, bytes;
        std::uint32_t kind, layer, chunk, rank;
    };
    static_assert(sizeof(OldHeader) == 64);
    OldHeader identity{0x3148584650584d4dULL, options.epoch, options.plan, sequence,
                       tag.request, payload.size(), tag.kind, tag.layer,
                       static_cast<std::uint32_t>(options.chunk) | (options.crc ? 0x80000000U : 0U), options.rank};
    std::vector<Message> result;
    result.emplace_back(sizeof(identity));
    std::memcpy(result.back().data(), &identity, sizeof(identity));
    for (std::size_t offset = 0; offset < payload.size(); offset += options.chunk) {
        const auto bytes = std::min(options.chunk, payload.size() - offset);
        result.emplace_back(payload.data() + offset, payload.data() + offset + bytes);
    }
    if (options.crc) {
        identity.magic = 0x3154584650584d4dULL;
        result.emplace_back(sizeof(identity) + crcs.size() * 4);
        std::memcpy(result.back().data(), &identity, sizeof(identity));
        std::memcpy(result.back().data() + sizeof(identity), crcs.data(), crcs.size() * 4);
    }
    return result;
}
struct FakeExchanger {
    struct View { const void* local; const void* peer; };
    Options settings;
    FakeTransport transport;
    Message tx_slot, rx_slot;
    std::vector<std::uint32_t> crcs;
    std::vector<char> operations;
    std::uint64_t sequence = 9;
    const Options& options() const { return settings; }
    std::uint64_t sent() const { return sequence; }
    void* tx() { return tx_slot.data(); }
    void send_copy(const void* source, std::size_t bytes, const px::Tag& tag) {
        operations.push_back('c');
        std::memcpy(tx_slot.data(), source, bytes);
        send_wire(bytes, tag);
    }
    void send(std::size_t bytes, const px::Tag& tag) {
        operations.push_back('p');
        send_wire(bytes, tag);
    }
    void send_wire(std::size_t bytes, const px::Tag& tag) {
        const auto identity = px::PrefillTraits::header(settings, ++sequence, bytes, tag);
        px::Header peer;
        transport.exchange({reinterpret_cast<const std::byte*>(&identity), sizeof(identity)},
                           {reinterpret_cast<std::byte*>(&peer), sizeof(peer)});
        for (std::size_t offset = 0; offset < bytes; offset += settings.chunk) {
            const auto count = std::min(settings.chunk, bytes - offset);
            transport.exchange({tx_slot.data() + offset, count}, {rx_slot.data() + offset, count});
        }
        if (settings.crc) {
            Message trailer(sizeof(px::Header) + crcs.size() * 4), incoming(trailer.size());
            px::PrefillTraits::trailer(trailer.data(), identity, crcs.data(), crcs.size());
            transport.exchange(trailer, incoming);
        }
    }
    View recv() { operations.push_back('w'); return {tx_slot.data(), rx_slot.data()}; }
    void retire() { operations.push_back('r'); }
    void drain() { operations.push_back('d'); }
};
void check(std::size_t rows, std::size_t width, bool crc, std::uint32_t rank, bool inplace) {
    FakeExchanger runtime;
    runtime.settings.crc = crc;
    runtime.settings.rank = rank;
    const auto bytes = px::PrefillTraits::payload_bytes(rows, width);
    Message payload(bytes);
    for (std::size_t i = 0; i < bytes; ++i) payload[i] = static_cast<std::byte>(i % 251);
    runtime.tx_slot = payload;
    runtime.rx_slot.resize(bytes);
    const auto chunks = (bytes + runtime.settings.chunk - 1) / runtime.settings.chunk;
    for (std::size_t i = 0; i < chunks; ++i) runtime.crcs.push_back(static_cast<std::uint32_t>(i) ^ 0x92abcdefU);
    const px::Tag tag{rank + 1, 17, 23};
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        runtime.transport.messages.clear();
        runtime.operations.clear();
        const auto expected = parent_messages(runtime.settings, runtime.sent() + 1, tag, payload, runtime.crcs);
        px::Route<FakeExchanger, px::Tag> route(runtime);
        const auto ticket = route.post(inplace ? runtime.tx() : payload.data(), bytes, tag);
        const auto view = route.wait(ticket);
        require(view.local == runtime.tx() && view.peer == runtime.rx_slot.data());
        require(runtime.rx_slot == payload);
        require(runtime.transport.messages == expected);
        // The engine enqueues its rank-ordered consumer here, then retires once.
        runtime.operations.push_back('k');
        runtime.retire();
        route.release(ticket);
        require(runtime.operations == std::vector<char>({inplace ? 'p' : 'c', 'w', 'k', 'r'}));
    }
}
} // namespace
int main() {
    for (const auto rows : std::array<std::size_t, 7>{1, 8, 32, 1024, 4096, 8192, 32768})
        for (const bool crc : {false, true})
            for (const std::uint32_t rank : {0U, 1U})
                for (const bool inplace : {false, true}) check(rows, 8, crc, rank, inplace);
    check(3, 6144, true, 0, false); // full hidden rows and a partial final wire chunk
}
