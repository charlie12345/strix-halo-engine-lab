#pragma once

#include "strix/minimax/prefill_types.hpp"
#include <array>
#include <cstring>
#include <stdexcept>

namespace strix::minimax::serve {
struct Control {
    std::uint32_t magic = 0x43534d4dU;  // "MMSC"
    std::uint32_t version = 3;
    std::uint32_t command = 0;
    std::uint64_t task_id = 0, parent_task_id = 0, slot_generation = 0;
    strix::minimax::engine::PfProgress active{}, parked{};
    std::uint32_t rows = 0;             // step: rows used in row[]
    std::uint32_t pslot = 0, plen = 0;  // prefill: slot and (suffix) token count sent after this control
    std::uint32_t pflags = 0, bank = 0;
    std::uint64_t round = 0, ack_round = ~std::uint64_t{0};
    strix::minimax::engine::Row row[8]{};
    std::uint32_t layers = 0;  // unit
    std::uint32_t append = 0;  // step
    std::uint32_t dslots = 0;  // draft
    std::uint32_t dslot[8]{}, danchor[8]{};
    std::uint64_t dpos[8]{};
    // prefill: first position; a reused prefix of the slot's cache precedes it (0 = the whole prompt). Last field on
    // purpose: it lands in the 512-byte frame's former zero padding, so version-3 peers still read every prior field and
    // a zero pstart (every command but a reusing prefill) leaves the wire bytes identical to before.
    std::uint32_t pstart = 0;
};
static_assert(sizeof(Control) <= 512);

// Hello reuses pflags; zero retains the ordinary version-2 wire encoding.
inline constexpr std::uint32_t kHelloSessionVote = 1U;
// This rank asks for prefix reuse and replays a prefill that opens at pstart. Reuse needs both votes: a peer that
// does not set the bit (the switch off, or a build that predates pstart) keeps every prefill whole.
inline constexpr std::uint32_t kHelloPrefixVote = 2U;

template<class Engine>
Control exchange(Engine& eng, const Control& out) try {
    std::array<std::byte, 512> o{}, i{};
    Control local = out;
    local.active = eng.prefill_progress();
    local.parked = eng.prefill_progress(true);
    std::memcpy(o.data(), &local, sizeof(Control));
    eng.exchange_bytes(o.data(), i.data(), o.size());
    Control in;
    std::memcpy(&in, i.data(), sizeof(Control));
    if (in.magic != out.magic) {
        eng.abort("control channel: bad magic");
        throw std::runtime_error("control channel: bad magic from the peer");
    }
    if (in.version != local.version || in.active != local.active || in.parked != local.parked) {
        eng.abort("control channel: protocol / task checkpoint mismatch");
        throw std::runtime_error("control channel: protocol / task checkpoint mismatch");
    }
    if (local.command != 0 && in.command == 0 &&
        in.ack_round != (out.round == 0 ? ~std::uint64_t{0} : out.round - 1)) {
        eng.abort("control channel: preceding round not acknowledged");
        throw std::runtime_error("control channel: preceding round not acknowledged");
    }
    return in;
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

// prefix: this rank's prefix-reuse vote; *prefix_agreed (when given) is true only if both ranks voted for it.
template<class Engine>
bool session_hello(Engine& eng, bool requested, bool prefix = false, bool* prefix_agreed = nullptr) {
    Control hello;
    hello.pflags = (requested ? kHelloSessionVote : 0U) | (prefix ? kHelloPrefixVote : 0U);
    const auto peer = exchange(eng, hello);
    if (prefix_agreed != nullptr) *prefix_agreed = (hello.pflags & peer.pflags & kHelloPrefixVote) != 0;
    return ((hello.pflags | peer.pflags) & kHelloSessionVote) != 0;
}
}  // namespace strix::minimax::serve
