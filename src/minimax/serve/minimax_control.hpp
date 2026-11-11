#pragma once

#include "strix/minimax/prefill_types.hpp"
#include "strix/pair/lockstep.hpp"
#include <array>
#include <cstddef>
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
    // prefill: first position; a reused prefix of the slot's cache precedes it (0 = the whole prompt). pstart and rank
    // land in the 512-byte frame's former zero padding: rank 0 with a zero pstart sends the same bytes as before.
    std::uint32_t pstart = 0;
    // The sender's pair rank. With round / ack_round it forms the strix::pair::LockstepHeader every exchange checks; a
    // peer built before the field reports rank 0 on both ranks and is refused as "same rank" instead of misrunning.
    std::uint32_t rank = 0;
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
    local.rank = static_cast<std::uint32_t>(eng.rank());
    // Copy semantic fields only: Control and PfProgress padding is indeterminate,
    // including on repeated hello calls. Every padding byte on the wire stays zero.
    const auto put = [&](std::size_t offset, const auto& value) {
        std::memcpy(o.data() + offset, &value, sizeof(value));
    };
#define STRIX_MM_CONTROL_FIELD(f) put(offsetof(Control, f), local.f)
    STRIX_MM_CONTROL_FIELD(magic); STRIX_MM_CONTROL_FIELD(version); STRIX_MM_CONTROL_FIELD(command);
    STRIX_MM_CONTROL_FIELD(task_id); STRIX_MM_CONTROL_FIELD(parent_task_id); STRIX_MM_CONTROL_FIELD(slot_generation);
    const auto progress = [&](std::size_t offset, const strix::minimax::engine::PfProgress& p) {
        using P = strix::minimax::engine::PfProgress;
#define STRIX_MM_PROGRESS_FIELD(f) put(offset + offsetof(P, f), p.f)
        STRIX_MM_PROGRESS_FIELD(id); STRIX_MM_PROGRESS_FIELD(parent); STRIX_MM_PROGRESS_FIELD(generation);
        STRIX_MM_PROGRESS_FIELD(c0); STRIX_MM_PROGRESS_FIELD(chunk); STRIX_MM_PROGRESS_FIELD(rows);
        STRIX_MM_PROGRESS_FIELD(slot); STRIX_MM_PROGRESS_FIELD(bank); STRIX_MM_PROGRESS_FIELD(layer);
        STRIX_MM_PROGRESS_FIELD(next); STRIX_MM_PROGRESS_FIELD(open);
#undef STRIX_MM_PROGRESS_FIELD
    };
    progress(offsetof(Control, active), local.active); progress(offsetof(Control, parked), local.parked);
    STRIX_MM_CONTROL_FIELD(rows); STRIX_MM_CONTROL_FIELD(pslot); STRIX_MM_CONTROL_FIELD(plen);
    STRIX_MM_CONTROL_FIELD(pflags); STRIX_MM_CONTROL_FIELD(bank);
    STRIX_MM_CONTROL_FIELD(round); STRIX_MM_CONTROL_FIELD(ack_round);
    static_assert(sizeof(strix::minimax::engine::Row) == 2 * sizeof(std::uint32_t) + sizeof(std::uint64_t));
    STRIX_MM_CONTROL_FIELD(row); STRIX_MM_CONTROL_FIELD(layers); STRIX_MM_CONTROL_FIELD(append);
    STRIX_MM_CONTROL_FIELD(dslots); STRIX_MM_CONTROL_FIELD(dslot); STRIX_MM_CONTROL_FIELD(danchor);
    STRIX_MM_CONTROL_FIELD(dpos); STRIX_MM_CONTROL_FIELD(pstart); STRIX_MM_CONTROL_FIELD(rank);
#undef STRIX_MM_CONTROL_FIELD
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
    try {
        strix::pair::check_pair({local.round, local.ack_round, local.rank}, {in.round, in.ack_round, in.rank});
    } catch (const strix::pair::ProtocolError& e) {
        eng.abort(e.what());
        throw;
    }
    return in;
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

// One lockstep round: stamps out's round / ack_round from lock, exchanges, then checks the peer's header and advances.
template<class Engine>
Control exchange(Engine& eng, Control out, strix::pair::Lockstep& lock) {
    const auto header = lock.header();
    out.round = header.round;
    out.ack_round = header.ack_round;
    Control in = exchange(eng, out);
    lock.accept({in.round, in.ack_round, in.rank});
    return in;
}

// prefix: this rank's prefix-reuse vote; *prefix_agreed (when given) is true only if both ranks voted for it.
template<class Engine>
bool session_hello(Engine& eng, bool requested, bool prefix = false, bool* prefix_agreed = nullptr,
                   std::uint64_t prefix_rows = 1) {
    Control hello;
    hello.pflags = (requested ? kHelloSessionVote : 0U) | (prefix ? kHelloPrefixVote : 0U);
    hello.dpos[0] = prefix ? prefix_rows : 0U;  // OFF keeps the legacy frame; ON agrees group alignment too
    const auto peer = exchange(eng, hello);
    if (prefix_agreed != nullptr) *prefix_agreed = (hello.pflags & peer.pflags & kHelloPrefixVote) != 0 &&
        hello.dpos[0] != 0 && hello.dpos[0] == peer.dpos[0];
    return ((hello.pflags | peer.pflags) & kHelloSessionVote) != 0;
}
}  // namespace strix::minimax::serve
