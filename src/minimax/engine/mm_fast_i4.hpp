#pragma once
// Host-only model scheduling glue. All conversion and arithmetic stay in the library.
#include <array>
#include <stdexcept>
#include <string_view>

namespace strix::minimax::engine::fast_i4 {
enum class Scope { all, gate_up };
inline Scope scope(const char* value) {
    const std::string_view s = value == nullptr ? "" : value;
    if (s.empty() || s == "all") return Scope::all;
    if (s == "gate_up") return Scope::gate_up;
    throw std::invalid_argument("STRIX_MMX_FAST_I4_SCOPE must be all or gate_up");
}

// One acquire/release surrounds ALL chunks of a layer. A new group/range must
// begin afresh; a pack is never treated as a resident layer across groups.
struct Schedule {
    static constexpr unsigned slots = 2;
    std::array<int, slots> layer{{-1, -1}};
    void begin() { layer.fill(-1); }

    // convert(slot, layer, side) must wait for the slot's last release before
    // side-stream writes, then publish ready. wait(slot) is an engine-stream wait.
    template<class Convert, class Wait>
    unsigned acquire(unsigned L, unsigned end, bool prefetch, Convert convert, Wait wait) {
        if (L >= end) throw std::invalid_argument("fast_i4: layer outside prefill range");
        const unsigned s = L & 1U;
        if (layer[s] != static_cast<int>(L)) {
            convert(s, L, false);  // serial on the engine stream, including the first layer
            layer[s] = static_cast<int>(L);
        } else {
            wait(s);
        }
        if (prefetch && L + 1U < end) {
            const unsigned next = L + 1U, other = next & 1U;
            convert(other, next, true);
            layer[other] = static_cast<int>(next);
        }
        return s;
    }
};
}  // namespace strix::minimax::engine::fast_i4
