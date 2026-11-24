#pragma once

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || verify),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

// The one prompt-row rule of an opted-in W4A4 expert class, shared by the CLI session and the served lanes: every
// prefill-kind row runs the class whatever the chunk geometry, and decode / verify rows keep the BF16 tile. A one-row
// prefill chunk therefore takes the grouped route (the packed one-row route has no W4A4 class).
constexpr bool mini_w4a4_rows(bool w4a4_bound, MiniAppendKind kind) {
    return w4a4_bound && kind == MiniAppendKind::prefill;
}

constexpr MiniResidentRoute mini_resident_chunk_route(MiniAppendKind kind, unsigned rows, bool packed_decode,
                                                      bool w4a4_bound) {
    return mini_resident_route(kind, rows, packed_decode && !mini_w4a4_rows(w4a4_bound, kind), false);
}

}  // namespace strix::inkling2
