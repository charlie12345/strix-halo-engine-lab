#pragma once
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "strix/env_flags.hpp"

namespace strix::core {
// Physical row stride of a row-major device buffer (one logical row per token / request).
//
// A kernel that walks many rows in lockstep (the same column of 16 or more rows at once: row-batched projections,
// tiled matrix loads, 16-bit epilogues) touches addresses that are one stride apart. When the stride is a multiple
// of the page size every one of those addresses has the same page offset, and the rows of one tile compete for the
// same cache sets and translation entries: measured 1.7x on a recompute projection at 20,480 bytes against 20,608.
//
// The policy keeps the logical row width and the caller's alignment and only moves rows apart: a stride whose rows
// repeat a page offset sooner than `min_period` rows is extended by `dealias_bytes` (a multiple of 64, so 64-byte
// row alignment for 512-bit loads is preserved) until it no longer does. Values and the order of every computation
// are untouched; consumers must take the stride from here (pointer + stride), never recompute it from the width.
struct RowLayoutPolicy {
    std::uint64_t alias_bytes = 4096;     // period of the aliasing resource (page / cache way size)
    std::uint64_t dealias_bytes = 128;    // step added to an aliasing stride; 0 keeps the tight stride
    std::uint32_t min_period = 2;         // rows before a page offset may repeat (2: only exact multiples move)
    std::uint32_t lockstep_rows = 16;     // buffers with fewer rows are never walked in lockstep
};

inline constexpr std::uint64_t kRowAlignment = 64;

// Rows after which stride * n returns to the same offset inside the aliasing period.
[[nodiscard]] constexpr std::uint64_t row_alias_period(std::uint64_t stride, std::uint64_t alias_bytes = 4096) {
    return stride && alias_bytes ? alias_bytes / std::gcd(stride, alias_bytes) : 1U;
}

[[nodiscard]] constexpr bool row_layout_valid(const RowLayoutPolicy& p) {
    return p.alias_bytes && !(p.alias_bytes & (p.alias_bytes - 1U)) && p.dealias_bytes % kRowAlignment == 0 &&
        p.min_period >= 1U && p.min_period <= p.alias_bytes / kRowAlignment &&
        (!p.dealias_bytes || row_alias_period(p.dealias_bytes, p.alias_bytes) >= p.min_period);
}

// `tight` is the stride the buffer would have without the policy (payload rounded to the buffer's own alignment).
// The result is >= tight, congruent to it modulo 64, and equal to it when rows < lockstep_rows, when the policy is
// off, or when the tight stride does not alias.
[[nodiscard]] constexpr std::uint64_t physical_row_stride(const RowLayoutPolicy& p, std::uint64_t tight,
                                                          std::uint64_t rows) {
    if (!row_layout_valid(p)) throw std::invalid_argument("row layout policy is invalid");
    if (!tight || !p.dealias_bytes || rows < p.lockstep_rows) return tight;
    auto stride = tight;
    // dealias_bytes has period >= min_period, so at most alias_bytes / 64 steps are ever needed.
    for (std::uint64_t step = 0; row_alias_period(stride, p.alias_bytes) < p.min_period; ++step) {
        if (step > p.alias_bytes / kRowAlignment || stride > UINT64_MAX - p.dealias_bytes)
            throw std::overflow_error("row layout stride overflows");
        stride += p.dealias_bytes;
    }
    return stride;
}

// The one resolved policy of the process: engine, server and fixtures all lay rows out through it.
//   STRIX_ROW_DEALIAS_BYTES  step in bytes (multiple of 64); 0 restores the tight strides
//   STRIX_ROW_DEALIAS_PERIOD rows before a page offset may repeat (power of two, 2..64)
[[nodiscard]] inline const RowLayoutPolicy& row_layout_policy() {
    static const RowLayoutPolicy resolved = [] {
        RowLayoutPolicy p;
        const int bytes = envcfg::integer("STRIX_ROW_DEALIAS_BYTES", static_cast<int>(p.dealias_bytes));
        const int period = envcfg::integer("STRIX_ROW_DEALIAS_PERIOD", static_cast<int>(p.min_period));
        if (bytes < 0 || period < 1) throw std::invalid_argument("row layout environment is invalid");
        p.dealias_bytes = static_cast<std::uint64_t>(bytes);
        p.min_period = static_cast<std::uint32_t>(period);
        if (!row_layout_valid(p)) throw std::invalid_argument("row layout environment is invalid");
        return p;
    }();
    return resolved;
}

[[nodiscard]] inline std::uint64_t physical_row_stride(std::uint64_t tight, std::uint64_t rows) {
    return physical_row_stride(row_layout_policy(), tight, rows);
}
}  // namespace strix::core
