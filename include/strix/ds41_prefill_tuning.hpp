#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strix::ds41::prefill_tuning {

inline bool enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

// Resource-based candidates, not a measured performance table. Not wired into dispatch:
// the split-K prefill kernels stay the policy until a workload campaign selects one.
constexpr unsigned attention_rows(unsigned positions, unsigned chunk_rows, unsigned heads,
                                  unsigned context_positions = 0U, unsigned lanes = 4U) {
    unsigned rows = positions >= 8192U && heads >= 32U ? 256U : positions >= 1024U ? 128U : 64U;
    const std::uint64_t context = context_positions ? context_positions : positions;
    // The current caller allocates partials for 64 heads even with head TP.
    // Bound all attention-lane scratch to 2 GiB (scores + keys + query prep +
    // partials + tickets). Candidate masks and row buffers are separate.
    const auto scratch = [&](unsigned r) {
        return std::uint64_t{r} * (64U * 10U * 514U * 4U + 8U * (context + 4U * r + 1U) + 4736U + 32U);
    };
    while (rows > 1U && (rows > chunk_rows || chunk_rows % rows != 0U ||
                         scratch(rows) * lanes > (std::uint64_t{2} << 30U))) rows /= 2U;
    return rows;
}

constexpr unsigned dense_outputs(unsigned width, unsigned outputs, unsigned rows) {
    // A 16-output tile halves the live class accumulators for long K chains.
    return width >= 4096U && outputs >= 512U && rows >= 16U ? 16U : 32U;
}

constexpr unsigned moe_rows(unsigned hidden, unsigned down_width, unsigned rows, unsigned top_k, unsigned experts) {
    // Two row tiles amortise weight decoding only when there are enough entries
    // and both activation stages are small. Long-K shapes keep the split tile.
    return hidden <= 1280U && down_width <= 1280U &&
                   std::uint64_t{rows} * top_k >= std::uint64_t{experts} * 32U ? 32U : 16U;
}

constexpr bool down_stage(unsigned width, unsigned outputs) {
    return width <= 1280U && outputs >= 512U;
}

}  // namespace strix::ds41::prefill_tuning
