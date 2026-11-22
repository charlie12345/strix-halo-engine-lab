#pragma once

// Geometry helpers run on the host (launch grids) and in the kernels (column and row maps).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_HD __host__ __device__
#else
#define STRIX_R1_HD
#endif

namespace strix::moe_core::microscaled_r1 {

// Launch geometry. A wave owns `columns` consecutive outputs and streams their rows together (gate_up: the gate and
// up rows of each column; down_combine: one row per column per routed expert). Each row keeps its own accumulator,
// lane->K mapping and butterfly, so the column count changes no output bit. It sets how many rows share one wave's
// dependent load chain (ids -> matrices -> codes -> scales).
struct Geometry {
    unsigned gate_columns, down_columns;
};
inline constexpr unsigned kWavesPerBlock = 8;
inline constexpr Geometry kHipGeometry{2U, 4U};
inline constexpr Geometry kCudaGeometry{1U, 1U};

STRIX_R1_HD constexpr unsigned gate_blocks(unsigned intermediate, unsigned columns) {
    return (intermediate - 1U) / (kWavesPerBlock * columns) + 1U;
}
STRIX_R1_HD constexpr unsigned gate_first_column(unsigned block, unsigned wave, unsigned columns) {
    return (block * kWavesPerBlock + wave) * columns;
}
STRIX_R1_HD constexpr unsigned down_blocks(unsigned hidden, unsigned columns) { return (hidden - 1U) / columns + 1U; }
// Weight row read by output slot i of a wave whose first column is col; `live` columns remain in the matrix.
// Slots past the live columns re-read the first valid row (their results are never stored).
STRIX_R1_HD constexpr unsigned slot_row(unsigned col, unsigned i, unsigned columns, unsigned stride, unsigned live) {
    const unsigned offset = i % columns < live ? i % columns : 0U;
    return col + offset + (i / columns) * stride;
}

}  // namespace strix::moe_core::microscaled_r1

#undef STRIX_R1_HD
