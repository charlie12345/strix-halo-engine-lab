#pragma once

#include <cstddef>
#include <limits>

namespace strix::cuda_core {

// The input-major F32 form gives each output thread a complete ascending-K
// fused chain per activation row. Staging does not partition the reduction.
struct F32InputMajorRows {
    static constexpr unsigned rows = 16, outputs = 128, k = 64;
};

struct F32InputMajorGrid {
    unsigned row_tiles = 0, output_tiles = 0;
    [[nodiscard]] constexpr bool valid() const noexcept { return output_tiles != 0; }
};

[[nodiscard]] constexpr F32InputMajorGrid plan_f32_input_major_rows(
    std::size_t rows, unsigned in, unsigned out, unsigned out_stride) noexcept {
    using Tile = F32InputMajorRows;
    if (in == 0 || in % Tile::k != 0 || out == 0 || out_stride < out) return {};
    const auto row_tiles = rows == 0 ? 0 : (rows - 1) / Tile::rows + 1;
    const auto output_tiles = (out - 1) / Tile::outputs + 1;
    constexpr auto max_floats = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (row_tiles > 0x7fffffffU || output_tiles > 65535U ||
        rows > max_floats / in || rows > max_floats / out_stride || in > max_floats / out) return {};
    return {static_cast<unsigned>(row_tiles), output_tiles};
}

}  // namespace strix::cuda_core
