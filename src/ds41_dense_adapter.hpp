#pragma once
#include "ds41_integer_operand.hpp"
#include "strix/hip_ds41_layer.hpp"
namespace strix::ds41::integer_core {
// Dense is one group on tiled_core; row/column tails and strides are model glue.
struct DenseAdapter {
    static constexpr bool kActive = true, kTiles = true;
    struct Tile {
        int rows = 0, first_col = 0, split = 0, splits = 1;
        const unsigned* packed = nullptr;
        unsigned begin = 0;
    };
    Q8A8DensePrefillView view;
    __device__ bool next(Tile& t, unsigned iter, int bn) const {
        if (iter) return false;
        t.begin = blockIdx.x * 16U;
        if (t.begin >= view.rows) return false;
        t.rows = int(min(16U, view.rows - t.begin));
        t.first_col = int(blockIdx.y) * bn;
        return true;
    }
    __device__ const signed char* code_row(const Tile& t, int r) const {
        return view.codes + std::uint64_t(t.begin + unsigned(r)) * view.x_stride;
    }
    __device__ float delta(const Tile& t, int r, int group) const {
        return view.deltas[std::uint64_t(t.begin + unsigned(r)) * (view.x_stride / 32U) + unsigned(group)];
    }
    __device__ const unsigned long long* grid() const { return nullptr; }
    __device__ const unsigned char* weight_row(const Tile&, int col) const {
        const unsigned row_bytes = (view.row_width ? view.row_width : view.width) / 32U * 34U;
        return view.weight + std::uint64_t(min(unsigned(col), view.count - 1U)) * row_bytes + view.k_offset / 32U * 34U;
    }
    __device__ void store_raw(const Tile& t, int r, int col, float value) const {
        if (unsigned(col) >= view.count) return;
        auto* dst = view.out + std::uint64_t(t.begin + unsigned(r)) * view.out_stride + unsigned(col);
        *dst = view.accumulate ? *dst + value : value;
    }
    __device__ void finish_tile(const Tile& t, const float* values, int bm, int bn) const {
        if (!view.a8_codes) return;
        const unsigned lane = threadIdx.x & 31U, waves = blockDim.x / 32U;
        for (unsigned r = threadIdx.x / 32U; r < unsigned(bm); r += waves) {
            if (r >= unsigned(t.rows)) continue;
            const auto a = strix_trellis::quantize_a8_row32(values[r * unsigned(bn) + lane]);
            const auto row = std::uint64_t(t.begin + r);
            const unsigned col = unsigned(t.first_col) + lane;
            if (col < view.count) view.a8_codes[row * view.a8_stride + col] = a.code;
            if (!lane && unsigned(t.first_col) < view.count) {
                const auto g = row * (view.a8_stride / 32U) + unsigned(t.first_col) / 32U;
                view.a8_deltas[g] = a.delta; view.a8_sums[g] = a.sum;
            }
        }
    }
};
}
