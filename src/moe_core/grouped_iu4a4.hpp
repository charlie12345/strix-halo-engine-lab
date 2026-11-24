#pragma once
// Native HIP launch binding for the SDK-free grouped IU4A4 contract.
#if defined(__HIPCC__)
#include "core/iu4a4_dense.hpp"
#include "moe_core/expert_activation.hpp"
#endif
#include "moe_core/grouped_iu4a4_contract.hpp"

namespace strix::moe_core::grouped_iu4a4 {

#if defined(__HIPCC__)
namespace dense = strix::core::iu4a4_dense;

// GroupCta source of a TileRows plan. Tile t: weights / rows / outputs of tiles[t]; its first assignment row is
// (index_tiles[t].input - index_base) / index_tiles[t].input_row (the gate/up tiles locate the assignment rows of both
// projections: the down tiles are the same rows, row for row); image row = map ? map[a] : a. Paired (h != nullptr):
// outputs go to h + a * h_row (assignment order) instead of the tile's output.
struct TileSource {
    const TileRows* tiles{};
    const TileRows* index_tiles{};
    const std::uint16_t* index_base{};
    const std::uint32_t* map{};
    const unsigned* tile_count{};
    const unsigned* failure{};
    const std::uint8_t* codes_from{};
    std::size_t codes_stride{};
    ExpertImage image{};
    unsigned in{}, n{};
    float* h{};
    std::size_t h_row{};
    __device__ __forceinline__ bool active(unsigned t) const {
        return (!failure || *failure == 0U) && (!tile_count || t < *tile_count);
    }
    __device__ __forceinline__ dense::GroupCta bind(unsigned t) const {
        const TileRows tile = tiles[t], index = index_tiles[t];
        const auto e = static_cast<std::size_t>(tile.weights.codes - codes_from) / codes_stride;
        const auto first = static_cast<unsigned>(static_cast<std::size_t>(index.input - index_base) / index.input_row);
        const dense::Weights w{image.codes + e * image.code_bytes, image.steps + e * image.step_floats, in, n};
        if (h) return {w, map, first, tile.rows, h + std::size_t(first) * h_row, h_row};
        return {w, map, first, tile.rows, tile.output, tile.output_row};
    }
};

// Paired epilogue: the incumbent expert activation (moe::SoftcappedSwiglu, BF16 result) stored as FP32.
struct SwigluOut {
    static constexpr bool kPaired = true;
    float gate_cap, up_cap;
    __device__ __forceinline__ float operator()(float gate, float up) const {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
        const moe::SoftcappedSwiglu<expert_activation::detail::DeviceMath> swiglu{gate_cap, up_cap};
        return weight::scalar_float::u2f(std::uint32_t{expert_activation::detail::activate_bf16(swiglu, gate, up)} << 16U);
    }
};

// Load time: expert matrix -> rotated image slot (codes, steps) on stream s.
template <packed::MatrixFormat Format>
inline void enqueue_requantise_expert(const packed::PackedMatrix& matrix, unsigned pair_rows, unsigned in, unsigned n,
                                      std::uint8_t* codes, float* steps, hipStream_t s) {
    dense::enqueue_requantise_source(ExpertSource<Format>{matrix, pair_rows}, in, n, codes, steps, s);
}

// The pass (see Pass) on stream s: four launches. Refuses (hipErrorInvalidValue, nothing enqueued) shapes the class
// does not take, a missing image or a workspace below act_bytes.
inline hipError_t launch(const Pass& p, hipStream_t s) {
    const unsigned gate_width = 2U * p.intermediate;
    if (!p.images || !p.tokens || !p.h || !p.act || !p.assignment_rows || !p.gate_up || !p.down || p.tiles == 0U ||
        !dense::grouped_supported(p.hidden, gate_width, true, p.tiles, p.max_tile_rows) ||
        !dense::grouped_supported(p.intermediate, p.hidden, false, p.tiles, p.max_tile_rows) ||
        reinterpret_cast<std::uintptr_t>(p.act) % 256U != 0U ||
        p.act_size < act_bytes(p.rows, p.assignments, p.hidden, p.intermediate))
        return hipErrorInvalidValue;
    const auto carve = [](std::uint8_t*& at, unsigned rows, unsigned in) {
        dense::Act a{};
        a.rows = rows;
        a.in = in;
        a.codes = at;
        at += align256(dense::act_code_bytes(rows, in));
        a.u = reinterpret_cast<std::uint32_t*>(at);
        at += align256(dense::act_u_bytes(rows, in));
        a.base = reinterpret_cast<float*>(at);
        at += align256(dense::act_base_bytes(rows));
        return a;
    };
    std::uint8_t* at = p.act;
    const dense::Act tokens = carve(at, p.rows, p.hidden), inner = carve(at, p.assignments, p.intermediate);
    TileSource gate{p.gate_up, p.gate_up, p.index_base, p.assignment_rows, p.tile_count, p.failure,
                    p.images->gate_up_codes, p.images->gate_up_stride, p.images->gate_up, p.hidden, gate_width, p.h, p.intermediate};
    TileSource down{p.down, p.gate_up, p.index_base, nullptr, p.tile_count, p.failure,
                    p.images->down_codes, p.images->down_stride, p.images->down, p.intermediate, p.hidden, nullptr, 0U};
    dense::enqueue_quantise_rows(p.tokens, p.hidden, tokens, s);
    dense::enqueue_grouped_gemm(gate, p.tiles, p.max_tile_rows, gate_width, tokens, s, SwigluOut{p.gate_cap, p.up_cap});
    dense::enqueue_quantise_rows(p.h, p.intermediate, inner, s);
    dense::enqueue_grouped_gemm(down, p.tiles, p.max_tile_rows, p.hidden, inner, s);
    return hipGetLastError();
}
#endif

}  // namespace strix::moe_core::grouped_iu4a4

