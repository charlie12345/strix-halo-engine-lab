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
// outputs go to h + a * h_row (assignment order) instead of the tile's output. A tile whose weights are not an expert
// slot of the image (address off the base + e * stride rule, or e >= experts), whose index row is not a whole row of
// the base, whose rows differ from its index tile's or exceed max_rows, or whose rows run past index_rows binds no rows
// (stores nothing) and sets kFaultBounds in fault.
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
    unsigned index_rows{}, max_rows{};
    unsigned* fault{};
    __device__ __forceinline__ bool active(unsigned t) const {
        return (!failure || *failure == 0U) && (!tile_count || t < *tile_count);
    }
    __device__ __forceinline__ dense::GroupCta bind(unsigned t) const {
        const TileRows tile = tiles[t], index = index_tiles[t];
        const auto wa = reinterpret_cast<std::uintptr_t>(tile.weights.codes), wb = reinterpret_cast<std::uintptr_t>(codes_from);
        const auto ia = reinterpret_cast<std::uintptr_t>(index.input), ib = reinterpret_cast<std::uintptr_t>(index_base);
        const std::size_t e = wa >= wb ? (wa - wb) / codes_stride : ~std::size_t{0};
        const std::size_t at = ia >= ib ? (ia - ib) / sizeof(std::uint16_t) : ~std::size_t{0};
        const std::size_t first = index.input_row != 0U ? at / index.input_row : ~std::size_t{0};
        const bool bound = wa >= wb && (wa - wb) % codes_stride == 0U && e < image.experts && ia >= ib &&
                           (ia - ib) % sizeof(std::uint16_t) == 0U && index.input_row != 0U && at % index.input_row == 0U &&
                           tile.rows == index.rows && tile.rows <= max_rows && first <= index_rows &&
                           tile.rows <= index_rows - first;
        if (!bound) {
            if (fault && threadIdx.x == 0U) atomicOr(fault, kFaultBounds);
            return {};  // no rows: the CTA leaves before its first barrier
        }
        const dense::Weights w{image.codes + e * image.code_bytes, image.steps + e * image.step_floats, in, n};
        const auto row0 = static_cast<unsigned>(first);
        if (h) return {w, map, row0, tile.rows, h + first * h_row, h_row, fault};
        return {w, map, row0, tile.rows, tile.output, tile.output_row, fault};
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

static_assert(kFaultBounds == dense::kBoundsFault, "one fault bit for the class and its core");

// The pass (see Pass) on stream s: four launches, each submitted only after the previous one was accepted. Refuses
// (hipErrorInvalidValue, nothing enqueued) what refusal() names and grids the grouped launch cannot take. After a planner
// failure or a bounds fault of the gate/up launch no later tile stores (the quantiser of h reads rows no tile consumes).
inline hipError_t launch(const Pass& p, hipStream_t s) {
    const unsigned gate_width = 2U * p.intermediate;
    if (refusal(p) != nullptr || !dense::grouped_supported(p.hidden, gate_width, true, p.tiles, p.max_tile_rows) ||
        !dense::grouped_supported(p.intermediate, p.hidden, false, p.tiles, p.max_tile_rows))
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
                    p.images->gate_up_codes, p.images->gate_up_stride, p.images->gate_up, p.hidden, gate_width, p.h, p.intermediate,
                    p.assignments, p.max_tile_rows, p.fault};
    TileSource down{p.down, p.gate_up, p.index_base, nullptr, p.tile_count, p.failure,
                    p.images->down_codes, p.images->down_stride, p.images->down, p.intermediate, p.hidden, nullptr, 0U,
                    p.assignments, p.max_tile_rows, p.fault};
    dense::enqueue_quantise_rows(p.tokens, p.hidden, tokens, s);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_grouped_gemm(gate, p.tiles, p.max_tile_rows, gate_width, tokens, s, SwigluOut{p.gate_cap, p.up_cap});
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_quantise_rows(p.h, p.intermediate, inner, s);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_grouped_gemm(down, p.tiles, p.max_tile_rows, p.hidden, inner, s);
    return hipGetLastError();
}
#endif

}  // namespace strix::moe_core::grouped_iu4a4

