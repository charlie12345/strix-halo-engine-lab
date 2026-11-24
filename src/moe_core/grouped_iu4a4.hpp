#pragma once
// Rotated integer A4W4 operand class of the grouped expert projection (Home: shared moe_core contract; the HIP body is the
// grouped trait of core/iu4a4_dense.hpp on v_wmma_i32_16x16x16_iu4; a backend without integer 4-bit WMMA declines by
// name). Sibling of grouped_w4a4.hpp (NVFP4 activations on block-scaled FP4 MMA): same plan (TileRows of routed rows of
// one expert), same outputs; both operands change:
//   weights: once at load, each expert matrix (NVFP4 or MXFP8 as stored) is requantised to the rotated IU4 image of
//     dense-iu4a4-rot-v1 (iu4a4_rot::requantise_values on the EXACT BF16 operand value of every weight); a gate/up matrix
//     is built in the paired column order (iu4a4_rot::paired_column) so the SwiGLU epilogue meets gate and up in a lane.
//   activations: the A4 image of dense-iu4a4-rot-v1 (row-major: the grouped launch gathers rows). Gate/up: the TOKEN rows quantised once (not once per assignment),
//     read through the plan's assignment -> token map. Down: the activated rows h (FP32 [assignment][ffn], the BF16 value
//     of the incumbent epilogue) quantised in assignment order.
// Arithmetic identity "grouped-iu4a4-rot-v1": every output is iu4a4_rot::dot of its row and rotated column (the dense
// contract per tile); h = activate_bf16(gate, up) as the incumbent gate/up epilogue. Bits differ from the BF16 tile:
// FAST / TURBO class only, behind gate v2.
#include "core/iu4a4_rot.hpp"
#include "moe_core/grouped_iu4a4_plan.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"
#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__HIPCC__)
#include "core/iu4a4_dense.hpp"
#include "moe_core/expert_activation.hpp"
#define STRIX_GI4_HD __host__ __device__
#else
#define STRIX_GI4_HD
#endif

namespace strix::moe_core::grouped_iu4a4 {

namespace packed = strix::weight::microscaled;
namespace rot = strix::core::iu4a4_rot;
using grouped_microscaled::TileRows;
inline constexpr unsigned kCtaRows = 128U;  // rows per CTA of the grouped GEMM; taller plan tiles run as row slices

constexpr std::size_t image_code_bytes(unsigned in, unsigned n) { return std::size_t(n) * in / 2U; }
constexpr std::size_t image_step_floats(unsigned in, unsigned n) { return std::size_t(n) * (in / rot::kBlock); }
inline bool supported(unsigned in, unsigned n, bool paired) {
    return in > 0U && in % rot::kBlock == 0U && in <= rot::kMaxInputs && n > 0U && n % (paired ? 32U : 16U) == 0U;
}

// Weight source of one expert matrix: rotated column n reads stored row paired_column(n, pair_rows) (pair_rows = 0:
// row n), value = the BF16 operand the EXACT tile decodes (PackedOperand::decode_pair), as FP32.
template <packed::MatrixFormat Format>
struct ExpertSource {
    packed::PackedMatrix matrix;
    unsigned pair_rows{};
    STRIX_GI4_HD float value(std::size_t row, unsigned k) const {
        const packed::PackedOperand<Format> op{matrix};
        const unsigned k0 = k & ~15U, j = k - k0;
        const auto address = op.tile16_address(row, k0);
        const std::uint8_t* codes = address.codes + (packed::PackedOperand<Format>::kNvfp4 ? j / 2U : (j & ~1U));
        const std::uint32_t pair = op.decode_pair(codes, packed::PackedOperand<Format>::decode_scale(*address.scale_code));
        const std::uint32_t half = (j & 1U) != 0U ? pair >> 16U : pair & 0xffffU;
        return weight::scalar_float::u2f(half << 16U);
    }
    STRIX_GI4_HD float requantise(unsigned col, unsigned b, unsigned n, std::uint8_t* dst) const {
        const std::size_t row = pair_rows != 0U ? rot::paired_column(col, pair_rows) : col;
        const auto v = [&](unsigned i) { return value(row, b * rot::kBlock + i); };
        return rot::requantise_values(v, b, col, n, dst);
    }
};

// Expert index of a tile from its weights' code address (host check before binding: every expert matrix of the layer
// starts at codes_from + e * codes_stride, so the address is an index; the image then holds expert e at e).
inline bool expert_codes_strided(std::span<const packed::PackedMatrix> experts, const std::uint8_t* codes_from, std::size_t codes_stride) {
    for (std::size_t e = 0; e < experts.size(); ++e)
        if (experts[e].codes != codes_from + e * codes_stride) return false;
    return codes_stride > 0U;
}

static_assert(rot::kBlock == kInputBlock, "act_image_bytes assumes the contract's 128-input block");

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

#undef STRIX_GI4_HD
