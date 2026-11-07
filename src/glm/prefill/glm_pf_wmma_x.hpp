#pragma once
// GLM prefill FAST identity glm-pf-wmma-x (hwops G-01b): the exact X cores' operands on v_wmma_f32_16x16x16_f16.
// Same FP16 operands as exact_x::core_s (trellis tiles decoded with mul1_pair, transformed FP16 rows), same epilogue
// (H128 on the 128-output block, svh scale). The accumulation is new: per (row, output) and stage of four k-units one
// WMMA chain from +0, stages added in ascending order into an FP32 total (v_add_f32), no K splits and no reduce4, so
// the bits are a new identity (FAST only, opt-in).
//   CTA  = 8 waves x 128 outputs (wave w = output tile w, the tile it decodes in the exact cores) x 16 * MT rows.
//   k    = stages of four 16-wide k-units: the activation rows are staged in LDS once per CTA, each wave transposes its
//          decoded tile through its own LDS rows into a B fragment (lane = output, 16 k).
#include "glm/moe/glm_trellis_device.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strix::glm::pf::wmma_x {
namespace d = moe::dev;
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));

// enabled() (STRIX_GLM_PF_WMMA=1) lives in glm_pf_wmma.hpp (host side, also read by the rank startup agreement).

// Full tiles of 128 rows (MT 8: each decoded weight tile feeds eight WMMAs; measured whole-call MoE -27..-30 % vs
// -24..-26 % at 64 rows); STRIX_GLM_PF_WMMA_ROWS=64 keeps 64-row tiles. Same arithmetic per (row, output): same bits.
inline unsigned tile_rows() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_ROWS");
    return p && std::strcmp(p, "64") == 0 ? 64U : 128U;
}

// 128-row tiles: each wave computes two output tiles x four row tiles (CP 2: A fragments reused across the pair,
// 1.5 instead of 2.25 b128 LDS reads per WMMA) or one output tile x eight row tiles (CP 1,
// STRIX_GLM_PF_WMMA_CP=1). Decode and every (row tile, output tile) WMMA chain are unchanged: same bits.
inline unsigned cols_per_wave() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_CP");
    return p && std::strcmp(p, "1") == 0 ? 1U : 2U;
}

// Routed MoE groups: a remainder of 65..127 rows joins the 128-row tiles (STRIX_GLM_PF_WMMA_TAIL128=0 keeps it on the
// 64-row tail kernel). Same arithmetic per (row, output): same bits.
inline unsigned tail128() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_TAIL128");
    return p && std::strcmp(p, "0") == 0 ? 0U : 1U;
}

// Routed MoE full tiles over S CTAs per (expert, output block): STRIX_GLM_PF_WMMA_SPLIT=1..8, default 2 (t8: MoE whole
// call -0.3..-1.3 % vs 1). Same bits.
inline unsigned split() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_SPLIT");
    const unsigned v = p ? static_cast<unsigned>(std::strtoul(p, nullptr, 10)) : 2U;
    return v >= 1U && v <= 8U ? v : 2U;
}

constexpr unsigned kRowsMax = 64;  // MT = 4 row tiles; the epilogue always works in 64-row halves
// Pair pitch 36 dwords (as exact_x): b128 reads stay aligned and rows rotate banks.
template<unsigned NR>
union alignas(16) StorageR {
    struct { std::uint32_t w[128][36], a[NR][36]; } stage;
    float total[kRowsMax][128];
};
using Storage = StorageR<64>;
using Storage128 = StorageR<128>;
static_assert(sizeof(Storage) == 32768 && sizeof(Storage128) == 36864);

__device__ __forceinline__ Half16 frag(const std::uint32_t* p) {
    const U4 lo = *reinterpret_cast<const U4*>(p), hi = *reinterpret_cast<const U4*>(p + 4);
    struct { U4 l, h; } v{lo, hi};
    return __builtin_bit_cast(Half16, v);
}

// Rows: input(r) = transformed FP16 row, output(r) = FP32 destination (exact_x interface). nr <= 16 * MT.
template<int Bits, unsigned MT, unsigned CP = 1, class Mat, class Rows, class Lds>
__device__ __forceinline__ void core(Mat m, Rows rows, unsigned nr, Lds& lds) {
    static_assert(MT == 1 || MT == 2 || MT == 4 || MT == 8);
    static_assert(CP == 1 || (CP == 2 && MT == 8));
    // CP 2: wave w owns output tiles 2 (w & 3), 2 (w & 3) + 1 and row tiles 4 (w >> 2) .. 4 (w >> 2) + 3;
    // acc[4 s + t] = (output tile 2 (w & 3) + s, row tile 4 (w >> 2) + t). CP 1: acc[t] = (output tile w, row tile t).
    constexpr unsigned NR = 16U * MT, Q = NR * 8U / 256U;  // 16-byte activation chunks per thread per stage
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid / 32U, c = lane & 15U;
    const unsigned col0 = blockIdx.x * 128U;
    const unsigned units = m.input / 16U;
    const d::LaneGeom<Bits> geo(lane);
    const auto* packed = reinterpret_cast<const unsigned char*>(d::uniform(m.trellis));
    // Chunk q of a stage: row q / 8, 16-byte column q % 8. Rows past nr read row index 0 (never published).
    const _Float16* src[Q > 0 ? Q : 1];
    unsigned qrow[Q > 0 ? Q : 1], qcol[Q > 0 ? Q : 1];
#pragma unroll
    for (unsigned j = 0; j < Q; ++j) {
        const unsigned q = tid + j * 256U;
        qrow[j] = q / 8U;
        qcol[j] = q % 8U;
        src[j] = rows.input(qrow[j] < nr ? qrow[j] : 0U);
    }
    // MT 1: 128 chunks, the upper half of the CTA stages nothing.
    const bool stager = Q > 0 || tid < NR * 8U;
    const unsigned q1 = tid, r1 = q1 / 8U, c1 = q1 % 8U;
    const _Float16* src1 = (Q == 0 && stager) ? rows.input(r1 < nr ? r1 : 0U) : nullptr;

    std::uint32_t raw[4][3];
    U4 xa[Q > 0 ? Q : 1];
    const auto fetch = [&](unsigned base, unsigned nk) {
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) if (kk < nk) {
            const auto* tile = packed + (std::size_t{base + kk} * (m.output / 16U) + col0 / 16U + wave) * Bits * 32U;
            raw[kk][0] = d::load_u32(tile, geo.oa);
            raw[kk][1] = d::load_u32(tile, geo.ob);
            raw[kk][2] = d::load_u32(tile, geo.oc);
        }
        if constexpr (Q > 0) {
#pragma unroll
            for (unsigned j = 0; j < Q; ++j) {
                xa[j] = U4{0, 0, 0, 0};
                if (qcol[j] < nk * 2U) xa[j] = *reinterpret_cast<const U4*>(src[j] + base * 16U + qcol[j] * 8U);
            }
        } else {
            xa[0] = U4{0, 0, 0, 0};
            if (stager && c1 < nk * 2U) xa[0] = *reinterpret_cast<const U4*>(src1 + base * 16U + c1 * 8U);
        }
    };
    Float8 acc[MT] = {};
    fetch(0, units < 4U ? units : 4U);
    for (unsigned base = 0; base < units; base += 4U) {
        const unsigned nk = units - base < 4U ? units - base : 4U;
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) if (kk < nk) {
            d::TileStates<Bits> st;
            st.high = __builtin_amdgcn_alignbit(raw[kk][0], raw[kk][1], geo.align);
            st.low = __builtin_amdgcn_alignbit(raw[kk][1], raw[kk][2], geo.align);
            st.tail = raw[kk][2] << (32U - geo.align);
            const unsigned o = wave * 16U + lane / 4U, p = kk * 8U + (lane & 3U);
            lds.stage.w[o][p] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<0>(), st.template state<1>()));
            lds.stage.w[o][p + 4U] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<2>(), st.template state<3>()));
            lds.stage.w[o + 8U][p] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<4>(), st.template state<5>()));
            lds.stage.w[o + 8U][p + 4U] = __builtin_bit_cast(std::uint32_t, d::mul1_pair(st.template state<6>(), st.template state<7>()));
        }
        if constexpr (Q > 0) {
#pragma unroll
            for (unsigned j = 0; j < Q; ++j) *reinterpret_cast<U4*>(&lds.stage.a[qrow[j]][qcol[j] * 4U]) = xa[j];
        } else if (stager) {
            *reinterpret_cast<U4*>(&lds.stage.a[r1][c1 * 4U]) = xa[0];
        }
        __syncthreads();
        if (base + 4U < units) {
            const unsigned next = base + 4U;
            fetch(next, units - next < 4U ? units - next : 4U);
        }
        asm volatile("" ::: "memory");
        // The WMMA accumulator add is not round-to-nearest (synthetic drift grew linearly with K when one chain ran
        // over all of K): chain only the stage's k-units from +0, then fold the stage into the total with v_add_f32.
        Float8 part[MT] = {};
        if constexpr (CP == 1) {
            for (unsigned kk = 0; kk < nk; ++kk) {
                const Half16 b = frag(&lds.stage.w[wave * 16U + c][kk * 8U]);
#pragma unroll
                for (unsigned t = 0; t < MT; ++t)
                    part[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag(&lds.stage.a[t * 16U + c][kk * 8U]), b, part[t]);
            }
        } else {
            const unsigned ot = 2U * (wave & 3U), rt = 4U * (wave >> 2U);
            for (unsigned kk = 0; kk < nk; ++kk) {
                const Half16 b0 = frag(&lds.stage.w[ot * 16U + c][kk * 8U]), b1 = frag(&lds.stage.w[(ot + 1U) * 16U + c][kk * 8U]);
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t) {
                    const Half16 a = frag(&lds.stage.a[(rt + t) * 16U + c][kk * 8U]);
                    part[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, part[t]);
                    part[4U + t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, part[4U + t]);
                }
            }
        }
#pragma unroll
        for (unsigned t = 0; t < MT; ++t) acc[t] = acc[t] + part[t];
        __syncthreads();
    }
    // acc[t][j] -> row index (16 t + 2 j + lane / 16), output wave * 16 + lane % 16; 64-row halves through lds.total.
    constexpr unsigned TH = MT < 4U ? MT : 4U;
#pragma unroll
    for (unsigned h = 0; h < MT / TH; ++h) {
        if constexpr (CP == 1) {
#pragma unroll
            for (unsigned t = 0; t < TH; ++t)
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) lds.total[t * 16U + 2U * j + lane / 16U][wave * 16U + c] = acc[h * TH + t][j];
        } else if ((wave >> 2U) == h) {  // the waves holding this half's row tiles
#pragma unroll
            for (unsigned s = 0; s < 2U; ++s)
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t)
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j)
                        lds.total[t * 16U + 2U * j + lane / 16U][(2U * (wave & 3U) + s) * 16U + c] = acc[4U * s + t][j];
        }
        __syncthreads();
        const unsigned h0 = h * kRowsMax, h1 = nr < h0 + kRowsMax ? nr : h0 + kRowsMax;
        for (unsigned r = h0 + wave; r < h1; r += 8U) {
            float v[4];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) v[j] = lds.total[r - h0][lane + 32U * j];
            d::hadamard128(v, lane);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const unsigned o = col0 + lane + 32U * j;
                rows.output(r)[o] = v[j] * d::half_bits(m.svh[o]);
            }
        }
        __syncthreads();
    }
}

// A row span [0, n) in tiles of 64 (or 128) rows, the last one on the smallest MT covering it (uniform per CTA).
template<int Bits, unsigned CP = 1, class Mat, class RowsAt, class Lds>
__device__ __forceinline__ void span(Mat m, RowsAt at, unsigned n, Lds& lds) {
    constexpr unsigned Tile = sizeof(lds.stage.a) / sizeof(lds.stage.a[0]);
    for (unsigned r = 0; r < n; r += Tile) {
        const unsigned k = n - r < Tile ? n - r : Tile;
        if constexpr (Tile == 128U) {
            if (k > 64U) { core<Bits, 8, CP>(m, at(r, k), k, lds); continue; }
        }
        if (k > 32U) core<Bits, 4>(m, at(r, k), k, lds);
        else if (k > 16U) core<Bits, 2>(m, at(r, k), k, lds);
        else core<Bits, 1>(m, at(r, k), k, lds);
    }
}
} // namespace strix::glm::pf::wmma_x
