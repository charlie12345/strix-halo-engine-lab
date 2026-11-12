#pragma once
// GLM prefill FAST identity glm-pf-kda-wmma-v1: the KDA plain-weight projections of a chunk (f_a / g_a / b of x, then
// f_b and g_b) on the shared dense FP16 WMMA core (core/f16_dense_tall.hpp) instead of the exact FP32 fma chains of
// dense_rows. Opt-in: STRIX_GLM_PF_KDA_WMMA=1 (unset or 0 = exact, the same bits as before). The router and the
// indexer projections stay on the exact kernels (they select experts / tokens).
// Adapter only: column-tile mapping of up to kDenseGroupMax projections of the same x, and the launches.
#include "core/f16_dense_tall.hpp"
#include "glm/prefill/glm_prefill.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::glm::pf::kda_wmma {

inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_KDA_WMMA");
    if (p && std::strcmp(p, "0") != 0 && std::strcmp(p, "1") != 0)
        throw std::invalid_argument("STRIX_GLM_PF_KDA_WMMA must be 0 or 1");
    return p && std::strcmp(p, "1") == 0;
}
inline std::atomic<std::uint64_t> calls{0};  // launches that ran on the WMMA core (identity reporting)

struct Segment {
    const _Float16* w;  // [n][in] row-major
    float* y;           // [rows][n]
    unsigned n;         // multiple of 16
};
struct Args {
    const float* x;
    unsigned in, rows, tile0, count;
    Segment seg[kDenseGroupMax];
};
struct Columns {
    const Args& a;
    __device__ __forceinline__ core::f16_dense_tall::ColumnTile tile(unsigned i) const {
        unsigned t = a.tile0 + i;
#pragma unroll
        for (unsigned s = 0; s < kDenseGroupMax; ++s) {
            const unsigned tiles = s < a.count ? a.seg[s].n / 16U : 0U;
            if (t < tiles) return {a.seg[s].w + static_cast<std::size_t>(t) * 16U * a.in, a.seg[s].y + t * 16U, a.seg[s].n, true};
            t -= tiles;
        }
        return {a.seg[0].w, a.seg[0].y, 0U, false};
    }
};

template <unsigned TC> __global__ __launch_bounds__(256) void dense_kernel(Args a) {
    __shared__ core::f16_dense_tall::Storage lds;
    core::f16_dense_tall::core<TC>(a.x, a.in, a.rows, Columns{a}, lds);
}

// True if the shapes fit the core (FP16 weights, in % 128 == 0, every n a positive multiple of 16).
inline bool supported(dev::WType wt, unsigned in, const DenseProj* p, unsigned count, unsigned rows) {
    if (wt != dev::WType::F16 || count == 0 || count > kDenseGroupMax || rows == 0 || in == 0 || in % 128U) return false;
    if (static_cast<std::uint64_t>(rows) * in * 4U >= (std::uint64_t{1} << 32U)) return false;
    for (unsigned i = 0; i < count; ++i)
        if (p[i].n == 0 || p[i].n % 16U || !p[i].w || !p[i].y) return false;
    return true;
}

// y_i[r][o] for every projection i; the caller checked supported(). 16-tile CTA columns first, the remainder in
// 4-tile columns.
inline void dense_group(const float* x, unsigned in, const DenseProj* p, unsigned count, unsigned rows, hipStream_t s) {
    Args a{x, in, rows, 0U, count, {}};
    unsigned tiles = 0;
    for (unsigned i = 0; i < count; ++i) {
        a.seg[i] = {static_cast<const _Float16*>(p[i].w) + static_cast<std::size_t>(p[i].o0) * in, p[i].y, p[i].n};
        tiles += p[i].n / 16U;
    }
    const unsigned gx = (rows + core::f16_dense_tall::kRows - 1U) / core::f16_dense_tall::kRows;
    const unsigned full = tiles / 16U, rest = tiles % 16U;
    if (full) dense_kernel<16><<<dim3(gx, full), 256, 0, s>>>(a);
    if (rest) {
        a.tile0 = full * 16U;
        dense_kernel<4><<<dim3(gx, (rest + 3U) / 4U), 256, 0, s>>>(a);
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("glm pf kda wmma dense: ") + hipGetErrorString(e));
    calls.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace strix::glm::pf::kda_wmma
