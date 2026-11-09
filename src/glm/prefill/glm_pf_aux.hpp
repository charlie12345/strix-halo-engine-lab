#pragma once
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <string>
#if defined(__HIPCC__)
#include "glm/moe/glm_trellis_device.hpp"
#endif

namespace strix::glm::pf::aux {
// Draft organization policies. Native qualification is required before changing
// the production default. Read per call so a gate can compare arms in one process.
// Default-on form for qualified aux levers: unset or "1" = on, exactly "0" = off.
inline bool enabled_default_on(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::string(value) == "1") return true;
    if (std::string(value) == "0") return false;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
inline bool enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
constexpr bool input_admitted(unsigned output) { return output != 0 && output <= 512U && output % 128U == 0; }
constexpr unsigned dense_columns(const unsigned* widths, unsigned count) {
    unsigned n = 0;
    for (unsigned p = 0; p < count; ++p) if (widths[p] > n) n = widths[p];
    return n;
}
// Two independent lanes of the original 256-lane RMSNorm. Folding its first
// tree level before the 128-lane tree preserves every chain and every add.
template<class Fma>
#if defined(__HIPCC__)
__host__ __device__
#endif
inline auto norm_pair(const float* x, unsigned width, unsigned lane, Fma fma) {
    float lo = 0.0F, hi = 0.0F;
    for (unsigned i = lane; i < width; i += 256U) lo = fma(x[i], x[i], lo);
    for (unsigned i = lane + 128U; i < width; i += 256U) hi = fma(x[i], x[i], hi);
    return lo + hi;
}
}  // namespace strix::glm::pf::aux

#if defined(__HIPCC__)
namespace strix::glm::pf::aux {
// Shared X/WMMA epilogue contract; custom policies receive an entire H128
// block per wave, after accumulation but before the final svh/store boundary.
struct Store {
    template<class Rows, class Mat>
    __device__ __forceinline__ void operator()(Rows rows, Mat m, unsigned r, unsigned col0,
                                               float (&v)[4], unsigned lane) const {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned o = col0 + lane + 32U * j;
            rows.output(r)[o] = v[j] * strix::glm::moe::dev::half_bits(m.svh[o]);
        }
    }
};
struct StagedInput {
    static constexpr bool kFused = false;
    __device__ StagedInput offset(unsigned) const { return {}; }
};
// The original H128 graph, including both RN16 boundaries. Two consecutive
// 64-wide WMMA stages consume one cached H128 input tile in the wider A image.
// Limit admission to narrow outputs: wider matrices multiply transform work
// across too many independent output CTAs.
struct H128Input {
    static constexpr bool kFused = true;
    const float* x;
    std::size_t stride;
    unsigned row0;
    __device__ H128Input offset(unsigned r) const { return {x, stride, row0 + r}; }
    template<unsigned NR, class Mat, class Lds>
    __device__ void stage(Mat m, unsigned nr, unsigned base, Lds& lds) const {
        const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U;
        if (base & 4U) return;  // retain the other half from the preceding stage
        const unsigned begin = base * 16U;
        for (unsigned r = wave; r < NR; r += 8U) {
            float v[4];
            strix::glm::moe::dev::transform_values(x + std::size_t{row0 + (r < nr ? r : 0U)} * stride,
                                                 m.suh, begin, lane, v);
            auto* dst = reinterpret_cast<_Float16*>(&lds.stage.a[r][0]);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const unsigned c = lane + j * 32U;
                dst[c] = static_cast<_Float16>(v[j]);
            }
        }
    }
};
}  // namespace strix::glm::pf::aux
#endif
