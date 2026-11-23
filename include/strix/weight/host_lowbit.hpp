#pragma once
// Low-bit resident row storage for dense linears (FAST weight formats of the shared dense rows family).
// One vendor-free storage contract; the device decoders (core/weight_decode.hpp Chunk) and the host encoder below
// read the same layout. Row-major [N, K] matrix, K % 128 == 0, each row a run of 128-element superblocks:
//   q4_g64 (68 B):  bytes [0, 64)  codes, element j in byte j / 2 (low nibble = even j), signed 4-bit [-8, 7]
//                   bytes [64, 68) FP16 scale of elements [0, 64) then of [64, 128)
//   q6_g64 (100 B): bytes [0, 64)  low 4 bits of every code (same addressing as q4_g64)
//                   bytes [64, 96) high 2 bits, element j in byte 64 + j / 4 at bit 2 * (j % 4); signed 6-bit [-32, 31]
//                   bytes [96, 100) FP16 scales as above
// Weight = float(fp16 scale) * float(code): one FP32 product, exact (11-bit by <= 6-bit significands).
// The code planes keep every 16-element load of the rows sweep 8-byte (codes) and 4-byte (high bits) aligned.
// Encoding: symmetric round-to-nearest-even with one FP16 scale per 64 elements; the scale is chosen per group from
// kClipSteps clip ratios of max(positive / qmax, negative / -qmin) by the smallest squared reconstruction error, and
// the codes are computed with the stored (FP16) scale.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_LOWBIT_HD __host__ __device__
#else
#define STRIX_LOWBIT_HD
#endif

namespace strix::weight {

template <unsigned Bits>
struct LowBitStorage {
    static_assert(Bits == 4U || Bits == 6U);
    static constexpr std::uint32_t elements = 128, group = 64, groups = elements / group;
    static constexpr std::uint32_t low_bytes = elements / 2U;                       // 4-bit plane
    static constexpr std::uint32_t high_bytes = Bits == 6U ? elements / 4U : 0U;    // 2-bit plane
    static constexpr std::uint32_t scale_offset = low_bytes + high_bytes;
    static constexpr std::uint32_t bytes = scale_offset + groups * 2U;
    static constexpr int qmax = (1 << (Bits - 1U)) - 1, qmin = -(1 << (Bits - 1U));
    static constexpr std::uint64_t row_bytes(std::uint64_t width) noexcept { return width / elements * bytes; }
    STRIX_LOWBIT_HD static std::uint16_t scale_bits(const std::uint8_t* block, unsigned group_index) noexcept {
        const auto* p = block + scale_offset + 2U * group_index;
        return static_cast<std::uint16_t>(p[0] | (std::uint16_t(p[1]) << 8U));
    }
    // Signed code of element j (0..127) of a superblock.
    STRIX_LOWBIT_HD static int code(const std::uint8_t* block, unsigned j) noexcept {
        const unsigned low = (block[j / 2U] >> ((j % 2U) * 4U)) & 15U;
        if constexpr (Bits == 4U) {
            return low < 8U ? int(low) : int(low) - 16;
        } else {
            const unsigned high = (block[low_bytes + j / 4U] >> ((j % 4U) * 2U)) & 3U;
            const unsigned v = low | (high << 4U);
            return v < 32U ? int(v) : int(v) - 64;
        }
    }
};
using Q4G64Storage = LowBitStorage<4>;
using Q6G64Storage = LowBitStorage<6>;

namespace lowbit {

inline constexpr int kClipSteps = 21;
inline constexpr float clip_ratio(int i) { return 1.0F - 0.02F * static_cast<float>(i); }

// IEEE binary16 round to nearest even, and its exact widening (finite inputs; the encoder rejects the rest).
inline std::uint16_t f32_to_f16_bits(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16U) & 0x8000U, e = (x >> 23U) & 0xffU, m = x & 0x7fffffU;
    if (e == 0xffU) return static_cast<std::uint16_t>(sign | 0x7c00U | (m ? 0x200U : 0U));
    const int exp = static_cast<int>(e) - 112;
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t full = m | 0x800000U, shift = static_cast<std::uint32_t>(14 - exp);
        std::uint32_t half = full >> shift;
        const std::uint32_t rem = full & ((1U << shift) - 1U), mid = 1U << (shift - 1U);
        if (rem > mid || (rem == mid && (half & 1U))) ++half;
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10U) | (m >> 13U);
    const std::uint32_t rem = m & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (half & 1U))) ++half;
    return static_cast<std::uint16_t>(sign | half);
}
inline float f16_bits_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (std::uint32_t(h) & 0x8000U) << 16U, e = (h >> 10U) & 0x1fU, m = h & 0x3ffU;
    float v;
    if (e == 0U) {
        v = static_cast<float>(m) * 0x1p-24F;
    } else if (e == 31U) {
        v = m ? NAN : INFINITY;
    } else {
        const std::uint32_t bits = ((e + 112U) << 23U) | (m << 13U);
        std::memcpy(&v, &bits, 4);
    }
    return sign ? -v : v;
}

template <class Storage>
inline int quantise(float w, float inv) {
    const float r = std::nearbyint(w * inv);  // default rounding mode: to nearest even
    return r > float(Storage::qmax) ? Storage::qmax : r < float(Storage::qmin) ? Storage::qmin : int(r);
}

// One 64-element group: returns the FP16 scale bits and the codes. Finite inputs only.
template <class Storage>
inline std::uint16_t encode_group(const float* w, int* q) {
    float pos = 0.0F, neg = 0.0F;
    for (unsigned j = 0; j < Storage::group; ++j) {
        pos = w[j] > pos ? w[j] : pos;
        neg = -w[j] > neg ? -w[j] : neg;
    }
    const float a = pos / float(Storage::qmax), b = neg / float(-Storage::qmin), base = a > b ? a : b;
    std::uint16_t best_bits = 0;
    if (base > 0.0F) {
        float best = -1.0F;
        for (int i = 0; i < kClipSteps; ++i) {
            const float raw = clip_ratio(i) * base;
            const auto bits = f32_to_f16_bits(raw > 0x1p-24F ? raw : 0x1p-24F);
            const float s = f16_bits_to_f32(bits), inv = 1.0F / s;
            if (!std::isfinite(s) || !std::isfinite(inv)) continue;
            float err = 0.0F;
            for (unsigned j = 0; j < Storage::group; ++j) {
                const float d = w[j] - s * static_cast<float>(quantise<Storage>(w[j], inv));
                err += d * d;
            }
            if (best < 0.0F || err < best) { best = err; best_bits = bits; }
        }
    }
    if (best_bits == 0U) {
        for (unsigned j = 0; j < Storage::group; ++j) q[j] = 0;
        return 0;
    }
    const float inv = 1.0F / f16_bits_to_f32(best_bits);
    for (unsigned j = 0; j < Storage::group; ++j) q[j] = quantise<Storage>(w[j], inv);
    return best_bits;
}

// One superblock of Storage::elements finite floats -> Storage::bytes. Returns false on a non-finite input or a
// scale that does not fit FP16.
template <class Storage>
inline bool encode_block(const float* w, std::uint8_t* out) {
    for (unsigned j = 0; j < Storage::elements; ++j)
        if (!std::isfinite(w[j])) return false;
    std::memset(out, 0, Storage::bytes);
    int q[Storage::elements];
    for (unsigned g = 0; g < Storage::groups; ++g) {
        const auto bits = encode_group<Storage>(w + g * Storage::group, q + g * Storage::group);
        if ((bits & 0x7c00U) == 0x7c00U) return false;
        out[Storage::scale_offset + 2U * g] = static_cast<std::uint8_t>(bits & 0xffU);
        out[Storage::scale_offset + 2U * g + 1U] = static_cast<std::uint8_t>(bits >> 8U);
    }
    for (unsigned j = 0; j < Storage::elements; ++j) {
        const auto v = static_cast<unsigned>(q[j]);
        out[j / 2U] = static_cast<std::uint8_t>(out[j / 2U] | ((v & 15U) << ((j % 2U) * 4U)));
        if constexpr (Storage::high_bytes != 0U)
            out[Storage::low_bytes + j / 4U] =
                static_cast<std::uint8_t>(out[Storage::low_bytes + j / 4U] | (((v >> 4U) & 3U) << ((j % 4U) * 2U)));
    }
    return true;
}

// Reference decode of element j of a superblock (host tests, gate tooling).
template <class Storage>
inline float decode(const std::uint8_t* block, unsigned j) {
    return f16_bits_to_f32(Storage::scale_bits(block, j / Storage::group)) * static_cast<float>(Storage::code(block, j));
}

}  // namespace lowbit
}  // namespace strix::weight

#undef STRIX_LOWBIT_HD
