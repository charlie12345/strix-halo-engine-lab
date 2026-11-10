#include "moe_core/grouped_integer_decode.hpp"
#include "core/integer_operand.hpp"
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
enum class Format { iq2_xxs, q2_k, q8_0, mxfp4 };
using V4 = std::array<int, 4>;
using V8 = std::array<int, 8>;
template<Format> struct Raw;
template<> struct Raw<Format::iq2_xxs> { unsigned short d; unsigned long long aux; };
template<> struct Raw<Format::q2_k> { unsigned q[8]; unsigned short scales; unsigned dm, shift; };
template<> struct Raw<Format::q8_0> { unsigned short d; unsigned q[8]; };
template<> struct Raw<Format::mxfp4> { unsigned e, q[4]; };
struct Group { V4 b0, b1; float scale, d, dmin; int m0, m1; unsigned dm_bits, sc_bits; };
struct Traits {
    using WeightFormat = Format;
    using Int4 = V4; using Int8 = V8; using OperandGroup = Group;
    template<Format F> using RawGroup = Raw<F>;
    static constexpr int magic_bits = 0x4b400000;
    static constexpr unsigned iq2_xxs_block_bytes = 66, q2_k_block_bytes = 84, q8_0_block_bytes = 34, mxfp4_block_bytes = 17;
    using Operations = strix::moe_core::GroupedIntegerDecodeOps<Traits>;
};
using Ops = Traits::Operations;
using namespace strix::core::integer_math;
const unsigned long long grid[256] = {
#include "strix/ds41_iq2_xxs_grid.inc"
};
void expect(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
int code(const Group& g, unsigned j) {
    const auto& v = j < 16 ? g.b0 : g.b1;
    return int(std::int8_t(unsigned(v[(j % 16) / 4]) >> (8 * (j % 4))));
}
void iq_cases() {
    // Unaligned GGUF blocks, every grid entry, every parity sign and subscale.
    std::array<unsigned char, 134> data{};
    unsigned char* row = data.data() + 1;
    for (unsigned gi = 0; gi < 256; ++gi)
        for (unsigned sign = 0; sign < 128; ++sign) {
            for (unsigned g = 0; g < 16; ++g) {
                unsigned char* block = row + (g / 8) * 66;
                const unsigned short d = 0x3400; std::memcpy(block, &d, 2);
                const unsigned a0 = gi * 0x01010101U;
                const unsigned a1 = sign | (sign << 7) | (sign << 14) | (sign << 21) | ((g % 16) << 28);
                std::memcpy(block + 2 + (g % 8) * 8, &a0, 4);
                std::memcpy(block + 6 + (g % 8) * 8, &a1, 4);
                using I = strix::core::IntegerOperand<Traits, Format::iq2_xxs>;
                const auto w = I::decode(row, g, grid);
                const unsigned parity = std::popcount(sign) & 1U, signs = sign | (parity << 7);
                for (unsigned j = 0; j < 32; ++j) {
                    const int mag = int((grid[gi] >> (8 * (j % 8))) & 255U);
                    const int want = (signs & (1U << (j % 8))) ? -mag : mag;
                    expect(code(w, j) == want, "IQ2 grid/sign/address decode");
                    expect(signed_nibble(nibble(iq4(want))) == iq4(want), "IQ2 signed nibble round trip");
                    const int q = iq4(want); expect(q >= -7 && q <= 7 && q != 0, "IQ2 IU4 range");
                }
                expect(w.scale == 0.25f * (0.5f + float(g)) * 0.25f, "IQ2 FP32 scale");
            }
        }
}
void q2_cases() {
    std::array<unsigned char, 170> data{}; unsigned char* row = data.data() + 1;
    std::mt19937 rng(31);
    for (unsigned sc = 0; sc < 256; ++sc) {
        for (unsigned b = 0; b < 2; ++b) {
            auto* p = row + 84 * b;
            for (unsigned j = 0; j < 16; ++j) p[j] = static_cast<unsigned char>((sc + 37 * j) & 255U);
            for (unsigned j = 16; j < 80; ++j) p[j] = static_cast<unsigned char>(rng());
            const unsigned dm = 0x30003400; std::memcpy(p + 80, &dm, 4);
        }
        for (unsigned g = 0; g < 16; ++g) {
            const auto raw = Ops::load_raw<Format::q2_k>(row, g);
            const auto w = Ops::decode(raw, nullptr), lean = Ops::decode_lean(raw, nullptr);
            const auto* block = row + (g / 8) * 84;
            const unsigned gg = g % 8, half = gg / 4;
            int a = 0, direct = 0, sums[2] = {}, dots[2] = {};
            for (unsigned j = 0; j < 32; ++j) {
                const unsigned s = block[2 * gg + j / 16];
                const int q = int((block[16 + half * 32 + j] >> (2 * (gg % 4))) & 3U);
                const int av = int(rng() % 255) - 127;
                expect(code(w, j) == q * int(s & 15U) && code(lean, j) == code(w, j), "Q2_K decode factors");
                sums[j / 16] += av; dots[j / 16] += q * av;
                a += code(w, j) * av; direct += q * int(s & 15U) * av;
                expect(signed_nibble(nibble(q)) == q, "Q2_K nibble carries no scale factor");
            }
            const unsigned s0 = block[2 * gg], s1 = block[2 * gg + 1];
            expect(a == direct && a == int(s0 & 15U) * dots[0] + int(s1 & 15U) * dots[1], "Q2_K half-dot factor fold");
            const int m = int(s0 >> 4) * sums[0] + int(s1 >> 4) * sums[1];
            expect(w.m0 * sums[0] + w.m1 * sums[1] == m, "Q2_K minimum half-row sums");
            expect(affine(a, m, w.d, w.dmin, 0.125f) == (0.25f * float(a) - 0.125f * float(m)) * 0.125f, "Q2_K RN32 correction order");
        }
    }
}
void q8_cases() {
    std::array<unsigned char, 103> row{};
    for (unsigned g = 0; g < 3; ++g) {
        const unsigned short d = 0x3800; std::memcpy(row.data() + 34 * g, &d, 2);
        for (unsigned j = 0; j < 32; ++j) row[34 * g + 2 + j] = static_cast<unsigned char>(int(j) * 7 - 127);
        const auto w = Ops::decode(Ops::load_raw<Format::q8_0>(row.data(), g), nullptr);
        for (unsigned j = 0; j < 32; ++j) expect(code(w, j) == int(std::int8_t(int(j) * 7 - 127)), "Q8_0 block stride/codes");
        expect(w.scale == 0.5f, "Q8_0 scale");
        const auto dot = Ops::group_dot(w.b0, w.b0, w.b1, w.b1);
        int want = 0; for (unsigned j = 0; j < 32; ++j) want += code(w, j) * code(w, j);
        expect(Ops::magic_float(dot[0]) == float(want), "IU8 magic-offset conversion bounds");
    }
}
void fragments() {
    using V2 = std::array<int, 2>;
    std::mt19937 rng(102);
    for (unsigned sample = 0; sample < 4096; ++sample) {
        V2 a{}, b{}; int expected = 0;
        for (unsigned j = 0; j < 16; ++j) {
            const int av = int(rng() % 15) - 7, bv = int(rng() % 15) - 7;
            a[j / 8] |= int(nibble(av) << (4 * (j % 8)));
            b[j / 8] |= int(nibble(bv) << (4 * (j % 8)));
            expected += av * bv;
        }
        const auto result = strix::core::integer_wmma<4>(a, b, V8{});
        expect(result[0] == expected, "IU4 fragment packing/sign semantics (CPU row shim)");
    }
}
void reductions() {
    std::mt19937 rng(88);
    for (unsigned groups : {1U, 2U, 7U, 8U, 31U, 32U, 33U, 64U, 72U, 144U, 160U, 256U}) {
        float cls[32] = {};
        for (unsigned g = 0; g < groups; ++g) cls[g % 32] += std::ldexp(float(int(rng() % 65535) - 32767), int(rng() % 30) - 15);
        float ref[32]; std::memcpy(ref, cls, sizeof(cls));
        for (unsigned stride = 16; stride; stride /= 2) {
            float next[32]; for (unsigned c = 0; c < 32; ++c) next[c] = ref[c] + ref[c ^ stride];
            std::memcpy(ref, next, sizeof(ref));
        }
        float q[8];
        for (unsigned c = 0; c < 8; ++c) q[c] = (cls[c] + cls[c + 16]) + (cls[c + 8] + cls[c + 24]);
        const float core = ((q[0] + q[4]) + (q[2] + q[6])) + ((q[1] + q[5]) + (q[3] + q[7]));
        expect(std::bit_cast<unsigned>(core) == std::bit_cast<unsigned>(ref[0]), "class-wave butterfly order");
        expect(std::bit_cast<unsigned>(tree32(cls)) == std::bit_cast<unsigned>(ref[0]), "portable Contract reduction");
    }
    for (int amax = 0; amax <= 128; ++amax)
        for (int value = -amax; value <= amax; ++value) {
            const int q = a4_code(value, a4_step(amax));
            expect(q >= -7 && q <= 7 && signed_nibble(nibble(q)) == q, "A4 quant/range/RNE");
        }
    expect(a4_code(1, 2.0f) == 0 && a4_code(3, 2.0f) == 2 && a4_code(-3, 2.0f) == -2, "A4 ties to even");
}
}
int main() try {
    iq_cases(); q2_cases(); q8_cases(); fragments(); reductions();
    std::cout << "PASS DS41 IntegerOperand host: GGUF decode, IU4 mapping, affine correction, class tree, A4 RNE\n";
    return 0;
} catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
