// CPU-only exhaustive E1 gate. Requires HIP headers/compiler, never a device.
#include "moe_core/exl3_trellis.hpp"

#include <cstdint>
#include <cstdio>

template <int Bits> bool check_decode() {
    using namespace strix::moe_core;
    for (std::uint32_t state = 0; state < 65536U; ++state) {
        std::uint32_t st[8];
        Half2 packed[4], scalar[4], high[4];
        std::uint32_t high_st[8];
        for (int i = 0; i < 8; ++i) {
            // Every state visits every lane; partners differ and wrap at 65535.
            st[i] = (state + static_cast<std::uint32_t>(i) * 8191U) & 65535U;
            high_st[i] = st[i] | 0xa5a50000U;
        }
        Mul1PairDecode<Bits>::decode8(st, packed);
        ScalarDecode<Bits>::decode8(st, scalar);
        Mul1PairDecode<Bits>::decode8(high_st, high);
        for (int i = 0; i < 8; ++i) {
            const auto expected = __builtin_bit_cast(std::uint16_t,
                static_cast<_Float16>(Exl3Trellis<Bits>::codebook_value(st[i])));
            const auto got = __builtin_bit_cast(std::uint16_t, packed[i / 2][i % 2]);
            if (got != expected || __builtin_bit_cast(std::uint16_t, scalar[i / 2][i % 2]) != expected ||
                __builtin_bit_cast(std::uint16_t, high[i / 2][i % 2]) != expected ||
                hash16<0x83dcd12dU>(high_st[i]) != st[i] * 0x83dcd12dU) {
                std::fprintf(stderr, "MISMATCH K%d state=%u lane=%d got=%04x expected=%04x; FAST-only\n",
                             Bits, st[i], i, static_cast<unsigned>(got), static_cast<unsigned>(expected));
                return false;
            }
        }
    }
    std::printf("EXACT K%d: 65536 states in all eight lanes (including high-bit hash emulation)\n", Bits);
    return true;
}

int main() {
    return check_decode<3>() && check_decode<4>() && check_decode<5>() && check_decode<6>() ? 0 : 1;
}
