#include "core/stored_i4_plan.hpp"
#include "core/iu4.hpp"
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace strix::core;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
int main() {
    CHECK(plan_stored_i4(100, 20, 30, 2, {20, 20, 20, 20, 20, 20}).layers == 5);
    CHECK(plan_stored_i4(31, 0, 30, 2, {100}).layers == 0); // final fit cannot bypass conversion peak
    CHECK(plan_stored_i4(32, 0, 30, 2, {20}).layers == 1);
    CHECK(plan_stored_i4(100, 100, 30, 2, {20}).layers == 0);
    CHECK(plan_stored_i4(100, 101, 30, 2, {20}).layers == 0);
    CHECK(plan_stored_i4(100, 0, 30, 2, {20}, 0).layers == 0);
    CHECK(plan_stored_i4(100, 0, 30, 2, {20}, 1).layers == 1);
    for (std::size_t free = 0; free < 100; ++free)
        for (std::size_t reserve = 0; reserve <= 100; reserve += 10)
            for (std::size_t pk = 1; pk <= 15; pk += 2) {
                const std::vector<std::size_t> old = {2, 9, 3, 8, 4};
                std::size_t expected = 0;
                long long occupied = 0;
                const auto usable = free > reserve ? free - reserve : 0;
                for (std::size_t n = 1; n <= old.size(); ++n) {
                    const long long peak = occupied + static_cast<long long>(pk + 1);
                    if (peak > static_cast<long long>(usable)) break;
                    ++expected;
                    occupied += static_cast<long long>(pk + 1) - static_cast<long long>(old[n - 1]);
                }
                const auto p = plan_stored_i4(free, reserve, pk, 1, old);
                CHECK(p.layers == expected);
                CHECK(static_cast<long long>(p.free_after) == static_cast<long long>(usable) - occupied);
            }
    bool rejected = false;
    try { (void)plan_stored_i4(100, 0, std::numeric_limits<std::size_t>::max(), 1, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { (void)plan_stored_i4(100, 0, 1, 0, {std::numeric_limits<std::size_t>::max()}); }
    catch (const std::overflow_error&) { rejected = true; }
    CHECK(rejected);
    for (int outputs : {16, 64, 128}) for (int inputs : {64, 128, 256}) {
        std::vector<unsigned> touched(std::size_t(inputs) * outputs, 0);
        for (int k = 0; k < inputs; ++k) for (int n = 0; n < outputs; ++n) {
            const auto at = iu4_weight_byte(k, n, outputs);
            CHECK(at < touched.size() / 2);
            CHECK(++touched[2 * at + (k & 1)] == 1);
        }
        for (auto count : touched) CHECK(count == 1);
    }
    // A wave reads one consecutive 512-byte stage (32 lanes x 16 bytes).
    for (int k = 0; k < 256; k += 16) for (int tile = 0; tile < 8; tile += 4)
        for (int lane = 0; lane < 32; ++lane)
            CHECK(iu4_weight_byte(k, (tile + lane / 8) * 16 + 2 * (lane % 8), 128) ==
                  iu4_weight_byte(k, tile * 16, 128) + std::size_t(lane) * 16);
    for (int low = -8; low <= 7; ++low) for (int high = -8; high <= 7; ++high) {
        std::uint8_t byte = 0;
        iu4_put(byte, 0, low); iu4_put(byte, 1, high);
        CHECK(iu4_unpack(byte, 0) == low && iu4_unpack(byte, 1) == high);
    }
    std::fesetround(FE_DOWNWARD);
    CHECK(iu4_rne(0.5f) == 0 && iu4_rne(1.5f) == 2 && iu4_rne(2.5f) == 2);
    CHECK(iu4_rne(-0.5f) == 0 && iu4_rne(-1.5f) == -2 && iu4_rne(-2.5f) == -2);
    std::fesetround(FE_TONEAREST);
    constexpr int K = 128, N = 32;
    std::vector<float> w(K * N, 0);
    auto zero = quantise_iu4_weights(w.data(), K, N);
    for (auto scale : zero.scales) CHECK(scale == 1);
    for (auto byte : zero.packed) CHECK(byte == 0);
    // Exactly representable scales isolate packing and K64 group indexing from FP16 rounding.
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n)
        w[k * N + n] = float(k % 15 - 7) * (k < 64 ? 0.5f : 2.0f);
    auto q = quantise_iu4_weights(w.data(), K, N);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
        CHECK(q.scales[n * 2 + k / 64] == (k < 64 ? 0.5f : 2.0f));
        CHECK(iu4_unpack(q.packed[iu4_weight_byte(k, n, N)], k & 1) == k % 15 - 7);
    }
    w[0] = std::numeric_limits<float>::infinity();
    rejected = false;
    try { (void)quantise_iu4_weights(w.data(), K, N); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    std::puts("stored-i4 host: PASS (budget peaks/reserve/overflow, layout bijection, streaming runs, signed nibbles, RNE, K64 packs)");
}
