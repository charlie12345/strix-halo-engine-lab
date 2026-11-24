// Device identity of grouped_rows::combine_image (no model): FP32 down sums stored through
// core::f16_tile_gemm::Bf16NanExactStore and combined from that BF16 image give the output bits of grouped_rows::combine
// over the FP32 sums, for quiet and signalling NaNs of both signs and payloads in every slot, +-Inf, Inf x 0 and
// Inf - Inf between slots, two NaN terms in one sum, NaN weights on NaN sums (the product of two NaNs), next to finite
// columns in the same wave. A launch without a NaN
// sum leaves nan_seen at 0 and never writes the FP32 rows; a raised nan_seen with stale FP32 rows changes nothing.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/f16_tile_gemm.hpp"
#include "moe_core/grouped_rows.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace grouped = strix::moe_core::grouped_rows;
namespace gemm = strix::core::f16_tile_gemm;
namespace sf = strix::weight::scalar_float;

constexpr unsigned kRows = 37, kTopK = 8, kWidth = 300, kStride = 304, kOutStride = 301;
constexpr unsigned kAssignments = kRows * kTopK;
constexpr float kGuard = -2468.0F;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL combine-image-hip %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* upload(const std::vector<T>& host) {
    T* d = nullptr;
    ok(hipMalloc(&d, host.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> host(n);
    ok(hipMemcpy(host.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return host;
}

// The down store of one tile, element by element: what the tile epilogue does with each completed sum.
__global__ void store_kernel(const float* sums, gemm::Bf16NanExactStore store, unsigned rows, unsigned width) {
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    for (unsigned row = blockIdx.y; row < rows; row += gridDim.y)
        store.put(0U, row, column, sums[std::size_t{row} * store.y_row + column]);
}

struct Case {
    const char* name;
    bool nans;          // the sums hold NaN / Inf patterns
    bool stale_flag;    // nan_seen already raised and the FP32 rows hold stale values before the launch
};

bool run(const Case& c, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0F, 3.0F);
    std::vector<float> sums(std::size_t{kAssignments} * kStride, 0.0F);
    for (auto& v : sums) v = dist(rng);
    std::vector<float> weights(kAssignments);
    for (auto& w : weights) w = dist(rng) * 0.25F;
    std::vector<std::uint32_t> combine_rows(kAssignments);
    for (unsigned i = 0; i < kAssignments; ++i) combine_rows[i] = (i * 53U + 11U) % kAssignments;  // a permutation
    if (c.nans) {
        constexpr std::uint32_t patterns[] = {0x7FC00000U, 0xFFC00000U, 0x7FC12345U, 0x7FC10000U, 0x7FC00001U,
                                              0x7F800001U, 0xFFBFFFFFU, 0x7F800000U, 0xFF800000U};
        // Row r slot s: pattern (r + s) % 9 in column (r * 7 + s) % width; finite elsewhere.
        for (unsigned row = 0; row < kRows; ++row)
            for (unsigned slot = 0; slot < kTopK; ++slot) {
                const auto expert_row = combine_rows[row * kTopK + slot];
                sums[std::size_t{expert_row} * kStride + (row * 7U + slot) % kWidth] = sf::u2f(patterns[(row + slot) % 9U]);
            }
        // Two NaN terms in one sum (different payloads), Inf - Inf, Inf x 0 and NaN weights.
        sums[std::size_t{combine_rows[0]} * kStride + 100U] = sf::u2f(0x7FC0ABCDU);
        sums[std::size_t{combine_rows[1]} * kStride + 100U] = sf::u2f(0xFFC01234U);
        sums[std::size_t{combine_rows[8]} * kStride + 101U] = sf::u2f(0x7F800000U);
        sums[std::size_t{combine_rows[9]} * kStride + 101U] = sf::u2f(0xFF800000U);
        weights[combine_rows[16]] = 0.0F;
        sums[std::size_t{combine_rows[16]} * kStride + 102U] = sf::u2f(0x7F800000U);
        // A NaN weight on a NaN sum: the product keeps one NaN by operand position (rows 3, 5, 8, 12, 17).
        weights[combine_rows[24]] = sf::u2f(0x7FC00042U);
        weights[combine_rows[41]] = sf::u2f(0xFFC00007U);
        weights[combine_rows[66]] = sf::u2f(0x7FD00000U);
        weights[combine_rows[99]] = sf::u2f(0x7F800003U);
        weights[combine_rows[140]] = sf::u2f(0xFFFFFFFFU);
        weights[combine_rows[25]] = sf::u2f(0x7F800000U);
    }
    float* d_sums = upload(sums);
    float* d_weights = upload(weights);
    std::uint32_t* d_rows = upload(combine_rows);
    float* d_out_a = upload(std::vector<float>(std::size_t{kRows} * kOutStride, kGuard));
    float* d_out_b = upload(std::vector<float>(std::size_t{kRows} * kOutStride, kGuard));
    const std::vector<float> stale(sums.size(), c.stale_flag ? 12345.0F : kGuard);
    float* d_exact = upload(stale);
    std::uint16_t* d_image = upload(std::vector<std::uint16_t>(sums.size(), 0xFF81U));
    unsigned* d_seen = upload(std::vector<unsigned>{c.stale_flag ? 1U : 0U});

    // Arm A: the incumbent combine over the FP32 sums.
    ok(grouped::combine({d_sums, kStride, d_rows, d_weights, kRows, kTopK, kWidth, d_out_a, kOutStride}, nullptr), "A");
    // Arm B: the down store's image + combine_image.
    const gemm::Bf16NanExactStore store{d_image, d_exact, d_seen, kStride, 0U};
    store_kernel<<<dim3((kWidth + 63U) / 64U, kAssignments), 64>>>(d_sums, store, kAssignments, kWidth);
    ok(hipGetLastError(), "store");
    ok(grouped::combine_image({d_image, d_exact, d_seen, kStride, d_rows, d_weights, kRows, kTopK, kWidth, d_out_b,
                               kOutStride}, nullptr), "B");
    // Refusals before any launch.
    if (grouped::combine_image({d_image, nullptr, d_seen, kStride, d_rows, d_weights, kRows, kTopK, kWidth, d_out_b,
                                kOutStride}, nullptr) != hipErrorInvalidValue ||
        grouped::combine_image({d_image, d_exact, d_seen, kWidth - 1U, d_rows, d_weights, kRows, kTopK, kWidth, d_out_b,
                                kOutStride}, nullptr) != hipErrorInvalidValue)
        ok(hipErrorInvalidValue, "refusal");
    ok(hipDeviceSynchronize(), "sync");

    const auto a = download(d_out_a, std::size_t{kRows} * kOutStride);
    const auto b = download(d_out_b, std::size_t{kRows} * kOutStride);
    const auto exact = download(d_exact, sums.size());
    const auto image = download(d_image, sums.size());
    const unsigned seen = download(d_seen, 1U)[0];
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    for (std::size_t i = 0, shown = 0; i < a.size() && shown < 6U; ++i) {
        if (sf::f2u(a[i]) == sf::f2u(b[i])) continue;
        ++shown;
        const auto row = i / kOutStride, column = i % kOutStride;
        std::printf("  differ row %zu column %zu combine %08x image %08x terms", row, column, sf::f2u(a[i]), sf::f2u(b[i]));
        for (unsigned slot = 0; slot < kTopK; ++slot) {
            const auto e = combine_rows[row * kTopK + slot];
            std::printf(" [w %08x d %08x]", sf::f2u(weights[e]), sf::f2u(sums[std::size_t{e} * kStride + column]));
        }
        std::printf("\n");
    }
    std::size_t nan_out = 0, image_ok = 0, exact_written = 0;
    for (const float v : a) nan_out += (sf::f2u(v) & 0x7fffffffU) > 0x7f800000U ? 1U : 0U;
    for (std::size_t i = 0; i < sums.size(); ++i) {
        if (i % kStride >= kWidth) {
            image_ok += image[i] == 0xFF81U ? 1U : 0U;  // stride padding untouched
            continue;
        }
        image_ok += image[i] == sf::bf16_encode(sums[i]) ? 1U : 0U;
        const bool nan = (sf::f2u(sums[i]) & 0x7fffffffU) > 0x7f800000U;
        if (nan && std::memcmp(&exact[i], &sums[i], 4U) == 0) ++exact_written;
        if (!nan && std::memcmp(&exact[i], &stale[i], 4U) != 0) return std::printf("FAIL combine-image-hip %s: FP32 row written at a number\n", c.name), false;
    }
    std::size_t nan_sums = 0;
    for (std::size_t i = 0; i < sums.size(); ++i)
        if (i % kStride < kWidth && (sf::f2u(sums[i]) & 0x7fffffffU) > 0x7f800000U) ++nan_sums;
    const bool seen_ok = c.stale_flag ? seen == 1U : seen == (nan_sums != 0U ? 1U : 0U);
    const bool pass = same && image_ok == sums.size() && exact_written == nan_sums && seen_ok &&
                      (!c.nans || nan_out != 0U);
    std::printf("%s combine-image-hip %s output_bitwise=%d nan_outputs=%zu nan_sums=%zu exact_written=%zu image_ok=%zu/%zu "
                "nan_seen=%u\n", pass ? "PASS" : "FAIL", c.name, same ? 1 : 0, nan_out, nan_sums, exact_written, image_ok,
                sums.size(), seen);
    for (void* p : {static_cast<void*>(d_sums), static_cast<void*>(d_weights), static_cast<void*>(d_rows),
                    static_cast<void*>(d_out_a), static_cast<void*>(d_out_b), static_cast<void*>(d_exact),
                    static_cast<void*>(d_image), static_cast<void*>(d_seen)})
        (void)hipFree(p);
    return pass;
}
}  // namespace

int main() {
    std::mt19937 rng(20261124U);
    bool all = true;
    for (const Case& c : {Case{"finite", false, false}, Case{"finite-stale-flag", false, true},
                          Case{"nan", true, false}, Case{"nan-stale-flag", true, true}})
        all = run(c, rng) && all;
    std::puts(all ? "combine_image_hip: PASS" : "combine_image_hip: FAIL");
    return all ? 0 : 1;
}
