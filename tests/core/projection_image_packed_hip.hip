// Image projections on the packed-operand core: enqueue_bf16_operand_image + enqueue_bf16_image_rows must store the
// bits enqueue_bf16_rows stores on the FP32 rows (the K64 packed schedule from kPackedImageMinRows rows on whole
// 128-column projections, the K32 image tile otherwise). Cases: a q/k/v-like group of unequal widths, a rounded single
// projection at K 3072, a 128-wide projection, a 64-wide projection (not admitted by the packed route), rows below the
// threshold, at it, and not a multiple of the CTA. Outputs are seeded with a NaN sentinel; every output must be finite.
// Routed cases (enqueue_bf16_rows_routed, one call of any count against enqueue_bf16_rows in runs of kGroupMax): a q-like
// call of six projections (packed, a 24-wide image projection), pitched weights whose pad holds NaN (packed with a row
// pitch), a narrow call the plan keeps on the FP32 rows; also at 8,209 rows (8,192 + 17).
// --bench: ms of the FP32 route and of image + image route (or the routed call) at 8,192 rows. Exit 1 on a failed check.
#include <hip/hip_runtime.h>
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"

#include "core/fma_bf16_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fma_rows = strix::core::fma_rows;
using strix::core::Bf16RowProjection;

constexpr unsigned kMaxOutputs = 1U << 20U;

void require(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::cerr << "HIP failure: " << what << ": " << hipGetErrorString(status) << '\n';
        std::exit(2);
    }
}

std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}

template <class T>
T* device(std::size_t n) {
    T* p = nullptr;
    require(hipMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

struct Case {
    const char* name;
    unsigned K;
    std::vector<unsigned> widths;
    bool round;
    bool routed = false;
    unsigned pitch = 0U;        // weight row elements (0 = K)
    bool expect_image = false;  // routed: the plan takes the image from kPackedImageMinRows rows
};

struct Run {
    std::vector<float*> outputs;
    std::vector<Bf16RowProjection> projections;
};

Run make_run(const Case& c, const std::vector<std::uint16_t*>& weights, unsigned rows) {
    Run run;
    for (std::size_t p = 0; p < c.widths.size(); ++p) {
        run.outputs.push_back(device<float>(std::size_t{rows} * c.widths[p]));
        run.projections.push_back({weights[p], c.widths[p], run.outputs.back(), c.round,
                                   c.pitch != 0U ? std::size_t{c.pitch} * 2U : 0U});
    }
    return run;
}

std::vector<float> fetch(const Run& run, unsigned rows) {
    std::vector<float> all;
    for (std::size_t p = 0; p < run.outputs.size(); ++p) {
        std::vector<float> h(std::size_t{rows} * run.projections[p].outputs);
        require(hipMemcpy(h.data(), run.outputs[p], h.size() * 4U, hipMemcpyDeviceToHost), "d2h");
        all.insert(all.end(), h.begin(), h.end());
    }
    return all;
}

void seed(const Run& run, unsigned rows) {
    for (std::size_t p = 0; p < run.outputs.size(); ++p)
        require(hipMemset(run.outputs[p], 0xFF, std::size_t{rows} * run.projections[p].outputs * 4U), "seed");
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const std::vector<Case> cases = {{"qkv_group", 2304U, {3072U, 1024U, 1024U}, false},
                                     {"o_round", 3072U, {2304U}, true},
                                     {"router", 2304U, {128U}, false},
                                     {"narrow64", 2304U, {64U}, false},
                                     {"routed_q_call", 2304U, {3072U, 384U, 24U, 3072U, 1024U, 1024U}, false, true, 0U, true},
                                     {"routed_o_pitched", 3072U, {2304U}, true, true, 3200U, true},
                                     {"routed_kv_narrow", 2304U, {1024U, 32U}, false, true, 0U, false}};
    const std::vector<unsigned> heights = {1000U, 1024U, 4100U, 8192U, 8209U};
    std::mt19937 rng(20261124U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const unsigned max_rows = 8209U, max_k = 3072U, bench_rows = 8192U;
    std::vector<float> host_a(std::size_t{max_rows} * max_k);
    for (auto& v : host_a) v = nd(rng);
    host_a[17] = 0.0F, host_a[18] = -0.0F, host_a[19] = 1.0e-40F;  // signed zero and a subnormal input
    float* a = device<float>(host_a.size());
    require(hipMemcpy(a, host_a.data(), host_a.size() * 4U, hipMemcpyHostToDevice), "a");
    auto* image = device<std::uint16_t>(host_a.size());
    bool ok = true;
    {  // refused before a launch: an image over its own input (whole or partial) and a grid past the launch limit
        auto* over = reinterpret_cast<std::uint16_t*>(a);
        const bool refused =
            fma_rows::enqueue_bf16_operand_image(a, 2048U, 2U, over, nullptr) == hipErrorInvalidValue &&
            fma_rows::enqueue_bf16_operand_image(a, 2048U, 2U, over + 2048U, nullptr) == hipErrorInvalidValue &&
            fma_rows::enqueue_bf16_operand_image(a + 8U, 2048U, 2U, over, nullptr) == hipErrorInvalidValue &&
            fma_rows::enqueue_bf16_operand_image(a, 8192U, (1U << 30U) + 1U, image, nullptr) == hipErrorInvalidValue &&
            hipGetLastError() == hipSuccess;
        std::cout << "rejected_arguments " << (refused ? "PASS" : "FAIL") << '\n';
        ok = ok && refused;
    }
    for (const auto& c : cases) {
        std::vector<std::uint16_t*> weights;
        const unsigned ld = c.pitch != 0U ? c.pitch : c.K;
        for (const unsigned width : c.widths) {
            std::vector<std::uint16_t> w(std::size_t{width} * ld, std::uint16_t{0x7FC0U});  // pad = NaN
            for (unsigned o = 0; o < width; ++o)
                for (unsigned k = 0; k < c.K; ++k) w[std::size_t{o} * ld + k] = bf16(0.05F * nd(rng));
            weights.push_back(device<std::uint16_t>(w.size()));
            require(hipMemcpy(weights.back(), w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
        }
        for (const unsigned rows : heights) {
            const auto ref = make_run(c, weights, rows);
            const auto img = make_run(c, weights, rows);
            seed(ref, rows);
            seed(img, rows);
            const std::span<std::uint16_t> scratch(image, host_a.size());
            const auto fp32 = [&](const Run& run) {
                return strix::core::for_each_bf16_projection_group<fma_rows::kGroupMax>(run.projections, hipSuccess,
                    [&](std::span<const Bf16RowProjection> group) {
                        return fma_rows::enqueue_bf16_rows(a, c.K, rows, group, kMaxOutputs, nullptr);
                    });
            };
            const auto image_arm = [&](const Run& run) {
                if (c.routed)
                    return fma_rows::enqueue_bf16_rows_routed(a, c.K, rows, run.projections, kMaxOutputs, scratch,
                                                              nullptr);
                if (const auto status = fma_rows::enqueue_bf16_operand_image(a, c.K, rows, image, nullptr);
                    status != hipSuccess)
                    return status;
                return fma_rows::enqueue_bf16_image_rows(image, c.K, rows, run.projections, kMaxOutputs, nullptr);
            };
            require(fp32(ref), "fp32 rows");
            bool admitted = fma_rows::admits_bf16_image_projections(c.K, rows, img.projections, kMaxOutputs);
            if (c.routed) {
                strix::core::ImageRoute routes[fma_rows::kImageRouteMaxProjections]{};
                const bool planned = strix::core::plan_image_routes(fma_rows::kImageRoutePolicy, c.K, rows,
                    img.projections, kMaxOutputs, routes, [&](const Bf16RowProjection& projection) {
                        return fma_rows::admits_bf16_image_projections(c.K, rows, {&projection, 1U}, kMaxOutputs);
                    });
                admitted = planned == (c.expect_image && rows >= fma_rows::kPackedImageMinRows);
            }
            require(image_arm(img), "image rows");
            require(hipDeviceSynchronize(), "sync");
            const auto x = fetch(ref, rows), y = fetch(img, rows);
            const bool finite = std::all_of(y.begin(), y.end(), [](float v) { return std::isfinite(v); });
            const bool same = x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4U) == 0;
            ok = ok && admitted && finite && same;
            std::cout << "check " << c.name << " rows " << rows << (admitted ? " admitted" : " NOT-ADMITTED")
                      << (finite ? " finite" : " NONFINITE") << (same ? " same-bits" : " DIFFER")
                      << (admitted && finite && same ? " PASS" : " FAIL") << '\n';
            if (bench && rows == bench_rows) {
                hipEvent_t e0, e1;
                require(hipEventCreate(&e0), "e");
                require(hipEventCreate(&e1), "e");
                double t[2];
                for (unsigned arm = 0; arm < 2U; ++arm) {
                    std::vector<float> ms;
                    for (unsigned rep = 0; rep < 8U; ++rep) {
                        require(hipEventRecord(e0, nullptr), "r");
                        require(arm == 0U ? fp32(ref) : image_arm(img), "b");
                        require(hipEventRecord(e1, nullptr), "r");
                        require(hipEventSynchronize(e1), "s");
                        float v = 0.0F;
                        require(hipEventElapsedTime(&v, e0, e1), "el");
                        if (rep > 0U) ms.push_back(v);
                    }
                    std::sort(ms.begin(), ms.end());
                    t[arm] = double(ms[ms.size() / 2U]);
                }
                std::cout << "bench " << c.name << " rows " << rows << " fp32_rows_ms " << t[0] << " image_route_ms "
                          << t[1] << " ratio " << t[1] / t[0] << '\n';
            }
            for (auto* p : ref.outputs) require(hipFree(p), "free");
            for (auto* p : img.outputs) require(hipFree(p), "free");
        }
        for (auto* w : weights) require(hipFree(w), "free");
    }
    std::cout << (ok ? "ALL PASS" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
