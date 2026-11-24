// Rescore trait of the screened head (core/hip_screened_head.hpp enqueue_rescore_rows, screened_head_plan RescoreLaunch),
// device test. Approximate scores = the incumbent full head (BF16 tight rows + bf16 softcap) plus noise; after the trait,
// per lane:
//   1. the device threshold equals the host reference screened_head_plan::rescore_threshold;
//   2. every selected row (key >= threshold) holds the incumbent full-head bits, every other row its approximate bits;
//   3. the selected count is the host count (every row when it exceeds the capacity or the lane has fewer than k rows).
// Cases: lanes 1 / 3 / 8, widths 512 / 2304, k 64 / 1, a pitch wider than the rows, all-equal scores (capacity overflow:
// every row exact), fewer rows than k, a NaN score; invalid launches are refused.
// --bench: no-model timing of the trait at rows 200058 x width 2304, k 64, lanes 1 / 8 / 16, beside the BF16 full head.
#include "core/hip_screened_head.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

namespace hs = strix::core::hip_screened_head;
namespace plan = strix::core::screened_head_plan;
namespace dl = strix::core::dense_linear;

int failures = 0;

void check(hipError_t status, const char* what) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(status));
    std::exit(2);
}

void expect(bool ok, const char* what, unsigned a = 0U, unsigned b = 0U) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s (%u, %u)\n", what, a, b);
}

__global__ void softcap_kernel(float* y, unsigned n, float cap) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = strix::core::row_transforms::bf16_softcap(y[i], cap);
}

template <class T>
T* device_alloc(std::size_t count) {
    void* p = nullptr;
    check(hipMalloc(&p, count * sizeof(T)), "alloc");
    return static_cast<T*>(p);
}

std::uint16_t to_bf16(float v) { return static_cast<std::uint16_t>(strix::weight::scalar_float::bf16_encode(v)); }

constexpr float kCap = 30.0F;

enum class Noise { gaussian, equal, nan_score };

void run(unsigned rows, unsigned width, unsigned lanes, unsigned k, Noise noise, std::mt19937& rng) {
    const unsigned stride = width + 8U;
    const unsigned pitch = rows + 7U;
    const unsigned capacity = plan::kCapacity;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(std::size_t{lanes} * stride, 0.0F);
    for (auto& v : x) v = normal(rng);
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    for (auto& v : w) v = to_bf16(0.05F * normal(rng));
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(x.size());
    auto* dy = device_alloc<float>(rows);
    auto* ds = device_alloc<float>(std::size_t{lanes} * pitch);
    const auto layout = plan::rescore_layout(rows, capacity, lanes);
    auto* scratch = device_alloc<std::uint8_t>(layout.bytes);
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "upload x");
    const dl::View view{reinterpret_cast<const std::uint8_t*>(dw), width, rows, std::size_t{width} * 2U};
    std::vector<float> ref(std::size_t{lanes} * pitch, 0.0F), approx(std::size_t{lanes} * pitch, -1.0F);
    for (unsigned l = 0; l < lanes; ++l) {
        check(dl::enqueue_bf16_tight_rows(view, dx + std::size_t{l} * stride, dy, 1U, nullptr), "full gemv");
        softcap_kernel<<<(rows + 255U) / 256U, 256>>>(dy, rows, kCap);
        check(hipMemcpy(ref.data() + std::size_t{l} * pitch, dy, rows * 4U, hipMemcpyDeviceToHost), "copy ref");
        for (unsigned i = 0; i < rows; ++i) {
            float& a = approx[std::size_t{l} * pitch + i];
            const float r = ref[std::size_t{l} * pitch + i];
            a = noise == Noise::equal ? 1.0F : r + 0.3F * normal(rng);
        }
        if (noise == Noise::nan_score) approx[std::size_t{l} * pitch + rows / 3U] = std::numeric_limits<float>::quiet_NaN();
    }
    check(hipMemcpy(ds, approx.data(), approx.size() * 4U, hipMemcpyHostToDevice), "upload scores");
    const plan::RescoreLaunch launch{{dw, rows, width, kCap}, dx, ds, scratch, lanes, stride, pitch, k, capacity, 1024U};
    check(hs::enqueue_rescore_rows(launch, nullptr), "rescore");
    check(hipDeviceSynchronize(), "sync");
    std::vector<float> out(approx.size());
    check(hipMemcpy(out.data(), ds, out.size() * 4U, hipMemcpyDeviceToHost), "copy scores");
    std::vector<plan::RescoreState> states(lanes);
    check(hipMemcpy(states.data(), scratch, lanes * sizeof(plan::RescoreState), hipMemcpyDeviceToHost), "copy states");
    unsigned max_selected = 0U;
    for (unsigned l = 0; l < lanes; ++l) {
        const std::span<const float> lane_approx(approx.data() + std::size_t{l} * pitch, rows);
        const auto threshold = plan::rescore_threshold(lane_approx, k);
        expect(states[l].key == threshold, "device threshold == host reference", states[l].key, threshold);
        unsigned selected = 0U, bad_in = 0U, bad_out = 0U;
        for (unsigned i = 0; i < rows; ++i) selected += plan::order_key(lane_approx[i]) >= threshold ? 1U : 0U;
        const bool all = selected > capacity;
        for (unsigned i = 0; i < rows; ++i) {
            const std::size_t at = std::size_t{l} * pitch + i;
            const bool in = all || plan::order_key(approx[at]) >= threshold;
            const auto got = std::bit_cast<std::uint32_t>(out[at]);
            if (in && got != std::bit_cast<std::uint32_t>(ref[at])) ++bad_in;
            if (!in && got != std::bit_cast<std::uint32_t>(approx[at])) ++bad_out;
        }
        for (unsigned i = rows; i < pitch; ++i)  // the pitch tail is never written
            if (std::bit_cast<std::uint32_t>(out[std::size_t{l} * pitch + i]) !=
                std::bit_cast<std::uint32_t>(approx[std::size_t{l} * pitch + i]))
                ++bad_out;
        expect(bad_in == 0U, "selected rows hold the incumbent bits", l, bad_in);
        expect(bad_out == 0U, "other rows keep the approximate bits", l, bad_out);
        expect(states[l].count == selected, "device count == host count", states[l].count, selected);
        expect(selected >= std::min(k, rows), "at least k rows selected", selected, k);
        max_selected = std::max(max_selected, selected);
    }
    std::printf("rows %u width %u lanes %u k %u noise %d: max selected %u\n", rows, width, lanes, k,
                static_cast<int>(noise), max_selected);
    // Invalid launches are refused before any launch.
    auto bad = launch;
    bad.k = 0U;
    expect(hs::enqueue_rescore_rows(bad, nullptr) == hipErrorInvalidValue, "k 0 refused");
    bad = launch;
    bad.capacity = k - 1U;
    if (k > 1U) expect(hs::enqueue_rescore_rows(bad, nullptr) == hipErrorInvalidValue, "capacity < k refused");
    bad = launch;
    bad.pitch = rows - 1U;
    expect(hs::enqueue_rescore_rows(bad, nullptr) == hipErrorInvalidValue, "pitch < rows refused");
    bad = launch;
    bad.scratch = scratch + 4;
    expect(hs::enqueue_rescore_rows(bad, nullptr) == hipErrorInvalidValue, "misaligned scratch refused");
    bad = launch;
    bad.head.softcap = 0.0F;
    expect(hs::enqueue_rescore_rows(bad, nullptr) == hipErrorInvalidValue, "no post-transform refused");
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
    check(hipFree(dy), "free");
    check(hipFree(ds), "free");
    check(hipFree(scratch), "free");
}

// No-model timing: the trait alone (scores already written) per lane count, and the BF16 full head per lane.
void bench() {
    constexpr unsigned rows = 200058U, width = 2304U, k = 64U, iters = 50U;
    std::mt19937 rng(7U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    for (auto& v : w) v = to_bf16(0.02F * normal(rng));
    constexpr unsigned max_lanes = 16U;
    std::vector<float> x(std::size_t{max_lanes} * width);
    for (auto& v : x) v = normal(rng);
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(x.size());
    auto* ds = device_alloc<float>(std::size_t{max_lanes} * rows);
    auto* scratch = device_alloc<std::uint8_t>(plan::rescore_layout(rows, plan::kCapacity, max_lanes).bytes);
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "upload x");
    const dl::View view{reinterpret_cast<const std::uint8_t*>(dw), width, rows, std::size_t{width} * 2U};
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "event");
    check(hipEventCreate(&stop), "event");
    const auto time = [&](auto&& body) {
        body();
        check(hipDeviceSynchronize(), "warm");
        check(hipEventRecord(start, nullptr), "record");
        for (unsigned i = 0; i < iters; ++i) body();
        check(hipEventRecord(stop, nullptr), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0.0F;
        check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        return 1000.0 * static_cast<double>(ms) / iters;
    };
    for (unsigned l = 0; l < max_lanes; ++l) {
        check(dl::enqueue_bf16_tight_rows(view, dx + std::size_t{l} * width, ds + std::size_t{l} * rows, 1U, nullptr),
              "scores");
        softcap_kernel<<<(rows + 255U) / 256U, 256>>>(ds + std::size_t{l} * rows, rows, kCap);
    }
    const double full_us = time([&] {
        check(dl::enqueue_bf16_tight_rows(view, dx, ds, 1U, nullptr), "full head");
    });
    std::printf("bench full BF16 head one lane us=%.1f GBps=%.1f\n", full_us,
                static_cast<double>(w.size()) * 2.0 / (full_us * 1e3));
    for (const unsigned lanes : {1U, 8U, 16U}) {
        const plan::RescoreLaunch launch{{dw, rows, width, kCap}, dx, ds, scratch, lanes, width, rows, k,
                                         plan::kCapacity, 1024U};
        const double us = time([&] { check(hs::enqueue_rescore_rows(launch, nullptr), "rescore"); });
        std::printf("bench rescore rows %u k %u lanes %u us=%.1f us_per_lane=%.1f\n", rows, k, lanes, us, us / lanes);
    }
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
    check(hipFree(ds), "free");
    check(hipFree(scratch), "free");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    std::mt19937 rng(20261124U);
    run(50021U, 512U, 1U, 64U, Noise::gaussian, rng);
    run(50021U, 512U, 3U, 64U, Noise::gaussian, rng);
    run(30011U, 2304U, 8U, 64U, Noise::gaussian, rng);
    run(20000U, 512U, 2U, 1U, Noise::gaussian, rng);
    run(20000U, 512U, 2U, 64U, Noise::equal, rng);      // every key equal: count > capacity, every row exact
    run(40U, 512U, 2U, 64U, Noise::gaussian, rng);      // fewer rows than k: every row exact
    run(20000U, 512U, 3U, 64U, Noise::nan_score, rng);  // a NaN score has the largest key: selected, recomputed
    std::printf(failures == 0 ? "screened head rescore: PASS\n" : "screened head rescore: FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
