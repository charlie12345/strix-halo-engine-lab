// No-model fixture of the resident dense routes per weight storage (bf16, q8_0, q4_g64, q6_g64): forced low-bit
// routes (sweep = dense rows family, small = small-M WMMA schedule, tile = 128-row projection tile), checks against the
// host reference decode, then weight-read bandwidth; a last section times q|k|v as ONE launch. Each timed pass walks
// enough distinct weight copies (>= 512 MiB) that every launch reads DRAM.
// Usage: strix-hip-dense-lowbit-bench [N K ...]   (default: 3072 2304  1024 2304  2304 3072  384 2304)
// Output: "check fmt=F N=.. K=.. route=R rows=.. splits=.. rel_err=.. PASS|FAIL" and "bench ... us=.. GBps=.." lines.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/resident_q8_rows.hpp"
#include "core/f16_tile_gemm_small.hpp"
#include "core/iu4_onfly.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace core = strix::core;
namespace wd = strix::core::weight_decode;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::cerr << what << ": " << hipGetErrorString(e) << '\n';
        std::exit(2);
    }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    return static_cast<std::uint16_t>((x + 0x7fffU + ((x >> 16U) & 1U)) >> 16U);
}

struct Packed {
    wd::Format format;
    wei::FormatId storage;
    std::size_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<float> dequant;  // [N][K] reference weights
};

Packed pack(wd::Format format, const std::vector<std::uint16_t>& bits, unsigned n, unsigned k) {
    Packed p{format, wei::kBf16WeightStorage, wd::row_bytes(format, k), {}, std::vector<float>(std::size_t(n) * k)};
    p.bytes.resize(p.row_bytes * n);
    std::vector<float> w(bits.size());
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = core::onfly::bf16_bits_to_f32(bits[i]);
    switch (format) {
    case wd::Format::bf16:
        std::memcpy(p.bytes.data(), bits.data(), p.bytes.size());
        p.dequant = w;
        break;
    case wd::Format::q8_0: {
        p.storage = wei::kQ8WeightStorage;
        p.bytes = core::onfly::bf16_to_q8_0_host(bits.data(), n, k);
        for (std::size_t i = 0; i < w.size(); ++i) {
            const auto* blk = p.bytes.data() + (i / k) * p.row_bytes + (i % k) / 32U * 34U;
            p.dequant[i] = core::onfly::f16_bits_to_f32(wei::Q8Storage::scale_bits(blk)) *
                           float(wei::Q8Storage::code(blk, (i % k) % 32U));
        }
        break;
    }
    case wd::Format::q4_g64:
    case wd::Format::q6_g64: {
        const bool q4 = format == wd::Format::q4_g64;
        p.storage = q4 ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage;
        const unsigned block = q4 ? wei::Q4G64Storage::bytes : wei::Q6G64Storage::bytes;
        for (std::size_t b = 0; b < w.size() / 128U; ++b) {
            auto* out = p.bytes.data() + b * block;
            const bool encoded = q4 ? wei::lowbit::encode_block<wei::Q4G64Storage>(w.data() + b * 128U, out)
                                    : wei::lowbit::encode_block<wei::Q6G64Storage>(w.data() + b * 128U, out);
            if (!encoded) std::exit(3);
            for (unsigned j = 0; j < 128U; ++j)
                p.dequant[b * 128U + j] = q4 ? wei::lowbit::decode<wei::Q4G64Storage>(out, j)
                                             : wei::lowbit::decode<wei::Q6G64Storage>(out, j);
        }
        break;
    }
    }
    return p;
}

const char* name(wd::Format f) {
    return f == wd::Format::bf16 ? "bf16" : f == wd::Format::q8_0 ? "q8_0" : f == wd::Format::q4_g64 ? "q4_g64" : "q6_g64";
}

enum class Route { sweep, small, tile };
const char* route_name(Route r) { return r == Route::sweep ? "sweep" : r == Route::small ? "small" : "tile"; }

// Projections whose weights are consecutive in one image (sizes n[i], same K); outputs [projection][rows][n].
struct Group {
    std::vector<unsigned> n;
    std::size_t outputs = 0;
};

hipError_t launch(const Packed& p, const std::uint8_t* w, const Group& g, unsigned k, const float* x, float* y,
                  unsigned rows, Route route, unsigned splits) {
    namespace tg = core::f16_tile_gemm;
    if (route == Route::sweep) {
        // One launch per projection (the incumbent per-projection loop).
        std::size_t wo = 0, yo = 0;
        for (const unsigned n : g.n) {
            hipError_t e = hipSuccess;
            const core::dense_linear::View view{w + wo, k, n, p.row_bytes};
            if (p.format == wd::Format::bf16) {
                if (rows == 1U) core::detail::launch_q8_tight_rows<1, wd::Format::bf16>(view, x, y + yo, rows, nullptr);
                else if (rows <= 8U) core::detail::launch_q8_tight_rows<8, wd::Format::bf16>(view, x, y + yo, rows, nullptr);
                else core::detail::launch_q8_tight_rows<16, wd::Format::bf16>(view, x, y + yo, rows, nullptr);
                e = hipGetLastError();
            } else if (p.format == wd::Format::q4_g64) {
                e = core::detail::launch_lowbit_rows<wd::Format::q4_g64>(view, x, y + yo, rows, nullptr);
            } else if (p.format == wd::Format::q6_g64) {
                e = core::detail::launch_lowbit_rows<wd::Format::q6_g64>(view, x, y + yo, rows, nullptr);
            } else {  // q8_0: the resident entry (rows > 16 -> its tile)
                const wei::ResidentMatrixView mv{{w + wo, std::size_t(n) * p.row_bytes}, p.storage, n, k, p.row_bytes};
                e = core::enqueue_resident_q8_rows(mv, n, x, rows, y + yo, nullptr);
            }
            if (e != hipSuccess) return e;
            wo += std::size_t(n) * p.row_bytes;
            yo += std::size_t(n) * rows;
        }
        return hipSuccess;
    }
    if (p.format == wd::Format::bf16) {  // the EXACT prefill route: BF16 projection tile
        tg::Bf16ProjectionGroup<4> bgroup{};
        std::size_t wo = 0, yo = 0;
        for (const unsigned n : g.n) {
            bgroup.append({reinterpret_cast<const std::uint16_t*>(w + wo), n, y + yo, false});
            wo += std::size_t(n) * p.row_bytes;
            yo += std::size_t(n) * rows;
        }
        tg::gemm_bf16_projections(x, rows, k, bgroup, nullptr);
        return hipGetLastError();
    }
    tg::Q8ProjectionGroup<4> group{};
    std::size_t wo = 0, yo = 0;
    for (const unsigned n : g.n) {
        group.append({w + wo, n, y + yo});
        wo += std::size_t(n) * p.row_bytes;
        yo += std::size_t(n) * rows;
    }
    const bool q4 = p.format == wd::Format::q4_g64;
    if (route == Route::small)
        return q4 ? tg::small_m::gemm_lowbit<wei::Q4G64Storage>(x, rows, k, group, splits, nullptr)
                  : tg::small_m::gemm_lowbit<wei::Q6G64Storage>(x, rows, k, group, splits, nullptr);
    return q4 ? tg::gemm_lowbit_projections<wei::Q4G64Storage>(x, rows, k, group, nullptr)
              : tg::gemm_lowbit_projections<wei::Q6G64Storage>(x, rows, k, group, nullptr);
}

constexpr unsigned kMaxRows = 512U, kRefRows = 130U;

struct Case {
    Route route;
    unsigned rows, splits;
};

// Reference rows 0..kRefRows-1 of the grouped projections (double, from the stored weights), row-major [row][output].
void reference(const std::vector<Packed>& parts, const Group& g, unsigned k, const std::vector<float>& x,
               std::vector<double>& ref, std::vector<double>& mag) {
    ref.assign(std::size_t(kRefRows) * g.outputs, 0.0);
    mag.assign(ref.size(), 0.0);
    for (unsigned r = 0; r < kRefRows; ++r) {
        std::size_t o0 = 0;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            for (unsigned o = 0; o < g.n[i]; ++o) {
                double acc = 0.0, m = 0.0;
                const float* wr = parts[i].dequant.data() + std::size_t(o) * k;
                const float* xr = x.data() + std::size_t(r) * k;
                for (unsigned kk = 0; kk < k; ++kk) {
                    const double t = double(wr[kk]) * xr[kk];
                    acc += t;
                    m += std::fabs(t);
                }
                ref[std::size_t(r) * g.outputs + o0 + o] = acc;
                mag[std::size_t(r) * g.outputs + o0 + o] = m;
            }
            o0 += g.n[i];
        }
    }
}

int run_group(const std::vector<Packed>& packs, const Group& g, unsigned k, const float* dx, const std::vector<float>& x,
              float* dy, const std::string& label) {
    int failures = 0;
    const auto& p = packs.front();
    // One image of the projections' bytes, then distinct copies so that a pass reads DRAM.
    std::vector<std::uint8_t> image;
    for (const auto& q : packs) image.insert(image.end(), q.bytes.begin(), q.bytes.end());
    const std::size_t copies = std::max<std::size_t>(8U, (std::size_t(512) << 20U) / image.size() + 1U);
    std::uint8_t* dw = nullptr;
    ok(hipMalloc(&dw, copies * image.size()), "malloc w");
    for (std::size_t c = 0; c < copies; ++c)
        ok(hipMemcpy(dw + c * image.size(), image.data(), image.size(), hipMemcpyHostToDevice), "copy w");
    const bool lowbit = p.format == wd::Format::q4_g64 || p.format == wd::Format::q6_g64;
    const bool single = g.n.size() == 1U;
    unsigned panels = 0;
    for (const unsigned n : g.n) panels += (n + 63U) / 64U;
    const unsigned plan_splits = core::lowbit_small_splits(panels, k / 64U);
    std::vector<Case> checks, benches;
    if (single) checks.push_back({Route::sweep, 8U, 1U});
    if (single && p.format == wd::Format::q8_0) checks.push_back({Route::sweep, 40U, 1U});  // its tile
    if (lowbit) {
        for (const unsigned s : {1U, 2U, 4U}) checks.push_back({Route::small, 37U, s});
        checks.push_back({Route::small, 1U, plan_splits});
        checks.push_back({Route::small, 64U, plan_splits});
        checks.push_back({Route::tile, kRefRows, 1U});
    }
    for (const unsigned r : {1U, 8U, 16U}) benches.push_back({Route::sweep, r, 1U});
    if (single && p.format == wd::Format::q8_0)  // rows > 16: the Q8_0 resident tile
        for (const unsigned r : {32U, 64U, 128U, 512U}) benches.push_back({Route::sweep, r, 1U});
    if (single && p.format == wd::Format::bf16)
        for (const unsigned r : {128U, 512U}) benches.push_back({Route::tile, r, 1U});
    if (lowbit) {
        for (const unsigned r : {1U, 2U, 4U, 8U, 16U, 32U, 48U, 64U}) benches.push_back({Route::small, r, plan_splits});
        for (const unsigned s : {1U, 2U, 4U})
            if (s != plan_splits) benches.push_back({Route::small, 16U, s});
        for (const unsigned r : {128U, 512U}) benches.push_back({Route::tile, r, 1U});
    }
    if (!checks.empty()) {
        std::vector<double> ref, mag;
        reference(packs, g, k, x, ref, mag);
        for (const auto& c : checks) {
            ok(hipMemset(dy, 0, std::size_t(kMaxRows) * g.outputs * 4U), "clear y");
            ok(launch(p, dw, g, k, dx, dy, c.rows, c.route, c.splits), "launch check");
            ok(hipDeviceSynchronize(), "check sync");
            std::vector<float> y(std::size_t(c.rows) * g.outputs);
            ok(hipMemcpy(y.data(), dy, y.size() * 4U, hipMemcpyDeviceToHost), "copy y");
            double worst = 0.0;
            std::size_t yo = 0, o0 = 0;
            for (const unsigned n : g.n) {
                for (unsigned r = 0; r < c.rows; ++r)
                    for (unsigned o = 0; o < n; ++o) {
                        const std::size_t ri = std::size_t(r) * g.outputs + o0 + o;
                        const double err = std::fabs(double(y[yo + std::size_t(r) * n + o]) - ref[ri]) / (mag[ri] + 1e-30);
                        worst = std::max(worst, err == err ? err : 1e30);
                    }
                yo += std::size_t(n) * c.rows;
                o0 += n;
            }
            const bool pass = worst < 1e-4;
            failures += pass ? 0 : 1;
            std::cout << "check fmt=" << name(p.format) << " " << label << " K=" << k << " route=" << route_name(c.route)
                      << " rows=" << c.rows << " splits=" << c.splits << " rel_err=" << worst << (pass ? " PASS" : " FAIL")
                      << std::endl;
        }
    }
    for (const auto& c : benches) {
        hipEvent_t a, b;
        ok(hipEventCreate(&a), "event");
        ok(hipEventCreate(&b), "event");
        for (std::size_t i = 0; i < copies; ++i)
            ok(launch(p, dw + i * image.size(), g, k, dx, dy, c.rows, c.route, c.splits), "warm");
        constexpr int passes = 3;
        ok(hipEventRecord(a, nullptr), "record");
        for (int it = 0; it < passes; ++it)
            for (std::size_t i = 0; i < copies; ++i)
                ok(launch(p, dw + i * image.size(), g, k, dx, dy, c.rows, c.route, c.splits), "run");
        ok(hipEventRecord(b, nullptr), "record");
        ok(hipEventSynchronize(b), "sync");
        float ms = 0.0F;
        ok(hipEventElapsedTime(&ms, a, b), "elapsed");
        const double us = 1000.0 * ms / (double(passes) * double(copies));
        std::cout << "bench fmt=" << name(p.format) << " " << label << " K=" << k << " route=" << route_name(c.route)
                  << " rows=" << c.rows << " splits=" << c.splits << " bytes=" << image.size() << " us=" << us
                  << " GBps=" << double(image.size()) / (us * 1e3)
                  << " TFLOPS=" << 2.0 * double(g.outputs) * k * c.rows / (us * 1e6) << std::endl;
        ok(hipEventDestroy(a), "event");
        ok(hipEventDestroy(b), "event");
    }
    ok(hipFree(dw), "free w");
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<unsigned> shapes{3072, 2304, 1024, 2304, 2304, 3072, 384, 2304};
    if (argc > 1) {
        shapes.clear();
        for (int i = 1; i < argc; ++i) shapes.push_back(static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10)));
    }
    const wd::Format formats[] = {wd::Format::bf16, wd::Format::q8_0, wd::Format::q4_g64, wd::Format::q6_g64};
    int failures = 0;
    const auto run_shapes = [&](const std::vector<unsigned>& ns, unsigned k, bool all_formats) {
        Group g;
        g.n = ns;
        for (const unsigned n : ns) g.outputs += n;
        std::mt19937 rng(11U + static_cast<unsigned>(g.outputs) + k);
        std::normal_distribution<float> normal(0.0F, 0.02F);
        std::vector<std::vector<std::uint16_t>> bits(ns.size());
        for (std::size_t i = 0; i < ns.size(); ++i) {
            bits[i].resize(std::size_t(ns[i]) * k);
            for (auto& b : bits[i]) b = to_bf16(normal(rng));
        }
        std::vector<float> x(std::size_t(kMaxRows) * k);
        for (auto& v : x) v = normal(rng) * 50.0F;
        float *dx = nullptr, *dy = nullptr;
        ok(hipMalloc(&dx, x.size() * 4U), "malloc x");
        ok(hipMalloc(&dy, std::size_t(kMaxRows) * g.outputs * 4U), "malloc y");
        ok(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "copy x");
        std::string label = "N=";
        for (std::size_t i = 0; i < ns.size(); ++i) label += (i ? "+" : "") + std::to_string(ns[i]);
        for (const auto format : formats) {
            const bool lowbit = format == wd::Format::q4_g64 || format == wd::Format::q6_g64;
            if (!all_formats && !lowbit) continue;
            std::vector<Packed> packs;
            for (std::size_t i = 0; i < ns.size(); ++i) packs.push_back(pack(format, bits[i], ns[i], k));
            failures += run_group(packs, g, k, dx, x, dy, label);
        }
        ok(hipFree(dx), "free x");
        ok(hipFree(dy), "free y");
    };
    for (std::size_t s = 0; s + 1 < shapes.size(); s += 2) run_shapes({shapes[s]}, shapes[s + 1], true);
    if (argc <= 1) run_shapes({3072U, 1024U, 1024U}, 2304U, false);  // q | k | v of one attention visit, one launch
    std::cout << (failures ? "lowbit bench FAIL" : "lowbit bench checks PASS") << std::endl;
    return failures ? 1 : 0;
}
