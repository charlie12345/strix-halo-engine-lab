// No-model fixture of the FAST small-M form of the grouped microscaled tile (grouped_microscaled::Form::small_f16) at the
// Inkling 2 Mini expert shapes (gate_up 2304 x 2304 interleaved halves, down 2304 x 1152; 128 experts, top 8), NVFP4 and
// MXFP8. Checks: (1) every output of a routed 8-row visit against a double reference of the decoded weights and the BF16
// inputs (normalised by sum |x w|), for the small form and the incumbent BF16 tile; (2) row invariance: the first row of
// each tile run alone gives the same bits. Then ms per visit (gate_up + down projections) of R1 rows (8-row chunks,
// activation and combine included), the incumbent tile and the small form, over rows 1..40, uniform and skewed routes,
// and GB/s over the distinct expert bytes. Exit 1 on a failed check.
#include <hip/hip_runtime.h>

#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_microscaled.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace packed = strix::weight::microscaled;
namespace sf = strix::weight::scalar_float;
namespace grouped = strix::moe_core::grouped_microscaled;
namespace r1 = strix::moe_core::microscaled_r1;
using strix::moe_core::device_grouped::ExpertWeights;
using Format = packed::MatrixFormat;

constexpr unsigned kExperts = 128U, kTopK = 8U, kHidden = 2304U, kInter = 1152U, kGate = 2U * kInter;
constexpr unsigned kVariants = 6U;  // route draws cycled per timed rep (distinct experts per rep, cold weights)

void require_hip(hipError_t status, const char* what) {
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
float widen(std::uint16_t h) {
    const std::uint32_t u = std::uint32_t{h} << 16U;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

template <class T>
T* device_copy(const std::vector<T>& host) {
    T* p = nullptr;
    require_hip(hipMalloc(&p, host.size() * sizeof(T)), "malloc");
    require_hip(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "copy");
    return p;
}

struct Matrix {  // host bytes of one expert matrix
    std::vector<std::uint8_t> codes, scales;
};

// Random valid codes; scales in a range that keeps every product finite.
Matrix make_matrix(Format f, unsigned rows, unsigned cols, std::mt19937& rng) {
    Matrix m;
    const bool nv = f == Format::nvfp4;
    m.codes.resize(std::size_t{rows} * (nv ? cols / 2U : cols));
    m.scales.resize(std::size_t{rows} * cols / (nv ? 16U : 32U));
    for (std::size_t i = 0; i < m.codes.size(); i += 4U) {
        const auto word = static_cast<std::uint32_t>(rng());
        for (std::size_t j = 0; j < 4U && i + j < m.codes.size(); ++j) {
            auto b = static_cast<std::uint8_t>(word >> (8U * j));
            if (!nv && (b & 0x7FU) == 0x7FU) b = static_cast<std::uint8_t>(b ^ 1U);  // no E4M3 NaN
            m.codes[i + j] = b;
        }
    }
    std::uniform_int_distribution<int> s4(0x20, 0x40), s8(118, 124);
    for (auto& s : m.scales) s = static_cast<std::uint8_t>(nv ? s4(rng) : s8(rng));
    return m;
}

double weight(Format f, const Matrix& m, float outer, unsigned cols, std::size_t prow, unsigned k) {
    if (f == Format::nvfp4) {
        const auto byte = m.codes[prow * (cols / 2U) + k / 2U];
        const auto code = static_cast<std::uint8_t>(k % 2U ? byte >> 4U : byte & 0x0FU);
        return double{sf::e2m1(code)} * double{sf::e4m3(m.scales[prow * (cols / 16U) + k / 16U])} * double{outer};
    }
    return double{sf::e4m3(m.codes[prow * cols + k])} * double{sf::e8m0(m.scales[prow * (cols / 32U) + k / 32U])};
}

struct Routes {
    std::vector<unsigned> ids;  // [rows][8] token-major
    unsigned distinct;
};

Routes draw(unsigned rows, bool skewed, std::mt19937& rng) {
    std::vector<double> p(kExperts, 1.0);
    if (skewed) {
        std::vector<unsigned> perm(kExperts);
        std::iota(perm.begin(), perm.end(), 0U);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (unsigned e = 0; e < kExperts; ++e) p[perm[e]] = 1.0 / std::pow(double(e + 1U), 0.9);
    }
    Routes r{{}, 0U};
    std::vector<int> seen(kExperts, 0);
    for (unsigned row = 0; row < rows; ++row) {
        auto w = p;
        for (unsigned k = 0; k < kTopK; ++k) {
            std::discrete_distribution<unsigned> d(w.begin(), w.end());
            const unsigned e = d(rng);
            w[e] = 0.0;
            r.ids.push_back(e);
            seen[e] = 1;
        }
    }
    r.distinct = static_cast<unsigned>(std::count(seen.begin(), seen.end(), 1));
    return r;
}

struct Format_ {
    Format f;
    std::vector<Matrix> gu, dn;
    std::vector<float> outer;
    std::vector<ExpertWeights> host;  // device addresses
    ExpertWeights* dev{};
    double expert_bytes{};
};

Format_ make_format(Format f, std::mt19937& rng) {
    Format_ F{f, {}, {}, {}, {}, nullptr, 0.0};
    const bool nv = f == Format::nvfp4;
    for (unsigned e = 0; e < kExperts; ++e) {
        F.gu.push_back(make_matrix(f, kGate, kHidden, rng));
        F.dn.push_back(make_matrix(f, kHidden, kInter, rng));
        F.outer.push_back(nv ? 0.75F + 0.001F * float(e) : 1.0F);
        const auto upload = [&](const Matrix& m, unsigned rows, unsigned cols, packed::RowOrder order) {
            packed::PackedMatrix pm{device_copy(m.codes), device_copy(m.scales), rows, cols, F.outer.back(), order};
            return pm;
        };
        F.host.push_back({upload(F.gu.back(), kGate, kHidden, packed::RowOrder::interleaved_halves),
                          upload(F.dn.back(), kHidden, kInter, packed::RowOrder::contiguous)});
    }
    F.dev = device_copy(F.host);
    const auto& m0 = F.gu[0];
    const auto& d0 = F.dn[0];
    F.expert_bytes = double(m0.codes.size() + m0.scales.size() + d0.codes.size() + d0.scales.size());
    return F;
}

// Expert-major tiles of one route draw over gathered rows (assignment a = position in expert-major order).
struct Plan {
    std::vector<grouped::TileRows> gu, dn;
    std::vector<unsigned> token;  // assignment -> token row
    unsigned max_rows{};
};

Plan plan_of(const Format_& F, const Routes& r, unsigned rows, std::uint16_t* gu_in, float* gu_out, std::uint16_t* dn_in,
             float* dn_out) {
    Plan p;
    for (unsigned e = 0; e < kExperts; ++e) {
        const unsigned first = static_cast<unsigned>(p.token.size());
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned k = 0; k < kTopK; ++k)
                if (r.ids[row * kTopK + k] == e) p.token.push_back(row);
        const unsigned n = static_cast<unsigned>(p.token.size()) - first;
        if (n == 0U) continue;
        p.max_rows = std::max(p.max_rows, n);
        p.gu.push_back({F.host[e].gate_up, gu_in + std::size_t{first} * kHidden, gu_out + std::size_t{first} * kGate,
                        kHidden, kGate, n});
        p.dn.push_back({F.host[e].down, dn_in + std::size_t{first} * kInter, dn_out + std::size_t{first} * kHidden,
                        kInter, kHidden, n});
    }
    return p;
}

struct Device {
    std::uint16_t *gu_in{}, *dn_in{};
    float *gu_out{}, *dn_out{};
    grouped::TileRows *gu_tiles{}, *dn_tiles{};
    // R1
    unsigned* ids{};
    float *weights{}, *x{}, *y{};
    std::uint16_t* act{};
    unsigned* failure{};
};

constexpr unsigned kMaxAssign = 48U * kTopK;

Device make_device() {
    Device d;
    require_hip(hipMalloc(&d.gu_in, std::size_t{kMaxAssign} * kHidden * 2U), "m");
    require_hip(hipMalloc(&d.dn_in, std::size_t{kMaxAssign} * kInter * 2U), "m");
    require_hip(hipMalloc(&d.gu_out, std::size_t{kMaxAssign} * kGate * 4U), "m");
    require_hip(hipMalloc(&d.dn_out, std::size_t{kMaxAssign} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.gu_tiles, kExperts * sizeof(grouped::TileRows)), "m");
    require_hip(hipMalloc(&d.dn_tiles, kExperts * sizeof(grouped::TileRows)), "m");
    require_hip(hipMalloc(&d.ids, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.weights, kMaxAssign * 4U), "m");
    require_hip(hipMalloc(&d.x, std::size_t{48U} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.y, std::size_t{48U} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.act, std::size_t{kMaxAssign} * kInter * 2U), "m");
    require_hip(hipMalloc(&d.failure, 48U * 4U), "m");
    require_hip(hipMemset(d.failure, 0, 48U * 4U), "m");
    return d;
}

grouped::Args args_of(const grouped::TileRows* tiles, unsigned count, unsigned inputs, unsigned outputs, unsigned max_rows,
                      grouped::Form form) {
    grouped::Args a{};
    a.tiles = tiles;
    a.count = count;
    a.inputs = inputs;
    a.outputs = outputs;
    a.max_tile_rows = max_rows;
    a.form = form;
    return a;
}

template <Format F>
void run_tiles(const Plan& p, const Device& d, grouped::Form form) {
    require_hip(grouped::launch<F>(args_of(d.gu_tiles, unsigned(p.gu.size()), kHidden, kGate, p.max_rows, form), nullptr),
                "gate_up");
    require_hip(grouped::launch<F>(args_of(d.dn_tiles, unsigned(p.dn.size()), kInter, kHidden, p.max_rows, form), nullptr),
                "down");
}

template <Format F>
void run_r1(const Format_& Fm, const Device& d, unsigned rows) {
    for (unsigned r0 = 0; r0 < rows; r0 += r1::kMaxRows) {
        const unsigned n = std::min(r1::kMaxRows, rows - r0);
        const r1::Args a{Fm.dev, d.ids + r0 * kTopK, d.weights + r0 * kTopK, d.failure + r0, d.x + std::size_t{r0} * kHidden,
                         d.act, d.y + std::size_t{r0} * kHidden, kExperts, kHidden, kInter, 4.0F, 25.0F};
        require_hip(r1::launch<F, kTopK>(a, nullptr, n), "r1");
    }
}

void upload_plan(const Plan& p, const Device& d) {
    require_hip(hipMemcpy(d.gu_tiles, p.gu.data(), p.gu.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice), "t");
    require_hip(hipMemcpy(d.dn_tiles, p.dn.data(), p.dn.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice), "t");
}

template <Format F>
bool check(const Format_& Fm, const Device& d, std::mt19937& rng) {
    constexpr unsigned rows = 8U;
    const auto routes = draw(rows, false, rng);
    const auto p = plan_of(Fm, routes, rows, d.gu_in, d.gu_out, d.dn_in, d.dn_out);
    const unsigned A = unsigned(p.token.size());
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> xg(std::size_t{A} * kHidden), xd(std::size_t{A} * kInter);
    for (auto& v : xg) v = bf16(nd(rng));
    for (auto& v : xd) v = bf16(nd(rng));
    require_hip(hipMemcpy(d.gu_in, xg.data(), xg.size() * 2U, hipMemcpyHostToDevice), "x");
    require_hip(hipMemcpy(d.dn_in, xd.data(), xd.size() * 2U, hipMemcpyHostToDevice), "x");
    upload_plan(p, d);
    // Reference (double) per tile row.
    std::vector<double> ref_gu(std::size_t{A} * kGate), ref_dn(std::size_t{A} * kHidden), mag_gu(ref_gu.size()),
        mag_dn(ref_dn.size());
    unsigned a = 0;
    for (std::size_t t = 0; t < p.gu.size(); ++t) {
        unsigned e = 0;
        while (Fm.host[e].gate_up.codes != p.gu[t].weights.codes) ++e;
        for (unsigned r = 0; r < p.gu[t].rows; ++r, ++a) {
            for (unsigned n = 0; n < kGate; ++n) {
                const auto prow = packed::physical_row(packed::RowOrder::interleaved_halves, kGate, n);
                double s = 0.0, m = 0.0;
                for (unsigned k = 0; k < kHidden; ++k) {
                    const double v = double(widen(xg[std::size_t{a} * kHidden + k])) * weight(F, Fm.gu[e], Fm.outer[e], kHidden, prow, k);
                    s += v;
                    m += std::fabs(v);
                }
                ref_gu[std::size_t{a} * kGate + n] = s;
                mag_gu[std::size_t{a} * kGate + n] = m;
            }
            for (unsigned n = 0; n < kHidden; ++n) {
                double s = 0.0, m = 0.0;
                for (unsigned k = 0; k < kInter; ++k) {
                    const double v = double(widen(xd[std::size_t{a} * kInter + k])) * weight(F, Fm.dn[e], Fm.outer[e], kInter, n, k);
                    s += v;
                    m += std::fabs(v);
                }
                ref_dn[std::size_t{a} * kHidden + n] = s;
                mag_dn[std::size_t{a} * kHidden + n] = m;
            }
        }
    }
    bool ok = true;
    std::vector<float> gu(ref_gu.size()), dn(ref_dn.size());
    double small_err = 0.0;
    for (auto form : {grouped::Form::bf16_tile, grouped::Form::small_f16}) {
        require_hip(hipMemset(d.gu_out, 0xFF, gu.size() * 4U), "m");
        require_hip(hipMemset(d.dn_out, 0xFF, dn.size() * 4U), "m");
        run_tiles<F>(p, d, form);
        require_hip(hipDeviceSynchronize(), "sync");
        require_hip(hipMemcpy(gu.data(), d.gu_out, gu.size() * 4U, hipMemcpyDeviceToHost), "y");
        require_hip(hipMemcpy(dn.data(), d.dn_out, dn.size() * 4U, hipMemcpyDeviceToHost), "y");
        double eg = 0.0, ed = 0.0;
        for (std::size_t i = 0; i < gu.size(); ++i) eg = std::max(eg, std::fabs(double(gu[i]) - ref_gu[i]) / (mag_gu[i] + 1e-30));
        for (std::size_t i = 0; i < dn.size(); ++i) ed = std::max(ed, std::fabs(double(dn[i]) - ref_dn[i]) / (mag_dn[i] + 1e-30));
        const bool small = form == grouped::Form::small_f16;
        if (small) small_err = std::max(eg, ed);
        std::cout << (F == Format::nvfp4 ? "nvfp4" : "mxfp8") << (small ? " small_f16" : " bf16_tile")
                  << " vs double: max |err| / sum|xw| gate_up " << eg << " down " << ed << '\n';
    }
    if (!(small_err < 1e-5)) ok = false;
    // Row invariance: the small form on one-row tiles (each tile's first row) gives row 0 of each tile bitwise.
    auto one = p;
    for (auto& t : one.gu) t.rows = 1U;
    for (auto& t : one.dn) t.rows = 1U;
    one.max_rows = 1U;
    upload_plan(one, d);
    std::vector<float> gu1(gu.size()), dn1(dn.size());
    require_hip(hipMemset(d.gu_out, 0xFF, gu.size() * 4U), "m");
    require_hip(hipMemset(d.dn_out, 0xFF, dn.size() * 4U), "m");
    run_tiles<F>(one, d, grouped::Form::small_f16);
    require_hip(hipDeviceSynchronize(), "sync");
    require_hip(hipMemcpy(gu1.data(), d.gu_out, gu.size() * 4U, hipMemcpyDeviceToHost), "y");
    require_hip(hipMemcpy(dn1.data(), d.dn_out, dn.size() * 4U, hipMemcpyDeviceToHost), "y");
    unsigned diff = 0U, first = 0U;
    for (std::size_t t = 0; t < p.gu.size(); ++t) {
        diff += std::memcmp(&gu[std::size_t{first} * kGate], &gu1[std::size_t{first} * kGate], kGate * 4U) != 0;
        diff += std::memcmp(&dn[std::size_t{first} * kHidden], &dn1[std::size_t{first} * kHidden], kHidden * 4U) != 0;
        first += p.gu[t].rows;
    }
    std::cout << (F == Format::nvfp4 ? "nvfp4" : "mxfp8") << " small_f16 row invariance (tile row 0 alone vs in its tile): "
              << (diff == 0U ? "SAME" : "DIFF") << " (" << p.gu.size() << " tiles)\n";
    if (diff != 0U) ok = false;
    std::cout << (F == Format::nvfp4 ? "nvfp4" : "mxfp8") << " check " << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

template <Format F>
void bench(const Format_& Fm, const Device& d, std::mt19937& rng) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    std::normal_distribution<float> nd(0.0F, 1.0F);
    {
        std::vector<std::uint16_t> xg(std::size_t{kMaxAssign} * kHidden), xd(std::size_t{kMaxAssign} * kInter);
        for (auto& v : xg) v = bf16(nd(rng));
        for (auto& v : xd) v = bf16(nd(rng));
        require_hip(hipMemcpy(d.gu_in, xg.data(), xg.size() * 2U, hipMemcpyHostToDevice), "x");
        require_hip(hipMemcpy(d.dn_in, xd.data(), xd.size() * 2U, hipMemcpyHostToDevice), "x");
        std::vector<float> x(std::size_t{48U} * kHidden);
        for (auto& v : x) v = nd(rng);
        require_hip(hipMemcpy(d.x, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x");
    }
    hipEvent_t e0, e1;
    require_hip(hipEventCreate(&e0), "e");
    require_hip(hipEventCreate(&e1), "e");
    for (unsigned rows : {1U, 2U, 4U, 8U, 16U, 40U}) {
        for (bool skewed : {false, true}) {
            std::vector<Routes> rs;
            std::vector<Plan> ps;
            double distinct = 0.0;
            for (unsigned v = 0; v < kVariants; ++v) {
                rs.push_back(draw(rows, skewed, rng));
                ps.push_back(plan_of(Fm, rs.back(), rows, d.gu_in, d.gu_out, d.dn_in, d.dn_out));
                distinct += rs.back().distinct;
            }
            distinct /= kVariants;
            const auto time = [&](auto&& body) {
                std::vector<float> ms;
                for (unsigned rep = 0; rep < 4U * kVariants; ++rep) {
                    const unsigned v = rep % kVariants;
                    std::vector<float> wts(rs[v].ids.size(), 0.125F);
                    require_hip(hipMemcpy(d.ids, rs[v].ids.data(), rs[v].ids.size() * 4U, hipMemcpyHostToDevice), "ids");
                    require_hip(hipMemcpy(d.weights, wts.data(), wts.size() * 4U, hipMemcpyHostToDevice), "w");
                    upload_plan(ps[v], d);
                    require_hip(hipDeviceSynchronize(), "sync");
                    require_hip(hipEventRecord(e0, nullptr), "rec");
                    body(ps[v]);
                    require_hip(hipEventRecord(e1, nullptr), "rec");
                    require_hip(hipEventSynchronize(e1), "sync");
                    float t = 0.0F;
                    require_hip(hipEventElapsedTime(&t, e0, e1), "el");
                    if (rep >= kVariants) ms.push_back(t);  // first round warms up
                }
                std::sort(ms.begin(), ms.end());
                return double(ms[ms.size() / 2U]);
            };
            const double t_r1 = time([&](const Plan&) { run_r1<F>(Fm, d, rows); });
            const double t_tile = time([&](const Plan& p) { run_tiles<F>(p, d, grouped::Form::bf16_tile); });
            const double t_small = time([&](const Plan& p) { run_tiles<F>(p, d, grouped::Form::small_f16); });
            const double bytes = distinct * Fm.expert_bytes;
            std::cout << name << " rows " << rows << (skewed ? " skewed " : " uniform") << " U " << distinct
                      << " ms r1 " << t_r1 << " tile " << t_tile << " small " << t_small << " | GB/s r1 "
                      << bytes / t_r1 * 1e-6 << " tile " << bytes / t_tile * 1e-6 << " small " << bytes / t_small * 1e-6
                      << " | small/r1 " << t_small / t_r1 << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench_only = argc > 1 && std::string(argv[1]) == "--bench-only";
    std::mt19937 rng(20261123U);
    const auto dev = make_device();
    bool ok = true;
    {
        auto nv = make_format(Format::nvfp4, rng);
        if (!bench_only) ok = check<Format::nvfp4>(nv, dev, rng) && ok;
        bench<Format::nvfp4>(nv, dev, rng);
    }
    {
        auto mx = make_format(Format::mxfp8, rng);
        if (!bench_only) ok = check<Format::mxfp8>(mx, dev, rng) && ok;
        bench<Format::mxfp8>(mx, dev, rng);
    }
    std::cout << (ok ? "ALL PASS" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
