// Kernel bench of the shared dense flash prefill core (src/attn_core/dense_flash_prefill.hpp) on the shapes the engines
// launch, synthetic data, no model. Per case: one warm-up launch, then REPS launches timed with hip events (median ms),
// TFLOP/s of the visited work (QK + PV over the visible keys) and a 64-bit FNV digest of the output bits (two builds of the
// core with the same digest per case produce the same bits per row).
//   mha      MHA sliding form: D 256, bf16, one head x 64 positions per work-group, 64 heads, window 512 (Hy4-flash shape)
//   g4s      Gemma4 E4B sliding layer form: D 256, f16, GQA group 4 (8 waves), window 512
//   g4g      Gemma4 E4B global layer form: D 512, f16, GQA group 4 (16 waves), full causal
//   qw       Qwen3.5 27B wide-window form (dense_decode_flash.hpp <256, 6>): D 256, bf16, group 6 (12 waves)
//   v80 v64  vision encoder form: D 80 / 64, f16 two-term (hi + lo) operands, 64 positions, 1024 rows from position
//            4095 (each row sees 4096-5119 keys, the size of a bidirectional encoder row)
//   ikl ikf  one-slice lane-score form: D 128, bf16, GQA group 3 (12 waves), 64 positions, 8192 rows from position 0,
//            window 1024 (ikl) / full causal (ikf)
//   ibl ibf  the same with a relative bias projected inside the flash (width 16, extent 1024)
// Usage: strix-dense-flash-bench [REPS] [CASE]   (CASE: one name above; default all)
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "attn_core/dense_flash_prefill.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace dense = strix::attn_core::dense;
namespace tile = strix::attn_core::tile;

struct BArgs {
    const float* q;
    float* out;
    const std::uint16_t *k, *vt;
    unsigned* failure;
    unsigned rows, p0, k0, window, vstride, heads, kv_heads;
    const float* feat = nullptr;          // projected relative bias (Rel policies only): [row][head][16] FP32
    const std::uint16_t* rw = nullptr;    // [16][extent] BF16
    unsigned extent = 0U;
};
template <unsigned D, unsigned G, unsigned Pos, tile::Operand Op>
struct Common {
    using Args = BArgs;
    using Shape = dense::Shape<D, G, (D >= 128U ? D / 128U : 1U), Pos>;
    static constexpr float kScaleLog2 = (D == 512U ? 0.044194173824159220F : 0.0625F) * 1.4426950408889634F;
    static constexpr tile::Operand kOp = Op;
    __device__ static bool valid(const Args& a) { return a.rows != 0U && a.p0 >= a.k0; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.p0; }
    __device__ static unsigned window(const Args& a) { return a.window; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) { return a.q + (std::size_t(row) * a.heads + h) * D; }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) { return a.out + (std::size_t(row) * a.heads + h) * D; }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};
template <unsigned D, unsigned G, unsigned Pos, tile::Operand Op>
struct Classic : Common<D, G, Pos, Op> {
    using Args = BArgs;
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * a.kv_heads + h) * D; }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * D + d) * a.vstride - a.k0; }
};
template <unsigned D>  // two-term operands (lo terms = the same arrays: timing only)
struct Classic2 : Classic<D, 1U, 64U, tile::Operand::f16> {
    using Args = BArgs;
    using Shape = dense::Shape<D, 1U, 1U, 64U>;
    static constexpr unsigned kTerms = 2U;
    __device__ static const std::uint16_t* key_lo(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * a.kv_heads + h) * D; }
    __device__ static const std::uint16_t* value_t_lo(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * D + d) * a.vstride - a.k0; }
};
template <unsigned D, unsigned G>  // one-slice form with the relative bias projected in the flash
struct Rel : Classic<D, G, 64U, tile::Operand::bf16> {
    using Args = BArgs;
    static constexpr unsigned kBiasWidth = 16U, kBiasMaxExtent = 1024U;
    __device__ static strix::attn_core::RelativeBiasProjection bias_projection(const Args& a) {
        return {a.feat, a.rw, a.heads, a.extent};
    }
};
constexpr unsigned kN = 192U;  // MHA case: own columns per head before the shared 64

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint16_t round_op(float x, bool f16) {
    if (f16) {
        const _Float16 h = static_cast<_Float16>(x);
        std::uint16_t b;
        std::memcpy(&b, &h, 2);
        return b;
    }
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ok(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

// Exceptional operand values (review r17: bit identity of the lane-score form against the incumbent kernel):
//   nan       V holds NaN operands of several payloads, quiet and signalling encodings, both signs;
//   overflow  the first key tile's V is the largest finite operand (the FP32 PV sum overflows to Inf), a later key
//             dominates the scores (alpha underflows to 0: Inf * 0 = NaN accumulators) and the following tiles keep
//             that maximum (alpha exactly 1 across the wave: the shortcut's case);
//   tiny      V holds operands whose products are FP32 subnormals, and signed zeros.
// Ordinary cases (none) must leave the failure word 0; exceptional cases report it for the cross-build comparison.
enum class Special : unsigned { none, nan, overflow, tiny };

struct Case {
    const char* name;
    unsigned D, heads, kv_heads, rows, p0, window;
    bool f16;
    Special special = Special::none;
};

bool g_flag_failed = false;

void apply_special(const Case& c, unsigned keys, unsigned KH, std::vector<std::uint16_t>& kfull,
                   std::vector<std::uint16_t>& vrows) {
    const unsigned D = c.D;
    const auto v_at = [&](unsigned p, unsigned h, unsigned d) -> std::uint16_t& {
        return vrows[(std::size_t(p) * KH + h) * D + d];
    };
    if (c.special == Special::nan) {
        const std::uint16_t bf[5] = {0x7FC1U, 0x7F81U, 0xFFC5U, 0xFF8AU, 0x7FFFU};  // quiet, signalling, both signs
        const std::uint16_t hf[5] = {0x7E01U, 0x7C01U, 0xFE05U, 0xFC0AU, 0x7FFFU};
        for (unsigned p = 5U; p < keys; p += 97U)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 3U; d < D; d += 7U) v_at(p, h, d) = (c.f16 ? hf : bf)[(p + d) % 5U];
    } else if (c.special == Special::overflow) {
        const std::uint16_t big = c.f16 ? std::uint16_t{0x7BFFU} : std::uint16_t{0x7F7FU};
        const std::uint16_t nbig = static_cast<std::uint16_t>(big | 0x8000U);
        for (unsigned p = 0; p < 16U && p < keys; ++p)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 0; d < D; ++d) v_at(p, h, d) = d % 3U == 2U ? nbig : big;
        const std::uint16_t dom = round_op(30.0F, c.f16);  // a dominating key at position 40
        if (keys > 40U)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 0; d < D; ++d) kfull[(std::size_t(40U) * KH + h) * D + d] = dom;
    } else if (c.special == Special::tiny) {
        for (unsigned p = 0; p < keys; ++p)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 0; d < D; ++d)
                    if ((p + 3U * d) % 5U == 0U)
                        v_at(p, h, d) = (p + d) % 4U == 0U ? std::uint16_t{0x8000U}
                                                           : static_cast<std::uint16_t>(1U + (p * 7U + d) % 127U);
    }
}

template <class P> void run(const Case& c, unsigned reps) {
    const unsigned D = c.D, H = c.heads, KH = c.kv_heads;
    const unsigned k0 = (c.window && c.p0 >= c.window ? c.p0 - c.window + 1U : 0U) & ~15U, keys = c.p0 + c.rows - k0 + 16U, vstride = keys + 16U;
    std::mt19937 g(c.rows * 131U + c.p0 + D);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<float> q(std::size_t(c.rows) * H * D);
    for (auto& x : q) x = u(g);
    std::vector<std::uint16_t> kfull(std::size_t(keys) * KH * D), vt(std::size_t(KH) * D * vstride, 0U);
    for (auto& x : kfull) x = round_op(u(g), c.f16);
    std::vector<std::uint16_t> vrows(std::size_t(keys) * KH * D);
    for (auto& x : vrows) x = round_op(u(g), c.f16);
    apply_special(c, keys, KH, kfull, vrows);
    for (unsigned p = 0; p < keys; ++p)
        for (unsigned h = 0; h < KH; ++h)
            for (unsigned d = 0; d < D; ++d) vt[std::size_t(h * D + d) * vstride + p] = vrows[(std::size_t(p) * KH + h) * D + d];
    if (D == 256U && KH == H)  // MHA case: columns 192..255 of every head = head 0's (the shared rope key of an MLA form)
        for (unsigned p = 0; p < keys; ++p)
            for (unsigned h = 1; h < H; ++h)
                for (unsigned d = kN; d < D; ++d) kfull[(std::size_t(p) * KH + h) * D + d] = kfull[std::size_t(p) * KH * D + d];
    BArgs a{};
    a.q = up(q);
    a.k = up(kfull);
    a.vt = up(vt);
    a.failure = up(std::vector<unsigned>{0U});
    a.rows = c.rows;
    a.p0 = c.p0;
    a.k0 = k0;
    a.window = c.window;
    a.vstride = vstride;
    a.heads = H;
    a.kv_heads = KH;
    if constexpr (dense::DenseProjectedBias<P>) {
        std::vector<float> feat(std::size_t(c.rows) * H * 16U);
        for (auto& x : feat) x = 0.25F * u(g);
        std::vector<std::uint16_t> rw(16U * 1024U);
        for (auto& x : rw) x = round_op(0.25F * u(g), false);
        a.feat = up(feat);
        a.rw = up(rw);
        a.extent = 1024U;
    }
    const std::size_t n = std::size_t(c.rows) * H * D;
    float* out = nullptr;
    ok(hipMalloc(&out, n * 4U), "malloc out");
    a.out = out;
    hipEvent_t e0, e1;
    ok(hipEventCreate(&e0), "event");
    ok(hipEventCreate(&e1), "event");
    dense::launch<P>(a, H, nullptr);
    ok(hipDeviceSynchronize(), "warm-up");
    std::vector<float> ms(reps);
    for (unsigned r = 0; r < reps; ++r) {
        ok(hipEventRecord(e0, nullptr), "record");
        dense::launch<P>(a, H, nullptr);
        ok(hipEventRecord(e1, nullptr), "record");
        ok(hipEventSynchronize(e1), "sync");
        ok(hipEventElapsedTime(&ms[r], e0, e1), "elapsed");
    }
    std::vector<float> host(n);
    ok(hipMemcpy(host.data(), out, n * 4U, hipMemcpyDeviceToHost), "download");
    unsigned failure = 0U;
    ok(hipMemcpy(&failure, a.failure, 4U, hipMemcpyDeviceToHost), "failure");
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const float x : host) {
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        for (unsigned i = 0; i < 4U; ++i) h = (h ^ ((b >> (8U * i)) & 0xffU)) * 0x100000001b3ULL;
    }
    double work = 0.0;  // visible (row, key) pairs x heads x D x 2 products x 2 flop
    for (unsigned r = 0; r < c.rows; ++r) {
        const unsigned pos = c.p0 + r, lo = c.window && pos + 1U >= c.window ? pos + 1U - c.window : 0U;
        work += double(pos - lo + 1U);
    }
    work *= double(H) * D * 4.0;
    std::sort(ms.begin(), ms.end());
    const float med = ms[reps / 2U];
    std::printf("case %-5s D %u heads %u/%u rows %u p0 %u window %u: median %.3f ms min %.3f max %.3f  %.2f TF/s  digest %016llx failure %u\n",
                c.name, D, H, KH, c.rows, c.p0, c.window, double(med), double(ms.front()), double(ms.back()), work / (double(med) * 1e9),
                static_cast<unsigned long long>(h), failure);
    if (c.special == Special::none && failure != 0U) {
        std::printf("FAIL flag case %s: failure %u on ordinary data\n", c.name, failure);
        g_flag_failed = true;
    }
    for (const void* p : {static_cast<const void*>(a.q), static_cast<const void*>(a.k), static_cast<const void*>(a.vt),
                          static_cast<const void*>(a.failure), static_cast<const void*>(out)})
        if (p) ok(hipFree(const_cast<void*>(p)), "free");
    ok(hipEventDestroy(e0), "event");
    ok(hipEventDestroy(e1), "event");
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned reps = argc > 1 ? unsigned(std::atoi(argv[1])) : 7U;
    const std::string only = argc > 2 ? argv[2] : "";
    const auto want = [&](const char* n) { return only.empty() || only == n; };
    try {
        // MHA sliding layer-window: 2048 rows, 64 heads, window 512, the halo keys before the rows.
        if (want("mha")) run<Classic<256U, 1U, 64U, tile::Operand::bf16>>({"mha", 256U, 64U, 64U, 2048U, 2048U, 512U, false}, reps);
        if (want("g4s")) run<Classic<256U, 4U, 16U, tile::Operand::f16>>({"g4s", 256U, 8U, 2U, 2048U, 2048U, 512U, true}, reps);
        if (want("g4g")) run<Classic<512U, 4U, 16U, tile::Operand::f16>>({"g4g", 512U, 8U, 2U, 2048U, 2048U, 0U, true}, reps);
        if (want("qw")) run<Classic<256U, 6U, 16U, tile::Operand::bf16>>({"qw", 256U, 24U, 4U, 2048U, 2048U, 0U, false}, reps);
        if (want("v80")) run<Classic2<80U>>({"v80", 80U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("v64")) run<Classic2<64U>>({"v64", 64U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("ikl")) run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl", 128U, 24U, 8U, 8192U, 0U, 1024U, false}, reps);
        if (want("ikf")) run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf", 128U, 24U, 8U, 8192U, 0U, 0U, false}, reps);
        if (want("ibl")) run<Rel<128U, 3U>>({"ibl", 128U, 24U, 8U, 8192U, 0U, 1024U, false}, reps);
        if (want("ibf")) run<Rel<128U, 3U>>({"ibf", 128U, 24U, 8U, 8192U, 0U, 0U, false}, reps);
        // One-term FP16 vision encoder forms (the FAST route's lane-score body; bidirectional: every row sees every key).
        if (want("o80")) run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80", 80U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("o64")) run<Classic<64U, 1U, 64U, tile::Operand::f16>>({"o64", 64U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        // Exceptional values on the lane-score forms (BF16 Inkling local / full / projected bias, FP16 one-term vision)
        // and on the two-term vision comparison form.
        using S = Special;
        if (want("xn")) {
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl-n", 128U, 24U, 8U, 2048U, 0U, 1024U, false, S::nan}, reps);
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf-n", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::nan}, reps);
            run<Rel<128U, 3U>>({"ibf-n", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::nan}, reps);
            run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80-n", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::nan}, reps);
            run<Classic2<80U>>({"v80-n", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::nan}, reps);
        }
        if (want("xo")) {
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl-o", 128U, 24U, 8U, 2048U, 0U, 1024U, false, S::overflow}, reps);
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf-o", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::overflow}, reps);
            run<Rel<128U, 3U>>({"ibf-o", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::overflow}, reps);
            run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80-o", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::overflow}, reps);
            run<Classic2<80U>>({"v80-o", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::overflow}, reps);
        }
        if (want("xt")) {
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl-t", 128U, 24U, 8U, 2048U, 0U, 1024U, false, S::tiny}, reps);
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf-t", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::tiny}, reps);
            run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80-t", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::tiny}, reps);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    return g_flag_failed ? 1 : 0;
}
