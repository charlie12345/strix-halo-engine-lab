// Included by mmx_prefill.hip inside namespace strix::minimax::moe::opt (Plan, Tables, PGroup, InputTransform,
// Epilogue, ok): FAST_I4 prefill MoE (identity minimax-fast-i4, lossy, opt-in) on the strix-trellis grouped IU4 core
// (wmma_tiled.hpp launch_wmma_tiled_iu4_grouped, W4A4 g64) for every routed expert and the shared expert (index 128).
//   input:   gather_transform_token_kernel's values (suh + had128, FP16), quantised in the same kernel (no FP16 rows,
//            no separate quantiser pass): rows 2 p + m of aq_gu / as_gu (core::iu4_quant128).
//   gate/up: two grouped IU4 launches, fused output transform (had128 x svh) -> FP32 [P][2][1536].
//   down:    SwiGLU + down suh + had128 + A4 quantiser in one kernel -> grouped IU4 with the fused output transform
//            -> FP32 [P][6144]; combine = down_combine_kernel's order (ascending expert id, then + shared).
// Launches per call: gather+quant, tile list, gate, up, act+quant, down, combine (7, no host sync).
namespace {
int g_iu4_override = -1;

__global__ __launch_bounds__(256) void iu4_gather_kernel(const float* y, Tables t, std::uint8_t* aq, float* as) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    const float* x = y + static_cast<std::size_t>(tok) * kHidden;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = x[b * 128U + lane + 32U * static_cast<unsigned>(j)];
#pragma unroll 1
    for (unsigned k = 0; k <= kTopE; ++k) {
        const unsigned p = k < kTopE ? t.tok_pair[tok * kTopE + k] : t.groups[t.shared_group].pair_begin + tok;
        const PGroup& g = t.groups[t.pair_group[p]];
#pragma unroll
        for (unsigned m = 0; m < 2; ++m) {
            const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                v[j] = static_cast<float>(static_cast<_Float16>(InputTransform::scale(xv[j], suh[c])));
            }
            InputTransform::h128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = static_cast<float>(static_cast<_Float16>(v[j]));
            const std::size_t row = std::size_t{p} * 2U + m;
            strix::core::iu4_quant128(v, aq + row * (kHidden / 2U), as + row * (kHidden / 64U), b * 128U, lane);
        }
    }
}

// gu = [P][2][1536] FP32 (finished gate, up); one wave = one (pair, 128-block): gate_up_tiled_store's values.
__global__ __launch_bounds__(256) void iu4_act_kernel(Tables t, const float* gu, std::uint8_t* aq, float* as) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * kIB) return;
    const unsigned p = q / kIB, b = q % kIB;
    const PGroup& g = t.groups[t.pair_group[p]];
    const float* gr = gu + std::size_t{p} * 2U * kInter;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        const float act = Epilogue::swiglu(gr[c], gr[kInter + c]);
        v[j] = static_cast<float>(static_cast<_Float16>(InputTransform::scale(act, g.w.down.suh[c])));
    }
    InputTransform::h128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = static_cast<float>(static_cast<_Float16>(v[j]));
    if (aq != nullptr) {
        strix::core::iu4_quant128(v, aq + std::size_t{p} * (kInter / 2U), as + std::size_t{p} * (kInter / 64U), b * 128U, lane);
    } else {  // gate_up scope: the existing row-major EXL3 down operand, without A4
#pragma unroll
        for (int j = 0; j < 4; ++j)
            t.at[std::size_t{p} * kInter + b * 128U + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
    }
}

// dn = [P][6144] FP32 finished down rows; down_combine_kernel's accumulation order.
__global__ __launch_bounds__(256) void iu4_combine_kernel(Tables t, const float* dn, float* out) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned k = 0; k < kTopE; ++k) {
        const float* r = dn + std::size_t{t.tok_pair[tok * kTopE + k]} * kHidden + b * 128U + lane;
        const float w = t.tok_w[tok * kTopE + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = Epilogue::router_accumulate(w, r[32U * static_cast<unsigned>(j)], acc[j]);
    }
    const float* s = dn + std::size_t{t.groups[t.shared_group].pair_begin + tok} * kHidden + b * 128U + lane;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[static_cast<std::size_t>(tok) * kHidden + b * 128 + lane + 32U * static_cast<unsigned>(j)] =
            Epilogue::shared_add(acc[j], s[32U * static_cast<unsigned>(j)]);
}

// STRIX_MMX_IU4_PLAN: library IU4 plan 0 (BM32), 1 (BM64), 2 (BM128); default by the mean group size of the call:
// plan 2 when the (expert, token) pairs average >= 128 rows per expert (one 128-row tile per group), else plan 1.
int iu4_plan(std::size_t pairs) {
    const char* p = std::getenv("STRIX_MMX_IU4_PLAN");
    if (p == nullptr || *p == '\0') return pairs >= std::size_t{128} * kI4Experts ? 2 : 1;
    if (std::strcmp(p, "0") == 0) return 0;
    if (std::strcmp(p, "1") == 0) return 1;
    if (std::strcmp(p, "2") == 0) return 2;
    throw std::invalid_argument("STRIX_MMX_IU4_PLAN must be 0, 1 or 2");
}

constexpr std::size_t kI4W = kI4Nibbles;                                   // nibble bytes per expert matrix
constexpr std::size_t kI4S = std::size_t{kInter} * (kHidden / 64U);      // FP16 scales per expert matrix (both shapes)
constexpr std::size_t kI4Tab = (kI4Experts * sizeof(void*) + 255U) & ~std::size_t{255};
static_assert(std::size_t{kHidden} * (kInter / 64U) == kI4S && kI4ExpertBytes == kI4W + 2U * kI4S && kI4ExpertBytes % 256U == 0U,
              "gate/up and down share one per-expert block: nibbles, then FP16 scales");

#if STRIX_MMX_IU4_LIB
void iu4_ok(hipError_t e, const char* what) { ok(e, what); }


void launch_iu4(const Plan& p, const Tables& t, const PrefillMoeView& v, unsigned char* ws, std::size_t ws_bytes, hipStream_t stream) {
    namespace st = strix::core;
    const I4Layer& q = *v.i4;
    const std::size_t P = t.pairs;
    const int plan = iu4_plan(P);
    const auto max_tiles = static_cast<std::uint32_t>(st::iu4_grouped_max_tiles(P, static_cast<int>(kI4Experts), plan));
    std::size_t off = p.off_xt;
    auto take = [&](std::size_t bytes) {
        const std::size_t at = off;
        off += align256(bytes);
        return ws + at;
    };
    auto* tiles = reinterpret_cast<st::IU4GroupTile*>(take(std::size_t{max_tiles} * sizeof(st::IU4GroupTile)));
    auto* ntiles = reinterpret_cast<std::uint32_t*>(take(4));
    auto* aq_gu = take(2 * P * (kHidden / 2U));
    auto* as_gu = reinterpret_cast<float*>(take(2 * P * (kHidden / 64U) * 4U));
    auto* gu = reinterpret_cast<float*>(take(2 * P * kInter * 4U));
    auto* aq_dn = v.i4_all ? take(P * (kInter / 2U)) : nullptr;
    auto* as_dn = v.i4_all ? reinterpret_cast<float*>(take(P * (kInter / 64U) * 4U)) : nullptr;
    auto* dn = v.i4_all ? reinterpret_cast<float*>(take(P * kHidden * 4U)) : nullptr;
    if (off > ws_bytes) throw std::logic_error("minimax prefill iu4: workspace smaller than the call");
    if (!v.i4_all && off > p.off_at) throw std::logic_error("minimax prefill iu4: gate/up overlaps EXL3 down workspace");
    hipLaunchKernelGGL(iu4_gather_kernel, dim3((t.rows * kHB + 7) / 8), dim3(256), 0, stream, v.y, t, aq_gu, as_gu);
    launched("iu4 gather + quantiser");
    mark("gather", stream);
    iu4_ok(st::build_iu4_grouped_tiles(reinterpret_cast<const st::IU4Group*>(ws + p.off_i4_groups), static_cast<int>(kI4Experts), plan, tiles,
                                       ntiles, max_tiles, stream), "iu4 tile list");
    for (unsigned m = 0; m < 2U; ++m) {
        st::IU4GroupedOperands o;
        o.a_q = aq_gu; o.a_scale = as_gu; o.w_q = q.wq[m]; o.w_scale = q.ws[m];
        o.w_expert_bytes = kI4ExpertBytes; o.w_scale_expert = kI4ExpertBytes / 2U;
        o.col_scale = q.svh[m]; o.map = nullptr; o.a_mul = 2U; o.a_add = m; o.o_mul = 2U; o.o_add = m;
        iu4_ok(st::launch_wmma_tiled_iu4_grouped(o, tiles, ntiles, max_tiles, gu, static_cast<int>(kHidden), static_cast<int>(kInter), stream, plan),
               "iu4 gate/up");
    }
    mark("gate_up", stream);
    hipLaunchKernelGGL(iu4_act_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t, gu, aq_dn, as_dn);
    launched("iu4 swiglu + down transform + quantiser");
    mark("finish", stream);
    if (!v.i4_all) {
        // Reuse the existing EXL3 grouped down and ordered finish/combine. No
        // second down implementation or decode-format policy is introduced.
        const auto* items = reinterpret_cast<const WItem*>(ws + p.off_items);
        const auto n16 = static_cast<unsigned>(p.dn_items16.size());
        const auto n64 = static_cast<unsigned>(p.dn_items64.size());
        if (n16 != 0) hipLaunchKernelGGL(wmma_grouped_kernel<16>, dim3(p.dn_ctas16), dim3(128), 0, stream, items, n16);
        if (n64 != 0) {
            if (prefill_tile() == 128) hipLaunchKernelGGL(wmma_grouped_kernel<128>, dim3(p.dn_ctas64), dim3(128), 0, stream, items + n16, n64);
            else hipLaunchKernelGGL(wmma_grouped_kernel<64>, dim3(p.dn_ctas64), dim3(128), 0, stream, items + n16, n64);
        }
        launched("iu4 gate/up EXL3 down");
        mark("down", stream);
        hipLaunchKernelGGL(down_combine_kernel, dim3((t.rows * kHB + 7) / 8), dim3(256), 0, stream, t, v.out);
        launched("iu4 gate/up EXL3 combine");
        mark("combine", stream);
        return;
    }
    st::IU4GroupedOperands o;
    o.a_q = aq_dn; o.a_scale = as_dn; o.w_q = q.wq[2]; o.w_scale = q.ws[2];
    o.w_expert_bytes = kI4ExpertBytes; o.w_scale_expert = kI4ExpertBytes / 2U;
    o.col_scale = q.svh[2]; o.map = nullptr;
    iu4_ok(st::launch_wmma_tiled_iu4_grouped(o, tiles, ntiles, max_tiles, dn, static_cast<int>(kInter), static_cast<int>(kHidden), stream, plan),
           "iu4 down");
    mark("down", stream);
    hipLaunchKernelGGL(iu4_combine_kernel, dim3((t.rows * kHB + 7) / 8), dim3(256), 0, stream, t, dn, v.out);
    launched("iu4 combine");
    mark("combine", stream);
}
#endif
}  // namespace

bool iu4_built() { return STRIX_MMX_IU4_LIB != 0; }
std::size_t i4_pack_bytes() { return kI4PackBytes; }
std::size_t i4_source_bytes() { return 3U * 10U * kI4Tab; }
std::size_t i4_slot_bytes() { return kI4PackBytes + 3U * kI4Tab; }
I4Layer i4_slot_view(std::uint8_t* base) {
    I4Layer out = i4_layer_view(base, {});
    for (unsigned m = 0; m < 3; ++m)
        out.svh[m] = reinterpret_cast<const std::uint16_t* const*>(base + kI4PackBytes + m * kI4Tab);
    return out;
}
I4Source i4_source_build(const ExpertSlices* experts, const ExpertSlices& shared, void* base) {
    I4Source src;
    auto* b = static_cast<std::uint8_t*>(base);
    for (unsigned m = 0; m < 3; ++m) {
        const unsigned K = m < 2 ? kHidden : kInter, N = m < 2 ? kInter : kHidden;
        std::vector<const void*> svh(kI4Experts);
        unsigned seen = 0;
        for (unsigned e = 0; e < kI4Experts; ++e) {
            const Exl3& w = (&(e < kExperts ? experts[e] : shared).gate)[m];
            if (w.input != K || w.output != N || w.bits < 2 || w.bits > 8 || w.trellis == nullptr || w.svh == nullptr || w.suh == nullptr)
                throw std::invalid_argument("fast_i4: expert " + std::to_string(e) + " matrix " + std::to_string(m) +
                                            " is not a resident 2..8-bit EXL3 " + std::to_string(K) + "x" + std::to_string(N));
            seen |= 1U << w.bits;
            svh[e] = w.svh;
        }
        src.svh[m] = reinterpret_cast<const std::uint16_t**>(b);
        ok(hipMemcpy(b, svh.data(), kI4Experts * sizeof(void*), hipMemcpyHostToDevice), "fast_i4 svh table");
        b += kI4Tab;
        for (unsigned bits = 2; bits <= 8; ++bits) {
            if (!(seen & (1U << bits))) { b += kI4Tab; continue; }
            std::vector<const void*> tr(kI4Experts, nullptr);
            for (unsigned e = 0; e < kI4Experts; ++e) {
                const Exl3& w = (&(e < kExperts ? experts[e] : shared).gate)[m];
                if (w.bits == bits) tr[e] = w.trellis;
            }
            src.trellis[m][bits] = reinterpret_cast<const std::uint32_t* const*>(b);
            ok(hipMemcpy(b, tr.data(), kI4Experts * sizeof(void*), hipMemcpyHostToDevice), "fast_i4 trellis table");
            b += kI4Tab;
        }
        b += kI4Tab;  // 10 tables per matrix (svh + widths 2..8 + spare)
    }
    return src;
}
I4Layer i4_layer_view(std::uint8_t* pack, const I4Source& src) {
    I4Layer out;
    for (unsigned m = 0; m < 3; ++m) {
        out.wq[m] = pack + std::size_t{m} * kI4Experts * kI4ExpertBytes;
        out.ws[m] = reinterpret_cast<_Float16*>(out.wq[m] + kI4W);
        out.svh[m] = src.svh[m];
    }
    return out;
}
void i4_convert(const I4Source& src, const I4Layer& out, hipStream_t stream, bool all) {
#if STRIX_MMX_IU4_LIB
    for (unsigned m = 0; m < (all ? 3U : 2U); ++m) {
        if (out.svh[m] != src.svh[m])
            iu4_ok(hipMemcpyAsync(const_cast<const std::uint16_t**>(out.svh[m]), src.svh[m],
                                 kI4Experts * sizeof(void*), hipMemcpyDeviceToDevice, stream), "fast_i4 slot svh");
        const unsigned K = m < 2 ? kHidden : kInter, N = m < 2 ? kInter : kHidden;
        for (unsigned bits = 2; bits <= 8; ++bits) {  // the shared core converter (strix::core::convert_exl3_iu4), one per width
            const auto* td = src.trellis[m][bits];
            if (td == nullptr) continue;
            iu4_ok(strix::core::convert_exl3_iu4(td, static_cast<int>(bits), static_cast<int>(kI4Experts), static_cast<int>(K),
                                                   static_cast<int>(N), out.wq[m], out.ws[m], kI4ExpertBytes, kI4ExpertBytes / 2U, stream),
                   "fast_i4 convert");
        }
    }
#else
    (void)src, (void)out, (void)stream, (void)all;
    throw std::invalid_argument("minimax fast_i4: this build has no IU4 core (needs the HIP runtime build)");
#endif
}
void set_prefill_iu4(int on) { g_iu4_override = on; }
bool prefill_iu4() {
    if (g_iu4_override >= 0) return g_iu4_override != 0;
    const char* p = std::getenv("STRIX_MMX_PF_IU4");
    if (p == nullptr || *p == '\0' || std::strcmp(p, "1") == 0) return true;
    if (std::strcmp(p, "0") == 0) return false;
    throw std::invalid_argument("STRIX_MMX_PF_IU4 must be 0 or 1");
}
bool prefill_fakeq_a4() {
    const char* p = std::getenv("STRIX_MMX_FAKEQ_A4");
    if (p == nullptr || *p == '\0' || std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_MMX_FAKEQ_A4 must be 0 or 1");
}
