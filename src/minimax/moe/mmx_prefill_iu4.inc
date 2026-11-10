// Included by mmx_prefill.hip inside namespace strix::minimax::moe::opt (Plan, Tables, PGroup, InputTransform,
// Epilogue, ok): FAST_I4 prefill MoE (identity minimax-fast-i4, lossy, opt-in) on the strix-trellis grouped IU4 core
// (wmma_tiled.hpp launch_wmma_tiled_iu4_grouped, W4A4 g64) for every routed expert and the shared expert (index 128).
// STRIX_MOE_I4_CHAIN=1: device A-load gather/H128/A4; paired gate/up with SwiGLU/down-H128/A4
// epilogue; down epilogue accumulates weighted FP32 directly. No global GU or down FP32 image.
// OFF: the unfused control uses the same shared arithmetic with materialized intermediates.
// All FP16/A4 boundaries stay fixed; the candidate combine uses product RN32 + atomic addition.
namespace {
int g_iu4_override = -1;

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


struct I4Binding {
    static constexpr unsigned H=kHidden, I=kInter;
    const float* x; float* out; Tables t; bool dense_groups=true; bool reference_group_ids=false;
    __device__ unsigned slot(unsigned tok,unsigned k) const {
        return k<kTopE ? t.tok_pair[tok*kTopE+k] : t.groups[t.shared_group].pair_begin+tok;
    }
    __device__ const unsigned* map() const { return nullptr; }
    __device__ unsigned pair(unsigned pos) const { return pos; }
    __device__ unsigned token(unsigned p) const { return t.pair_row[p]; }
    __device__ float weight(unsigned p) const {
        const unsigned tok=token(p);
        if (t.pair_group[p]==t.shared_group) return 1.f;
        for (unsigned k=0;k<kTopE;++k) if(t.tok_pair[tok*kTopE+k]==p) return t.tok_w[tok*kTopE+k];
        return 0.f;
    }
    __device__ float suh(unsigned e,unsigned m,unsigned c) const {
        unsigned gi=e;
        if(!dense_groups && !reference_group_ids) { gi=0; while(t.groups[gi].expert!=e) ++gi; }
        const auto& w=t.groups[gi].w;
        return strix::moe_core::half_bits((m==0 ? w.gate.suh : m==1 ? w.up.suh : w.down.suh)[c]);
    }
    __device__ float activation(float g,float u) const { return Epilogue::swiglu(g,u); }
};
struct I4ReferenceSchedule {
    using Binding=I4Binding;
    static constexpr bool kCachedInput=true;
    Binding b; uint8_t* aq; float* as; const float* gu=nullptr;
    __device__ unsigned source_pair(unsigned row) const { return row/2; }
    __device__ bool input(unsigned i,strix::core::moe_i4::ReferenceRow& r) const {
        const unsigned q=blockIdx.x*8+threadIdx.x/32;
        if(q>=b.t.rows*kHB || i>=2*(kTopE+1)) return false;
        const unsigned tok=q/kHB,k=i/2,m=i%2;
        const unsigned p=k<kTopE ? b.t.tok_pair[tok*kTopE+k] : b.t.groups[b.t.shared_group].pair_begin+tok;
        r={b.t.pair_group[p],2*p+m,m,(q%kHB)*128,kHidden,nullptr,aq,as,nullptr};return true;
    }
    __device__ bool activation(unsigned i,strix::core::moe_i4::ReferenceRow& r) const {
        const unsigned q=blockIdx.x*8+threadIdx.x/32;
        if(i || q>=b.t.pairs*kIB) return false;
        const unsigned p=q/kIB;
        r={b.t.pair_group[p],p,2,(q%kIB)*128,kInter,gu,aq,as,b.t.at};return true;
    }
};

void launch_i4_chain(const Tables& t,const I4Layer& q,const float* x,float* out,
    const strix::core::IU4Group* groups,unsigned char* ws,std::size_t bytes,hipStream_t stream,bool dense_groups=true) {
    namespace st=strix::core;
    const int plan=iu4_plan(t.pairs);
    const auto max_tiles=static_cast<unsigned>(st::iu4_grouped_max_tiles(t.pairs,kI4Experts,plan));
    const std::size_t tile_bytes=align256(max_tiles*sizeof(st::IU4GroupTile));
    const std::size_t qbytes=align256(std::size_t(t.pairs)*(kInter/2));
    const std::size_t sbytes=align256(std::size_t(t.pairs)*(kInter/64)*4);
    if(bytes<tile_bytes+256+qbytes+sbytes) throw std::logic_error("minimax fused i4 scratch too small");
    auto* tiles=reinterpret_cast<st::IU4GroupTile*>(ws);
    auto* ntiles=reinterpret_cast<unsigned*>(ws+tile_bytes);
    auto* aq=ws+tile_bytes+256;
    auto* as=reinterpret_cast<float*>(aq+qbytes);
    st::moe_i4::Weights w{};
    for(unsigned m=0;m<3;++m) {
        w.q[m]=q.wq[m]; w.s[m]=q.ws[m]; w.svh[m]=q.svh[m];
        w.q_stride[m]=kI4ExpertBytes; w.s_stride[m]=kI4ExpertBytes/2;
    }
    I4Binding b{x,out,t,dense_groups};
    iu4_ok(st::build_iu4_grouped_tiles(groups,kI4Experts,plan,tiles,ntiles,max_tiles,stream),"chain tiles");
    iu4_ok(st::moe_i4::launch<true>(plan,b,w,tiles,ntiles,max_tiles,aq,as,stream),"chain paired gate/up");
    mark("gate_up",stream);
    iu4_ok(hipMemsetAsync(out,0,std::size_t(t.rows)*kHidden*4,stream),"chain output init");
    iu4_ok(st::moe_i4::launch<false>(plan,b,w,tiles,ntiles,max_tiles,aq,as,stream),"chain down/combine");
    mark("combine",stream);
}

void launch_iu4(const Plan& p, const Tables& t, const PrefillMoeView& v, unsigned char* ws, std::size_t ws_bytes, hipStream_t stream) {
    namespace st = strix::core;
    const I4Layer& q = *v.i4;
    if(v.i4_all && st::moe_i4::enabled()) {
        launch_i4_chain(t,q,v.y,v.out,reinterpret_cast<const st::IU4Group*>(ws+p.off_i4_groups),
            ws+p.off_xt,ws_bytes-p.off_xt,stream,false);
        return;
    }
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
    hipLaunchKernelGGL((st::moe_i4::reference_input_kernel<I4ReferenceSchedule>), dim3((t.rows*kHB+7)/8), dim3(256),0,stream, (I4ReferenceSchedule{{v.y,v.out,t,false,true},aq_gu,as_gu}));
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
    hipLaunchKernelGGL((st::moe_i4::reference_activation_kernel<I4ReferenceSchedule>),dim3((t.pairs*kIB+7)/8),dim3(256),0,stream,(I4ReferenceSchedule{{v.y,v.out,t,false,true},aq_dn,as_dn,gu}));
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
    hipLaunchKernelGGL((st::moe_i4::reference_combine_kernel<I4Binding>),dim3((t.rows*kHB+7)/8),dim3(256),0,stream,(I4Binding{v.y,v.out,t,false}),dn,t.rows,kTopE);
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
