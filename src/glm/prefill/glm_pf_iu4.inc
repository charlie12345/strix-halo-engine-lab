// Included inside glm_pf_exact_moe.hip's anonymous namespace (Args, Tile, E, H, I, P, ok, activation_kernel,
// combine_kernel, launch_linear_w): the FAST_I4 stored routed experts on the strix-trellis grouped
// IU4 (W4A4, g64) core. Lossy, a new identity on top of glm-pf-wmma-x; shared expert and every other linear unchanged.
//   weights: EXL3 tiles decoded with the engine's own mul1_pair (the FP16 values the WMMA-X cores multiply), then per
//            output column and K64 group s = FP16(max|w| / 7), q = RNE(w / s) in [-7, 7], nibbles in the trellis K16 x N16
//            tile order (strix::core::iu4_weight_byte), all in the had128-rotated basis (the EXL3 suh / svh stay).
//   activations: the transformed FP16 rows (input_kernel: suh + had128; activation_kernel: R15 + down suh + had128),
//            per row and K64 group (library quantiser), gathered through the plan's pair map.
//   epilogue: H128 times svh. STRIX_MOE_I4_CHAIN=1 binds the shared register-paired chain;
//            FP16/A4 boundaries stay fixed; the down epilogue writes the routed rows and the control's
//            ordered combine (aux or R16) sums them, so the chain is deterministic.
//            Shared expert remains on its existing WMMA-X arithmetic.
#if STRIX_GLM_PF_IU4_LIB
// STRIX_GLM_PF_IU4_PLAN: library IU4 plan 0 (BM32), 1 (BM64), 2 (BM128); unset = auto: plan 2 when the call averages
// >= min_rows rows per routed expert, else plan 1. The unfused control uses min_rows 128; the chain passes the
// shared moe_i4::kPlan2Rows policy (4096-row chunk calls average ~114 rows per expert and take plan 2).
int iu4_plan(unsigned pairs, unsigned min_rows = 128) {
    const char* p = std::getenv("STRIX_GLM_PF_IU4_PLAN");
    if (!p || !*p) return pairs >= min_rows * E ? 2 : 1;
    if (std::strcmp(p, "0") == 0) return 0;
    if (std::strcmp(p, "1") == 0) return 1;
    if (std::strcmp(p, "2") == 0) return 2;
    throw std::invalid_argument("STRIX_GLM_PF_IU4_PLAN must be 0, 1 or 2");
}
void iu4_ok(hipError_t e, const char* what) { ok(e, what); }
// STRIX_GLM_PF_IU4_FUSEQ (default 1): quantise in the input transform / activation kernels (routed rows write packed
// nibbles + scales instead of FP16 rows); 0 = FP16 rows then the library quantiser. Same bits either way: the fused
// form rounds each value to FP16 first and repeats the library quantiser's arithmetic (max|x|/7, RNE, [-7, 7]).
bool iu4_fuse_quant() {
    const char* p = std::getenv("STRIX_GLM_PF_IU4_FUSEQ");
    if (!p || !*p || std::strcmp(p, "1") == 0) return true;
    if (std::strcmp(p, "0") == 0) return false;
    throw std::invalid_argument("STRIX_GLM_PF_IU4_FUSEQ must be 0 or 1");
}
// Model binding only; the A-load and paired SwiGLU/A4 and down epilogues are shared.
struct I4Binding {
    static constexpr unsigned H=::strix::glm::pf::exact_moe::H, I=::strix::glm::pf::exact_moe::I;
    const float* x; float* out; Args a;
    __host__ __device__ __forceinline__ const unsigned* map() const { return a.pairs; }
    __device__ __forceinline__ unsigned pair(unsigned pos) const { return a.pairs[pos]; }
    __device__ __forceinline__ unsigned token(unsigned p) const { return p/P; }
    __device__ __forceinline__ float weight(unsigned p) const { return a.weights[(p/P)*8+p%P]; }
    __device__ __forceinline__ float* contribution(unsigned p) const { return a.output+size_t(p)*H; }
    __device__ __forceinline__ float suh(unsigned e,unsigned m,unsigned c) const { return d::half_bits(matrix(a,e,m).suh[c]); }
    __device__ __forceinline__ float activation(float g,float u) const {
        const float gg=g>kClamp ? kClamp : g, uu=u>kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
        return silu(gg)*uu;
    }
};
struct I4ReferenceSchedule {
    using Binding=I4Binding;
    static constexpr bool kCachedInput=false;
    Binding b; uint8_t* aq; float* as;
    __device__ __forceinline__ unsigned source_pair(unsigned row) const { return row/2; }
    __device__ __forceinline__ bool input(unsigned i,strix::core::moe_i4::ReferenceRow& r) const {
        const auto& a=b.a;
        if(a.ntiles && blockIdx.y>=*a.ntiles) return false;
        const auto tile=a.tiles[blockIdx.y];if(i>=tile.count) return false;
        const unsigned p=a.pairs[tile.first+i],m=blockIdx.z;
        r={tile.expert,2*p+m,m,(blockIdx.x*8+threadIdx.x/32)*128,H,nullptr,tile.expert==E ? nullptr:aq,as,a.input};return true;
    }
    __device__ __forceinline__ bool activation(unsigned i,strix::core::moe_i4::ReferenceRow& r) const {
        const auto& a=b.a;
        if(a.ntiles && blockIdx.y>=*a.ntiles) return false;
        const auto tile=a.tiles[blockIdx.y];const unsigned row=threadIdx.x/32+8*i;if(row>=tile.count) return false;
        const unsigned p=a.pairs[tile.first+row];
        r={tile.expert,p,2,blockIdx.x*128,I,a.gu,tile.expert==E ? nullptr:aq,as,a.at};return true;
    }
};

void launch_i4_chain(Args a,unsigned grid_tiles,const moe::ExpertSlices& shared,const iu4::Layer& q,
    const iu4::Scratch& s,hipStream_t stream) {
    namespace st=strix::core;
    // Plan 2 (BM128) paired gate/up spills on gfx1151: the paired call runs at most plan 1 on its own tile list.
    const int plan=iu4_plan(a.rows*8U,strix::core::moe_i4::kPlan2Rows), pplan=plan<1 ? plan : 1;
    const auto ptiles=unsigned(st::iu4_grouped_max_tiles(size_t(a.rows)*8,E,pplan));
    const auto dtiles=unsigned(st::iu4_grouped_max_tiles(size_t(a.rows)*8,E,plan));
    if(a.rows>s.max_rows || ptiles>s.max_tiles || dtiles>s.max_tiles) throw std::logic_error("glm chain scratch too small");
    auto* tiles=static_cast<st::IU4GroupTile*>(s.tiles);
    st::moe_i4::Weights w{};
    for(unsigned m=0;m<3;++m) {
        const unsigned K=m==2 ? I:H,N=m==2 ? H:I;
        w.q[m]=q.wq[m]; w.s[m]=reinterpret_cast<const _Float16*>(q.ws[m]); w.svh[m]=q.svh[m];
        w.q_stride[m]=size_t(K)*N/2; w.s_stride[m]=size_t(N)*(K/64);
    }
    I4Binding b{a.x,a.out,a};
    const auto* groups=reinterpret_cast<const st::IU4Group*>(a.groups);
    // The control's own pre-pass: routed rows -> gate/up A4 image (row 2*pair+m), shared rows -> FP16
    // input rows. Quantised once per row instead of once per N tile inside the paired K-loop.
    st::moe_i4::reference_input_kernel<<<dim3(H/1024,grid_tiles,2),256,0,stream>>>(I4ReferenceSchedule{{a.x,a.out,a},s.aq_gu,s.as_gu});
    ok(hipGetLastError(),"chain gate/up input transform + quantiser");
    iu4_ok(st::build_iu4_grouped_tiles(groups,E,pplan,tiles,s.ntiles,ptiles,stream),"chain tiles");
    iu4_ok(st::moe_i4::launch<true>(pplan,b,w,tiles,s.ntiles,ptiles,s.aq_gu,s.as_gu,s.aq_down,s.as_down,stream),"chain paired gate/up");
    if(pplan!=plan) iu4_ok(st::build_iu4_grouped_tiles(groups,E,plan,tiles,s.ntiles,dtiles,stream),"chain down tiles");
    iu4_ok(st::moe_i4::launch<false>(plan,b,w,tiles,s.ntiles,dtiles,nullptr,nullptr,s.aq_down,s.as_down,stream),"chain down");
    // Shared expert retains its existing exact FP16 operand and WMMA-X arithmetic; the combine is the
    // control's ordered one (shared-down aux epilogue, or R16), not atomics.
    a.aux_act=false; a.aux_combine=aux::enabled_default_on("STRIX_GLM_PF_AUX_COMBINE");
    launch_linear_w<false,false>(a,shared.gate.bits,stream);
    st::moe_i4::fp16_activation_rows<<<dim3(I/128,(a.rows+15)/16),256,0,stream>>>(b,a.gu,a.at,a.rows,E,P,8);
    ok(hipGetLastError(),"chain shared activation");
    launch_linear_w<true,false>(a,shared.down.bits,stream);
    if(!a.aux_combine) {
        combine_kernel<<<dim3(H/256,a.rows),256,0,stream>>>(a);
        ok(hipGetLastError(),"chain R16");
    }
}

// Routed gate/up/down on IU4; shared expert, activation and combine as the WMMA-X path (aux_act off). Runs the whole
// call from the input transform on (launch_grouped returns before its own input_kernel).
void launch_iu4(Args a, unsigned grid_tiles, const moe::ExpertSlices& shared, const iu4::Layer& q, const iu4::Scratch& s,
                hipStream_t stream) {
    namespace st = strix::core;
    if(st::moe_i4::enabled()) { launch_i4_chain(a,grid_tiles,shared,q,s,stream); return; }
    wmma_x::iu4_moe_calls.fetch_add(1, std::memory_order_relaxed);
    a.aux_act = false;
    a.aux_combine = aux::enabled_default_on("STRIX_GLM_PF_AUX_COMBINE");
    const unsigned pairs = a.rows * P;
    const int plan = iu4_plan(pairs);
    const auto max_tiles = static_cast<std::uint32_t>(st::iu4_grouped_max_tiles(std::size_t{a.rows} * 8U, static_cast<int>(E), plan));
    if (pairs > s.max_rows * P || max_tiles > s.max_tiles) throw std::logic_error("glm pf iu4: scratch smaller than the call");
    auto* tiles = static_cast<st::IU4GroupTile*>(s.tiles);
    static_assert(sizeof(Tile) == sizeof(st::IU4Group) && offsetof(Tile, first) == offsetof(st::IU4Group, first) &&
                  offsetof(Tile, count) == offsetof(st::IU4Group, count), "plan groups are the library's IU4Group");
    const bool fuse = iu4_fuse_quant();
    if (fuse) {
        st::moe_i4::reference_input_kernel<<<dim3(H/1024,grid_tiles,2),256,0,stream>>>(I4ReferenceSchedule{{a.x,a.out,a},s.aq_gu,s.as_gu});
        ok(hipGetLastError(), "iu4 gate/up input transform + quantiser");
    } else {
        input_kernel<<<dim3(H / 1024, grid_tiles, 2), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "iu4 gate/up input transform");
        iu4_ok(st::quantise_wmma_iu4_activations(a.input, s.aq_gu, s.as_gu, static_cast<int>(H), static_cast<int>(pairs * 2U), stream, 64),
               "iu4 gate/up activation quantiser");
    }
    iu4_ok(st::build_iu4_grouped_tiles(reinterpret_cast<const st::IU4Group*>(a.groups), static_cast<int>(E), plan, tiles, s.ntiles,
                                       max_tiles, stream), "iu4 tile list");
    for (unsigned m = 0; m < 2U; ++m) {
        st::IU4GroupedOperands o;
        o.a_q = s.aq_gu; o.a_scale = s.as_gu; o.w_q = q.wq[m]; o.w_scale = reinterpret_cast<const _Float16*>(q.ws[m]);
        o.w_expert_bytes = std::size_t{H} * I / 2U; o.w_scale_expert = std::size_t{I} * (H / 64U);
        o.col_scale = q.svh[m]; o.map = a.pairs; o.a_mul = 2U; o.a_add = m; o.o_mul = 2U; o.o_add = m;
        iu4_ok(st::launch_wmma_tiled_iu4_grouped(o, tiles, s.ntiles, max_tiles, a.gu, static_cast<int>(H), static_cast<int>(I), stream, plan),
               "iu4 routed gate/up");
    }
    launch_linear_w<false, false>(a, shared.gate.bits, stream);  // shared expert only
    if (fuse) {
        st::moe_i4::reference_activation_kernel<<<dim3(I/128,grid_tiles),256,0,stream>>>(I4ReferenceSchedule{{a.x,a.out,a},s.aq_down,s.as_down});
        ok(hipGetLastError(), "iu4 R15/down transform + quantiser");
    } else {
        activation_kernel<<<dim3(I / 128, grid_tiles), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "iu4 R15/down transform");
        iu4_ok(st::quantise_wmma_iu4_activations(a.at, s.aq_down, s.as_down, static_cast<int>(I), static_cast<int>(pairs), stream, 64),
               "iu4 down activation quantiser");
    }
    st::IU4GroupedOperands o;
    o.a_q = s.aq_down; o.a_scale = s.as_down; o.w_q = q.wq[2]; o.w_scale = reinterpret_cast<const _Float16*>(q.ws[2]);
    o.w_expert_bytes = std::size_t{I} * H / 2U; o.w_scale_expert = std::size_t{H} * (I / 64U);
    o.col_scale = q.svh[2]; o.map = a.pairs;
    iu4_ok(st::launch_wmma_tiled_iu4_grouped(o, tiles, s.ntiles, max_tiles, a.output, static_cast<int>(I), static_cast<int>(H), stream, plan),
           "iu4 routed down");
    launch_linear_w<true, false>(a, shared.down.bits, stream);  // shared down (its aux combine reads the routed outputs)
    if (!a.aux_combine) {
        combine_kernel<<<dim3(H / 256, a.rows), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "iu4 R16");
    }
}
#endif
