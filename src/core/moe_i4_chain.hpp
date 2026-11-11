#pragma once
// Candidate FAST_I4 chain. Bindings name model geometry, row maps and nonlinear arithmetic;
// all gather/transform/quantise, paired mainloops and output accumulation live on this core.
#include "tiled_core.hpp"
#include "activation_quant.hpp"
#include "linear_plan.hpp"
#include "moe_i4_arithmetic.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::core::moe_i4 {
// Chain plan policy: auto-select BM128 (plan 2) when the call averages >= kPlan2Rows rows per group.
// Waves past a ragged tile's rows skip their WMMA (tiled_core), so BM128's padding no longer costs its
// compute: grouped down on MiniMax-like routing (gfx1151) BM128 vs BM64 -4 % at ~159 rows per group and
// equal at 64. The paired GEMM is BM64 always (BM128 paired spills).
inline constexpr unsigned kPlan2Rows = 64;
inline bool enabled() {
    const char* p = std::getenv("STRIX_MOE_I4_CHAIN");
    if (!p || !*p || std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_MOE_I4_CHAIN must be 0 or 1 (candidate, native gates pending)");
}
struct Weights {
    const uint8_t* q[3];
    const _Float16* s[3];
    const uint16_t* const* svh[3];
    size_t q_stride[3], s_stride[3];
};
struct WaveOps {
    __device__ __forceinline__ static unsigned channel(unsigned j) { return (threadIdx.x&31U)+32*j; }
    __device__ __forceinline__ static float round16(float x) { return float(_Float16(x)); }
    __device__ __forceinline__ static float materialize(float x) { return materialize_fp32(x); }
    __device__ __forceinline__ static void transform(float (&v)[4]) { hadamard128(v); }
};
template<class Binding> struct InputPolicy {
    Binding b;
    __device__ __forceinline__ void load(unsigned e,unsigned p,unsigned m,unsigned c,float (&v)[4],const float* cached=nullptr) const {
        input_values<WaveOps>(b,e,p,m,c,v,cached);
    }
};
template<class Binding> struct EpiloguePolicy {
    Binding b;
    __device__ __forceinline__ void activation(unsigned e,unsigned c,const float (&g)[4],const float (&u)[4],float (&v)[4]) const {
        activation_values<WaveOps>(b,e,c,g,u,v);
    }
    // Deterministic combine, stage 1: the down epilogue stores each routed pair's FP32 output row
    // (the value the unfused control writes to its down image). Stage 2 is the model's existing
    // ordered combine (ascending top-k FMA, shared expert last), so the chain adds no rounding
    // point, no atomics and no run-to-run or cross-rank order dependence.
    __device__ __forceinline__ void contribution(unsigned pair, unsigned col, const float (&v)[4]) const {
        float* row = b.contribution(pair);
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) row[col + (threadIdx.x & 31U) + 32*j] = v[j];
    }
};
// Unfused control uses the same policies; schedules only name rows and destinations.
struct ReferenceRow {
    unsigned expert,pair,m,col,width;
    const float* gu=nullptr;
    uint8_t* aq=nullptr; float* as=nullptr; _Float16* fp16=nullptr;
};
__device__ __forceinline__ void reference_store(const ReferenceRow& r,const float (&v)[4]) {
    if(r.aq) iu4_quant128(v,r.aq+size_t(r.pair)*(r.width/2),r.as+size_t(r.pair)*(r.width/64),r.col,threadIdx.x&31U);
    else {
        #pragma unroll
        for(unsigned j=0;j<4;++j) r.fp16[size_t(r.pair)*r.width+r.col+(threadIdx.x&31U)+32*j]=_Float16(v[j]);
    }
}
template<class Schedule> __global__ __launch_bounds__(256) void reference_input_kernel(Schedule s) {
    ReferenceRow r{};
    float cached[4];
    if constexpr (Schedule::kCachedInput) {
        if(!s.input(0,r)) return;
        const unsigned token=s.b.token(s.source_pair(r.pair));
        #pragma unroll
        for(unsigned j=0;j<4;++j) cached[j]=s.b.x[size_t(token)*Schedule::Binding::H+r.col+(threadIdx.x&31U)+32*j];
    }
    for(unsigned i=0;s.input(i,r);++i) {
        float v[4]; InputPolicy<typename Schedule::Binding>{s.b}.load(r.expert,s.source_pair(r.pair),r.m,r.col,v,Schedule::kCachedInput ? cached : nullptr);
        reference_store(r,v);
    }
}
template<class Schedule> __global__ __launch_bounds__(256) void reference_activation_kernel(Schedule s) {
    ReferenceRow r{};
    for(unsigned i=0;s.activation(i,r);++i) {
        float g[4],u[4],v[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j) { const unsigned c=r.col+(threadIdx.x&31U)+32*j;g[j]=r.gu[size_t(r.pair)*2*r.width+c];u[j]=r.gu[(size_t(r.pair)*2+1)*r.width+c]; }
        EpiloguePolicy<typename Schedule::Binding>{s.b}.activation(r.expert,r.col,g,u,v);
        reference_store(r,v);
    }
}
template<class Binding> __global__ __launch_bounds__(256) void reference_combine_kernel(Binding b,const float* dn,unsigned rows,unsigned topk) {
    const unsigned q=blockIdx.x*8+threadIdx.x/32,blocks=Binding::H/128,lane=threadIdx.x&31U;
    if(q>=rows*blocks) return;
    const unsigned tok=q/blocks,col=(q%blocks)*128;
    float acc[4]={};
    for(unsigned k=0;k<topk;++k) {
        const unsigned pair=b.slot(tok,k);const float weight=b.weight(pair);
        #pragma unroll
        for(unsigned j=0;j<4;++j) acc[j]=__builtin_fmaf(weight,dn[size_t(pair)*Binding::H+col+lane+32*j],acc[j]);
    }
    const unsigned shared=b.slot(tok,topk);
    #pragma unroll
    for(unsigned j=0;j<4;++j) b.out[size_t(tok)*Binding::H+col+lane+32*j]=acc[j]+dn[size_t(shared)*Binding::H+col+lane+32*j];
}
// Existing FP16 shared experts can use the same policy arithmetic without entering A4.
template<class Binding> __global__ void fp16_input_rows(Binding b,_Float16* input,unsigned rows,unsigned expert,unsigned stride,unsigned slot) {
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32,col=(blockIdx.x*8+wave)*128;
    for(unsigned r=0;r<16;++r) {
        const unsigned token=blockIdx.y*16+r;
        if(token>=rows) return;
        const unsigned pair=token*stride+slot;
        float v[4]; InputPolicy<Binding>{b}.load(expert,pair,blockIdx.z,col,v);
        #pragma unroll
        for(unsigned j=0;j<4;++j) input[(size_t(pair)*2+blockIdx.z)*Binding::H+col+lane+32*j]=_Float16(v[j]);
    }
}
template<class Binding> __global__ void fp16_activation_rows(Binding b,const float* gu,_Float16* at,unsigned rows,unsigned expert,unsigned stride,unsigned slot) {
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32,col=blockIdx.x*128;
    for(unsigned r=wave;r<16;r+=8) {
        const unsigned token=blockIdx.y*16+r;
        if(token>=rows) return;
        const unsigned pair=token*stride+slot;
        float g[4],u[4],v[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j) { const unsigned c=col+lane+32*j;g[j]=gu[size_t(pair)*2*Binding::I+c];u[j]=gu[(size_t(pair)*2+1)*Binding::I+c]; }
        EpiloguePolicy<Binding>{b}.activation(expert,col,g,u,v);
        #pragma unroll
        for(unsigned j=0;j<4;++j) at[size_t(pair)*Binding::I+col+lane+32*j]=_Float16(v[j]);
    }
}
// Paired gate/up reads the interleaved pre-quantised A4 image written once by the shared input
// pre-pass (row 2*pair+m, m=0 gate, m=1 up): the pass selects the row, so the K loop never
// re-gathers, re-transforms or re-quantises the input. Bytes equal the LDS-staged ones.
template<class Binding,bool Pair> struct Chain {
    static constexpr bool kActive = true;
    static constexpr int kPasses = Pair ? 2 : 1;
    static constexpr uint32_t kPassRow = Pair ? 1 : 0;
    Binding b; Weights w;
    uint8_t* aq; float* as;
    unsigned expert = 0;
    __device__ __forceinline__ void bind(unsigned e, int pass, const uint32_t* __restrict__& q, const _Float16*& s, const uint16_t* __restrict__& svh) {
        expert=e;
        // Select, never index, the by-value argument arrays with the runtime pass: a dynamic index
        // would copy the whole kernel argument to scratch.
        auto pick=[&](const auto (&v)[3]) { if constexpr (Pair) return pass==0 ? v[0] : v[1]; else return v[2]; };
        q=reinterpret_cast<const uint32_t*>(pick(w.q)+size_t(e)*pick(w.q_stride));
        s=pick(w.s)+size_t(e)*pick(w.s_stride); svh=pick(w.svh)[e];
    }
    __device__ __forceinline__ void store(unsigned pos,unsigned col,int pass,int slot,float (&v)[4],float* saved) const {
        const unsigned pair=b.pair(pos);
        if constexpr (Pair) {
            if (!pass) {
                #pragma unroll
                for (unsigned j=0;j<4;++j) saved[unsigned(slot)*4+j]=v[j];
            } else {
                float g[4],a[4];
                #pragma unroll
                for (unsigned j=0;j<4;++j) g[j]=saved[unsigned(slot)*4+j];
                EpiloguePolicy<Binding>{b}.activation(expert,col,g,v,a);
                iu4_quant128(a,aq+size_t(pair)*(Binding::I/2),as+size_t(pair)*(Binding::I/64),col,threadIdx.x&31U);
            }
        } else EpiloguePolicy<Binding>{b}.contribution(pair,col,v);
    }
};
template<int Plan,bool Pair,class Binding> hipError_t launch(const Binding& b,const Weights& w,
    const IU4GroupTile* tiles,const uint32_t* ntiles,unsigned max_tiles,const uint8_t* gq,const float* gs,
    uint8_t* aq,float* as,hipStream_t stream) {
    constexpr auto c=iu4_tiles[Plan];
    using G=wmma_tiled::Geometry<c.bm,c.bn,c.wm,c.wn,PackedIU4>;
    using C=Chain<Binding,Pair>;
    wmma_tiled::GroupedArgs gp; gp.tiles=tiles; gp.ntiles=ntiles; gp.col_scale=w.svh[0]; gp.map=b.map();
    // Paired: A is the pre-quantised gate/up image gq/gs, row 2*pair+pass (pass offset via kPassRow).
    // Down: A is the down image aq/as written by the paired epilogue, row = pair.
    gp.a_mul=Pair ? 2U : 1U;
    const unsigned K=Pair ? Binding::H : Binding::I, N=Pair ? Binding::I : Binding::H;
    const auto kernel=wmma_tiled::tiled_core<TiledContract<PackedIU4>,c.bm,c.bn,c.wm,c.wn,true,0,false,false,64,true,wmma_tiled::NoAdapter,C>;
    // Reject spills/insufficient residency before launching. No token/route readback.
    hipFuncAttributes attr{}; auto status=hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(kernel));
    if (status!=hipSuccess) return status;
    int blocks=0; status=hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,G::threads,0);
    if (status!=hipSuccess) return status;
    if (attr.localSizeBytes || blocks*G::waves<4) return hipErrorNotSupported;
    kernel<<<dim3(max_tiles*(N/c.bn)),G::threads,0,stream>>>(reinterpret_cast<const uint32_t*>(w.q[0]),nullptr,nullptr,
        K,N,0,1,nullptr,nullptr,nullptr,1.f,0.f,1,Pair ? gq : aq,Pair ? gs : as,w.s[0],nullptr,gp,{},C{b,w,aq,as});
    return hipGetLastError();
}
template<bool Pair,class Binding> hipError_t launch(int plan,const Binding& b,const Weights& w,
    const IU4GroupTile* tiles,const uint32_t* ntiles,unsigned max_tiles,const uint8_t* gq,const float* gs,
    uint8_t* aq,float* as,hipStream_t stream) {
    switch(plan) {
        case 0:return launch<0,Pair>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,stream);
        case 1:return launch<1,Pair>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,stream);
        case 2:return launch<2,Pair>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,stream);
        default:return hipErrorInvalidValue;
    }
}
} // namespace strix::core::moe_i4
