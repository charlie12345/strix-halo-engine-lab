#pragma once
// Portable arithmetic contract of the FAST_I4 policies. Ops supplies the lane layout,
// RN32/RN16 boundaries and H128 implementation; the Binding supplies model math/scales.
#if defined(__HIPCC__)
#define STRIX_I4_INLINE __device__ __forceinline__
#else
#define STRIX_I4_INLINE inline
#endif
namespace strix::core::moe_i4 {
template<class Ops,class Binding,unsigned N>
STRIX_I4_INLINE void input_values(const Binding& b,unsigned expert,unsigned pair,unsigned m,unsigned col,float (&v)[N],const float* cached=nullptr) {
    for(unsigned j=0;j<N;++j) {
        const unsigned c=col+Ops::channel(j);
        const float x=cached ? cached[j] : b.x[static_cast<unsigned long long>(b.token(pair))*Binding::H+c];
        v[j]=Ops::round16(Ops::materialize(x*b.suh(expert,m,c)));
    }
    Ops::transform(v);
    for(unsigned j=0;j<N;++j) v[j]=Ops::round16(v[j]);
}
template<class Ops,class Binding,unsigned N>
STRIX_I4_INLINE void activation_values(const Binding& b,unsigned expert,unsigned col,const float (&g)[N],const float (&u)[N],float (&v)[N]) {
    for(unsigned j=0;j<N;++j) {
        const unsigned c=col+Ops::channel(j);
        v[j]=Ops::round16(Ops::materialize(b.activation(g[j],u[j])*b.suh(expert,2,c)));
    }
    Ops::transform(v);
    for(unsigned j=0;j<N;++j) v[j]=Ops::round16(v[j]);
}
} // namespace strix::core::moe_i4
#undef STRIX_I4_INLINE
