#pragma once
// Device-only fragments. Types supplies the engine's existing vector types.
namespace strix::moe_core {
template<class Types> struct GroupedF16Fragments {
    using FragmentWord8=typename Types::Word8;
    using FragmentHalf16=typename Types::Half16;
    using FragmentFloat8=typename Types::Float8;
    using FragmentWord4=typename Types::Word4;
template<bool Contiguous> static __device__ __forceinline__ FragmentWord8 f16_frag(const _Float16* row,unsigned lo_half,unsigned hi_half) {
    if constexpr(Contiguous) { (void)hi_half; return *reinterpret_cast<const FragmentWord8*>(row+lo_half); }
    else {
        const FragmentWord4 lo=*reinterpret_cast<const FragmentWord4*>(row+lo_half),hi=*reinterpret_cast<const FragmentWord4*>(row+hi_half);
        return FragmentWord8{lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
    }
}
static __device__ __forceinline__ FragmentFloat8 f16_wmma(FragmentWord8 a,FragmentWord8 b,FragmentFloat8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(FragmentHalf16,a),__builtin_bit_cast(FragmentHalf16,b),c);
}
};
} // namespace strix::moe_core
