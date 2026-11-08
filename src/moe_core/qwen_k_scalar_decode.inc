struct QwenKScalarDecode {
static __device__ __forceinline__ uint2 f16_affine_u8(unsigned w,float a,float b) {
    const float v0=fmaf(a,static_cast<float>(w&255U),b),v1=fmaf(a,static_cast<float>((w>>8U)&255U),b);
    const float v2=fmaf(a,static_cast<float>((w>>16U)&255U),b),v3=fmaf(a,static_cast<float>(w>>24U),b);
    return {f16_pair(v0,v1),f16_pair(v2,v3)};
}
};
