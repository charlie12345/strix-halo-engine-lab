// Shared only by the DA16 storage epilogue and its matching ordered merge readers.
template<bool BF16,bool I8=false>
__device__ __forceinline__ float moe_route_load(const float* base, std::uint64_t index,unsigned hidden=0U,unsigned top_k=0U) {
    if constexpr (I8) {
        static_assert(BF16);
        const auto layout=strix::moe_route_i8_layout(hidden,top_k);
        const auto* bytes=reinterpret_cast<const unsigned char*>(base);
        const auto group=(index/hidden)*layout.groups_per_slot+(index%hidden)/32U;
        const auto e=reinterpret_cast<const std::int16_t*>(bytes+layout.exponents)[group];
        if(e==strix::moe_route_i8_fallback)
            return __uint_as_float(static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(bytes+layout.fallback)[index])<<16U);
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(bytes)[index])*ldexpf(1.0F,static_cast<int>(e));
    } else if constexpr (BF16)
        return __uint_as_float(static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(base)[index]) << 16U);
    else return base[index];
}

// Vec4 admission guarantees hidden and index are multiples of four: one group
// exponent serves the whole vector, including the final partial group.
__device__ __forceinline__ float4 moe_route_i8_load4(const float* base,std::uint64_t index,unsigned hidden,unsigned top_k) {
    const auto layout=strix::moe_route_i8_layout(hidden,top_k);
    const auto* bytes=reinterpret_cast<const unsigned char*>(base);
    const auto group=(index/hidden)*layout.groups_per_slot+(index%hidden)/32U;
    const auto e=reinterpret_cast<const std::int16_t*>(bytes+layout.exponents)[group];
    if(e==strix::moe_route_i8_fallback) {
        const auto* values=reinterpret_cast<const std::uint16_t*>(bytes+layout.fallback)+index;
        return make_float4(__uint_as_float(static_cast<unsigned>(values[0])<<16U),__uint_as_float(static_cast<unsigned>(values[1])<<16U),
            __uint_as_float(static_cast<unsigned>(values[2])<<16U),__uint_as_float(static_cast<unsigned>(values[3])<<16U));
    }
    const auto* codes=reinterpret_cast<const std::int8_t*>(bytes)+index;
    const float scale=ldexpf(1.0F,static_cast<int>(e));
    return make_float4(static_cast<float>(codes[0])*scale,static_cast<float>(codes[1])*scale,
        static_cast<float>(codes[2])*scale,static_cast<float>(codes[3])*scale);
}

// Each half-wave owns two 16-column WMMA fragments of one row. Only epilogue
// temporaries cross these shuffles; no LDS, barrier or K-pipeline live-range growth.
__device__ __forceinline__ float2 moe_route_i8_store_pair(float* base,std::uint64_t first,unsigned lane16,unsigned live,
    unsigned hidden,unsigned top_k,float d0,float d1) {
    unsigned magnitude=max(lane16<live?__float_as_uint(d0)&0x7fffffffU:0U,
        lane16+16U<live?__float_as_uint(d1)&0x7fffffffU:0U);
#pragma unroll
    for(unsigned delta=8U;delta;delta>>=1U)magnitude=max(magnitude,__shfl_xor(magnitude,static_cast<int>(delta),16));
    const auto e=strix::moe_route_i8_exponent(magnitude);
    const auto layout=strix::moe_route_i8_layout(hidden,top_k);
    auto* bytes=reinterpret_cast<unsigned char*>(base);
    if(lane16==0U)reinterpret_cast<std::int16_t*>(bytes+layout.exponents)[(first/hidden)*layout.groups_per_slot+(first%hidden)/32U]=e;
    const auto store=[&](unsigned offset,float dot) {
        const auto index=first+offset;
        if(e==strix::moe_route_i8_fallback) {
            const auto bits=strix::moe_route_bf16_bits(__float_as_uint(dot));
            reinterpret_cast<std::uint16_t*>(bytes+layout.fallback)[index]=bits;
            return __uint_as_float(static_cast<unsigned>(bits)<<16U);
        }
        const auto code=static_cast<std::int8_t>(__float2int_rn(dot*ldexpf(1.0F,-static_cast<int>(e))));
        reinterpret_cast<std::int8_t*>(bytes)[index]=code;
        return static_cast<float>(code)*ldexpf(1.0F,static_cast<int>(e));
    };
    return make_float2(lane16<live?store(lane16,d0):0.0F,lane16+16U<live?store(lane16+16U,d1):0.0F);
}
template<bool BF16>
__device__ __forceinline__ float moe_route_store(float* base, std::uint64_t index, float dot) {
    if constexpr (BF16) {
        const auto bits = strix::moe_route_bf16_bits(__float_as_uint(dot));
        reinterpret_cast<std::uint16_t*>(base)[index] = bits;
        return __uint_as_float(static_cast<unsigned>(bits) << 16U);
    } else { base[index] = dot; return dot; }
}
