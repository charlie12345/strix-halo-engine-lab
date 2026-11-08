// The K pipeline is finished, but other waves may still be reading the stage.
// Reuse it in 32-row passes: 32 KiB FP32 / 16 KiB BF16, within every admitted
// stage. Each value uses the original unscale and (if selected) BF16 rounding.
#ifdef STRIX_DA16_CERTIFICATE
    constexpr bool EpiBF16=RouteBF16;
    bool unsafe=false;
#else
    constexpr bool EpiBF16=false;
#endif
    static_assert(sizeof(Stage)>=32U*Columns*(EpiBF16?2U:4U));
    typedef unsigned EpiWord __attribute__((ext_vector_type(4),aligned(16)));
    float* const ep=reinterpret_cast<float*>(sb);
    for(unsigned base=0;base<count;base+=32U) {
        __syncthreads();
#pragma unroll
        for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
        for(unsigned e=0;e<8U;++e) {
            const unsigned a=rbase+16U*mi+2U*e+h;
            if(a<count&&a>=base&&a<base+32U) {
                const float unscale=ldexpf(1.0F,-exp_of(a));
                const unsigned index=(a-base)*Columns+32U*cg+i;
                const float d0=moe_route_store<EpiBF16>(ep,index,acc[mi][0][e]*unscale);
                const float d1=moe_route_store<EpiBF16>(ep,index+16U,acc[mi][1][e]*unscale);
#ifdef STRIX_DA16_CERTIFICATE
                unsafe|=(__float_as_uint(d0)&0x7fffffffU)>=0x7b800000U;
                unsafe|=(__float_as_uint(d1)&0x7fffffffU)>=0x7b800000U;
#else
                (void)d0;(void)d1;
#endif
            }
        }
        __syncthreads();
        constexpr unsigned Per=Columns*(EpiBF16?2U:4U)/16U;
        for(unsigned j=t;j<min(32U,count-base)*Per;j+=T::Threads) {
            const unsigned a=base+j/Per,c=j%Per;
            auto* out=reinterpret_cast<unsigned char*>(at(view.rawdots,row_of(a)))+
                (static_cast<std::size_t>(slot_of(a))*hidden+tile*Columns)*(EpiBF16?2U:4U);
            *reinterpret_cast<__attribute__((address_space(1))) EpiWord*>(reinterpret_cast<std::uintptr_t>(out+16U*c))=
                reinterpret_cast<const EpiWord*>(ep)[j];
        }
    }
#ifdef STRIX_DA16_CERTIFICATE
    const unsigned bad=static_cast<unsigned>(__ballot(unsafe));
    if(bad&&lane==0U)atomicOr(view.unsafe,1U);
#endif
