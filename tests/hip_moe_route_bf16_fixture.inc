// Native candidate gate: production DA16 down -> compact route dots -> ordered MC tail.
// The CPU reference deliberately does not call the implementation's conversion helper.
std::uint16_t route_bf16_reference(float x) {
    const auto u=std::bit_cast<std::uint32_t>(x);
    if ((u&0x7fffffffU)>0x7f800000U) return static_cast<std::uint16_t>((u>>16U)|0x40U);
    const unsigned hi=u>>16U,lo=u&65535U;
    return static_cast<std::uint16_t>(hi+(lo>32768U||(lo==32768U&&(hi&1U))?1U:0U));
}
float route_bf16_widen(std::uint16_t x) { return std::bit_cast<float>(static_cast<unsigned>(x)<<16U); }
template<class T> T route_fixture_at(const std::vector<unsigned char>& bytes,std::uint64_t at) {
    T value;std::memcpy(&value,bytes.data()+at,sizeof(T));return value;
}
void route_fixture_require(bool ok,const char* what) { if(!ok) throw std::runtime_error(what); }
// The same storage epilogue as DA16, on exact synthetic FP32 dots. Single-threaded
// merge avoids cross-CTA header races in nonfinite cases; the production status logic is shared.
template<bool BF16, bool Vec4>
__global__ void route_bf16_edge_kernel(const float* input,float* raw,const unsigned* routes,const float* weights,
    float* acc,std::uint16_t* output,unsigned* header,unsigned* status,const unsigned* unsafe) {
    constexpr unsigned H = Vec4 ? 4U : 1U;
    for(unsigned s=threadIdx.x;s<10U*H;s+=blockDim.x) moe_route_store<BF16>(raw,s,input[s/H]);
    __syncthreads();
    if(threadIdx.x==0U) {
        const FixtureTailView view{{routes,0},{weights,0},{status,0},{raw,0},{acc,0},{output,0}};
        if constexpr (Vec4)
            moe_prefill_merge_z_body_vec4<true,BF16>(view,header,H,16U,10U,
                [](auto v,unsigned r){return fixture_row(v,r);},unsafe);
        else
            moe_prefill_merge_z_body<true,true,BF16>(view,header,H,16U,10U,
                [](auto v,unsigned r){return fixture_row(v,r);},unsafe);
    }
}
void route_bf16_tests() {
    unsigned cases=0;
    kr::set_prefill_core2(0);kr::set_prefill_route(14);kr::set_prefill_da16(1);kr::set_prefill_small(0);kr::set_prefill_xor64(1);
    // Automatic tunes 80/82/88, including XOR rows, with top_k=10 and partial CSR batches.
    for(unsigned rows:{48U,208U,3072U}) for(unsigned gate:{12U,13U}) for(unsigned down:{7U,8U}) {
        const unsigned experts = rows == 3072U ? 512U : 16U;
        Fixture f({"route BF16",experts,rows,gate,down,4U,true,true,false});
        const auto incumbent=f.run(3); // unchanged non-certificate DA16 writer
        f.view.unsafe=static_cast<unsigned*>(f.admission.base);
        const auto original=f.run(3);
        route_fixture_require(original==incumbent,"OFF certificate changed incumbent DA16 bytes");
        route_fixture_require(kr::prefill_certificate_produced(),"route BF16 original certificate missing");
        const auto raw=original[9];
        const auto weights=f.weights.read();
        f.view.route_bf16=true;
        const auto compact=f.run(3);
        route_fixture_require(kr::prefill_certificate_produced()&&kr::prefill_da16_ran(),"route BF16 writer not covered");
        route_fixture_require(kr::prefill_effective_tune(f.request,rows,down,3)==(rows==48U?80U:rows==208U?82U:88U),"route BF16 tune coverage");
        route_fixture_require(route_fixture_at<unsigned>(compact.back(),0U)==0U,"route BF16 upstream status");
        for(unsigned r=0;r<rows;++r) {
            for(unsigned i=0;i<Fixture::K*Fixture::H;++i) {
                const float dot=route_fixture_at<float>(raw,r*f.rawdots.stride+4ULL*i);
                const auto stored=route_fixture_at<std::uint16_t>(compact[9],r*f.rawdots.stride+2ULL*i);
                route_fixture_require(stored==route_bf16_reference(dot),"DA16 compact route bit mismatch");
            }
            for(std::uint64_t i=2ULL*Fixture::K*Fixture::H;i<f.rawdots.stride;++i)
                route_fixture_require(compact[9][r*f.rawdots.stride+i]==0xa5U,"route BF16 wrote padded tail");
        }
        Buffer out(rows,2U*Fixture::H),record(1,sizeof(strix::DeferredMoeFirstError));
        // Independent upstream FP32 merge/commit, with only its input dots rounded for ON.
        std::array<std::vector<unsigned char>,2> reference_outputs,reference_accumulations;
        for(bool bf16:{false,true}) {
            f.view.route_bf16=false;(void)f.run(3);
            auto reference_raw=raw;
            if(bf16) for(unsigned r=0;r<rows;++r) for(unsigned i=0;i<Fixture::K*Fixture::H;++i) {
                const auto at=r*f.rawdots.stride+4ULL*i;
                const float dot=route_bf16_widen(route_bf16_reference(route_fixture_at<float>(raw,at)));
                std::memcpy(reference_raw.data()+at,&dot,sizeof(dot));
            }
            check(hipMemcpyAsync(f.rawdots.base,reference_raw.data(),reference_raw.size(),hipMemcpyHostToDevice,f.stream),"reference route dots");
            out.poison(f.stream);record.poison(f.stream);
            hipLaunchKernelGGL(reference_merge_kernel_z,dim3(10U,rows),dim3(256),0,f.stream,
                f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K);
            hipLaunchKernelGGL(reference_commit_kernel,dim3(10U,rows),dim3(256),0,f.stream,
                f.view,f.view.csr.summary,rows,Fixture::H,out.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
            check(hipGetLastError(),"reference merge/commit launch");check(hipStreamSynchronize(f.stream),"reference merge/commit");
            reference_outputs[bf16]=out.read();reference_accumulations[bf16]=f.clear.read();
        }
        std::array<std::vector<unsigned char>,6> scalar_outputs,scalar_accumulations;
        for(bool vec:{false,true}) for(bool bf16:{false,true}) for(unsigned tail:{0U,1U,2U}) {
            // tail 0 = MC mode 1; 1 = MC with accumulation for oracle inspection;
            // 2 = force unsafe: unchanged merge_z chain followed by finish/commit.
            f.view.route_bf16=bf16;(void)f.run(3);
            const unsigned unsafe=tail==2U?1U:0U;
            check(hipMemcpyAsync(f.view.unsafe,&unsafe,4U,hipMemcpyHostToDevice,f.stream),"route BF16 tail choice");
            out.poison(f.stream);record.poison(f.stream);
            if(vec) {
                const auto grid = vec4_grid(rows);
                if(bf16) {
                    if(tail==0U)
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_vec4_kernel<false,true>),grid,dim3(128),0,f.stream,
                            f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
                    else
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_vec4_kernel<true,true>),grid,dim3(128),0,f.stream,
                            f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
                } else {
                    if(tail==0U)
                        hipLaunchKernelGGL(candidate_merge_commit_vec4_kernel<false>,grid,dim3(128),0,f.stream,
                            f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
                    else
                        hipLaunchKernelGGL(candidate_merge_commit_vec4_kernel<true>,grid,dim3(128),0,f.stream,
                            f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
                }
            } else if(bf16) {
                if(tail==0U)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<false,true>),dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<true,true>),dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
            } else {
                if(tail==0U)
                    hipLaunchKernelGGL(candidate_merge_commit_kernel<false>,dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
                else
                    hipLaunchKernelGGL(candidate_merge_commit_kernel<true>,dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,experts,Fixture::K,out.rows<std::uint16_t>());
            }
            hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,f.stream,f.view,f.view.csr.summary,rows,Fixture::H,
                out.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
            check(hipGetLastError(),"route BF16 merge/finish launch");check(hipStreamSynchronize(f.stream),"route BF16 merge/finish");
            const auto actual=out.read(),acc=f.clear.read(),statuses=f.status.read();
            route_fixture_require(actual==reference_outputs[bf16],"route output differs from upstream FP32 reference");
            if(tail!=0U) route_fixture_require(acc==reference_accumulations[bf16],"route accumulation differs from upstream FP32 reference");
            const unsigned arm = (bf16 ? 3U : 0U) + tail;
            if(!vec) { scalar_outputs[arm]=actual;scalar_accumulations[arm]=acc; }
            else {
                route_fixture_require(actual==scalar_outputs[arm],"vector/scalar route output byte mismatch");
                if(tail!=0U) route_fixture_require(acc==scalar_accumulations[arm],"vector/scalar route accumulation byte mismatch");
            }
            unsigned header=0;check(hipMemcpy(&header,f.view.csr.summary,4U,hipMemcpyDeviceToHost),"route BF16 merge header");
            route_fixture_require(header==0U,"route BF16 finite merge header");
            for(unsigned r=0;r<rows;++r) {
                route_fixture_require(route_fixture_at<unsigned>(statuses,r*f.status.stride)==0U,"route BF16 row status");
                for(unsigned h=0;h<Fixture::H;++h) {
                    float sum=0.0F;
                    for(unsigned slot=0;slot<10U;++slot) {
                        float dot=route_fixture_at<float>(raw,r*f.rawdots.stride+4ULL*(slot*Fixture::H+h));
                        if(bf16)dot=route_bf16_widen(route_bf16_reference(dot));
                        const float w=route_fixture_at<float>(weights,r*f.weights.stride+4ULL*slot);
                        sum=std::fma(w,dot,sum);
                    }
                    route_fixture_require(route_fixture_at<std::uint16_t>(actual,r*out.stride+2ULL*h)==route_bf16_reference(sum),
                        "route BF16 CPU slot-order fmaf output mismatch");
                    if(tail!=0U) route_fixture_require(route_fixture_at<unsigned>(acc,r*f.clear.stride+4ULL*h)==std::bit_cast<unsigned>(sum),
                        "route BF16 CPU slot-order fmaf accumulation mismatch");
                }
            }
            ++cases;
        }
        f.view.route_bf16=false;
        route_fixture_require(f.run(3)==original,"route BF16 OFF changed original written bytes");
    }
    // RNE ties (both signs/parities), signed zeros, FP32/BF16 subnormals,
    // finite overflow, infinities and low-payload NaNs. Ten routes plus a poisoned tail.
    Buffer inputs(1,40U),raw(1,192U),routes(1,40U),weights(1,40U),acc(1,16U),output(1,8U),header(1,4U),status(1,4U),admission(1,4U),record(1,sizeof(strix::DeferredMoeFirstError));
    const unsigned rt[10]{0,1,2,3,4,5,6,7,8,9};float w[10];std::fill_n(w,10,1.0F);
    check(hipMemcpy(routes.base,rt,sizeof(rt),hipMemcpyHostToDevice),"edge routes");
    check(hipMemcpy(weights.base,w,sizeof(w),hipMemcpyHostToDevice),"edge weights");
    for(unsigned bits:{0U,0x80000000U,0x00000001U,0x00008000U,0x00018000U,0x007fffffU,0x00800000U,
        0x3f808000U,0x3f818000U,0xbf808000U,0xbf818000U,0x7f7fffffU,0xff7fffffU,0x7f800000U,0xff800000U,
        0x7f800001U,0xff800001U,0x7fc00000U,0x7fffffffU}) for(bool vec:{false,true}) for(bool bf16:{false,true}) for(unsigned fallback:{0U,1U}) {
        float dots[10]{};dots[0]=std::bit_cast<float>(bits);
        check(hipMemcpy(inputs.base,dots,sizeof(dots),hipMemcpyHostToDevice),"edge input");
        check(hipMemset(header.base,0,4U),"edge header");check(hipMemset(status.base,0,4U),"edge status");
        raw.poison(nullptr);output.poison(nullptr);record.poison(nullptr);
        check(hipMemcpy(admission.base,&fallback,4U,hipMemcpyHostToDevice),"edge admission");
        const auto launch_edge = [&]<bool BF16, bool Vec4>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(route_bf16_edge_kernel<BF16,Vec4>),dim3(1),dim3(32),0,nullptr,
                static_cast<float*>(inputs.base),static_cast<float*>(raw.base),static_cast<unsigned*>(routes.base),
                static_cast<float*>(weights.base),static_cast<float*>(acc.base),static_cast<std::uint16_t*>(output.base),
                static_cast<unsigned*>(header.base),static_cast<unsigned*>(status.base),static_cast<unsigned*>(admission.base));
        };
        if(vec) { if(bf16)launch_edge.template operator()<true,true>();else launch_edge.template operator()<false,true>(); }
        else { if(bf16)launch_edge.template operator()<true,false>();else launch_edge.template operator()<false,false>(); }
        const unsigned hidden=vec?4U:1U;
        check(hipGetLastError(),"edge kernel");
        kr::PrefillRows v{};v.routes=routes.rows<const unsigned>();v.route_weights=weights.rows<const float>();
        v.status=status.rows<unsigned>();v.rawdots=raw.rows<float>();v.clear=acc.rows<float>();v.unsafe=static_cast<unsigned*>(admission.base);
        hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,nullptr,v,static_cast<unsigned*>(header.base),1U,hidden,
            output.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
        check(hipGetLastError(),"edge finish");check(hipStreamSynchronize(nullptr),"edge completion");
        float sum=0.0F;
        for(float dot:dots)sum=std::fma(1.0F,bf16?route_bf16_widen(route_bf16_reference(dot)):dot,sum);
        const auto sum_bits=std::bit_cast<unsigned>(sum);
        const unsigned expected_status=(sum_bits&0x7fffffffU)>=0x7f7f8000U?8U:0U;
        route_fixture_require(route_fixture_at<unsigned>(header.read(),0)==expected_status&&
            route_fixture_at<unsigned>(status.read(),0)==expected_status,"edge nonfinite status mismatch");
        // Nonfinite sums are refused before output publication, so the public output
        // remains the sentinel even when host/device NaN propagation payloads differ.
        const auto expected_output=expected_status?std::uint16_t{0xa5a5U}:route_bf16_reference(sum);
        const auto edge_output=output.read(),edge_acc=acc.read();
        for(unsigned j=0;j<hidden;++j) {
            route_fixture_require(route_fixture_at<std::uint16_t>(edge_output,2ULL*j)==expected_output,"edge output bits mismatch");
            if(!std::isnan(sum)) route_fixture_require(route_fixture_at<unsigned>(edge_acc,4ULL*j)==sum_bits,"edge accumulation bits mismatch");
        }
        const auto stored=raw.read();
        if(bf16)route_fixture_require(route_fixture_at<std::uint16_t>(stored,0)==route_bf16_reference(dots[0]),"edge route rounding bits");
        else route_fixture_require(route_fixture_at<unsigned>(stored,0)==bits,"edge OFF route bits");
        for(unsigned i=(bf16?20U:40U)*hidden;i<192U;++i)route_fixture_require(stored[i]==0xa5U,"edge padded tail overwritten");
        ++cases;
    }
    kr::set_prefill_core2(-1);kr::set_prefill_route(-1);kr::set_prefill_da16(-1);kr::set_prefill_small(-1);kr::set_prefill_xor64(-1);
    std::printf("route BF16: %u down/merge and edge cases passed; native model qualification still required\n",cases);
}


// Independent double-precision RNE oracle; no dependency on the device exponent or code helpers.
int route_i8_round(double x) {
    const double a=std::abs(x),lo=std::floor(a),fraction=a-lo;
    const int n=static_cast<int>(lo);
    const int rounded=n+((fraction>0.5||(fraction==0.5&&(n&1)))?1:0);
    return x<0.0?-rounded:rounded;
}
struct RouteI8Reference {
    std::vector<unsigned char> storage,widened;
    std::vector<double> bounds;
};
RouteI8Reference route_i8_reference(const std::vector<unsigned char>& raw,std::uint64_t stride,unsigned rows,unsigned hidden,unsigned top_k) {
    RouteI8Reference ref{std::vector<unsigned char>(raw.size(),0xa5U),raw,
        std::vector<double>(static_cast<std::size_t>(rows)*hidden*top_k)};
    const std::uint64_t n=static_cast<std::uint64_t>(hidden)*top_k,groups=(hidden+31U)/32U;
    const auto exponent_at=(n+1U)/2U*2U,fallback_at=exponent_at+2U*groups*top_k;
    route_fixture_require(fallback_at+2U*n<=4U*n,"I8 fallback does not fit reserved FP32 row");
    for(unsigned r=0;r<rows;++r)for(unsigned slot=0;slot<top_k;++slot)for(unsigned g=0;g<groups;++g) {
        const unsigned live=std::min(32U,hidden-32U*g);
        double peak=0.0;bool exceptional=false;
        for(unsigned c=0;c<live;++c) {
            const auto index=static_cast<std::uint64_t>(slot)*hidden+32U*g+c;
            const float dot=route_fixture_at<float>(raw,r*stride+4U*index);
            exceptional|=!std::isfinite(dot);peak=std::max(peak,std::abs(static_cast<double>(dot)));
        }
        int e=0;
        if(!exceptional&&peak!=0.0) {
            e=std::ilogb(peak)-7;
            while(route_i8_round(std::ldexp(peak,-e))>127)++e;
            exceptional=e < -126 || e > 120;
        }
        const auto stored_e=static_cast<std::int16_t>(exceptional?-32768:e);
        std::memcpy(ref.storage.data()+r*stride+exponent_at+2U*(slot*groups+g),&stored_e,2U);
        for(unsigned c=0;c<live;++c) {
            const auto index=static_cast<std::uint64_t>(slot)*hidden+32U*g+c,at=r*stride+4U*index;
            const float dot=route_fixture_at<float>(raw,at);float reconstructed;
            if(exceptional) {
                const auto bits=route_bf16_reference(dot);
                std::memcpy(ref.storage.data()+r*stride+fallback_at+2U*index,&bits,2U);
                reconstructed=route_bf16_widen(bits);
            } else {
                const int rounded=route_i8_round(std::ldexp(static_cast<double>(dot),-e));
                route_fixture_require(rounded>=-127&&rounded<=127,"I8 oracle saturation");
                const auto code=static_cast<std::int8_t>(rounded);
                std::memcpy(ref.storage.data()+r*stride+index,&code,1U);
                reconstructed=std::ldexp(static_cast<float>(code),e);
                const double bound=std::ldexp(1.0,e-1);
                route_fixture_require(std::abs(static_cast<double>(reconstructed)-dot)<=bound,"I8 per-group half-step bound");
                ref.bounds[static_cast<std::size_t>(r*n+index)]=bound;
            }
            // Exceptional groups retain the BF16 boundary, including its separate finite rounding error.
            if(exceptional&&std::isfinite(dot)&&std::isfinite(reconstructed))
                ref.bounds[static_cast<std::size_t>(r*n+index)]=std::abs(static_cast<double>(reconstructed)-dot);
            std::memcpy(ref.widened.data()+at,&reconstructed,4U);
        }
    }
    return ref;
}

// Same half-wave ownership as the down epilogue, including inactive channels in partial groups.
__global__ void route_i8_edge_store(kr::PrefillRows v,const float* input,unsigned hidden,unsigned top_k) {
    const unsigned groups=(hidden+31U)/32U,group=blockIdx.x*2U+(threadIdx.x>>4U),lane16=threadIdx.x&15U;
    if(group>=groups*top_k)return;
    const unsigned slot=group/groups,column=(group%groups)*32U,live=min(32U,hidden-column);
    const std::uint64_t first=static_cast<std::uint64_t>(slot)*hidden+column;
    const auto* source=fixture_row(kr::StridedRows<const float>{input,v.rawdots.stride},blockIdx.y);
    const auto stored=moe_route_i8_store_pair(fixture_row(v.rawdots,blockIdx.y),first,lane16,live,hidden,top_k,
        lane16<live?source[first+lane16]:0.0F,lane16+16U<live?source[first+lane16+16U]:0.0F);
    const bool unsafe=(__float_as_uint(stored.x)&0x7fffffffU)>=0x7b800000U||(__float_as_uint(stored.y)&0x7fffffffU)>=0x7b800000U;
    if(__ballot(unsafe)&&threadIdx.x==0U)atomicOr(v.unsafe,1U);
}
template<bool Vec4>
__global__ void route_i8_edge_load(kr::PrefillRows v,float* output,unsigned hidden,unsigned top_k) {
    const auto index=(static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)*(Vec4?4U:1U);
    if(index>=static_cast<std::uint64_t>(hidden)*top_k)return;
    auto* destination=fixture_row(kr::StridedRows<float>{output,v.rawdots.stride},blockIdx.y);
    if constexpr(Vec4)*reinterpret_cast<float4*>(destination+index)=moe_route_i8_load4(fixture_row(v.rawdots,blockIdx.y),index,hidden,top_k);
    else destination[index]=moe_route_load<true,true>(fixture_row(v.rawdots,blockIdx.y),index,hidden,top_k);
}

// Check the real-arithmetic dot perturbation bound separately from the ordered FP32 FMA oracle.
void route_i8_merge_checks(kr::PrefillRows v,const RouteI8Reference& ref,const std::vector<unsigned char>& original,
    const std::vector<unsigned char>& weights,unsigned rows,unsigned hidden,unsigned experts,unsigned top_k,hipStream_t stream) {
    Buffer output(rows,2U*hidden),record(1,sizeof(strix::DeferredMoeFirstError));
    for(bool vec:{false,true})for(unsigned tail:{0U,1U,2U}) {
        if(vec&&hidden%4U)continue;
        const unsigned unsafe=tail==2U?1U:0U;
        check(hipMemcpyAsync(v.unsafe,&unsafe,4U,hipMemcpyHostToDevice,stream),"I8 merge admission");
        output.poison(stream);record.poison(stream);
        const auto launch=[&]<bool Acc>() {
            if(vec)hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_vec4_kernel<Acc,true,true>),
                dim3((hidden/4U+127U)/128U,rows),dim3(128),0,stream,v,v.csr.summary,hidden,experts,top_k,output.rows<std::uint16_t>());
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<Acc,true,true>),
                dim3((hidden+255U)/256U,rows),dim3(256),0,stream,v,v.csr.summary,hidden,experts,top_k,output.rows<std::uint16_t>());
        };
        if(tail==0U)launch.template operator()<false>();else launch.template operator()<true>();
        hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,stream,v,v.csr.summary,rows,hidden,
            output.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
        check(hipGetLastError(),"I8 merge/finish launch");check(hipStreamSynchronize(stream),"I8 merge/finish");
        const auto actual=output.read();
        std::vector<unsigned char> accumulation(static_cast<std::size_t>(rows*v.clear.stride));
        check(hipMemcpy(accumulation.data(),v.clear.base,accumulation.size(),hipMemcpyDeviceToHost),"I8 accumulation");
        unsigned header=0;check(hipMemcpy(&header,v.csr.summary,4U,hipMemcpyDeviceToHost),"I8 merge header");
        route_fixture_require(header==0U,"I8 finite merge failed");
        for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<hidden;++c) {
            float sum=0.0F;double exact=0.0,quantized=0.0,bound=0.0,scale=0.0;
            for(unsigned slot=0;slot<top_k;++slot) {
                const auto index=static_cast<std::uint64_t>(slot)*hidden+c,at=r*v.rawdots.stride+4U*index;
                const float dot=route_fixture_at<float>(ref.widened,at),w=route_fixture_at<float>(weights,r*v.route_weights.stride+4ULL*slot);
                sum=std::fma(w,dot,sum);
                exact+=static_cast<double>(w)*route_fixture_at<float>(original,at);
                quantized+=static_cast<double>(w)*dot;
                bound+=std::abs(static_cast<double>(w))*ref.bounds[static_cast<std::size_t>((static_cast<std::uint64_t>(r)*top_k*hidden)+index)];
                scale+=std::abs(static_cast<double>(w))*(std::abs(static_cast<double>(dot))+std::abs(static_cast<double>(route_fixture_at<float>(original,at))));
            }
            route_fixture_require(std::abs(quantized-exact)<=bound+scale*0x1p-48,"I8 weighted merge perturbation bound");
            route_fixture_require(route_fixture_at<std::uint16_t>(actual,r*output.stride+2ULL*c)==route_bf16_reference(sum),"I8 slot-ordered FMA output");
            if(tail!=0U)route_fixture_require(route_fixture_at<unsigned>(accumulation,r*v.clear.stride+4ULL*c)==std::bit_cast<unsigned>(sum),"I8 slot-ordered FMA accumulation");
        }
    }
}

void route_i8_tests() {
    route_fixture_require(strix::moe_route_i8_enabled(),"route-i8 requires STRIX_PREFILL_MOE_ROUTE_I8=1 (default OFF)");
    route_fixture_require(strix::moe_prefill_storage_name(true)=="fp16-v1+moeroute16+moeroutei8"&&
        strix::moe_prefill_storage_identity(true,true)!=strix::moe_prefill_storage_identity(true,false)&&
        strix::moe_prefill_storage_identity(false,true)==strix::moe_prefill_storage_identity(false,false),"I8 storage identity binding");
    unsigned cases=0;
    for(unsigned hidden:{2U,3U,17U,31U,32U,33U,63U,64U,65U,68U,97U})for(unsigned top_k:{1U,3U,10U})for(unsigned flavor=0;flavor<10U;++flavor) {
        constexpr unsigned Rows=2U;
        Buffer input(Rows,4ULL*hidden*top_k),raw(Rows,4ULL*hidden*top_k),decoded(Rows,4ULL*hidden*top_k),unsafe(1,4U);
        std::vector<unsigned char> source(static_cast<std::size_t>(input.bytes),0xa5U);
        for(unsigned r=0;r<Rows;++r)for(unsigned s=0;s<top_k;++s)for(unsigned c=0;c<hidden;++c) {
            const float ties[8]{0.5F,-0.5F,1.5F,-1.5F,2.5F,-2.5F,127.49F,-127.49F};
            float dot=ties[(c+s+r)%8U];
            if(flavor==0U)dot=(c&1U)?-0.0F:0.0F;
            if(flavor==2U)dot=c%32U==0U?127.5F:dot;
            if(flavor==3U)dot=std::bit_cast<float>(1U+(c%32U));
            if(flavor==4U&&c==0U)dot=std::bit_cast<float>(0x7f7fffffU);
            if(flavor==5U&&c==0U)dot=std::bit_cast<float>((s&1U)?0xff800000U:0x7f800000U);
            if(flavor==6U&&c==0U)dot=std::bit_cast<float>((s&1U)?0xff800001U:0x7f800001U);
            if(flavor==7U)dot=std::ldexp(dot,-120);
            if(flavor==8U)dot=std::ldexp(dot,120);
            if(flavor==9U&&c%32U==0U)dot=std::nextafter(127.5F,0.0F);
            std::memcpy(source.data()+r*input.stride+4ULL*(s*hidden+c),&dot,4U);
        }
        const auto ref=route_i8_reference(source,input.stride,Rows,hidden,top_k);
        check(hipMemcpy(input.base,source.data(),source.size(),hipMemcpyHostToDevice),"I8 edge upload");
        raw.poison(nullptr);decoded.poison(nullptr);check(hipMemset(unsafe.base,0,4U),"I8 edge proof clear");
        kr::PrefillRows v{};v.rawdots=raw.rows<float>();v.unsafe=static_cast<unsigned*>(unsafe.base);
        hipLaunchKernelGGL(route_i8_edge_store,dim3((((hidden+31U)/32U)*top_k+1U)/2U,Rows),dim3(32),0,nullptr,
            v,static_cast<const float*>(input.base),hidden,top_k);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(route_i8_edge_load<false>),dim3((hidden*top_k+255U)/256U,Rows),dim3(256),0,nullptr,
            v,static_cast<float*>(decoded.base),hidden,top_k);
        check(hipGetLastError(),"I8 edge launch");check(hipStreamSynchronize(nullptr),"I8 edge completion");
        route_fixture_require(raw.read()==ref.storage,"I8 edge codes/exponents/fallback or poisoned tail mismatch");
        route_fixture_require(decoded.read()==ref.widened,"I8 edge FP32 reconstruction mismatch");
        if(hidden%4U==0U) {
            decoded.poison(nullptr);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(route_i8_edge_load<true>),dim3((hidden*top_k/4U+255U)/256U,Rows),dim3(256),0,nullptr,
                v,static_cast<float*>(decoded.base),hidden,top_k);
            check(hipGetLastError(),"I8 edge vec4 launch");check(hipStreamSynchronize(nullptr),"I8 edge vec4 completion");
            route_fixture_require(decoded.read()==ref.widened,"I8 edge vector reconstruction mismatch");
        }
        unsigned expected_unsafe=0;
        for(unsigned r=0;r<Rows;++r)for(unsigned i=0;i<hidden*top_k;++i)
            expected_unsafe|=(route_fixture_at<unsigned>(ref.widened,r*input.stride+4ULL*i)&0x7fffffffU)>=0x7b800000U?1U:0U;
        route_fixture_require(route_fixture_at<unsigned>(unsafe.read(),0)==expected_unsafe,"I8 stored-dot certificate mismatch");
        if(flavor==0U||flavor==1U||flavor==2U||flavor==7U||flavor==9U) {
            Buffer routes(Rows,4U*top_k),weights(Rows,4U*top_k),status(Rows,4U),header(1,96U),acc(Rows,4U*hidden);
            std::vector<unsigned char> route_bytes(static_cast<std::size_t>(routes.bytes),0U),weight_bytes(static_cast<std::size_t>(weights.bytes),0U);
            for(unsigned r=0;r<Rows;++r)for(unsigned slot=0;slot<top_k;++slot) {
                const float weight=slot%3U==0U?0.0F:slot%2U?-0.25F:0.75F;
                std::memcpy(route_bytes.data()+r*routes.stride+4ULL*slot,&slot,4U);
                std::memcpy(weight_bytes.data()+r*weights.stride+4ULL*slot,&weight,4U);
            }
            check(hipMemcpy(routes.base,route_bytes.data(),route_bytes.size(),hipMemcpyHostToDevice),"I8 synthetic routes");
            check(hipMemcpy(weights.base,weight_bytes.data(),weight_bytes.size(),hipMemcpyHostToDevice),"I8 synthetic weights");
            check(hipMemset(status.base,0,static_cast<std::size_t>(status.bytes)),"I8 synthetic statuses");
            check(hipMemset(header.base,0,static_cast<std::size_t>(header.bytes)),"I8 synthetic header");
            v.routes=routes.rows<const unsigned>();v.route_weights=weights.rows<const float>();v.status=status.rows<unsigned>();
            v.clear=acc.rows<float>();v.csr.summary=static_cast<unsigned*>(header.base);
            route_i8_merge_checks(v,ref,source,weight_bytes,Rows,hidden,16U,top_k,nullptr);
        }
        ++cases;
    }
    route_fixture_require(strix::moe_route_i8_layout(1U,1U).bytes>4U,"I8 tiny arena must decline");
    kr::set_prefill_core2(0);kr::set_prefill_route(14);kr::set_prefill_da16(1);kr::set_prefill_small(0);kr::set_prefill_xor64(1);
    for(unsigned rows:{48U,208U,3072U})for(unsigned gate:{12U,13U})for(unsigned down:{7U,8U}) {
        const unsigned experts=rows==3072U?512U:16U;
        Fixture f({"route I8",experts,rows,gate,down,4U,true,true,false});
        f.view.unsafe=static_cast<unsigned*>(f.admission.base);
        const auto fp32=f.run(3);
        f.view.route_bf16=true;const auto bf16=f.run(3);
        f.view.route_i8=true;const auto compact=f.run(3); // compiled resource gate runs before device work
        route_fixture_require(kr::prefill_certificate_produced()&&kr::prefill_da16_ran(),"I8 DA16 certificate missing");
        const auto ref=route_i8_reference(fp32[9],f.rawdots.stride,rows,Fixture::H,Fixture::K);
        route_fixture_require(compact[9]==ref.storage,"I8 DA16 bytes differ from independent quantizer");
        for(unsigned i=0;i<compact.size();++i)if(i!=9U)route_fixture_require(compact[i]==bf16[i],"I8 changed non-route producer bytes");
        route_i8_merge_checks(f.view,ref,fp32[9],f.weights.read(),rows,Fixture::H,experts,Fixture::K,f.stream);
        f.view.route_i8=false;
        route_fixture_require(f.run(3)==bf16,"I8 OFF did not restore route BF16 bytes");
        ++cases;
    }
    kr::set_prefill_core2(-1);kr::set_prefill_route(-1);kr::set_prefill_da16(-1);kr::set_prefill_small(-1);kr::set_prefill_xor64(-1);
    std::printf("route I8: %u storage/DA16 cases plus scalar/vector MC/unsafe merge arms passed; model qualification still required\n",cases);
}
