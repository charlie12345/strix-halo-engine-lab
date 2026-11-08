// Exact device word comparison, not a digest. Includes padding and poison left
// by skipped work. Reading every word also covers signed zeros and NaN payloads.
__global__ void tiers_compare_words(const unsigned* a,const unsigned* b,std::uint64_t n,unsigned* different) {
    bool bad=false;
    for(std::uint64_t i=blockIdx.x*static_cast<std::uint64_t>(blockDim.x)+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)bad|=a[i]!=b[i];
    if(bad)atomicOr(different,1U);
}
struct TiersSnapshot {
    std::vector<Buffer> saved;
    std::vector<std::pair<const void*,std::uint64_t>> sources;
    Buffer different;
    explicit TiersSnapshot(const std::vector<std::pair<const void*,std::uint64_t>>& src,hipStream_t stream):sources(src),different(1,4U) {
        for(const auto& [p,n]:sources) {
            if(n%4U)throw std::runtime_error("tiers snapshot needs whole words");
            saved.emplace_back(1,n);
            check(hipMemcpyAsync(saved.back().base,p,n,hipMemcpyDeviceToDevice,stream),"tiers snapshot");
        }
    }
    void compare(hipStream_t stream,const std::string& tag) const {
        for(std::size_t k=0;k<sources.size();++k) {
            check(hipMemsetAsync(different.base,0,4U,stream),"tiers compare reset");
            hipLaunchKernelGGL(tiers_compare_words,dim3(1024),dim3(256),0,stream,static_cast<const unsigned*>(sources[k].first),
                static_cast<const unsigned*>(saved[k].base),sources[k].second/4U,static_cast<unsigned*>(different.base));
            unsigned bad=0;
            check(hipMemcpyAsync(&bad,different.base,4U,hipMemcpyDeviceToHost,stream),"tiers compare read");
            check(hipStreamSynchronize(stream),"tiers compare");
            if(bad)throw std::runtime_error("tiers byte mismatch: "+tag+" view "+std::to_string(k));
        }
    }
};
std::vector<std::string> tiers_parse(const std::string& spec) {
    std::vector<std::string> f;std::size_t at=0;
    for(;;){const auto end=spec.find(':',at);f.push_back(spec.substr(at,end==std::string::npos?end:end-at));if(end==std::string::npos)break;at=end+1U;}
    if(f.size()!=6U)throw std::invalid_argument("layer spec needs 6 fields");
    return f;
}
void tiers_safe_weights(Fixture& f) {
    std::vector<float> w(f.weights.bytes/4U);
    check(hipMemcpy(w.data(),f.weights.base,f.weights.bytes,hipMemcpyDeviceToHost),"tiers route weights read");
    for(unsigned r=0;r<f.c.rows;++r)for(unsigned s=0;s<Fixture::K;++s)w[r*(f.weights.stride/4U)+s]*=0.5F;
    check(hipMemcpy(f.weights.base,w.data(),f.weights.bytes,hipMemcpyHostToDevice),"tiers bounded route weights");
}
// Slot-major construction gives expert e exactly e assignments, for e=1..256,
// plus 257, 512 and 513 assignments and an empty expert. This covers all eight
// expert-fill buckets, including repeated full batches followed by a partial one.
void tiers_all_fills(Fixture& f) {
    if(f.c.rows<4096U||f.c.experts<512U)throw std::runtime_error("tiers fill geometry");
    std::vector<unsigned> seq;
    for(unsigned e=1;e<=256U;++e)for(unsigned n=0;n<e;++n)seq.push_back(e);
    for(unsigned e=257U;e<=259U;++e)for(unsigned n=0;n<(e==257U?257U:e==258U?512U:513U);++n)seq.push_back(e);
    const unsigned rest=f.c.rows*Fixture::K-static_cast<unsigned>(seq.size());
    for(unsigned n=0;n<rest;++n)seq.push_back(260U+n*252U/rest);
    std::vector<unsigned> routes(f.routes.bytes/4U,0U),counts(f.c.experts,0U);
    for(unsigned s=0;s<Fixture::K;++s)for(unsigned r=0;r<f.c.rows;++r) {
        const unsigned e=seq[static_cast<std::size_t>(s)*f.c.rows+r];
        routes[r*(f.routes.stride/4U)+s]=e;++counts[e];
    }
    for(unsigned e=0;e<=256U;++e)if(counts[e]!=e)throw std::runtime_error("tiers exact fill construction");
    if(counts[257U]!=257U||counts[258U]!=512U||counts[259U]!=513U)throw std::runtime_error("tiers large fill construction");
    check(hipMemcpy(f.routes.base,routes.data(),f.routes.bytes,hipMemcpyHostToDevice),"tiers crafted fills");
}
// Raw little-endian captures: T x top_k u32 routes, optionally T x top_k FP32
// weights and T x hidden BF16 activations, with no row padding. Larger captures
// supply their prefix to smaller T. Output labels say which routing source ran.
template<class V> void tiers_capture(Buffer& buffer,unsigned rows,unsigned columns,const char* name,bool required) {
    const char* path=std::getenv(name);
    if(!path||!path[0]) {if(required)throw std::invalid_argument(std::string("missing ")+name);return;}
    std::ifstream file(path,std::ios::binary);
    std::vector<unsigned char> bytes=buffer.read();
    for(unsigned r=0;r<rows;++r) {
        file.read(reinterpret_cast<char*>(bytes.data()+r*buffer.stride),static_cast<std::streamsize>(columns*sizeof(V)));
        if(!file)throw std::runtime_error(std::string("short/unreadable capture: ")+name);
    }
    check(hipMemcpy(buffer.base,bytes.data(),buffer.bytes,hipMemcpyHostToDevice),"tiers captured rows");
}
// Without a capture: per-expert fills follow the production-layer histogram
// measured at T8192 (48 calls x 512 experts, per call 32/107/47/60/76/94/62/34
// experts with 0/1-16/17-32/33-64/65-128/129-256/257-512/>512 assignments).
// Counts are spread evenly inside each bucket and scaled by T/8192; the >512
// experts absorb the remainder. Slot-major runs (no row repeats an expert), a
// fixed expert-ID permutation and a fixed row shuffle; deterministic per T.
void tiers_histogram_routing(Fixture& f) {
    if(f.c.experts!=512U||f.c.rows<64U)throw std::runtime_error("tiers histogram routing geometry");
    constexpr std::array<unsigned,8> per_call{32U,107U,47U,60U,76U,94U,62U,34U};
    constexpr std::array<unsigned,8> lo{0U,1U,17U,33U,65U,129U,257U,513U},hi{0U,16U,32U,64U,128U,256U,512U,0U};
    const unsigned total=f.c.rows*Fixture::K;
    std::vector<unsigned> counts;unsigned fixed=0;
    for(unsigned b=0;b<7U;++b)for(unsigned k=0;k<per_call[b];++k) {
        const double at8k=b?lo[b]+(hi[b]-lo[b])*(k+0.5)/per_call[b]:0.0;
        const unsigned n=b?std::max(1U,static_cast<unsigned>(std::lround(at8k*f.c.rows/8192.0))):0U;
        counts.push_back(n);fixed+=n;
    }
    if(fixed>=total)throw std::runtime_error("tiers histogram routing overflow");
    for(unsigned k=0,rest=total-fixed;k<per_call[7];++k)counts.push_back(rest/per_call[7]+(k<rest%per_call[7]?1U:0U));
    std::vector<unsigned> seq;seq.reserve(total);
    for(unsigned k=0;k<512U;++k) {
        if(counts[k]>f.c.rows)throw std::runtime_error("tiers histogram routing fill above T");
        seq.insert(seq.end(),counts[k],(k*167U)%512U);  // odd multiplier: a permutation of expert IDs
    }
    std::vector<unsigned> order(f.c.rows);
    for(unsigned r=0;r<f.c.rows;++r)order[r]=r;
    std::uint64_t state=0x9e3779b97f4a7c15ULL^f.c.rows;
    for(unsigned r=f.c.rows;r>1U;--r) {
        state^=state<<13U;state^=state>>7U;state^=state<<17U;
        std::swap(order[r-1U],order[static_cast<unsigned>(state%r)]);
    }
    std::vector<unsigned> routes(f.routes.bytes/4U,0U);
    for(unsigned s=0;s<Fixture::K;++s)for(unsigned r=0;r<f.c.rows;++r)
        routes[order[r]*(f.routes.stride/4U)+s]=seq[static_cast<std::size_t>(s)*f.c.rows+r];
    check(hipMemcpy(f.routes.base,routes.data(),f.routes.bytes,hipMemcpyHostToDevice),"tiers histogram routes");
}
// TIERS_ROUTES_FILE selects captured routing; otherwise the measured histogram.
// The routing source is printed with its realised fill histogram.
void tiers_real_routing(Fixture& f) {
    const char* captured=std::getenv("TIERS_ROUTES_FILE");
    const bool capture=captured&&captured[0];
    if(capture)tiers_capture<unsigned>(f.routes,f.c.rows,Fixture::K,"TIERS_ROUTES_FILE",true);
    else tiers_histogram_routing(f);
    tiers_capture<float>(f.weights,f.c.rows,Fixture::K,"TIERS_ROUTE_WEIGHTS_FILE",false);
    tiers_capture<std::uint16_t>(f.input,f.c.rows,Fixture::H,"TIERS_INPUT_FILE",false);
    const auto bytes=f.routes.read();std::vector<unsigned> counts(f.c.experts,0U);
    for(unsigned r=0;r<f.c.rows;++r)for(unsigned s=0;s<Fixture::K;++s) {
        const unsigned e=route_fixture_at<unsigned>(bytes,r*f.routes.stride+4ULL*s);
        if(e>=f.c.experts)throw std::runtime_error("route outside expert range");
        ++counts[e];
    }
    std::array<unsigned,8> buckets{};
    for(unsigned n:counts)++buckets[n==0U?0U:n<=16U?1U:n<=32U?2U:n<=64U?3U:n<=128U?4U:n<=256U?5U:n<=512U?6U:7U];
    std::printf("tiers %s routing T%u expert fills 0/1-16/17-32/33-64/65-128/129-256/257-512/>512:",
        capture?"captured":"histogram-matched",f.c.rows);
    for(unsigned n:buckets)std::printf(" %u",n);
    std::printf("\n");
}
int tiers_tests(int argc,char** argv) {
    const std::string mode=argv[1];
    const bool coexist=mode=="tiers-coexist";
    const unsigned max_rows=[] {
        const char* v=std::getenv("TIERS_MAX_ROWS");const unsigned n=v?static_cast<unsigned>(std::stoul(v)):16384U;
        if(n!=8192U&&n!=16384U)throw std::invalid_argument("TIERS_MAX_ROWS must be 8192 or 16384");return n;
    }();
    kr::set_prefill_core2(0);kr::set_prefill_small(0);kr::set_prefill_route(14);kr::set_prefill_da16(1);kr::set_prefill_hdr(1);kr::set_prefill_xor64(0);
    const auto reset=[] {
        kr::set_prefill_tiers(-1);kr::set_prefill_down_lds(-1);kr::set_prefill_core2(-1);kr::set_prefill_small(-1);
        kr::set_prefill_route(-1);kr::set_prefill_da16(-1);kr::set_prefill_hdr(-1);kr::set_prefill_xor64(-1);
        kr::set_prefill_tile64(-1);
        kr::prefill_test_da16_events(nullptr);
    };
    // Non-hashed bytes, before and after the real ordered merge/commit/finish.
    unsigned comparisons=0,certified=0,safe_cases=0,unsafe_cases=0,tunes[3]{};
    const auto gate=[&](Fixture& f,unsigned rows,int form,unsigned cert,int hdr,bool xor64) {
        kr::set_prefill_hdr(hdr);kr::set_prefill_xor64(xor64?1:0);
        f.view.unsafe=cert?static_cast<unsigned*>(f.admission.base):nullptr;
        f.view.route_bf16=cert==2U;
        Buffer output(f.c.rows,2U*Fixture::H),record(1,8U);
        const auto run=[&](int tiers,int epi) {
            kr::set_prefill_tiers(tiers);kr::set_prefill_down_lds(epi);
            // A combined request must preserve the admitted tier/LDS transaction.
            kr::set_prefill_tile64(coexist&&(tiers||epi)?1:0);
            for(const auto* b:f.written())b->poison(f.stream);
            f.csr.poison(f.stream);output.poison(f.stream);record.poison(f.stream);
            check(hipMemsetAsync(f.admission.base,0,4U,f.stream),"tiers certificate reset");
            kr::prefill_test_poison_da16(f.stream,rows);
            f.enqueue(form,rows);
            if(!kr::prefill_da16_ran()||kr::prefill_tiers_ran()!=(tiers!=0&&kr::prefill_last_tune()==82U))throw std::runtime_error("tiers path not exercised");
            if(kr::prefill_tiers_mode_ran()!=(kr::prefill_last_tune()==82U?tiers:0))throw std::runtime_error("tiers mode not exercised");
            const bool aligned=reinterpret_cast<std::uintptr_t>(f.view.rawdots.base)%16U==0U&&f.view.rawdots.stride%16U==0U;
            if(kr::prefill_down_lds_ran()!=(epi!=0&&aligned))throw std::runtime_error("tiers LDS epilogue/fallback not exercised");
            if(coexist&&aligned&&kr::prefill_tile64_ran())throw std::runtime_error("tile64 bypassed tier/LDS precedence");
            if(kr::prefill_certificate_produced()!=(cert!=0))throw std::runtime_error("tiers certificate not exercised");
            check(hipStreamSynchronize(f.stream),"tiers producers");
        };
        const auto sources=[&] {
            std::vector<std::pair<const void*,std::uint64_t>> v;
            for(const auto* b:f.written())v.emplace_back(b->base,b->bytes);
            v.emplace_back(f.csr.base,f.csr.bytes);v.emplace_back(f.admission.base,4U);
            v.emplace_back(kr::prefill_test_da16_data(),static_cast<std::uint64_t>(rows)*Fixture::K*Fixture::I*2U);
            const unsigned cap=kr::prefill_last_tune()&2U?128U:64U;
            const unsigned n=std::min(rows*Fixture::K,f.c.experts+(rows*Fixture::K+cap-1U)/cap);
            v.emplace_back(kr::prefill_test_da16_arrivals(),n*4ULL);
            return v;
        };
        const auto tail=[&] {
            const dim3 grid(Fixture::H/256U,rows);
            if(cert) {
                if(f.view.route_bf16)hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<true,true>),grid,dim3(256),0,f.stream,
                    f.view,f.view.csr.summary,Fixture::H,f.c.experts,Fixture::K,output.rows<std::uint16_t>());
                else hipLaunchKernelGGL(candidate_merge_commit_kernel<true>,grid,dim3(256),0,f.stream,
                    f.view,f.view.csr.summary,Fixture::H,f.c.experts,Fixture::K,output.rows<std::uint16_t>());
            } else {
                hipLaunchKernelGGL(reference_merge_kernel_z,grid,dim3(256),0,f.stream,f.view,f.view.csr.summary,Fixture::H,f.c.experts,Fixture::K);
                hipLaunchKernelGGL(reference_commit_kernel,grid,dim3(256),0,f.stream,f.view,f.view.csr.summary,rows,Fixture::H,output.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
            }
            hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,f.stream,f.view,f.view.csr.summary,rows,Fixture::H,
                output.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
            check(hipGetLastError(),"tiers complete tail");
        };
        run(0,0);
        const auto src=sources();TiersSnapshot before(src,f.stream);
        tail();auto after_src=src;after_src.emplace_back(output.base,output.bytes);after_src.emplace_back(record.base,record.bytes);
        TiersSnapshot after(after_src,f.stream);
        const unsigned tune=kr::prefill_last_tune();
        if(tune==80U)++tunes[0];else if(tune==82U)++tunes[1];else if(tune==88U)++tunes[2];else throw std::runtime_error("unexpected tiers tune");
        for(const auto& arm:std::array<std::array<int,2>,6>{{{0,0},{1,0},{2,0},{0,1},{1,1},{2,1}}}) {
            run(arm[0],arm[1]);
            const std::string tag=std::string(f.c.name)+" T"+std::to_string(rows)+" tune"+std::to_string(tune)+" cert"+std::to_string(cert)+" hdr"+std::to_string(hdr)+" arm"+std::to_string(arm[0])+std::to_string(arm[1]);
            before.compare(f.stream,tag+" producers");tail();after.compare(f.stream,tag+" tail");++comparisons;
        }
        if(cert){++certified;unsigned u=0,h=0;check(hipMemcpy(&u,f.admission.base,4U,hipMemcpyDeviceToHost),"tiers unsafe");
            check(hipMemcpy(&h,f.view.csr.summary,4U,hipMemcpyDeviceToHost),"tiers summary");unsafe_cases+=u!=0U?1U:0U;safe_cases+=u==0U&&h==0U?1U:0U;}
        f.view.unsafe=nullptr;f.view.route_bf16=false;
    };
    if(mode=="tiersbench") {
        if(argc!=5)throw std::invalid_argument("tiersbench <reps> <layer spec> <rows>");
        const int reps=std::stoi(argv[2]);const unsigned rows=static_cast<unsigned>(std::stoul(argv[4]));
        if(reps<=0||rows<48U||rows>16384U)throw std::invalid_argument("tiersbench reps/rows");
        const auto choices=[](const char* name,const char* fallback,int maximum) {
            const char* env=std::getenv(name);const std::string spec=env?env:fallback;
            std::vector<int> v;std::size_t at=0;
            for(;;) {
                const auto end=spec.find(',',at);const auto field=spec.substr(at,end==std::string::npos?end:end-at);
                if(field.size()!=1U||field[0]<'0'||field[0]>'0'+maximum)throw std::invalid_argument(std::string(name)+" invalid arm");
                const int n=field[0]-'0';
                if(std::find(v.begin(),v.end(),n)!=v.end())throw std::invalid_argument(std::string(name)+" duplicate arm");
                v.push_back(n);if(end==std::string::npos)break;at=end+1U;
            }
            if(std::find(v.begin(),v.end(),0)==v.end())throw std::invalid_argument(std::string(name)+" needs baseline 0");
            return v;
        };
        const auto tier_modes=choices("BENCH_TIERS","0,1,2",2),epi_modes=choices("BENCH_EPI","0,1",1);
        std::vector<std::array<int,2>> arms;
        // Baseline first, independent of the caller's ordering.
        arms.push_back({0,0});
        for(int epi:epi_modes)for(int tier:tier_modes)if(tier||epi)arms.push_back({tier,epi});
        const auto fl=tiers_parse(argv[3]);
        Case c{"tiersbench",512U,rows,static_cast<unsigned>(std::stoul(fl[4])),static_cast<unsigned>(std::stoul(fl[5])),0U,false,true,false,0U,1.5,
            fl[0].c_str(),std::stoull(fl[1]),std::stoull(fl[2]),std::stoull(fl[3])};
        if(const char* v=std::getenv("BENCH_FULL128");v&&v[0]=='1'){c.crafted=true;c.full128=true;c.skew=false;}
        if(const char* v=std::getenv("BENCH_UNIFORM");v&&v[0]=='1')c.skew=false;
        Fixture f(c);
        tiers_safe_weights(f);
        tiers_real_routing(f);
        // Gate exactly this layer and shape with both raw-dot storage forms.
        for(unsigned cert:{0U,1U,2U})for(int hdr:{0,1})gate(f,rows,3,cert,hdr,true);
        const char* storage=std::getenv("TIERS_STORAGE");
        if(storage&&std::strcmp(storage,"fp32")&&std::strcmp(storage,"bf16"))throw std::invalid_argument("TIERS_STORAGE must be fp32 or bf16");
        f.view.unsafe=static_cast<unsigned*>(f.admission.base);f.view.route_bf16=!storage||std::strcmp(storage,"fp32")!=0;
        std::printf("tiersbench storage %s; two warmups, %d mirrored pairs per arm\n",f.view.route_bf16?"bf16":"fp32",reps);
        Buffer output(rows,2U*Fixture::H),record(1,8U);
        hipEvent_t e0,e1;check(hipEventCreate(&e0),"tiers event");check(hipEventCreate(&e1),"tiers event");
        std::array<hipEvent_t,3> phases{};
        for(auto& event:phases)check(hipEventCreate(&event),"tiers phase event");
        kr::prefill_test_da16_events(&phases);
        const auto timed=[&](int on,int epi) {
            kr::set_prefill_tiers(on);kr::set_prefill_down_lds(epi);
            check(hipEventRecord(e0,f.stream),"tiers start");
            f.enqueue(3,rows);
            if(f.view.route_bf16)hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<true,true>),dim3(Fixture::H/256U,rows),dim3(256),0,f.stream,
                f.view,f.view.csr.summary,Fixture::H,f.c.experts,Fixture::K,output.rows<std::uint16_t>());
            else hipLaunchKernelGGL(candidate_merge_commit_kernel<true>,dim3(Fixture::H/256U,rows),dim3(256),0,f.stream,
                f.view,f.view.csr.summary,Fixture::H,f.c.experts,Fixture::K,output.rows<std::uint16_t>());
            hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,f.stream,f.view,f.view.csr.summary,rows,Fixture::H,
                output.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
            check(hipEventRecord(e1,f.stream),"tiers stop");check(hipEventSynchronize(e1),"tiers timed sync");
            unsigned status=0;check(hipMemcpy(&status,f.view.csr.summary,4U,hipMemcpyDeviceToHost),"tiers timed status");
            if(status||!kr::prefill_certificate_produced()||kr::prefill_tiers_ran()!=(on!=0&&kr::prefill_last_tune()==82U)||kr::prefill_down_lds_ran()!=(epi!=0))throw std::runtime_error("tiersbench failed/inactive path");
            if(kr::prefill_tiers_mode_ran()!=(kr::prefill_last_tune()==82U?on:0))throw std::runtime_error("tiersbench mode not exercised");
            unsigned unsafe=0;check(hipMemcpy(&unsafe,f.admission.base,4U,hipMemcpyDeviceToHost),"tiers timed certificate");
            if(unsafe)throw std::runtime_error("tiersbench requires a safe certificate");
            std::array<double,3> result{};float ms=0;
            check(hipEventElapsedTime(&ms,e0,e1),"tiers elapsed");result[0]=ms;
            check(hipEventElapsedTime(&ms,phases[0],phases[1]),"tiers gate/up elapsed");result[1]=ms;
            check(hipEventElapsedTime(&ms,phases[1],phases[2]),"tiers down elapsed");result[2]=ms;
            return result;
        };
        const auto med=[](std::vector<double> v){std::sort(v.begin(),v.end());const auto n=v.size();return (v[(n-1U)/2U]+v[n/2U])/2.0;};
        // One mirrored interleave, A B C ... C B A: every arm has the same
        // temporal centre in each round, extending ABBA to the entire arm set.
        // Events enclose the complete enqueue + ordered merge/commit/finish.
        for(unsigned warm=0;warm<2U;++warm)for(const auto& arm:arms)(void)timed(arm[0],arm[1]);
        using Metrics=std::array<std::vector<double>,3>;
        std::vector<Metrics> samples(arms.size()),paired(arms.size());
        constexpr const char* labels[3]{"complete","gate/up","down"};
        for(int r=0;r<reps;++r) {
            std::vector<std::array<double,3>> first(arms.size()),last(arms.size());
            for(std::size_t k=0;k<arms.size();++k)first[k]=timed(arms[k][0],arms[k][1]);
            for(std::size_t k=arms.size();k-->0U;)last[k]=timed(arms[k][0],arms[k][1]);
            for(std::size_t k=0;k<arms.size();++k)for(unsigned metric=0;metric<3U;++metric) {
                samples[k][metric].push_back(first[k][metric]);samples[k][metric].push_back(last[k][metric]);
                paired[k][metric].push_back((first[0][metric]+last[0][metric]-first[k][metric]-last[k][metric])/2.0);
                std::printf("tiersbench sample T%u round%d tiers%d epi%d %s: %.4f %.4f ms/call\n",rows,r,arms[k][0],arms[k][1],labels[metric],first[k][metric],last[k][metric]);
            }
        }
        for(std::size_t k=0;k<arms.size();++k)for(unsigned metric=0;metric<3U;++metric) {
            const double baseline=med(samples[0][metric]),ms=med(samples[k][metric]);
            std::printf("tiersbench T%u tune%u tiers%d epi%d reps%d n%zu %s: median %.4f ms/call baseline %.4f ms saving %.4f ms (%+.2f%%), paired %.4f ms\n",
                rows,kr::prefill_last_tune(),arms[k][0],arms[k][1],reps,samples[k][metric].size(),labels[metric],ms,baseline,baseline-ms,100.0*(baseline-ms)/baseline,med(paired[k][metric]));
        }
        kr::prefill_test_da16_events(nullptr);
        for(auto event:phases)check(hipEventDestroy(event),"tiers phase event destroy");
        check(hipEventDestroy(e0),"tiers event destroy");check(hipEventDestroy(e1),"tiers event destroy");reset();return 0;
    }
    for(unsigned g:{12U,13U})for(unsigned d:{7U,8U})for(unsigned flavor:{0U,1U,2U,3U,4U,5U}) {
        Fixture f({"tiers small/flavors",40U,48U,g,d,flavor,true,true,flavor==2U});
        tiers_safe_weights(f);
        for(unsigned cert:{0U,1U,2U})for(int hdr:{0,1})gate(f,48U,3,cert,hdr,false);
    }
    for(unsigned g:{12U,13U})for(unsigned d:{7U,8U}) {
        Fixture f({"tiers projection nonfinite",40U,48U,g,d,2U,false,true,false});
        for(unsigned cert:{0U,1U,2U})gate(f,48U,3,cert,1,false);
    }
    for(unsigned g:{12U,13U})for(unsigned d:{7U,8U}) {
        Fixture f({"tiers all fills 1..256",512U,8192U,g,d,4U,false,false,false});tiers_all_fills(f);tiers_safe_weights(f);
        // Explicit forms cover all tunes without changing the CSR cap for an A/B pair.
        for(int form:{51,53,59})gate(f,8192U,form,0U,1,false);
        for(unsigned cert:{1U,2U})for(int hdr:{0,1})gate(f,8192U,3,cert,hdr,true);
        Case nonfinite{"tiers partial projection nonfinite",64U,1024U,g,d,4U,false,false,false};nonfinite.crafted=true;
        Fixture fn(nonfinite);
        const unsigned short nan=0x7e00U;
        check(hipMemcpy(static_cast<unsigned char*>(fn.gate.base)+Fixture::I*kind_row_bytes(g,Fixture::H),&nan,2U,hipMemcpyHostToDevice),"tiers NaN gate header");
        for(unsigned cert:{0U,1U,2U})gate(fn,1024U,3,cert,1,false);
        for(unsigned kind:{0U,1U,2U,3U,4U,5U,6U,7U,8U}) {
            Case c{"tiers routes/failures",64U,1024U,g,d,kind==5U?3U:kind==6U?5U:4U,false,false,false};
            c.crafted=true;c.full128=kind==0U;c.bad_route=kind==1U;c.nan_input=kind==2U;
            Fixture fc(c);
            if(kind==7U||kind==8U) {
                const auto old_stride=fc.rawdots.stride;
                fc.rawdots=Buffer(fc.c.rows,old_stride+256U);
                fc.view.rawdots={reinterpret_cast<float*>(static_cast<unsigned char*>(fc.rawdots.base)+(kind==7U?4U:0U)),
                    kind==8U?old_stride+4U:fc.rawdots.stride};
            }
            tiers_safe_weights(fc);
            if(kind==3U){const float bad=std::nextafter(1.0F,2.0F);check(hipMemcpy(fc.weights.base,&bad,4U,hipMemcpyHostToDevice),"tiers unproven certificate");}
            if(kind==4U){const float bad=std::numeric_limits<float>::quiet_NaN();check(hipMemcpy(fc.weights.base,&bad,4U,hipMemcpyHostToDevice),"tiers nonfinite route weight");}
            for(unsigned cert:{1U,2U})gate(fc,1024U,3,cert,1,false);
        }
    }
    // Large skew/uniform/full128, and a decreasing/increasing tail ladder to catch
    // stale lists, counters and DA16 images. The empty list and padded CTA slots
    // must return before touching any output on every shape.
    for(unsigned kind:{0U,1U,2U}) {
        Case c{"tiers large/tail ladder",512U,max_rows,12U,8U,4U,false,kind==0U,false,0U,1.5};
        if(kind==2U){c.crafted=true;c.full128=true;}
        Fixture f(c);
        tiers_safe_weights(f);
        for(unsigned rows:{16384U,8192U,4864U,4096U,3072U,2048U,1320U,208U,65U,64U,48U,16384U})
            if(rows<=max_rows)
            for(unsigned cert:{1U,2U})gate(f,rows,3,cert,1,rows>=3072U&&rows<4916U);
    }
    for(int a=2;a<argc;++a) {
        const auto fl=tiers_parse(argv[a]);
        Case c{"tiers real layer",512U,max_rows,static_cast<unsigned>(std::stoul(fl[4])),static_cast<unsigned>(std::stoul(fl[5])),0U,false,true,false,0U,1.5,
            fl[0].c_str(),std::stoull(fl[1]),std::stoull(fl[2]),std::stoull(fl[3])};
        Fixture f(c);
        tiers_safe_weights(f);
        tiers_real_routing(f);
        for(unsigned rows:{4096U,8192U,16384U})if(rows<=max_rows)for(unsigned cert:{0U,1U,2U})for(int hdr:{0,1})gate(f,rows,3,cert,hdr,rows==4096U);
    }
    reset();
    if(!comparisons||!tunes[0]||!tunes[1]||!tunes[2]||!certified||!safe_cases||!unsafe_cases)throw std::runtime_error("tiers missing required coverage");
    std::printf("tiers: %u exact byte comparisons, tune80/82/88 %u/%u/%u, %u certified, %u safe, %u unsafe; PASS\n",comparisons,tunes[0],tunes[1],tunes[2],certified,safe_cases,unsafe_cases);
    return 0;
}
