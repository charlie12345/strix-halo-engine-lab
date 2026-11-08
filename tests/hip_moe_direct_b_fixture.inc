// Included after Fixture. Every comparison is over complete byte arrays, including
// padding poison, unwritten rows, maxima and status; digests are not a pass gate.
namespace qp=strix::qwen_moe_prepared;
qp::PreparedTensor prepare_direct_b_fixture(const Buffer& raw,unsigned experts,qp::Role role) {
    const qp::Spec spec{qp::Format::q4_k,role,experts,Fixture::I,Fixture::H};
    auto source=qwen_moe_prepared_fixture::source();
    source.file_offset=0;source.tensor_index=static_cast<unsigned>(role);
    return qp::prepare(spec,source,raw.bytes,[&](std::uint64_t at,std::span<std::byte> bytes) {
        check(hipMemcpy(bytes.data(),static_cast<const char*>(raw.base)+at,bytes.size(),hipMemcpyDeviceToHost),"read fixture preparation slab");
    },qp::allocate_hip);
}
// Deterministic certificate passes/failures, signed zeros, subnormals and FP16
// midpoint-producing coefficients. These augment the randomized tensor body.
__global__ void direct_b_edges(unsigned char* bytes,unsigned experts,unsigned outputs,unsigned hidden) {
    const unsigned k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=experts*outputs)return;
    const unsigned short d[8]{0x0000U,0x8000U,0x0001U,0x03ffU,0x7bffU,0x1401U,0x0400U,0x9800U};
    const unsigned short m[8]{0x8000U,0x0000U,0x03ffU,0x0001U,0x0001U,0x0001U,0x0400U,0x1800U};
    auto* block=bytes+static_cast<std::uint64_t>(k)*(hidden/256U)*144U;
    // First superblock of each output; every output code position participates.
    reinterpret_cast<unsigned short*>(block)[0]=d[k%8U];
    reinterpret_cast<unsigned short*>(block)[1]=m[k%8U];
    if(k%8U==5U) { // odd mantissa and tiny subtraction exercise FP16 ties
        for(unsigned j=4;j<16;++j)block[j]=1U;
    }
}
void direct_b_fragments(bool paired_planes) {
    constexpr unsigned pairs=256U,bytes=pairs*2U*2304U;
    auto raw=qwen_moe_prepared_fixture::input(bytes,71U);std::vector<std::byte> packed(bytes);
    for(unsigned unit=0;unit<pairs*2U;++unit) {
        auto block=std::span(raw).subspan(unit*2304U,2304U);
        for(unsigned row=0;row<16;++row) {
            const unsigned short heads[8]{0x0000U,0x8000U,0x0001U,0x03ffU,0x0400U,0x1401U,0x7bffU,0x1bffU};
            const unsigned short d=unit%2U?heads[(unit/2U)%8U]:0x1801U;
            const unsigned short m=unit%2U?heads[((unit/2U)+3U)%8U]:0x1803U;
            std::memcpy(block.data()+144U*row,&d,2);std::memcpy(block.data()+144U*row+2,&m,2);
            for(unsigned k=0;k<128;++k)block[144U*row+16U+k]=std::byte((unit+k+row)%256U);
        }
        // Pin one nonzero slope and zero minimum for the fault-injection check.
        if(unit==0U){block[144U*5U+5U]=std::byte{1};block[144U*5U+9U]=std::byte{0};}
        qp::repack_unit(qp::Format::q4_k,block,std::span(packed).subspan(unit*2304U,2304U));
    }
    Buffer original(1,bytes),prepared(1,bytes);
    check(hipMemcpy(original.base,raw.data(),bytes,hipMemcpyHostToDevice),"fragment source upload");
    check(hipMemcpy(prepared.base,packed.data(),bytes,hipMemcpyHostToDevice),"fragment prepared upload");
    const auto result=kr::direct_b_fragment_test(static_cast<const unsigned char*>(original.base),static_cast<const unsigned char*>(prepared.base),pairs,paired_planes);
    constexpr unsigned long long elements=static_cast<unsigned long long>(pairs)*2U*16U*2U*16U*16U;
    if(result.mismatched_elements)std::fprintf(stderr,
        "direct-b first mismatch: unit=%u matrix=%s lane=%u replica=%u hardware_lane=%u plane=%u element=%u K=%u expected_fp16_bits=0x%04x actual_fp16_bits=0x%04x\n",
        result.unit,result.unit%2U?"up":"gate",result.lane,result.replica,result.lane+16U*result.replica,
        result.plane,result.element,result.plane*16U+result.element,result.expected,result.actual);
    if(result.mismatched_elements||!result.pass_lanes||!result.fallback_lanes)throw std::runtime_error(
        "direct B device operand bytes / certificate coverage failed: mismatched_elements="+std::to_string(result.mismatched_elements)+
        " compared_elements="+std::to_string(elements)+" compared_bytes="+std::to_string(2U*elements)+
        " pass_lanes="+std::to_string(result.pass_lanes)+" fallback_lanes="+std::to_string(result.fallback_lanes));
    // A single prepared nibble at (unit 0, column 5, K37) changes 0 to 1.
    // Both hardware replicas must detect it, including its nonzero element
    // index, and the deterministic first record must retain the matching bits.
    constexpr unsigned corrupt=256U+(16U+5U)*16U+2U;
    packed[corrupt]^=std::byte{0x10};
    check(hipMemcpy(static_cast<unsigned char*>(prepared.base)+corrupt,packed.data()+corrupt,1,hipMemcpyHostToDevice),"fragment fault upload");
    const auto fault=kr::direct_b_fragment_test(static_cast<const unsigned char*>(original.base),static_cast<const unsigned char*>(prepared.base),pairs,paired_planes);
    if(fault.mismatched_elements!=2U||fault.unit!=0U||fault.lane!=5U||fault.replica!=0U||fault.plane!=2U||fault.element!=5U||
        fault.expected!=0U||fault.actual!=0x1801U||fault.pass_lanes!=result.pass_lanes||fault.fallback_lanes!=result.fallback_lanes)
        throw std::runtime_error("direct B fragment fault-injection diagnostic failed");
    std::printf("direct-b fragments: compared_elements=%llu compared_bytes=%llu; every gate/up lane, replica and K16 plane; paired_planes=%u pass_lanes=%llu fallback_lanes=%llu; fault diagnostic PASS\n",
        elements,2U*elements,static_cast<unsigned>(paired_planes),result.pass_lanes,result.fallback_lanes);
}
void direct_b_fixture(bool timing,int repeats) {
    if(repeats<1)throw std::invalid_argument("direct-b-timing repeats must be positive");
    kr::set_prefill_core2(0);kr::set_prefill_small(0);kr::set_prefill_route(14);kr::set_prefill_hdr(1);kr::set_prefill_xor64(1);
    unsigned cases=0;
    direct_b_fragments(false);direct_b_fragments(true);
    const auto certificate=kr::fast_dequant_selftest(0x12345678ULL,4096);
    if(!certificate[1]||certificate[1]>=certificate[0]||certificate[2])throw std::runtime_error("affine certificate/fallback fixture did not cover both paths exactly");
    for(unsigned T:{4096U,8192U})for(unsigned flavor:timing?std::vector<unsigned>{4U}:std::vector<unsigned>{4U,0U,1U,2U}) {
        Case c{"direct-b",512U,T,12U,7U,flavor,false,true,flavor==2U,0U,1.5};
        c.crafted=flavor!=4U; // row16/64/128 tails, empty experts and masked rows
        Fixture f(c);
        if(flavor==1U) {
            hipLaunchKernelGGL(direct_b_edges,dim3((c.experts*Fixture::I+255U)/256U),dim3(256),0,f.stream,
                static_cast<unsigned char*>(f.gate.base),c.experts,Fixture::I,Fixture::H);
            hipLaunchKernelGGL(direct_b_edges,dim3((c.experts*Fixture::I+255U)/256U),dim3(256),0,f.stream,
                static_cast<unsigned char*>(f.up.base),c.experts,Fixture::I,Fixture::H);
            check(hipGetLastError(),"direct B adversarial weights");
        }
        check(hipStreamSynchronize(f.stream),"direct B preparation boundary");
        auto gate=prepare_direct_b_fixture(f.gate,c.experts,qp::Role::gate);
        auto up=prepare_direct_b_fixture(f.up,c.experts,qp::Role::up);
        f.direct.gate=reinterpret_cast<const unsigned char*>(gate.allocation->address());
        f.direct.up=reinterpret_cast<const unsigned char*>(up.allocation->address());
        // Exercise both production DA16 and ordinary down consumers.
        for(int da16:timing?std::vector<int>{1}:std::vector<int>{0,1}) {
            kr::set_prefill_da16(da16);
            f.view.unsafe=da16?static_cast<unsigned*>(f.admission.base):nullptr;
            kr::set_prefill_direct_b(0);const auto expected=f.run(3);
            const auto unsafe=f.admission.read();
            if(kr::prefill_direct_b_ran())throw std::runtime_error("direct B OFF engaged");
            kr::set_prefill_direct_b(1);const auto got=f.run(3);
            if(!kr::prefill_direct_b_ran())throw std::runtime_error("direct B did not engage (resource ceiling or selector)");
            if(da16 && f.admission.read()!=unsafe)throw std::runtime_error("direct B admission certificate differs");
            if(got!=expected)throw std::runtime_error("direct B output/status/maxima byte mismatch");
            ++cases;
            // Complete CSR receipt bytes, not just its summary word.
            const auto csr=f.csr.read();
            kr::set_prefill_direct_b(0);f.enqueue(3);
            check(hipStreamSynchronize(f.stream),"direct B reference CSR");
            if(f.csr.read()!=csr)throw std::runtime_error("direct B CSR differs");
        }
        if(!timing) {
            // Missing assets and a non-covered prompt tail retain the old launch.
            kr::set_prefill_direct_b(1);const auto saved=f.direct.up;f.direct.up=nullptr;
            f.enqueue(3);check(hipStreamSynchronize(f.stream),"direct B missing asset decline");
            if(kr::prefill_direct_b_ran())throw std::runtime_error("missing prepared up accepted");
            f.direct.up=saved;f.enqueue(3,T-8U);
            check(hipStreamSynchronize(f.stream),"direct B tail decline");
            if(kr::prefill_direct_b_ran())throw std::runtime_error("unsupported prompt tail accepted");
        } else {
            // Resolve the production effective tune (including XOR rows).
            // Both timed kernels include the last-CTA DA16 completion.
            kr::set_prefill_da16(1);kr::set_prefill_direct_b(0);f.enqueue(3);
            check(hipStreamSynchronize(f.stream),"direct B timing setup");
            const unsigned tune=kr::prefill_last_tune();
            unsigned batches=0;check(hipMemcpy(&batches,f.view.csr.summary+2,4,hipMemcpyDeviceToHost),"timing batch count");
            Buffer arrive(1,static_cast<std::uint64_t>(batches)*sizeof(unsigned));
            Buffer image(1,static_cast<std::uint64_t>(T)*Fixture::K*Fixture::I*2U);
            const auto reset=[&] {
                check(hipMemsetAsync(arrive.base,0,arrive.bytes,f.stream),"timing arrival reset");
                check(hipMemsetAsync(f.dsums.base,0,f.dsums.bytes,f.stream),"timing maxima reset");
            };
            const auto project=[&](bool direct) {
                kr::prefill_project_test(f.view,Fixture::H,Fixture::I,static_cast<const unsigned char*>(f.gate.base),
                    static_cast<const unsigned char*>(f.up.base),batches,tune,direct,f.stream,f.direct,
                    static_cast<unsigned*>(arrive.base),static_cast<uint4*>(image.base));
            };
            // Byte-gate these exact kernels, including the complete down image.
            const auto project_bytes=[&](bool direct) {
                reset();f.output.poison(f.stream);image.poison(f.stream);
                project(direct);
                check(hipStreamSynchronize(f.stream),"direct B timing byte gate");
                std::vector<std::vector<unsigned char>> out;
                for(const auto* buffer:f.written())out.push_back(buffer->read());
                out.push_back(f.csr.read());out.push_back(image.read());return out;
            };
            const auto expected=project_bytes(false);
            if(project_bytes(true)!=expected)throw std::runtime_error("direct B timed DA16 gate/up output/status/maxima/image byte mismatch");
            hipEvent_t start{},end{};check(hipEventCreate(&start),"event");check(hipEventCreate(&end),"event");
            const auto once=[&](bool direct) {
                reset(); // identical preparation outside the measured interval
                check(hipEventRecord(start,f.stream),"timing start");
                project(direct);
                check(hipEventRecord(end,f.stream),"timing end");check(hipEventSynchronize(end),"timing sync");
                float ms=0;check(hipEventElapsedTime(&ms,start,end),"timing elapsed");return ms;
            };
            once(false);once(true);
            std::vector<float> old_ms,new_ms;
            for(int r=0;r<repeats;++r) {old_ms.push_back(once(false));new_ms.push_back(once(true));new_ms.push_back(once(true));old_ms.push_back(once(false));}
            const auto median=[](std::vector<float> v){std::sort(v.begin(),v.end());return (v[(v.size()-1U)/2U]+v[v.size()/2U])*0.5F;};
            std::printf("direct-b-timing T=%u E=512 K=2560 N=640 tune=%u repeats=%d production_da16_mg_ms=%.6f direct_b_da16_ms=%.6f; gate/up plus completion; old/new/new/old medians; setup excluded\n",
                T,tune,repeats,median(old_ms),median(new_ms));
            check(hipEventDestroy(start),"event destroy");check(hipEventDestroy(end),"event destroy");
        }
        std::printf("direct-b T=%u flavor=%u every output/status/maxima byte PASS\n",T,flavor);
    }
    std::printf("direct-b cases=%u; native operator fixture only; model/workload qualification pending\n",cases);
    kr::set_prefill_direct_b(-1);kr::set_prefill_da16(-1);kr::set_prefill_core2(-1);kr::set_prefill_small(-1);
    kr::set_prefill_route(-1);kr::set_prefill_hdr(-1);kr::set_prefill_xor64(-1);
}
