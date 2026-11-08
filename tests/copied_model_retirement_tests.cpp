#include "strix/copied_stream_transport.hpp"
#include "strix/copied_host_channel.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <deque>
#include <iostream>
#include <cstring>
using namespace strix;
using namespace strix::transport;
namespace {
std::uint64_t checks=0;
void check(bool b) {++checks;if(!b) throw std::runtime_error("copied retirement check failed");}
template<class F> bool rejects(F f) {try {f();}catch(const std::exception&){return true;}return false;}
// Mock host transport and consumers only. No GPU/native/model gate.
struct Pipe {
    std::mutex mutex;std::condition_variable changed;
    std::deque<std::vector<std::byte>> incoming[2];bool closed=false;
};
struct MockTransport {
    std::shared_ptr<Pipe> pipe;unsigned rank;
    void send(std::span<const std::byte> b) {
        std::lock_guard lock(pipe->mutex);if(pipe->closed)throw std::runtime_error("closed mock");
        pipe->incoming[1U-rank].emplace_back(b.begin(),b.end());pipe->changed.notify_all();
    }
    void receive(std::span<std::byte> b) {
        std::unique_lock lock(pipe->mutex);
        if(!pipe->changed.wait_for(lock,std::chrono::seconds(2),[&]{return pipe->closed || !pipe->incoming[rank].empty();}))throw std::runtime_error("mock timeout");
        if(pipe->closed)throw std::runtime_error("closed mock");
        auto v=std::move(pipe->incoming[rank].front());pipe->incoming[rank].pop_front();
        if(v.size()!=b.size())throw std::runtime_error("mock truncated");
        std::copy(v.begin(),v.end(),b.begin());
    }
    void abort(const std::string&) {std::lock_guard lock(pipe->mutex);pipe->closed=true;pipe->changed.notify_all();}
    std::string name() const {return "MOCK-HOST-ONLY";}
};
using Channel=CopiedHostByteChannel<MockTransport>;
struct Local final : DistributedNativeLocalExecutor {void execute_local_step(const DistributedStepPlacement&) override {}};
struct Control final : DistributedNativeControlPlane {
    void execute_remote_step(const DistributedStepPlacement&) override {}
    void publish_boundary_transfer(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer&) override {}
    DistributedBoundaryTransfer await_boundary_transfer(const DistributedExecutionBoundary&) override {throw std::runtime_error("unused");}
};
struct Source final : CopiedStreamBoundarySourceProvider {
    CopiedStreamOwnedBuffer source;
    CopiedStreamPreparedSource stage_outbound_owned(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer& t) override {
        return {source,make_copied_stream_synchronous_proof({t.descriptor,source.view,CopiedStreamCompletionScope::producer_released})};
    }
};
struct Done final : DistributedInboundCompletion {DistributedInboundCompletionStatus poll() override {return DistributedInboundCompletionStatus::complete;}};
struct Consumer final : DistributedBoundaryStagingProvider {
    std::vector<std::byte> actual;bool fault=false;int cleanups=0;
    TransportBufferView stage_outbound(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer&) override {throw std::runtime_error("unused");}
    TransportBufferView stage_inbound(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer&) override {throw std::runtime_error("unused");}
    void consume_inbound(const DistributedExecutionBoundary&,const TransportBufferView& v) override {
        actual.assign(reinterpret_cast<const std::byte*>(v.address),reinterpret_cast<const std::byte*>(v.address)+v.bytes);
        if(fault)throw std::runtime_error("mock consumer failure after synchronous copy");
    }
    std::unique_ptr<DistributedInboundCompletion> finish_inbound(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer&) override {return std::make_unique<Done>();}
    std::unique_ptr<DistributedInboundCompletion> cleanup_inbound(const DistributedExecutionBoundary&,const DistributedBoundaryTransfer&) override {++cleanups;return std::make_unique<Done>();}
};
DistributedPlacementPlan plan(unsigned node) {
    DistributedPlacementPlan p;p.plan_identity=31;p.local_node=node;p.executable=true;p.has_remote_work=true;
    p.steps={{0,SerialOp::token_embedding,0,node!=0,false},{1,SerialOp::lm_head,1,node!=1,false}};
    p.boundaries={{91,0,1,0,1,16,{{{ExecutionBufferClass::expanded_hidden,0},1}}}};return p;
}
struct Pair {
    std::shared_ptr<Pipe> pipe=std::make_shared<Pipe>();
    ModelContentIdentity manifest=sha256_bytes(std::array<std::byte,1>{std::byte{9}});
    Channel left{std::make_unique<MockTransport>(MockTransport{pipe,0}),0,9,1,manifest,4096};
    Channel right{std::make_unique<MockTransport>(MockTransport{pipe,1}),1,9,1,manifest,4096};
    CopiedStreamConnectionState ls{{"mock-only",9,0,1,0}},rs{{"mock-only",9,1,0,0}};
    CopiedStreamTransportBackend tx{left,ls,{128,4096,4096}},rx{right,rs,{128,4096,4096}};
    AllocationGenerationSource generations;
};
}
int main() {
#ifdef NDEBUG
    std::cout<<"NDEBUG=1 ";
#else
    std::cout<<"NDEBUG=0 ";
#endif
    for(unsigned fault=0;fault<2;++fault) {
        Pair pair;auto lp=plan(0),rp=plan(1);Local local;Control control;Source source;Consumer consumer;consumer.fault=fault!=0;
        auto storage=std::make_shared<CopiedStreamHostBuffer>(16,pair.generations);source.source={storage,storage->view()};
        for(std::size_t i=0;i<16;++i)storage->writable_bytes()[i]=static_cast<std::byte>(i);
        auto pin=std::make_shared<int>(1);std::weak_ptr<int> weak=pin;
        std::optional<NativeDistributedTokenRetirement> last;
        for(std::uint64_t generation=1;generation<=5;++generation) {
            DistributedRuntimeRequest req{9,11,generation,7};
            DistributedControlReceiver lq(rp,req,0,local),rq(lp,req,1,local);
            DistributedBoundaryTransfer t;t.descriptor=make_distributed_boundary_descriptor(lp,lp.boundaries[0],9,11,generation,1,7);
            t.flow=classify_distributed_boundary_flow(lp,lp.boundaries[0]);t.link_plan.kind=PlanKind::single_link;t.link_plan.slices={{0,0,16}};
            rq.receive_boundary_transfer(t);
            CopiedStreamDistributedBackend a(pair.tx,lp,req,control,lq,local,source,consumer,pair.generations);
            CopiedStreamDistributedBackend b(pair.rx,rp,req,control,rq,local,source,consumer,pair.generations);
            const auto binding=100+generation;a.begin_token_retirement(binding,pin);b.begin_token_retirement(binding,pin);
            check(rejects([&]{a.begin_token_retirement(binding,pin);}));
            std::exception_ptr error;std::thread peer([&]{try{(void)b.receive_boundary(rp.boundaries[0]);}catch(...){error=std::current_exception();pair.right.poison();}});
            bool send_failed=false;try{a.send_boundary(lp.boundaries[0],t);}catch(...){send_failed=true;}
            peer.join();
            if(fault) {
                check(send_failed && error && a.failed() && b.failed());check(consumer.cleanups==1);
                check(rejects([&]{(void)a.retire_token();}));check(rejects([&]{(void)b.retire_token();}));
                check(pair.rx.drain_retained()==CopiedStreamCleanupStatus::released);
                check(pair.tx.drain_retained()==CopiedStreamCleanupStatus::released);break;
            }
            if(error)std::rethrow_exception(error);
            check(!send_failed);check(consumer.actual==std::vector<std::byte>(storage->bytes().begin(),storage->bytes().end()));
            auto left=a.retire_token();auto right=b.retire_token();
            left.validate_matches(binding,31,req,0,NativeDistributedRetirementOutcome::completed_boundaries);
            right.validate_matches(binding,31,req,1,NativeDistributedRetirementOutcome::completed_boundaries);check(true);
            check(rejects([&]{(void)a.retire_token();}));check(rejects([&]{a.execute_remote_step(lp.steps[1]);}));
            for(unsigned changed=0;changed<8;++changed) {
                auto x=req;if(changed==0)++x.connection_epoch;if(changed==1)++x.request_id;if(changed==2)++x.generation;if(changed==3)++x.dtype_id;
                check(rejects([&]{left.validate_matches(binding+(changed==4?1U:0U),31+(changed==5?1U:0U),x,changed==6?1U:0U,
                    changed==7?NativeDistributedRetirementOutcome::aborted_attempt:NativeDistributedRetirementOutcome::completed_boundaries);}));
            }
            auto moved=std::move(left);check(rejects([&]{left.validate_matches(binding,31,req,0,NativeDistributedRetirementOutcome::completed_boundaries);}));
            moved.validate_matches(binding,31,req,0,NativeDistributedRetirementOutcome::completed_boundaries);
            last.emplace(std::move(moved));
            check(!pair.tx.retains_buffer() && !pair.rx.retains_buffer());
        }
        if(!fault) {
            pair.tx.poison();
            check(rejects([&]{last->validate_matches(105,31,{9,11,5,7},0,NativeDistributedRetirementOutcome::completed_boundaries);}));
            pin.reset();check(!weak.expired());last.reset();check(weak.expired());
        }
    }
    for(unsigned fault=0;fault<5;++fault) {
        Pair pair;auto p=plan(0),other=plan(1);Local local;Control control;Source source;Consumer consumer;
        if(fault==0)p.boundaries[0].boundary_id=0;
        if(fault==1)p.boundaries.push_back(p.boundaries[0]);
        if(fault==2){p.boundaries[0].payload_bytes=0;p.boundaries[0].payloads.clear();}
        DistributedControlReceiver queue(other,{9,11,1,7},0,local);
        CopiedStreamDistributedBackend a(pair.tx,p,{9,11,1,7},control,queue,local,source,consumer,pair.generations);
        auto pin=std::make_shared<int>(2);
        if(fault<=2) {check(rejects([&]{a.begin_token_retirement(101,pin);}));check(!pair.tx.retains_buffer());}
        else {
            a.begin_token_retirement(101,pin);
            if(fault==3) {a.execute_local_step(p.steps[0]);check(rejects([&]{(void)a.retire_token();}));}
            else check(rejects([&]{a.complete_position_dependencies(NativeDistributedPositionCompletion{});}));
            check(a.failed() && pair.tx.failed());check(rejects([&]{a.execute_local_step(p.steps[0]);}));
        }
    }
    {
        NativeDistributedTokenRetirement empty;
        check(rejects([&]{empty.validate_matches(101,31,{9,11,1,7},0,NativeDistributedRetirementOutcome::completed_boundaries);}));
    }
    std::cout<<"PASS "<<checks<<" copied retirement/bilateral/continuation checks; MOCK consumers; NONnative\n";
}
