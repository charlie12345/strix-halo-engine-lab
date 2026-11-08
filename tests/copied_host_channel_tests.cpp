#include "strix/copied_host_channel.hpp"
#include <deque>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace strix;
namespace {
std::uint64_t checks=0;
void check(bool b) {++checks;if(!b) throw std::runtime_error("host byte channel check failed");}
template<class F> bool rejects(F f) {try {f();} catch(const std::exception&) {return true;}return false;}
struct Pipe {std::deque<std::vector<std::byte>> incoming[2];int aborts=0;};
// MOCK ONLY: no native, physical, provider or performance qualification.
struct MockTransport {
    std::shared_ptr<Pipe> pipe;unsigned rank;
    void send(std::span<const std::byte> b) {pipe->incoming[1U-rank].emplace_back(b.begin(),b.end());}
    void receive(std::span<std::byte> b) {
        auto& q=pipe->incoming[rank];if(q.empty()) throw std::runtime_error("mock short channel");
        auto row=std::move(q.front());q.pop_front();if(row.size()!=b.size()) throw std::runtime_error("mock truncated message");
        std::copy(row.begin(),row.end(),b.begin());
    }
    void abort(const std::string&) {++pipe->aborts;}
    std::string name() const {return "MOCK-HOST-ONLY";}
};
// Real host worker held past an injected callback exception. Destruction
// joins it while the channel's borrowed header/payload remains alive.
struct WorkerState {
    std::mutex mutex;std::condition_variable changed;bool release=false,retired=false,bytes_valid=false;
};
struct ThrowingWorkerTransport {
    std::shared_ptr<WorkerState> state;
    std::deque<std::vector<std::byte>> input;
    std::thread worker;
    unsigned call=0,fail_call=1;bool writing=false;
    ThrowingWorkerTransport(std::shared_ptr<WorkerState> value,unsigned fail,bool write,
        std::deque<std::vector<std::byte>> frames):state(std::move(value)),input(std::move(frames)),fail_call(fail),writing(write) {}
    ~ThrowingWorkerTransport() {
        {std::lock_guard lock(state->mutex);state->release=true;state->changed.notify_all();}
        if(worker.joinable())worker.join();
        state->retired=true;
    }
    void launch(std::span<const std::byte> view,std::vector<std::byte> expected) {
        worker=std::thread([this,view,expected=std::move(expected)] {
            {std::unique_lock lock(state->mutex);state->changed.wait(lock,[&]{return state->release;});}
            state->bytes_valid=std::equal(view.begin(),view.end(),expected.begin(),expected.end());
        });throw std::runtime_error("injected callback exception with a live host worker");
    }
    void send(std::span<const std::byte> view) {
        if(!writing)throw std::runtime_error("wrong test direction");
        if(++call==fail_call)launch(view,{view.begin(),view.end()});
    }
    void receive(std::span<std::byte> view) {
        if(writing || input.empty())throw std::runtime_error("wrong test direction");
        auto frame=std::move(input.front());input.pop_front();if(view.size()!=frame.size())throw std::runtime_error("wrong frame size");
        std::copy(frame.begin(),frame.end(),view.begin());if(++call==fail_call)launch(view,std::move(frame));
    }
    void abort(const std::string&) {}
    std::string name() const {return "MOCK-THROWING-HOST-WORKER";}
};
using Channel=CopiedHostByteChannel<MockTransport>;
std::unique_ptr<MockTransport> mock_transport(std::shared_ptr<Pipe> p,unsigned r) {return std::make_unique<MockTransport>(MockTransport{p,r});}
ModelContentIdentity manifest() {return sha256_bytes(std::array<std::byte,1>{std::byte{7}});}
}
int main() {
#ifdef NDEBUG
    std::cout<<"NDEBUG=1 ";
#else
    std::cout<<"NDEBUG=0 ";
#endif
    const std::vector<std::byte> data{std::byte{1},std::byte{2},std::byte{0},std::byte{255}};
    for(unsigned direction=0;direction<2;++direction) {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,direction),direction,19,3,manifest(),1024);
        Channel b(mock_transport(p,1U-direction),1U-direction,19,3,manifest(),1024);
        for(unsigned i=0;i<9;++i) {a.send(data);check(b.receive()==data);b.send(data);check(a.receive()==data);}
        check(a.transport_identity()=="host-copy-sha256-v1:MOCK-HOST-ONLY");
    }
    for(std::size_t offset=0;offset<128;++offset) {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,0),0,19,3,manifest(),1024),b(mock_transport(p,1),1,19,3,manifest(),1024);
        a.send(data);p->incoming[1].front()[offset]^=std::byte{1};
        check(rejects([&]{(void)b.receive();}));check(b.failed() && p->aborts==1);
        check(rejects([&]{b.send(data);}));check(rejects([&]{(void)b.receive();}));
    }
    for(std::size_t cut=0;cut<128;++cut) {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,0),0,19,3,manifest(),1024),b(mock_transport(p,1),1,19,3,manifest(),1024);
        a.send(data);p->incoming[1].front().resize(cut);check(rejects([&]{(void)b.receive();}));check(b.failed());
    }
    for(std::size_t offset=0;offset<data.size();++offset) {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,0),0,19,3,manifest(),1024),b(mock_transport(p,1),1,19,3,manifest(),1024);
        a.send(data);p->incoming[1][1][offset]^=std::byte{1};check(rejects([&]{(void)b.receive();}));check(b.failed());
    }
    for(std::size_t cut=0;cut<data.size();++cut) {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,0),0,19,3,manifest(),1024),b(mock_transport(p,1),1,19,3,manifest(),1024);
        a.send(data);p->incoming[1][1].resize(cut);check(rejects([&]{(void)b.receive();}));
    }
    for(unsigned fault=0;fault<5;++fault) {
        auto p=std::make_shared<Pipe>();auto m=manifest();if(fault==3)m.bytes[0]^=1;
        Channel a(mock_transport(p,0),0,19,3,manifest(),1024),b(mock_transport(p,1),fault==0?0U:1U,fault==1?20U:19U,fault==2?4U:3U,m,fault==4?1U:1024U);
        a.send(data);check(rejects([&]{(void)b.receive();}));check(b.failed());
    }
    {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,0),0,19,3,manifest(),1024),b(mock_transport(p,1),1,19,3,manifest(),1024);
        a.send(data);const auto replay=p->incoming[1];check(b.receive()==data);p->incoming[1]=replay;
        check(rejects([&]{(void)b.receive();}));check(b.failed());
    }
    {
        auto p=std::make_shared<Pipe>();Channel a(mock_transport(p,0),0,19,3,manifest(),1);
        check(rejects([&]{a.send(data);}));check(a.failed());check(p->incoming[1].empty());
    }
    for(unsigned fault=0;fault<5;++fault) {
        auto p=std::make_shared<Pipe>();auto m=manifest();if(fault==4)m={};
        check(rejects([&]{Channel bad(mock_transport(p,0),fault==0?2U:0U,fault==1?0U:19U,fault==2?0U:3U,m,fault==3?0U:1024U);}));
    }
    for(bool writing:{false,true}) for(unsigned failure:{1U,2U}) {
        auto seed=std::make_shared<Pipe>();Channel producer(mock_transport(seed,0),0,19,3,manifest(),1024);producer.send(data);
        auto state=std::make_shared<WorkerState>();
        CopiedHostByteChannel<ThrowingWorkerTransport> actual(
            std::make_unique<ThrowingWorkerTransport>(state,failure,writing,seed->incoming[1]),
            writing?0U:1U,19,3,manifest(),1024);
        check(rejects([&]{if(writing)actual.send(data);else (void)actual.receive();}));
        check(actual.failed());check(state->retired && state->bytes_valid);
        actual.poison();check(state->retired); // idempotent after transport destruction
    }
    std::cout<<"PASS "<<checks<<" actual host channel checks; MOCK transport; NONnative\n";
}
