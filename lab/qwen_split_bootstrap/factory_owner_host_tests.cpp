// NONnative ownership test of the ACTUAL factory, compiled with declaration-only
// HIP headers and mock project Transport implementations. No GPU/model/network.
#include "strix/hip_copied_host_sequence.hpp"
#include <iostream>
using namespace strix;
namespace {
unsigned checks=0;
void check(bool b) {++checks;if(!b)throw std::runtime_error("actual factory ownership check failed");}
struct Observation {unsigned destroyed=0,aborted=0;};
class Mock final : public dist::Transport {
    std::shared_ptr<Observation> observed_;
public:
    explicit Mock(std::shared_ptr<Observation> o):observed_(std::move(o)) {}
    ~Mock() override {++observed_->destroyed;}
    void send(std::span<const std::byte>) override {throw std::logic_error("no I/O in ownership test");}
    void receive(std::span<std::byte>) override {throw std::logic_error("no I/O in ownership test");}
    void exchange(std::span<const std::byte>,std::span<std::byte>) override {throw std::logic_error("no I/O in ownership test");}
    void abort(const std::string&) override {++observed_->aborted;}
    std::string name() const override {return "MOCK-NONNATIVE-OWNERSHIP";}
};
}
int main() {
#ifdef NDEBUG
    std::cout<<"NDEBUG=1 ";
#else
    std::cout<<"NDEBUG=0 ";
#endif
    std::array<std::shared_ptr<Observation>,4> seen;
    for(auto& o:seen)o=std::make_shared<Observation>();
    HipCopiedHostChannels inputs;inputs.payload=std::make_unique<Mock>(seen[0]);inputs.step=std::make_unique<Mock>(seen[1]);
    inputs.token=std::make_unique<Mock>(seen[2]);inputs.greedy=std::make_unique<Mock>(seen[3]);
    inputs.rank=0;inputs.epoch=19;inputs.agreed_physical_route="MOCK-ONLY";
    inputs.agreed_manifest=sha256_bytes(std::array<std::byte,1>{std::byte{9}});
    auto route=make_hip_copied_host_sequence_route(std::move(inputs));
    std::weak_ptr<const void> owner=route.sequence.actual_owner;
    auto backend=std::move(route.sequence.copied);
    check(!backend.owner_before(route.sequence.actual_owner) && !route.sequence.actual_owner.owner_before(backend));
    route.sequence.actual_owner.reset();route.sequence.step_control.reset();route.sequence.token_control.reset();
    route.sequence.copied_generations.reset();route.greedy.reset();
    check(!owner.expired());for(const auto& o:seen)check(o->destroyed==0);
    check(backend->connection().physical_epoch==19 && backend->connection().physical_link_id=="MOCK-ONLY");
    check(backend->capabilities().buffer_mode==transport::TransportBufferMode::copy_stream && !backend->capabilities().external_dma);
    backend->poison();check(seen[0]->destroyed==1 && seen[0]->aborted==1);
    check(!owner.expired());backend.reset();check(owner.expired());
    for(const auto& o:seen)check(o->destroyed==1);
    std::cout<<"PASS "<<checks<<" actual factory alias ownership checks; MOCK transports; NONnative\n";
}
