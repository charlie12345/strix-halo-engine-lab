#pragma once
#include "strix/distributed_control_wire.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix {
// Adapter for the project's synchronous whole-message host Transport contract.
// HostTransport is instantiated as dist::Transport by the actual HIP factory.
// Each channel exclusively owns a different transport, with bounded native I/O.
// No GPU address, readiness, DMA lease or driver qualification is fabricated.
template<class HostTransport>
class CopiedHostByteChannel final : public DistributedControlByteStream {
    std::unique_ptr<HostTransport> transport_;
    ModelContentIdentity manifest_;
    std::uint64_t epoch_,channel_,maximum_,sent_=0,received_=0;
    std::uint32_t rank_;
    bool failed_=false;
    std::string transport_name_;
    static constexpr std::size_t header_bytes=128;
    using Header=std::array<std::byte,header_bytes>;
    static void put(Header& h,std::size_t offset,std::uint64_t v,unsigned bytes=8) {
        for(unsigned i=0;i<bytes;++i) h[offset+i]=static_cast<std::byte>((v>>(i*8U))&255U);
    }
    static std::uint64_t get(const Header& h,std::size_t offset,unsigned bytes=8) {
        std::uint64_t v=0;for(unsigned i=0;i<bytes;++i) v|=std::uint64_t(std::to_integer<unsigned char>(h[offset+i]))<<(i*8U);return v;
    }
    void healthy() const {if(failed_) throw std::logic_error("host-copy channel is poisoned");}
    void fail() noexcept {
        failed_=true;
        auto owner=std::move(transport_);
        if(owner) {try {owner->abort("host-copy frame failure");} catch(...) {} owner.reset();}
    }
    template<class Io> void io(Io&& work) {
        // This nested error boundary runs BEFORE the caller's header/vector
        // unwinds. Abort and destroy the real transport here: its persistent
        // host workers join while every borrowed I/O buffer is still alive.
        try {work();} catch(...) {fail();throw;}
    }
public:
    CopiedHostByteChannel(std::unique_ptr<HostTransport> transport,std::uint32_t rank,std::uint64_t epoch,
        std::uint64_t channel,ModelContentIdentity manifest,std::uint64_t maximum)
        :transport_(std::move(transport)),manifest_(manifest),epoch_(epoch),channel_(channel),maximum_(maximum),rank_(rank) {
        if(!transport_ || rank>1U || !epoch || !channel || !manifest.valid() || !maximum || maximum>UINT32_MAX)
            throw std::invalid_argument("host-copy channel requires complete world2 identity and frame bound");
        transport_name_=transport_->name();
    }
    std::uint64_t maximum_frame_bytes() const noexcept override {return maximum_;}
    std::string transport_identity() const {return "host-copy-sha256-v1:"+transport_name_;}
    bool failed() const noexcept {return failed_;}
    void poison() noexcept {fail();}
    void abort() noexcept override {fail();}
    void send(std::span<const std::byte> frame) override {
        healthy();try {
            if(frame.empty() || frame.size()>maximum_ || sent_==UINT64_MAX)
                throw std::invalid_argument("host-copy send exceeds admitted frame or sequence capacity");
            Header h{};put(h,0,0x3150434858545353ULL);put(h,8,1,4);put(h,12,rank_,4);put(h,16,1U-rank_,4);put(h,20,2,4);
            put(h,24,epoch_);put(h,32,channel_);put(h,40,sent_+1U);put(h,48,frame.size());
            const auto hash=sha256_bytes(frame);
            for(std::size_t i=0;i<32;++i) {h[56+i]=static_cast<std::byte>(manifest_.bytes[i]);h[88+i]=static_cast<std::byte>(hash.bytes[i]);}
            io([&]{transport_->send(h);});io([&]{transport_->send(frame);});++sent_;
        } catch(...) {fail();throw;}
    }
    std::vector<std::byte> receive() override {
        healthy();try {
            Header h{};io([&]{transport_->receive(h);});
            if(received_==UINT64_MAX || get(h,0)!=0x3150434858545353ULL || get(h,8,4)!=1U ||
                get(h,12,4)!=1U-rank_ || get(h,16,4)!=rank_ || get(h,20,4)!=2U ||
                get(h,24)!=epoch_ || get(h,32)!=channel_ || get(h,40)!=received_+1U || get(h,120)!=0U)
                throw std::invalid_argument("host-copy frame rank/world/epoch/channel/sequence differs");
            for(std::size_t i=0;i<32;++i) if(h[56+i]!=static_cast<std::byte>(manifest_.bytes[i]))
                throw std::invalid_argument("host-copy source/model/target/admission manifest differs");
            const auto bytes=get(h,48);
            if(!bytes || bytes>maximum_ || bytes>std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("host-copy received frame exceeds admitted bound");
            std::vector<std::byte> frame(static_cast<std::size_t>(bytes));io([&]{transport_->receive(frame);});
            const auto hash=sha256_bytes(frame);
            for(std::size_t i=0;i<32;++i) if(h[88+i]!=static_cast<std::byte>(hash.bytes[i]))
                throw std::invalid_argument("host-copy payload checksum differs");
            ++received_;return frame;
        } catch(...) {fail();throw;}
    }
};
} // namespace strix
