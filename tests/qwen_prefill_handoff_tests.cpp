#include "strix/qwen_prefill_handoff.hpp"
#include <cstring>
#include <atomic>
#include <iostream>
#include <stdexcept>
using namespace strix::qwen_pp;
static void test_mtp_handoff(std::uint64_t position) {
    const MtpStateDescriptor descriptor{mtp_state_magic,1U,position-1U,123U,20480U,129U,2U,77U,88U};
    const auto extension=mtp_state_parts(position,descriptor);
    std::vector<StatePart> layout(extension.begin(),extension.end());
    layout.push_back({24U,1U,31U}); // target state still shares the checked stream
    layout.push_back({48U,0xffffffffU,64U}); // terminal FP32 logits
    const auto identity=mtp_state_identity(456U,descriptor);
    std::vector<std::vector<std::byte>> frames;
    const auto pattern=[](std::size_t index,std::uint64_t offset) {
        return static_cast<std::byte>((static_cast<std::uint64_t>(index)*37U+offset)%251U);
    };
    const auto sent=handoff(true,layout,identity,position,24U,
        [&](std::span<const std::byte> data){frames.emplace_back(data.begin(),data.end());},
        [&](std::span<std::byte> data) {
            const auto index=frames.back().size()==sizeof(StateHeader)?frames.size()-1U:frames.size()-2U;
            std::copy(frames[index].begin(),frames[index].end(),data.begin());
        },
        [&](bool save,std::size_t index,std::uint64_t offset,std::span<std::byte> data) {
            if(!save) throw std::runtime_error("MTP producer direction");
            if(index==0U) {
                const auto header=std::as_bytes(std::span(descriptor)).subspan(static_cast<std::size_t>(offset),data.size());
                std::copy(header.begin(),header.end(),data.begin());
            } else for(std::size_t byte=0;byte<data.size();++byte) data[byte]=pattern(index,offset+byte);
        });
    const auto receive=[&](const auto& wire,const auto& parts,const MtpStateDescriptor& expected,bool reject,bool no_install) {
        std::size_t cursor=0,installed=0;
        bool resumed=false;
        auto actual=expected;
        std::vector<std::uint64_t> counts(parts.size());
        try {
            const auto bytes=handoff(false,parts,mtp_state_identity(456U,expected),position,24U,
                [](std::span<const std::byte>){},
                [&](std::span<std::byte> data) {
                    if(cursor>=wire.size() || wire[cursor].size()!=data.size()) throw std::runtime_error("MTP truncated frame");
                    std::copy(wire[cursor].begin(),wire[cursor].end(),data.begin());++cursor;
                },
                [&](bool save,std::size_t index,std::uint64_t offset,std::span<std::byte> data) {
                    if(save || counts[index]!=offset) throw std::runtime_error("MTP consumer order");
                    if(index==0U) {
                        auto header=std::as_writable_bytes(std::span(actual)).subspan(static_cast<std::size_t>(offset),data.size());
                        std::copy(data.begin(),data.end(),header.begin());
                        if(actual!=expected) throw std::runtime_error("MTP descriptor payload");
                    } else {
                        for(std::size_t byte=0;byte<data.size();++byte)
                            if(data[byte]!=pattern(index,offset+byte)) throw std::runtime_error("MTP payload bytes");
                        ++installed;
                    }
                    counts[index]+=data.size();
                });
            if(bytes!=sent || cursor!=wire.size()) throw std::runtime_error("MTP completion bytes");
            for(std::size_t index=0;index<parts.size();++index)
                if(counts[index]!=parts[index].bytes) throw std::runtime_error("MTP missing part");
            resumed=true; // publication is permitted only after the complete receipt
        } catch(const std::runtime_error&) {}
        if(resumed==reject || (no_install && installed)) throw std::runtime_error("MTP admission verdict");
    };
    receive(frames,layout,descriptor,false,false);
    auto wrong=descriptor;wrong[7]^=1U;receive(frames,layout,wrong,true,true);
    wrong=descriptor;wrong[8]^=1U;receive(frames,layout,wrong,true,true);
    wrong=descriptor;++wrong[4];receive(frames,layout,wrong,true,true);
    wrong=descriptor;--wrong[2];receive(frames,layout,wrong,true,true);
    auto bad_layout=layout;std::swap(bad_layout[1],bad_layout[2]);receive(frames,bad_layout,descriptor,true,true);
    auto corrupt=frames;corrupt[3][0]^=std::byte{1};receive(corrupt,layout,descriptor,true,true);
    corrupt=frames;corrupt.pop_back();receive(corrupt,layout,descriptor,true,false);
    // Valid transport CRC with a semantically wrong descriptor must also fail before installation.
    corrupt=frames;corrupt[1][0]^=std::byte{1};
    StateHeader header{};std::memcpy(header.data(),corrupt[0].data(),sizeof(header));
    SeamCrc crc;crc.update(corrupt[1]);header.back()=crc.value();
    std::memcpy(corrupt[0].data(),header.data(),sizeof(header));receive(corrupt,layout,descriptor,true,true);
    const auto invalid=[&](std::uint64_t end,const MtpStateDescriptor& value) {
        bool failed=false;
        try {(void)mtp_state_parts(end,value);} catch(const std::exception&){failed=true;}
        if(!failed) throw std::runtime_error("MTP invalid descriptor accepted");
    };
    invalid(position,wrong);
    wrong=descriptor;wrong[5]=std::numeric_limits<std::uint64_t>::max();invalid(position,wrong);
    wrong=descriptor;wrong[4]=0U;invalid(position,wrong);
    invalid(0U,descriptor);
}

struct HostStateStaging final : StateStaging {
    std::array<std::vector<std::byte>,2> slots;
    HostStateStaging() { for(auto& slot:slots) slot.resize(seam_piece_bytes); }
    std::span<std::byte> buffer(std::size_t slot) override { return slots.at(slot); }
    void copy_device(bool,void*,std::span<std::byte>) override { throw std::logic_error("host-only staging"); }
};
static void test_overlap_handoff() {
    const std::array<StatePart,3> parts{{{24U,1U,2U*seam_piece_bytes+17U},{25U,1U,0U},{48U,0xffffffffU,64U}}};
    HostStateStaging staging;
    std::vector<std::vector<std::byte>> wire;
    const auto checksum=[](std::span<const std::byte> data) { SeamCrc crc;crc.update(data);return crc.value(); };
    const auto pattern=[](std::size_t part,std::uint64_t offset) {return static_cast<std::byte>((part+offset)%251U);};
    std::mutex lock; std::condition_variable cv;
    bool lookahead=false;
    const auto sent=handoff_overlap(true,parts,123U,32768U,24U,staging,
        [&](std::span<const std::byte> data) {
            if(wire.size()==1U) {
                std::unique_lock guard(lock);
                if(!cv.wait_for(guard,std::chrono::seconds(5),[&]{return lookahead;}))
                    throw std::runtime_error("D2H did not overlap send");
            }
            wire.emplace_back(data.begin(),data.end());
        },
        [&](std::span<std::byte> data) {std::copy(wire.back().begin(),wire.back().end(),data.begin());},
        [&](bool save,std::size_t part,std::uint64_t offset,std::span<std::byte> data) {
            if(!save) throw std::logic_error("producer direction");
            if(offset==seam_piece_bytes) {std::lock_guard guard(lock);lookahead=true;cv.notify_all();}
            for(std::size_t i=0;i<data.size();++i) data[i]=pattern(part,offset+i);
        },checksum,[]{});
    if(sent!=2U*seam_piece_bytes+81U || wire.size()!=9U) throw std::runtime_error("overlap framing/total");
    const auto replay=[&](bool corrupt,bool fail_copy,bool truncate) {
        std::size_t cursor=0,acks=0;
        std::atomic<std::uint64_t> installed{0};
        bool next_header=false,aborted=false,rejected=false;
        try {
            const auto received=handoff_overlap(false,parts,123U,32768U,24U,staging,
                [&](std::span<const std::byte>) {++acks;},
                [&](std::span<std::byte> data) {
                    if(truncate && cursor==wire.size()-1U) throw std::runtime_error("truncated completion");
                    if(cursor>=wire.size() || wire[cursor].size()!=data.size()) throw std::runtime_error("wire shape");
                    std::copy(wire[cursor].begin(),wire[cursor].end(),data.begin());
                    if(corrupt && cursor==1U) data[0]^=std::byte{1};
                    if(cursor==2U) {std::lock_guard guard(lock);next_header=true;cv.notify_all();}
                    ++cursor;
                },
                [&](bool save,std::size_t part,std::uint64_t offset,std::span<std::byte> data) {
                    if(save || fail_copy) throw std::runtime_error("copy failure");
                    if(part==0U && offset==0U) {
                        std::unique_lock guard(lock);
                        if(!cv.wait_for(guard,std::chrono::seconds(5),[&]{return next_header;}))
                            throw std::runtime_error("receive did not overlap H2D");
                    }
                    for(std::size_t i=0;i<data.size();++i)
                        if(data[i]!=pattern(part,offset+i)) throw std::runtime_error("slot reused before install");
                    installed+=data.size();
                },checksum,[&]{aborted=true;});
            if(received!=sent || installed!=sent || acks!=1U) throw std::runtime_error("completion before install");
        } catch(const std::runtime_error&) {rejected=true;}
        if(rejected!=(corrupt||fail_copy||truncate) || aborted!=rejected || (rejected && acks))
            throw std::runtime_error("overlap error/completion guard");
        if(corrupt && installed!=0U) throw std::runtime_error("corrupt piece installed");
    };
    replay(false,false,false); replay(true,false,false); replay(false,true,false); replay(false,false,true);
}

int main() {
    try {
        test_overlap_handoff();
        // Reject a stage-B shared-history overwrite before even the preceding
        // valid layer can be copied or sent.
        for (const auto& position : {32768U, 131072U}) {
            for (const auto& producer : {false, true}) {
                const std::array<StatePart, 2> wrong{{{24U, 1U, 17U}, {48U, 3U, 184320U}}};
                bool touched = false, rejected = false;
                try {
                    (void)handoff(producer, wrong, 123U, position, 24U,
                        [&](std::span<const std::byte>) { touched = true; },
                        [&](std::span<std::byte>) { touched = true; },
                        [&](bool, std::size_t, std::uint64_t, std::span<std::byte>) { touched = true; });
                } catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected || touched) throw std::runtime_error("shared PLE owner guard");
            }
        }
        const std::array<StatePart, 4> parts{{{24, 1, 17}, {25, 2, seam_piece_bytes + 3U}, {27, 0, 31}, {48, 0xffffffffU, 0}}};
        const auto expected = 17U + seam_piece_bytes + 3U + 31U;
        std::vector<std::vector<std::byte>> wire;
        std::vector<std::uint64_t> produced(parts.size()), consumed(parts.size());
        const auto sent = handoff(true, parts, 123U, 32768U, 24U,
            [&](std::span<const std::byte> data) { wire.emplace_back(data.begin(), data.end()); },
            [&](std::span<std::byte> data) {
                const auto index = wire.back().size() == sizeof(StateHeader) ? wire.size()-1U : wire.size()-2U;
                std::copy(wire[index].begin(), wire[index].end(), data.begin());
            },
            [&](bool save, std::size_t index, std::uint64_t offset, std::span<std::byte> data) {
                if (!save || offset != produced[index]) throw std::runtime_error("producer ordering");
                std::fill(data.begin(), data.end(), static_cast<std::byte>(index + 1U));
                produced[index] += data.size();
            });
        if (sent != expected) throw std::runtime_error("producer count");
        auto receive = [&](const auto& frames, const auto& layout, bool expect_failure) {
            std::size_t cursor = 0, acks = 0;
            std::fill(consumed.begin(), consumed.end(), 0U);
            bool failed = false;
            try {
                const auto bytes = handoff(false, layout, 123U, 32768U, 24U,
                    [&](std::span<const std::byte>) { ++acks; },
                    [&](std::span<std::byte> data) {
                        if (cursor >= frames.size() || frames[cursor].size() != data.size())
                            throw std::runtime_error("wire size");
                        std::copy(frames[cursor].begin(), frames[cursor].end(), data.begin()); ++cursor;
                    },
                    [&](bool save, std::size_t index, std::uint64_t offset, std::span<std::byte> data) {
                        if (save || offset != consumed[index]) throw std::runtime_error("consumer ordering");
                        for (const auto value : data) if (value != static_cast<std::byte>(index + 1U))
                            throw std::runtime_error("consumer content");
                        consumed[index] += data.size();
                    });
                if (bytes != expected || cursor != frames.size() || acks != 5U)
                    throw std::runtime_error("completion count");
            } catch (const std::runtime_error&) { failed = true; }
            if (failed != expect_failure) throw std::runtime_error("unexpected verdict");
        };
        receive(wire, parts, false);
        for (std::size_t index = 0; index < parts.size(); ++index)
            if (produced[index] != parts[index].bytes || consumed[index] != parts[index].bytes)
                throw std::runtime_error("per-layer count");
        auto corrupt = wire; corrupt[1][0] ^= std::byte{1}; receive(corrupt, parts, true);
        corrupt = wire; std::swap(corrupt[0], corrupt[2]); receive(corrupt, parts, true);
        auto wrong_parts = parts; ++wrong_parts[0].bytes; receive(wire, wrong_parts, true);
        corrupt = wire; corrupt.back()[0] ^= std::byte{1}; receive(corrupt, parts, true);
        corrupt = wire; corrupt.pop_back(); receive(corrupt, parts, true);
        bool bad_ack = false;
        try {
            handoff(true, parts, 123U, 32768U, 24U,
                [](std::span<const std::byte>) {},
                [](std::span<std::byte> data) { std::fill(data.begin(), data.end(), std::byte{}); },
                [](bool, std::size_t, std::uint64_t, std::span<std::byte>) {});
        } catch (const std::runtime_error&) { bad_ack = true; }
        if (!bad_ack) throw std::runtime_error("bad ACK accepted");
        test_mtp_handoff(32768U);
        test_mtp_handoff(131072U);
        std::cout << "pair handoff MTP framing PASS: tap, cache rows, frontier, identity, corruption, completion\n";
        std::cout << "pair handoff framing PASS: pieces, ordering, per-layer bytes, CRC, completion, ACK\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
