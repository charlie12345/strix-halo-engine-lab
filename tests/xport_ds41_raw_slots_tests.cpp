#include "xport/ds41_adapter.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

namespace {
using Slots = std::array<unsigned char, 96>;
struct Tag { unsigned kind, layer; std::uint64_t request; };
struct ReadyView { void* data; unsigned token; };
void require(bool condition) {
    if (!condition) throw std::runtime_error("DS41 raw_slots fixture failed");
}
struct FakeCommunicator {
    Slots tx{}, rx{}, peer{};
    std::vector<unsigned char> sent;
    std::vector<unsigned> calls;
    std::vector<std::size_t> sizes;
    std::vector<unsigned> layers;
    std::size_t bytes = 0, messages = 0;
    bool held = false;
    FakeCommunicator() {
        for (std::size_t at = 0; at < peer.size(); ++at)
            peer[at] = static_cast<unsigned char>((at * 37U) & 255U);
    }
    void start(unsigned call_stream, void* payload, std::size_t payload_size, const Tag& call_tag) {
        require(call_stream == 7U && call_tag.kind == 4U && call_tag.request == 29U);
        const auto* payload_begin = static_cast<const unsigned char*>(payload);
        sent.insert(sent.end(), payload_begin, payload_begin + payload_size);
        sizes.push_back(payload_size);
        layers.push_back(call_tag.layer);
        bytes += payload_size;
        ++messages;
        calls.push_back(1U);
    }
    void finish(unsigned call_stream, void* destination) {
        require(call_stream == 7U);
        std::memcpy(destination, peer.data(), sizes.back());
        calls.push_back(3U);
    }
    void* send_staging(std::size_t staging_bytes) {
        require(staging_bytes <= tx.size());
        calls.push_back(0U);
        return tx.data();
    }
    ReadyView prepare_staged_ready(unsigned call_stream, std::size_t staging_bytes) {
        require(call_stream == 7U && staging_bytes <= tx.size());
        calls.push_back(6U);
        return {tx.data(), 41U};
    }
    void start_staged(unsigned call_stream, std::size_t payload_size, const Tag& call_tag) {
        start(call_stream, tx.data(), payload_size, call_tag);
    }
    const void* finish_staged(unsigned call_stream) {
        finish(call_stream, rx.data());
        held = true;
        return rx.data();
    }
    void release_staged(unsigned call_stream) {
        require(call_stream == 7U && held);
        calls.push_back(5U);
        held = false;
    }
    void start_staged_ready(unsigned call_stream, std::size_t payload_size, const Tag& call_tag) {
        start_staged(call_stream, payload_size, call_tag);
    }
    ReadyView finish_staged_ready() {
        (void)finish_staged(7U);
        return {rx.data(), 73U};
    }
    void release_staged_ready(unsigned call_stream) { release_staged(call_stream); }
};
struct Result { FakeCommunicator comm; Slots received{}; };
Result plain(bool adapted, unsigned chunks, std::size_t slot_bytes, unsigned slot_layer, bool cross_op) {
    Result result;
    Slots own{};
    for (std::size_t at = 0; at < own.size(); ++at) own[at] = static_cast<unsigned char>(255U - at);
    auto& test_comm = result.comm;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        const Tag slot_tag{4U, slot_layer, 29U};
        auto* local_chunk = own.data() + chunk * slot_bytes;
        auto* remote_chunk = result.received.data() + chunk * slot_bytes;
        const auto overlap = [&] {
            require(test_comm.messages == chunk + 1U && !test_comm.held);
            // Model next-product / previous-add / late-mix / prefetch ordering.
            test_comm.calls.push_back(2U);
            if (chunks > 1U) {
                if (chunk + 1U < chunks) test_comm.calls.push_back(8U);
                if (chunk > 0U) test_comm.calls.push_back(9U);
            }
        };
        if (!adapted) {
            test_comm.start(7U, local_chunk, slot_bytes, slot_tag);
            overlap();
            test_comm.finish(7U, remote_chunk);
        } else if (cross_op) {
            using Exchange = strix::xport::Ds41RawSlotsExchange<FakeCommunicator, unsigned, Tag>;
            std::optional<Exchange> split_exchange;
            split_exchange.emplace(test_comm, 7U, local_chunk, remote_chunk, slot_bytes, slot_tag);
            overlap();
            split_exchange->finish(7U);
            split_exchange.reset();
        } else {
            strix::xport::ds41_exchange_raw_slots(test_comm, 7U, local_chunk, remote_chunk,
                                                 slot_bytes, slot_tag, overlap);
        }
        test_comm.calls.push_back(4U);
    }
    return result;
}
template<bool Ready>
Result staged(bool adapted, bool split_phase = false) {
    Result result;
    auto& test_comm = result.comm;
    void* prepared;
    if constexpr (Ready) prepared = test_comm.prepare_staged_ready(7U, sizeof(Slots)).data;
    else prepared = test_comm.send_staging(sizeof(Slots));
    for (std::size_t at = 0; at < test_comm.tx.size(); ++at)
        test_comm.tx[at] = static_cast<unsigned char>(at ^ 0xa5U);
    const Tag slot_tag{4U, 11U, 29U};
    const auto overlap = [&] {
        require(!test_comm.held && test_comm.messages == 1U);
        test_comm.calls.push_back(2U);
    };
    const auto consume = [&](const void* payload) {
        require(test_comm.held && payload == test_comm.rx.data());
        std::memcpy(result.received.data(), payload, sizeof(Slots));
        test_comm.calls.push_back(4U);
    };
    const auto consume_ready = [&](const ReadyView& ready_peer) {
        require(ready_peer.token == 73U);
        consume(ready_peer.data);
    };
    if (adapted) {
        if constexpr (Ready)
            strix::xport::ds41_exchange_raw_slots_staged<true>(test_comm, 7U, prepared,
                sizeof(Slots), slot_tag, overlap, consume_ready);
        else if (split_phase) {
            strix::xport::Ds41RawSlotsStagedExchange<FakeCommunicator, unsigned, Tag> staged_pending{
                test_comm, 7U, prepared, sizeof(Slots), slot_tag};
            overlap();
            consume(staged_pending.finish(7U));
            staged_pending.release(7U);
        } else
            strix::xport::ds41_exchange_raw_slots_staged<false>(test_comm, 7U, prepared,
                sizeof(Slots), slot_tag, overlap, consume);
    } else {
        if constexpr (Ready) test_comm.start_staged_ready(7U, sizeof(Slots), slot_tag);
        else test_comm.start_staged(7U, sizeof(Slots), slot_tag);
        overlap();
        if constexpr (Ready) consume_ready(test_comm.finish_staged_ready());
        else consume(test_comm.finish_staged(7U));
        if constexpr (Ready) test_comm.release_staged_ready(7U);
        else test_comm.release_staged(7U);
    }
    return result;
}
void compare(const Result& old_result, const Result& adapted_result, std::size_t expected_messages,
             std::size_t expected_bytes) {
    require(old_result.received == adapted_result.received);
    require(old_result.comm.sent == adapted_result.comm.sent);
    require(old_result.comm.calls == adapted_result.comm.calls);
    require(old_result.comm.sizes == adapted_result.comm.sizes);
    require(old_result.comm.layers == adapted_result.comm.layers);
    require(old_result.comm.bytes == expected_bytes && adapted_result.comm.bytes == expected_bytes);
    require(old_result.comm.messages == expected_messages && adapted_result.comm.messages == expected_messages);
    require(!old_result.comm.held && !adapted_result.comm.held);
}
}
int main() {
    try {
        // Engram packed rows and distinguished tag; ordinary trellis and Q8 OB.
        compare(plain(false, 1U, 96U, 0x1000bU, false), plain(true, 1U, 96U, 0x1000bU, false), 1U, 96U);
        compare(plain(false, 1U, 96U, 11U, false), plain(true, 1U, 96U, 11U, false), 1U, 96U);
        compare(plain(false, 4U, 24U, 11U, false), plain(true, 4U, 24U, 11U, false), 4U, 96U);
        // Split-shared activation codes followed by scales, for either producer.
        compare(plain(false, 1U, 36U, 11U, true), plain(true, 1U, 36U, 11U, true), 1U, 36U);
        compare(staged<false>(false), staged<false>(true), 1U, 96U);
        compare(staged<false>(false), staged<false>(true, true), 1U, 96U);
        compare(staged<true>(false), staged<true>(true), 1U, 96U);
        std::cout << "DS41 raw_slots byte, message, route and lifetime checks passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
