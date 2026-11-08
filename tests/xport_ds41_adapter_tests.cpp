#include "xport/ds41_adapter.hpp"
#include "strix/ds41_layer.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {
using Candidate = strix::ds41::ArgmaxCandidate;
struct Tag { std::uint32_t kind, layer; std::uint64_t request; };
void require(bool condition) {
    if (!condition) throw std::runtime_error("DS41 adapter fixture failed");
}
struct FakeCommunicator {
    std::span<const Candidate> peer;
    std::vector<std::byte> sent;
    std::vector<unsigned> calls;
    std::size_t payload_bytes = 0, messages = 0;
    Tag observed_tag{};
    unsigned observed_stream = 0;
    void start(unsigned call_stream, void* payload, std::size_t payload_size, const Tag& call_tag) {
        require(payload_size == peer.size_bytes());
        observed_stream = call_stream;
        observed_tag = call_tag;
        const auto* payload_begin = static_cast<const std::byte*>(payload);
        sent.insert(sent.end(), payload_begin, payload_begin + payload_size);
        payload_bytes += payload_size;
        ++messages;
        calls.push_back(1);
    }
    void finish(unsigned call_stream, void* destination) {
        require(call_stream == observed_stream);
        std::memcpy(destination, peer.data(), peer.size_bytes());
        calls.push_back(2);
    }
    // The former call site's exact start/finish contract.
    void exchange(unsigned call_stream, void* payload, void* destination,
                  std::size_t payload_size, const Tag& call_tag) {
        start(call_stream, payload, payload_size, call_tag);
        finish(call_stream, destination);
    }
};

// Independent host rendering of the unchanged device combine rule. Explicit
// expected winners below keep a shared comparator mistake from hiding a tie bug.
Candidate combine(const Candidate& rank_zero, const Candidate& rank_one) {
    Candidate winner = (rank_one.value > rank_zero.value ||
        (rank_one.value == rank_zero.value && rank_one.index < rank_zero.index)) ? rank_one : rank_zero;
    winner.finite = rank_zero.finite & rank_one.finite;
    return winner;
}
void run() {
    constexpr float infinity = std::numeric_limits<float>::infinity();
    const std::array<Candidate, 8> rank_zero{{
        {3, 9, 1}, {1, 2, 1}, {4, 9, 1}, {4, 2, 1},
        {-0.0F, 7, 1}, {-infinity, 0xffffffffU, 1}, {2, 3, 0}, {0.0F, 5, 1}}};
    const std::array<Candidate, 8> rank_one{{
        {1, 2, 1}, {3, 9, 1}, {4, 2, 1}, {4, 9, 1},
        {0.0F, 7, 1}, {-infinity, 0xffffffffU, 0}, {1, 2, 1}, {-0.0F, 2, 1}}};
    const std::array<std::uint32_t, 8> expected_indices{{9, 9, 2, 2, 7, 0xffffffffU, 3, 2}};
    const std::array<std::uint32_t, 8> expected_finite{{1, 1, 1, 1, 1, 0, 0, 1}};
    // Both ranks, the decode size (one row), and a multi-row payload. Repeated
    // requests exercise communicator reuse and preservation of tag/request bytes.
    for (unsigned rank_id = 0; rank_id < 2; ++rank_id) {
        for (const std::size_t row_count : {std::size_t{1}, rank_zero.size()}) {
            const auto& own = rank_id == 0 ? rank_zero : rank_one;
            const auto& peer = rank_id == 0 ? rank_one : rank_zero;
            FakeCommunicator legacy{std::span<const Candidate>(peer.data(), row_count), {}, {}};
            FakeCommunicator adapted{std::span<const Candidate>(peer.data(), row_count), {}, {}};
            for (std::uint64_t request_id = 0; request_id < 8; ++request_id) {
                std::array<Candidate, 8> local = own, legacy_remote{}, adapted_remote{};
                const Tag call_tag{3, 0, request_id};
                legacy.exchange(17, local.data(), legacy_remote.data(), row_count * sizeof(Candidate), call_tag);
                strix::xport::ds41_exchange_argmax(adapted, 17U, local.data(), adapted_remote.data(), row_count, call_tag);
                require(legacy.sent == adapted.sent && legacy.calls == adapted.calls);
                require(legacy.payload_bytes == adapted.payload_bytes && legacy.messages == adapted.messages);
                require(adapted.messages == request_id + 1 && adapted.payload_bytes == (request_id + 1) * row_count * sizeof(Candidate));
                require(adapted.observed_stream == 17 && adapted.observed_tag.kind == 3 &&
                        adapted.observed_tag.layer == 0 && adapted.observed_tag.request == request_id);
                require(std::memcmp(legacy_remote.data(), adapted_remote.data(), row_count * sizeof(Candidate)) == 0);
                require(std::memcmp(local.data(), own.data(), sizeof(local)) == 0);
                for (std::size_t row_id = 0; row_id < row_count; ++row_id) {
                    const auto legacy_winner = rank_id == 0 ? combine(local[row_id], legacy_remote[row_id]) :
                                                               combine(legacy_remote[row_id], local[row_id]);
                    const auto adapted_winner = rank_id == 0 ? combine(local[row_id], adapted_remote[row_id]) :
                                                                combine(adapted_remote[row_id], local[row_id]);
                    require(std::memcmp(&legacy_winner, &adapted_winner, sizeof(Candidate)) == 0);
                    require(adapted_winner.index == expected_indices[row_id] && adapted_winner.finite == expected_finite[row_id]);
                    if (row_id == 4) require(std::memcmp(&adapted_winner.value, &rank_zero[4].value, sizeof(float)) == 0);
                }
            }
        }
    }
}
} // namespace
int main() {
    try { run(); std::cout << "DS41 adapter: candidates, ties, bytes, messages and call order passed\n"; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
