// Compare the original MiniMax progress serializer with the production shared serializer.
// Transport::exchange sees payload + trailer: MiniMax adds no separate wire header.
#include "minimax/xfer/minimax_traits.hpp"
#include "xport/wire.hpp"
#include <array>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace strix::minimax::xfer;
using Bytes = std::vector<std::byte>;

// Independent scalar CRC-32C oracle for the transport mock (same polynomial as dist::crc32c).
std::uint32_t crc32c(const Bytes& bytes) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const auto b : bytes) {
        crc ^= static_cast<std::uint32_t>(b);
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0x82F63B78U : 0U);
    }
    return ~crc;
}
// Original call shape retained so the frozen process() region needs no edits.
namespace dist {
struct View { const std::byte* data; std::size_t size; };
std::uint32_t crc32c(View v) { return ::crc32c(Bytes(v.data, v.data + v.size)); }
}
// The fixture's std::span is replaced only at its call site during extraction.
struct Job { Tag tag; std::uint64_t sequence; std::size_t bytes; };
struct Options { std::uint64_t epoch = 7, plan = 0x46584d4dULL; std::uint32_t rank = 0; bool crc = true; };
struct Legacy {
    Options options_;
    bool prev_deferred_ = false;
    std::uint32_t prev_tx_crc_ = 0;
    Trailer trailer(const Job& job, const Bytes& payload, bool deferred) const {
        const auto* tx = payload.data();
#include "fixtures/minimax_xfer_legacy_trailer.inc"
    }
};

struct MockTransport {
    std::array<std::deque<Bytes>, 2> pending;
    std::uint64_t messages = 0, bytes = 0;
    void exchange(std::uint32_t rank, Bytes message) {
        bytes += static_cast<std::uint64_t>(message.size());
        ++messages;
        pending.at(rank).push_back(std::move(message));
    }
    struct Endpoint {
        MockTransport& owner;
        std::uint32_t rank;
        void exchange(std::span<const std::byte> out, std::span<std::byte> in) {
            require_size(out.size() == in.size());
            owner.exchange(rank, Bytes(out.begin(), out.end()));
        }
        static void require_size(bool ok) {
            if (!ok) throw std::runtime_error("mock transport frame size differs");
        }
    };
};

void require(bool ok) { if (!ok) throw std::runtime_error("MiniMax wire mismatch"); }
void run(unsigned rows, bool prefill, unsigned crc_mode) {
    MockTransport old_transport, new_transport;
    std::array<Legacy, 2> old{};
    std::array<bool, 2> previous{};
    std::array<std::uint32_t, 2> previous_crc{};
    std::uint64_t sequence = 0;
    unsigned compared = 0;
    const auto flush = [&] {
        for (std::size_t rank = 0; rank < 2; ++rank) {
            auto& a = old_transport.pending[rank];
            auto& b = new_transport.pending[rank];
            require(a.size() == b.size() && a.size() <= 8);
            while (!a.empty()) {
                require(a.front() == b.front()); // includes every payload and trailer/header byte
                a.pop_front(); b.pop_front(); ++compared;
            }
        }
    };
    const auto post = [&](Tag tag, std::size_t bytes, bool seal) {
        if (!seal) ++sequence;
        for (std::uint32_t rank = 0; rank < 2; ++rank) {
            Bytes payload(bytes);
            for (std::size_t i = 0; i < bytes; ++i)
                payload[i] = static_cast<std::byte>((i + sequence * 13U + rank * 71U) & 255U);
            const bool deferred = crc_mode == 2 && bytes > kInlineCrcMax;
            old[rank].options_.rank = rank;
            old[rank].options_.crc = crc_mode != 0;
            const Job job{tag, sequence, bytes};
            const auto a = old[rank].trailer(job, payload, deferred);
            const auto site = static_cast<unsigned>(sequence - 1);
            const auto new_tag = seal ? Tag{Kind::control, 0xFFFFFFFFU, 0} : MinimaxTraits::tag_at(site, 11);
            const auto new_bytes = seal ? std::size_t{0} : MinimaxTraits::payload_bytes(site, rows);
            require(new_bytes == bytes);
            const auto b = strix::xport::make_trailer<MinimaxTraits>(new_tag, 7, MinimaxTraits::kPlan,
                sequence, new_bytes, rank, deferred, crc_mode != 0 ? crc32c(payload) : 0U,
                previous[rank], previous_crc[rank]);
            const auto pack = [&](const Trailer& t) {
                Bytes result = payload;
                result.resize(bytes + sizeof(t));
                std::memcpy(result.data() + bytes, &t, sizeof(t));
                return result;
            };
            old_transport.exchange(rank, pack(a));
            Bytes tx = payload, rx(bytes + sizeof(Trailer));
            tx.resize(bytes + sizeof(Trailer));
            MockTransport::Endpoint endpoint{new_transport, rank};
            strix::xport::exchange_frame(endpoint, tx.data(), rx.data(), bytes, b);
            old[rank].prev_deferred_ = deferred;
            previous[rank] = deferred;
            if (deferred) {
                old[rank].prev_tx_crc_ = crc32c(payload);
                previous_crc[rank] = crc32c(payload);
            }
        }
        if (old_transport.pending[0].size() == 8) flush();
    };
    // Frozen old schedule is literal; the new schedule's sizes/counts must agree independently.
    require(MinimaxTraits::Schedule::kLayers == 60 && MinimaxTraits::Schedule::kPerStep == 121);
    require(MinimaxTraits::Schedule::row_bytes(rows) == std::size_t{rows} * 6144U * sizeof(float));
    for (std::uint32_t layer = 0; layer < 60; ++layer) {
        post(Tag{Kind::attn_sum, layer, 11}, std::size_t{rows} * 6144U * sizeof(float), false);
        post(Tag{Kind::ffn_sum, layer, 11}, std::size_t{rows} * 6144U * sizeof(float), false);
    }
    // A prefill reduction block ends before head selection; drain carries the deferred tail CRC.
    if (!prefill) post(Tag{Kind::argmax, 60, 11}, std::size_t{rows} * 8U, false);
    if (previous[0]) post(Tag{Kind::control, 0xFFFFFFFFU, 0}, 0, true);
    flush();
    require(old_transport.messages == new_transport.messages && old_transport.bytes == new_transport.bytes);
    require(compared == (prefill && crc_mode != 2 ? 240U : 242U));
}
}
int main() {
    static_assert(MinimaxTraits::kCrc && MinimaxTraits::kCrcDeferred); // frozen source default
    static_assert(kHidden == 6144 && kRowF32 == 24576 && sizeof(Trailer) == 64);
    static_assert(static_cast<std::uint32_t>(Kind::gather) == 5);
    static_assert(MinimaxTraits::kHeadLayer == 60);
    static_assert(MinimaxTraits::memory_kind("host") == 0);
    static_assert(MinimaxTraits::memory_kind("device") == 1);
    static_assert(MinimaxTraits::memory_kind("uncached") == 2);
    static_assert(MinimaxTraits::memory_kind("invalid") == -1);
    try {
        require(crc32c(Bytes{std::byte{'1'}, std::byte{'2'}, std::byte{'3'}, std::byte{'4'},
            std::byte{'5'}, std::byte{'6'}, std::byte{'7'}, std::byte{'8'}, std::byte{'9'}}) == 0xE3069283U);
        require(std::string_view(MinimaxTraits::kEnvironment) == "STRIX_MMX_");
        require(std::string_view(MinimaxTraits::crc_description(true, false)) == "inline");
        require(std::string_view(MinimaxTraits::crc_description(true, true)) == "deferred");
        require(std::string_view(MinimaxTraits::crc_description(false, true)) == "off");
        for (unsigned mode = 0; mode < 3; ++mode) {
            run(1, false, mode); run(8, false, mode); run(32, true, mode);
        }
        std::cout << "PASS: MiniMax decode R=1/R=8, prefill block R=32; both ranks; N=8; CRC off/inline/deferred; all bytes/counts equal\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
