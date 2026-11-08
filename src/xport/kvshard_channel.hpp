#pragma once
#include "kvshard_traits.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace strix::xport::kvshard {
enum class Phase : std::uint64_t { queries = 1, partials = 2 };
struct AttentionTag {
    std::uint64_t request, layer, position, ownership_epoch;
    Phase phase;
    auto key() const { return std::make_tuple(ownership_epoch, request, position, layer); }
};

// A deliberately synchronous, host-coherent Channel adapter. Runtime supplies
// stream(), publish() and synchronize(stream). synchronize MUST complete the
// Channel's stream wait before the CPU inspects RX; check() then rejects a peer
// failure (the Channel may release a failed stream with unusable payload bytes).
// One adapter exclusively owns one Channel's post/wait sequence. TX/RX storage
// must hold max_frame_bytes. Destroy both after cancellation/protocol failure;
// there is no in-place recovery or reuse of an outstanding staging slot.
template<class Channel, class Runtime> class AttentionChannel {
public:
    static constexpr std::size_t query_floats = PartialDims::local_heads * PartialDims::accumulator;
    static constexpr std::size_t partial_floats = PartialDims::local_heads * PartialDims::fp32_per_head;
    using Header = std::array<std::uint64_t, 9>;
    static constexpr std::size_t max_frame_bytes = sizeof(Header) + partial_floats * sizeof(float);
    struct Ticket {
        typename Channel::Ticket wire;
        Header header;
    };
    AttentionChannel(Channel& channel, Runtime& runtime, bool enabled)
        : channel_(channel), runtime_(runtime), enabled_(enabled) {}
    AttentionChannel(const AttentionChannel&) = delete;
    AttentionChannel& operator=(const AttentionChannel&) = delete;

    Ticket post(const AttentionTag& tag, const typename Channel::Tag& wire_tag,
                const float* payload, std::size_t count) {
        require_live();
        if (pending_) throw std::logic_error("kvshard: one outstanding frame (pressure)");
        const bool queries = tag.phase == Phase::queries;
        if ((!queries && tag.phase != Phase::partials) ||
            count != (queries ? query_floats : partial_floats) || payload == nullptr)
            throw std::invalid_argument("kvshard: payload dimensions / phase");
        if (tag.layer > std::numeric_limits<std::uint16_t>::max() ||
            wire_tag.layer != tag.layer || wire_tag.step != tag.position)
            throw std::invalid_argument("kvshard: physical tag differs");
        if ((queries && (awaiting_partials_ || (have_last_ && tag.key() <= last_.key()))) ||
            (!queries && (!awaiting_partials_ || tag.key() != last_.key())))
            throw std::invalid_argument("kvshard: stale tag / phase order");
        if (sequence_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("kvshard: sequence exhausted");
        Header header{magic, 1, tag.request, tag.layer, tag.position,
                      static_cast<std::uint64_t>(tag.phase), tag.ownership_epoch, count, sequence_ + 1};
        try {
            channel_.check();
            auto staging = channel_.staging();
            std::memcpy(staging.data, header.data(), sizeof(header));
            std::memcpy(static_cast<unsigned char*>(staging.data) + sizeof(header), payload, count * sizeof(float));
            auto posted_ticket = channel_.post(wire_tag, staging, sizeof(header) + count * sizeof(float), runtime_.publish());
            expected_ = header;
            last_ = tag;
            have_last_ = pending_ = true;
            ++sequence_;
            return {posted_ticket, header};
        } catch (...) { failed_ = true; throw; }
    }

    std::vector<float> wait(const Ticket& ticket) {
        require_live();
        if (!pending_ || ticket.header != expected_)
            throw std::invalid_argument("kvshard: stale / unordered wait");
        try {
            const void* remote = channel_.wait(ticket.wire, runtime_.stream());
            runtime_.synchronize(runtime_.stream());
            channel_.check();
            if (remote == nullptr) throw std::runtime_error("kvshard: missing peer frame");
            Header received{};
            std::memcpy(received.data(), remote, sizeof(received));
            if (received != expected_) throw std::runtime_error("kvshard: peer tag / epoch / shape mismatch");
            const auto count = static_cast<std::size_t>(received[7]);
            std::vector<float> result(count);
            std::memcpy(result.data(), static_cast<const unsigned char*>(remote) + sizeof(Header), count * sizeof(float));
            pending_ = false;
            awaiting_partials_ = last_.phase == Phase::queries;
            return result;
        } catch (...) { failed_ = true; throw; }
    }
    void cancel() noexcept { failed_ = true; }
private:
    void require_live() const {
        if (!enabled_) throw std::logic_error("kvshard: opt-in required");
        if (failed_) throw std::runtime_error("kvshard: cancelled or failed; recreate channel");
    }
    static constexpr std::uint64_t magic = 0x4b56534841524431ULL;
    Channel& channel_;
    Runtime& runtime_;
    bool enabled_, pending_ = false, failed_ = false, awaiting_partials_ = false, have_last_ = false;
    std::uint64_t sequence_ = 0;
    Header expected_{};
    AttentionTag last_{};
};
} // namespace strix::xport::kvshard
