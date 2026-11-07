#include "xport/ds41_adapter.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
using Values = std::array<float, 4>;
struct Tag { unsigned kind, layer; std::uint64_t request; };
struct ReadyView { void* data; unsigned token; };
void require(bool condition) {
    if (!condition) throw std::runtime_error("DS41 half_sum fixture failed");
}
struct FakeCommunicator {
    Values peer_values{1.0F, -4.0F, 8.0F, -0.0F};
    Values tx_values{}, rx_values{};
    std::vector<std::byte> sent;
    std::vector<unsigned> calls;
    std::size_t payload_bytes = 0, messages = 0;
    bool held = false;
    void start(unsigned call_stream, void* payload, std::size_t payload_size, const Tag& call_tag) {
        require(call_stream == 7 && payload_size == sizeof(Values));
        require(call_tag.kind == 3 && call_tag.layer == 11 && call_tag.request == 29);
        const auto* first_byte = static_cast<const std::byte*>(payload);
        sent.insert(sent.end(), first_byte, first_byte + payload_size);
        payload_bytes += payload_size;
        ++messages;
        calls.push_back(1);
    }
    void finish(unsigned call_stream, void* destination) {
        require(call_stream == 7);
        std::memcpy(destination, peer_values.data(), sizeof(Values));
        calls.push_back(3);
    }
    void exchange(unsigned call_stream, void* payload, void* destination,
                  std::size_t payload_size, const Tag& call_tag) {
        start(call_stream, payload, payload_size, call_tag);
        finish(call_stream, destination);
    }
    ReadyView prepare_staged_ready(unsigned call_stream, std::size_t payload_size) {
        require(call_stream == 7 && payload_size == sizeof(Values));
        calls.push_back(0);
        return {tx_values.data(), 41};
    }
    void start_staged_ready(unsigned call_stream, std::size_t payload_size, const Tag& call_tag) {
        start(call_stream, tx_values.data(), payload_size, call_tag);
    }
    ReadyView finish_staged_ready() {
        finish(7, rx_values.data());
        held = true;
        return {rx_values.data(), 43};
    }
    void release_staged_ready(unsigned call_stream) {
        require(call_stream == 7 && held);
        held = false;
        calls.push_back(5);
        rx_values.fill(999.0F); // Detect consumption after release.
    }
    void start_staged(unsigned staged_stream, std::size_t staged_bytes, const Tag& staged_tag) {
        start_staged_ready(staged_stream, staged_bytes, staged_tag);
    }
    const void* finish_staged(unsigned staged_stream) {
        require(staged_stream == 7);
        return finish_staged_ready().data;
    }
    void release_staged(unsigned staged_stream) { release_staged_ready(staged_stream); }
};

// Host rank-order oracle; device consumer kernels remain unchanged by the seam.
Values reduce(const Values& own_values, const Values& remote_values, unsigned rank_value,
              bool shared_value) {
    const Values& p0 = rank_value == 0 ? own_values : remote_values;
    const Values& p1 = rank_value == 0 ? remote_values : own_values;
    Values result_values{};
    for (std::size_t item = 0; item < result_values.size(); ++item) {
        const float rank_sum = p0[item] + p1[item];
        result_values[item] = shared_value ? 2.0F + rank_sum : rank_sum;
    }
    return result_values;
}
void run_case(bool staged_case, bool overlap_case, bool shared_case, unsigned rank_case) {
    FakeCommunicator old_comm, new_comm;
    Values old_own{4.0F, 1.0F, -8.0F, 0.0F}, new_own = old_own;
    Values old_other{}, new_other{}, old_result{}, new_result{};
    const Tag half_tag{3, 11, 29};
    const auto overlap_old = [&] { if (overlap_case) old_comm.calls.push_back(2); };
    const auto overlap_new = [&] { if (overlap_case) new_comm.calls.push_back(2); };
    if (staged_case) {
        const auto old_stage = old_comm.prepare_staged_ready(7, sizeof(Values));
        const auto new_stage = new_comm.prepare_staged_ready(7, sizeof(Values));
        std::memcpy(old_stage.data, old_own.data(), sizeof(Values));
        std::memcpy(new_stage.data, new_own.data(), sizeof(Values));
        old_comm.start_staged_ready(7, sizeof(Values), half_tag);
        overlap_old();
        const auto old_peer = old_comm.finish_staged_ready();
        require(old_peer.token == 43 && old_comm.held);
        old_result = reduce(old_comm.tx_values, old_comm.rx_values, rank_case, shared_case);
        old_comm.calls.push_back(4);
        old_comm.release_staged_ready(7);
        strix::xport::ds41_exchange_half_sum_ready(new_comm, 7U, new_stage.data, sizeof(Values), half_tag,
            overlap_new, [&](const auto& new_peer) {
                require(new_peer.token == 43 && new_peer.data == new_comm.rx_values.data() && new_comm.held);
                new_result = reduce(new_comm.tx_values, new_comm.rx_values, rank_case, shared_case);
                new_comm.calls.push_back(4);
            });
    } else {
        if (overlap_case) {
            old_comm.start(7, old_own.data(), sizeof(Values), half_tag);
            overlap_old();
            old_comm.finish(7, old_other.data());
        } else {
            old_comm.exchange(7, old_own.data(), old_other.data(), sizeof(Values), half_tag);
        }
        strix::xport::ds41_exchange_half_sum(new_comm, 7U, new_own.data(), new_other.data(),
            sizeof(Values), half_tag, overlap_new);
        old_result = reduce(old_own, old_other, rank_case, shared_case);
        new_result = reduce(new_own, new_other, rank_case, shared_case);
        require(std::memcmp(old_other.data(), new_other.data(), sizeof(Values)) == 0);
    }
    require(std::memcmp(old_result.data(), new_result.data(), sizeof(Values)) == 0);
    require(old_comm.calls == new_comm.calls && old_comm.sent == new_comm.sent);
    require(old_comm.payload_bytes == new_comm.payload_bytes && new_comm.payload_bytes == sizeof(Values));
    require(old_comm.messages == new_comm.messages && new_comm.messages == 1);
    require(!new_comm.held);
    const Values expected_result = shared_case ? Values{7.0F, -1.0F, 2.0F, 2.0F} : Values{5.0F, -3.0F, 0.0F, 0.0F};
    require(std::memcmp(expected_result.data(), new_result.data(), sizeof(Values)) == 0);
}
} // namespace
void run_prefill_case(bool joint_case, unsigned prefill_rank) {
    FakeCommunicator parent_comm, adapter_comm;
    const Values produced_half{1.0e20F, 1.0F, -8.0F, -0.0F};
    parent_comm.peer_values = {-1.0e20F, -4.0F, 8.0F, 0.0F};
    adapter_comm.peer_values = parent_comm.peer_values;
    parent_comm.tx_values = produced_half;
    adapter_comm.tx_values = produced_half;
    const Tag prefill_tag{3, 11, 29};
    Values parent_sum{}, adapter_sum{};
    const auto consume_half = [&](FakeCommunicator& active_comm, const void* peer_data, Values& sum_values) {
        require(active_comm.held && peer_data == active_comm.rx_values.data());
        const auto* peer_half_values = static_cast<const float*>(peer_data);
        for (std::size_t value_index = 0; value_index < sum_values.size(); ++value_index) {
            const float low_value = prefill_rank == 0 ? produced_half[value_index] : peer_half_values[value_index];
            const float high_value = prefill_rank == 0 ? peer_half_values[value_index] : produced_half[value_index];
            const float shared_low_value = 2.0F + low_value;
            sum_values[value_index] = shared_low_value + high_value;
        }
        active_comm.calls.push_back(4);
    };
    parent_comm.start_staged(7, sizeof(Values), prefill_tag);
    if (!joint_case) parent_comm.calls.push_back(2); // Own-row work after start.
    parent_comm.calls.push_back(6); // moe.sum boundary, also before finish.
    const void* parent_peer = parent_comm.finish_staged(7);
    consume_half(parent_comm, parent_peer, parent_sum);
    parent_comm.release_staged(7);

    strix::xport::Ds41HalfSumStagedExchange<FakeCommunicator, unsigned, Tag> prefill_exchange{
        adapter_comm, 7U, adapter_comm.tx_values.data(), sizeof(Values), prefill_tag};
    if (!joint_case) adapter_comm.calls.push_back(2);
    adapter_comm.calls.push_back(6);
    const void* adapter_peer = prefill_exchange.finish(7U);
    consume_half(adapter_comm, adapter_peer, adapter_sum);
    prefill_exchange.release(7U);
    require(!parent_comm.held && !adapter_comm.held);
    require(parent_comm.calls == adapter_comm.calls && parent_comm.sent == adapter_comm.sent);
    require(parent_comm.messages == 1 && adapter_comm.messages == 1);
    require(parent_comm.payload_bytes == sizeof(Values) && adapter_comm.payload_bytes == sizeof(Values));
    require(std::memcmp(parent_sum.data(), adapter_sum.data(), sizeof(Values)) == 0);
    require(adapter_sum[0] == 0.0F); // (shared + low) + high, not shared + (low + high).
}

int main() {
    for (unsigned prefill_rank = 0; prefill_rank < 2; ++prefill_rank) {
        run_prefill_case(false, prefill_rank);
        run_prefill_case(true, prefill_rank);
    }
    for (unsigned rank_case = 0; rank_case < 2; ++rank_case) {
        for (bool overlap_case : {false, true}) {
            run_case(true, overlap_case, true, rank_case);
            run_case(false, overlap_case, true, rank_case);
        }
        run_case(false, false, false, rank_case); // shexp TP: S_r + P_r already in the payload.
    }
    std::cout << "DS41 half_sum: 10 decode + 4 prefill staged/rank-order cases passed\n";
}
