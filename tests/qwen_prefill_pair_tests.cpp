#include "strix/qwen_prefill_pair.hpp"
#include "strix/qwen_prefill_balance.hpp"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>

namespace {
struct Wire {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> messages;
    bool closed = false;
    void send(std::span<const std::byte> bytes) {
        std::lock_guard lock(mutex);
        messages.emplace_back(bytes.begin(), bytes.end()); cv.notify_all();
    }
    void receive(std::span<std::byte> bytes) {
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return closed || !messages.empty(); }))
            throw std::runtime_error("fake transport timed out");
        if (messages.empty()) throw std::runtime_error("fake transport closed");
        auto message = std::move(messages.front()); messages.pop_front();
        if (message.size() != bytes.size()) throw std::runtime_error("fake transport byte count mismatch");
        std::copy(message.begin(), message.end(), bytes.begin());
    }
    void stop() { std::lock_guard lock(mutex); closed = true; cv.notify_all(); }
};
void require(bool value) { if (!value) throw std::runtime_error("pair prefill schedule assertion failed"); }
struct MockStaging final : strix::qwen_pp::PipeStaging {
    std::vector<std::byte> storage;
    std::mutex& mutex;
    std::condition_variable& cv;
    std::uint64_t& received;
    MockStaging(std::mutex& lock, std::condition_variable& condition, std::uint64_t& count)
        : mutex(lock), cv(condition), received(count) {}
    std::span<std::byte> resize(std::size_t bytes) override { storage.resize(bytes); return storage; }
    void copy(bool, std::size_t, void*, std::size_t) override {}
    void seal(std::size_t) override {}
    void await_piece(std::size_t) override {}
    void received_piece(std::size_t, std::size_t) override {}
    void finish_receive() override { std::lock_guard lock(mutex); ++received; cv.notify_all(); }
};
void balance_checks() {
    using namespace strix::qwen_pp;
    LegalCuts legal{}; legal.fill(true);
    require(parse_cut(nullptr, legal) == 24U);
    for (const char* bad : {"0", "48", "-1", "24x", "", " 24"}) {
        bool rejected = false;
        try { (void)parse_cut(bad, legal); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    legal[25] = false;
    bool rejected = false;
    try { (void)parse_cut("25", legal); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    LayerCosts costs; costs.layers.fill(1); costs.lm_head = 8;
    require(choose_cut(costs, legal).cut == 28U);
    legal[28] = false; require(choose_cut(costs, legal).cut == 27U);
    auto prediction = predict_cut(costs, 24);
    require(prediction.a == 24 && prediction.b == 32);
    std::string layers = "{";
    for (unsigned layer = 0; layer < 48U; ++layer) {
        if (layer != 0U) layers += ',';
        layers += '"' + std::to_string(layer) + "\":1";
    }
    layers += '}';
    const auto bucket = [&](unsigned context, unsigned head) {
        return "{\"context\":" + std::to_string(context) + ",\"layers\":" + layers + ",\"lm_head\":" + std::to_string(head) + '}';
    };
    const auto json = "{\"buckets\":[" + bucket(131072, 8) + ',' + bucket(32768, 0) + "]}";
    require(parse_costs(json, 32000).context == 32768U);
    require(parse_costs(json, 32769).lm_head == 8);
    require(parse_costs(json, 262144).context == 131072U);
    require(choose_cut(parse_costs(json, 32769), legal).cut == 27U);
    for (const std::string& bad : {std::string("{\"buckets\":[]}"),
        "{\"buckets\":[" + bucket(1, 0) + ',' + bucket(1, 0) + "]}",
        std::string("{\"buckets\":[{\"context\":1,\"layers\":{},\"lm_head\":0}]}")}) {
        rejected = false;
        try { (void)parse_costs(bad, 1); } catch (const std::exception&) { rejected = true; }
        require(rejected);
    }
    legal.fill(false); rejected = false;
    try { (void)choose_cut(costs, legal); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
void chunk_checks() {
    using namespace strix::qwen_pp;
    LayerCosts costs; costs.layers.fill(0.001); costs.has_transfer = true;
    costs.transfer_ms_per_row = 0.01; costs.fixed_ms = 20;
    const auto prediction = predict_cut(costs, 24);
    require(choose_chunk("auto", 16384, 32768, &costs, prediction).rows == 2048);
    require(choose_chunk(nullptr, 16384, 32768, &costs, prediction).rows == 4096);   // measured default
    require(choose_chunk(nullptr, 16384, 131072, nullptr, prediction).rows == 4096);
    require(choose_chunk(nullptr, 16384, 4096, nullptr, prediction).rows == 16384);  // 4096 not < prompt: legacy
    require(choose_chunk(nullptr, 2048, 32768, nullptr, prediction).rows == 2048);   // above admitted tile: legacy
    require(choose_chunk("auto", 8192, 131072, &costs, prediction).rows == 2048);
    require(choose_chunk("legacy", 16384, 32768, &costs, prediction).rows == 16384);
    require(choose_chunk("auto", 16384, 32768, nullptr, prediction).rows == 16384);
    costs.has_transfer = false;
    require(choose_chunk("auto", 8192, 32768, &costs, prediction).rows == 8192);
    costs.has_transfer = true; costs.fixed_ms = 1e6;
    require(choose_chunk("auto", 8192, 32768, &costs, prediction).rows == 8192);
    for (const char* value : {"2048", "4096", "8192", "16", "32"})
        require(valid_chunk(choose_chunk(value, 16384, 32768, nullptr, prediction).rows, 32768));
    for (const char* value : {"", "0", "8", "17", "-2048", "4096x", "32768"}) {
        bool rejected = false;
        try { (void)choose_chunk(value, 16384, 32768, nullptr, prediction); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    require(!valid_chunk(2048, 32776)); // eight-row terminal tail cannot execute
    require(!valid_chunk(2048, 0));
    costs.fixed_ms = 20;
    require(legal_prefix(1193, 4096) == 1192);
    require(legal_prefix(32777, 4096) == 32768);
    require(legal_prefix(32793, 4096) == 32792);
    require(choose_chunk(nullptr, 16384, 32769, nullptr, prediction).rows == 4096);
    require(choose_prompt_route(1193, 4096, nullptr, prediction).prefix == 0);
    require(choose_prompt_route(32771, 4096, nullptr, prediction).prefix == 32768);
    require(choose_prompt_route(24576, 4096, nullptr, prediction).prefix == 0);
    require(choose_prompt_route(28672, 4096, nullptr, prediction).prefix == 28672);
    require(choose_prompt_route(32777, 4096, nullptr, prediction).prefix == 32768);
    require(choose_prompt_route(32793, 4096, nullptr, prediction).prefix == 32792);
    auto route_costs = costs;
    route_costs.fixed_ms = 0; route_costs.transfer_ms_per_row = 0;
    require(choose_prompt_route(131072, 4096, &route_costs, CutPrediction{24U, 0.3, 0.3}).prefix != 0);
    route_costs.fixed_ms = 1e9;
    require(choose_prompt_route(131072, 4096, &route_costs, CutPrediction{24U, 0.3, 0.3}).prefix == 0);
    const auto bucket = [&](const std::string& extra) {
        std::string text = "{\"buckets\":[{\"context\":32768,\"layers\":{";
        for (unsigned i = 0; i < 48; ++i) text += (i ? "," : "") + std::string("\"") + std::to_string(i) + "\":0.001";
        return text + "},\"lm_head\":0" + extra + "}]}";
    };
    require(parse_costs(bucket(",\"transfer_ms_per_row\":0.01,\"fixed_ms\":20"), 32768).has_transfer);
    for (const auto extra : {",\"fixed_ms\":20", ",\"transfer_ms_per_row\":-1,\"fixed_ms\":20"}) {
        bool rejected = false;
        try { (void)parse_costs(bucket(extra), 32768); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
}
void blocked_send_overlap_checks() {
    using namespace strix::qwen_pp;
    Wire forward, backward;
    std::mutex gate; std::condition_variable cv;
    bool sending = false, posted_next = false;
    const auto abort = [&] { forward.stop(); backward.stop(); };
    Pipe producer(0, 2, [&](auto bytes) {
        if (bytes.size() == sizeof(PipeFrame::header)) {
            std::unique_lock lock(gate);
            if (!sending) {
                sending = true; cv.notify_all();
                if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return posted_next; }))
                    throw std::runtime_error("stage A did not post next chunk during blocked send");
            }
        }
        forward.send(bytes);
    }, [&](auto bytes) { backward.receive(bytes); }, abort);
    Pipe consumer(1, 2, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort);
    for (std::uint64_t i = 0; i < 2U; ++i) {
        auto& frame = producer.acquire(i);
        frame.header[0] = Seam::magic; frame.header[1] = i;
        frame.header[3] = 16; frame.header[4] = 24; frame.header[41] = 2;
        frame.bytes.assign(16U * Seam::row_bytes, static_cast<std::byte>(i));
        producer.release(i);
        std::unique_lock lock(gate);
        if (i == 0U) require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return sending; }));
        else { posted_next = true; cv.notify_all(); }
    }
    for (std::uint64_t i = 0; i < 2U; ++i) {
        auto& frame = consumer.acquire(i);
        require(frame.header[1] == i);
        require(std::all_of(frame.payload.begin(), frame.payload.end(), [&](auto byte) { return byte == static_cast<std::byte>(i); }));
        consumer.release(i);
    }
    consumer.finish(2); producer.finish(2);
}
void crc_checks() {
    using namespace strix::qwen_pp;
    const std::string known = "123456789";
    SeamCrc check; check.update(std::as_bytes(std::span(known)));
    require(check.value() == 0xe3069283U);
    std::vector<std::byte> data(seam_piece_bytes * 2U + 17U);
    for (std::size_t index = 0; index < data.size(); ++index) data[index] = static_cast<std::byte>(index % 251U);
    SeamCrc whole, pieces, combined; whole.update(data);
    for (std::size_t offset = 0; offset < data.size(); offset += seam_piece_bytes) {
        const auto piece = std::span(data).subspan(offset, std::min(seam_piece_bytes, data.size() - offset));
        pieces.update(piece); SeamCrc checksum; checksum.update(piece); combined.append(checksum.value(), piece.size());
    }
    require(whole.value() == pieces.value() && whole.value() == combined.value());
    SeamCrc empty; combined.append(empty.value(), 0); require(whole.value() == combined.value());
}
void fault_checks() {
    using namespace strix::qwen_pp;
    // Exercise actual multi-piece wire framing, and refuse corruption and peer cut drift.
    for (unsigned fault = 0; fault < 4U; ++fault) {
        Wire forward, backward;
        const auto abort = [&] { forward.stop(); backward.stop(); };
        unsigned messages = 0;
        Pipe producer(0, 1, [&](auto bytes) {
            ++messages;
            if ((fault == 1U && messages == 3U) || (fault == 3U && messages == 6U)) {
                std::vector<std::byte> corrupt(bytes.begin(), bytes.end()); corrupt[0] ^= std::byte{1}; forward.send(corrupt);
            } else forward.send(bytes);
        }, [&](auto bytes) { backward.receive(bytes); }, abort, 28);
        Pipe consumer(1, 1, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort,
            fault == 2U ? 24U : 28U);
        auto& sent = producer.acquire(0);
        sent.header[0] = Seam::magic; sent.header[1] = 0; sent.header[3] = 700; sent.header[4] = 28; sent.header[41] = 1;
        sent.bytes.assign(700U * Seam::row_bytes, std::byte{37}); producer.release(0);
        bool rejected = false;
        try {
            auto& received = consumer.acquire(0);
            require(received.payload.size() == sent.bytes.size());
            require(std::all_of(received.payload.begin(), received.payload.end(), [](auto byte) { return byte == std::byte{37}; }));
            consumer.release(0); consumer.finish(1); producer.finish(1);
        } catch (const std::exception&) { rejected = true; }
        require(rejected == (fault != 0U));
    }
}

}
int main() {
    try {
        balance_checks(); chunk_checks(); blocked_send_overlap_checks(); crc_checks(); fault_checks();
        Wire forward, backward;
        std::mutex received_mutex; std::condition_variable received_cv; std::uint64_t received_chunks = 0;
        const auto abort = [&] { forward.stop(); backward.stop(); };
        strix::qwen_pp::Pipe producer(0, 7, [&](auto bytes) { forward.send(bytes); }, [&](auto bytes) { backward.receive(bytes); }, abort);
        strix::qwen_pp::Pipe consumer(1, 7, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort, 24,
            [&] { return std::make_shared<MockStaging>(received_mutex, received_cv, received_chunks); });
        std::exception_ptr producer_error;
        std::array<const void*, 2> addresses{};
        std::mutex observed_mutex;
        std::condition_variable observed_cv;
        std::uint64_t submitted = 0;
        std::thread stage_a([&] {
            try {
                for (std::uint64_t sequence = 0; sequence < 7U; ++sequence) {
                    auto& frame = producer.acquire(sequence);
                    const auto slot = static_cast<std::size_t>(sequence % 2U);
                    if (sequence < 2U) addresses[slot] = &frame;
                    else require(addresses[slot] == &frame);
                    const auto rows = static_cast<std::size_t>(sequence + 1U);
                    frame.header[0] = strix::qwen_pp::Seam::magic; frame.header[1] = sequence;
                    frame.header[41] = 7U;
                    frame.header[2] = sequence * 16U; frame.header[3] = rows; frame.header[4] = 24U;
                    frame.bytes.assign(rows * strix::qwen_pp::Seam::row_bytes, static_cast<std::byte>(sequence));
                    producer.release(sequence);
                    { std::lock_guard lock(observed_mutex); submitted = sequence + 1U; observed_cv.notify_all(); }
                }
                producer.finish(7);
            } catch (...) { producer_error = std::current_exception(); abort(); }
        });
        std::size_t total_bytes = 0;
        try {
            for (std::uint64_t sequence = 0; sequence < 7U; ++sequence) {
                auto& frame = consumer.acquire(sequence);
                if (sequence == 0U) {
                    std::unique_lock lock(observed_mutex);
                    require(observed_cv.wait_for(lock, std::chrono::seconds(5), [&] { return submitted == 2U; }));
                    {
                        std::unique_lock received_lock(received_mutex);
                        require(received_cv.wait_for(received_lock, std::chrono::seconds(5), [&] { return received_chunks == 2U; }));
                    }
                    // Stage A has completed the next chunk while stage B owns chunk 0.
                    // Slot 0 cannot be acquired for chunk 2 before this consumer releases it.
                    require(submitted == 2U);
                }
                require(frame.header[1] == sequence);
                require(frame.payload.size() == static_cast<std::size_t>(sequence + 1U) * strix::qwen_pp::Seam::row_bytes);
                require(std::all_of(frame.payload.begin(), frame.payload.end(), [&](auto value) { return value == static_cast<std::byte>(sequence); }));
                total_bytes += frame.payload.size();
                consumer.release(sequence);
            }
            consumer.finish(7);
        } catch (...) { abort(); stage_a.join(); throw; }
        stage_a.join();
        if (producer_error) std::rethrow_exception(producer_error);
        require(total_bytes == 28U * strix::qwen_pp::Seam::row_bytes);
        std::cout << "PASS: chunk policy/transfer schema, blocked-send overlap, cut/model/bucket validation, multi-piece CRC/corruption/cut drift, H2D lookahead before B release, 7 ordered chunks, 2 slots per rank, consumer acknowledgement before reuse, " << total_bytes << " seam bytes\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
