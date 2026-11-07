#include "xport/staging.hpp"
#include "xport/communicator_backend.hpp"
#include "glm/xfer/glm_traits.hpp"
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>

namespace {
void require(bool value) { if (!value) throw std::runtime_error("staging simulator assertion"); }
using Tag = strix::glm::xfer::Tag;
using Key = std::tuple<unsigned, unsigned, std::uint64_t>;
Key key(const Tag& tag) { return {static_cast<unsigned>(tag.kind), tag.layer, tag.step}; }
struct Signal { bool done = false; };
struct Event { std::shared_ptr<Signal> signal; unsigned slot = 0; };
struct Network {
    std::map<Key, float> payload[2];
    std::map<Key, bool> delivered[2];
};
struct Ops {
    using Stream = unsigned;
    using Event = ::Event;
    Network& network;
    unsigned rank;
    std::array<float, 8> storage{};
    std::array<std::pair<std::shared_ptr<Signal>, std::shared_ptr<Signal>>, 8> dependencies{};
    std::array<bool, 8> occupied{};
    std::map<Key, unsigned> locations;
    std::map<Key, std::shared_ptr<Signal>> source;
    std::map<unsigned, std::shared_ptr<Signal>> consumers;
    unsigned initialized = 0, waits = 0, blocked = 0;
    void destroy(const Event&) {}
    void init(Event& src, Event& last, Stream) {
        if (src.signal) return;
        src = {std::make_shared<Signal>(Signal{true}), initialized};
        last = {std::make_shared<Signal>(Signal{true}), initialized++};
    }
    void reuse(const Event& src, const Event& last, Stream) {
        dependencies[src.slot] = {src.signal, last.signal};
        waits += 2;
    }
    void wait_source(const Event&, Stream) {}
    void* remote(unsigned slot, unsigned) { return &storage[slot]; }
    bool ready(unsigned slot) const {
        const auto& d = dependencies[slot];
        return (!d.first || d.first->done) && (!d.second || d.second->done);
    }
    template<class View>
    std::size_t post(const Tag& tag, View view, std::size_t bytes, bool,
                     Event& src, Event&) {
        require(!occupied[view.slot]);
        occupied[view.slot] = true;
        if (!ready(view.slot)) ++blocked;
        network.payload[rank][key(tag)] = *static_cast<float*>(view.local);
        locations[key(tag)] = view.slot;
        src.signal = std::make_shared<Signal>();
        source[key(tag)] = src.signal;
        return bytes;
    }
    template<class View>
    const void* wait(View view, Stream, Event&) {
        require(ready(view.slot)); // an enqueued write cannot pass either event
        auto it = locations.begin();
        while (it != locations.end() && it->second != view.slot) ++it;
        require(it != locations.end());
        require(network.delivered[1 - rank].at(it->first));
        storage[view.slot] = network.payload[1 - rank].at(it->first);
        locations.erase(it);
        return &storage[view.slot];
    }
    void release(Event& last, Stream) {
        occupied[last.slot] = false;
        last.signal = std::make_shared<Signal>();
        consumers[last.slot] = last.signal;
    }
};
using Channel = strix::xport::StagingChannel<Tag, Ops>;
Channel::Counters run(unsigned depth, bool trace = true) {
    Network network;
    Ops op0{network, 0}, op1{network, 1};
    Channel c0(op0, depth, trace), c1(op1, depth, trace);
    std::array<float, 16> local0{}, local1{};
    std::array<Channel::Ticket, 16> t0{}, t1{};
    for (unsigned base = 0; base < 16; base += depth) {
        for (unsigned i = base; i < base + depth; ++i) {
            local0[i] = static_cast<float>(i + 1);
            local1[i] = static_cast<float>(100 + i);
            const Tag tag{strix::glm::xfer::Kind::attn_sum, i % 3, i};
            t0[i] = c0.post(tag, c0.staging(&local0[i], 0, 0, i % depth), sizeof(float), true);
            t1[i] = c1.post(tag, c1.staging(&local1[i], 0, 0, i % depth), sizeof(float), true);
        }
        bool full = false;
        try { (void)c0.staging(&local0[base], 0, 0, 0); }
        catch (const std::logic_error&) { full = true; }
        require(full);
        // Source completion alone cannot authorize the second use of a view.
        if (base) {
            require(op0.blocked != 0 && op1.blocked != 0);
            if (base == 8) {
                for (unsigned slot = 0; slot < depth; ++slot) {
                    op0.consumers.at(slot)->done = true;
                    op1.consumers.at(slot)->done = true;
                    require(!op0.ready(slot) && !op1.ready(slot));
                }
            }
            for (auto& entry : op0.source) if (std::get<2>(entry.first) < base) entry.second->done = true;
            for (auto& entry : op1.source) if (std::get<2>(entry.first) < base) entry.second->done = true;
            for (unsigned slot = 0; slot < depth; ++slot) {
                if (base != 8) require(!op0.ready(slot) && !op1.ready(slot));
                op0.consumers.at(slot)->done = true;
                op1.consumers.at(slot)->done = true;
                require(op0.ready(slot) && op1.ready(slot));
            }
        }
        auto bad = t0[base];
        ++bad.tag.layer;
        bool rejected = false;
        try { (void)c0.wait(bad, 0); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
        // Deliver and consume in reverse order, independently of slot position.
        for (unsigned j = depth; j != 0; --j) {
            const unsigned i = base + j - 1;
            network.delivered[0][key(t0[i].tag)] = true;
            network.delivered[1][key(t1[i].tag)] = true;
            const auto v0 = c0.wait(t0[i], 0), v1 = c1.wait(t1[i], 0);
            const float *p0 = nullptr, *p1 = nullptr;
            strix::xport::rank_pair(0, &local0[i], static_cast<const float*>(v0.remote), &p0, &p1);
            require(*p0 == local0[i] && *p1 == local1[i]);
            const float sum0 = *p0 + *p1;
            strix::xport::rank_pair(1, &local1[i], static_cast<const float*>(v1.remote), &p0, &p1);
            require(*p0 == local0[i] && *p1 == local1[i] && sum0 == *p0 + *p1);
            c0.release(t0[i], 0);
            c1.release(t1[i], 0);
        }
        // Release records a consumer fence, but does not complete either fence.
        if (base == 0) {
            for (unsigned slot = 0; slot < depth; ++slot)
                require(!op0.consumers.at(slot)->done && !op1.consumers.at(slot)->done);
        }
    }
    require(c0.counters().bytes == c1.counters().bytes);
    require(c0.counters().exchanges == (trace ? 16U : 0U) && c1.counters().exchanges == (trace ? 16U : 0U));
    return c0.counters();
}
}

namespace backend_fixture {
struct Runtime {
    using Stream = unsigned;
    using Event = unsigned;
    unsigned stream = 0, source_records = 0, consumer_records = 0, copies = 0;
    bool source_done[9]{}, consumer_done[9]{};
    float storage[8]{};
    unsigned initialized = 0, reuse_waits = 0;
    void destroy(Event) {}
    void init(Event& source, Event& consumer, Stream s) {
        if (!source) source = consumer = ++initialized;
        stream = s;
    }
    void reuse(Event source, Event consumer, Stream) {
        require(source_done[source] && consumer_done[consumer]);
        reuse_waits += 2;
    }
    void wait_source(Event source, Stream) { require(source_done[source]); }
    void* remote(unsigned slot, unsigned) { return &storage[slot]; }
    void copy(void* from, void* to, std::size_t, Event consumer) {
        require(consumer_done[consumer]);
        *static_cast<float*>(to) = *static_cast<float*>(from); ++copies;
    }
    void record_source(Event, Stream, bool) { ++source_records; }
    void record_remote(Event, Stream) { ++consumer_records; }
};
struct Communicator {
    Network& network;
    unsigned rank;
    float tx = 0, rx = 0;
    Tag tag{};
    std::size_t bytes = 0, messages = 0;
    bool staging = true, asynchronous = true;
    void* send_staging(std::size_t) { return staging ? &tx : nullptr; }
    void start_staged(unsigned, std::size_t n, const Tag& t) { start(0, &tx, n, t); }
    void start(unsigned, void* local, std::size_t n, const Tag& t) {
        tag = t; network.payload[rank][key(t)] = *static_cast<float*>(local);
        bytes += n; ++messages;
    }
    bool start_async(unsigned stream, void* local, void*, std::size_t n,
                     const Tag& t, unsigned, unsigned) {
        start(stream, local, n, t); return asynchronous;
    }
    void finish(unsigned, void* remote) {
        *static_cast<float*>(remote) = network.payload[1-rank].at(key(tag));
    }
    const void* finish_staged(unsigned stream) { finish(stream, &rx); return &rx; }
    void release_staged(unsigned) {}
};
using B = strix::xport::CommunicatorBackend<Communicator, Runtime, Tag>;
static_assert(strix::xport::Backend<B> && B::max_outstanding == 1);
void run() {
    // Eight reserved/tagged requests per direction, serialized on one wire credit.
    // Run staged, async and synchronous/fallback outcomes against step-3 Ops.
    for (unsigned mode = 0; mode < 4; ++mode) {
        Network network, reference;
        Communicator comm[2]{{network, 0}, {network, 1}};
        Runtime runtime[2];
        B back[2]{{comm[0], runtime[0]}, {comm[1], runtime[1]}};
        Ops old[2]{{reference, 0}, {reference, 1}};
        Channel path[2]{Channel(old[0], 8, true), Channel(old[1], 8, true)};
        using Bridge = strix::xport::BackendStagingOps<B>;
        Bridge bridge[2]{Bridge(back[0]), Bridge(back[1])};
        using Staged = strix::xport::StagingChannel<Tag, Bridge>;
        Staged adapted[2]{Staged(bridge[0], 8, true), Staged(bridge[1], 8, true)};
        Staged::Ticket leases[2];
        float local[2][8]{};
        std::size_t copied[2]{};
        for (unsigned round = 0; round < 2; ++round) {
            for (unsigned i = 0; i < 8; ++i) {
                const Tag tag{strix::glm::xfer::Kind::attn_sum, i % 3, round * 8 + i};
                Channel::Ticket previous[2];
                for (unsigned rank = 0; rank < 2; ++rank) {
                    auto& rt = runtime[rank];
                    if (round) {
                        require(!rt.source_done[i+1] && !rt.consumer_done[i+1]);
                        rt.source_done[i+1] = true;
                        require(!rt.consumer_done[i+1]); // source fence alone is insufficient
                        rt.consumer_done[i+1] = true;
                        old[rank].source.at(key(Tag{tag.kind, i % 3, i}))->done = true;
                        old[rank].consumers.at(i)->done = true;
                    } else rt.source_done[i+1] = rt.consumer_done[i+1] = true;
                    require(rt.source_done[i+1] && rt.consumer_done[i+1]);
                    local[rank][i] = static_cast<float>(100 * rank + round * 8 + i);
                    comm[rank].staging = mode == 0;
                    comm[rank].asynchronous = mode != 2;
                    const auto slot = adapted[rank].staging(&local[rank][i], 0, 0, i);
                    leases[rank] = adapted[rank].post(tag, slot, sizeof(float),
                                                     strix::xport::BackendPublish{mode != 3, mode != 3});
                    const auto view = B::Buffer{slot.local, slot.remote,
                        strix::xport::AddressSpace::device, i+1, i+1, slot.lease};
                    copied[rank] += back[rank].copied_bytes();
                    bool refused = false;
                    try { (void)back[rank].post(tag, view, sizeof(float), {true, true}); }
                    catch (const std::logic_error&) { refused = true; }
                    require(refused);
                    previous[rank] = path[rank].post(tag,
                        path[rank].staging(&local[rank][i], 0, 0, i), sizeof(float), true);
                    reference.delivered[rank][key(tag)] = true;
                }
                for (unsigned rank = 0; rank < 2; ++rank) {
                    const auto rx = adapted[rank].wait(leases[rank], 0);
                    const auto was = path[rank].wait(previous[rank], 0);
                    require(*static_cast<float*>(rx.remote) == *static_cast<float*>(was.remote));
                    bool refused = false;
                    try { back[rank].drain(); } catch (const std::logic_error&) { refused = true; }
                    require(refused);
                    adapted[rank].release(leases[rank], 0);
                    path[rank].release(previous[rank], 0);
                    runtime[rank].source_done[i+1] = runtime[rank].consumer_done[i+1] = false;
                }
            }
        }
        for (unsigned rank = 0; rank < 2; ++rank) {
            back[rank].drain();
            require(runtime[rank].reuse_waits == 16);
            require(adapted[rank].counters().bytes == path[rank].counters().bytes);
            require(adapted[rank].counters().exchanges == path[rank].counters().exchanges);
            require(comm[rank].bytes == path[rank].counters().bytes);
            require(comm[rank].messages == path[rank].counters().exchanges);
            require(copied[rank] == (mode == 0 ? path[rank].counters().staging_bytes : 0));
            require(runtime[rank].source_records == (mode == 1 ? 0U : 16U));
            require(runtime[rank].consumer_records == 16);
        }
    }
}
} // namespace backend_fixture

int main() {
    backend_fixture::run();
    try {
        const auto two = run(2), eight = run(8), off = run(8, false);
        require(off.bytes == 0 && off.staging_bytes == 0 && off.exchanges == 0 && off.exposed_wait == 0);
        require(two.bytes == eight.bytes && two.staging_bytes == eight.staging_bytes);
        require(two.exchanges == eight.exchanges && eight.exposed_wait == 16);
        std::cout << "PASS xport staging: two ranks, N=8, tags, reverse completion, both fences, R17, counters\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL xport staging: " << error.what() << '\n';
        return 1;
    }
}
