// Exercise the production host scheduler with independently interleaved device
// queues. Consumers must see their own layer even when the side queue runs first.
#include "minimax/engine/mm_fast_i4.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace fi = strix::minimax::engine::fast_i4;
void require(bool value) { if (!value) std::abort(); }
struct Queues {
    struct Node { std::vector<int> deps; unsigned slot, layer; int kind; };
    std::vector<Node> nodes;
    int tail[2] = {-1, -1}, free[2] = {-1, -1}, ready[2] = {-1, -1};
    unsigned conversions = 0, side_conversions = 0, waits = 0;
    int add(unsigned queue, unsigned s, unsigned L, int kind, int dependency = -1) {
        Node n{{}, s, L, kind};
        if (tail[queue] >= 0) n.deps.push_back(tail[queue]);
        if (dependency >= 0) n.deps.push_back(dependency);
        nodes.push_back(n);
        return tail[queue] = static_cast<int>(nodes.size()) - 1;
    }
    void convert(unsigned s, unsigned L, bool side) {
        ++conversions;
        if (side) ++side_conversions;
        const int id = add(side ? 1U : 0U, s, L, 0, side ? free[s] : -1);
        if (side) ready[s] = id;
    }
    void wait(unsigned s) {
        require(ready[s] >= 0);
        ++waits;
        add(0, s, 0, 2, ready[s]);
    }
    void run(unsigned seed) const {
        std::mt19937 random(seed);
        std::vector<bool> done(nodes.size(), false);
        int contents[2] = {-1, -1};
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            std::vector<std::size_t> runnable;
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                if (!done[j] && std::all_of(nodes[j].deps.begin(), nodes[j].deps.end(),
                                          [&](int d) { return done[static_cast<std::size_t>(d)]; })) runnable.push_back(j);
            }
            require(!runnable.empty());
            const auto j = runnable[random() % runnable.size()];
            const auto& n = nodes[j];
            if (n.kind == 0) contents[n.slot] = static_cast<int>(n.layer);
            if (n.kind == 1) require(contents[n.slot] == static_cast<int>(n.layer));
            done[j] = true;
        }
    }
};
void range(fi::Schedule& schedule, Queues& q, unsigned first, unsigned end, unsigned chunks, bool prefetch) {
    const auto before = q.conversions;
    schedule.begin();
    for (unsigned L = first; L < end; ++L) {
        const unsigned s = schedule.acquire(L, end, prefetch,
            [&](unsigned slot, unsigned layer, bool side) { q.convert(slot, layer, side); },
            [&](unsigned slot) { q.wait(slot); });
        require(s == (L & 1U));
        for (unsigned c = 0; c < chunks; ++c) q.add(0, s, L, 1);
        q.free[s] = q.add(0, s, L, 2);  // release after every chunk
    }
    require(q.conversions - before == end - first);
}
int main() {
    require(fi::scope(nullptr) == fi::Scope::all && fi::scope("") == fi::Scope::all);
    require(fi::scope("all") == fi::Scope::all && fi::scope("gate_up") == fi::Scope::gate_up);
    for (const char* bad : {"ALL", "gate", "all ", "0"}) {
        bool refused = false;
        try { (void)fi::scope(bad); } catch (const std::invalid_argument&) { refused = true; }
        require(refused);
    }
    for (bool prefetch : {false, true}) for (unsigned chunks : {1U, 2U, 4U}) {
        Queues q;
        fi::Schedule schedule;
        // Full group, an OFF arm (no acquire), ungrouped tail, paused ranges,
        // parity changes and another prompt: no conversion is cached across units.
        range(schedule, q, 3, 60, chunks, prefetch);
        q.add(0, 0, 0, 2);
        range(schedule, q, 3, 60, 1, prefetch);
        range(schedule, q, 3, 4, chunks, prefetch);
        range(schedule, q, 4, 19, chunks, prefetch);
        range(schedule, q, 19, 60, chunks, prefetch);
        require(prefetch ? q.side_conversions > 0 && q.waits == q.side_conversions : q.side_conversions == 0 && q.waits == 0);
        for (unsigned seed = 0; seed < 32; ++seed) q.run(seed);
    }
    std::cout << "PASS: scope validation, parity slots, 1/2/4 chunks, OFF/resume/repeat, serial/prefetch; 192 queue interleavings\n";
}
