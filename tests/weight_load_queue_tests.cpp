#include "core/weight_load_queue.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#include <malloc.h>
#endif

namespace {
using namespace strix::core;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// A file in memory; `align` > 1 rejects what an unbuffered descriptor would reject.
class MemorySource final : public WeightSource {
public:
    MemorySource(std::size_t bytes, std::uint64_t align, unsigned seed) : data_(bytes), align_(align) {
        std::mt19937_64 rng(seed);
        for (auto& b : data_) b = static_cast<std::uint8_t>(rng());
    }
    std::uint64_t alignment() const override { return align_; }
    std::uint64_t read(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) const override {
        if (fail_at != 0 && offset >= fail_at) throw std::runtime_error("injected read failure");
        if (align_ > 1 && ((offset | bytes | reinterpret_cast<std::uintptr_t>(dst)) & (align_ - 1)) != 0)
            throw std::runtime_error("unaligned unbuffered read");
        if (offset >= data_.size()) return 0;
        const std::uint64_t n = std::min<std::uint64_t>(bytes, data_.size() - offset);
        const std::uint64_t part = short_reads && n > 8192 ? (n / 2) & ~std::uint64_t{4095} : n;  // short, still aligned
        std::memcpy(dst, data_.data() + offset, static_cast<std::size_t>(part));
        return part;
    }
    const std::uint8_t* bytes() const { return data_.data(); }
    std::size_t size() const { return data_.size(); }
    std::uint64_t fail_at = 0;
    bool short_reads = false;

private:
    std::vector<std::uint8_t> data_;
    std::uint64_t align_;
};

// "Device" memory is host memory. A copy reads its staging only when it completes (`delay` after the submit), so a
// staging buffer recycled before its event corrupts the result.
class HostUploader final : public WeightUploader {
public:
    explicit HostUploader(std::chrono::milliseconds delay = {}) : delay_(delay) {}
    std::uint8_t* allocate(std::uint64_t bytes) override {
        void* p = nullptr;
#ifdef _MSC_VER
        p = _aligned_malloc(static_cast<std::size_t>(bytes), 4096);
#else
        if (posix_memalign(&p, 4096, static_cast<std::size_t>(bytes)) != 0) p = nullptr;
#endif
        if (p == nullptr) throw std::bad_alloc();
        const std::uint64_t now = live_bytes += bytes;
        std::uint64_t peak = peak_bytes.load();
        while (now > peak && !peak_bytes.compare_exchange_weak(peak, now)) {
        }
        return static_cast<std::uint8_t*>(p);
    }
    void release(std::uint8_t* staging, std::uint64_t bytes) noexcept override {
        live_bytes -= bytes;
#ifdef _MSC_VER
        _aligned_free(staging);
#else
        std::free(staging);
#endif
    }
    Token submit(const std::uint8_t* staging, const Copy* copies, std::size_t count) override {
        std::lock_guard<std::mutex> lock(m_);
        Pending p;
        p.staging = staging;
        p.copies.assign(copies, copies + count);
        p.due = std::chrono::steady_clock::now() + delay_;
        pending_.emplace(next_, std::move(p));
        return next_++;
    }
    bool done(Token token) override {
        std::lock_guard<std::mutex> lock(m_);
        return finish(token, false);
    }
    void wait(Token token) override {
        std::lock_guard<std::mutex> lock(m_);
        (void)finish(token, true);
    }
    void retire(Token token) noexcept override {
        std::lock_guard<std::mutex> lock(m_);
        if (pending_.count(token) != 0) late_retire = true;  // retired before its copy completed: a queue bug
    }
    void drain() override {
        std::lock_guard<std::mutex> lock(m_);
        ++drains;
        while (!pending_.empty()) (void)finish(pending_.begin()->first, true);
    }
    std::atomic<std::uint64_t> live_bytes{0}, peak_bytes{0};
    std::uint64_t drains = 0;
    bool late_retire = false;

private:
    struct Pending {
        const std::uint8_t* staging = nullptr;
        std::vector<Copy> copies;
        std::chrono::steady_clock::time_point due;
    };
    bool finish(Token token, bool block) {
        const auto it = pending_.find(token);
        if (it == pending_.end()) return true;
        if (std::chrono::steady_clock::now() < it->second.due) {
            if (!block) return false;
            std::this_thread::sleep_until(it->second.due);
        }
        for (const auto& c : it->second.copies)
            std::memcpy(c.device, it->second.staging + c.staging_offset, static_cast<std::size_t>(c.bytes));
        pending_.erase(it);
        return true;
    }
    std::mutex m_;
    std::map<Token, Pending> pending_;
    Token next_ = 1;
    std::chrono::milliseconds delay_;
};

void identical_bytes(std::uint64_t align, bool short_reads, unsigned workers, std::uint64_t budget) {
    MemorySource a(std::size_t{3} << 20, align, 1), b((std::size_t{2} << 20) + 777, align, 2);
    a.short_reads = b.short_reads = short_reads;
    HostUploader up(std::chrono::milliseconds(1));
    std::mt19937_64 rng(align * 31 + workers);
    struct Want {
        std::vector<std::uint8_t> device, expect;
        std::uint64_t n1 = 0;
        WeightLoadQueue::Ticket ticket = 0;
    };
    std::vector<Want> wants(200);
    {
        WeightLoadQueue q(up, {workers, budget, nullptr, true});
        for (auto& w : wants) {
            WeightLoadManifest m;
            const MemorySource& s = rng() % 2 ? a : b;
            const std::uint64_t n1 = 1 + rng() % 200000, o1 = rng() % (s.size() - n1);
            const std::uint64_t n2 = 1 + rng() % 5000, o2 = s.size() - n2;  // the tail of the file: a short last window
            const std::uint64_t at1 = m.read(s, o1, n1), at2 = m.read(s, o2, n2);
            w.n1 = n1;
            w.expect.assign(s.bytes() + o1, s.bytes() + o1 + n1);
            w.expect.insert(w.expect.end(), s.bytes() + o2, s.bytes() + o2 + n2);
            w.device.resize(n1 + n2);
            if (rng() % 3 == 0) {  // a conversion: the first range reversed into scratch
                const std::uint64_t out = m.scratch(n1);
                m.convert([=](std::uint8_t* st) {
                    for (std::uint64_t i = 0; i < n1; ++i) st[out + i] = st[at1 + n1 - 1 - i];
                });
                std::reverse(w.expect.begin(), w.expect.begin() + static_cast<std::ptrdiff_t>(n1));
                m.upload(out, n1, w.device.data());
            } else {
                m.upload(at1, n1, w.device.data());
            }
            m.upload(at2, n2, w.device.data() + n1);
            w.ticket = q.submit(std::move(m));
        }
        q.wait(wants[7].ticket);  // the first consumer of one tensor: ready before publication
        require(wants[7].device == wants[7].expect, "a waited ticket is in device memory");
        q.publish();
        const auto st = q.stats();
        require(st.tickets == wants.size() && st.upload_events == wants.size(), "one ready event per ticket");
        require(st.peak_staging_bytes <= budget && up.peak_bytes <= budget, "resident staging stays inside the budget");
        for (auto& w : wants) {
            require(w.device == w.expect, "loaded bytes are identical");
            const std::uint64_t h = weight_hash64(w.expect.data() + w.n1, w.expect.size() - w.n1, weight_hash64(w.expect.data(), w.n1));
            require(q.hash(w.ticket) == h, "the ticket hash chains its upload ranges");
        }
    }
    require(up.drains >= 1 && !up.late_retire, "staging is retired only after its upload event");
    require(up.live_bytes == 0, "all staging returned");
}

void held_in_order() {
    MemorySource s(std::size_t{4} << 20, 4096, 3);
    HostUploader up;
    const std::uint64_t budget = std::uint64_t{2} << 20;  // two 1 MiB tickets resident under eight workers
    {
        WeightLoadQueue q(up, {8, budget, nullptr, true});
        std::vector<WeightLoadQueue::Ticket> tickets;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> where;
        for (unsigned i = 0; i < 64; ++i) {
            WeightLoadManifest m;
            const std::uint64_t off = (i * 60001U) % (3U << 20), n = 500000;
            const std::uint64_t at = m.read(s, off, n);
            m.hold();
            m.output(at, n);
            where.emplace_back(off, at);
            tickets.push_back(q.submit(std::move(m)));
        }
        for (unsigned i = 0; i < 64; ++i) {
            const std::uint8_t* st = q.take(tickets[i]);
            require(std::memcmp(st + where[i].second, s.bytes() + where[i].first, 500000) == 0, "held bytes are identical");
            require(q.hash(tickets[i]) == weight_hash64(s.bytes() + where[i].first, 500000), "held ticket hash");
            if (i != 63) q.release(tickets[i]);  // the last one is still held when the queue goes away
        }
        q.publish();
        require(q.stats().peak_staging_bytes <= budget, "held tickets stay inside the budget");
        require(throws([&] { q.release(tickets[0]); }), "a double release is rejected");
    }
    require(up.live_bytes == 0, "held staging returned at destruction");
}

void failure_cancels() {
    MemorySource s(std::size_t{1} << 20, 1, 4);
    s.fail_at = 600000;
    HostUploader up;
    std::vector<std::uint8_t> dev(1000);
    WeightLoadQueue q(up, {3, std::uint64_t{8} << 20, nullptr, false});
    std::vector<WeightLoadQueue::Ticket> t;
    for (unsigned i = 0; i < 40; ++i) {
        WeightLoadManifest m;
        m.upload(m.read(s, i == 5 ? 700000 : 1000 * i, 1000), 1000, dev.data());
        t.push_back(q.submit(std::move(m)));
    }
    require(throws([&] { q.wait(t[5]); }), "the failed ticket rethrows at its consumer");
    require(throws([&] { q.publish(); }), "publication rethrows the first failure");
    WeightLoadManifest big;
    (void)big.scratch(std::uint64_t{9} << 20);
    require(throws([&] { (void)q.submit(std::move(big)); }), "a ticket larger than the budget is rejected");
    WeightLoadManifest bad;
    require(throws([&] { bad.upload(0, 1, nullptr); }), "a null destination is rejected");
}
}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        require(weight_hash64("abc", 3) != weight_hash64("abd", 3) && weight_hash64("abc", 3) == weight_hash64("abc", 3), "hash");
        identical_bytes(1, false, 1, std::uint64_t{4} << 20);
        identical_bytes(1, true, 4, std::uint64_t{2} << 20);
        identical_bytes(512, false, 4, std::uint64_t{3} << 20);
        identical_bytes(4096, true, 6, std::uint64_t{1} << 20);
        held_in_order();
        failure_cancels();
    } catch (const std::exception& e) {
        std::cerr << "weight load queue tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "weight load queue tests ok\n";
    return 0;
}
