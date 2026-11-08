#include "glm/prefill/glm_pf_load_admission.hpp"
#include "glm_rank_defaults.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

namespace pf = strix::glm::pf;
namespace mem = strix::glm::engine::mem;
namespace {
void check(bool ok) {
    if (!ok) throw std::runtime_error("GLM prefill workspace CPU check failed");
}
template<class F> bool refused(F&& f) {
    try { f(); }
    catch (const std::runtime_error& e) {
        check(std::string(e.what()).find("glm pf path: refusing to allocate ") == 0);
        return true;
    }
    return false;
}
// Inject an identity physical model so these tests isolate owner accounting.
// The runtime supplies the calibrated AllocModel, including its real granule.
struct PhysicalModel { std::uint64_t physical(std::uint64_t bytes) const { return bytes; } };
struct Admission {
    std::vector<mem::Term> terms;
    mem::Snapshot snapshot;
    mem::Margins margins{0, 0};
    std::uint64_t prefill_install = 0, device_free = 0;
    std::uint64_t post_load_device_bytes = 0, post_load_host_bytes = 0;
    PhysicalModel model;
};
}

int main() {
    namespace rd = strix::glm::rank_defaults;
    check(rd::resolve_prefill_auto(false, false, false, false, true, false, 0, false) == "gemv");
    check(rd::resolve_prefill_auto(true, false, false, false, true, false, 0, false, true) == "gemv");
    check(rd::resolve_prefill_auto(false, false, false, false, false, false, 0, false) == "gemv");
    check(rd::resolve_prefill_auto(false, false, true, false, true, false, 0, false) == "ref");
    check(pf::indexer_workspace_bytes(8192) == 144 * mem::kMiB);
    check(pf::indexer_workspace_bytes(4096) == 72 * mem::kMiB);
    pf::admit_prefill_bytes(100, 20, 8, 128, 128);
    check(refused([] { pf::admit_prefill_bytes(100, 20, 8, 127, 128); }));
    check(refused([] { pf::admit_prefill_bytes(100, 20, 8, 128, 127); }));
    check(refused([] { pf::admit_prefill_bytes(std::numeric_limits<std::uint64_t>::max(), 1, 0, 0, 0); }));

    Admission a;
    a.snapshot.meminfo_ok = true;
    a.post_load_device_bytes = 1000;
    a.post_load_host_bytes = 1100;
    // These are load-only guards, not post-load hipMemGetInfo/MemAvailable.
    a.terms = {{"temporary load staging and reserves", 5000, 5000}};
    a.prefill_install = 5000;
    a.snapshot.gpu = true;
    a.snapshot.gtt_total = 1;
    a.margins = {8000, 8000};
    // Device: retained 1000 + comm 40 + path 200 + workspace 30 + margin 20.
    // Host: retained 1100 + comm 40 + the same path/workspace/margin.
    a.device_free = 1290;
    a.snapshot.mem_available = 1390;
    pf::admit_prefill_load(a, 200, 30, 20, 40);
    --a.device_free;
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));
    ++a.device_free;
    --a.snapshot.mem_available;
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));
    ++a.snapshot.mem_available;
    a.snapshot.meminfo_ok = false;
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));
    a.snapshot.meminfo_ok = true;
    a.post_load_device_bytes = std::numeric_limits<std::uint64_t>::max();
    check(refused([&] { pf::admit_prefill_load(a, 200, 30, 20, 40); }));

    // Receipt-derived, rounded 8192-row fixture, not a device qualification.
    // The post-load oracle is independent of the startup ledger subtraction.
    constexpr std::uint64_t path = 6336 * mem::kMiB, workspace = 144 * mem::kMiB;
    constexpr std::uint64_t margin = 8 * mem::kGiB;
    for (const std::uint64_t capacity : {1090000ULL, 1190000ULL, 1500000ULL, 2000000ULL}) {
        const auto post_device = 306 * mem::kGiB / 1000 + (2027520 - capacity) * 17737 + 43 * mem::kGiB / 10;
        const auto post_host = post_device + 2 * mem::kGiB;
        const bool expected = post_device >= path + workspace + margin && post_host >= path + workspace + margin;
        a.post_load_device_bytes = 65 * mem::kGiB + capacity * 17737;
        a.post_load_host_bytes = a.post_load_device_bytes + 7 * mem::kGiB;
        a.device_free = post_device + a.post_load_device_bytes;
        a.snapshot.mem_available = post_host + a.post_load_host_bytes;
        const bool startup = !refused([&] { pf::admit_prefill_load(a, path, workspace, margin, 0); });
        const bool loaded = !refused([&] { pf::admit_prefill_bytes(path, workspace, margin, post_device, post_host); });
        check(startup == expected && loaded == expected);
        check(expected == (capacity <= 1190000));
        std::cout << "capacity " << capacity << ": startup=" << (startup ? "ADMIT" : "REFUSE")
                  << " post-load=" << (loaded ? "ADMIT" : "REFUSE") << " PASS\n";
    }

    // Check the integer top-k decomposition independently of score arithmetic:
    // retained prefix + next window gives the global set AND ascending tie order.
    std::mt19937 gen(41);
    unsigned cases = 0;
    for (unsigned size : {0U, 1U, 511U, 512U, 513U, 4095U, 4096U, 4097U, 8193U, 16385U}) {
        for (unsigned pattern = 0; pattern < 5; ++pattern) {
            std::vector<std::uint32_t> keys(size);
            for (unsigned i = 0; i < size; ++i)
                keys[i] = pattern == 0 ? 7U : pattern == 1 ? i : pattern == 2 ? size - i : pattern == 3 ? gen() % 5U : gen();
            const auto better = [&](unsigned x, unsigned y) { return keys[x] != keys[y] ? keys[x] > keys[y] : x < y; };
            std::vector<unsigned> all, kept;
            for (unsigned i = 0; i < size; ++i) all.push_back(i);
            auto top = [&](auto& v) {
                std::sort(v.begin(), v.end(), better);
                if (v.size() > pf::kIndexerRetainedPools) v.resize(pf::kIndexerRetainedPools);
                std::sort(v.begin(), v.end());
            };
            top(all);
            for (unsigned begin = 0; begin < size; begin += pf::kIndexerPoolWindow) {
                for (unsigned i = begin; i < std::min(size, begin + pf::kIndexerPoolWindow); ++i) kept.push_back(i);
                top(kept);
            }
            check(kept == all);
            ++cases;
        }
    }
    std::cout << "PASS: workspace/admission boundaries and " << cases << " ordered top-k cases\n";
    return EXIT_SUCCESS;
}
