// Ladder rule (strix/engine_profile.hpp): TURBO is never slower than FAST, FAST never slower than EXACT, in any
// regime. Fixtures check the verdicts and the route line; the walk resolves every lossy lever of every registered engine
// at every probe regime of its ladder rows and fails when a regime would run a cell that is not measured faster than
// the cell below it (or a ladder row is malformed).
#include "strix/engine_profile.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "minimax/engine/mm_defaults.hpp"

#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace p = strix::profile;

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr std::string_view kLever = "fixture-w4a4@1";
// FAST cell "a8", TURBO cell "a4": both lossy, both ready.
constexpr std::array<p::Lever, 1> kRows{{
    {kLever, "STRIX_FIXTURE_W4A4", "", {"", "a8", "", "a4"}, "off", p::Kind::bit_changing,
     {true, true, false, true}, {true, true, false, true}, "", "off", "fixture receipts"},
}};

p::Resolution resolve(const p::Engine& engine, p::Name tier) {
    p::Inputs in;
    in.profile = tier;
    in.env = [](const char*) -> const char* { return nullptr; };
    return p::resolve(engine, in);
}

void verdicts() {
    constexpr p::LadderRow fast_pf{kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.20, 0.02, "r/fast-pf"};
    constexpr p::LadderRow turbo_pf{kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 2049, 4096, 1, 1, 1.30, 0.02,
                                    "r/turbo-pf-4k"};
    constexpr p::LadderRow turbo_pf_noise{kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 4097, 8192, 1, 1, 1.003,
                                          0.02, "r/turbo-pf-8k"};
    constexpr p::LadderRow turbo_dec_slow{kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536, 8, 16, 0.95,
                                          0.02, "r/turbo-dec"};
    constexpr p::LadderRow fast_dec_slow{kLever, "a8", p::Name::fast, p::Phase::decode, "dev", 1, 65536, 1, 1, 0.97, 0.01,
                                         "r/fast-dec"};
    constexpr p::LadderRow turbo_dec_b1{kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536, 1, 1, 1.25, 0.02,
                                        "r/turbo-dec-b1"};
    constexpr std::array rows{fast_pf, turbo_pf, turbo_pf_noise, turbo_dec_slow, fast_dec_slow, turbo_dec_b1};
    const p::Engine engine{"fixture", "", kRows, false, "fixture-turbo-v1", {}, rows};
    require(p::ladder_defect(engine).empty(), "fixture ladder is well formed: " + p::ladder_defect(engine));

    // Predicate.
    static_assert(!p::ladder_gain(1.005, 0.0) && p::ladder_gain(1.011, 0.0), "noise floor 1 %");
    static_assert(!p::ladder_gain(1.015, 0.02) && p::ladder_gain(1.021, 0.02), "stated margin");
    static_assert(!p::ladder_gain(0.0, 0.02), "unmeasured is never a gain");
    static_assert(p::tier_below(p::Name::turbo) == p::Name::fast && p::tier_below(p::Name::fast) == p::Name::standard);

    const auto turbo = resolve(engine, p::Name::turbo);
    const auto fast = resolve(engine, p::Name::fast);
    const auto exact = resolve(engine, p::Name::standard);
    const auto at = [](const p::Resolution& r, p::Phase phase, std::uint64_t ctx, std::uint32_t lanes,
                       std::string_view device = "dev") { return p::ladder_value(r, kLever, {phase, ctx, lanes, device}); };

    // Measured gain in band: TURBO runs its cell.
    auto cell = at(turbo, p::Phase::prefill, 4096, 1);
    require(cell.value == "a4" && cell.tier == p::Name::turbo, "turbo applies at 4K prefill (x1.30)");
    require(p::ladder_route(kLever, {p::Phase::prefill, 4096, 1, "dev"}, cell) ==
                "strix-ladder/v1 fixture-w4a4@1 prefill ctx=4096 lanes=1 dev=dev: turbo=a4 applies x1.300 noise 0.020 "
                "(r/turbo-pf-4k) -> turbo=a4",
            "route line of an applied cell: " + p::ladder_route(kLever, {p::Phase::prefill, 4096, 1, "dev"}, cell));
    // Within noise at 8K: TURBO resolves to FAST (FAST measured faster than EXACT there).
    cell = at(turbo, p::Phase::prefill, 8192, 1);
    require(cell.value == "a8" && cell.tier == p::Name::fast, "turbo within noise at 8K resolves to fast");
    require(cell.route.find("turbo=a4 within noise x1.003") != std::string::npos, "route says within noise: " + cell.route);
    // Unmeasured context: no TURBO row past 8K, no FAST row either -> EXACT (the builtin).
    cell = at(turbo, p::Phase::prefill, 16384, 1);
    require(cell.value == "off" && cell.tier == p::Name::standard, "unmeasured 16K prefill resolves to exact");
    require(cell.route.find("turbo=a4 unmeasured regime") != std::string::npos &&
                cell.route.find("fast=a8 unmeasured regime") != std::string::npos,
            "route names both unmeasured rungs: " + cell.route);
    // Unknown context (0) is outside every band.
    require(at(turbo, p::Phase::prefill, 0, 1).value == "off", "unknown prompt length never meets a band");
    // Another device never matches.
    require(at(turbo, p::Phase::prefill, 4096, 1, "other").value == "off", "a row of another device never applies");
    // Slower decode at C8..C16: TURBO resolves to FAST; FAST has only a B=1 decode row (slower) -> EXACT.
    cell = at(turbo, p::Phase::decode, 4096, 8);
    require(cell.value == "off" && cell.route.find("turbo=a4 slower x0.950") != std::string::npos,
            "slower turbo decode resolves down: " + cell.route);
    // B=1 decode: TURBO measured faster than FAST, but FAST is slower than EXACT there, so FAST does not run and the
    // TURBO row (measured over FAST) has no measurement against what runs: EXACT.
    cell = at(turbo, p::Phase::decode, 4096, 1);
    require(cell.value == "off" && cell.tier == p::Name::standard,
            "turbo measured over a fast cell that does not run resolves to exact");
    require(cell.route.find("measured over fast=a8, which does not run here") != std::string::npos,
            "route says why: " + cell.route);
    // FAST and EXACT tiers.
    require(at(fast, p::Phase::prefill, 8192, 1).value == "a8", "fast applies inside its band");
    require(at(fast, p::Phase::decode, 4096, 1).value == "off", "fast slower at B=1 decode resolves to exact");
    require(at(exact, p::Phase::prefill, 4096, 1).value == "off", "exact runs the builtin");

    // An explicit setting is the user's own: it runs as set in every regime.
    p::Inputs in;
    in.profile = p::Name::turbo;
    in.env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_FIXTURE_W4A4" ? "a4" : nullptr;
    };
    const auto opted = p::resolve(engine, in);
    require(p::ladder_value(opted, kLever, {p::Phase::prefill, 16384, 1, "dev"}).value == "a4",
            "explicit setting bypasses the ladder (qualification and timing arms)");

    // An engine without ladder rows: every lossy cell resolves to the builtin; an exact lever keeps its cell.
    const p::Engine bare{"bare", "", kRows, false, "bare-turbo-v1"};
    require(p::ladder_value(resolve(bare, p::Name::turbo), kLever, {p::Phase::prefill, 4096, 1, "dev"}).value == "off",
            "no ladder rows: lossy cells do not apply");
    constexpr std::array<p::Lever, 1> exact_rows{{
        {"sched", "STRIX_FIXTURE_SCHED", "", {"", "2", "", ""}, "1", p::Kind::exact, {true, true, false, true},
         {true, true, false, true}, "", "", "fixture scheduling receipt"},
    }};
    const p::Engine scheduling{"sched", "", exact_rows, false, "sched-turbo-v1"};
    require(p::ladder_value(resolve(scheduling, p::Name::turbo), "sched", {p::Phase::decode, 1, 1, "dev"}).value == "2",
            "exact (scheduling) levers are outside the ladder");
}

void defects() {
    const auto defect = [](p::LadderRow row) {
        const std::array rows{row};
        const p::Engine engine{"fixture", "", kRows, false, "fixture-turbo-v1", {}, rows};
        return p::ladder_defect(engine);
    };
    const p::LadderRow good{kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 4096, 1, 1, 1.3, 0.02, "r"};
    require(defect(good).empty(), "good row");
    auto row = good;
    row.lever = "missing";
    require(!defect(row).empty(), "row naming no lever");
    row = good;
    row.value = "a8";
    require(!defect(row).empty(), "row value that is not the tier's cell");
    row = good;
    row.tier = p::Name::standard;
    require(!defect(row).empty(), "exact column row");
    row = good;
    row.receipt = "";
    require(!defect(row).empty(), "row without receipt");
    row = good;
    row.device = "";
    require(!defect(row).empty(), "row without device");
    row = good;
    row.speedup = 0.0;
    require(!defect(row).empty(), "unmeasured speedup");
    row = good;
    row.noise = 0.001;
    require(!defect(row).empty(), "noise below the floor");
    row = good;
    row.min_context = 0;
    require(!defect(row).empty(), "band covering the unknown context");
    row = good;
    row.min_lanes = 4;
    row.max_lanes = 2;
    require(!defect(row).empty(), "unordered lane band");
}

// Walk: every registered engine, every lossy lever, every tier, every probe regime of its ladder. A cell other than
// the tier-below's cell may run only through an `applies` rung whose row shows a gain and whose tier-below cell runs.
void walk(const p::Engine& engine) {
    const auto where = std::string(engine.tag);
    require(p::ladder_defect(engine).empty(), where + ": " + p::ladder_defect(engine));
    auto probes = p::ladder_probes(engine);
    probes.push_back({p::Phase::prefill, 0, 1, {}});  // no row: every lossy cell must resolve to its builtin
    for (const auto& lever : engine.table) {
        if (lever.kind == p::Kind::exact) continue;
        for (const auto tier : {p::Name::fast, p::Name::turbo}) {
            for (const auto& regime : probes) {
                const auto cell = p::ladder_cell(lever, engine.ladder, tier, false, regime);
                if (cell.value == lever.builtin) continue;
                const auto below = p::ladder_cell(lever, engine.ladder, p::tier_below(cell.tier), false, regime);
                if (cell.value == below.value) continue;
                const auto verdict = p::ladder_judge(engine.ladder, lever.name, cell.value, cell.tier, regime);
                require(verdict.rung == p::Rung::applies && verdict.row &&
                            p::ladder_gain(verdict.row->speedup, verdict.row->noise) &&
                            below.value == p::column_cell(lever, p::tier_below(cell.tier), false),
                        where + "/" + std::string(lever.name) + ": " + p::ladder_route(lever.name, regime, cell));
            }
        }
    }
}

}  // namespace

int main() {
    verdicts();
    defects();
    walk(strix::glm::rank_defaults::product_profile_engine);
    walk(strix::hy4::profile::engine());
    walk(strix::inkling2::kMiniProfile);
    walk(strix::minimax::defaults::profile_engine());
    if (failures) return 1;
    std::puts("profile ladder tests: PASS");
    return 0;
}
