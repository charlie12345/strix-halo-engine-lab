#pragma once
// Load-cohort schedules of a staged CTA tile pipeline.
//
// A cohort is a set of raw global loads issued together. Its destination registers are not read
// until the cohort is published (decoded if the format needs it, then stored to the stage's LDS
// buffer). A schedule only orders three kernel-supplied operations around the stage's matrix work:
//   issue(stage, Set<c>{})            raw loads of that stage, no use of their destinations
//   publish(stage, buffer, Set<c>{})  consume the cohorts' registers into LDS buffer `buffer`
//   sync()                           the CTA barrier that hands a published buffer to all waves
// It never changes an operand, an address or the arithmetic: every schedule of one kernel must
// produce the same bits. The kernel names its schedule through its contract (`Contract::Schedule`),
// so the load order is a trait of the instantiation and not code inside one kernel body.
#include <hip/hip_runtime.h>
#include <type_traits>

namespace strix::core::pipeline {
enum Cohort : unsigned { cohort_a = 1U, cohort_b = 2U, cohort_all = cohort_a | cohort_b };
// A cohort set as a type: the kernel selects its loads with `if constexpr`, so a schedule adds no
// branch and no operand to the kernel body.
template<unsigned Mask> struct Set { static constexpr unsigned mask = Mask; };
template<class S> constexpr bool has(unsigned cohort) { return (S::mask & cohort) != 0U; }

// One stage of register lookahead, two LDS stage buffers, one barrier per stage:
//   prologue  issue(0) -> publish(0, 0) -> sync
//   stage s   issue(s+1) -> matrix work on buffer s&1 -> publish(s+1, (s&1)^1) -> sync
// The next stage's loads are in flight across the whole matrix work of the current stage; the
// publish is their first use, so the only memory wait of a steady-state stage sits after its
// matrix work. `Early` names the cohorts issued before the matrix work; the remaining cohorts
// are issued after it, immediately before the publish. cohort_all (everything early) is the
// production schedule; the others expose that cohort's memory latency once per stage and exist
// so a fixture can prove schedule independence and measure what the overlap is worth.
template<unsigned Early = cohort_all> struct CtaStageAhead {
    static constexpr int lookahead = 1;       // stages held in registers ahead of their LDS publication
    static constexpr int lds_stages = 2;      // simultaneously live LDS stage buffers
    static constexpr unsigned early = Early;
    static_assert((Early & ~unsigned(cohort_all)) == 0U);

    template<class Issue, class Publish, class Sync>
    __device__ __forceinline__ static void prologue(int stages, Issue&& issue, Publish&& publish, Sync&& sync) {
        if (stages > 0) {
            issue(0, Set<cohort_all>{});
            publish(0, 0, Set<cohort_all>{});
            sync();
        }
    }
    template<class Issue>
    __device__ __forceinline__ static void before_compute(int stage, int stages, Issue&& issue) {
        if constexpr (Early != 0U) { if (stage + 1 < stages) issue(stage + 1, Set<Early>{}); }
        else { (void)stage; (void)stages; (void)issue; }
    }
    template<class Issue, class Publish, class Sync>
    __device__ __forceinline__ static void after_compute(int stage, int stages, int buffer, Issue&& issue, Publish&& publish, Sync&& sync) {
        if (stage + 1 < stages) {
            if constexpr (Early != unsigned(cohort_all)) issue(stage + 1, Set<unsigned(cohort_all) & ~Early>{});
            publish(stage + 1, buffer ^ 1, Set<cohort_all>{});
        }
        sync();
    }
};

// Contract trait: the contract's `Schedule`, else the stage-ahead schedule with every cohort early.
template<class C, class = void> struct contract_schedule { using type = CtaStageAhead<>; };
template<class C> struct contract_schedule<C, std::void_t<typename C::Schedule>> { using type = typename C::Schedule; };
}
