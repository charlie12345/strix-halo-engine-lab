#include "moe_core/grouped_policy.hpp"

using namespace strix::moe_core;
struct HostDecodeOps { static constexpr unsigned marker = 7U; };
using Q4 = GroupedWeightDecode<GroupedWeightFormat::q4_k, HostDecodeOps>;
using Q2 = GroupedWeightDecode<GroupedWeightFormat::q2_k, HostDecodeOps>;
using IQX = GroupedWeightDecode<GroupedWeightFormat::iq2_xxs, HostDecodeOps>;
using IQS = GroupedWeightDecode<GroupedWeightFormat::iq2_s, HostDecodeOps>;
static_assert(!GroupedPrefillCore<Q4>::xor64);
static_assert(GroupedPrefillCore<Q4>::WeightDecode::marker == 7U);
static_assert(Q2::format != Q4::format && IQX::format != IQS::format);
static_assert(GroupedLdsRing<>::pitch == 72U);
using Xor = GroupedLdsRing<GroupedPrefillPolicy<true>>;
static_assert(Xor::pitch == 64U);
static_assert((64U+256U)*Xor::pitch*2U == 40960U);
static_assert((64U+256U)*GroupedLdsRing<>::pitch*2U == 46080U);
template<GroupedWeightFormat Format> constexpr bool trait_case() {
    using Decode = GroupedWeightDecode<Format, HostDecodeOps>;
    using Core = GroupedPrefillCore<Decode>;
    return Core::WeightDecode::format == Format && !Core::xor64;
}
static_assert(trait_case<GroupedWeightFormat::q4_k>());
static_assert(trait_case<GroupedWeightFormat::q5_k>());
static_assert(trait_case<GroupedWeightFormat::q2_k>());
static_assert(trait_case<GroupedWeightFormat::q5_1>());
static_assert(trait_case<GroupedWeightFormat::q8_0>());
static_assert(trait_case<GroupedWeightFormat::q2_0>());
static_assert(trait_case<GroupedWeightFormat::iq2_xxs>());
static_assert(trait_case<GroupedWeightFormat::iq2_s>());
static_assert(trait_case<GroupedWeightFormat::iq3_xxs>());
static_assert(trait_case<GroupedWeightFormat::iq3_s>());
static_assert(trait_case<GroupedWeightFormat::iq4_xs>());
static_assert(trait_case<GroupedWeightFormat::iq4_nl>());

constexpr bool mapping_matches() {
    // Every half, including both fragment halves, occupies the original slot.
    for (unsigned row = 0U; row < 256U; ++row) {
        bool seen[64]{};
        for (unsigned half = 0U; half < 64U; ++half) {
            const unsigned old = row*64U+((half/8U)^(row%8U))*8U+half%8U;
            const unsigned got = Xor::offset(row, half);
            if (got != old || got/64U != row || seen[got%64U]) return false;
            seen[got%64U] = true;
            if (GroupedLdsRing<>::offset(row, half) != row*72U+half) return false;
        }
    }
    return true;
}

static_assert(mapping_matches());
int main() { return mapping_matches() ? 0 : 1; }
