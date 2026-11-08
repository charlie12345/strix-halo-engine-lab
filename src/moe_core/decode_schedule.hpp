#pragma once
namespace strix::moe_core {
// A schedule describes arithmetic partitions, not a performance qualification.
template <unsigned GateSplits, unsigned DownSplits, unsigned SplitChannels, unsigned Waves>
struct DecodeSchedule {
    static_assert(GateSplits > 0 && DownSplits > 0);
    static_assert(SplitChannels % 128U == 0 && Waves > 0);
    static constexpr unsigned kGateSplits = GateSplits;
    static constexpr unsigned kDownSplits = DownSplits;
    static constexpr unsigned kSplitChannels = SplitChannels;
    static constexpr unsigned kWaves = Waves;
    static constexpr bool kAscendingSplitSums = true;
    static constexpr bool kHalfMaterializedDownInput = true;
};
}  // namespace strix::moe_core
