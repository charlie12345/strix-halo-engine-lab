#include "moe_core/decode_schedule.hpp"
using GlmDecode = strix::moe_core::DecodeSchedule<32U, 4U, 128U, 4U>;
using MiniMaxDecode = strix::moe_core::DecodeSchedule<16U, 4U, 384U, 4U>;
static_assert(GlmDecode::kGateSplits * GlmDecode::kSplitChannels == 4096U);
static_assert(GlmDecode::kDownSplits * 256U == 1024U);
static_assert(MiniMaxDecode::kGateSplits * MiniMaxDecode::kSplitChannels == 6144U);
static_assert(MiniMaxDecode::kDownSplits * MiniMaxDecode::kSplitChannels == 1536U);
static_assert(GlmDecode::kAscendingSplitSums && MiniMaxDecode::kAscendingSplitSums);
static_assert(GlmDecode::kHalfMaterializedDownInput && MiniMaxDecode::kHalfMaterializedDownInput);
static_assert(GlmDecode::kWaves == 4U && MiniMaxDecode::kWaves == 4U);
int main() { return 0; }
