#pragma once

// Compatibility names for MiniMax decode and existing device consumers.
#include "../../moe_core/exl3_trellis.hpp"

namespace strix::minimax::moe::opt::dev {
using strix::moe_core::Half2;
using strix::moe_core::vector_splits;
using strix::moe_core::materialize;
using strix::moe_core::hadamard128;
using strix::moe_core::half_bits;
using strix::moe_core::transform_block;
using strix::moe_core::transform_block_at;
using strix::moe_core::hash16;
using strix::moe_core::mul1_pair;
using strix::moe_core::LaneGeom;
using strix::moe_core::GlobalU32;
using strix::moe_core::LocalU32;
using strix::moe_core::uni;
using strix::moe_core::uniform;
using strix::moe_core::load_u32;
using strix::moe_core::TileStates;
using strix::moe_core::load_act;
using strix::moe_core::core_tile;
using strix::moe_core::reduce4;
using strix::moe_core::Slot;
using strix::moe_core::load_slot;
using strix::moe_core::compute_slot;
using strix::moe_core::LdsRing;
using strix::moe_core::LdsRingU;
using strix::moe_core::lds_barrier;
using strix::moe_core::stream_column_lds;
using strix::moe_core::stream_column_lds_u;
using strix::moe_core::stream_column;
}  // namespace strix::minimax::moe::opt::dev
