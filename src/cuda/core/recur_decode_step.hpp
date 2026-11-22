#pragma once

#include "core/decode_state_rows.hpp"
#include "recur_decode_plan.hpp"

#include <cstddef>
#include <limits>

namespace strix::cuda::recur {

struct DecodeRowStrides {
    // Element strides, with beta and log_decay using the same stride.
    std::size_t input = Gfx1151Decode::hidden;
    std::size_t qkv = Gfx1151Decode::channels;
    std::size_t z = Gfx1151Decode::value_width;
    std::size_t prepared = Gfx1151Decode::channels;
    std::size_t gate = Gfx1151Decode::value_heads;
    std::size_t out = Gfx1151Decode::value_width;
    std::size_t raw_capture = Gfx1151Decode::channels;
    std::size_t a8_codes = Gfx1151Decode::value_width;
};

struct DecodeStep {
    const exec::StepBlock* descriptor;
    DecodeRow rows;
    DecodeRowStrides strides;
    ::strix::core::decode_state::SlotView history, matrix;
    // Disjoint, contiguous images indexed by descriptor row, including padding.
    // The caller publishes the selected pair by changing the two slot entries.
    // Pending images must be disjoint from all published images for this step.
    std::uint16_t* pending_history;
    void* pending_matrix;
    RecurrentStateType type;
    DecodeA8 a8; // Deltas are [value_width / 32][a8_pitch], column r.
};

[[nodiscard]] inline bool valid_decode_step(unsigned width, const DecodeStep& step) noexcept {
    using L = Gfx1151Decode;
    if (width > 16U || !session::is_step_replay_width(width) ||
        !step.descriptor || !step.history.valid() || !step.matrix.valid() ||
        !step.pending_history || !step.pending_matrix ||
        (step.type != RecurrentStateType::f32 && step.type != RecurrentStateType::f16 &&
         step.type != RecurrentStateType::bf16))
        return false;

    const auto fits = [width](std::size_t stride, std::size_t extent, std::size_t element) {
        const auto limit = std::numeric_limits<std::size_t>::max() / element;
        return stride >= extent && (width == 1U || stride <= (limit - extent) / (width - 1U));
    };
    const auto& r = step.rows;
    const auto& s = step.strides;
    return r.input && r.qkv && r.z && r.prepared && r.beta && r.log_decay && r.out &&
        reinterpret_cast<std::uintptr_t>(r.prepared) % 16U == 0U && s.prepared % 4U == 0U &&
        (r.qkv != r.prepared || s.qkv == s.prepared) &&
        fits(s.input, L::hidden, sizeof(std::uint16_t)) && fits(s.qkv, L::channels, sizeof(float)) &&
        fits(s.z, L::value_width, sizeof(float)) && fits(s.prepared, L::channels, sizeof(float)) &&
        fits(s.gate, L::value_heads, sizeof(float)) && fits(s.out, L::value_width, sizeof(float)) &&
        (!r.raw_capture || fits(s.raw_capture, L::channels, sizeof(float))) &&
        bool(step.a8.codes) == bool(step.a8.deltas) &&
        (!step.a8.codes || fits(s.a8_codes, L::value_width, sizeof(std::int8_t)));
}

struct DecodeSource {
    const state::SlotEntry* history;
    const state::SlotEntry* matrix;
    unsigned parent;
};

#if defined(__CUDACC__)
__host__ __device__
#endif
[[nodiscard]] inline DecodeSource decode_source(const DecodeStep& args, unsigned r) noexcept {
    using namespace ::strix::core::decode_state;
    const auto& step = *args.descriptor;
    if (!exec::step_row_valid(step, r)) return {};
    const auto& row = step.row[r];
    const auto* history = args.history.entry(row.slot);
    const auto* matrix = args.matrix.entry(row.slot);
    if (!accepts(history, row.position) || !accepts(matrix, row.position)) return {};
    const auto parent = row.parent;
    // Each link is in the same slot and advances one position. Checking the
    // root against both frontiers also rejects children of an inadmissible row.
    unsigned root = r;
    while (step.row[root].parent != exec::kStepNoParent) {
        root = parent_row(step, root);
        if (root == exec::kStepNoParent) return {};
    }
    const auto position = step.row[root].position;
    if (history->frontier != position || matrix->frontier != position) return {};
    return {history, matrix, parent};
}

#if defined(__CUDACC__)
__host__ __device__
#endif
[[nodiscard]] inline DecodeRow decode_row(const DecodeStep& step, std::size_t r) noexcept {
    const auto& b = step.rows;
    const auto& s = step.strides;
    return {b.input + r * s.input, b.qkv + r * s.qkv, b.z + r * s.z,
            b.prepared + r * s.prepared, b.beta + r * s.gate, b.log_decay + r * s.gate,
            b.out + r * s.out, b.raw_capture ? b.raw_capture + r * s.raw_capture : nullptr};
}

}  // namespace strix::cuda::recur
