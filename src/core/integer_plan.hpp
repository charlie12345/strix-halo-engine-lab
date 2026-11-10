#pragma once
#if defined(__HIPCC__)
#define STRIX_PLAN_HD __host__ __device__
#else
#define STRIX_PLAN_HD
#endif
namespace strix::core {
enum class IntegerStructure { per_group, grouped_chunk, dense_tall, dense_wide };
struct IntegerPlan {
    static constexpr IntegerStructure structure = IntegerStructure::per_group;
    static constexpr unsigned class_waves = 8, chunk_groups = 32;
    static constexpr unsigned waves_per_eu = 0;
};
template<unsigned Waves = 8> struct GroupedIntegerPlan : IntegerPlan {
    static constexpr IntegerStructure structure = IntegerStructure::grouped_chunk;
    static constexpr unsigned waves_per_eu = Waves;
    static constexpr bool weight_lds = true, partial_blocks = true;
    static constexpr bool float_half_sums = true, lds_only_barriers = true;
    static constexpr bool half_exchange = true, opaque_magic = true;
};
template<unsigned Lean = 0, bool ColFirst = false, unsigned Parts = 1>
struct DenseIntegerPlan : IntegerPlan {
    static constexpr IntegerStructure structure = IntegerStructure::dense_tall;
    static constexpr unsigned lean = Lean, parts = Parts;
    static constexpr bool column_first = ColFirst;
};
template<bool Staged = true> struct WideIntegerPlan : IntegerPlan {
    static constexpr IntegerStructure structure = IntegerStructure::dense_wide;
    static constexpr bool staged = Staged;
};
struct IntegerStageItem { unsigned row, half, group; };
STRIX_PLAN_HD constexpr IntegerStageItem integer_stage_item(unsigned thread, unsigned item) {
    const unsigned linear = thread + 512 * item;
    return {linear % 16, linear / 16 % 2, linear / 32};
}
STRIX_PLAN_HD constexpr unsigned integer_weight_offset(unsigned word, unsigned row_words, unsigned row_bytes) {
    return word / row_words * row_bytes + 4 * (word % row_words);
}
struct IntegerBlock { unsigned batch, column; };
STRIX_PLAN_HD constexpr IntegerBlock integer_block(unsigned x, unsigned y, unsigned X, unsigned Y, unsigned band) {
    if (band > 1) {
        const unsigned linear = x + X * y, full = X / band;
        if (linear < full * band * Y) {
            const unsigned in = linear % (band * Y);
            return {linear / (band * Y) * band + in % band, in / band};
        }
        const unsigned width = X - full * band, in = linear - full * band * Y;
        return {full * band + in % width, in / width};
    }
    return {x, y};
}
STRIX_PLAN_HD constexpr unsigned integer_row_position(unsigned r) {
    return (r & ~15U) | ((r & 1U) << 3U) | ((r & 15U) >> 1U);
}
STRIX_PLAN_HD constexpr bool integer_dense_fuse_parts(unsigned tall, unsigned rows, unsigned row_width,
    unsigned width, unsigned count, bool lean_ok, bool diagnostic, bool staged) {
    return tall == 1 && rows >= 896 && row_width == 8192 && width == 4096 && count != 1024 &&
        lean_ok && !diagnostic && staged;
}
// Production dense shape selector. Returned form describes geometry and load policy.
STRIX_PLAN_HD constexpr unsigned integer_dense_form(unsigned rows, unsigned width, unsigned count,
                                                    unsigned row_width, bool lean_ok) {
    unsigned pick = 0;
    if (rows >= 256) {
        if (width == 5120 && count <= 1536) pick = rows >= 6144 ? 914 : 814;
        else if (width <= 1536 && count >= 4096) pick = 814;
        else if (width == 4096 && count == 1024) pick = rows >= 512 ? 914 : 822;
        else if (row_width == 8192 && width == 4096) pick = 822;
    }
    if (pick && !lean_ok) pick = pick == 914 ? 114 : pick == 822 ? 22 : 14;
    return pick ? pick : rows >= 3072 || (rows >= 2048 && count >= 1024) ? 14
                        : rows >= 256 && count >= 4096 && width >= 4096 ? 12 : 0;
}
}
#undef STRIX_PLAN_HD
