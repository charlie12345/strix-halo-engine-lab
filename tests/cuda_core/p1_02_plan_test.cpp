#include "core/bf16_dense_row_plan.hpp"
#include "core/dense_mma_group_plan.hpp"
#include "core/dense_row_plan.hpp"
#include "core/head_argmax_plan.hpp"
#include "core/residual_mix_plan.hpp"
#include "core/step_rows_plan.hpp"

#include <cstdio>

namespace {

using strix::core::Bf16RowLimits;
using strix::core::Bf16RowPath;
using strix::core::plan_bf16_row_projection;
using enum Bf16RowPath;

constexpr unsigned cuda_grid_rows = 65535U * 8U;
constexpr Bf16RowLimits incumbent_limits{32U, cuda_grid_rows, 32U, 32U, 16U};

struct Case {
    unsigned in, out, rows;
    bool tall;
    Bf16RowPath expected;
};

// The expected paths are the decisions of the incumbent CUDA row planner.
constexpr Case cases[]{
    {32U, 16U, 0U, true, short_rows},
    {32U, 16U, 32U, true, short_rows},
    {32U, 48U, 33U, false, short_rows},
    {0U, 16U, 33U, true, short_rows},
    {32U, 0U, 33U, true, short_rows},
    {31U, 16U, 33U, true, fp32_grid},
    {32U, 15U, 33U, true, fp32_grid},
    {32U, 24U, cuda_grid_rows, true, fp32_grid},
    {32U, 24U, cuda_grid_rows + 1U, true, short_rows},
    {32U, 16U, 33U, true, big_compact},
    {32U, 32U, 33U, true, big_compact},
    {32U, 48U, 33U, true, big},
    {32U, 48U, cuda_grid_rows + 1U, true, big},
};

static_assert(plan_bf16_row_projection(16U, 8U, 17U, true, {16U, 32U, 8U, 16U, 8U}) == big_compact);
static_assert(plan_bf16_row_projection(16U, 8U, 17U, true, {16U, 32U, 8U, 0U, 8U}) == fp32_grid);
static_assert(plan_bf16_row_projection(16U, 8U, 17U, true, {16U, 0U, 8U, 16U, 0U}) == short_rows);

bool bf16_groups() {
    namespace cc = strix::cuda_core;
    namespace rows = strix::core::rows;
    alignas(16) float input[32]{}, output[16]{}, residual[16]{};
    alignas(16) std::uint16_t weights[16 * 32]{};
    const strix::core::Bf16RowProjection projection{weights, 16U, output, true};
    const cc::Bf16DenseRowGroup group{input, 32U, std::span(&projection, 1)};
    if (!cc::detail::valid_bf16_dense_row_group(group, 1, true) ||
        cc::detail::valid_bf16_dense_row_group(group, 1, false) ||
        cc::detail::valid_bf16_dense_row_group(group, 0, true)) return false;

    constexpr rows::ChainId add = 1, round = 2;
    rows::Launch launch{};
    launch.tiles.count = 1;
    launch.tiles.first[1] = 1;
    auto& job = launch.jobs[0];
    job.w.data = weights;
    job.x.data = input;
    job.y.data = output;
    job.in = 32;
    job.out = 16;
    job.epilogue = round;
    auto args = cc::mma_group_args(launch, add, round);
    if (!args.jobs[0].round_bf16 || !cc::valid_bf16_mma3_r1(args)) return false;
    args.tiles.first[1] = 0;
    if (cc::valid_bf16_mma3_r1(args)) return false;
    job.epilogue = add;
    job.aux[0].data = residual;
    args = cc::mma_group_args(launch, add, round);
    return args.jobs[0].residual == residual && !args.jobs[0].round_bf16 && !cc::valid_bf16_mma3_r1(args);
}

bool head_view() {
    namespace head = strix::cuda::core::head_argmax;
    float logits[8]{};
    head::ha::Pick partial{};
    head::View view{.logits = logits, .count = 8, .rows = 1, .partial = &partial};
    if (!head::detail::prepare_split_view(view) || view.stride != 8 || view.chunks != head::ha::default_chunks(8))
        return false;
    view.stride = 7;
    return !head::detail::prepare_split_view(view);
}

bool residual_validation() {
    namespace mix = strix::core::residual_mix;
    constexpr unsigned width = 4;
    float storage[3 * width]{};
    float* const bank = storage;
    float* const output = storage + width;
    const float* const query = storage + 2 * width;
    mix::NormalizeArgs norm{bank, output, 1, width, 1e-5F};
    if (!mix::fits(norm)) return false;
    norm.output_stride = 3;
    if (mix::fits(norm)) return false;
    mix::MixArgs args{.banks = {bank}, .queries = query, .output = {output},
                      .rows = 1, .width = width, .bank_count = 1, .query_count = 1, .normalized = true};
    if (!mix::fits(args)) return false;
    args.output[0] = bank;
    return !mix::fits(args);
}

bool step_plan() {
    namespace cc = strix::cuda::core;
    const auto width = []<unsigned Rows>() { return Rows; };
    for (unsigned rows : {1U, 2U, 4U, 8U, 16U}) {
        if (cc::dispatch_step_rows(rows, 0U, width) != rows) return false;
    }
    if (cc::dispatch_step_rows(32U, 0U, width) != 0U) return false;
    strix::exec::StepBlock step{};
    step.h.rows = 4;
    step.h.valid_mask = 5;
    step.row[0].flags = strix::exec::step_valid;
    step.row[2] = strix::exec::StepRow{};
    step.row[2].position = 17;
    step.row[2].flags = strix::exec::step_valid;
    const cc::StepRows<4> rows{&step};
    return rows.valid(0) && !rows.valid(1) && rows.valid(2) && rows.position(2) == 17 && !rows.valid(4);
}

}  // namespace

int main() {
    for (const auto& c : cases) {
        if (plan_bf16_row_projection(c.in, c.out, c.rows, c.tall, incumbent_limits) != c.expected) {
            std::fprintf(stderr, "P1-02 BF16 row path mismatch: in=%u out=%u rows=%u tall=%d\n",
                         c.in, c.out, c.rows, static_cast<int>(c.tall));
            return 1;
        }
    }
    std::puts("PASS P1-02 BF16 row plan (13 incumbent cases)");
    struct Check { const char* name; bool (*passes)(); };
    const Check checks[] = {
        {"BF16 group", bf16_groups}, {"head view", head_view},
        {"residual validation", residual_validation}, {"step rows", step_plan},
    };
    for (const auto& check : checks) {
        if (!check.passes()) {
            std::fprintf(stderr, "P1-02 %s plan mismatch\n", check.name);
            return 1;
        }
    }
    std::puts("PASS P1-02 shared BF16 group, head, residual, and step plans");
}
