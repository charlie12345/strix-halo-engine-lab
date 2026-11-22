#include "core/dense_row_plan.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

enum class Status { ok, failed };

bool check_row_chunks() {
    std::vector<std::pair<unsigned, unsigned>> chunks;
    const auto record = [&](unsigned first, unsigned count) {
        chunks.emplace_back(first, count);
        return Status::ok;
    };
    using strix::core::for_each_dense_chunk;
    if (for_each_dense_chunk<32U>(0U, Status::ok, record) != Status::ok || !chunks.empty()) return false;
    if (for_each_dense_chunk<32U>(32U, Status::ok, record) != Status::ok ||
        chunks != decltype(chunks){{0U, 32U}}) return false;
    chunks.clear();
    if (for_each_dense_chunk<32U>(65U, Status::ok, record) != Status::ok ||
        chunks != decltype(chunks){{0U, 32U}, {32U, 32U}, {64U, 1U}}) return false;
    chunks.clear();
    constexpr auto max_rows = std::numeric_limits<unsigned>::max();
    return for_each_dense_chunk<max_rows - 1U>(max_rows, Status::ok, record) == Status::ok &&
           chunks == decltype(chunks){{0U, max_rows - 1U}, {max_rows - 1U, 1U}};
}

bool check_projection_groups() {
    using namespace strix::core;
    std::array<Bf16RowProjection, 9> projections{};
    for (unsigned i = 0; i < projections.size(); ++i) projections[i].outputs = i + 1U;
    std::vector<unsigned> seen;
    std::vector<std::size_t> counts;
    const auto record = [&](std::span<const Bf16RowProjection> group) {
        counts.push_back(group.size());
        for (const auto& projection : group) seen.push_back(projection.outputs);
        return Status::ok;
    };
    if (for_each_bf16_projection_group<4U>(projections, Status::ok, record) != Status::ok ||
        counts != decltype(counts){4U, 4U, 1U} || seen != decltype(seen){1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U})
        return false;
    unsigned calls = 0;
    const auto fail_second = [&](std::span<const Bf16RowProjection>) {
        return ++calls == 2U ? Status::failed : Status::ok;
    };
    return for_each_bf16_projection_group<4U>(projections, Status::ok, fail_second) == Status::failed && calls == 2U;
}

bool check_cuda_bf16_paths() {
    using strix::core::Bf16RowPath;
    constexpr strix::core::Bf16RowLimits limits{32U, 65535U * 8U, 32U, 32U, 16U};
    const auto plan = [&](unsigned in, unsigned out, unsigned rows, bool tall) {
        return strix::core::plan_bf16_row_projection(in, out, rows, tall, limits);
    };
    constexpr unsigned input = 2304U;
    for (const unsigned rows : {33U, 2048U}) {
        for (const unsigned out : {384U, 1024U, 3072U})
            if (plan(input, out, rows, true) != Bf16RowPath::big) return false;
        for (const unsigned out : {16U, 32U})
            if (plan(input, out, rows, true) != Bf16RowPath::big_compact) return false;
        if (plan(input, 24U, rows, true) != Bf16RowPath::fp32_grid ||
            plan(input + 8U, 32U, rows, true) != Bf16RowPath::fp32_grid)
            return false;
    }
    return plan(input, 24U, limits.grid_rows, true) == Bf16RowPath::fp32_grid &&
           plan(input, 24U, limits.grid_rows + 1U, true) == Bf16RowPath::short_rows &&
           plan(input, 32U, limits.short_rows, true) == Bf16RowPath::short_rows &&
           plan(input, 3072U, 2048U, false) == Bf16RowPath::short_rows;
}

}  // namespace

int main() {
    if (!check_row_chunks() || !check_projection_groups() || !check_cuda_bf16_paths()) {
        std::cerr << "Dense row or projection chunk policy failed\n";
        return 1;
    }
    return 0;
}
