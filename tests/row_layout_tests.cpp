// Shared row layout policy (src/core/row_layout.hpp): the pure stride rule, and the known-rows arenas that take
// every row stride from it. Host only.
#include "../src/core/row_layout.hpp"
#include "strix/execution_arena.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
}  // namespace

int main() {
    using strix::core::RowLayoutPolicy;
    using strix::core::physical_row_stride;
    using strix::core::row_alias_period;
    constexpr RowLayoutPolicy page{4096, 128, 2, 16};
    static_assert(physical_row_stride(page, 20480, 8192) == 20608, "a five-page row moves off the page multiple");
    static_assert(physical_row_stride(page, 20480, 8) == 20480, "small windows keep the tight stride");
    static_assert(physical_row_stride(page, 5120, 8192) == 5120, "period 2 moves exact page multiples only");
    static_assert(physical_row_stride(page, 42240, 8192) == 42240, "a stride that does not alias is unchanged");
    static_assert(physical_row_stride(RowLayoutPolicy{4096, 0, 2, 16}, 20480, 8192) == 20480, "step 0 is the tight layout");
    static_assert(row_alias_period(20480) == 1 && row_alias_period(20608) == 32 && row_alias_period(20544) == 64 &&
                  row_alias_period(5120) == 4 && row_alias_period(42240) == 16, "alias periods");
    constexpr RowLayoutPolicy wide{4096, 64, 64, 16};
    for (std::uint64_t tight = 64; tight <= 3U * 65536U; tight += 64) {
        for (const auto& p : {page, wide, RowLayoutPolicy{4096, 192, 16, 16}, RowLayoutPolicy{4096, 256, 16, 16}}) {
            const auto stride = physical_row_stride(p, tight, 2048);
            if (stride < tight || stride % 64 || stride - tight > p.alias_bytes ||
                row_alias_period(stride, p.alias_bytes) < p.min_period ||
                (row_alias_period(tight, p.alias_bytes) >= p.min_period && stride != tight)) {
                expect(false, "stride keeps 64-byte rows, moves only aliasing strides, and reaches the period");
                tight = UINT64_MAX - 64; break;
            }
        }
    }
    bool rejected = false;
    try { (void)physical_row_stride(RowLayoutPolicy{4096, 100, 2, 16}, 20480, 64); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "a step that breaks 64-byte row alignment is rejected");
    rejected = false;
    try { (void)physical_row_stride(RowLayoutPolicy{4096, 4096, 2, 16}, 20480, 64); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "a step that can never leave the page multiple is rejected");

    // Arena: one five-page buffer and one that does not alias; the resolved process policy decides both.
    strix::ExecutionArenaPlan serial;
    serial.alignment = 256;
    strix::ExecutionArenaAllocation stream, hidden;
    stream.use.buffer.buffer_class = strix::ExecutionBufferClass::expanded_hidden; stream.bytes = 20480;
    hidden.use.buffer.buffer_class = strix::ExecutionBufferClass::hyper_workspace; hidden.offset = 20480; hidden.bytes = 42240;
    serial.allocations = {stream, hidden};
    serial.total_bytes = 20480 + 42240; serial.naive_nonaliased_bytes = serial.total_bytes;
    const auto& policy = strix::core::row_layout_policy();
    for (const std::uint32_t rows : {16U, 64U, 2048U}) {
        const auto arena = strix::build_known_prefill_arena(serial, 256, rows);
        expect(arena.row_stride.size() == 2 && arena.row_view.size() == rows, "prefill arena shape");
        for (std::size_t i = 0; i < arena.physical.allocations.size(); ++i) {
            const auto& physical = arena.physical.allocations[i];
            const auto* original = strix::find_arena_allocation(serial, physical.use);
            const auto expected = original ? physical_row_stride(policy, original->bytes, rows) : 0;
            expect(original && arena.row_stride[i] == expected && physical.bytes == expected * rows,
                "every prefill row stride is the policy's stride");
            expect(!policy.dealias_bytes || arena.row_stride[i] % 4096 != 0, "no prefill row stride is a page multiple");
            for (std::uint32_t r = 0; r < rows; ++r) {
                const auto& view = arena.row_view[r].allocations[i];
                if (view.bytes != original->bytes || view.offset != physical.offset + std::uint64_t{r} * expected) {
                    expect(false, "row views keep the payload and step by the policy stride"); break;
                }
            }
        }
    }
    const auto window = strix::build_known_window_arena(serial, 256, 8);
    for (std::size_t i = 0; i < window.physical.allocations.size(); ++i)
        expect(window.row_stride[i] == strix::find_arena_allocation(serial, window.physical.allocations[i].use)->bytes,
            "closed windows of two, four or eight rows keep the tight stride");
    if (failures) return 1;
    std::cout << "row layout tests passed\n";
    return 0;
}
