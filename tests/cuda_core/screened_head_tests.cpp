#include "screened_head_cases.hpp"

#include <cstdio>

namespace {
using namespace b7_fixture;

void check_plan() {
    const auto plan = head::Plan::make(264, 67, 13).value();
    require(plan.packed_bytes() == 8844 && plan.packed_stride() == 132 && plan.partial_count() == 3,
            "screen and partial layout");
    require(!head::Plan::make(0, 32) && !head::Plan::make(7, 32) && !head::Plan::make(8, 0) &&
            !head::Plan::make(8, 2, head::ha::kNone - 1U) && !head::Plan::make(0xfffffff8U, 1),
            "invalid or uncertifiable plans");
    try {
        static_cast<void>(head::build_screen(plan, {}));
        require(false, "short weights were accepted");
    } catch (const std::invalid_argument&) {
    }
    const auto small = head::Plan::make(8, 1).value();
    const std::array weights{bf16(7.0F), bf16(-7.0F), bf16(3.0F), bf16(-2.0F),
                             bf16(0.0F), bf16(0.0F), bf16(1.0F), bf16(-1.0F)};
    const auto screen = head::build_screen(small, weights);
    require(screen.packed == std::vector<std::uint8_t>{0x1fU, 0x6bU, 0x88U, 0x79U}, "nibble order");
    const std::array ranges{head::Interval{5, 6}, head::Interval{4, 7}, head::Interval{1, 4},
                            head::Interval{1, 3}, head::Interval{}};
    const auto cutoff = head::second_lower(ranges.data(), static_cast<unsigned>(ranges.size()));
    require(cutoff == 4 && head::survives(ranges[2], cutoff) && !head::survives(ranges[3], cutoff) &&
            head::survives(ranges[4], cutoff), "cutoff equality and uncertified row");
    std::puts("b7 host plan-and-cutoff PASS");
}

double input_max(const Case& test) {
    double maximum = 0;
    for (const auto x : test.activation) {
        if (!std::isfinite(x)) return head::kInfinity;
        maximum = std::max(maximum, std::abs(static_cast<double>(x)));
    }
    return maximum;
}

head::Interval enclosure(const Case& test, const head::Screen& screen, unsigned row, double maximum) {
    head::Interval dot{0, 0};
    if (std::isfinite(maximum) && std::isfinite(screen.certificates[row].weight_l1)) {
        const auto offset = std::size_t{row} * test.plan.packed_stride();
        for (unsigned col = 0; col < test.plan.width(); ++col) {
            const int q = head::unpack(screen.packed[offset + col / 2U], col);
            const auto product = static_cast<double>(q) * test.activation[col];
            dot.lower = head::HostOutward::add_down(dot.lower, product);
            dot.upper = head::HostOutward::add_up(dot.upper, product);
        }
    }
    return head::certify<head::HostOutward>(screen.certificates[row], maximum, test.plan.relative_error(),
                                           test.plan.underflow_error(), dot);
}

void check_case(const Case& test) {
    const auto& plan = test.plan;
    const auto screen = head::build_screen(plan, test.weights);
    const auto maximum = input_max(test);
    const std::span weights(test.weights);
    std::vector<float> logits(plan.vocab());
    for (unsigned row = 0; row < plan.vocab(); ++row) {
        logits[row] = incumbent_score(weights.subspan(std::size_t{row} * plan.width(), plan.width()), test.activation);
    }

    head::ha::Pick2 result;
    unsigned pruned = 0;
    unsigned certified = 0;
    for (unsigned begin = 0; begin < plan.vocab(); begin += head::kTileRows) {
        std::array<head::Interval, head::kTileRows> ranges;
        const auto count = std::min(plan.vocab() - begin, head::kTileRows);
        for (unsigned local = 0; local < count; ++local) {
            const unsigned row = begin + local;
            const auto range = enclosure(test, screen, row, maximum);
            ranges[local] = range;
            if (std::isfinite(range.upper)) {
                ++certified;
                require(std::isfinite(logits[row]) && range.lower <= logits[row] && logits[row] <= range.upper,
                        "certificate missed incumbent score");
            }
        }
        const auto cutoff = head::second_lower(ranges.data(), count);
        head::ha::Pick2 pick;
        for (unsigned local = 0; local < count; ++local) {
            const unsigned row = begin + local;
            if (!head::survives(ranges[local], cutoff)) ++pruned;
            else if (!std::isfinite(logits[row])) ++pick.first.nonfinite;
            else head::ha::push2(pick, logits[row], plan.first() + row);
        }
        result = head::ha::merge2(result, pick);
    }
    const auto reference = head::ha::reference2(logits, plan.first());
    require(same(head::emit(result), head::emit(reference)) &&
            std::bit_cast<std::uint32_t>(result.value2) == std::bit_cast<std::uint32_t>(reference.value2),
            "full rescore identity");
    if (std::string_view(test.name) == "pruning-ties") {
        require(pruned > 0 && result.first.index == 15 && result.index2 == 25, "actual pruning and ID tie rule");
    }
    if (std::string_view(test.name) == "incumbent-order") {
        require(result.first.index == 49 && result.first.value == 1.0F && result.index2 == 50,
                "lane order changes the winner relative to sequential scoring");
    }
    if (std::string_view(test.name) == "overflow-and-nonfinite") {
        require(certified > 0 && result.first.nonfinite == 4 && result.first.index == 12 && result.index2 == 61,
                "uncertified fallback and cross-tile runner-up");
    }
    if (std::string_view(test.name) == "nonfinite-activation") {
        require(pruned == 0 && certified == 0 && result.first.nonfinite == plan.vocab() &&
                result.first.index == head::ha::kNone && result.index2 == head::ha::kNone, "unbounded activation");
    }
    std::printf("b7 host %s PASS\n", test.name);
}
}  // namespace

int main() {
    try {
        check_plan();
        for (const auto& test : b7_fixture::cases()) check_case(test);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "b7 host FAIL %s\n", e.what());
        return 1;
    }
}
