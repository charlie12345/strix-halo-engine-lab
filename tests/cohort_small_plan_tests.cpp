#include "strix/cohort_small_plan.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
enum class Decline { none, wide, no_rows, padding, alone };
struct Item {
    std::size_t member;
    std::uint32_t rows;
    bool operator==(const Item&) const = default;
};
struct Plan {
    std::vector<std::vector<Item>> groups;
    std::vector<Item> solo;
    std::vector<Decline> why;
    bool operator==(const Plan&) const = default;
};
void require(bool pass) {
    if (!pass) throw std::runtime_error("small cohort regression");
}
// Independent model of the established default first-fit and padding rules.
Plan reference(std::vector<Item> items, std::uint32_t fits_mask) {
    Plan plan{{}, {}, std::vector<Decline>(items.size(), Decline::none)};
    std::stable_sort(items.begin(), items.end(), [](Item a, Item b) { return a.rows > b.rows; });
    for (const auto item : items) {
        if (item.rows == 0 || item.rows >= 8) {
            plan.solo.push_back(item);
            plan.why[item.member] = item.rows ? Decline::wide : Decline::no_rows;
            continue;
        }
        auto destination = plan.groups.end();
        for (auto it = plan.groups.begin(); it != plan.groups.end(); ++it) {
            std::uint32_t total = item.rows;
            for (const auto x : *it) total += x.rows;
            if (total <= 8) { destination = it; break; }
        }
        if (destination == plan.groups.end()) plan.groups.push_back({item});
        else destination->push_back(item);
    }
    for (auto& group : plan.groups) {
        while (group.size() >= 2) {
            std::uint32_t total = 0;
            for (const auto x : group) total += x.rows;
            if (total == 4 || total == 8) break;
            const auto target = total < 4 ? 4U : 8U;
            bool repaired = false;
            for (std::size_t i = group.size(); i > 0 && !repaired; --i) {
                auto& x = group[i - 1];
                for (const auto factor : {2U, 4U}) {
                    const auto grown = x.rows * factor;
                    if (grown <= 8 && total - x.rows + grown == target &&
                        (fits_mask & (1U << x.member))) {
                        x.rows = grown; repaired = true; break;
                    }
                }
            }
            if (!repaired) {
                plan.why[group.back().member] = Decline::padding;
                plan.solo.push_back(group.back()); group.pop_back();
            }
        }
        if (group.size() == 1) {
            plan.why[group.front().member] = Decline::alone;
            plan.solo.push_back(group.front()); group.clear();
        }
    }
    return plan;
}
Plan actual(const std::vector<Item>& items, std::uint32_t fits_mask) {
    Plan plan;
    strix::cohort_default_small_plan<Item, Decline>(items, items.size(),
        [fits_mask](Item x, std::uint32_t) { return (fits_mask & (1U << x.member)) != 0; },
        [&](const auto& groups, const auto& solo, const auto& why) { plan = {groups, solo, why}; });
    return plan;
}
} // namespace

int main() {
    std::size_t checked = 0;
    // All C1..C4 row vectors with total <=8, including irregular padding,
    // zero-row declines, wide solos and every combination of owner fits.
    for (std::size_t count = 1; count <= 4; ++count) {
        std::size_t combinations = 1;
        for (std::size_t i = 0; i < count; ++i) combinations *= 9;
        for (std::size_t code = 0; code < combinations; ++code) {
            auto remainder = code;
            std::vector<Item> items;
            std::uint32_t total = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto rows = static_cast<std::uint32_t>(remainder % 9);
                remainder /= 9; total += rows; items.push_back({i, rows});
            }
            if (total > 8) continue;
            require(strix::cohort_default_small_group(total, true));
            require(!strix::cohort_default_small_group(total, false));
            for (std::uint32_t mask = 0; mask < (1U << count); ++mask) {
                require(actual(items, mask) == reference(items, mask));
                ++checked;
            }
        }
    }
    for (std::uint32_t rows = 9; rows <= 16; ++rows)
        require(!strix::cohort_default_small_group(rows, true));
    // C4 depth-one windows form one eight-row verify with no padding.
    const auto c4 = actual({{0, 2}, {1, 2}, {2, 2}, {3, 2}}, 15);
    require(c4.groups.size() == 1 && c4.groups.front().size() == 4 && c4.solo.empty());
    std::cout << "PASS C1..C4 default/budget-small planner: " << checked << " cases\n";
}
