#include "strix/cohort_step_plan.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void require(bool pass) { if (!pass) throw std::runtime_error("cohort step plan mismatch"); }
void check(std::span<const std::uint32_t> rows, std::size_t old_steps, std::size_t new_steps) {
    const auto old_plan = strix::cohort_step_plan(rows, 8U);
    const auto new_plan = strix::cohort_step_plan(rows, 16U);
    require(old_plan.size() == old_steps && new_plan.size() == new_steps);
    // Row identity is (session, window offset); preserve every complete session
    // window in order, including its padding rows, across both execution plans.
    std::vector<std::pair<std::size_t, std::uint32_t>> reference;
    for (std::size_t session = 0; session < rows.size(); ++session)
        for (std::uint32_t offset = 0; offset < rows[session]; ++offset) reference.emplace_back(session, offset);
    for (const auto* plan : {&old_plan, &new_plan}) {
        std::vector<std::pair<std::size_t, std::uint32_t>> actual;
        const std::uint32_t cap = plan == &old_plan ? 8U : 16U;
        for (const auto& chunk : *plan) {
            std::uint32_t total = 0;
            for (const auto session : chunk) {
                total += rows[session];
                for (std::uint32_t offset = 0; offset < rows[session]; ++offset) actual.emplace_back(session, offset);
            }
            require(total <= cap);
            require(chunk.size() < 2U || total == 4U || total == 8U || total == 16U);
        }
        require(actual == reference);
    }
}
}
int main(int argc, char* argv[]) {
    if (argc > 1) require(strix::cohort_step_rows() == (std::string_view(argv[1]) == "16" ? 16U : 8U));
    check(std::array<std::uint32_t, 8>{2,2,2,2,2,2,2,2}, 2, 1);
    check(std::array<std::uint32_t, 6>{4,4,2,2,2,2}, 2, 1);
    check(std::array<std::uint32_t, 2>{8,8}, 2, 1);
    check(std::array<std::uint32_t, 3>{4,4,4}, 2, 2);
    check(std::array<std::uint32_t, 3>{2,2,2}, 2, 2);
    check(std::array<std::uint32_t, 0>{}, 0, 0);
    check(std::array<std::uint32_t, 1>{8}, 1, 1);
    require(strix::cohort_step_rows() == 8U || strix::cohort_step_rows() == 16U);
    std::cout << "PASS: row order, session windows, padding and step counts\n";
}
