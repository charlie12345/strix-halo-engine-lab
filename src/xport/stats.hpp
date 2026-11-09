#pragma once
#include <cstdint>
namespace strix::xport {
struct Stats {
    std::uint64_t exchanges = 0, bytes = 0;
    double producer_wait = 0.0;
    double wire = 0.0;
    double check = 0.0;
    std::uint64_t staging_bytes = 0, reuse_dependencies = 0; // event edges, never elapsed time
};
} // namespace strix::xport
