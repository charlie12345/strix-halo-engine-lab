#include "glm/prefill/glm_moe_traits.hpp"

using Policy = strix::glm::pf::MoePolicy;
static_assert(Policy::kSplits == 1);
static_assert(Policy::kRowBlock == 16 && Policy::kGateUpInterleave == 2);
static_assert(Policy::kAscendingRouteOrder);
static_assert(Policy::kG3Pitch == 136 && Policy::kG3Blocks == 8);
static_assert(!Policy::accepts_bits(2) && Policy::accepts_bits(3));
static_assert(Policy::accepts_bits(6) && !Policy::accepts_bits(7));
int main() { return 0; }
