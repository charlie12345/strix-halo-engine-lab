// CUDA preprocessing contract, compiled by the HOST compiler only.
#define __CUDACC__ 1
#define __host__
#define __device__
#include "strix/moe_route_bf16.hpp"
static_assert(strix::moe_route_bf16_bits(0x3f818000U)==0x3f82U);
int main() { return strix::moe_route_bf16_bits(0x7f800001U)==0x7fc0U ? 0 : 1; }
