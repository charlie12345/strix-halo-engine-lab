#pragma once
#include <cstdint>

namespace strix {
// Post-projection R1 fixture boundary; ggml small-weight kinds 0/1/30 are
// F32/F16/BF16. Does not allocate, synchronize, or publish session state.
struct HipGdnDecodeFuseFixture {
    float *qkv, *beta, *gate, *core, *matrix;
    const float* z;
    std::uint16_t* conv_state;
    const void *conv, *dt, *a, *norm;
    std::uint32_t small_type = 0, key_heads = 2, value_heads = 4, kernel = 4;
    float qk_epsilon = 1e-6F, rms_epsilon = 1e-6F;
};
void hip_gdn_decode_fuse_fixture(const HipGdnDecodeFuseFixture& fixture, bool fuse,
    std::uint32_t* graph_status, void* stream);
} // namespace strix
