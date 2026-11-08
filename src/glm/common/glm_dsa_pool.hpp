#pragma once
#include "glm/common/glm_math.hpp"

namespace strix::glm {
// One R12 channel. The address is physical; APE is relative to the four-token
// pool, so compacting whole pools must not change this arithmetic.
STRIX_GLM_HD inline float dsa_pool_channel(const std::uint16_t* key, const std::uint16_t* gate,
                                         const float* ape, unsigned channel) {
    float logits[kKpool], values[kKpool], exponent[kKpool], probability[kKpool];
    for (unsigned j = 0; j < kKpool; ++j) {
        const unsigned offset = j * kIdxDim + channel;
        logits[j] = bf16_to_float(gate[offset]) + ape[offset];
        values[j] = bf16_to_float(key[offset]);
    }
    float maximum = logits[0];
    for (unsigned j = 1; j < kKpool; ++j) maximum = logits[j] > maximum ? logits[j] : maximum;
    for (unsigned j = 0; j < kKpool; ++j) exponent[j] = declared_exp(logits[j] - maximum);
    const float sum = ((exponent[0] + exponent[1]) + exponent[2]) + exponent[3];
    for (unsigned j = 0; j < kKpool; ++j) probability[j] = exponent[j] / sum;
    return ((probability[0] * values[0] + probability[1] * values[1]) +
            probability[2] * values[2]) + probability[3] * values[3];
}
} // namespace strix::glm
