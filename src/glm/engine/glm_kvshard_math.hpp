#pragma once
#include "glm/common/glm_math.hpp"

namespace strix::glm::kvshard {
// The GLM softmax contract uses declared_exp, including for rank rescaling.
// A generic expf-based attention oracle does not check this engine contract.
STRIX_GLM_HD inline float partial_weight(float score, float maximum) {
    return declared_exp(score - maximum);
}

STRIX_GLM_HD inline float merge_component(float m0, float l0, float a0,
                                          float m1, float l1, float a1) {
    // Test emptiness before subtracting maxima: an empty rank has m = -inf.
    if (l0 == 0.0F) return l1 == 0.0F ? 0.0F : a1 / l1;
    if (l1 == 0.0F) return a0 / l0;
    const float maximum = m1 > m0 ? m1 : m0;
    const float w0 = partial_weight(m0, maximum), w1 = partial_weight(m1, maximum);
    const float denominator = l0 * w0 + l1 * w1;
    const float numerator = a0 * w0 + a1 * w1;
    return numerator / denominator;
}
} // namespace strix::glm::kvshard
