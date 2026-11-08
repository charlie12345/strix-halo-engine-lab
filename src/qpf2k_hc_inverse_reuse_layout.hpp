#pragma once
// Exact experimental organization only; no numerical profile or identity.
#if defined(__HIPCC__)
#define STRIX_HC_REUSE_HD __host__ __device__
#else
#define STRIX_HC_REUSE_HD
#endif
namespace hc_inverse_reuse {
STRIX_HC_REUSE_HD constexpr unsigned row(unsigned t, unsigned e, unsigned lane) {
    return 16U * t + 2U * e + lane / 16U;
}
STRIX_HC_REUSE_HD constexpr unsigned feature(unsigned first, unsigned c) { return first + 16U * c; }
STRIX_HC_REUSE_HD constexpr unsigned inverse_half(unsigned r, unsigned s) { return 40U * r + 32U + 2U * s; }
STRIX_HC_REUSE_HD constexpr unsigned clamp(unsigned r, unsigned count) { return r < count ? r : count - 1U; }
constexpr bool admitted(bool recompute, bool out16, bool explicit_candidate, bool requested_auto) {
    return recompute && out16 && (explicit_candidate || requested_auto);
}
}
#undef STRIX_HC_REUSE_HD
