#pragma once

// Instruction/fragment availability, not tuning or numerical qualification.
// Keep target detection in one place. Unknown targets fail closed.
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define STRIX_HIP_ARCH_RDNA3 1
#else
#define STRIX_HIP_ARCH_RDNA3 0
#endif
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define STRIX_HIP_ARCH_RDNA35 1
#else
#define STRIX_HIP_ARCH_RDNA35 0
#endif

// The packed IU4/IU8 operand ABI used by integer_projection is RDNA 3.5 only.
#define STRIX_HAS_RDNA35_WMMA (STRIX_HIP_ARCH_RDNA35 && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32))
#define STRIX_HAS_WMMA_BF16_W32 ((STRIX_HIP_ARCH_RDNA3 || STRIX_HIP_ARCH_RDNA35) && \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32))
#define STRIX_HAS_DPP_ROW ((STRIX_HIP_ARCH_RDNA3 || STRIX_HIP_ARCH_RDNA35) && \
    __has_builtin(__builtin_amdgcn_update_dpp))
#define STRIX_HAS_SUDOT4 ((STRIX_HIP_ARCH_RDNA3 || STRIX_HIP_ARCH_RDNA35) && \
    __has_builtin(__builtin_amdgcn_sudot4))
