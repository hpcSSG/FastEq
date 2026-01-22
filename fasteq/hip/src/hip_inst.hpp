#ifndef __HIP_INST_HPP__
#define __HIP_INST_HPP__

#include <hip/hip_runtime.h>

#define WARP_SIZE 64  // AMD GPUs use 64-thread wavefronts

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

#define FETCH_16B(src) (reinterpret_cast<float4*>(&(src))[0])

/*
 * HIP does not have direct equivalents for NVIDIA's MMA instructions (Tensor Cores).
 * For AMD GPUs, you would use MFMA (Matrix Fused Multiply-Add) instructions via
 * rocWMMA or inline assembly for MI architectures.
 *
 * The following are placeholder macros. Real implementations would need to use:
 * - rocWMMA library for portable matrix operations
 * - AMD MFMA inline assembly for specific GPU architectures (gfx90a, gfx942)
 *
 * Example MFMA for MI200 (gfx90a):
 *   __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0)
 */

// Placeholder for MMA m8n8k4 - needs MFMA implementation for AMD
#define asm_mma_m8n8k4_f64_f64_f64_f64(RD0, RD1, RA0, RB0, RC0, RC1) \
    do { \
        /* Fallback: simple scalar operations - replace with MFMA for performance */ \
        RD0 = RC0 + RA0 * RB0; \
        RD1 = RC1 + RA0 * RB0; \
    } while(0)

// Placeholder for MMA m16n8k4 - needs MFMA implementation for AMD
#define asm_mma_m16n8k4_f64_f64_f64_f64(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RC0, RC1, RC2, RC3) \
    do { \
        /* Fallback: simple scalar operations - replace with MFMA for performance */ \
        RD0 = RC0 + RA0 * RB0; \
        RD1 = RC1 + RA0 * RB0; \
        RD2 = RC2 + RA1 * RB0; \
        RD3 = RC3 + RA1 * RB0; \
    } while(0)

/*
 * HIP async copy operations
 * AMD GPUs do not have direct equivalents to NVIDIA's cp.async instructions.
 * Use regular memory operations or AMD-specific async copy when available.
 */

// Regular copy fallback - no async copy on AMD
#define asm_cp_async_ca(SMEM_ADDR, GMEM_ADDR, _Byte) \
    do { \
        char* dst = reinterpret_cast<char*>(SMEM_ADDR); \
        const char* src = reinterpret_cast<const char*>(GMEM_ADDR); \
        for (int i = 0; i < _Byte; ++i) dst[i] = src[i]; \
    } while(0)

#define asm_cp_async_cg(SMEM_ADDR, GMEM_ADDR, _Byte) \
    asm_cp_async_ca(SMEM_ADDR, GMEM_ADDR, _Byte)

#define asm_cp_async_ca_l2_prefetch_64B(SMEM_ADDR, GMEM_ADDR, _Byte) \
    asm_cp_async_ca(SMEM_ADDR, GMEM_ADDR, _Byte)

#define asm_cp_async_ca_l2_prefetch_128B(SMEM_ADDR, GMEM_ADDR, _Byte) \
    asm_cp_async_ca(SMEM_ADDR, GMEM_ADDR, _Byte)

#define asm_cp_async_ca_l2_prefetch_256B(SMEM_ADDR, GMEM_ADDR, _Byte) \
    asm_cp_async_ca(SMEM_ADDR, GMEM_ADDR, _Byte)

// No async commit group on AMD - these are no-ops
#define asm_cp_async_commit_group() do {} while(0)
#define asm_cp_async_waitgroup(_N) __syncthreads()

/*
 * ldmatrix instructions - AMD equivalent would use LDS (Local Data Share) operations
 * These are placeholders using regular shared memory loads
 */
#define asm_ldmatrix_x1(R, addr) \
    do { R = *reinterpret_cast<unsigned int*>(addr); } while(0)

#define asm_ldmatrix_x2(R0, R1, addr) \
    do { \
        unsigned int* p = reinterpret_cast<unsigned int*>(addr); \
        R0 = p[0]; R1 = p[1]; \
    } while(0)

#define asm_ldmatrix_x4(R0, R1, R2, R3, addr) \
    do { \
        unsigned int* p = reinterpret_cast<unsigned int*>(addr); \
        R0 = p[0]; R1 = p[1]; R2 = p[2]; R3 = p[3]; \
    } while(0)

#endif  // __HIP_INST_HPP__