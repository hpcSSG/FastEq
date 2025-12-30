#ifndef __FCTP_PTX_INST_CUH__
#define __FCTP_PTX_INST_CUH__

#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

// typedef double double2_ __attribute__((ext_vector_type(2)));
// typedef double double4_ __attribute__((ext_vector_type(4)));

// union MMARegUnion32B {
//     double4_ v_32B;
//     double v_8B[4];
// };

// union MMARegUnion16B {
//     double2_ v_16B;
//     double v_8B[2];
// };

#define FETCH_16B(src) (reinterpret_cast<float4*>(&(src))[0])

/* Double presision MMA instruction
 * mma.sync.aligned.shape.row.col.f64.f64.f64.f64 d, a, b, c;
 * .shape = {.m8n8k4, .m16n8k4, .m16n8k8, .m16n8k16};
 * "h" = .u16 reg, "r" = .u32 reg, "l" = .u64 reg, "f" = .f32 reg, "d" = .f64 reg
 */
#define asm_mma_m8n8k4_f64_f64_f64_f64(RD0, RD1, RA0, RB0, RC0, RC1)                            \
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};\n"  \
                 : "=d"(RD0), "=d"(RD1)                                                               \
                 : "d"(RA0), "d"(RB0), "d"(RC0), "d"(RC1))

#define asm_mma_m16n8k4_f64_f64_f64_f64(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RC0, RC1, RC2, RC3)                                  \
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"      \
                 : "=d"(RD0), "=d"(RD1), "=d"(RD2), "=d"(RD3)                                                                   \
                 : "d"(RA0), "d"(RA1), "d"(RB0), "d"(RC0), "d"(RC1), "d"(RC2), "d"(RC3))

#define asm_cp_async_ca(SMEM_ADDR, GMEM_ADDR, _Byte)                    \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"          \
                 :                                                      \
                 : "r"(SMEM_ADDR), "l"(GMEM_ADDR), "n"(_Byte))

#define asm_cp_async_cg(SMEM_ADDR, GMEM_ADDR, _Byte)                    \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"          \
                 :                                                      \
                 : "r"(SMEM_ADDR), "l"(GMEM_ADDR), "n"(_Byte))

#define asm_cp_async_ca_l2_prefetch_64B(SMEM_ADDR, GMEM_ADDR, _Byte)        \
    asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], %2;\n"      \
                 :                                                          \
                 : "r"(SMEM_ADDR), "l"(GMEM_ADDR), "n"(_Byte))

#define asm_cp_async_ca_l2_prefetch_128B(SMEM_ADDR, GMEM_ADDR, _Byte)       \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"     \
                 :                                                          \
                 : "r"(SMEM_ADDR), "l"(GMEM_ADDR), "n"(_Byte))

#define asm_cp_async_ca_l2_prefetch_256B(SMEM_ADDR, GMEM_ADDR, _Byte)       \
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n"     \
                 :                                                          \
                 : "r"(SMEM_ADDR), "l"(GMEM_ADDR), "n"(_Byte))

#define asm_cp_async_commit_group() \
    asm volatile("cp.async.commit_group;\n" ::)

#define asm_cp_async_waitgroup(_N)           \
    asm volatile("cp.async.wait_group %0;\n" \
                 :                           \
                 : "n"(_N))

#define asm_ldmatrix_x1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define asm_ldmatrix_x2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define asm_ldmatrix_x4(R0, R1, R2, R3, addr)                                         \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#endif  // __FCTP_PTX_INST_CUH__