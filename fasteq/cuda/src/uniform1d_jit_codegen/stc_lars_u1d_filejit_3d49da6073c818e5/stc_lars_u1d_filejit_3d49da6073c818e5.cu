#include <stdint.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPGuard.h>
  using gpuStream_t = hipStream_t;
  #define getCurrentGPUStream at::hip::getCurrentHIPStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpuStream_t = cudaStream_t;
  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

using GPU_Guard = c10::DeviceGuard;

template <typename scalar_t, typename index_t>
__global__ void stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5(
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x0,
    scalar_t* __restrict__ out,
    int B, int X1, int X0, int V, int U)
{
    const int b = (int)blockIdx.x;
    if (b >= B) return;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;
    const index_t x1_base = (index_t)b * (index_t)X1 * (index_t)U;
    const index_t x0_base = (index_t)b * (index_t)X0 * (index_t)U;
    const index_t out_base = (index_t)b * (index_t)V * (index_t)U;
    for (int u_base = 0; u_base < U; u_base += 32) {
        const int u = u_base + lane;
        if (u < U) {
            scalar_t r0;
            scalar_t r1;
            scalar_t r2;
            scalar_t r3;
            scalar_t r4;
            scalar_t r5;
            scalar_t r6;
            scalar_t r7;
            scalar_t r8;
            scalar_t r9;
            scalar_t r10;
            scalar_t r11;
            scalar_t r12;
            scalar_t r13;
            scalar_t r14;
            scalar_t r15;

            scalar_t out_acc_v_0 = scalar_t(0);

            // inst 0: load | global key=(0, 0, (0, 0, 0, 0, -1, 1), "('x0', 5)")
            r0 = x0[x0_base + (index_t)1120 + (index_t)u];
            // inst 1: load | global key=(1, 1, (2, 1, 1, 0, -20, 18), "('x1', 0)")
            r1 = x1[x1_base + (index_t)0 + (index_t)u];
            // inst 2: mul_stc_resident | path#17: out[0] += c * product
            out_acc_v_0 += scalar_t(0.3535533905932738) * (((r1 * r1) * r1) * r0);
            // inst 3: release | last use after path#17
            // inst 4: load | global key=(1, 1, (1, 1, 1, 0, 0, 1), "('x0', 1)")
            r0 = x0[x0_base + (index_t)224 + (index_t)u];
            // inst 5: mul_stc_resident | path#1: out[0] += c * product
            out_acc_v_0 += scalar_t(0.5000000000000001) * ((r1 * r1) * r0);
            // inst 6: release | last use after path#1
            // inst 7: load | global key=(1, 1, (1, 1, 1, 0, 0, 1), "('x0', 0)")
            r0 = x0[x0_base + (index_t)0 + (index_t)u];
            // inst 8: mul_stc_resident | path#0: out[0] += c * product
            out_acc_v_0 += scalar_t(1.0) * (r1 * r0);
            // inst 9: release | last use after path#0
            // inst 10: load | global key=(0, 0, (0, 0, 7, 0, -7, 7), "('x0', 8)")
            r0 = x0[x0_base + (index_t)1792 + (index_t)u];
            // inst 11: load | global key=(1, 0, (0, 1, 2, 0, -14, 10), "('x1', 14)")
            r2 = x1[x1_base + (index_t)3136 + (index_t)u];
            // inst 12: mul_stc_resident | path#31: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r2) * r2) * r0);
            // inst 13: load | global key=(1, 0, (0, 1, 3, 0, -13, 9), "('x1', 9)")
            r3 = x1[x1_base + (index_t)2016 + (index_t)u];
            // inst 14: mul_stc_resident | path#26: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r3) * r3) * r0);
            // inst 15: load | global key=(2, 0, (0, 2, 2, 0, -5, 7), "('x0', 4)")
            r4 = x0[x0_base + (index_t)896 + (index_t)u];
            // inst 16: mul_stc_resident | path#15: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r2 * r2) * r4);
            // inst 17: mul_stc_resident | path#10: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r3 * r3) * r4);
            // inst 18: load | global key=(2, 0, (0, 2, 5, 0, -14, 14), "('x1', 13)")
            r5 = x1[x1_base + (index_t)2912 + (index_t)u];
            // inst 19: mul_stc_resident | path#30: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r5) * r5) * r0);
            // inst 20: mul_stc_resident | path#14: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r5 * r5) * r4);
            // inst 21: load | global key=(2, 0, (0, 2, 6, 0, -13, 14), "('x1', 11)")
            r6 = x1[x1_base + (index_t)2464 + (index_t)u];
            // inst 22: mul_stc_resident | path#28: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r6) * r6) * r0);
            // inst 23: mul_stc_resident | path#12: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r6 * r6) * r4);
            // inst 24: load | global key=(2, 0, (0, 2, 6, 0, -10, 10), "('x1', 10)")
            r7 = x1[x1_base + (index_t)2240 + (index_t)u];
            // inst 25: mul_stc_resident | path#27: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r7) * r7) * r0);
            // inst 26: mul_stc_resident | path#11: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r7 * r7) * r4);
            // inst 27: load | global key=(2, 0, (0, 2, 7, 0, -9, 9), "('x1', 15)")
            r8 = x1[x1_base + (index_t)3360 + (index_t)u];
            // inst 28: mul_stc_resident | path#32: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r8) * r8) * r0);
            // inst 29: mul_stc_resident | path#16: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r8 * r8) * r4);
            // inst 30: load | global key=(2, 1, (4, 2, 7, 0, -10, 10), "('x1', 12)")
            r9 = x1[x1_base + (index_t)2688 + (index_t)u];
            // inst 31: mul_stc_resident | path#29: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313785) * (((r1 * r9) * r9) * r0);
            // inst 32: release | last use after path#29
            // inst 33: mul_stc_resident | path#13: out[0] += c * product
            out_acc_v_0 += scalar_t(0.18898223650461363) * ((r9 * r9) * r4);
            // inst 34: release | last use after path#13
            // inst 35: load | global key=(0, 0, (0, 0, 41, 0, -5, 24), "('x0', 12)")
            r4 = x0[x0_base + (index_t)2688 + (index_t)u];
            // inst 36: load | global key=(5, 0, (0, 5, 20, 0, -12, 15), "('x1', 7)")
            r0 = x1[x1_base + (index_t)1568 + (index_t)u];
            // inst 37: mul_stc_resident | path#85: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1636634176769943) * (((r0 * r7) * r6) * r4);
            // inst 38: mul_stc_resident | path#86: out[0] += c * product
            out_acc_v_0 += scalar_t(0.08451542547285167) * (((r0 * r9) * r5) * r4);
            // inst 39: mul_stc_resident | path#87: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1636634176769943) * (((r0 * r5) * r2) * r4);
            // inst 40: mul_stc_resident | path#84: out[0] += c * product
            out_acc_v_0 += scalar_t(0.21128856368212917) * (((r0 * r3) * r7) * r4);
            // inst 41: mul_stc_resident | path#88: out[0] += c * product
            out_acc_v_0 += scalar_t(0.21128856368212917) * (((r0 * r2) * r8) * r4);
            // inst 42: load | global key=(5, 0, (0, 5, 21, 0, -11, 15), "('x1', 5)")
            r10 = x1[x1_base + (index_t)1120 + (index_t)u];
            // inst 43: mul_stc_resident | path#73: out[0] += c * product
            out_acc_v_0 += scalar_t(0.08451542547285167) * (((r10 * r6) * r9) * r4);
            // inst 44: mul_stc_resident | path#71: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1636634176769943) * (((r10 * r7) * r5) * r4);
            // inst 45: mul_stc_resident | path#74: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.1636634176769943) * (((r10 * r6) * r2) * r4);
            // inst 46: mul_stc_resident | path#70: out[0] += c * product
            out_acc_v_0 += scalar_t(0.21128856368212917) * (((r10 * r3) * r2) * r4);
            // inst 47: mul_stc_resident | path#72: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.21128856368212917) * (((r10 * r7) * r8) * r4);
            // inst 48: load | global key=(5, 0, (0, 5, 21, 0, -9, 17), "('x1', 8)")
            r11 = x1[x1_base + (index_t)1792 + (index_t)u];
            // inst 49: mul_stc_resident | path#89: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.13363062095621223) * (((r11 * r3) * r6) * r4);
            // inst 50: mul_stc_resident | path#93: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.13363062095621223) * (((r11 * r5) * r8) * r4);
            // inst 51: mul_stc_resident | path#91: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.26726124191242445) * (((r11 * r9) * r2) * r4);
            // inst 52: mul_stc_resident | path#90: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.10350983390135315) * (((r11 * r6) * r6) * r4);
            // inst 53: mul_stc_resident | path#92: out[0] += c * product
            out_acc_v_0 += scalar_t(0.10350983390135315) * (((r11 * r5) * r5) * r4);
            // inst 54: load | global key=(5, 0, (0, 5, 17, 0, -9, 18), "('x1', 6)")
            r12 = x1[x1_base + (index_t)1344 + (index_t)u];
            // inst 55: mul_stc_resident | path#79: out[0] += c * product
            out_acc_v_0 += scalar_t(0.08964214570007953) * (((r12 * r6) * r6) * r4);
            // inst 56: mul_stc_resident | path#81: out[0] += c * product
            out_acc_v_0 += scalar_t(0.08964214570007953) * (((r12 * r5) * r5) * r4);
            // inst 57: mul_stc_resident | path#80: out[0] += c * product
            out_acc_v_0 += scalar_t(0.11952286093343939) * (((r12 * r9) * r9) * r4);
            // inst 58: mul_stc_resident | path#78: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.1494035761667992) * (((r12 * r3) * r3) * r4);
            // inst 59: mul_stc_resident | path#82: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.1494035761667992) * (((r12 * r8) * r8) * r4);
            // inst 60: load | global key=(6, 0, (0, 6, 14, 0, -1, 8), "('x0', 11)")
            r13 = x0[x0_base + (index_t)2464 + (index_t)u];
            // inst 61: mul_stc_resident | path#77: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.253546276418555) * (((r12 * r11) * r11) * r13);
            // inst 62: mul_stc_resident | path#68: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1267731382092775) * (((r10 * r10) * r12) * r13);
            // inst 63: mul_stc_resident | path#76: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1267731382092775) * (((r12 * r0) * r0) * r13);
            // inst 64: mul_stc_resident | path#69: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.21957751641342) * (((r10 * r10) * r11) * r13);
            // inst 65: mul_stc_resident | path#83: out[0] += c * product
            out_acc_v_0 += scalar_t(0.21957751641342) * (((r0 * r0) * r11) * r13);
            // inst 66: mul_stc_resident | path#75: out[0] += c * product
            out_acc_v_0 += scalar_t(0.08451542547285167) * (((r12 * r12) * r12) * r13);
            // inst 67: load | global key=(6, 0, (0, 6, 23, 0, -7, 14), "('x1', 4)")
            r14 = x1[x1_base + (index_t)896 + (index_t)u];
            // inst 68: mul_stc_resident | path#63: out[0] += c * product
            out_acc_v_0 += scalar_t(0.43915503282683993) * (((r14 * r10) * r0) * r13);
            // inst 69: mul_stc_resident | path#62: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.253546276418555) * (((r14 * r14) * r12) * r13);
            // inst 70: release | last use after path#62
            // inst 71: mul_stc_resident | path#66: out[0] += c * product
            out_acc_v_0 += scalar_t(0.2070196678027063) * (((r14 * r6) * r5) * r4);
            // inst 72: mul_stc_resident | path#64: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.13363062095621223) * (((r14 * r3) * r5) * r4);
            // inst 73: mul_stc_resident | path#65: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.26726124191242445) * (((r14 * r7) * r9) * r4);
            // inst 74: mul_stc_resident | path#67: out[0] += c * product
            out_acc_v_0 += scalar_t(0.13363062095621223) * (((r14 * r6) * r8) * r4);
            // inst 75: release | last use after path#67
            // inst 76: load | global key=(5, 0, (0, 5, 10, 0, 0, 5), "('x0', 7)")
            r4 = x0[x0_base + (index_t)1568 + (index_t)u];
            // inst 77: mul_stc_resident | path#25: out[0] += c * product
            out_acc_v_0 += scalar_t(0.27386127875258304) * (((r1 * r11) * r11) * r4);
            // inst 78: mul_stc_resident | path#21: out[0] += c * product
            out_acc_v_0 += scalar_t(0.27386127875258304) * (((r1 * r14) * r14) * r4);
            // inst 79: mul_stc_resident | path#23: out[0] += c * product
            out_acc_v_0 += scalar_t(0.27386127875258304) * (((r1 * r12) * r12) * r4);
            // inst 80: mul_stc_resident | path#22: out[0] += c * product
            out_acc_v_0 += scalar_t(0.27386127875258304) * (((r1 * r10) * r10) * r4);
            // inst 81: mul_stc_resident | path#24: out[0] += c * product
            out_acc_v_0 += scalar_t(0.27386127875258304) * (((r1 * r0) * r0) * r4);
            // inst 82: release | last use after path#24
            // inst 83: load | global key=(5, 0, (0, 5, 5, 0, 0, 5), "('x0', 3)")
            r4 = x0[x0_base + (index_t)672 + (index_t)u];
            // inst 84: mul_stc_resident | path#9: out[0] += c * product
            out_acc_v_0 += scalar_t(0.223606797749979) * ((r11 * r11) * r4);
            // inst 85: mul_stc_resident | path#5: out[0] += c * product
            out_acc_v_0 += scalar_t(0.223606797749979) * ((r14 * r14) * r4);
            // inst 86: mul_stc_resident | path#7: out[0] += c * product
            out_acc_v_0 += scalar_t(0.223606797749979) * ((r12 * r12) * r4);
            // inst 87: mul_stc_resident | path#6: out[0] += c * product
            out_acc_v_0 += scalar_t(0.223606797749979) * ((r10 * r10) * r4);
            // inst 88: mul_stc_resident | path#8: out[0] += c * product
            out_acc_v_0 += scalar_t(0.223606797749979) * ((r0 * r0) * r4);
            // inst 89: release | last use after path#8
            // inst 90: load | global key=(0, 0, (0, 0, 42, 0, -3, 21), "('x0', 10)")
            r4 = x0[x0_base + (index_t)2240 + (index_t)u];
            // inst 91: load | global key=(8, 0, (0, 8, 29, 0, -5, 14), "('x1', 3)")
            r13 = x1[x1_base + (index_t)672 + (index_t)u];
            // inst 92: mul_stc_resident | path#60: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.05976143046671969) * (((r13 * r11) * r5) * r4);
            // inst 93: mul_stc_resident | path#55: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.05976143046671969) * (((r13 * r14) * r6) * r4);
            // inst 94: mul_stc_resident | path#57: out[0] += c * product
            out_acc_v_0 += scalar_t(0.2070196678027063) * (((r13 * r12) * r5) * r4);
            // inst 95: mul_stc_resident | path#56: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1889822365046136) * (((r13 * r10) * r7) * r4);
            // inst 96: mul_stc_resident | path#58: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.14638501094227999) * (((r13 * r0) * r9) * r4);
            // inst 97: mul_stc_resident | path#61: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313788) * (((r13 * r11) * r8) * r4);
            // inst 98: mul_stc_resident | path#54: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313788) * (((r13 * r14) * r3) * r4);
            // inst 99: mul_stc_resident | path#59: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1889822365046136) * (((r13 * r0) * r2) * r4);
            // inst 100: load | global key=(8, 1, (4, 8, 30, 0, -4, 14), "('x1', 1)")
            r15 = x1[x1_base + (index_t)224 + (index_t)u];
            // inst 101: mul_stc_resident | path#43: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23145502494313788) * (((r15 * r11) * r3) * r4);
            // inst 102: release | last use after path#43
            // inst 103: mul_stc_resident | path#38: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.23145502494313788) * (((r15 * r14) * r8) * r4);
            // inst 104: release | last use after path#38
            // inst 105: mul_stc_resident | path#41: out[0] += c * product
            out_acc_v_0 += scalar_t(0.2070196678027063) * (((r15 * r12) * r6) * r4);
            // inst 106: mul_stc_resident | path#39: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.14638501094227999) * (((r15 * r10) * r9) * r4);
            // inst 107: mul_stc_resident | path#44: out[0] += c * product
            out_acc_v_0 += scalar_t(0.05976143046671969) * (((r15 * r11) * r6) * r4);
            // inst 108: mul_stc_resident | path#37: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.05976143046671969) * (((r15 * r14) * r5) * r4);
            // inst 109: mul_stc_resident | path#40: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.1889822365046136) * (((r15 * r10) * r2) * r4);
            // inst 110: mul_stc_resident | path#42: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1889822365046136) * (((r15 * r0) * r7) * r4);
            // inst 111: load | global key=(5, 1, (10, 5, 21, 0, -3, 10), "('x1', 2)")
            r8 = x1[x1_base + (index_t)448 + (index_t)u];
            // inst 112: mul_stc_resident | path#49: out[0] += c * product
            out_acc_v_0 += scalar_t(0.253546276418555) * (((r8 * r12) * r9) * r4);
            // inst 113: release | last use after path#49
            // inst 114: mul_stc_resident | path#51: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1889822365046136) * (((r8 * r11) * r2) * r4);
            // inst 115: release | last use after path#51
            // inst 116: mul_stc_resident | path#47: out[0] += c * product
            out_acc_v_0 += scalar_t(0.1889822365046136) * (((r8 * r14) * r7) * r4);
            // inst 117: release | last use after path#47
            // inst 118: mul_stc_resident | path#48: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23904572186687875) * (((r8 * r10) * r6) * r4);
            // inst 119: release | last use after path#48
            // inst 120: mul_stc_resident | path#50: out[0] += c * product
            out_acc_v_0 += scalar_t(0.23904572186687875) * (((r8 * r0) * r5) * r4);
            // inst 121: release | last use after path#50
            // inst 122: release | last use after path#50
            // inst 123: load | global key=(8, 1, (6, 8, 19, 0, 0, 8), "('x0', 9)")
            r4 = x0[x0_base + (index_t)2016 + (index_t)u];
            // inst 124: mul_stc_resident | path#36: out[0] += c * product
            out_acc_v_0 += scalar_t(0.38729833462074176) * (((r15 * r13) * r14) * r4);
            // inst 125: release | last use after path#36
            // inst 126: mul_stc_resident | path#35: out[0] += c * product
            out_acc_v_0 += scalar_t(0.38729833462074176) * (((r15 * r8) * r10) * r4);
            // inst 127: release | last use after path#35
            // inst 128: mul_stc_resident | path#46: out[0] += c * product
            out_acc_v_0 += scalar_t(0.38729833462074176) * (((r8 * r13) * r0) * r4);
            // inst 129: release | last use after path#46
            // inst 130: mul_stc_resident | path#33: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.1118033988749895) * (((r15 * r15) * r12) * r4);
            // inst 131: mul_stc_resident | path#52: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.1118033988749895) * (((r13 * r13) * r12) * r4);
            // inst 132: mul_stc_resident | path#45: out[0] += c * product
            out_acc_v_0 += scalar_t(0.223606797749979) * (((r8 * r8) * r12) * r4);
            // inst 133: release | last use after path#45
            // inst 134: mul_stc_resident | path#34: out[0] += c * product
            out_acc_v_0 += scalar_t(-0.19364916731037085) * (((r15 * r15) * r11) * r4);
            // inst 135: mul_stc_resident | path#53: out[0] += c * product
            out_acc_v_0 += scalar_t(0.19364916731037085) * (((r13 * r13) * r11) * r4);
            // inst 136: release | last use after path#53
            // inst 137: release | last use after path#53
            // inst 138: load | global key=(3, 0, (0, 3, 6, 0, 0, 3), "('x0', 6)")
            r4 = x0[x0_base + (index_t)1344 + (index_t)u];
            // inst 139: mul_stc_resident | path#18: out[0] += c * product
            out_acc_v_0 += scalar_t(0.35355339059327373) * (((r1 * r15) * r15) * r4);
            // inst 140: mul_stc_resident | path#19: out[0] += c * product
            out_acc_v_0 += scalar_t(0.35355339059327373) * (((r1 * r8) * r8) * r4);
            // inst 141: mul_stc_resident | path#20: out[0] += c * product
            out_acc_v_0 += scalar_t(0.35355339059327373) * (((r1 * r13) * r13) * r4);
            // inst 142: release | last use after path#20
            // inst 143: release | last use after path#20
            // inst 144: load | global key=(3, 1, (6, 3, 3, 0, 0, 3), "('x0', 2)")
            r4 = x0[x0_base + (index_t)448 + (index_t)u];
            // inst 145: mul_stc_resident | path#2: out[0] += c * product
            out_acc_v_0 += scalar_t(0.2886751345948129) * ((r15 * r15) * r4);
            // inst 146: release | last use after path#2
            // inst 147: mul_stc_resident | path#3: out[0] += c * product
            out_acc_v_0 += scalar_t(0.2886751345948129) * ((r8 * r8) * r4);
            // inst 148: release | last use after path#3
            // inst 149: mul_stc_resident | path#4: out[0] += c * product
            out_acc_v_0 += scalar_t(0.2886751345948129) * ((r13 * r13) * r4);
            // inst 150: release | last use after path#4
            // inst 151: release | last use after path#4

            // resident output accumulator writeback
            out[out_base + (index_t)0 + (index_t)u] += out_acc_v_0;
        }
    }
}

static inline bool mul_fits_int32(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}
static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32(a, b)) return false;
    return mul_fits_int32(a * b, c);
}

template <typename scalar_t, typename index_t>
void launch_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5_typed(
    const scalar_t* x1, const scalar_t* x0, scalar_t* out,
    int B, int X1, int X0, int V, int U, gpuStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5<scalar_t, index_t><<<grid, block, 0, stream>>>(x1, x0, out, B, X1, X0, V, U);
}

template <typename scalar_t>
void launch_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5(
    const scalar_t* x1, const scalar_t* x0, scalar_t* out,
    int B, int X1, int X0, int V, int U, gpuStream_t stream)
{
    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&
                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&
                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);
    if (use_i32) {
        launch_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5_typed<scalar_t, int32_t>(x1, x0, out, B, X1, X0, V, U, stream);
    } else {
        launch_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5_typed<scalar_t, int64_t>(x1, x0, out, B, X1, X0, V, U, stream);
    }
}

torch::Tensor launcher_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5(torch::Tensor x1, torch::Tensor x0, int64_t V64) {
    TORCH_CHECK(x1.is_cuda() && x0.is_cuda(), "x1/x0 must be CUDA/HIP");
    TORCH_CHECK(x1.is_contiguous() && x0.is_contiguous(), "x1/x0 must be contiguous");
    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3, "x1/x0 must be [B,S,U]");
    TORCH_CHECK(x1.scalar_type() == x0.scalar_type(), "x1/x0 dtype mismatch");
    int B = (int)x1.size(0);
    int X1 = (int)x1.size(1);
    int U = (int)x1.size(2);
    int X0 = (int)x0.size(1);
    int V = (int)V64;
    TORCH_CHECK((int)x0.size(0) == B, "x0 batch mismatch");
    TORCH_CHECK((int)x0.size(2) == U, "x0 U mismatch");
    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    auto out = torch::zeros({B, V, U}, x1.options());
    GPU_Guard device_guard(x1.device());
    gpuStream_t stream = getCurrentGPUStream(x1.device().index());
    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5", [&] {
        launch_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5<scalar_t>((const scalar_t*)x1.data_ptr<scalar_t>(),
            (const scalar_t*)x0.data_ptr<scalar_t>(),
            (scalar_t*)out.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);
    });
    GPU_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5, "stc_lars_u1d_filejit_3d49da6073c818e5_u224_path94_maxlen5 STC forward jit impl");
}