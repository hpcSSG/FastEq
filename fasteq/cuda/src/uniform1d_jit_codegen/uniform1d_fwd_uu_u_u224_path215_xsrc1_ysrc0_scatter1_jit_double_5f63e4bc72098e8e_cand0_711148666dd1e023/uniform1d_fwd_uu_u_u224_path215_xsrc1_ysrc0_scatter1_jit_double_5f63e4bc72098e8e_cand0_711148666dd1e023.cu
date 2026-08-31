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
__global__ void uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)
{
    const int e_local = (int)blockIdx.x;
    if (e_local >= B) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    constexpr int U_CONST = 224;
    (void)U_CONST;

    const int e_orig = e_local;
    const int w_row  = (WB == 1 ? 0 : e_orig);

    const int x_row = src_idx[e_orig];
    const int y_row = e_orig;
    const int out_row = dst_idx[e_orig];

    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;
    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;
    const index_t y_base = (index_t)y_row * (index_t)Ky;
    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;

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
            scalar_t r16;
            scalar_t r17;
            scalar_t r18;
            scalar_t r19;
            scalar_t r20;
            scalar_t r21;
            scalar_t r22;
            scalar_t r23;

            scalar_t out_acc_v_1 = scalar_t(0);
            scalar_t out_acc_v_2 = scalar_t(0);
            scalar_t out_acc_v_9 = scalar_t(0);
            scalar_t out_acc_v_10 = scalar_t(0);
            scalar_t out_acc_v_11 = scalar_t(0);
            scalar_t out_acc_v_12 = scalar_t(0);
            scalar_t out_acc_v_13 = scalar_t(0);
            scalar_t out_acc_v_14 = scalar_t(0);
            scalar_t out_acc_v_15 = scalar_t(0);
            scalar_t out_acc_v_16 = scalar_t(0);
            scalar_t out_acc_v_17 = scalar_t(0);
            scalar_t out_acc_v_23 = scalar_t(0);
            scalar_t out_acc_v_24 = scalar_t(0);
            scalar_t out_acc_v_25 = scalar_t(0);
            scalar_t out_acc_v_26 = scalar_t(0);
            scalar_t out_acc_v_27 = scalar_t(0);
            scalar_t out_acc_v_28 = scalar_t(0);
            scalar_t out_acc_v_29 = scalar_t(0);
            scalar_t out_acc_v_30 = scalar_t(0);
            scalar_t out_acc_v_31 = scalar_t(0);
            scalar_t out_acc_v_32 = scalar_t(0);
            scalar_t out_acc_v_38 = scalar_t(0);
            scalar_t out_acc_v_39 = scalar_t(0);
            scalar_t out_acc_v_40 = scalar_t(0);
            scalar_t out_acc_v_41 = scalar_t(0);
            scalar_t out_acc_v_42 = scalar_t(0);
            scalar_t out_acc_v_50 = scalar_t(0);
            scalar_t out_acc_v_51 = scalar_t(0);
            scalar_t out_acc_v_52 = scalar_t(0);
            scalar_t out_acc_v_53 = scalar_t(0);
            scalar_t out_acc_v_54 = scalar_t(0);
            scalar_t out_acc_v_55 = scalar_t(0);
            scalar_t out_acc_v_56 = scalar_t(0);
            scalar_t out_acc_v_57 = scalar_t(0);
            scalar_t out_acc_v_58 = scalar_t(0);
            scalar_t out_acc_v_59 = scalar_t(0);
            scalar_t out_acc_v_60 = scalar_t(0);
            scalar_t out_acc_v_61 = scalar_t(0);
            scalar_t out_acc_v_62 = scalar_t(0);
            scalar_t out_acc_v_63 = scalar_t(0);
            scalar_t out_acc_v_64 = scalar_t(0);
            scalar_t out_acc_v_65 = scalar_t(0);
            scalar_t out_acc_v_66 = scalar_t(0);
            scalar_t out_acc_v_67 = scalar_t(0);
            scalar_t out_acc_v_68 = scalar_t(0);
            scalar_t out_acc_v_69 = scalar_t(0);
            scalar_t out_acc_v_70 = scalar_t(0);

            // direct single-use output writeback enabled for 24 out accumulator(s)

            // inst 0: load | global key=(0, 0, (0, 0, 0, 0, -2, 1), "('w', 0)")
            r0 = w[w_base + (index_t)0 + (index_t)u];
            // inst 1: load | global key=(0, 0, (0, 0, 1, 0, -11, 9), "('y', 0)")
            r1 = y[y_base + (index_t)0];
            // inst 2: load | global key=(1, 1, (2, 1, 2, 0, -18, 16), "('x', 0)")
            r2 = x[x_base + (index_t)0 + (index_t)u];
            // inst 3: fma_u1d_direct | path#0: direct out[0] += x[0] * y[0] * w[0] * 1.0
            atomicAdd(&out[out_base + (index_t)0 + (index_t)u], scalar_t(1.0) * (r0 * r2) * r1);
            // inst 4: release | last use after path#0
            // inst 5: load | global key=(0, 0, (0, 0, 7, 0, -7, 7), "('w', 13)")
            r0 = w[w_base + (index_t)2912 + (index_t)u];
            // inst 6: load | global key=(1, 0, (0, 1, 2, 0, -9, 12), "('y', 14)")
            r3 = y[y_base + (index_t)14];
            // inst 7: fma_u1d_direct | path#130: direct out[48] += x[0] * y[14] * w[13] * 0.9999999999999998
            atomicAdd(&out[out_base + (index_t)10752 + (index_t)u], scalar_t(0.9999999999999998) * (r0 * r2) * r3);
            // inst 8: load | global key=(1, 0, (0, 1, 2, 0, -9, 12), "('y', 10)")
            r4 = y[y_base + (index_t)10];
            // inst 9: fma_u1d_direct | path#126: direct out[44] += x[0] * y[10] * w[13] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)9856 + (index_t)u], scalar_t(1.0000000000000004) * (r0 * r2) * r4);
            // inst 10: load | global key=(1, 0, (0, 1, 2, 0, -10, 10), "('y', 9)")
            r5 = y[y_base + (index_t)9];
            // inst 11: fma_u1d_direct | path#125: direct out[43] += x[0] * y[9] * w[13] * 1.0000000000000007
            atomicAdd(&out[out_base + (index_t)9632 + (index_t)u], scalar_t(1.0000000000000007) * (r0 * r2) * r5);
            // inst 12: load | global key=(1, 0, (0, 1, 2, 0, -10, 10), "('y', 15)")
            r6 = y[y_base + (index_t)15];
            // inst 13: fma_u1d_direct | path#131: direct out[49] += x[0] * y[15] * w[13] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)10976 + (index_t)u], scalar_t(1.0000000000000004) * (r0 * r2) * r6);
            // inst 14: load | global key=(1, 0, (0, 1, 2, 0, -11, 17), "('y', 13)")
            r7 = y[y_base + (index_t)13];
            // inst 15: fma_u1d_direct | path#129: direct out[47] += x[0] * y[13] * w[13] * 0.9999999999999996
            atomicAdd(&out[out_base + (index_t)10528 + (index_t)u], scalar_t(0.9999999999999996) * (r0 * r2) * r7);
            // inst 16: load | global key=(1, 0, (0, 1, 2, 0, -11, 17), "('y', 11)")
            r8 = y[y_base + (index_t)11];
            // inst 17: fma_u1d_direct | path#127: direct out[45] += x[0] * y[11] * w[13] * 0.9999999999999991
            atomicAdd(&out[out_base + (index_t)10080 + (index_t)u], scalar_t(0.9999999999999991) * (r0 * r2) * r8);
            // inst 18: load | global key=(1, 1, (2, 1, 2, 0, -11, 12), "('y', 12)")
            r9 = y[y_base + (index_t)12];
            // inst 19: fma_u1d_direct | path#128: direct out[46] += x[0] * y[12] * w[13] * 0.9999999999999996
            atomicAdd(&out[out_base + (index_t)10304 + (index_t)u], scalar_t(0.9999999999999996) * (r0 * r2) * r9);
            // inst 20: release | last use after path#128
            // inst 21: load | global key=(0, 0, (0, 0, 41, 0, -5, 41), "('w', 16)")
            r0 = w[w_base + (index_t)3584 + (index_t)u];
            // inst 22: load | global key=(10, 0, (0, 10, 25, 0, -14, 28), "('x', 7)")
            r10 = x[x_base + (index_t)1568 + (index_t)u];
            // inst 23: fma_u1d_resident | path#183: out[65] += x[7] * y[11] * w[16] * 0.35355339059327395
            out_acc_v_65 += scalar_t(0.35355339059327395) * (r0 * r10) * r8;
            // inst 24: fma_u1d_resident | path#195: out[67] += x[7] * y[13] * w[16] * 0.18257418583505516
            out_acc_v_67 += scalar_t(0.18257418583505516) * (r0 * r10) * r7;
            // inst 25: fma_u1d_resident | path#207: out[69] += x[7] * y[13] * w[16] * 0.3535533905932733
            out_acc_v_69 += scalar_t(0.3535533905932733) * (r0 * r10) * r7;
            // inst 26: fma_u1d_resident | path#177: out[64] += x[7] * y[10] * w[16] * 0.456435464587639
            out_acc_v_64 += scalar_t(0.456435464587639) * (r0 * r10) * r4;
            // inst 27: fma_u1d_resident | path#201: out[68] += x[7] * y[12] * w[16] * 0.1825741858350552
            out_acc_v_68 += scalar_t(0.1825741858350552) * (r0 * r10) * r9;
            // inst 28: fma_u1d_resident | path#202: out[68] += x[7] * y[14] * w[16] * 0.35355339059327356
            out_acc_v_68 += scalar_t(0.35355339059327356) * (r0 * r10) * r3;
            // inst 29: fma_u1d_resident | path#189: out[66] += x[7] * y[10] * w[16] * 0.35355339059327323
            out_acc_v_66 += scalar_t(0.35355339059327323) * (r0 * r10) * r4;
            // inst 30: fma_u1d_resident | path#213: out[70] += x[7] * y[14] * w[16] * 0.45643546458763906
            out_acc_v_70 += scalar_t(0.45643546458763906) * (r0 * r10) * r3;
            // inst 31: fma_u1d_resident | path#182: out[65] += x[7] * y[9] * w[16] * 0.4564354645876384
            out_acc_v_65 += scalar_t(0.4564354645876384) * (r0 * r10) * r5;
            // inst 32: fma_u1d_resident | path#208: out[69] += x[7] * y[15] * w[16] * 0.45643546458763845
            out_acc_v_69 += scalar_t(0.45643546458763845) * (r0 * r10) * r6;
            // inst 33: load | global key=(10, 0, (0, 10, 25, 0, -14, 28), "('x', 5)")
            r11 = x[x_base + (index_t)1120 + (index_t)u];
            // inst 34: fma_u1d_resident | path#193: out[67] += x[5] * y[11] * w[16] * 0.18257418583505508
            out_acc_v_67 += scalar_t(0.18257418583505508) * (r0 * r11) * r8;
            // inst 35: fma_u1d_resident | path#180: out[65] += x[5] * y[13] * w[16] * 0.3535533905932737
            out_acc_v_65 += scalar_t(0.3535533905932737) * (r0 * r11) * r7;
            // inst 36: fma_u1d_resident | path#206: out[69] += x[5] * y[11] * w[16] * -0.35355339059327345
            out_acc_v_69 += scalar_t(-0.35355339059327345) * (r0 * r11) * r8;
            // inst 37: fma_u1d_resident | path#186: out[66] += x[5] * y[12] * w[16] * 0.18257418583505508
            out_acc_v_66 += scalar_t(0.18257418583505508) * (r0 * r11) * r9;
            // inst 38: fma_u1d_resident | path#175: out[64] += x[5] * y[14] * w[16] * 0.4564354645876388
            out_acc_v_64 += scalar_t(0.4564354645876388) * (r0 * r11) * r3;
            // inst 39: fma_u1d_resident | path#199: out[68] += x[5] * y[10] * w[16] * 0.35355339059327423
            out_acc_v_68 += scalar_t(0.35355339059327423) * (r0 * r11) * r4;
            // inst 40: fma_u1d_resident | path#181: out[65] += x[5] * y[15] * w[16] * -0.4564354645876381
            out_acc_v_65 += scalar_t(-0.4564354645876381) * (r0 * r11) * r6;
            // inst 41: fma_u1d_resident | path#187: out[66] += x[5] * y[14] * w[16] * -0.35355339059327345
            out_acc_v_66 += scalar_t(-0.35355339059327345) * (r0 * r11) * r3;
            // inst 42: fma_u1d_resident | path#205: out[69] += x[5] * y[9] * w[16] * 0.4564354645876381
            out_acc_v_69 += scalar_t(0.4564354645876381) * (r0 * r11) * r5;
            // inst 43: fma_u1d_resident | path#211: out[70] += x[5] * y[10] * w[16] * -0.4564354645876387
            out_acc_v_70 += scalar_t(-0.4564354645876387) * (r0 * r11) * r4;
            // inst 44: load | global key=(8, 0, (0, 8, 29, 0, -3, 21), "('w', 7)")
            r12 = w[w_base + (index_t)1568 + (index_t)u];
            // inst 45: fma_u1d_resident | path#46: out[16] += x[5] * y[11] * w[7] * 0.4780914437337572
            out_acc_v_16 += scalar_t(0.4780914437337572) * (r12 * r11) * r8;
            // inst 46: fma_u1d_resident | path#48: out[16] += x[7] * y[13] * w[7] * 0.4780914437337571
            out_acc_v_16 += scalar_t(0.4780914437337571) * (r12 * r10) * r7;
            // inst 47: fma_u1d_resident | path#39: out[15] += x[5] * y[12] * w[7] * -0.29277002188455964
            out_acc_v_15 += scalar_t(-0.29277002188455964) * (r12 * r11) * r9;
            // inst 48: fma_u1d_resident | path#54: out[17] += x[7] * y[12] * w[7] * -0.2927700218845598
            out_acc_v_17 += scalar_t(-0.2927700218845598) * (r12 * r10) * r9;
            // inst 49: fma_u1d_resident | path#40: out[15] += x[5] * y[14] * w[7] * -0.3779644730092273
            out_acc_v_15 += scalar_t(-0.3779644730092273) * (r12 * r11) * r3;
            // inst 50: fma_u1d_resident | path#42: out[15] += x[7] * y[10] * w[7] * 0.37796447300922714
            out_acc_v_15 += scalar_t(0.37796447300922714) * (r12 * r10) * r4;
            // inst 51: fma_u1d_resident | path#52: out[17] += x[5] * y[10] * w[7] * 0.37796447300922736
            out_acc_v_17 += scalar_t(0.37796447300922736) * (r12 * r11) * r4;
            // inst 52: fma_u1d_resident | path#55: out[17] += x[7] * y[14] * w[7] * 0.37796447300922753
            out_acc_v_17 += scalar_t(0.37796447300922753) * (r12 * r10) * r3;
            // inst 53: load | global key=(13, 0, (0, 13, 27, 0, -12, 26), "('x', 8)")
            r13 = x[x_base + (index_t)1792 + (index_t)u];
            // inst 54: fma_u1d_resident | path#178: out[64] += x[8] * y[11] * w[16] * -0.2886751345948129
            out_acc_v_64 += scalar_t(-0.2886751345948129) * (r0 * r13) * r8;
            // inst 55: fma_u1d_resident | path#203: out[68] += x[8] * y[13] * w[16] * 0.4472135954999584
            out_acc_v_68 += scalar_t(0.4472135954999584) * (r0 * r13) * r7;
            // inst 56: fma_u1d_resident | path#191: out[66] += x[8] * y[11] * w[16] * -0.4472135954999582
            out_acc_v_66 += scalar_t(-0.4472135954999582) * (r0 * r13) * r8;
            // inst 57: fma_u1d_resident | path#214: out[70] += x[8] * y[13] * w[16] * -0.28867513459481314
            out_acc_v_70 += scalar_t(-0.28867513459481314) * (r0 * r13) * r7;
            // inst 58: fma_u1d_resident | path#190: out[66] += x[8] * y[9] * w[16] * -0.2886751345948133
            out_acc_v_66 += scalar_t(-0.2886751345948133) * (r0 * r13) * r5;
            // inst 59: fma_u1d_resident | path#44: out[15] += x[8] * y[11] * w[7] * 0.11952286093343914
            out_acc_v_15 += scalar_t(0.11952286093343914) * (r12 * r13) * r8;
            // inst 60: fma_u1d_resident | path#204: out[68] += x[8] * y[15] * w[16] * -0.2886751345948126
            out_acc_v_68 += scalar_t(-0.2886751345948126) * (r0 * r13) * r6;
            // inst 61: fma_u1d_resident | path#56: out[17] += x[8] * y[13] * w[7] * -0.11952286093343928
            out_acc_v_17 += scalar_t(-0.11952286093343928) * (r12 * r13) * r7;
            // inst 62: fma_u1d_resident | path#209: out[69] += x[8] * y[12] * w[16] * -0.5773502691896265
            out_acc_v_69 += scalar_t(-0.5773502691896265) * (r0 * r13) * r9;
            // inst 63: fma_u1d_resident | path#196: out[67] += x[8] * y[14] * w[16] * -0.5773502691896264
            out_acc_v_67 += scalar_t(-0.5773502691896264) * (r0 * r13) * r3;
            // inst 64: fma_u1d_resident | path#43: out[15] += x[8] * y[9] * w[7] * 0.4629100498862758
            out_acc_v_15 += scalar_t(0.4629100498862758) * (r12 * r13) * r5;
            // inst 65: fma_u1d_resident | path#57: out[17] += x[8] * y[15] * w[7] * 0.4629100498862764
            out_acc_v_17 += scalar_t(0.4629100498862764) * (r12 * r13) * r6;
            // inst 66: fma_u1d_resident | path#49: out[16] += x[8] * y[14] * w[7] * 0.3779644730092272
            out_acc_v_16 += scalar_t(0.3779644730092272) * (r12 * r13) * r3;
            // inst 67: load | global key=(13, 0, (0, 13, 27, 0, -12, 26), "('x', 4)")
            r14 = x[x_base + (index_t)896 + (index_t)u];
            // inst 68: fma_u1d_resident | path#174: out[64] += x[4] * y[13] * w[16] * -0.288675134594813
            out_acc_v_64 += scalar_t(-0.288675134594813) * (r0 * r14) * r7;
            // inst 69: fma_u1d_resident | path#198: out[68] += x[4] * y[11] * w[16] * 0.44721359549995815
            out_acc_v_68 += scalar_t(0.44721359549995815) * (r0 * r14) * r8;
            // inst 70: fma_u1d_resident | path#184: out[66] += x[4] * y[13] * w[16] * 0.44721359549995765
            out_acc_v_66 += scalar_t(0.44721359549995765) * (r0 * r14) * r7;
            // inst 71: fma_u1d_resident | path#210: out[70] += x[4] * y[11] * w[16] * 0.2886751345948129
            out_acc_v_70 += scalar_t(0.2886751345948129) * (r0 * r14) * r8;
            // inst 72: fma_u1d_resident | path#37: out[15] += x[4] * y[13] * w[7] * -0.11952286093343926
            out_acc_v_15 += scalar_t(-0.11952286093343926) * (r12 * r14) * r7;
            // inst 73: fma_u1d_resident | path#179: out[65] += x[4] * y[12] * w[16] * -0.5773502691896258
            out_acc_v_65 += scalar_t(-0.5773502691896258) * (r0 * r14) * r9;
            // inst 74: fma_u1d_resident | path#51: out[17] += x[4] * y[11] * w[7] * -0.11952286093343928
            out_acc_v_17 += scalar_t(-0.11952286093343928) * (r12 * r14) * r8;
            // inst 75: fma_u1d_resident | path#185: out[66] += x[4] * y[15] * w[16] * 0.2886751345948125
            out_acc_v_66 += scalar_t(0.2886751345948125) * (r0 * r14) * r6;
            // inst 76: fma_u1d_resident | path#192: out[67] += x[4] * y[10] * w[16] * -0.5773502691896257
            out_acc_v_67 += scalar_t(-0.5773502691896257) * (r0 * r14) * r4;
            // inst 77: fma_u1d_resident | path#50: out[17] += x[4] * y[9] * w[7] * 0.46291004988627626
            out_acc_v_17 += scalar_t(0.46291004988627626) * (r12 * r14) * r5;
            // inst 78: fma_u1d_resident | path#197: out[68] += x[4] * y[9] * w[16] * -0.2886751345948127
            out_acc_v_68 += scalar_t(-0.2886751345948127) * (r0 * r14) * r5;
            // inst 79: fma_u1d_resident | path#38: out[15] += x[4] * y[15] * w[7] * -0.4629100498862759
            out_acc_v_15 += scalar_t(-0.4629100498862759) * (r12 * r14) * r6;
            // inst 80: fma_u1d_resident | path#45: out[16] += x[4] * y[10] * w[7] * 0.37796447300922714
            out_acc_v_16 += scalar_t(0.37796447300922714) * (r12 * r14) * r4;
            // inst 81: load | global key=(8, 0, (0, 8, 17, 0, -13, 21), "('x', 6)")
            r15 = x[x_base + (index_t)1344 + (index_t)u];
            // inst 82: fma_u1d_resident | path#188: out[66] += x[6] * y[11] * w[16] * 0.3872983346207423
            out_acc_v_66 += scalar_t(0.3872983346207423) * (r0 * r15) * r8;
            // inst 83: fma_u1d_resident | path#200: out[68] += x[6] * y[13] * w[16] * 0.38729833462074215
            out_acc_v_68 += scalar_t(0.38729833462074215) * (r0 * r15) * r7;
            // inst 84: fma_u1d_resident | path#41: out[15] += x[6] * y[11] * w[7] * 0.41403933560541256
            out_acc_v_15 += scalar_t(0.41403933560541256) * (r12 * r15) * r8;
            // inst 85: fma_u1d_resident | path#194: out[67] += x[6] * y[12] * w[16] * 0.5163977794943223
            out_acc_v_67 += scalar_t(0.5163977794943223) * (r0 * r15) * r9;
            // inst 86: fma_u1d_resident | path#53: out[17] += x[6] * y[13] * w[7] * 0.41403933560541256
            out_acc_v_17 += scalar_t(0.41403933560541256) * (r12 * r15) * r7;
            // inst 87: fma_u1d_resident | path#47: out[16] += x[6] * y[12] * w[7] * 0.5070925528371097
            out_acc_v_16 += scalar_t(0.5070925528371097) * (r12 * r15) * r9;
            // inst 88: release | last use after path#47
            // inst 89: fma_u1d_resident | path#176: out[64] += x[6] * y[9] * w[16] * -0.645497224367902
            out_acc_v_64 += scalar_t(-0.645497224367902) * (r0 * r15) * r5;
            // inst 90: fma_u1d_resident | path#212: out[70] += x[6] * y[15] * w[16] * -0.6454972243679024
            out_acc_v_70 += scalar_t(-0.6454972243679024) * (r0 * r15) * r6;
            // inst 91: release | last use after path#212
            // inst 92: load | global key=(5, 0, (0, 5, 10, 0, 0, 5), "('w', 11)")
            r0 = w[w_base + (index_t)2464 + (index_t)u];
            // inst 93: fma_u1d_direct | path#96: direct out[34] += x[5] * y[0] * w[11] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)7616 + (index_t)u], scalar_t(1.0000000000000002) * (r0 * r11) * r1);
            // inst 94: fma_u1d_direct | path#98: direct out[36] += x[7] * y[0] * w[11] * 1.0
            atomicAdd(&out[out_base + (index_t)8064 + (index_t)u], scalar_t(1.0) * (r0 * r10) * r1);
            // inst 95: fma_u1d_direct | path#95: direct out[33] += x[4] * y[0] * w[11] * 0.9999999999999993
            atomicAdd(&out[out_base + (index_t)7392 + (index_t)u], scalar_t(0.9999999999999993) * (r0 * r14) * r1);
            // inst 96: fma_u1d_direct | path#97: direct out[35] += x[6] * y[0] * w[11] * 0.9999999999999997
            atomicAdd(&out[out_base + (index_t)7840 + (index_t)u], scalar_t(0.9999999999999997) * (r0 * r15) * r1);
            // inst 97: fma_u1d_direct | path#99: direct out[37] += x[8] * y[0] * w[11] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)8288 + (index_t)u], scalar_t(1.0000000000000004) * (r0 * r13) * r1);
            // inst 98: release | last use after path#99
            // inst 99: load | global key=(0, 0, (0, 0, 25, 0, -5, 25), "('w', 12)")
            r0 = w[w_base + (index_t)2688 + (index_t)u];
            // inst 100: load | global key=(6, 0, (0, 6, 14, 0, -7, 14), "('y', 7)")
            r12 = y[y_base + (index_t)7];
            // inst 101: fma_u1d_resident | path#101: out[38] += x[5] * y[7] * w[12] * 0.46291004988627604
            out_acc_v_38 += scalar_t(0.46291004988627604) * (r0 * r11) * r12;
            // inst 102: fma_u1d_resident | path#113: out[40] += x[7] * y[7] * w[12] * 0.2672612419124247
            out_acc_v_40 += scalar_t(0.2672612419124247) * (r0 * r10) * r12;
            // inst 103: fma_u1d_resident | path#104: out[39] += x[4] * y[7] * w[12] * 0.4629100498862763
            out_acc_v_39 += scalar_t(0.4629100498862763) * (r0 * r14) * r12;
            // inst 104: fma_u1d_resident | path#117: out[41] += x[6] * y[7] * w[12] * 0.26726124191242445
            out_acc_v_41 += scalar_t(0.26726124191242445) * (r0 * r15) * r12;
            // inst 105: fma_u1d_resident | path#120: out[41] += x[8] * y[7] * w[12] * 0.46291004988627565
            out_acc_v_41 += scalar_t(0.46291004988627565) * (r0 * r13) * r12;
            // inst 106: fma_u1d_resident | path#123: out[42] += x[7] * y[7] * w[12] * 0.4629100498862761
            out_acc_v_42 += scalar_t(0.4629100498862761) * (r0 * r10) * r12;
            // inst 107: load | global key=(6, 0, (0, 6, 14, 0, -7, 14), "('y', 5)")
            r16 = y[y_base + (index_t)5];
            // inst 108: fma_u1d_resident | path#111: out[40] += x[5] * y[5] * w[12] * 0.26726124191242484
            out_acc_v_40 += scalar_t(0.26726124191242484) * (r0 * r11) * r16;
            // inst 109: fma_u1d_resident | path#103: out[38] += x[7] * y[5] * w[12] * 0.46291004988627577
            out_acc_v_38 += scalar_t(0.46291004988627577) * (r0 * r10) * r16;
            // inst 110: fma_u1d_resident | path#107: out[39] += x[6] * y[5] * w[12] * 0.26726124191242423
            out_acc_v_39 += scalar_t(0.26726124191242423) * (r0 * r15) * r16;
            // inst 111: fma_u1d_resident | path#109: out[39] += x[8] * y[5] * w[12] * -0.4629100498862755
            out_acc_v_39 += scalar_t(-0.4629100498862755) * (r0 * r13) * r16;
            // inst 112: fma_u1d_resident | path#115: out[41] += x[4] * y[5] * w[12] * 0.4629100498862755
            out_acc_v_41 += scalar_t(0.4629100498862755) * (r0 * r14) * r16;
            // inst 113: fma_u1d_resident | path#121: out[42] += x[5] * y[5] * w[12] * -0.46291004988627593
            out_acc_v_42 += scalar_t(-0.46291004988627593) * (r0 * r11) * r16;
            // inst 114: load | global key=(5, 0, (0, 5, 12, 0, -7, 13), "('y', 6)")
            r17 = y[y_base + (index_t)6];
            // inst 115: fma_u1d_resident | path#100: out[38] += x[4] * y[6] * w[12] * -0.5345224838248491
            out_acc_v_38 += scalar_t(-0.5345224838248491) * (r0 * r14) * r17;
            // inst 116: fma_u1d_resident | path#105: out[39] += x[5] * y[6] * w[12] * 0.2672612419124244
            out_acc_v_39 += scalar_t(0.2672612419124244) * (r0 * r11) * r17;
            // inst 117: fma_u1d_resident | path#112: out[40] += x[6] * y[6] * w[12] * 0.5345224838248488
            out_acc_v_40 += scalar_t(0.5345224838248488) * (r0 * r15) * r17;
            // inst 118: fma_u1d_resident | path#118: out[41] += x[7] * y[6] * w[12] * 0.26726124191242395
            out_acc_v_41 += scalar_t(0.26726124191242395) * (r0 * r10) * r17;
            // inst 119: fma_u1d_resident | path#124: out[42] += x[8] * y[6] * w[12] * -0.5345224838248492
            out_acc_v_42 += scalar_t(-0.5345224838248492) * (r0 * r13) * r17;
            // inst 120: load | global key=(4, 0, (0, 4, 10, 0, -7, 13), "('y', 8)")
            r18 = y[y_base + (index_t)8];
            // inst 121: fma_u1d_resident | path#106: out[39] += x[5] * y[8] * w[12] * -0.4629100498862753
            out_acc_v_39 += scalar_t(-0.4629100498862753) * (r0 * r11) * r18;
            // inst 122: fma_u1d_resident | path#114: out[40] += x[8] * y[8] * w[12] * -0.5345224838248485
            out_acc_v_40 += scalar_t(-0.5345224838248485) * (r0 * r13) * r18;
            // inst 123: fma_u1d_resident | path#119: out[41] += x[7] * y[8] * w[12] * 0.46291004988627527
            out_acc_v_41 += scalar_t(0.46291004988627527) * (r0 * r10) * r18;
            // inst 124: fma_u1d_resident | path#122: out[42] += x[6] * y[8] * w[12] * -0.5345224838248491
            out_acc_v_42 += scalar_t(-0.5345224838248491) * (r0 * r15) * r18;
            // inst 125: load | global key=(4, 0, (0, 4, 10, 0, -7, 13), "('y', 4)")
            r19 = y[y_base + (index_t)4];
            // inst 126: fma_u1d_resident | path#110: out[40] += x[4] * y[4] * w[12] * -0.5345224838248482
            out_acc_v_40 += scalar_t(-0.5345224838248482) * (r0 * r14) * r19;
            // inst 127: fma_u1d_resident | path#102: out[38] += x[6] * y[4] * w[12] * -0.5345224838248488
            out_acc_v_38 += scalar_t(-0.5345224838248488) * (r0 * r15) * r19;
            // inst 128: fma_u1d_resident | path#108: out[39] += x[7] * y[4] * w[12] * 0.4629100498862756
            out_acc_v_39 += scalar_t(0.4629100498862756) * (r0 * r10) * r19;
            // inst 129: fma_u1d_resident | path#116: out[41] += x[5] * y[4] * w[12] * 0.4629100498862755
            out_acc_v_41 += scalar_t(0.4629100498862755) * (r0 * r11) * r19;
            // inst 130: release | last use after path#116
            // inst 131: load | global key=(5, 0, (0, 5, 10, 0, 0, 5), "('w', 8)")
            r0 = w[w_base + (index_t)1792 + (index_t)u];
            // inst 132: fma_u1d_direct | path#58: direct out[18] += x[0] * y[4] * w[8] * 0.9999999999999993
            atomicAdd(&out[out_base + (index_t)4032 + (index_t)u], scalar_t(0.9999999999999993) * (r0 * r2) * r19);
            // inst 133: fma_u1d_direct | path#62: direct out[22] += x[0] * y[8] * w[8] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)4928 + (index_t)u], scalar_t(1.0000000000000004) * (r0 * r2) * r18);
            // inst 134: fma_u1d_direct | path#59: direct out[19] += x[0] * y[5] * w[8] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)4256 + (index_t)u], scalar_t(1.0000000000000002) * (r0 * r2) * r16);
            // inst 135: fma_u1d_direct | path#60: direct out[20] += x[0] * y[6] * w[8] * 0.9999999999999997
            atomicAdd(&out[out_base + (index_t)4480 + (index_t)u], scalar_t(0.9999999999999997) * (r0 * r2) * r17);
            // inst 136: fma_u1d_direct | path#61: direct out[21] += x[0] * y[7] * w[8] * 1.0
            atomicAdd(&out[out_base + (index_t)4704 + (index_t)u], scalar_t(1.0) * (r0 * r2) * r12);
            // inst 137: release | last use after path#61
            // inst 138: load | global key=(5, 0, (0, 5, 10, 0, 0, 5), "('w', 2)")
            r0 = w[w_base + (index_t)448 + (index_t)u];
            // inst 139: fma_u1d_resident | path#4: out[2] += x[4] * y[4] * w[2] * 0.4472135954999577
            out_acc_v_2 += scalar_t(0.4472135954999577) * (r0 * r14) * r19;
            // inst 140: fma_u1d_resident | path#8: out[2] += x[8] * y[8] * w[2] * 0.4472135954999582
            out_acc_v_2 += scalar_t(0.4472135954999582) * (r0 * r13) * r18;
            // inst 141: fma_u1d_resident | path#5: out[2] += x[5] * y[5] * w[2] * 0.44721359549995815
            out_acc_v_2 += scalar_t(0.44721359549995815) * (r0 * r11) * r16;
            // inst 142: fma_u1d_resident | path#6: out[2] += x[6] * y[6] * w[2] * 0.44721359549995787
            out_acc_v_2 += scalar_t(0.44721359549995787) * (r0 * r15) * r17;
            // inst 143: fma_u1d_resident | path#7: out[2] += x[7] * y[7] * w[2] * 0.447213595499958
            out_acc_v_2 += scalar_t(0.447213595499958) * (r0 * r10) * r12;
            // inst 144: release | last use after path#7
            // inst 145: load | global key=(0, 0, (0, 0, 21, 0, -3, 21), "('w', 15)")
            r0 = w[w_base + (index_t)3360 + (index_t)u];
            // inst 146: load | global key=(8, 0, (0, 8, 21, 0, -7, 18), "('y', 3)")
            r20 = y[y_base + (index_t)3];
            // inst 147: fma_u1d_resident | path#153: out[57] += x[4] * y[3] * w[15] * 0.7071067811865482
            out_acc_v_57 += scalar_t(0.7071067811865482) * (r0 * r14) * r20;
            // inst 148: fma_u1d_resident | path#168: out[61] += x[8] * y[3] * w[15] * -0.18257418583505522
            out_acc_v_61 += scalar_t(-0.18257418583505522) * (r0 * r13) * r20;
            // inst 149: fma_u1d_resident | path#156: out[58] += x[5] * y[3] * w[15] * 0.577350269189626
            out_acc_v_58 += scalar_t(0.577350269189626) * (r0 * r11) * r20;
            // inst 150: fma_u1d_resident | path#158: out[59] += x[4] * y[3] * w[15] * -0.18257418583505522
            out_acc_v_59 += scalar_t(-0.18257418583505522) * (r0 * r14) * r20;
            // inst 151: fma_u1d_resident | path#164: out[60] += x[7] * y[3] * w[15] * -0.44721359549995765
            out_acc_v_60 += scalar_t(-0.44721359549995765) * (r0 * r10) * r20;
            // inst 152: fma_u1d_resident | path#166: out[61] += x[6] * y[3] * w[15] * 0.6324555320336759
            out_acc_v_61 += scalar_t(0.6324555320336759) * (r0 * r15) * r20;
            // inst 153: fma_u1d_resident | path#173: out[63] += x[8] * y[3] * w[15] * 0.7071067811865485
            out_acc_v_63 += scalar_t(0.7071067811865485) * (r0 * r13) * r20;
            // inst 154: fma_u1d_resident | path#170: out[62] += x[7] * y[3] * w[15] * 0.5773502691896263
            out_acc_v_62 += scalar_t(0.5773502691896263) * (r0 * r10) * r20;
            // inst 155: load | global key=(8, 0, (0, 8, 21, 0, -7, 18), "('y', 1)")
            r21 = y[y_base + (index_t)1];
            // inst 156: fma_u1d_resident | path#154: out[57] += x[8] * y[1] * w[15] * 0.7071067811865476
            out_acc_v_57 += scalar_t(0.7071067811865476) * (r0 * r13) * r21;
            // inst 157: fma_u1d_resident | path#160: out[59] += x[6] * y[1] * w[15] * 0.6324555320336759
            out_acc_v_59 += scalar_t(0.6324555320336759) * (r0 * r15) * r21;
            // inst 158: fma_u1d_resident | path#162: out[60] += x[5] * y[1] * w[15] * -0.4472135954999573
            out_acc_v_60 += scalar_t(-0.4472135954999573) * (r0 * r11) * r21;
            // inst 159: fma_u1d_resident | path#165: out[61] += x[4] * y[1] * w[15] * -0.1825741858350552
            out_acc_v_61 += scalar_t(-0.1825741858350552) * (r0 * r14) * r21;
            // inst 160: fma_u1d_resident | path#157: out[58] += x[7] * y[1] * w[15] * 0.5773502691896256
            out_acc_v_58 += scalar_t(0.5773502691896256) * (r0 * r10) * r21;
            // inst 161: fma_u1d_resident | path#161: out[59] += x[8] * y[1] * w[15] * 0.182574185835055
            out_acc_v_59 += scalar_t(0.182574185835055) * (r0 * r13) * r21;
            // inst 162: fma_u1d_resident | path#169: out[62] += x[5] * y[1] * w[15] * -0.5773502691896258
            out_acc_v_62 += scalar_t(-0.5773502691896258) * (r0 * r11) * r21;
            // inst 163: fma_u1d_resident | path#172: out[63] += x[4] * y[1] * w[15] * -0.7071067811865477
            out_acc_v_63 += scalar_t(-0.7071067811865477) * (r0 * r14) * r21;
            // inst 164: load | global key=(8, 0, (0, 8, 19, 0, -1, 11), "('w', 6)")
            r22 = w[w_base + (index_t)1344 + (index_t)u];
            // inst 165: fma_u1d_resident | path#28: out[12] += x[6] * y[1] * w[6] * -0.3162277660168376
            out_acc_v_12 += scalar_t(-0.3162277660168376) * (r22 * r15) * r21;
            // inst 166: fma_u1d_resident | path#26: out[12] += x[4] * y[3] * w[6] * 0.5477225575051665
            out_acc_v_12 += scalar_t(0.5477225575051665) * (r22 * r14) * r20;
            // inst 167: fma_u1d_resident | path#29: out[12] += x[8] * y[1] * w[6] * -0.5477225575051664
            out_acc_v_12 += scalar_t(-0.5477225575051664) * (r22 * r13) * r21;
            // inst 168: fma_u1d_resident | path#32: out[13] += x[7] * y[3] * w[6] * 0.5477225575051669
            out_acc_v_13 += scalar_t(0.5477225575051669) * (r22 * r10) * r20;
            // inst 169: fma_u1d_resident | path#30: out[13] += x[5] * y[1] * w[6] * 0.5477225575051657
            out_acc_v_13 += scalar_t(0.5477225575051657) * (r22 * r11) * r21;
            // inst 170: fma_u1d_resident | path#34: out[14] += x[6] * y[3] * w[6] * -0.31622776601683794
            out_acc_v_14 += scalar_t(-0.31622776601683794) * (r22 * r15) * r20;
            // inst 171: fma_u1d_resident | path#33: out[14] += x[4] * y[1] * w[6] * 0.5477225575051664
            out_acc_v_14 += scalar_t(0.5477225575051664) * (r22 * r14) * r21;
            // inst 172: fma_u1d_resident | path#36: out[14] += x[8] * y[3] * w[6] * 0.5477225575051664
            out_acc_v_14 += scalar_t(0.5477225575051664) * (r22 * r13) * r20;
            // inst 173: load | global key=(8, 1, (4, 8, 17, 0, -6, 13), "('y', 2)")
            r23 = y[y_base + (index_t)2];
            // inst 174: fma_u1d_resident | path#155: out[58] += x[4] * y[2] * w[15] * 0.5773502691896256
            out_acc_v_58 += scalar_t(0.5773502691896256) * (r0 * r14) * r23;
            // inst 175: release | last use after path#155
            // inst 176: fma_u1d_resident | path#171: out[62] += x[8] * y[2] * w[15] * 0.5773502691896257
            out_acc_v_62 += scalar_t(0.5773502691896257) * (r0 * r13) * r23;
            // inst 177: release | last use after path#171
            // inst 178: fma_u1d_resident | path#27: out[12] += x[5] * y[2] * w[6] * 0.5477225575051657
            out_acc_v_12 += scalar_t(0.5477225575051657) * (r22 * r11) * r23;
            // inst 179: fma_u1d_resident | path#159: out[59] += x[5] * y[2] * w[15] * 0.7302967433402211
            out_acc_v_59 += scalar_t(0.7302967433402211) * (r0 * r11) * r23;
            // inst 180: release | last use after path#159
            // inst 181: fma_u1d_resident | path#31: out[13] += x[6] * y[2] * w[6] * 0.6324555320336755
            out_acc_v_13 += scalar_t(0.6324555320336755) * (r22 * r15) * r23;
            // inst 182: fma_u1d_resident | path#35: out[14] += x[7] * y[2] * w[6] * 0.5477225575051656
            out_acc_v_14 += scalar_t(0.5477225575051656) * (r22 * r10) * r23;
            // inst 183: release | last use after path#35
            // inst 184: fma_u1d_resident | path#163: out[60] += x[6] * y[2] * w[15] * 0.774596669241483
            out_acc_v_60 += scalar_t(0.774596669241483) * (r0 * r15) * r23;
            // inst 185: release | last use after path#163
            // inst 186: fma_u1d_resident | path#167: out[61] += x[7] * y[2] * w[15] * 0.7302967433402209
            out_acc_v_61 += scalar_t(0.7302967433402209) * (r0 * r10) * r23;
            // inst 187: release | last use after path#167
            // inst 188: release | last use after path#167
            // inst 189: load | global key=(3, 0, (0, 3, 6, 0, 0, 3), "('w', 3)")
            r0 = w[w_base + (index_t)672 + (index_t)u];
            // inst 190: fma_u1d_direct | path#9: direct out[3] += x[0] * y[1] * w[3] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)672 + (index_t)u], scalar_t(1.0000000000000002) * (r0 * r2) * r21);
            // inst 191: fma_u1d_direct | path#11: direct out[5] += x[0] * y[3] * w[3] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)1120 + (index_t)u], scalar_t(1.0000000000000002) * (r0 * r2) * r20);
            // inst 192: fma_u1d_direct | path#10: direct out[4] += x[0] * y[2] * w[3] * 0.9999999999999998
            atomicAdd(&out[out_base + (index_t)896 + (index_t)u], scalar_t(0.9999999999999998) * (r0 * r2) * r23);
            // inst 193: release | last use after path#10
            // inst 194: release | last use after path#10
            // inst 195: load | global key=(0, 0, (0, 0, 26, 0, -6, 26), "('x', 3)")
            r0 = x[x_base + (index_t)672 + (index_t)u];
            // inst 196: load | global key=(8, 0, (0, 8, 29, 0, -2, 21), "('w', 14)")
            r2 = w[w_base + (index_t)3136 + (index_t)u];
            // inst 197: fma_u1d_resident | path#133: out[50] += x[3] * y[4] * w[14] * 0.7071067811865482
            out_acc_v_50 += scalar_t(0.7071067811865482) * (r2 * r0) * r19;
            // inst 198: fma_u1d_resident | path#147: out[54] += x[3] * y[8] * w[14] * -0.18257418583505522
            out_acc_v_54 += scalar_t(-0.18257418583505522) * (r2 * r0) * r18;
            // inst 199: fma_u1d_resident | path#136: out[51] += x[3] * y[5] * w[14] * 0.577350269189626
            out_acc_v_51 += scalar_t(0.577350269189626) * (r2 * r0) * r16;
            // inst 200: fma_u1d_resident | path#140: out[52] += x[3] * y[4] * w[14] * -0.18257418583505522
            out_acc_v_52 += scalar_t(-0.18257418583505522) * (r2 * r0) * r19;
            // inst 201: fma_u1d_resident | path#143: out[53] += x[3] * y[7] * w[14] * -0.44721359549995765
            out_acc_v_53 += scalar_t(-0.44721359549995765) * (r2 * r0) * r12;
            // inst 202: fma_u1d_resident | path#146: out[54] += x[3] * y[6] * w[14] * 0.6324555320336759
            out_acc_v_54 += scalar_t(0.6324555320336759) * (r2 * r0) * r17;
            // inst 203: fma_u1d_resident | path#152: out[56] += x[3] * y[8] * w[14] * 0.7071067811865485
            out_acc_v_56 += scalar_t(0.7071067811865485) * (r2 * r0) * r18;
            // inst 204: fma_u1d_resident | path#150: out[55] += x[3] * y[7] * w[14] * 0.5773502691896263
            out_acc_v_55 += scalar_t(0.5773502691896263) * (r2 * r0) * r12;
            // inst 205: load | global key=(8, 0, (0, 8, 34, 0, -5, 26), "('x', 1)")
            r10 = x[x_base + (index_t)224 + (index_t)u];
            // inst 206: fma_u1d_resident | path#132: out[50] += x[1] * y[8] * w[14] * 0.7071067811865476
            out_acc_v_50 += scalar_t(0.7071067811865476) * (r2 * r10) * r18;
            // inst 207: fma_u1d_resident | path#137: out[52] += x[1] * y[6] * w[14] * 0.6324555320336759
            out_acc_v_52 += scalar_t(0.6324555320336759) * (r2 * r10) * r17;
            // inst 208: fma_u1d_resident | path#141: out[53] += x[1] * y[5] * w[14] * -0.4472135954999573
            out_acc_v_53 += scalar_t(-0.4472135954999573) * (r2 * r10) * r16;
            // inst 209: fma_u1d_resident | path#144: out[54] += x[1] * y[4] * w[14] * -0.1825741858350552
            out_acc_v_54 += scalar_t(-0.1825741858350552) * (r2 * r10) * r19;
            // inst 210: fma_u1d_resident | path#134: out[51] += x[1] * y[7] * w[14] * 0.5773502691896256
            out_acc_v_51 += scalar_t(0.5773502691896256) * (r2 * r10) * r12;
            // inst 211: fma_u1d_resident | path#138: out[52] += x[1] * y[8] * w[14] * 0.182574185835055
            out_acc_v_52 += scalar_t(0.182574185835055) * (r2 * r10) * r18;
            // inst 212: fma_u1d_resident | path#148: out[55] += x[1] * y[5] * w[14] * -0.5773502691896258
            out_acc_v_55 += scalar_t(-0.5773502691896258) * (r2 * r10) * r16;
            // inst 213: fma_u1d_resident | path#151: out[56] += x[1] * y[4] * w[14] * -0.7071067811865477
            out_acc_v_56 += scalar_t(-0.7071067811865477) * (r2 * r10) * r19;
            // inst 214: load | global key=(16, 0, (0, 16, 37, 0, -1, 21), "('w', 10)")
            r15 = w[w_base + (index_t)2240 + (index_t)u];
            // inst 215: fma_u1d_resident | path#74: out[28] += x[1] * y[13] * w[10] * -0.15430334996209177
            out_acc_v_28 += scalar_t(-0.15430334996209177) * (r15 * r10) * r7;
            // inst 216: fma_u1d_resident | path#78: out[28] += x[3] * y[11] * w[10] * -0.1543033499620918
            out_acc_v_28 += scalar_t(-0.1543033499620918) * (r15 * r0) * r8;
            // inst 217: fma_u1d_resident | path#79: out[29] += x[1] * y[12] * w[10] * -0.37796447300922675
            out_acc_v_29 += scalar_t(-0.37796447300922675) * (r15 * r10) * r9;
            // inst 218: fma_u1d_resident | path#82: out[29] += x[3] * y[10] * w[10] * 0.4879500364742667
            out_acc_v_29 += scalar_t(0.4879500364742667) * (r15 * r0) * r4;
            // inst 219: fma_u1d_resident | path#80: out[29] += x[1] * y[14] * w[10] * -0.48795003647426666
            out_acc_v_29 += scalar_t(-0.48795003647426666) * (r15 * r10) * r3;
            // inst 220: fma_u1d_resident | path#85: out[30] += x[3] * y[13] * w[10] * 0.5345224838248488
            out_acc_v_30 += scalar_t(0.5345224838248488) * (r15 * r0) * r7;
            // inst 221: fma_u1d_resident | path#83: out[30] += x[1] * y[11] * w[10] * 0.5345224838248488
            out_acc_v_30 += scalar_t(0.5345224838248488) * (r15 * r10) * r8;
            // inst 222: fma_u1d_resident | path#77: out[28] += x[3] * y[9] * w[10] * 0.5976143046671976
            out_acc_v_28 += scalar_t(0.5976143046671976) * (r15 * r0) * r5;
            // inst 223: fma_u1d_resident | path#90: out[32] += x[1] * y[9] * w[10] * 0.5976143046671969
            out_acc_v_32 += scalar_t(0.5976143046671969) * (r15 * r10) * r5;
            // inst 224: release | last use after path#90
            // inst 225: fma_u1d_resident | path#88: out[31] += x[3] * y[12] * w[10] * -0.377964473009227
            out_acc_v_31 += scalar_t(-0.377964473009227) * (r15 * r0) * r9;
            // inst 226: fma_u1d_resident | path#75: out[28] += x[1] * y[15] * w[10] * -0.597614304667197
            out_acc_v_28 += scalar_t(-0.597614304667197) * (r15 * r10) * r6;
            // inst 227: fma_u1d_resident | path#94: out[32] += x[3] * y[15] * w[10] * 0.5976143046671977
            out_acc_v_32 += scalar_t(0.5976143046671977) * (r15 * r0) * r6;
            // inst 228: release | last use after path#94
            // inst 229: fma_u1d_resident | path#86: out[31] += x[1] * y[10] * w[10] * 0.4879500364742665
            out_acc_v_31 += scalar_t(0.4879500364742665) * (r15 * r10) * r4;
            // inst 230: fma_u1d_resident | path#89: out[31] += x[3] * y[14] * w[10] * 0.487950036474267
            out_acc_v_31 += scalar_t(0.487950036474267) * (r15 * r0) * r3;
            // inst 231: fma_u1d_resident | path#91: out[32] += x[1] * y[11] * w[10] * 0.1543033499620916
            out_acc_v_32 += scalar_t(0.1543033499620916) * (r15 * r10) * r8;
            // inst 232: fma_u1d_resident | path#93: out[32] += x[3] * y[13] * w[10] * -0.1543033499620918
            out_acc_v_32 += scalar_t(-0.1543033499620918) * (r15 * r0) * r7;
            // inst 233: load | global key=(10, 1, (10, 10, 28, 0, -4, 18), "('x', 2)")
            r6 = x[x_base + (index_t)448 + (index_t)u];
            // inst 234: fma_u1d_resident | path#76: out[28] += x[2] * y[10] * w[10] * 0.4879500364742665
            out_acc_v_28 += scalar_t(0.4879500364742665) * (r15 * r6) * r4;
            // inst 235: release | last use after path#76
            // inst 236: fma_u1d_resident | path#81: out[29] += x[2] * y[11] * w[10] * 0.6172133998483673
            out_acc_v_29 += scalar_t(0.6172133998483673) * (r15 * r6) * r8;
            // inst 237: release | last use after path#81
            // inst 238: fma_u1d_resident | path#84: out[30] += x[2] * y[12] * w[10] * 0.6546536707079768
            out_acc_v_30 += scalar_t(0.6546536707079768) * (r15 * r6) * r9;
            // inst 239: release | last use after path#84
            // inst 240: fma_u1d_resident | path#87: out[31] += x[2] * y[13] * w[10] * 0.6172133998483672
            out_acc_v_31 += scalar_t(0.6172133998483672) * (r15 * r6) * r7;
            // inst 241: release | last use after path#87
            // inst 242: fma_u1d_resident | path#92: out[32] += x[2] * y[14] * w[10] * 0.48795003647426655
            out_acc_v_32 += scalar_t(0.48795003647426655) * (r15 * r6) * r3;
            // inst 243: release | last use after path#92
            // inst 244: release | last use after path#92
            // inst 245: fma_u1d_resident | path#142: out[53] += x[2] * y[6] * w[14] * 0.774596669241483
            out_acc_v_53 += scalar_t(0.774596669241483) * (r2 * r6) * r17;
            // inst 246: fma_u1d_resident | path#135: out[51] += x[2] * y[4] * w[14] * 0.5773502691896256
            out_acc_v_51 += scalar_t(0.5773502691896256) * (r2 * r6) * r19;
            // inst 247: fma_u1d_resident | path#139: out[52] += x[2] * y[5] * w[14] * 0.7302967433402211
            out_acc_v_52 += scalar_t(0.7302967433402211) * (r2 * r6) * r16;
            // inst 248: fma_u1d_resident | path#145: out[54] += x[2] * y[7] * w[14] * 0.7302967433402209
            out_acc_v_54 += scalar_t(0.7302967433402209) * (r2 * r6) * r12;
            // inst 249: fma_u1d_resident | path#149: out[55] += x[2] * y[8] * w[14] * 0.5773502691896257
            out_acc_v_55 += scalar_t(0.5773502691896257) * (r2 * r6) * r18;
            // inst 250: release | last use after path#149
            // inst 251: load | global key=(11, 0, (0, 11, 22, 0, 0, 11), "('w', 9)")
            r2 = w[w_base + (index_t)2016 + (index_t)u];
            // inst 252: fma_u1d_resident | path#63: out[23] += x[1] * y[3] * w[9] * 0.7071067811865479
            out_acc_v_23 += scalar_t(0.7071067811865479) * (r2 * r10) * r20;
            // inst 253: fma_u1d_resident | path#64: out[23] += x[3] * y[1] * w[9] * 0.7071067811865478
            out_acc_v_23 += scalar_t(0.7071067811865478) * (r2 * r0) * r21;
            // inst 254: fma_u1d_resident | path#65: out[24] += x[1] * y[2] * w[9] * 0.7071067811865469
            out_acc_v_24 += scalar_t(0.7071067811865469) * (r2 * r10) * r23;
            // inst 255: fma_u1d_resident | path#69: out[25] += x[3] * y[3] * w[9] * -0.40824829046386296
            out_acc_v_25 += scalar_t(-0.40824829046386296) * (r2 * r0) * r20;
            // inst 256: fma_u1d_resident | path#66: out[24] += x[2] * y[1] * w[9] * 0.707106781186547
            out_acc_v_24 += scalar_t(0.707106781186547) * (r2 * r6) * r21;
            // inst 257: fma_u1d_resident | path#67: out[25] += x[1] * y[1] * w[9] * -0.4082482904638625
            out_acc_v_25 += scalar_t(-0.4082482904638625) * (r2 * r10) * r21;
            // inst 258: fma_u1d_resident | path#71: out[26] += x[3] * y[2] * w[9] * 0.7071067811865468
            out_acc_v_26 += scalar_t(0.7071067811865468) * (r2 * r0) * r23;
            // inst 259: fma_u1d_resident | path#70: out[26] += x[2] * y[3] * w[9] * 0.7071067811865483
            out_acc_v_26 += scalar_t(0.7071067811865483) * (r2 * r6) * r20;
            // inst 260: fma_u1d_resident | path#72: out[27] += x[1] * y[1] * w[9] * -0.7071067811865478
            out_acc_v_27 += scalar_t(-0.7071067811865478) * (r2 * r10) * r21;
            // inst 261: fma_u1d_resident | path#73: out[27] += x[3] * y[3] * w[9] * 0.7071067811865478
            out_acc_v_27 += scalar_t(0.7071067811865478) * (r2 * r0) * r20;
            // inst 262: fma_u1d_resident | path#68: out[25] += x[2] * y[2] * w[9] * 0.8164965809277256
            out_acc_v_25 += scalar_t(0.8164965809277256) * (r2 * r6) * r23;
            // inst 263: release | last use after path#68
            // inst 264: load | global key=(11, 0, (0, 11, 22, 0, 0, 11), "('w', 5)")
            r2 = w[w_base + (index_t)1120 + (index_t)u];
            // inst 265: fma_u1d_resident | path#15: out[9] += x[1] * y[6] * w[5] * -0.3162277660168376
            out_acc_v_9 += scalar_t(-0.3162277660168376) * (r2 * r10) * r17;
            // inst 266: fma_u1d_resident | path#18: out[9] += x[3] * y[4] * w[5] * 0.5477225575051664
            out_acc_v_9 += scalar_t(0.5477225575051664) * (r2 * r0) * r19;
            // inst 267: fma_u1d_resident | path#22: out[11] += x[1] * y[4] * w[5] * 0.5477225575051665
            out_acc_v_11 += scalar_t(0.5477225575051665) * (r2 * r10) * r19;
            // inst 268: release | last use after path#22
            // inst 269: fma_u1d_resident | path#17: out[9] += x[2] * y[5] * w[5] * 0.5477225575051657
            out_acc_v_9 += scalar_t(0.5477225575051657) * (r2 * r6) * r16;
            // inst 270: fma_u1d_resident | path#19: out[10] += x[1] * y[5] * w[5] * 0.5477225575051657
            out_acc_v_10 += scalar_t(0.5477225575051657) * (r2 * r10) * r16;
            // inst 271: release | last use after path#19
            // inst 272: fma_u1d_resident | path#21: out[10] += x[3] * y[7] * w[5] * 0.5477225575051656
            out_acc_v_10 += scalar_t(0.5477225575051656) * (r2 * r0) * r12;
            // inst 273: fma_u1d_resident | path#23: out[11] += x[2] * y[7] * w[5] * 0.5477225575051669
            out_acc_v_11 += scalar_t(0.5477225575051669) * (r2 * r6) * r12;
            // inst 274: release | last use after path#23
            // inst 275: fma_u1d_resident | path#24: out[11] += x[3] * y[6] * w[5] * -0.31622776601683794
            out_acc_v_11 += scalar_t(-0.31622776601683794) * (r2 * r0) * r17;
            // inst 276: fma_u1d_resident | path#20: out[10] += x[2] * y[6] * w[5] * 0.6324555320336755
            out_acc_v_10 += scalar_t(0.6324555320336755) * (r2 * r6) * r17;
            // inst 277: release | last use after path#20
            // inst 278: fma_u1d_resident | path#16: out[9] += x[1] * y[8] * w[5] * -0.5477225575051664
            out_acc_v_9 += scalar_t(-0.5477225575051664) * (r2 * r10) * r18;
            // inst 279: fma_u1d_resident | path#25: out[11] += x[3] * y[8] * w[5] * 0.5477225575051664
            out_acc_v_11 += scalar_t(0.5477225575051664) * (r2 * r0) * r18;
            // inst 280: release | last use after path#25
            // inst 281: release | last use after path#25
            // inst 282: load | global key=(3, 1, (6, 3, 6, 0, 0, 3), "('w', 1)")
            r2 = w[w_base + (index_t)224 + (index_t)u];
            // inst 283: fma_u1d_resident | path#1: out[1] += x[1] * y[1] * w[1] * 0.5773502691896258
            out_acc_v_1 += scalar_t(0.5773502691896258) * (r2 * r10) * r21;
            // inst 284: release | last use after path#1
            // inst 285: fma_u1d_resident | path#2: out[1] += x[2] * y[2] * w[1] * 0.5773502691896256
            out_acc_v_1 += scalar_t(0.5773502691896256) * (r2 * r6) * r23;
            // inst 286: release | last use after path#2
            // inst 287: fma_u1d_resident | path#3: out[1] += x[3] * y[3] * w[1] * 0.5773502691896258
            out_acc_v_1 += scalar_t(0.5773502691896258) * (r2 * r0) * r20;
            // inst 288: release | last use after path#3
            // inst 289: release | last use after path#3
            // inst 290: load | global key=(3, 1, (6, 3, 6, 0, 0, 3), "('w', 4)")
            r2 = w[w_base + (index_t)896 + (index_t)u];
            // inst 291: fma_u1d_direct | path#12: direct out[6] += x[1] * y[0] * w[4] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)1344 + (index_t)u], scalar_t(1.0000000000000002) * (r2 * r10) * r1);
            // inst 292: release | last use after path#12
            // inst 293: fma_u1d_direct | path#13: direct out[7] += x[2] * y[0] * w[4] * 0.9999999999999998
            atomicAdd(&out[out_base + (index_t)1568 + (index_t)u], scalar_t(0.9999999999999998) * (r2 * r6) * r1);
            // inst 294: release | last use after path#13
            // inst 295: fma_u1d_direct | path#14: direct out[8] += x[3] * y[0] * w[4] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)1792 + (index_t)u], scalar_t(1.0000000000000002) * (r2 * r0) * r1);
            // inst 296: release | last use after path#14
            // inst 297: release | last use after path#14
            // inst 298: release | last use after path#14

            // resident output accumulator writeback
            atomicAdd(&out[out_base + (index_t)224 + (index_t)u], out_acc_v_1);
            atomicAdd(&out[out_base + (index_t)448 + (index_t)u], out_acc_v_2);
            atomicAdd(&out[out_base + (index_t)2016 + (index_t)u], out_acc_v_9);
            atomicAdd(&out[out_base + (index_t)2240 + (index_t)u], out_acc_v_10);
            atomicAdd(&out[out_base + (index_t)2464 + (index_t)u], out_acc_v_11);
            atomicAdd(&out[out_base + (index_t)2688 + (index_t)u], out_acc_v_12);
            atomicAdd(&out[out_base + (index_t)2912 + (index_t)u], out_acc_v_13);
            atomicAdd(&out[out_base + (index_t)3136 + (index_t)u], out_acc_v_14);
            atomicAdd(&out[out_base + (index_t)3360 + (index_t)u], out_acc_v_15);
            atomicAdd(&out[out_base + (index_t)3584 + (index_t)u], out_acc_v_16);
            atomicAdd(&out[out_base + (index_t)3808 + (index_t)u], out_acc_v_17);
            atomicAdd(&out[out_base + (index_t)5152 + (index_t)u], out_acc_v_23);
            atomicAdd(&out[out_base + (index_t)5376 + (index_t)u], out_acc_v_24);
            atomicAdd(&out[out_base + (index_t)5600 + (index_t)u], out_acc_v_25);
            atomicAdd(&out[out_base + (index_t)5824 + (index_t)u], out_acc_v_26);
            atomicAdd(&out[out_base + (index_t)6048 + (index_t)u], out_acc_v_27);
            atomicAdd(&out[out_base + (index_t)6272 + (index_t)u], out_acc_v_28);
            atomicAdd(&out[out_base + (index_t)6496 + (index_t)u], out_acc_v_29);
            atomicAdd(&out[out_base + (index_t)6720 + (index_t)u], out_acc_v_30);
            atomicAdd(&out[out_base + (index_t)6944 + (index_t)u], out_acc_v_31);
            atomicAdd(&out[out_base + (index_t)7168 + (index_t)u], out_acc_v_32);
            atomicAdd(&out[out_base + (index_t)8512 + (index_t)u], out_acc_v_38);
            atomicAdd(&out[out_base + (index_t)8736 + (index_t)u], out_acc_v_39);
            atomicAdd(&out[out_base + (index_t)8960 + (index_t)u], out_acc_v_40);
            atomicAdd(&out[out_base + (index_t)9184 + (index_t)u], out_acc_v_41);
            atomicAdd(&out[out_base + (index_t)9408 + (index_t)u], out_acc_v_42);
            atomicAdd(&out[out_base + (index_t)11200 + (index_t)u], out_acc_v_50);
            atomicAdd(&out[out_base + (index_t)11424 + (index_t)u], out_acc_v_51);
            atomicAdd(&out[out_base + (index_t)11648 + (index_t)u], out_acc_v_52);
            atomicAdd(&out[out_base + (index_t)11872 + (index_t)u], out_acc_v_53);
            atomicAdd(&out[out_base + (index_t)12096 + (index_t)u], out_acc_v_54);
            atomicAdd(&out[out_base + (index_t)12320 + (index_t)u], out_acc_v_55);
            atomicAdd(&out[out_base + (index_t)12544 + (index_t)u], out_acc_v_56);
            atomicAdd(&out[out_base + (index_t)12768 + (index_t)u], out_acc_v_57);
            atomicAdd(&out[out_base + (index_t)12992 + (index_t)u], out_acc_v_58);
            atomicAdd(&out[out_base + (index_t)13216 + (index_t)u], out_acc_v_59);
            atomicAdd(&out[out_base + (index_t)13440 + (index_t)u], out_acc_v_60);
            atomicAdd(&out[out_base + (index_t)13664 + (index_t)u], out_acc_v_61);
            atomicAdd(&out[out_base + (index_t)13888 + (index_t)u], out_acc_v_62);
            atomicAdd(&out[out_base + (index_t)14112 + (index_t)u], out_acc_v_63);
            atomicAdd(&out[out_base + (index_t)14336 + (index_t)u], out_acc_v_64);
            atomicAdd(&out[out_base + (index_t)14560 + (index_t)u], out_acc_v_65);
            atomicAdd(&out[out_base + (index_t)14784 + (index_t)u], out_acc_v_66);
            atomicAdd(&out[out_base + (index_t)15008 + (index_t)u], out_acc_v_67);
            atomicAdd(&out[out_base + (index_t)15232 + (index_t)u], out_acc_v_68);
            atomicAdd(&out[out_base + (index_t)15456 + (index_t)u], out_acc_v_69);
            atomicAdd(&out[out_base + (index_t)15680 + (index_t)u], out_acc_v_70);
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

static inline bool should_use_int32_index_fwd(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && out_ok;
}

template <typename scalar_t, typename index_t>
void launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd_typed(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd<scalar_t, index_t><<<grid, block, 0, stream>>>(
        w, x, y, out,
        src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    constexpr bool kUseXSrc = true;
    constexpr bool kUseYSrc = false;
    constexpr bool kUseScatter = true;
    constexpr bool kModeScalarY = true;
    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,
                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd_typed<scalar_t, int32_t>(
            w, x, y, out, src_idx, dst_idx,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd_typed<scalar_t, int64_t>(
            w, x, y, out, src_idx, dst_idx,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd_auto<scalar_t>(
        w, x, y, out, src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S, stream);
}



torch::Tensor launcher_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd(
    torch::Tensor w,
    torch::Tensor x_all,
    torch::Tensor y,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    int64_t V64)
{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : [B,Ky,1] or [S,Ky,1]
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "w/x_all/y must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");

    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");

    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");


    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int)dst_idx.size(0);
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");

    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(1) > 0, "Iw must be > 0");

    TORCH_CHECK((int)x_all.size(1) > 0, "Ix must be > 0");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");

    TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");

    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");
    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");


    auto out = torch::zeros({S, V, U}, w.options());

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd", [&] {

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd, "uniform1d_lars_all_inputs_u224_path215_uu_u_xsrc1_ysrc0_scatter1_fwd forward jit impl");
}
