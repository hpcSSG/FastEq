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

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum_xor_lars_bwd(scalar_t v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        v += __shfl_xor(v, offset);
#else
        v += __shfl_xor_sync(0xffffffff, v, offset);
#endif
    }
    return v;
}

template <typename scalar_t, typename index_t>
__global__ void uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_w,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_y,
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

    const int src = src_idx[e_orig];
    const int dst = dst_idx[e_orig];

    const int x_row = src;
    const int y_row = e_orig;
    const int go_row = dst;

    const index_t w_base  = (index_t)w_row * (index_t)3808;
    const index_t gw_base = (index_t)w_row * (index_t)3808;
    const index_t x_base  = (index_t)x_row * (index_t)2016;
    const index_t gx_base = (index_t)x_row * (index_t)2016;
    const index_t y_base  = (index_t)y_row * (index_t)Ky;
    const index_t gy_base = (index_t)y_row * (index_t)Ky;
    const index_t go_base = (index_t)go_row * (index_t)15904;

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
            scalar_t r24;
            scalar_t r25;
            scalar_t r26;
            scalar_t r27;
            scalar_t r28;

            scalar_t gw_acc_i_0 = scalar_t(0);
            scalar_t gw_acc_i_1 = scalar_t(0);
            scalar_t gw_acc_i_2 = scalar_t(0);
            scalar_t gw_acc_i_3 = scalar_t(0);
            scalar_t gw_acc_i_4 = scalar_t(0);
            scalar_t gw_acc_i_5 = scalar_t(0);
            scalar_t gw_acc_i_6 = scalar_t(0);
            scalar_t gw_acc_i_7 = scalar_t(0);
            scalar_t gw_acc_i_8 = scalar_t(0);
            scalar_t gw_acc_i_9 = scalar_t(0);
            scalar_t gw_acc_i_10 = scalar_t(0);
            scalar_t gw_acc_i_11 = scalar_t(0);
            scalar_t gw_acc_i_12 = scalar_t(0);
            scalar_t gw_acc_i_13 = scalar_t(0);
            scalar_t gw_acc_i_14 = scalar_t(0);
            scalar_t gw_acc_i_15 = scalar_t(0);
            scalar_t gw_acc_i_16 = scalar_t(0);
            scalar_t gx_acc_j_0 = scalar_t(0);
            scalar_t gx_acc_j_1 = scalar_t(0);
            scalar_t gx_acc_j_2 = scalar_t(0);
            scalar_t gx_acc_j_3 = scalar_t(0);
            scalar_t gx_acc_j_4 = scalar_t(0);
            scalar_t gx_acc_j_5 = scalar_t(0);
            scalar_t gx_acc_j_6 = scalar_t(0);
            scalar_t gx_acc_j_7 = scalar_t(0);
            scalar_t gx_acc_j_8 = scalar_t(0);
            scalar_t gy_acc_k_0 = scalar_t(0);
            scalar_t gy_acc_k_1 = scalar_t(0);
            scalar_t gy_acc_k_2 = scalar_t(0);
            scalar_t gy_acc_k_3 = scalar_t(0);
            scalar_t gy_acc_k_4 = scalar_t(0);
            scalar_t gy_acc_k_5 = scalar_t(0);
            scalar_t gy_acc_k_6 = scalar_t(0);
            scalar_t gy_acc_k_7 = scalar_t(0);
            scalar_t gy_acc_k_8 = scalar_t(0);
            scalar_t gy_acc_k_9 = scalar_t(0);
            scalar_t gy_acc_k_10 = scalar_t(0);
            scalar_t gy_acc_k_11 = scalar_t(0);
            scalar_t gy_acc_k_12 = scalar_t(0);
            scalar_t gy_acc_k_13 = scalar_t(0);
            scalar_t gy_acc_k_14 = scalar_t(0);
            scalar_t gy_acc_k_15 = scalar_t(0);

            // inst 0: load | global key=(0, 0, (0, 0, 0, 0, -3, 1), "('w', 0)")
            r0 = w[w_base + (index_t)0 + (index_t)u];
            // inst 1: load | global key=(0, 0, (0, 0, 1, 0, -2, 1), "('go', 0)")
            r1 = grad_out[go_base + (index_t)0 + (index_t)u];
            // inst 2: load | global key=(0, 0, (0, 0, 2, 0, -19, 9), "('y', 0)")
            r2 = y[y_base + (index_t)0];
            // inst 3: load | global key=(1, 2, (4, 1, 3, 0, -33, 16), "('x', 0)")
            r3 = x[x_base + (index_t)0 + (index_t)u];
            // inst 4: bwd_fma_resident | path#0: gw[0] += go[0]*x[0]*y[0], gx[0] += w[0]*go[0]*y[0], gy[0] += w[0]*go[0]*x[0]
            gw_acc_i_0 += scalar_t(1.0) * r1 * r3 * r2;
            gx_acc_j_0 += scalar_t(1.0) * (r0 * r1) * r2;
            gy_acc_k_0 += scalar_t(1.0) * (r0 * r1) * r3;
            // inst 5: release | last use after path#0
            // inst 6: release | last use after path#0
            // inst 7: load | global key=(0, 0, (0, 0, 7, 0, -14, 7), "('w', 13)")
            r1 = w[w_base + (index_t)2912 + (index_t)u];
            // inst 8: load | global key=(0, 0, (0, 0, 5, 0, -10, 5), "('w', 8)")
            r0 = w[w_base + (index_t)1792 + (index_t)u];
            // inst 9: load | global key=(0, 0, (0, 0, 5, 0, -10, 5), "('w', 11)")
            r4 = w[w_base + (index_t)2464 + (index_t)u];
            // inst 10: load | global key=(0, 0, (0, 0, 3, 0, -6, 3), "('w', 4)")
            r5 = w[w_base + (index_t)896 + (index_t)u];
            // inst 11: load | global key=(0, 0, (0, 0, 3, 0, -6, 3), "('w', 3)")
            r6 = w[w_base + (index_t)672 + (index_t)u];
            // inst 12: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('go', 8)")
            r7 = grad_out[go_base + (index_t)1792 + (index_t)u];
            // inst 13: load | global key=(1, 1, (2, 1, 3, 0, -40, 26), "('x', 3)")
            r8 = x[x_base + (index_t)672 + (index_t)u];
            // inst 14: bwd_fma_resident | path#14: gw[4] += go[8]*x[3]*y[0], gx[3] += w[4]*go[8]*y[0], gy[0] += w[4]*go[8]*x[3]
            gw_acc_i_4 += scalar_t(1.0000000000000002) * r7 * r8 * r2;
            gx_acc_j_3 += scalar_t(1.0000000000000002) * (r5 * r7) * r2;
            gy_acc_k_0 += scalar_t(1.0000000000000002) * (r5 * r7) * r8;
            // inst 15: release | last use after path#14
            // inst 16: load | global key=(0, 0, (0, 0, 8, 0, -14, 21), "('w', 14)")
            r7 = w[w_base + (index_t)3136 + (index_t)u];
            // inst 17: load | global key=(0, 0, (0, 0, 10, 0, -22, 13), "('y', 8)")
            r9 = y[y_base + (index_t)8];
            // inst 18: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 22)")
            r10 = grad_out[go_base + (index_t)4928 + (index_t)u];
            // inst 19: bwd_fma_resident | path#62: gw[8] += go[22]*x[0]*y[8], gx[0] += w[8]*go[22]*y[8], gy[8] += w[8]*go[22]*x[0]
            gw_acc_i_8 += scalar_t(1.0000000000000004) * r10 * r3 * r9;
            gx_acc_j_0 += scalar_t(1.0000000000000004) * (r0 * r10) * r9;
            gy_acc_k_8 += scalar_t(1.0000000000000004) * (r0 * r10) * r3;
            // inst 20: release | last use after path#62
            // inst 21: load | global key=(1, 0, (0, 1, 7, 0, -5, 4), "('go', 54)")
            r10 = grad_out[go_base + (index_t)12096 + (index_t)u];
            // inst 22: bwd_fma_resident | path#147: gw[14] += go[54]*x[3]*y[8], gx[3] += w[14]*go[54]*y[8], gy[8] += w[14]*go[54]*x[3]
            gw_acc_i_14 += scalar_t(-0.18257418583505522) * r10 * r8 * r9;
            gx_acc_j_3 += scalar_t(-0.18257418583505522) * (r7 * r10) * r9;
            gy_acc_k_8 += scalar_t(-0.18257418583505522) * (r7 * r10) * r8;
            // inst 23: load | global key=(1, 0, (0, 1, 8, 0, -22, 13), "('y', 6)")
            r11 = y[y_base + (index_t)6];
            // inst 24: bwd_fma_resident | path#146: gw[14] += go[54]*x[3]*y[6], gx[3] += w[14]*go[54]*y[6], gy[6] += w[14]*go[54]*x[3]
            gw_acc_i_14 += scalar_t(0.6324555320336759) * r10 * r8 * r11;
            gx_acc_j_3 += scalar_t(0.6324555320336759) * (r7 * r10) * r11;
            gy_acc_k_6 += scalar_t(0.6324555320336759) * (r7 * r10) * r8;
            // inst 25: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 20)")
            r12 = grad_out[go_base + (index_t)4480 + (index_t)u];
            // inst 26: bwd_fma_resident | path#60: gw[8] += go[20]*x[0]*y[6], gx[0] += w[8]*go[20]*y[6], gy[6] += w[8]*go[20]*x[0]
            gw_acc_i_8 += scalar_t(0.9999999999999997) * r12 * r3 * r11;
            gx_acc_j_0 += scalar_t(0.9999999999999997) * (r0 * r12) * r11;
            gy_acc_k_6 += scalar_t(0.9999999999999997) * (r0 * r12) * r3;
            // inst 27: release | last use after path#60
            // inst 28: load | global key=(1, 0, (0, 1, 4, 0, -2, 2), "('go', 56)")
            r12 = grad_out[go_base + (index_t)12544 + (index_t)u];
            // inst 29: bwd_fma_resident | path#152: gw[14] += go[56]*x[3]*y[8], gx[3] += w[14]*go[56]*y[8], gy[8] += w[14]*go[56]*x[3]
            gw_acc_i_14 += scalar_t(0.7071067811865485) * r12 * r8 * r9;
            gx_acc_j_3 += scalar_t(0.7071067811865485) * (r7 * r12) * r9;
            gy_acc_k_8 += scalar_t(0.7071067811865485) * (r7 * r12) * r8;
            // inst 30: load | global key=(0, 0, (0, 0, 17, 0, -36, 26), "('x', 1)")
            r13 = x[x_base + (index_t)224 + (index_t)u];
            // inst 31: load | global key=(2, 1, (2, 2, 15, 0, -19, 13), "('y', 4)")
            r14 = y[y_base + (index_t)4];
            // inst 32: bwd_fma_resident | path#151: gw[14] += go[56]*x[1]*y[4], gx[1] += w[14]*go[56]*y[4], gy[4] += w[14]*go[56]*x[1]
            gw_acc_i_14 += scalar_t(-0.7071067811865477) * r12 * r13 * r14;
            gx_acc_j_1 += scalar_t(-0.7071067811865477) * (r7 * r12) * r14;
            gy_acc_k_4 += scalar_t(-0.7071067811865477) * (r7 * r12) * r13;
            // inst 33: release | last use after path#151
            // inst 34: bwd_fma_resident | path#144: gw[14] += go[54]*x[1]*y[4], gx[1] += w[14]*go[54]*y[4], gy[4] += w[14]*go[54]*x[1]
            gw_acc_i_14 += scalar_t(-0.1825741858350552) * r10 * r13 * r14;
            gx_acc_j_1 += scalar_t(-0.1825741858350552) * (r7 * r10) * r14;
            gy_acc_k_4 += scalar_t(-0.1825741858350552) * (r7 * r10) * r13;
            // inst 35: load | global key=(3, 0, (0, 3, 10, 0, -2, 4), "('go', 52)")
            r12 = grad_out[go_base + (index_t)11648 + (index_t)u];
            // inst 36: bwd_fma_resident | path#137: gw[14] += go[52]*x[1]*y[6], gx[1] += w[14]*go[52]*y[6], gy[6] += w[14]*go[52]*x[1]
            gw_acc_i_14 += scalar_t(0.6324555320336759) * r12 * r13 * r11;
            gx_acc_j_1 += scalar_t(0.6324555320336759) * (r7 * r12) * r11;
            gy_acc_k_6 += scalar_t(0.6324555320336759) * (r7 * r12) * r13;
            // inst 37: bwd_fma_resident | path#138: gw[14] += go[52]*x[1]*y[8], gx[1] += w[14]*go[52]*y[8], gy[8] += w[14]*go[52]*x[1]
            gw_acc_i_14 += scalar_t(0.182574185835055) * r12 * r13 * r9;
            gx_acc_j_1 += scalar_t(0.182574185835055) * (r7 * r12) * r9;
            gy_acc_k_8 += scalar_t(0.182574185835055) * (r7 * r12) * r13;
            // inst 38: bwd_fma_resident | path#140: gw[14] += go[52]*x[3]*y[4], gx[3] += w[14]*go[52]*y[4], gy[4] += w[14]*go[52]*x[3]
            gw_acc_i_14 += scalar_t(-0.18257418583505522) * r12 * r8 * r14;
            gx_acc_j_3 += scalar_t(-0.18257418583505522) * (r7 * r12) * r14;
            gy_acc_k_4 += scalar_t(-0.18257418583505522) * (r7 * r12) * r8;
            // inst 39: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 50)")
            r15 = grad_out[go_base + (index_t)11200 + (index_t)u];
            // inst 40: bwd_fma_resident | path#132: gw[14] += go[50]*x[1]*y[8], gx[1] += w[14]*go[50]*y[8], gy[8] += w[14]*go[50]*x[1]
            gw_acc_i_14 += scalar_t(0.7071067811865476) * r15 * r13 * r9;
            gx_acc_j_1 += scalar_t(0.7071067811865476) * (r7 * r15) * r9;
            gy_acc_k_8 += scalar_t(0.7071067811865476) * (r7 * r15) * r13;
            // inst 41: bwd_fma_resident | path#133: gw[14] += go[50]*x[3]*y[4], gx[3] += w[14]*go[50]*y[4], gy[4] += w[14]*go[50]*x[3]
            gw_acc_i_14 += scalar_t(0.7071067811865482) * r15 * r8 * r14;
            gx_acc_j_3 += scalar_t(0.7071067811865482) * (r7 * r15) * r14;
            gy_acc_k_4 += scalar_t(0.7071067811865482) * (r7 * r15) * r8;
            // inst 42: release | last use after path#133
            // inst 43: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 6)")
            r15 = grad_out[go_base + (index_t)1344 + (index_t)u];
            // inst 44: bwd_fma_resident | path#12: gw[4] += go[6]*x[1]*y[0], gx[1] += w[4]*go[6]*y[0], gy[0] += w[4]*go[6]*x[1]
            gw_acc_i_4 += scalar_t(1.0000000000000002) * r15 * r13 * r2;
            gx_acc_j_1 += scalar_t(1.0000000000000002) * (r5 * r15) * r2;
            gy_acc_k_0 += scalar_t(1.0000000000000002) * (r5 * r15) * r13;
            // inst 45: release | last use after path#12
            // inst 46: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 18)")
            r15 = grad_out[go_base + (index_t)4032 + (index_t)u];
            // inst 47: bwd_fma_resident | path#58: gw[8] += go[18]*x[0]*y[4], gx[0] += w[8]*go[18]*y[4], gy[4] += w[8]*go[18]*x[0]
            gw_acc_i_8 += scalar_t(0.9999999999999993) * r15 * r3 * r14;
            gx_acc_j_0 += scalar_t(0.9999999999999993) * (r0 * r15) * r14;
            gy_acc_k_4 += scalar_t(0.9999999999999993) * (r0 * r15) * r3;
            // inst 48: release | last use after path#58
            // inst 49: load | global key=(0, 0, (0, 0, 16, 0, -13, 21), "('w', 10)")
            r15 = w[w_base + (index_t)2240 + (index_t)u];
            // inst 50: load | global key=(0, 0, (0, 0, 18, 0, -29, 18), "('x', 2)")
            r16 = x[x_base + (index_t)448 + (index_t)u];
            // inst 51: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 7)")
            r17 = grad_out[go_base + (index_t)1568 + (index_t)u];
            // inst 52: bwd_fma_resident | path#13: gw[4] += go[7]*x[2]*y[0], gx[2] += w[4]*go[7]*y[0], gy[0] += w[4]*go[7]*x[2]
            gw_acc_i_4 += scalar_t(0.9999999999999998) * r17 * r16 * r2;
            gx_acc_j_2 += scalar_t(0.9999999999999998) * (r5 * r17) * r2;
            gy_acc_k_0 += scalar_t(0.9999999999999998) * (r5 * r17) * r16;
            // inst 53: release | last use after path#13
            // inst 54: release | last use after path#13
            // inst 55: load | global key=(1, 1, (2, 1, 13, 0, -20, 14), "('y', 7)")
            r17 = y[y_base + (index_t)7];
            // inst 56: bwd_fma_resident | path#145: gw[14] += go[54]*x[2]*y[7], gx[2] += w[14]*go[54]*y[7], gy[7] += w[14]*go[54]*x[2]
            gw_acc_i_14 += scalar_t(0.7302967433402209) * r10 * r16 * r17;
            gx_acc_j_2 += scalar_t(0.7302967433402209) * (r7 * r10) * r17;
            gy_acc_k_7 += scalar_t(0.7302967433402209) * (r7 * r10) * r16;
            // inst 57: release | last use after path#145
            // inst 58: load | global key=(2, 0, (0, 2, 8, 0, -1, 3), "('go', 55)")
            r10 = grad_out[go_base + (index_t)12320 + (index_t)u];
            // inst 59: bwd_fma_resident | path#150: gw[14] += go[55]*x[3]*y[7], gx[3] += w[14]*go[55]*y[7], gy[7] += w[14]*go[55]*x[3]
            gw_acc_i_14 += scalar_t(0.5773502691896263) * r10 * r8 * r17;
            gx_acc_j_3 += scalar_t(0.5773502691896263) * (r7 * r10) * r17;
            gy_acc_k_7 += scalar_t(0.5773502691896263) * (r7 * r10) * r8;
            // inst 60: bwd_fma_resident | path#149: gw[14] += go[55]*x[2]*y[8], gx[2] += w[14]*go[55]*y[8], gy[8] += w[14]*go[55]*x[2]
            gw_acc_i_14 += scalar_t(0.5773502691896257) * r10 * r16 * r9;
            gx_acc_j_2 += scalar_t(0.5773502691896257) * (r7 * r10) * r9;
            gy_acc_k_8 += scalar_t(0.5773502691896257) * (r7 * r10) * r16;
            // inst 61: load | global key=(2, 1, (4, 2, 14, 0, -19, 14), "('y', 5)")
            r5 = y[y_base + (index_t)5];
            // inst 62: bwd_fma_resident | path#148: gw[14] += go[55]*x[1]*y[5], gx[1] += w[14]*go[55]*y[5], gy[5] += w[14]*go[55]*x[1]
            gw_acc_i_14 += scalar_t(-0.5773502691896258) * r10 * r13 * r5;
            gx_acc_j_1 += scalar_t(-0.5773502691896258) * (r7 * r10) * r5;
            gy_acc_k_5 += scalar_t(-0.5773502691896258) * (r7 * r10) * r13;
            // inst 63: release | last use after path#148
            // inst 64: bwd_fma_resident | path#139: gw[14] += go[52]*x[2]*y[5], gx[2] += w[14]*go[52]*y[5], gy[5] += w[14]*go[52]*x[2]
            gw_acc_i_14 += scalar_t(0.7302967433402211) * r12 * r16 * r5;
            gx_acc_j_2 += scalar_t(0.7302967433402211) * (r7 * r12) * r5;
            gy_acc_k_5 += scalar_t(0.7302967433402211) * (r7 * r12) * r16;
            // inst 65: release | last use after path#139
            // inst 66: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 53)")
            r12 = grad_out[go_base + (index_t)11872 + (index_t)u];
            // inst 67: bwd_fma_resident | path#141: gw[14] += go[53]*x[1]*y[5], gx[1] += w[14]*go[53]*y[5], gy[5] += w[14]*go[53]*x[1]
            gw_acc_i_14 += scalar_t(-0.4472135954999573) * r12 * r13 * r5;
            gx_acc_j_1 += scalar_t(-0.4472135954999573) * (r7 * r12) * r5;
            gy_acc_k_5 += scalar_t(-0.4472135954999573) * (r7 * r12) * r13;
            // inst 68: bwd_fma_resident | path#143: gw[14] += go[53]*x[3]*y[7], gx[3] += w[14]*go[53]*y[7], gy[7] += w[14]*go[53]*x[3]
            gw_acc_i_14 += scalar_t(-0.44721359549995765) * r12 * r8 * r17;
            gx_acc_j_3 += scalar_t(-0.44721359549995765) * (r7 * r12) * r17;
            gy_acc_k_7 += scalar_t(-0.44721359549995765) * (r7 * r12) * r8;
            // inst 69: bwd_fma_resident | path#142: gw[14] += go[53]*x[2]*y[6], gx[2] += w[14]*go[53]*y[6], gy[6] += w[14]*go[53]*x[2]
            gw_acc_i_14 += scalar_t(0.774596669241483) * r12 * r16 * r11;
            gx_acc_j_2 += scalar_t(0.774596669241483) * (r7 * r12) * r11;
            gy_acc_k_6 += scalar_t(0.774596669241483) * (r7 * r12) * r16;
            // inst 70: release | last use after path#142
            // inst 71: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 51)")
            r12 = grad_out[go_base + (index_t)11424 + (index_t)u];
            // inst 72: bwd_fma_resident | path#134: gw[14] += go[51]*x[1]*y[7], gx[1] += w[14]*go[51]*y[7], gy[7] += w[14]*go[51]*x[1]
            gw_acc_i_14 += scalar_t(0.5773502691896256) * r12 * r13 * r17;
            gx_acc_j_1 += scalar_t(0.5773502691896256) * (r7 * r12) * r17;
            gy_acc_k_7 += scalar_t(0.5773502691896256) * (r7 * r12) * r13;
            // inst 73: bwd_fma_resident | path#136: gw[14] += go[51]*x[3]*y[5], gx[3] += w[14]*go[51]*y[5], gy[5] += w[14]*go[51]*x[3]
            gw_acc_i_14 += scalar_t(0.577350269189626) * r12 * r8 * r5;
            gx_acc_j_3 += scalar_t(0.577350269189626) * (r7 * r12) * r5;
            gy_acc_k_5 += scalar_t(0.577350269189626) * (r7 * r12) * r8;
            // inst 74: bwd_fma_resident | path#135: gw[14] += go[51]*x[2]*y[4], gx[2] += w[14]*go[51]*y[4], gy[4] += w[14]*go[51]*x[2]
            gw_acc_i_14 += scalar_t(0.5773502691896256) * r12 * r16 * r14;
            gx_acc_j_2 += scalar_t(0.5773502691896256) * (r7 * r12) * r14;
            gy_acc_k_4 += scalar_t(0.5773502691896256) * (r7 * r12) * r16;
            // inst 75: release | last use after path#135
            // inst 76: release | last use after path#135
            // inst 77: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 21)")
            r12 = grad_out[go_base + (index_t)4704 + (index_t)u];
            // inst 78: bwd_fma_resident | path#61: gw[8] += go[21]*x[0]*y[7], gx[0] += w[8]*go[21]*y[7], gy[7] += w[8]*go[21]*x[0]
            gw_acc_i_8 += scalar_t(1.0) * r12 * r3 * r17;
            gx_acc_j_0 += scalar_t(1.0) * (r0 * r12) * r17;
            gy_acc_k_7 += scalar_t(1.0) * (r0 * r12) * r3;
            // inst 79: release | last use after path#61
            // inst 80: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 19)")
            r12 = grad_out[go_base + (index_t)4256 + (index_t)u];
            // inst 81: bwd_fma_resident | path#59: gw[8] += go[19]*x[0]*y[5], gx[0] += w[8]*go[19]*y[5], gy[5] += w[8]*go[19]*x[0]
            gw_acc_i_8 += scalar_t(1.0000000000000002) * r12 * r3 * r5;
            gx_acc_j_0 += scalar_t(1.0000000000000002) * (r0 * r12) * r5;
            gy_acc_k_5 += scalar_t(1.0000000000000002) * (r0 * r12) * r3;
            // inst 82: release | last use after path#59
            // inst 83: release | last use after path#59
            // inst 84: load | global key=(0, 0, (0, 0, 25, 0, -10, 25), "('w', 12)")
            r12 = w[w_base + (index_t)2688 + (index_t)u];
            // inst 85: load | global key=(0, 0, (0, 0, 22, 0, -3, 11), "('w', 5)")
            r0 = w[w_base + (index_t)1120 + (index_t)u];
            // inst 86: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 9)")
            r7 = grad_out[go_base + (index_t)2016 + (index_t)u];
            // inst 87: bwd_fma_resident | path#15: gw[5] += go[9]*x[1]*y[6], gx[1] += w[5]*go[9]*y[6], gy[6] += w[5]*go[9]*x[1]
            gw_acc_i_5 += scalar_t(-0.3162277660168376) * r7 * r13 * r11;
            gx_acc_j_1 += scalar_t(-0.3162277660168376) * (r0 * r7) * r11;
            gy_acc_k_6 += scalar_t(-0.3162277660168376) * (r0 * r7) * r13;
            // inst 88: bwd_fma_resident | path#18: gw[5] += go[9]*x[3]*y[4], gx[3] += w[5]*go[9]*y[4], gy[4] += w[5]*go[9]*x[3]
            gw_acc_i_5 += scalar_t(0.5477225575051664) * r7 * r8 * r14;
            gx_acc_j_3 += scalar_t(0.5477225575051664) * (r0 * r7) * r14;
            gy_acc_k_4 += scalar_t(0.5477225575051664) * (r0 * r7) * r8;
            // inst 89: bwd_fma_resident | path#16: gw[5] += go[9]*x[1]*y[8], gx[1] += w[5]*go[9]*y[8], gy[8] += w[5]*go[9]*x[1]
            gw_acc_i_5 += scalar_t(-0.5477225575051664) * r7 * r13 * r9;
            gx_acc_j_1 += scalar_t(-0.5477225575051664) * (r0 * r7) * r9;
            gy_acc_k_8 += scalar_t(-0.5477225575051664) * (r0 * r7) * r13;
            // inst 90: bwd_fma_resident | path#17: gw[5] += go[9]*x[2]*y[5], gx[2] += w[5]*go[9]*y[5], gy[5] += w[5]*go[9]*x[2]
            gw_acc_i_5 += scalar_t(0.5477225575051657) * r7 * r16 * r5;
            gx_acc_j_2 += scalar_t(0.5477225575051657) * (r0 * r7) * r5;
            gy_acc_k_5 += scalar_t(0.5477225575051657) * (r0 * r7) * r16;
            // inst 91: release | last use after path#17
            // inst 92: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 11)")
            r7 = grad_out[go_base + (index_t)2464 + (index_t)u];
            // inst 93: bwd_fma_resident | path#24: gw[5] += go[11]*x[3]*y[6], gx[3] += w[5]*go[11]*y[6], gy[6] += w[5]*go[11]*x[3]
            gw_acc_i_5 += scalar_t(-0.31622776601683794) * r7 * r8 * r11;
            gx_acc_j_3 += scalar_t(-0.31622776601683794) * (r0 * r7) * r11;
            gy_acc_k_6 += scalar_t(-0.31622776601683794) * (r0 * r7) * r8;
            // inst 94: bwd_fma_resident | path#22: gw[5] += go[11]*x[1]*y[4], gx[1] += w[5]*go[11]*y[4], gy[4] += w[5]*go[11]*x[1]
            gw_acc_i_5 += scalar_t(0.5477225575051665) * r7 * r13 * r14;
            gx_acc_j_1 += scalar_t(0.5477225575051665) * (r0 * r7) * r14;
            gy_acc_k_4 += scalar_t(0.5477225575051665) * (r0 * r7) * r13;
            // inst 95: bwd_fma_resident | path#25: gw[5] += go[11]*x[3]*y[8], gx[3] += w[5]*go[11]*y[8], gy[8] += w[5]*go[11]*x[3]
            gw_acc_i_5 += scalar_t(0.5477225575051664) * r7 * r8 * r9;
            gx_acc_j_3 += scalar_t(0.5477225575051664) * (r0 * r7) * r9;
            gy_acc_k_8 += scalar_t(0.5477225575051664) * (r0 * r7) * r8;
            // inst 96: bwd_fma_resident | path#23: gw[5] += go[11]*x[2]*y[7], gx[2] += w[5]*go[11]*y[7], gy[7] += w[5]*go[11]*x[2]
            gw_acc_i_5 += scalar_t(0.5477225575051669) * r7 * r16 * r17;
            gx_acc_j_2 += scalar_t(0.5477225575051669) * (r0 * r7) * r17;
            gy_acc_k_7 += scalar_t(0.5477225575051669) * (r0 * r7) * r16;
            // inst 97: release | last use after path#23
            // inst 98: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 10)")
            r7 = grad_out[go_base + (index_t)2240 + (index_t)u];
            // inst 99: bwd_fma_resident | path#19: gw[5] += go[10]*x[1]*y[5], gx[1] += w[5]*go[10]*y[5], gy[5] += w[5]*go[10]*x[1]
            gw_acc_i_5 += scalar_t(0.5477225575051657) * r7 * r13 * r5;
            gx_acc_j_1 += scalar_t(0.5477225575051657) * (r0 * r7) * r5;
            gy_acc_k_5 += scalar_t(0.5477225575051657) * (r0 * r7) * r13;
            // inst 100: bwd_fma_resident | path#21: gw[5] += go[10]*x[3]*y[7], gx[3] += w[5]*go[10]*y[7], gy[7] += w[5]*go[10]*x[3]
            gw_acc_i_5 += scalar_t(0.5477225575051656) * r7 * r8 * r17;
            gx_acc_j_3 += scalar_t(0.5477225575051656) * (r0 * r7) * r17;
            gy_acc_k_7 += scalar_t(0.5477225575051656) * (r0 * r7) * r8;
            // inst 101: bwd_fma_resident | path#20: gw[5] += go[10]*x[2]*y[6], gx[2] += w[5]*go[10]*y[6], gy[6] += w[5]*go[10]*x[2]
            gw_acc_i_5 += scalar_t(0.6324555320336755) * r7 * r16 * r11;
            gx_acc_j_2 += scalar_t(0.6324555320336755) * (r0 * r7) * r11;
            gy_acc_k_6 += scalar_t(0.6324555320336755) * (r0 * r7) * r16;
            // inst 102: release | last use after path#20
            // inst 103: release | last use after path#20
            // inst 104: load | global key=(0, 0, (0, 0, 15, 0, -38, 28), "('x', 7)")
            r7 = x[x_base + (index_t)1568 + (index_t)u];
            // inst 105: load | global key=(2, 0, (0, 2, 14, 0, -4, 6), "('go', 41)")
            r0 = grad_out[go_base + (index_t)9184 + (index_t)u];
            // inst 106: bwd_fma_resident | path#118: gw[12] += go[41]*x[7]*y[6], gx[7] += w[12]*go[41]*y[6], gy[6] += w[12]*go[41]*x[7]
            gw_acc_i_12 += scalar_t(0.26726124191242395) * r0 * r7 * r11;
            gx_acc_j_7 += scalar_t(0.26726124191242395) * (r12 * r0) * r11;
            gy_acc_k_6 += scalar_t(0.26726124191242395) * (r12 * r0) * r7;
            // inst 107: bwd_fma_resident | path#119: gw[12] += go[41]*x[7]*y[8], gx[7] += w[12]*go[41]*y[8], gy[8] += w[12]*go[41]*x[7]
            gw_acc_i_12 += scalar_t(0.46291004988627527) * r0 * r7 * r9;
            gx_acc_j_7 += scalar_t(0.46291004988627527) * (r12 * r0) * r9;
            gy_acc_k_8 += scalar_t(0.46291004988627527) * (r12 * r0) * r7;
            // inst 108: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 36)")
            r10 = grad_out[go_base + (index_t)8064 + (index_t)u];
            // inst 109: bwd_fma_resident | path#98: gw[11] += go[36]*x[7]*y[0], gx[7] += w[11]*go[36]*y[0], gy[0] += w[11]*go[36]*x[7]
            gw_acc_i_11 += scalar_t(1.0) * r10 * r7 * r2;
            gx_acc_j_7 += scalar_t(1.0) * (r4 * r10) * r2;
            gy_acc_k_0 += scalar_t(1.0) * (r4 * r10) * r7;
            // inst 110: release | last use after path#98
            // inst 111: load | global key=(1, 0, (0, 1, 16, 0, -37, 28), "('x', 5)")
            r10 = x[x_base + (index_t)1120 + (index_t)u];
            // inst 112: bwd_fma_resident | path#116: gw[12] += go[41]*x[5]*y[4], gx[5] += w[12]*go[41]*y[4], gy[4] += w[12]*go[41]*x[5]
            gw_acc_i_12 += scalar_t(0.4629100498862755) * r0 * r10 * r14;
            gx_acc_j_5 += scalar_t(0.4629100498862755) * (r12 * r0) * r14;
            gy_acc_k_4 += scalar_t(0.4629100498862755) * (r12 * r0) * r10;
            // inst 113: load | global key=(3, 0, (0, 3, 15, 0, -3, 6), "('go', 39)")
            r18 = grad_out[go_base + (index_t)8736 + (index_t)u];
            // inst 114: bwd_fma_resident | path#105: gw[12] += go[39]*x[5]*y[6], gx[5] += w[12]*go[39]*y[6], gy[6] += w[12]*go[39]*x[5]
            gw_acc_i_12 += scalar_t(0.2672612419124244) * r18 * r10 * r11;
            gx_acc_j_5 += scalar_t(0.2672612419124244) * (r12 * r18) * r11;
            gy_acc_k_6 += scalar_t(0.2672612419124244) * (r12 * r18) * r10;
            // inst 115: bwd_fma_resident | path#106: gw[12] += go[39]*x[5]*y[8], gx[5] += w[12]*go[39]*y[8], gy[8] += w[12]*go[39]*x[5]
            gw_acc_i_12 += scalar_t(-0.4629100498862753) * r18 * r10 * r9;
            gx_acc_j_5 += scalar_t(-0.4629100498862753) * (r12 * r18) * r9;
            gy_acc_k_8 += scalar_t(-0.4629100498862753) * (r12 * r18) * r10;
            // inst 116: bwd_fma_resident | path#108: gw[12] += go[39]*x[7]*y[4], gx[7] += w[12]*go[39]*y[4], gy[4] += w[12]*go[39]*x[7]
            gw_acc_i_12 += scalar_t(0.4629100498862756) * r18 * r7 * r14;
            gx_acc_j_7 += scalar_t(0.4629100498862756) * (r12 * r18) * r14;
            gy_acc_k_4 += scalar_t(0.4629100498862756) * (r12 * r18) * r7;
            // inst 117: load | global key=(2, 0, (0, 2, 15, 0, -32, 21), "('x', 6)")
            r19 = x[x_base + (index_t)1344 + (index_t)u];
            // inst 118: bwd_fma_resident | path#107: gw[12] += go[39]*x[6]*y[5], gx[6] += w[12]*go[39]*y[5], gy[5] += w[12]*go[39]*x[6]
            gw_acc_i_12 += scalar_t(0.26726124191242423) * r18 * r19 * r5;
            gx_acc_j_6 += scalar_t(0.26726124191242423) * (r12 * r18) * r5;
            gy_acc_k_5 += scalar_t(0.26726124191242423) * (r12 * r18) * r19;
            // inst 119: bwd_fma_resident | path#117: gw[12] += go[41]*x[6]*y[7], gx[6] += w[12]*go[41]*y[7], gy[7] += w[12]*go[41]*x[6]
            gw_acc_i_12 += scalar_t(0.26726124191242445) * r0 * r19 * r17;
            gx_acc_j_6 += scalar_t(0.26726124191242445) * (r12 * r0) * r17;
            gy_acc_k_7 += scalar_t(0.26726124191242445) * (r12 * r0) * r19;
            // inst 120: load | global key=(3, 0, (0, 3, 13, 0, -2, 5), "('go', 40)")
            r20 = grad_out[go_base + (index_t)8960 + (index_t)u];
            // inst 121: bwd_fma_resident | path#111: gw[12] += go[40]*x[5]*y[5], gx[5] += w[12]*go[40]*y[5], gy[5] += w[12]*go[40]*x[5]
            gw_acc_i_12 += scalar_t(0.26726124191242484) * r20 * r10 * r5;
            gx_acc_j_5 += scalar_t(0.26726124191242484) * (r12 * r20) * r5;
            gy_acc_k_5 += scalar_t(0.26726124191242484) * (r12 * r20) * r10;
            // inst 122: bwd_fma_resident | path#113: gw[12] += go[40]*x[7]*y[7], gx[7] += w[12]*go[40]*y[7], gy[7] += w[12]*go[40]*x[7]
            gw_acc_i_12 += scalar_t(0.2672612419124247) * r20 * r7 * r17;
            gx_acc_j_7 += scalar_t(0.2672612419124247) * (r12 * r20) * r17;
            gy_acc_k_7 += scalar_t(0.2672612419124247) * (r12 * r20) * r7;
            // inst 123: bwd_fma_resident | path#112: gw[12] += go[40]*x[6]*y[6], gx[6] += w[12]*go[40]*y[6], gy[6] += w[12]*go[40]*x[6]
            gw_acc_i_12 += scalar_t(0.5345224838248488) * r20 * r19 * r11;
            gx_acc_j_6 += scalar_t(0.5345224838248488) * (r12 * r20) * r11;
            gy_acc_k_6 += scalar_t(0.5345224838248488) * (r12 * r20) * r19;
            // inst 124: load | global key=(3, 0, (0, 3, 14, 0, -33, 26), "('x', 8)")
            r21 = x[x_base + (index_t)1792 + (index_t)u];
            // inst 125: bwd_fma_resident | path#109: gw[12] += go[39]*x[8]*y[5], gx[8] += w[12]*go[39]*y[5], gy[5] += w[12]*go[39]*x[8]
            gw_acc_i_12 += scalar_t(-0.4629100498862755) * r18 * r21 * r5;
            gx_acc_j_8 += scalar_t(-0.4629100498862755) * (r12 * r18) * r5;
            gy_acc_k_5 += scalar_t(-0.4629100498862755) * (r12 * r18) * r21;
            // inst 126: bwd_fma_resident | path#120: gw[12] += go[41]*x[8]*y[7], gx[8] += w[12]*go[41]*y[7], gy[7] += w[12]*go[41]*x[8]
            gw_acc_i_12 += scalar_t(0.46291004988627565) * r0 * r21 * r17;
            gx_acc_j_8 += scalar_t(0.46291004988627565) * (r12 * r0) * r17;
            gy_acc_k_7 += scalar_t(0.46291004988627565) * (r12 * r0) * r21;
            // inst 127: bwd_fma_resident | path#114: gw[12] += go[40]*x[8]*y[8], gx[8] += w[12]*go[40]*y[8], gy[8] += w[12]*go[40]*x[8]
            gw_acc_i_12 += scalar_t(-0.5345224838248485) * r20 * r21 * r9;
            gx_acc_j_8 += scalar_t(-0.5345224838248485) * (r12 * r20) * r9;
            gy_acc_k_8 += scalar_t(-0.5345224838248485) * (r12 * r20) * r21;
            // inst 128: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 42)")
            r22 = grad_out[go_base + (index_t)9408 + (index_t)u];
            // inst 129: bwd_fma_resident | path#121: gw[12] += go[42]*x[5]*y[5], gx[5] += w[12]*go[42]*y[5], gy[5] += w[12]*go[42]*x[5]
            gw_acc_i_12 += scalar_t(-0.46291004988627593) * r22 * r10 * r5;
            gx_acc_j_5 += scalar_t(-0.46291004988627593) * (r12 * r22) * r5;
            gy_acc_k_5 += scalar_t(-0.46291004988627593) * (r12 * r22) * r10;
            // inst 130: bwd_fma_resident | path#123: gw[12] += go[42]*x[7]*y[7], gx[7] += w[12]*go[42]*y[7], gy[7] += w[12]*go[42]*x[7]
            gw_acc_i_12 += scalar_t(0.4629100498862761) * r22 * r7 * r17;
            gx_acc_j_7 += scalar_t(0.4629100498862761) * (r12 * r22) * r17;
            gy_acc_k_7 += scalar_t(0.4629100498862761) * (r12 * r22) * r7;
            // inst 131: bwd_fma_resident | path#124: gw[12] += go[42]*x[8]*y[6], gx[8] += w[12]*go[42]*y[6], gy[6] += w[12]*go[42]*x[8]
            gw_acc_i_12 += scalar_t(-0.5345224838248492) * r22 * r21 * r11;
            gx_acc_j_8 += scalar_t(-0.5345224838248492) * (r12 * r22) * r11;
            gy_acc_k_6 += scalar_t(-0.5345224838248492) * (r12 * r22) * r21;
            // inst 132: bwd_fma_resident | path#122: gw[12] += go[42]*x[6]*y[8], gx[6] += w[12]*go[42]*y[8], gy[8] += w[12]*go[42]*x[6]
            gw_acc_i_12 += scalar_t(-0.5345224838248491) * r22 * r19 * r9;
            gx_acc_j_6 += scalar_t(-0.5345224838248491) * (r12 * r22) * r9;
            gy_acc_k_8 += scalar_t(-0.5345224838248491) * (r12 * r22) * r19;
            // inst 133: release | last use after path#122
            // inst 134: load | global key=(3, 1, (6, 3, 14, 0, -33, 26), "('x', 4)")
            r22 = x[x_base + (index_t)896 + (index_t)u];
            // inst 135: bwd_fma_resident | path#104: gw[12] += go[39]*x[4]*y[7], gx[4] += w[12]*go[39]*y[7], gy[7] += w[12]*go[39]*x[4]
            gw_acc_i_12 += scalar_t(0.4629100498862763) * r18 * r22 * r17;
            gx_acc_j_4 += scalar_t(0.4629100498862763) * (r12 * r18) * r17;
            gy_acc_k_7 += scalar_t(0.4629100498862763) * (r12 * r18) * r22;
            // inst 136: release | last use after path#104
            // inst 137: bwd_fma_resident | path#110: gw[12] += go[40]*x[4]*y[4], gx[4] += w[12]*go[40]*y[4], gy[4] += w[12]*go[40]*x[4]
            gw_acc_i_12 += scalar_t(-0.5345224838248482) * r20 * r22 * r14;
            gx_acc_j_4 += scalar_t(-0.5345224838248482) * (r12 * r20) * r14;
            gy_acc_k_4 += scalar_t(-0.5345224838248482) * (r12 * r20) * r22;
            // inst 138: release | last use after path#110
            // inst 139: bwd_fma_resident | path#115: gw[12] += go[41]*x[4]*y[5], gx[4] += w[12]*go[41]*y[5], gy[5] += w[12]*go[41]*x[4]
            gw_acc_i_12 += scalar_t(0.4629100498862755) * r0 * r22 * r5;
            gx_acc_j_4 += scalar_t(0.4629100498862755) * (r12 * r0) * r5;
            gy_acc_k_5 += scalar_t(0.4629100498862755) * (r12 * r0) * r22;
            // inst 140: release | last use after path#115
            // inst 141: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 38)")
            r0 = grad_out[go_base + (index_t)8512 + (index_t)u];
            // inst 142: bwd_fma_resident | path#100: gw[12] += go[38]*x[4]*y[6], gx[4] += w[12]*go[38]*y[6], gy[6] += w[12]*go[38]*x[4]
            gw_acc_i_12 += scalar_t(-0.5345224838248491) * r0 * r22 * r11;
            gx_acc_j_4 += scalar_t(-0.5345224838248491) * (r12 * r0) * r11;
            gy_acc_k_6 += scalar_t(-0.5345224838248491) * (r12 * r0) * r22;
            // inst 143: bwd_fma_resident | path#101: gw[12] += go[38]*x[5]*y[7], gx[5] += w[12]*go[38]*y[7], gy[7] += w[12]*go[38]*x[5]
            gw_acc_i_12 += scalar_t(0.46291004988627604) * r0 * r10 * r17;
            gx_acc_j_5 += scalar_t(0.46291004988627604) * (r12 * r0) * r17;
            gy_acc_k_7 += scalar_t(0.46291004988627604) * (r12 * r0) * r10;
            // inst 144: bwd_fma_resident | path#103: gw[12] += go[38]*x[7]*y[5], gx[7] += w[12]*go[38]*y[5], gy[5] += w[12]*go[38]*x[7]
            gw_acc_i_12 += scalar_t(0.46291004988627577) * r0 * r7 * r5;
            gx_acc_j_7 += scalar_t(0.46291004988627577) * (r12 * r0) * r5;
            gy_acc_k_5 += scalar_t(0.46291004988627577) * (r12 * r0) * r7;
            // inst 145: bwd_fma_resident | path#102: gw[12] += go[38]*x[6]*y[4], gx[6] += w[12]*go[38]*y[4], gy[4] += w[12]*go[38]*x[6]
            gw_acc_i_12 += scalar_t(-0.5345224838248488) * r0 * r19 * r14;
            gx_acc_j_6 += scalar_t(-0.5345224838248488) * (r12 * r0) * r14;
            gy_acc_k_4 += scalar_t(-0.5345224838248488) * (r12 * r0) * r19;
            // inst 146: release | last use after path#102
            // inst 147: release | last use after path#102
            // inst 148: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 37)")
            r0 = grad_out[go_base + (index_t)8288 + (index_t)u];
            // inst 149: bwd_fma_resident | path#99: gw[11] += go[37]*x[8]*y[0], gx[8] += w[11]*go[37]*y[0], gy[0] += w[11]*go[37]*x[8]
            gw_acc_i_11 += scalar_t(1.0000000000000004) * r0 * r21 * r2;
            gx_acc_j_8 += scalar_t(1.0000000000000004) * (r4 * r0) * r2;
            gy_acc_k_0 += scalar_t(1.0000000000000004) * (r4 * r0) * r21;
            // inst 150: release | last use after path#99
            // inst 151: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 35)")
            r0 = grad_out[go_base + (index_t)7840 + (index_t)u];
            // inst 152: bwd_fma_resident | path#97: gw[11] += go[35]*x[6]*y[0], gx[6] += w[11]*go[35]*y[0], gy[0] += w[11]*go[35]*x[6]
            gw_acc_i_11 += scalar_t(0.9999999999999997) * r0 * r19 * r2;
            gx_acc_j_6 += scalar_t(0.9999999999999997) * (r4 * r0) * r2;
            gy_acc_k_0 += scalar_t(0.9999999999999997) * (r4 * r0) * r19;
            // inst 153: release | last use after path#97
            // inst 154: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 34)")
            r0 = grad_out[go_base + (index_t)7616 + (index_t)u];
            // inst 155: bwd_fma_resident | path#96: gw[11] += go[34]*x[5]*y[0], gx[5] += w[11]*go[34]*y[0], gy[0] += w[11]*go[34]*x[5]
            gw_acc_i_11 += scalar_t(1.0000000000000002) * r0 * r10 * r2;
            gx_acc_j_5 += scalar_t(1.0000000000000002) * (r4 * r0) * r2;
            gy_acc_k_0 += scalar_t(1.0000000000000002) * (r4 * r0) * r10;
            // inst 156: release | last use after path#96
            // inst 157: load | global key=(1, 3, (5, 1, 3, 0, 0, 1), "('go', 33)")
            r0 = grad_out[go_base + (index_t)7392 + (index_t)u];
            // inst 158: bwd_fma_resident | path#95: gw[11] += go[33]*x[4]*y[0], gx[4] += w[11]*go[33]*y[0], gy[0] += w[11]*go[33]*x[4]
            gw_acc_i_11 += scalar_t(0.9999999999999993) * r0 * r22 * r2;
            gx_acc_j_4 += scalar_t(0.9999999999999993) * (r4 * r0) * r2;
            gy_acc_k_0 += scalar_t(0.9999999999999993) * (r4 * r0) * r22;
            // inst 159: release | last use after path#95
            // inst 160: release | last use after path#95
            // inst 161: release | last use after path#95
            // inst 162: load | global key=(0, 0, (0, 0, 41, 0, -14, 41), "('w', 16)")
            r0 = w[w_base + (index_t)3584 + (index_t)u];
            // inst 163: load | global key=(0, 0, (0, 0, 30, 0, -16, 17), "('y', 13)")
            r2 = y[y_base + (index_t)13];
            // inst 164: load | global key=(2, 0, (0, 2, 18, 0, -6, 8), "('go', 68)")
            r4 = grad_out[go_base + (index_t)15232 + (index_t)u];
            // inst 165: bwd_fma_resident | path#203: gw[16] += go[68]*x[8]*y[13], gx[8] += w[16]*go[68]*y[13], gy[13] += w[16]*go[68]*x[8]
            gw_acc_i_16 += scalar_t(0.4472135954999584) * r4 * r21 * r2;
            gx_acc_j_8 += scalar_t(0.4472135954999584) * (r0 * r4) * r2;
            gy_acc_k_13 += scalar_t(0.4472135954999584) * (r0 * r4) * r21;
            // inst 166: bwd_fma_resident | path#200: gw[16] += go[68]*x[6]*y[13], gx[6] += w[16]*go[68]*y[13], gy[13] += w[16]*go[68]*x[6]
            gw_acc_i_16 += scalar_t(0.38729833462074215) * r4 * r19 * r2;
            gx_acc_j_6 += scalar_t(0.38729833462074215) * (r0 * r4) * r2;
            gy_acc_k_13 += scalar_t(0.38729833462074215) * (r0 * r4) * r19;
            // inst 167: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 47)")
            r12 = grad_out[go_base + (index_t)10528 + (index_t)u];
            // inst 168: bwd_fma_resident | path#129: gw[13] += go[47]*x[0]*y[13], gx[0] += w[13]*go[47]*y[13], gy[13] += w[13]*go[47]*x[0]
            gw_acc_i_13 += scalar_t(0.9999999999999996) * r12 * r3 * r2;
            gx_acc_j_0 += scalar_t(0.9999999999999996) * (r1 * r12) * r2;
            gy_acc_k_13 += scalar_t(0.9999999999999996) * (r1 * r12) * r3;
            // inst 169: release | last use after path#129
            // inst 170: load | global key=(1, 0, (0, 1, 31, 0, -15, 17), "('y', 11)")
            r12 = y[y_base + (index_t)11];
            // inst 171: bwd_fma_resident | path#198: gw[16] += go[68]*x[4]*y[11], gx[4] += w[16]*go[68]*y[11], gy[11] += w[16]*go[68]*x[4]
            gw_acc_i_16 += scalar_t(0.44721359549995815) * r4 * r22 * r12;
            gx_acc_j_4 += scalar_t(0.44721359549995815) * (r0 * r4) * r12;
            gy_acc_k_11 += scalar_t(0.44721359549995815) * (r0 * r4) * r22;
            // inst 172: load | global key=(3, 0, (0, 3, 19, 0, -5, 8), "('go', 66)")
            r20 = grad_out[go_base + (index_t)14784 + (index_t)u];
            // inst 173: bwd_fma_resident | path#191: gw[16] += go[66]*x[8]*y[11], gx[8] += w[16]*go[66]*y[11], gy[11] += w[16]*go[66]*x[8]
            gw_acc_i_16 += scalar_t(-0.4472135954999582) * r20 * r21 * r12;
            gx_acc_j_8 += scalar_t(-0.4472135954999582) * (r0 * r20) * r12;
            gy_acc_k_11 += scalar_t(-0.4472135954999582) * (r0 * r20) * r21;
            // inst 174: bwd_fma_resident | path#184: gw[16] += go[66]*x[4]*y[13], gx[4] += w[16]*go[66]*y[13], gy[13] += w[16]*go[66]*x[4]
            gw_acc_i_16 += scalar_t(0.44721359549995765) * r20 * r22 * r2;
            gx_acc_j_4 += scalar_t(0.44721359549995765) * (r0 * r20) * r2;
            gy_acc_k_13 += scalar_t(0.44721359549995765) * (r0 * r20) * r22;
            // inst 175: bwd_fma_resident | path#188: gw[16] += go[66]*x[6]*y[11], gx[6] += w[16]*go[66]*y[11], gy[11] += w[16]*go[66]*x[6]
            gw_acc_i_16 += scalar_t(0.3872983346207423) * r20 * r19 * r12;
            gx_acc_j_6 += scalar_t(0.3872983346207423) * (r0 * r20) * r12;
            gy_acc_k_11 += scalar_t(0.3872983346207423) * (r0 * r20) * r19;
            // inst 176: load | global key=(2, 0, (0, 2, 23, 0, -11, 12), "('y', 14)")
            r18 = y[y_base + (index_t)14];
            // inst 177: bwd_fma_resident | path#187: gw[16] += go[66]*x[5]*y[14], gx[5] += w[16]*go[66]*y[14], gy[14] += w[16]*go[66]*x[5]
            gw_acc_i_16 += scalar_t(-0.35355339059327345) * r20 * r10 * r18;
            gx_acc_j_5 += scalar_t(-0.35355339059327345) * (r0 * r20) * r18;
            gy_acc_k_14 += scalar_t(-0.35355339059327345) * (r0 * r20) * r10;
            // inst 178: bwd_fma_resident | path#202: gw[16] += go[68]*x[7]*y[14], gx[7] += w[16]*go[68]*y[14], gy[14] += w[16]*go[68]*x[7]
            gw_acc_i_16 += scalar_t(0.35355339059327356) * r4 * r7 * r18;
            gx_acc_j_7 += scalar_t(0.35355339059327356) * (r0 * r4) * r18;
            gy_acc_k_14 += scalar_t(0.35355339059327356) * (r0 * r4) * r7;
            // inst 179: load | global key=(3, 0, (0, 3, 13, 0, -2, 5), "('go', 70)")
            r23 = grad_out[go_base + (index_t)15680 + (index_t)u];
            // inst 180: bwd_fma_resident | path#210: gw[16] += go[70]*x[4]*y[11], gx[4] += w[16]*go[70]*y[11], gy[11] += w[16]*go[70]*x[4]
            gw_acc_i_16 += scalar_t(0.2886751345948129) * r23 * r22 * r12;
            gx_acc_j_4 += scalar_t(0.2886751345948129) * (r0 * r23) * r12;
            gy_acc_k_11 += scalar_t(0.2886751345948129) * (r0 * r23) * r22;
            // inst 181: bwd_fma_resident | path#214: gw[16] += go[70]*x[8]*y[13], gx[8] += w[16]*go[70]*y[13], gy[13] += w[16]*go[70]*x[8]
            gw_acc_i_16 += scalar_t(-0.28867513459481314) * r23 * r21 * r2;
            gx_acc_j_8 += scalar_t(-0.28867513459481314) * (r0 * r23) * r2;
            gy_acc_k_13 += scalar_t(-0.28867513459481314) * (r0 * r23) * r21;
            // inst 182: bwd_fma_resident | path#213: gw[16] += go[70]*x[7]*y[14], gx[7] += w[16]*go[70]*y[14], gy[14] += w[16]*go[70]*x[7]
            gw_acc_i_16 += scalar_t(0.45643546458763906) * r23 * r7 * r18;
            gx_acc_j_7 += scalar_t(0.45643546458763906) * (r0 * r23) * r18;
            gy_acc_k_14 += scalar_t(0.45643546458763906) * (r0 * r23) * r7;
            // inst 183: load | global key=(3, 0, (0, 3, 24, 0, -10, 12), "('y', 10)")
            r24 = y[y_base + (index_t)10];
            // inst 184: bwd_fma_resident | path#199: gw[16] += go[68]*x[5]*y[10], gx[5] += w[16]*go[68]*y[10], gy[10] += w[16]*go[68]*x[5]
            gw_acc_i_16 += scalar_t(0.35355339059327423) * r4 * r10 * r24;
            gx_acc_j_5 += scalar_t(0.35355339059327423) * (r0 * r4) * r24;
            gy_acc_k_10 += scalar_t(0.35355339059327423) * (r0 * r4) * r10;
            // inst 185: bwd_fma_resident | path#189: gw[16] += go[66]*x[7]*y[10], gx[7] += w[16]*go[66]*y[10], gy[10] += w[16]*go[66]*x[7]
            gw_acc_i_16 += scalar_t(0.35355339059327323) * r20 * r7 * r24;
            gx_acc_j_7 += scalar_t(0.35355339059327323) * (r0 * r20) * r24;
            gy_acc_k_10 += scalar_t(0.35355339059327323) * (r0 * r20) * r7;
            // inst 186: bwd_fma_resident | path#211: gw[16] += go[70]*x[5]*y[10], gx[5] += w[16]*go[70]*y[10], gy[10] += w[16]*go[70]*x[5]
            gw_acc_i_16 += scalar_t(-0.4564354645876387) * r23 * r10 * r24;
            gx_acc_j_5 += scalar_t(-0.4564354645876387) * (r0 * r23) * r24;
            gy_acc_k_10 += scalar_t(-0.4564354645876387) * (r0 * r23) * r10;
            // inst 187: load | global key=(4, 0, (0, 4, 14, 0, -1, 5), "('go', 67)")
            r25 = grad_out[go_base + (index_t)15008 + (index_t)u];
            // inst 188: bwd_fma_resident | path#193: gw[16] += go[67]*x[5]*y[11], gx[5] += w[16]*go[67]*y[11], gy[11] += w[16]*go[67]*x[5]
            gw_acc_i_16 += scalar_t(0.18257418583505508) * r25 * r10 * r12;
            gx_acc_j_5 += scalar_t(0.18257418583505508) * (r0 * r25) * r12;
            gy_acc_k_11 += scalar_t(0.18257418583505508) * (r0 * r25) * r10;
            // inst 189: bwd_fma_resident | path#195: gw[16] += go[67]*x[7]*y[13], gx[7] += w[16]*go[67]*y[13], gy[13] += w[16]*go[67]*x[7]
            gw_acc_i_16 += scalar_t(0.18257418583505516) * r25 * r7 * r2;
            gx_acc_j_7 += scalar_t(0.18257418583505516) * (r0 * r25) * r2;
            gy_acc_k_13 += scalar_t(0.18257418583505516) * (r0 * r25) * r7;
            // inst 190: bwd_fma_resident | path#192: gw[16] += go[67]*x[4]*y[10], gx[4] += w[16]*go[67]*y[10], gy[10] += w[16]*go[67]*x[4]
            gw_acc_i_16 += scalar_t(-0.5773502691896257) * r25 * r22 * r24;
            gx_acc_j_4 += scalar_t(-0.5773502691896257) * (r0 * r25) * r24;
            gy_acc_k_10 += scalar_t(-0.5773502691896257) * (r0 * r25) * r22;
            // inst 191: bwd_fma_resident | path#196: gw[16] += go[67]*x[8]*y[14], gx[8] += w[16]*go[67]*y[14], gy[14] += w[16]*go[67]*x[8]
            gw_acc_i_16 += scalar_t(-0.5773502691896264) * r25 * r21 * r18;
            gx_acc_j_8 += scalar_t(-0.5773502691896264) * (r0 * r25) * r18;
            gy_acc_k_14 += scalar_t(-0.5773502691896264) * (r0 * r25) * r21;
            // inst 192: load | global key=(4, 0, (0, 4, 14, 0, -1, 5), "('go', 64)")
            r26 = grad_out[go_base + (index_t)14336 + (index_t)u];
            // inst 193: bwd_fma_resident | path#178: gw[16] += go[64]*x[8]*y[11], gx[8] += w[16]*go[64]*y[11], gy[11] += w[16]*go[64]*x[8]
            gw_acc_i_16 += scalar_t(-0.2886751345948129) * r26 * r21 * r12;
            gx_acc_j_8 += scalar_t(-0.2886751345948129) * (r0 * r26) * r12;
            gy_acc_k_11 += scalar_t(-0.2886751345948129) * (r0 * r26) * r21;
            // inst 194: bwd_fma_resident | path#174: gw[16] += go[64]*x[4]*y[13], gx[4] += w[16]*go[64]*y[13], gy[13] += w[16]*go[64]*x[4]
            gw_acc_i_16 += scalar_t(-0.288675134594813) * r26 * r22 * r2;
            gx_acc_j_4 += scalar_t(-0.288675134594813) * (r0 * r26) * r2;
            gy_acc_k_13 += scalar_t(-0.288675134594813) * (r0 * r26) * r22;
            // inst 195: bwd_fma_resident | path#175: gw[16] += go[64]*x[5]*y[14], gx[5] += w[16]*go[64]*y[14], gy[14] += w[16]*go[64]*x[5]
            gw_acc_i_16 += scalar_t(0.4564354645876388) * r26 * r10 * r18;
            gx_acc_j_5 += scalar_t(0.4564354645876388) * (r0 * r26) * r18;
            gy_acc_k_14 += scalar_t(0.4564354645876388) * (r0 * r26) * r10;
            // inst 196: bwd_fma_resident | path#177: gw[16] += go[64]*x[7]*y[10], gx[7] += w[16]*go[64]*y[10], gy[10] += w[16]*go[64]*x[7]
            gw_acc_i_16 += scalar_t(0.456435464587639) * r26 * r7 * r24;
            gx_acc_j_7 += scalar_t(0.456435464587639) * (r0 * r26) * r24;
            gy_acc_k_10 += scalar_t(0.456435464587639) * (r0 * r26) * r7;
            // inst 197: load | global key=(3, 1, (2, 3, 24, 0, -10, 12), "('y', 12)")
            r27 = y[y_base + (index_t)12];
            // inst 198: bwd_fma_resident | path#194: gw[16] += go[67]*x[6]*y[12], gx[6] += w[16]*go[67]*y[12], gy[12] += w[16]*go[67]*x[6]
            gw_acc_i_16 += scalar_t(0.5163977794943223) * r25 * r19 * r27;
            gx_acc_j_6 += scalar_t(0.5163977794943223) * (r0 * r25) * r27;
            gy_acc_k_12 += scalar_t(0.5163977794943223) * (r0 * r25) * r19;
            // inst 199: release | last use after path#194
            // inst 200: bwd_fma_resident | path#186: gw[16] += go[66]*x[5]*y[12], gx[5] += w[16]*go[66]*y[12], gy[12] += w[16]*go[66]*x[5]
            gw_acc_i_16 += scalar_t(0.18257418583505508) * r20 * r10 * r27;
            gx_acc_j_5 += scalar_t(0.18257418583505508) * (r0 * r20) * r27;
            gy_acc_k_12 += scalar_t(0.18257418583505508) * (r0 * r20) * r10;
            // inst 201: bwd_fma_resident | path#201: gw[16] += go[68]*x[7]*y[12], gx[7] += w[16]*go[68]*y[12], gy[12] += w[16]*go[68]*x[7]
            gw_acc_i_16 += scalar_t(0.1825741858350552) * r4 * r7 * r27;
            gx_acc_j_7 += scalar_t(0.1825741858350552) * (r0 * r4) * r27;
            gy_acc_k_12 += scalar_t(0.1825741858350552) * (r0 * r4) * r7;
            // inst 202: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 31)")
            r25 = grad_out[go_base + (index_t)6944 + (index_t)u];
            // inst 203: bwd_fma_resident | path#88: gw[10] += go[31]*x[3]*y[12], gx[3] += w[10]*go[31]*y[12], gy[12] += w[10]*go[31]*x[3]
            gw_acc_i_10 += scalar_t(-0.377964473009227) * r25 * r8 * r27;
            gx_acc_j_3 += scalar_t(-0.377964473009227) * (r15 * r25) * r27;
            gy_acc_k_12 += scalar_t(-0.377964473009227) * (r15 * r25) * r8;
            // inst 204: bwd_fma_resident | path#86: gw[10] += go[31]*x[1]*y[10], gx[1] += w[10]*go[31]*y[10], gy[10] += w[10]*go[31]*x[1]
            gw_acc_i_10 += scalar_t(0.4879500364742665) * r25 * r13 * r24;
            gx_acc_j_1 += scalar_t(0.4879500364742665) * (r15 * r25) * r24;
            gy_acc_k_10 += scalar_t(0.4879500364742665) * (r15 * r25) * r13;
            // inst 205: bwd_fma_resident | path#87: gw[10] += go[31]*x[2]*y[13], gx[2] += w[10]*go[31]*y[13], gy[13] += w[10]*go[31]*x[2]
            gw_acc_i_10 += scalar_t(0.6172133998483672) * r25 * r16 * r2;
            gx_acc_j_2 += scalar_t(0.6172133998483672) * (r15 * r25) * r2;
            gy_acc_k_13 += scalar_t(0.6172133998483672) * (r15 * r25) * r16;
            // inst 206: bwd_fma_resident | path#89: gw[10] += go[31]*x[3]*y[14], gx[3] += w[10]*go[31]*y[14], gy[14] += w[10]*go[31]*x[3]
            gw_acc_i_10 += scalar_t(0.487950036474267) * r25 * r8 * r18;
            gx_acc_j_3 += scalar_t(0.487950036474267) * (r15 * r25) * r18;
            gy_acc_k_14 += scalar_t(0.487950036474267) * (r15 * r25) * r8;
            // inst 207: release | last use after path#89
            // inst 208: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 29)")
            r25 = grad_out[go_base + (index_t)6496 + (index_t)u];
            // inst 209: bwd_fma_resident | path#79: gw[10] += go[29]*x[1]*y[12], gx[1] += w[10]*go[29]*y[12], gy[12] += w[10]*go[29]*x[1]
            gw_acc_i_10 += scalar_t(-0.37796447300922675) * r25 * r13 * r27;
            gx_acc_j_1 += scalar_t(-0.37796447300922675) * (r15 * r25) * r27;
            gy_acc_k_12 += scalar_t(-0.37796447300922675) * (r15 * r25) * r13;
            // inst 210: bwd_fma_resident | path#81: gw[10] += go[29]*x[2]*y[11], gx[2] += w[10]*go[29]*y[11], gy[11] += w[10]*go[29]*x[2]
            gw_acc_i_10 += scalar_t(0.6172133998483673) * r25 * r16 * r12;
            gx_acc_j_2 += scalar_t(0.6172133998483673) * (r15 * r25) * r12;
            gy_acc_k_11 += scalar_t(0.6172133998483673) * (r15 * r25) * r16;
            // inst 211: bwd_fma_resident | path#80: gw[10] += go[29]*x[1]*y[14], gx[1] += w[10]*go[29]*y[14], gy[14] += w[10]*go[29]*x[1]
            gw_acc_i_10 += scalar_t(-0.48795003647426666) * r25 * r13 * r18;
            gx_acc_j_1 += scalar_t(-0.48795003647426666) * (r15 * r25) * r18;
            gy_acc_k_14 += scalar_t(-0.48795003647426666) * (r15 * r25) * r13;
            // inst 212: bwd_fma_resident | path#82: gw[10] += go[29]*x[3]*y[10], gx[3] += w[10]*go[29]*y[10], gy[10] += w[10]*go[29]*x[3]
            gw_acc_i_10 += scalar_t(0.4879500364742667) * r25 * r8 * r24;
            gx_acc_j_3 += scalar_t(0.4879500364742667) * (r15 * r25) * r24;
            gy_acc_k_10 += scalar_t(0.4879500364742667) * (r15 * r25) * r8;
            // inst 213: release | last use after path#82
            // inst 214: load | global key=(3, 1, (2, 3, 21, 0, -8, 10), "('y', 9)")
            r25 = y[y_base + (index_t)9];
            // inst 215: bwd_fma_resident | path#176: gw[16] += go[64]*x[6]*y[9], gx[6] += w[16]*go[64]*y[9], gy[9] += w[16]*go[64]*x[6]
            gw_acc_i_16 += scalar_t(-0.645497224367902) * r26 * r19 * r25;
            gx_acc_j_6 += scalar_t(-0.645497224367902) * (r0 * r26) * r25;
            gy_acc_k_9 += scalar_t(-0.645497224367902) * (r0 * r26) * r19;
            // inst 216: release | last use after path#176
            // inst 217: bwd_fma_resident | path#190: gw[16] += go[66]*x[8]*y[9], gx[8] += w[16]*go[66]*y[9], gy[9] += w[16]*go[66]*x[8]
            gw_acc_i_16 += scalar_t(-0.2886751345948133) * r20 * r21 * r25;
            gx_acc_j_8 += scalar_t(-0.2886751345948133) * (r0 * r20) * r25;
            gy_acc_k_9 += scalar_t(-0.2886751345948133) * (r0 * r20) * r21;
            // inst 218: bwd_fma_resident | path#197: gw[16] += go[68]*x[4]*y[9], gx[4] += w[16]*go[68]*y[9], gy[9] += w[16]*go[68]*x[4]
            gw_acc_i_16 += scalar_t(-0.2886751345948127) * r4 * r22 * r25;
            gx_acc_j_4 += scalar_t(-0.2886751345948127) * (r0 * r4) * r25;
            gy_acc_k_9 += scalar_t(-0.2886751345948127) * (r0 * r4) * r22;
            // inst 219: load | global key=(4, 0, (0, 4, 14, 0, -1, 5), "('go', 69)")
            r26 = grad_out[go_base + (index_t)15456 + (index_t)u];
            // inst 220: bwd_fma_resident | path#206: gw[16] += go[69]*x[5]*y[11], gx[5] += w[16]*go[69]*y[11], gy[11] += w[16]*go[69]*x[5]
            gw_acc_i_16 += scalar_t(-0.35355339059327345) * r26 * r10 * r12;
            gx_acc_j_5 += scalar_t(-0.35355339059327345) * (r0 * r26) * r12;
            gy_acc_k_11 += scalar_t(-0.35355339059327345) * (r0 * r26) * r10;
            // inst 221: bwd_fma_resident | path#207: gw[16] += go[69]*x[7]*y[13], gx[7] += w[16]*go[69]*y[13], gy[13] += w[16]*go[69]*x[7]
            gw_acc_i_16 += scalar_t(0.3535533905932733) * r26 * r7 * r2;
            gx_acc_j_7 += scalar_t(0.3535533905932733) * (r0 * r26) * r2;
            gy_acc_k_13 += scalar_t(0.3535533905932733) * (r0 * r26) * r7;
            // inst 222: bwd_fma_resident | path#209: gw[16] += go[69]*x[8]*y[12], gx[8] += w[16]*go[69]*y[12], gy[12] += w[16]*go[69]*x[8]
            gw_acc_i_16 += scalar_t(-0.5773502691896265) * r26 * r21 * r27;
            gx_acc_j_8 += scalar_t(-0.5773502691896265) * (r0 * r26) * r27;
            gy_acc_k_12 += scalar_t(-0.5773502691896265) * (r0 * r26) * r21;
            // inst 223: bwd_fma_resident | path#205: gw[16] += go[69]*x[5]*y[9], gx[5] += w[16]*go[69]*y[9], gy[9] += w[16]*go[69]*x[5]
            gw_acc_i_16 += scalar_t(0.4564354645876381) * r26 * r10 * r25;
            gx_acc_j_5 += scalar_t(0.4564354645876381) * (r0 * r26) * r25;
            gy_acc_k_9 += scalar_t(0.4564354645876381) * (r0 * r26) * r10;
            // inst 224: load | global key=(4, 1, (8, 4, 22, 0, -7, 10), "('y', 15)")
            r28 = y[y_base + (index_t)15];
            // inst 225: bwd_fma_resident | path#185: gw[16] += go[66]*x[4]*y[15], gx[4] += w[16]*go[66]*y[15], gy[15] += w[16]*go[66]*x[4]
            gw_acc_i_16 += scalar_t(0.2886751345948125) * r20 * r22 * r28;
            gx_acc_j_4 += scalar_t(0.2886751345948125) * (r0 * r20) * r28;
            gy_acc_k_15 += scalar_t(0.2886751345948125) * (r0 * r20) * r22;
            // inst 226: release | last use after path#185
            // inst 227: bwd_fma_resident | path#204: gw[16] += go[68]*x[8]*y[15], gx[8] += w[16]*go[68]*y[15], gy[15] += w[16]*go[68]*x[8]
            gw_acc_i_16 += scalar_t(-0.2886751345948126) * r4 * r21 * r28;
            gx_acc_j_8 += scalar_t(-0.2886751345948126) * (r0 * r4) * r28;
            gy_acc_k_15 += scalar_t(-0.2886751345948126) * (r0 * r4) * r21;
            // inst 228: release | last use after path#204
            // inst 229: bwd_fma_resident | path#208: gw[16] += go[69]*x[7]*y[15], gx[7] += w[16]*go[69]*y[15], gy[15] += w[16]*go[69]*x[7]
            gw_acc_i_16 += scalar_t(0.45643546458763845) * r26 * r7 * r28;
            gx_acc_j_7 += scalar_t(0.45643546458763845) * (r0 * r26) * r28;
            gy_acc_k_15 += scalar_t(0.45643546458763845) * (r0 * r26) * r7;
            // inst 230: release | last use after path#208
            // inst 231: bwd_fma_resident | path#212: gw[16] += go[70]*x[6]*y[15], gx[6] += w[16]*go[70]*y[15], gy[15] += w[16]*go[70]*x[6]
            gw_acc_i_16 += scalar_t(-0.6454972243679024) * r23 * r19 * r28;
            gx_acc_j_6 += scalar_t(-0.6454972243679024) * (r0 * r23) * r28;
            gy_acc_k_15 += scalar_t(-0.6454972243679024) * (r0 * r23) * r19;
            // inst 232: release | last use after path#212
            // inst 233: load | global key=(5, 0, (0, 5, 15, 0, 0, 5), "('go', 65)")
            r23 = grad_out[go_base + (index_t)14560 + (index_t)u];
            // inst 234: bwd_fma_resident | path#183: gw[16] += go[65]*x[7]*y[11], gx[7] += w[16]*go[65]*y[11], gy[11] += w[16]*go[65]*x[7]
            gw_acc_i_16 += scalar_t(0.35355339059327395) * r23 * r7 * r12;
            gx_acc_j_7 += scalar_t(0.35355339059327395) * (r0 * r23) * r12;
            gy_acc_k_11 += scalar_t(0.35355339059327395) * (r0 * r23) * r7;
            // inst 235: bwd_fma_resident | path#180: gw[16] += go[65]*x[5]*y[13], gx[5] += w[16]*go[65]*y[13], gy[13] += w[16]*go[65]*x[5]
            gw_acc_i_16 += scalar_t(0.3535533905932737) * r23 * r10 * r2;
            gx_acc_j_5 += scalar_t(0.3535533905932737) * (r0 * r23) * r2;
            gy_acc_k_13 += scalar_t(0.3535533905932737) * (r0 * r23) * r10;
            // inst 236: bwd_fma_resident | path#179: gw[16] += go[65]*x[4]*y[12], gx[4] += w[16]*go[65]*y[12], gy[12] += w[16]*go[65]*x[4]
            gw_acc_i_16 += scalar_t(-0.5773502691896258) * r23 * r22 * r27;
            gx_acc_j_4 += scalar_t(-0.5773502691896258) * (r0 * r23) * r27;
            gy_acc_k_12 += scalar_t(-0.5773502691896258) * (r0 * r23) * r22;
            // inst 237: bwd_fma_resident | path#181: gw[16] += go[65]*x[5]*y[15], gx[5] += w[16]*go[65]*y[15], gy[15] += w[16]*go[65]*x[5]
            gw_acc_i_16 += scalar_t(-0.4564354645876381) * r23 * r10 * r28;
            gx_acc_j_5 += scalar_t(-0.4564354645876381) * (r0 * r23) * r28;
            gy_acc_k_15 += scalar_t(-0.4564354645876381) * (r0 * r23) * r10;
            // inst 238: bwd_fma_resident | path#182: gw[16] += go[65]*x[7]*y[9], gx[7] += w[16]*go[65]*y[9], gy[9] += w[16]*go[65]*x[7]
            gw_acc_i_16 += scalar_t(0.4564354645876384) * r23 * r7 * r25;
            gx_acc_j_7 += scalar_t(0.4564354645876384) * (r0 * r23) * r25;
            gy_acc_k_9 += scalar_t(0.4564354645876384) * (r0 * r23) * r7;
            // inst 239: release | last use after path#182
            // inst 240: release | last use after path#182
            // inst 241: load | global key=(5, 0, (0, 5, 15, 0, 0, 5), "('go', 32)")
            r23 = grad_out[go_base + (index_t)7168 + (index_t)u];
            // inst 242: bwd_fma_resident | path#91: gw[10] += go[32]*x[1]*y[11], gx[1] += w[10]*go[32]*y[11], gy[11] += w[10]*go[32]*x[1]
            gw_acc_i_10 += scalar_t(0.1543033499620916) * r23 * r13 * r12;
            gx_acc_j_1 += scalar_t(0.1543033499620916) * (r15 * r23) * r12;
            gy_acc_k_11 += scalar_t(0.1543033499620916) * (r15 * r23) * r13;
            // inst 243: bwd_fma_resident | path#93: gw[10] += go[32]*x[3]*y[13], gx[3] += w[10]*go[32]*y[13], gy[13] += w[10]*go[32]*x[3]
            gw_acc_i_10 += scalar_t(-0.1543033499620918) * r23 * r8 * r2;
            gx_acc_j_3 += scalar_t(-0.1543033499620918) * (r15 * r23) * r2;
            gy_acc_k_13 += scalar_t(-0.1543033499620918) * (r15 * r23) * r8;
            // inst 244: bwd_fma_resident | path#90: gw[10] += go[32]*x[1]*y[9], gx[1] += w[10]*go[32]*y[9], gy[9] += w[10]*go[32]*x[1]
            gw_acc_i_10 += scalar_t(0.5976143046671969) * r23 * r13 * r25;
            gx_acc_j_1 += scalar_t(0.5976143046671969) * (r15 * r23) * r25;
            gy_acc_k_9 += scalar_t(0.5976143046671969) * (r15 * r23) * r13;
            // inst 245: bwd_fma_resident | path#94: gw[10] += go[32]*x[3]*y[15], gx[3] += w[10]*go[32]*y[15], gy[15] += w[10]*go[32]*x[3]
            gw_acc_i_10 += scalar_t(0.5976143046671977) * r23 * r8 * r28;
            gx_acc_j_3 += scalar_t(0.5976143046671977) * (r15 * r23) * r28;
            gy_acc_k_15 += scalar_t(0.5976143046671977) * (r15 * r23) * r8;
            // inst 246: bwd_fma_resident | path#92: gw[10] += go[32]*x[2]*y[14], gx[2] += w[10]*go[32]*y[14], gy[14] += w[10]*go[32]*x[2]
            gw_acc_i_10 += scalar_t(0.48795003647426655) * r23 * r16 * r18;
            gx_acc_j_2 += scalar_t(0.48795003647426655) * (r15 * r23) * r18;
            gy_acc_k_14 += scalar_t(0.48795003647426655) * (r15 * r23) * r16;
            // inst 247: release | last use after path#92
            // inst 248: load | global key=(5, 0, (0, 5, 15, 0, 0, 5), "('go', 28)")
            r23 = grad_out[go_base + (index_t)6272 + (index_t)u];
            // inst 249: bwd_fma_resident | path#78: gw[10] += go[28]*x[3]*y[11], gx[3] += w[10]*go[28]*y[11], gy[11] += w[10]*go[28]*x[3]
            gw_acc_i_10 += scalar_t(-0.1543033499620918) * r23 * r8 * r12;
            gx_acc_j_3 += scalar_t(-0.1543033499620918) * (r15 * r23) * r12;
            gy_acc_k_11 += scalar_t(-0.1543033499620918) * (r15 * r23) * r8;
            // inst 250: bwd_fma_resident | path#74: gw[10] += go[28]*x[1]*y[13], gx[1] += w[10]*go[28]*y[13], gy[13] += w[10]*go[28]*x[1]
            gw_acc_i_10 += scalar_t(-0.15430334996209177) * r23 * r13 * r2;
            gx_acc_j_1 += scalar_t(-0.15430334996209177) * (r15 * r23) * r2;
            gy_acc_k_13 += scalar_t(-0.15430334996209177) * (r15 * r23) * r13;
            // inst 251: bwd_fma_resident | path#75: gw[10] += go[28]*x[1]*y[15], gx[1] += w[10]*go[28]*y[15], gy[15] += w[10]*go[28]*x[1]
            gw_acc_i_10 += scalar_t(-0.597614304667197) * r23 * r13 * r28;
            gx_acc_j_1 += scalar_t(-0.597614304667197) * (r15 * r23) * r28;
            gy_acc_k_15 += scalar_t(-0.597614304667197) * (r15 * r23) * r13;
            // inst 252: bwd_fma_resident | path#76: gw[10] += go[28]*x[2]*y[10], gx[2] += w[10]*go[28]*y[10], gy[10] += w[10]*go[28]*x[2]
            gw_acc_i_10 += scalar_t(0.4879500364742665) * r23 * r16 * r24;
            gx_acc_j_2 += scalar_t(0.4879500364742665) * (r15 * r23) * r24;
            gy_acc_k_10 += scalar_t(0.4879500364742665) * (r15 * r23) * r16;
            // inst 253: bwd_fma_resident | path#77: gw[10] += go[28]*x[3]*y[9], gx[3] += w[10]*go[28]*y[9], gy[9] += w[10]*go[28]*x[3]
            gw_acc_i_10 += scalar_t(0.5976143046671976) * r23 * r8 * r25;
            gx_acc_j_3 += scalar_t(0.5976143046671976) * (r15 * r23) * r25;
            gy_acc_k_9 += scalar_t(0.5976143046671976) * (r15 * r23) * r8;
            // inst 254: release | last use after path#77
            // inst 255: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 30)")
            r23 = grad_out[go_base + (index_t)6720 + (index_t)u];
            // inst 256: bwd_fma_resident | path#83: gw[10] += go[30]*x[1]*y[11], gx[1] += w[10]*go[30]*y[11], gy[11] += w[10]*go[30]*x[1]
            gw_acc_i_10 += scalar_t(0.5345224838248488) * r23 * r13 * r12;
            gx_acc_j_1 += scalar_t(0.5345224838248488) * (r15 * r23) * r12;
            gy_acc_k_11 += scalar_t(0.5345224838248488) * (r15 * r23) * r13;
            // inst 257: bwd_fma_resident | path#85: gw[10] += go[30]*x[3]*y[13], gx[3] += w[10]*go[30]*y[13], gy[13] += w[10]*go[30]*x[3]
            gw_acc_i_10 += scalar_t(0.5345224838248488) * r23 * r8 * r2;
            gx_acc_j_3 += scalar_t(0.5345224838248488) * (r15 * r23) * r2;
            gy_acc_k_13 += scalar_t(0.5345224838248488) * (r15 * r23) * r8;
            // inst 258: bwd_fma_resident | path#84: gw[10] += go[30]*x[2]*y[12], gx[2] += w[10]*go[30]*y[12], gy[12] += w[10]*go[30]*x[2]
            gw_acc_i_10 += scalar_t(0.6546536707079768) * r23 * r16 * r27;
            gx_acc_j_2 += scalar_t(0.6546536707079768) * (r15 * r23) * r27;
            gy_acc_k_12 += scalar_t(0.6546536707079768) * (r15 * r23) * r16;
            // inst 259: release | last use after path#84
            // inst 260: release | last use after path#84
            // inst 261: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 49)")
            r23 = grad_out[go_base + (index_t)10976 + (index_t)u];
            // inst 262: bwd_fma_resident | path#131: gw[13] += go[49]*x[0]*y[15], gx[0] += w[13]*go[49]*y[15], gy[15] += w[13]*go[49]*x[0]
            gw_acc_i_13 += scalar_t(1.0000000000000004) * r23 * r3 * r28;
            gx_acc_j_0 += scalar_t(1.0000000000000004) * (r1 * r23) * r28;
            gy_acc_k_15 += scalar_t(1.0000000000000004) * (r1 * r23) * r3;
            // inst 263: release | last use after path#131
            // inst 264: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 48)")
            r23 = grad_out[go_base + (index_t)10752 + (index_t)u];
            // inst 265: bwd_fma_resident | path#130: gw[13] += go[48]*x[0]*y[14], gx[0] += w[13]*go[48]*y[14], gy[14] += w[13]*go[48]*x[0]
            gw_acc_i_13 += scalar_t(0.9999999999999998) * r23 * r3 * r18;
            gx_acc_j_0 += scalar_t(0.9999999999999998) * (r1 * r23) * r18;
            gy_acc_k_14 += scalar_t(0.9999999999999998) * (r1 * r23) * r3;
            // inst 266: release | last use after path#130
            // inst 267: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 46)")
            r23 = grad_out[go_base + (index_t)10304 + (index_t)u];
            // inst 268: bwd_fma_resident | path#128: gw[13] += go[46]*x[0]*y[12], gx[0] += w[13]*go[46]*y[12], gy[12] += w[13]*go[46]*x[0]
            gw_acc_i_13 += scalar_t(0.9999999999999996) * r23 * r3 * r27;
            gx_acc_j_0 += scalar_t(0.9999999999999996) * (r1 * r23) * r27;
            gy_acc_k_12 += scalar_t(0.9999999999999996) * (r1 * r23) * r3;
            // inst 269: release | last use after path#128
            // inst 270: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 45)")
            r23 = grad_out[go_base + (index_t)10080 + (index_t)u];
            // inst 271: bwd_fma_resident | path#127: gw[13] += go[45]*x[0]*y[11], gx[0] += w[13]*go[45]*y[11], gy[11] += w[13]*go[45]*x[0]
            gw_acc_i_13 += scalar_t(0.9999999999999991) * r23 * r3 * r12;
            gx_acc_j_0 += scalar_t(0.9999999999999991) * (r1 * r23) * r12;
            gy_acc_k_11 += scalar_t(0.9999999999999991) * (r1 * r23) * r3;
            // inst 272: release | last use after path#127
            // inst 273: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 44)")
            r23 = grad_out[go_base + (index_t)9856 + (index_t)u];
            // inst 274: bwd_fma_resident | path#126: gw[13] += go[44]*x[0]*y[10], gx[0] += w[13]*go[44]*y[10], gy[10] += w[13]*go[44]*x[0]
            gw_acc_i_13 += scalar_t(1.0000000000000004) * r23 * r3 * r24;
            gx_acc_j_0 += scalar_t(1.0000000000000004) * (r1 * r23) * r24;
            gy_acc_k_10 += scalar_t(1.0000000000000004) * (r1 * r23) * r3;
            // inst 275: release | last use after path#126
            // inst 276: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 43)")
            r23 = grad_out[go_base + (index_t)9632 + (index_t)u];
            // inst 277: bwd_fma_resident | path#125: gw[13] += go[43]*x[0]*y[9], gx[0] += w[13]*go[43]*y[9], gy[9] += w[13]*go[43]*x[0]
            gw_acc_i_13 += scalar_t(1.0000000000000007) * r23 * r3 * r25;
            gx_acc_j_0 += scalar_t(1.0000000000000007) * (r1 * r23) * r25;
            gy_acc_k_9 += scalar_t(1.0000000000000007) * (r1 * r23) * r3;
            // inst 278: release | last use after path#125
            // inst 279: release | last use after path#125
            // inst 280: load | global key=(0, 0, (0, 0, 42, 0, -3, 21), "('w', 7)")
            r23 = w[w_base + (index_t)1568 + (index_t)u];
            // inst 281: load | global key=(8, 0, (0, 8, 24, 0, 0, 8), "('go', 17)")
            r1 = grad_out[go_base + (index_t)3808 + (index_t)u];
            // inst 282: bwd_fma_resident | path#51: gw[7] += go[17]*x[4]*y[11], gx[4] += w[7]*go[17]*y[11], gy[11] += w[7]*go[17]*x[4]
            gw_acc_i_7 += scalar_t(-0.11952286093343928) * r1 * r22 * r12;
            gx_acc_j_4 += scalar_t(-0.11952286093343928) * (r23 * r1) * r12;
            gy_acc_k_11 += scalar_t(-0.11952286093343928) * (r23 * r1) * r22;
            // inst 283: bwd_fma_resident | path#56: gw[7] += go[17]*x[8]*y[13], gx[8] += w[7]*go[17]*y[13], gy[13] += w[7]*go[17]*x[8]
            gw_acc_i_7 += scalar_t(-0.11952286093343928) * r1 * r21 * r2;
            gx_acc_j_8 += scalar_t(-0.11952286093343928) * (r23 * r1) * r2;
            gy_acc_k_13 += scalar_t(-0.11952286093343928) * (r23 * r1) * r21;
            // inst 284: bwd_fma_resident | path#50: gw[7] += go[17]*x[4]*y[9], gx[4] += w[7]*go[17]*y[9], gy[9] += w[7]*go[17]*x[4]
            gw_acc_i_7 += scalar_t(0.46291004988627626) * r1 * r22 * r25;
            gx_acc_j_4 += scalar_t(0.46291004988627626) * (r23 * r1) * r25;
            gy_acc_k_9 += scalar_t(0.46291004988627626) * (r23 * r1) * r22;
            // inst 285: bwd_fma_resident | path#52: gw[7] += go[17]*x[5]*y[10], gx[5] += w[7]*go[17]*y[10], gy[10] += w[7]*go[17]*x[5]
            gw_acc_i_7 += scalar_t(0.37796447300922736) * r1 * r10 * r24;
            gx_acc_j_5 += scalar_t(0.37796447300922736) * (r23 * r1) * r24;
            gy_acc_k_10 += scalar_t(0.37796447300922736) * (r23 * r1) * r10;
            // inst 286: bwd_fma_resident | path#54: gw[7] += go[17]*x[7]*y[12], gx[7] += w[7]*go[17]*y[12], gy[12] += w[7]*go[17]*x[7]
            gw_acc_i_7 += scalar_t(-0.2927700218845598) * r1 * r7 * r27;
            gx_acc_j_7 += scalar_t(-0.2927700218845598) * (r23 * r1) * r27;
            gy_acc_k_12 += scalar_t(-0.2927700218845598) * (r23 * r1) * r7;
            // inst 287: bwd_fma_resident | path#57: gw[7] += go[17]*x[8]*y[15], gx[8] += w[7]*go[17]*y[15], gy[15] += w[7]*go[17]*x[8]
            gw_acc_i_7 += scalar_t(0.4629100498862764) * r1 * r21 * r28;
            gx_acc_j_8 += scalar_t(0.4629100498862764) * (r23 * r1) * r28;
            gy_acc_k_15 += scalar_t(0.4629100498862764) * (r23 * r1) * r21;
            // inst 288: bwd_fma_resident | path#53: gw[7] += go[17]*x[6]*y[13], gx[6] += w[7]*go[17]*y[13], gy[13] += w[7]*go[17]*x[6]
            gw_acc_i_7 += scalar_t(0.41403933560541256) * r1 * r19 * r2;
            gx_acc_j_6 += scalar_t(0.41403933560541256) * (r23 * r1) * r2;
            gy_acc_k_13 += scalar_t(0.41403933560541256) * (r23 * r1) * r19;
            // inst 289: bwd_fma_resident | path#55: gw[7] += go[17]*x[7]*y[14], gx[7] += w[7]*go[17]*y[14], gy[14] += w[7]*go[17]*x[7]
            gw_acc_i_7 += scalar_t(0.37796447300922753) * r1 * r7 * r18;
            gx_acc_j_7 += scalar_t(0.37796447300922753) * (r23 * r1) * r18;
            gy_acc_k_14 += scalar_t(0.37796447300922753) * (r23 * r1) * r7;
            // inst 290: release | last use after path#55
            // inst 291: load | global key=(8, 1, (4, 8, 24, 0, 0, 8), "('go', 15)")
            r1 = grad_out[go_base + (index_t)3360 + (index_t)u];
            // inst 292: bwd_fma_resident | path#38: gw[7] += go[15]*x[4]*y[15], gx[4] += w[7]*go[15]*y[15], gy[15] += w[7]*go[15]*x[4]
            gw_acc_i_7 += scalar_t(-0.4629100498862759) * r1 * r22 * r28;
            gx_acc_j_4 += scalar_t(-0.4629100498862759) * (r23 * r1) * r28;
            gy_acc_k_15 += scalar_t(-0.4629100498862759) * (r23 * r1) * r22;
            // inst 293: release | last use after path#38
            // inst 294: bwd_fma_resident | path#43: gw[7] += go[15]*x[8]*y[9], gx[8] += w[7]*go[15]*y[9], gy[9] += w[7]*go[15]*x[8]
            gw_acc_i_7 += scalar_t(0.4629100498862758) * r1 * r21 * r25;
            gx_acc_j_8 += scalar_t(0.4629100498862758) * (r23 * r1) * r25;
            gy_acc_k_9 += scalar_t(0.4629100498862758) * (r23 * r1) * r21;
            // inst 295: release | last use after path#43
            // inst 296: bwd_fma_resident | path#44: gw[7] += go[15]*x[8]*y[11], gx[8] += w[7]*go[15]*y[11], gy[11] += w[7]*go[15]*x[8]
            gw_acc_i_7 += scalar_t(0.11952286093343914) * r1 * r21 * r12;
            gx_acc_j_8 += scalar_t(0.11952286093343914) * (r23 * r1) * r12;
            gy_acc_k_11 += scalar_t(0.11952286093343914) * (r23 * r1) * r21;
            // inst 297: bwd_fma_resident | path#37: gw[7] += go[15]*x[4]*y[13], gx[4] += w[7]*go[15]*y[13], gy[13] += w[7]*go[15]*x[4]
            gw_acc_i_7 += scalar_t(-0.11952286093343926) * r1 * r22 * r2;
            gx_acc_j_4 += scalar_t(-0.11952286093343926) * (r23 * r1) * r2;
            gy_acc_k_13 += scalar_t(-0.11952286093343926) * (r23 * r1) * r22;
            // inst 298: bwd_fma_resident | path#39: gw[7] += go[15]*x[5]*y[12], gx[5] += w[7]*go[15]*y[12], gy[12] += w[7]*go[15]*x[5]
            gw_acc_i_7 += scalar_t(-0.29277002188455964) * r1 * r10 * r27;
            gx_acc_j_5 += scalar_t(-0.29277002188455964) * (r23 * r1) * r27;
            gy_acc_k_12 += scalar_t(-0.29277002188455964) * (r23 * r1) * r10;
            // inst 299: bwd_fma_resident | path#40: gw[7] += go[15]*x[5]*y[14], gx[5] += w[7]*go[15]*y[14], gy[14] += w[7]*go[15]*x[5]
            gw_acc_i_7 += scalar_t(-0.3779644730092273) * r1 * r10 * r18;
            gx_acc_j_5 += scalar_t(-0.3779644730092273) * (r23 * r1) * r18;
            gy_acc_k_14 += scalar_t(-0.3779644730092273) * (r23 * r1) * r10;
            // inst 300: bwd_fma_resident | path#41: gw[7] += go[15]*x[6]*y[11], gx[6] += w[7]*go[15]*y[11], gy[11] += w[7]*go[15]*x[6]
            gw_acc_i_7 += scalar_t(0.41403933560541256) * r1 * r19 * r12;
            gx_acc_j_6 += scalar_t(0.41403933560541256) * (r23 * r1) * r12;
            gy_acc_k_11 += scalar_t(0.41403933560541256) * (r23 * r1) * r19;
            // inst 301: bwd_fma_resident | path#42: gw[7] += go[15]*x[7]*y[10], gx[7] += w[7]*go[15]*y[10], gy[10] += w[7]*go[15]*x[7]
            gw_acc_i_7 += scalar_t(0.37796447300922714) * r1 * r7 * r24;
            gx_acc_j_7 += scalar_t(0.37796447300922714) * (r23 * r1) * r24;
            gy_acc_k_10 += scalar_t(0.37796447300922714) * (r23 * r1) * r7;
            // inst 302: release | last use after path#42
            // inst 303: load | global key=(5, 1, (10, 5, 15, 0, 0, 5), "('go', 16)")
            r1 = grad_out[go_base + (index_t)3584 + (index_t)u];
            // inst 304: bwd_fma_resident | path#45: gw[7] += go[16]*x[4]*y[10], gx[4] += w[7]*go[16]*y[10], gy[10] += w[7]*go[16]*x[4]
            gw_acc_i_7 += scalar_t(0.37796447300922714) * r1 * r22 * r24;
            gx_acc_j_4 += scalar_t(0.37796447300922714) * (r23 * r1) * r24;
            gy_acc_k_10 += scalar_t(0.37796447300922714) * (r23 * r1) * r22;
            // inst 305: release | last use after path#45
            // inst 306: bwd_fma_resident | path#49: gw[7] += go[16]*x[8]*y[14], gx[8] += w[7]*go[16]*y[14], gy[14] += w[7]*go[16]*x[8]
            gw_acc_i_7 += scalar_t(0.3779644730092272) * r1 * r21 * r18;
            gx_acc_j_8 += scalar_t(0.3779644730092272) * (r23 * r1) * r18;
            gy_acc_k_14 += scalar_t(0.3779644730092272) * (r23 * r1) * r21;
            // inst 307: release | last use after path#49
            // inst 308: bwd_fma_resident | path#46: gw[7] += go[16]*x[5]*y[11], gx[5] += w[7]*go[16]*y[11], gy[11] += w[7]*go[16]*x[5]
            gw_acc_i_7 += scalar_t(0.4780914437337572) * r1 * r10 * r12;
            gx_acc_j_5 += scalar_t(0.4780914437337572) * (r23 * r1) * r12;
            gy_acc_k_11 += scalar_t(0.4780914437337572) * (r23 * r1) * r10;
            // inst 309: release | last use after path#46
            // inst 310: bwd_fma_resident | path#47: gw[7] += go[16]*x[6]*y[12], gx[6] += w[7]*go[16]*y[12], gy[12] += w[7]*go[16]*x[6]
            gw_acc_i_7 += scalar_t(0.5070925528371097) * r1 * r19 * r27;
            gx_acc_j_6 += scalar_t(0.5070925528371097) * (r23 * r1) * r27;
            gy_acc_k_12 += scalar_t(0.5070925528371097) * (r23 * r1) * r19;
            // inst 311: release | last use after path#47
            // inst 312: bwd_fma_resident | path#48: gw[7] += go[16]*x[7]*y[13], gx[7] += w[7]*go[16]*y[13], gy[13] += w[7]*go[16]*x[7]
            gw_acc_i_7 += scalar_t(0.4780914437337571) * r1 * r7 * r2;
            gx_acc_j_7 += scalar_t(0.4780914437337571) * (r23 * r1) * r2;
            gy_acc_k_13 += scalar_t(0.4780914437337571) * (r23 * r1) * r7;
            // inst 313: release | last use after path#48
            // inst 314: release | last use after path#48
            // inst 315: release | last use after path#48
            // inst 316: load | global key=(0, 0, (0, 0, 21, 0, -10, 21), "('w', 15)")
            r1 = w[w_base + (index_t)3360 + (index_t)u];
            // inst 317: load | global key=(0, 0, (0, 0, 27, 0, -19, 18), "('y', 3)")
            r2 = y[y_base + (index_t)3];
            // inst 318: load | global key=(2, 0, (0, 2, 10, 0, -2, 4), "('go', 61)")
            r23 = grad_out[go_base + (index_t)13664 + (index_t)u];
            // inst 319: bwd_fma_resident | path#168: gw[15] += go[61]*x[8]*y[3], gx[8] += w[15]*go[61]*y[3], gy[3] += w[15]*go[61]*x[8]
            gw_acc_i_15 += scalar_t(-0.18257418583505522) * r23 * r21 * r2;
            gx_acc_j_8 += scalar_t(-0.18257418583505522) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(-0.18257418583505522) * (r1 * r23) * r21;
            // inst 320: bwd_fma_resident | path#166: gw[15] += go[61]*x[6]*y[3], gx[6] += w[15]*go[61]*y[3], gy[3] += w[15]*go[61]*x[6]
            gw_acc_i_15 += scalar_t(0.6324555320336759) * r23 * r19 * r2;
            gx_acc_j_6 += scalar_t(0.6324555320336759) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.6324555320336759) * (r1 * r23) * r19;
            // inst 321: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 5)")
            r27 = grad_out[go_base + (index_t)1120 + (index_t)u];
            // inst 322: bwd_fma_resident | path#11: gw[3] += go[5]*x[0]*y[3], gx[0] += w[3]*go[5]*y[3], gy[3] += w[3]*go[5]*x[0]
            gw_acc_i_3 += scalar_t(1.0000000000000002) * r27 * r3 * r2;
            gx_acc_j_0 += scalar_t(1.0000000000000002) * (r6 * r27) * r2;
            gy_acc_k_3 += scalar_t(1.0000000000000002) * (r6 * r27) * r3;
            // inst 323: release | last use after path#11
            // inst 324: load | global key=(1, 0, (0, 1, 28, 0, -18, 18), "('y', 1)")
            r27 = y[y_base + (index_t)1];
            // inst 325: bwd_fma_resident | path#165: gw[15] += go[61]*x[4]*y[1], gx[4] += w[15]*go[61]*y[1], gy[1] += w[15]*go[61]*x[4]
            gw_acc_i_15 += scalar_t(-0.1825741858350552) * r23 * r22 * r27;
            gx_acc_j_4 += scalar_t(-0.1825741858350552) * (r1 * r23) * r27;
            gy_acc_k_1 += scalar_t(-0.1825741858350552) * (r1 * r23) * r22;
            // inst 326: load | global key=(3, 0, (0, 3, 11, 0, -1, 4), "('go', 59)")
            r12 = grad_out[go_base + (index_t)13216 + (index_t)u];
            // inst 327: bwd_fma_resident | path#161: gw[15] += go[59]*x[8]*y[1], gx[8] += w[15]*go[59]*y[1], gy[1] += w[15]*go[59]*x[8]
            gw_acc_i_15 += scalar_t(0.182574185835055) * r12 * r21 * r27;
            gx_acc_j_8 += scalar_t(0.182574185835055) * (r1 * r12) * r27;
            gy_acc_k_1 += scalar_t(0.182574185835055) * (r1 * r12) * r21;
            // inst 328: bwd_fma_resident | path#158: gw[15] += go[59]*x[4]*y[3], gx[4] += w[15]*go[59]*y[3], gy[3] += w[15]*go[59]*x[4]
            gw_acc_i_15 += scalar_t(-0.18257418583505522) * r12 * r22 * r2;
            gx_acc_j_4 += scalar_t(-0.18257418583505522) * (r1 * r12) * r2;
            gy_acc_k_3 += scalar_t(-0.18257418583505522) * (r1 * r12) * r22;
            // inst 329: bwd_fma_resident | path#160: gw[15] += go[59]*x[6]*y[1], gx[6] += w[15]*go[59]*y[1], gy[1] += w[15]*go[59]*x[6]
            gw_acc_i_15 += scalar_t(0.6324555320336759) * r12 * r19 * r27;
            gx_acc_j_6 += scalar_t(0.6324555320336759) * (r1 * r12) * r27;
            gy_acc_k_1 += scalar_t(0.6324555320336759) * (r1 * r12) * r19;
            // inst 330: load | global key=(2, 1, (4, 2, 21, 0, -14, 13), "('y', 2)")
            r18 = y[y_base + (index_t)2];
            // inst 331: bwd_fma_resident | path#159: gw[15] += go[59]*x[5]*y[2], gx[5] += w[15]*go[59]*y[2], gy[2] += w[15]*go[59]*x[5]
            gw_acc_i_15 += scalar_t(0.7302967433402211) * r12 * r10 * r18;
            gx_acc_j_5 += scalar_t(0.7302967433402211) * (r1 * r12) * r18;
            gy_acc_k_2 += scalar_t(0.7302967433402211) * (r1 * r12) * r10;
            // inst 332: release | last use after path#159
            // inst 333: bwd_fma_resident | path#167: gw[15] += go[61]*x[7]*y[2], gx[7] += w[15]*go[61]*y[2], gy[2] += w[15]*go[61]*x[7]
            gw_acc_i_15 += scalar_t(0.7302967433402209) * r23 * r7 * r18;
            gx_acc_j_7 += scalar_t(0.7302967433402209) * (r1 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.7302967433402209) * (r1 * r23) * r7;
            // inst 334: release | last use after path#167
            // inst 335: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 62)")
            r23 = grad_out[go_base + (index_t)13888 + (index_t)u];
            // inst 336: bwd_fma_resident | path#169: gw[15] += go[62]*x[5]*y[1], gx[5] += w[15]*go[62]*y[1], gy[1] += w[15]*go[62]*x[5]
            gw_acc_i_15 += scalar_t(-0.5773502691896258) * r23 * r10 * r27;
            gx_acc_j_5 += scalar_t(-0.5773502691896258) * (r1 * r23) * r27;
            gy_acc_k_1 += scalar_t(-0.5773502691896258) * (r1 * r23) * r10;
            // inst 337: bwd_fma_resident | path#170: gw[15] += go[62]*x[7]*y[3], gx[7] += w[15]*go[62]*y[3], gy[3] += w[15]*go[62]*x[7]
            gw_acc_i_15 += scalar_t(0.5773502691896263) * r23 * r7 * r2;
            gx_acc_j_7 += scalar_t(0.5773502691896263) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.5773502691896263) * (r1 * r23) * r7;
            // inst 338: bwd_fma_resident | path#171: gw[15] += go[62]*x[8]*y[2], gx[8] += w[15]*go[62]*y[2], gy[2] += w[15]*go[62]*x[8]
            gw_acc_i_15 += scalar_t(0.5773502691896257) * r23 * r21 * r18;
            gx_acc_j_8 += scalar_t(0.5773502691896257) * (r1 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.5773502691896257) * (r1 * r23) * r21;
            // inst 339: release | last use after path#171
            // inst 340: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 60)")
            r23 = grad_out[go_base + (index_t)13440 + (index_t)u];
            // inst 341: bwd_fma_resident | path#162: gw[15] += go[60]*x[5]*y[1], gx[5] += w[15]*go[60]*y[1], gy[1] += w[15]*go[60]*x[5]
            gw_acc_i_15 += scalar_t(-0.4472135954999573) * r23 * r10 * r27;
            gx_acc_j_5 += scalar_t(-0.4472135954999573) * (r1 * r23) * r27;
            gy_acc_k_1 += scalar_t(-0.4472135954999573) * (r1 * r23) * r10;
            // inst 342: bwd_fma_resident | path#164: gw[15] += go[60]*x[7]*y[3], gx[7] += w[15]*go[60]*y[3], gy[3] += w[15]*go[60]*x[7]
            gw_acc_i_15 += scalar_t(-0.44721359549995765) * r23 * r7 * r2;
            gx_acc_j_7 += scalar_t(-0.44721359549995765) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(-0.44721359549995765) * (r1 * r23) * r7;
            // inst 343: bwd_fma_resident | path#163: gw[15] += go[60]*x[6]*y[2], gx[6] += w[15]*go[60]*y[2], gy[2] += w[15]*go[60]*x[6]
            gw_acc_i_15 += scalar_t(0.774596669241483) * r23 * r19 * r18;
            gx_acc_j_6 += scalar_t(0.774596669241483) * (r1 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.774596669241483) * (r1 * r23) * r19;
            // inst 344: release | last use after path#163
            // inst 345: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 58)")
            r23 = grad_out[go_base + (index_t)12992 + (index_t)u];
            // inst 346: bwd_fma_resident | path#157: gw[15] += go[58]*x[7]*y[1], gx[7] += w[15]*go[58]*y[1], gy[1] += w[15]*go[58]*x[7]
            gw_acc_i_15 += scalar_t(0.5773502691896256) * r23 * r7 * r27;
            gx_acc_j_7 += scalar_t(0.5773502691896256) * (r1 * r23) * r27;
            gy_acc_k_1 += scalar_t(0.5773502691896256) * (r1 * r23) * r7;
            // inst 347: bwd_fma_resident | path#156: gw[15] += go[58]*x[5]*y[3], gx[5] += w[15]*go[58]*y[3], gy[3] += w[15]*go[58]*x[5]
            gw_acc_i_15 += scalar_t(0.577350269189626) * r23 * r10 * r2;
            gx_acc_j_5 += scalar_t(0.577350269189626) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.577350269189626) * (r1 * r23) * r10;
            // inst 348: bwd_fma_resident | path#155: gw[15] += go[58]*x[4]*y[2], gx[4] += w[15]*go[58]*y[2], gy[2] += w[15]*go[58]*x[4]
            gw_acc_i_15 += scalar_t(0.5773502691896256) * r23 * r22 * r18;
            gx_acc_j_4 += scalar_t(0.5773502691896256) * (r1 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.5773502691896256) * (r1 * r23) * r22;
            // inst 349: release | last use after path#155
            // inst 350: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 63)")
            r23 = grad_out[go_base + (index_t)14112 + (index_t)u];
            // inst 351: bwd_fma_resident | path#172: gw[15] += go[63]*x[4]*y[1], gx[4] += w[15]*go[63]*y[1], gy[1] += w[15]*go[63]*x[4]
            gw_acc_i_15 += scalar_t(-0.7071067811865477) * r23 * r22 * r27;
            gx_acc_j_4 += scalar_t(-0.7071067811865477) * (r1 * r23) * r27;
            gy_acc_k_1 += scalar_t(-0.7071067811865477) * (r1 * r23) * r22;
            // inst 352: bwd_fma_resident | path#173: gw[15] += go[63]*x[8]*y[3], gx[8] += w[15]*go[63]*y[3], gy[3] += w[15]*go[63]*x[8]
            gw_acc_i_15 += scalar_t(0.7071067811865485) * r23 * r21 * r2;
            gx_acc_j_8 += scalar_t(0.7071067811865485) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.7071067811865485) * (r1 * r23) * r21;
            // inst 353: release | last use after path#173
            // inst 354: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 57)")
            r23 = grad_out[go_base + (index_t)12768 + (index_t)u];
            // inst 355: bwd_fma_resident | path#154: gw[15] += go[57]*x[8]*y[1], gx[8] += w[15]*go[57]*y[1], gy[1] += w[15]*go[57]*x[8]
            gw_acc_i_15 += scalar_t(0.7071067811865476) * r23 * r21 * r27;
            gx_acc_j_8 += scalar_t(0.7071067811865476) * (r1 * r23) * r27;
            gy_acc_k_1 += scalar_t(0.7071067811865476) * (r1 * r23) * r21;
            // inst 356: bwd_fma_resident | path#153: gw[15] += go[57]*x[4]*y[3], gx[4] += w[15]*go[57]*y[3], gy[3] += w[15]*go[57]*x[4]
            gw_acc_i_15 += scalar_t(0.7071067811865482) * r23 * r22 * r2;
            gx_acc_j_4 += scalar_t(0.7071067811865482) * (r1 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.7071067811865482) * (r1 * r23) * r22;
            // inst 357: release | last use after path#153
            // inst 358: release | last use after path#153
            // inst 359: load | global key=(1, 1, (1, 1, 3, 0, 0, 1), "('go', 4)")
            r23 = grad_out[go_base + (index_t)896 + (index_t)u];
            // inst 360: bwd_fma_resident | path#10: gw[3] += go[4]*x[0]*y[2], gx[0] += w[3]*go[4]*y[2], gy[2] += w[3]*go[4]*x[0]
            gw_acc_i_3 += scalar_t(0.9999999999999998) * r23 * r3 * r18;
            gx_acc_j_0 += scalar_t(0.9999999999999998) * (r6 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.9999999999999998) * (r6 * r23) * r3;
            // inst 361: release | last use after path#10
            // inst 362: load | global key=(1, 3, (5, 1, 3, 0, 0, 1), "('go', 3)")
            r23 = grad_out[go_base + (index_t)672 + (index_t)u];
            // inst 363: bwd_fma_resident | path#9: gw[3] += go[3]*x[0]*y[1], gx[0] += w[3]*go[3]*y[1], gy[1] += w[3]*go[3]*x[0]
            gw_acc_i_3 += scalar_t(1.0000000000000002) * r23 * r3 * r27;
            gx_acc_j_0 += scalar_t(1.0000000000000002) * (r6 * r23) * r27;
            gy_acc_k_1 += scalar_t(1.0000000000000002) * (r6 * r23) * r3;
            // inst 364: release | last use after path#9
            // inst 365: release | last use after path#9
            // inst 366: release | last use after path#9
            // inst 367: load | global key=(0, 0, (0, 0, 22, 0, -3, 11), "('w', 6)")
            r23 = w[w_base + (index_t)1344 + (index_t)u];
            // inst 368: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 14)")
            r3 = grad_out[go_base + (index_t)3136 + (index_t)u];
            // inst 369: bwd_fma_resident | path#34: gw[6] += go[14]*x[6]*y[3], gx[6] += w[6]*go[14]*y[3], gy[3] += w[6]*go[14]*x[6]
            gw_acc_i_6 += scalar_t(-0.31622776601683794) * r3 * r19 * r2;
            gx_acc_j_6 += scalar_t(-0.31622776601683794) * (r23 * r3) * r2;
            gy_acc_k_3 += scalar_t(-0.31622776601683794) * (r23 * r3) * r19;
            // inst 370: bwd_fma_resident | path#33: gw[6] += go[14]*x[4]*y[1], gx[4] += w[6]*go[14]*y[1], gy[1] += w[6]*go[14]*x[4]
            gw_acc_i_6 += scalar_t(0.5477225575051664) * r3 * r22 * r27;
            gx_acc_j_4 += scalar_t(0.5477225575051664) * (r23 * r3) * r27;
            gy_acc_k_1 += scalar_t(0.5477225575051664) * (r23 * r3) * r22;
            // inst 371: bwd_fma_resident | path#36: gw[6] += go[14]*x[8]*y[3], gx[8] += w[6]*go[14]*y[3], gy[3] += w[6]*go[14]*x[8]
            gw_acc_i_6 += scalar_t(0.5477225575051664) * r3 * r21 * r2;
            gx_acc_j_8 += scalar_t(0.5477225575051664) * (r23 * r3) * r2;
            gy_acc_k_3 += scalar_t(0.5477225575051664) * (r23 * r3) * r21;
            // inst 372: bwd_fma_resident | path#35: gw[6] += go[14]*x[7]*y[2], gx[7] += w[6]*go[14]*y[2], gy[2] += w[6]*go[14]*x[7]
            gw_acc_i_6 += scalar_t(0.5477225575051656) * r3 * r7 * r18;
            gx_acc_j_7 += scalar_t(0.5477225575051656) * (r23 * r3) * r18;
            gy_acc_k_2 += scalar_t(0.5477225575051656) * (r23 * r3) * r7;
            // inst 373: release | last use after path#35
            // inst 374: load | global key=(4, 0, (0, 4, 12, 0, 0, 4), "('go', 12)")
            r3 = grad_out[go_base + (index_t)2688 + (index_t)u];
            // inst 375: bwd_fma_resident | path#28: gw[6] += go[12]*x[6]*y[1], gx[6] += w[6]*go[12]*y[1], gy[1] += w[6]*go[12]*x[6]
            gw_acc_i_6 += scalar_t(-0.3162277660168376) * r3 * r19 * r27;
            gx_acc_j_6 += scalar_t(-0.3162277660168376) * (r23 * r3) * r27;
            gy_acc_k_1 += scalar_t(-0.3162277660168376) * (r23 * r3) * r19;
            // inst 376: bwd_fma_resident | path#26: gw[6] += go[12]*x[4]*y[3], gx[4] += w[6]*go[12]*y[3], gy[3] += w[6]*go[12]*x[4]
            gw_acc_i_6 += scalar_t(0.5477225575051665) * r3 * r22 * r2;
            gx_acc_j_4 += scalar_t(0.5477225575051665) * (r23 * r3) * r2;
            gy_acc_k_3 += scalar_t(0.5477225575051665) * (r23 * r3) * r22;
            // inst 377: bwd_fma_resident | path#27: gw[6] += go[12]*x[5]*y[2], gx[5] += w[6]*go[12]*y[2], gy[2] += w[6]*go[12]*x[5]
            gw_acc_i_6 += scalar_t(0.5477225575051657) * r3 * r10 * r18;
            gx_acc_j_5 += scalar_t(0.5477225575051657) * (r23 * r3) * r18;
            gy_acc_k_2 += scalar_t(0.5477225575051657) * (r23 * r3) * r10;
            // inst 378: bwd_fma_resident | path#29: gw[6] += go[12]*x[8]*y[1], gx[8] += w[6]*go[12]*y[1], gy[1] += w[6]*go[12]*x[8]
            gw_acc_i_6 += scalar_t(-0.5477225575051664) * r3 * r21 * r27;
            gx_acc_j_8 += scalar_t(-0.5477225575051664) * (r23 * r3) * r27;
            gy_acc_k_1 += scalar_t(-0.5477225575051664) * (r23 * r3) * r21;
            // inst 379: release | last use after path#29
            // inst 380: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 13)")
            r3 = grad_out[go_base + (index_t)2912 + (index_t)u];
            // inst 381: bwd_fma_resident | path#30: gw[6] += go[13]*x[5]*y[1], gx[5] += w[6]*go[13]*y[1], gy[1] += w[6]*go[13]*x[5]
            gw_acc_i_6 += scalar_t(0.5477225575051657) * r3 * r10 * r27;
            gx_acc_j_5 += scalar_t(0.5477225575051657) * (r23 * r3) * r27;
            gy_acc_k_1 += scalar_t(0.5477225575051657) * (r23 * r3) * r10;
            // inst 382: bwd_fma_resident | path#32: gw[6] += go[13]*x[7]*y[3], gx[7] += w[6]*go[13]*y[3], gy[3] += w[6]*go[13]*x[7]
            gw_acc_i_6 += scalar_t(0.5477225575051669) * r3 * r7 * r2;
            gx_acc_j_7 += scalar_t(0.5477225575051669) * (r23 * r3) * r2;
            gy_acc_k_3 += scalar_t(0.5477225575051669) * (r23 * r3) * r7;
            // inst 383: bwd_fma_resident | path#31: gw[6] += go[13]*x[6]*y[2], gx[6] += w[6]*go[13]*y[2], gy[2] += w[6]*go[13]*x[6]
            gw_acc_i_6 += scalar_t(0.6324555320336755) * r3 * r19 * r18;
            gx_acc_j_6 += scalar_t(0.6324555320336755) * (r23 * r3) * r18;
            gy_acc_k_2 += scalar_t(0.6324555320336755) * (r23 * r3) * r19;
            // inst 384: release | last use after path#31
            // inst 385: release | last use after path#31
            // inst 386: load | global key=(0, 0, (0, 0, 22, 0, -5, 11), "('w', 9)")
            r3 = w[w_base + (index_t)2016 + (index_t)u];
            // inst 387: load | global key=(3, 0, (0, 3, 9, 0, 0, 3), "('go', 25)")
            r23 = grad_out[go_base + (index_t)5600 + (index_t)u];
            // inst 388: bwd_fma_resident | path#67: gw[9] += go[25]*x[1]*y[1], gx[1] += w[9]*go[25]*y[1], gy[1] += w[9]*go[25]*x[1]
            gw_acc_i_9 += scalar_t(-0.4082482904638625) * r23 * r13 * r27;
            gx_acc_j_1 += scalar_t(-0.4082482904638625) * (r3 * r23) * r27;
            gy_acc_k_1 += scalar_t(-0.4082482904638625) * (r3 * r23) * r13;
            // inst 389: bwd_fma_resident | path#69: gw[9] += go[25]*x[3]*y[3], gx[3] += w[9]*go[25]*y[3], gy[3] += w[9]*go[25]*x[3]
            gw_acc_i_9 += scalar_t(-0.40824829046386296) * r23 * r8 * r2;
            gx_acc_j_3 += scalar_t(-0.40824829046386296) * (r3 * r23) * r2;
            gy_acc_k_3 += scalar_t(-0.40824829046386296) * (r3 * r23) * r8;
            // inst 390: bwd_fma_resident | path#68: gw[9] += go[25]*x[2]*y[2], gx[2] += w[9]*go[25]*y[2], gy[2] += w[9]*go[25]*x[2]
            gw_acc_i_9 += scalar_t(0.8164965809277256) * r23 * r16 * r18;
            gx_acc_j_2 += scalar_t(0.8164965809277256) * (r3 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.8164965809277256) * (r3 * r23) * r16;
            // inst 391: release | last use after path#68
            // inst 392: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 27)")
            r23 = grad_out[go_base + (index_t)6048 + (index_t)u];
            // inst 393: bwd_fma_resident | path#72: gw[9] += go[27]*x[1]*y[1], gx[1] += w[9]*go[27]*y[1], gy[1] += w[9]*go[27]*x[1]
            gw_acc_i_9 += scalar_t(-0.7071067811865478) * r23 * r13 * r27;
            gx_acc_j_1 += scalar_t(-0.7071067811865478) * (r3 * r23) * r27;
            gy_acc_k_1 += scalar_t(-0.7071067811865478) * (r3 * r23) * r13;
            // inst 394: bwd_fma_resident | path#73: gw[9] += go[27]*x[3]*y[3], gx[3] += w[9]*go[27]*y[3], gy[3] += w[9]*go[27]*x[3]
            gw_acc_i_9 += scalar_t(0.7071067811865478) * r23 * r8 * r2;
            gx_acc_j_3 += scalar_t(0.7071067811865478) * (r3 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.7071067811865478) * (r3 * r23) * r8;
            // inst 395: release | last use after path#73
            // inst 396: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 26)")
            r23 = grad_out[go_base + (index_t)5824 + (index_t)u];
            // inst 397: bwd_fma_resident | path#70: gw[9] += go[26]*x[2]*y[3], gx[2] += w[9]*go[26]*y[3], gy[3] += w[9]*go[26]*x[2]
            gw_acc_i_9 += scalar_t(0.7071067811865483) * r23 * r16 * r2;
            gx_acc_j_2 += scalar_t(0.7071067811865483) * (r3 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.7071067811865483) * (r3 * r23) * r16;
            // inst 398: bwd_fma_resident | path#71: gw[9] += go[26]*x[3]*y[2], gx[3] += w[9]*go[26]*y[2], gy[2] += w[9]*go[26]*x[3]
            gw_acc_i_9 += scalar_t(0.7071067811865468) * r23 * r8 * r18;
            gx_acc_j_3 += scalar_t(0.7071067811865468) * (r3 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.7071067811865468) * (r3 * r23) * r8;
            // inst 399: release | last use after path#71
            // inst 400: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 24)")
            r23 = grad_out[go_base + (index_t)5376 + (index_t)u];
            // inst 401: bwd_fma_resident | path#65: gw[9] += go[24]*x[1]*y[2], gx[1] += w[9]*go[24]*y[2], gy[2] += w[9]*go[24]*x[1]
            gw_acc_i_9 += scalar_t(0.7071067811865469) * r23 * r13 * r18;
            gx_acc_j_1 += scalar_t(0.7071067811865469) * (r3 * r23) * r18;
            gy_acc_k_2 += scalar_t(0.7071067811865469) * (r3 * r23) * r13;
            // inst 402: bwd_fma_resident | path#66: gw[9] += go[24]*x[2]*y[1], gx[2] += w[9]*go[24]*y[1], gy[1] += w[9]*go[24]*x[2]
            gw_acc_i_9 += scalar_t(0.707106781186547) * r23 * r16 * r27;
            gx_acc_j_2 += scalar_t(0.707106781186547) * (r3 * r23) * r27;
            gy_acc_k_1 += scalar_t(0.707106781186547) * (r3 * r23) * r16;
            // inst 403: release | last use after path#66
            // inst 404: load | global key=(2, 0, (0, 2, 6, 0, 0, 2), "('go', 23)")
            r23 = grad_out[go_base + (index_t)5152 + (index_t)u];
            // inst 405: bwd_fma_resident | path#63: gw[9] += go[23]*x[1]*y[3], gx[1] += w[9]*go[23]*y[3], gy[3] += w[9]*go[23]*x[1]
            gw_acc_i_9 += scalar_t(0.7071067811865479) * r23 * r13 * r2;
            gx_acc_j_1 += scalar_t(0.7071067811865479) * (r3 * r23) * r2;
            gy_acc_k_3 += scalar_t(0.7071067811865479) * (r3 * r23) * r13;
            // inst 406: bwd_fma_resident | path#64: gw[9] += go[23]*x[3]*y[1], gx[3] += w[9]*go[23]*y[1], gy[1] += w[9]*go[23]*x[3]
            gw_acc_i_9 += scalar_t(0.7071067811865478) * r23 * r8 * r27;
            gx_acc_j_3 += scalar_t(0.7071067811865478) * (r3 * r23) * r27;
            gy_acc_k_1 += scalar_t(0.7071067811865478) * (r3 * r23) * r8;
            // inst 407: release | last use after path#64
            // inst 408: release | last use after path#64
            // inst 409: load | global key=(0, 0, (0, 0, 10, 0, -1, 5), "('w', 2)")
            r23 = w[w_base + (index_t)448 + (index_t)u];
            // inst 410: load | global key=(5, 2, (20, 5, 15, 0, 0, 5), "('go', 2)")
            r3 = grad_out[go_base + (index_t)448 + (index_t)u];
            // inst 411: bwd_fma_resident | path#4: gw[2] += go[2]*x[4]*y[4], gx[4] += w[2]*go[2]*y[4], gy[4] += w[2]*go[2]*x[4]
            gw_acc_i_2 += scalar_t(0.4472135954999577) * r3 * r22 * r14;
            gx_acc_j_4 += scalar_t(0.4472135954999577) * (r23 * r3) * r14;
            gy_acc_k_4 += scalar_t(0.4472135954999577) * (r23 * r3) * r22;
            // inst 412: release | last use after path#4
            // inst 413: release | last use after path#4
            // inst 414: bwd_fma_resident | path#5: gw[2] += go[2]*x[5]*y[5], gx[5] += w[2]*go[2]*y[5], gy[5] += w[2]*go[2]*x[5]
            gw_acc_i_2 += scalar_t(0.44721359549995815) * r3 * r10 * r5;
            gx_acc_j_5 += scalar_t(0.44721359549995815) * (r23 * r3) * r5;
            gy_acc_k_5 += scalar_t(0.44721359549995815) * (r23 * r3) * r10;
            // inst 415: release | last use after path#5
            // inst 416: release | last use after path#5
            // inst 417: bwd_fma_resident | path#6: gw[2] += go[2]*x[6]*y[6], gx[6] += w[2]*go[2]*y[6], gy[6] += w[2]*go[2]*x[6]
            gw_acc_i_2 += scalar_t(0.44721359549995787) * r3 * r19 * r11;
            gx_acc_j_6 += scalar_t(0.44721359549995787) * (r23 * r3) * r11;
            gy_acc_k_6 += scalar_t(0.44721359549995787) * (r23 * r3) * r19;
            // inst 418: release | last use after path#6
            // inst 419: release | last use after path#6
            // inst 420: bwd_fma_resident | path#7: gw[2] += go[2]*x[7]*y[7], gx[7] += w[2]*go[2]*y[7], gy[7] += w[2]*go[2]*x[7]
            gw_acc_i_2 += scalar_t(0.447213595499958) * r3 * r7 * r17;
            gx_acc_j_7 += scalar_t(0.447213595499958) * (r23 * r3) * r17;
            gy_acc_k_7 += scalar_t(0.447213595499958) * (r23 * r3) * r7;
            // inst 421: release | last use after path#7
            // inst 422: release | last use after path#7
            // inst 423: bwd_fma_resident | path#8: gw[2] += go[2]*x[8]*y[8], gx[8] += w[2]*go[2]*y[8], gy[8] += w[2]*go[2]*x[8]
            gw_acc_i_2 += scalar_t(0.4472135954999582) * r3 * r21 * r9;
            gx_acc_j_8 += scalar_t(0.4472135954999582) * (r23 * r3) * r9;
            gy_acc_k_8 += scalar_t(0.4472135954999582) * (r23 * r3) * r21;
            // inst 424: release | last use after path#8
            // inst 425: release | last use after path#8
            // inst 426: release | last use after path#8
            // inst 427: release | last use after path#8
            // inst 428: load | global key=(0, 0, (0, 0, 6, 0, -1, 3), "('w', 1)")
            r3 = w[w_base + (index_t)224 + (index_t)u];
            // inst 429: load | global key=(3, 2, (12, 3, 9, 0, 0, 3), "('go', 1)")
            r9 = grad_out[go_base + (index_t)224 + (index_t)u];
            // inst 430: bwd_fma_resident | path#1: gw[1] += go[1]*x[1]*y[1], gx[1] += w[1]*go[1]*y[1], gy[1] += w[1]*go[1]*x[1]
            gw_acc_i_1 += scalar_t(0.5773502691896258) * r9 * r13 * r27;
            gx_acc_j_1 += scalar_t(0.5773502691896258) * (r3 * r9) * r27;
            gy_acc_k_1 += scalar_t(0.5773502691896258) * (r3 * r9) * r13;
            // inst 431: release | last use after path#1
            // inst 432: release | last use after path#1
            // inst 433: bwd_fma_resident | path#2: gw[1] += go[1]*x[2]*y[2], gx[2] += w[1]*go[1]*y[2], gy[2] += w[1]*go[1]*x[2]
            gw_acc_i_1 += scalar_t(0.5773502691896256) * r9 * r16 * r18;
            gx_acc_j_2 += scalar_t(0.5773502691896256) * (r3 * r9) * r18;
            gy_acc_k_2 += scalar_t(0.5773502691896256) * (r3 * r9) * r16;
            // inst 434: release | last use after path#2
            // inst 435: release | last use after path#2
            // inst 436: bwd_fma_resident | path#3: gw[1] += go[1]*x[3]*y[3], gx[3] += w[1]*go[1]*y[3], gy[3] += w[1]*go[1]*x[3]
            gw_acc_i_1 += scalar_t(0.5773502691896258) * r9 * r8 * r2;
            gx_acc_j_3 += scalar_t(0.5773502691896258) * (r3 * r9) * r2;
            gy_acc_k_3 += scalar_t(0.5773502691896258) * (r3 * r9) * r8;
            // inst 437: release | last use after path#3
            // inst 438: release | last use after path#3
            // inst 439: release | last use after path#3
            // inst 440: release | last use after path#3

            // mixed register/shared-memory backward accumulator writeback
            grad_w[gw_base + (index_t)0 + (index_t)u] = gw_acc_i_0;
            grad_w[gw_base + (index_t)224 + (index_t)u] = gw_acc_i_1;
            grad_w[gw_base + (index_t)448 + (index_t)u] = gw_acc_i_2;
            grad_w[gw_base + (index_t)672 + (index_t)u] = gw_acc_i_3;
            grad_w[gw_base + (index_t)896 + (index_t)u] = gw_acc_i_4;
            grad_w[gw_base + (index_t)1120 + (index_t)u] = gw_acc_i_5;
            grad_w[gw_base + (index_t)1344 + (index_t)u] = gw_acc_i_6;
            grad_w[gw_base + (index_t)1568 + (index_t)u] = gw_acc_i_7;
            grad_w[gw_base + (index_t)1792 + (index_t)u] = gw_acc_i_8;
            grad_w[gw_base + (index_t)2016 + (index_t)u] = gw_acc_i_9;
            grad_w[gw_base + (index_t)2240 + (index_t)u] = gw_acc_i_10;
            grad_w[gw_base + (index_t)2464 + (index_t)u] = gw_acc_i_11;
            grad_w[gw_base + (index_t)2688 + (index_t)u] = gw_acc_i_12;
            grad_w[gw_base + (index_t)2912 + (index_t)u] = gw_acc_i_13;
            grad_w[gw_base + (index_t)3136 + (index_t)u] = gw_acc_i_14;
            grad_w[gw_base + (index_t)3360 + (index_t)u] = gw_acc_i_15;
            grad_w[gw_base + (index_t)3584 + (index_t)u] = gw_acc_i_16;
            atomicAdd(&grad_x[gx_base + (index_t)0 + (index_t)u], gx_acc_j_0);
            atomicAdd(&grad_x[gx_base + (index_t)224 + (index_t)u], gx_acc_j_1);
            atomicAdd(&grad_x[gx_base + (index_t)448 + (index_t)u], gx_acc_j_2);
            atomicAdd(&grad_x[gx_base + (index_t)672 + (index_t)u], gx_acc_j_3);
            atomicAdd(&grad_x[gx_base + (index_t)896 + (index_t)u], gx_acc_j_4);
            atomicAdd(&grad_x[gx_base + (index_t)1120 + (index_t)u], gx_acc_j_5);
            atomicAdd(&grad_x[gx_base + (index_t)1344 + (index_t)u], gx_acc_j_6);
            atomicAdd(&grad_x[gx_base + (index_t)1568 + (index_t)u], gx_acc_j_7);
            atomicAdd(&grad_x[gx_base + (index_t)1792 + (index_t)u], gx_acc_j_8);
            scalar_t gy_sum_resident_0 = warp_sum_xor_lars_bwd(gy_acc_k_0);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)0], gy_sum_resident_0);
            }
            scalar_t gy_sum_resident_1 = warp_sum_xor_lars_bwd(gy_acc_k_1);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)1], gy_sum_resident_1);
            }
            scalar_t gy_sum_resident_2 = warp_sum_xor_lars_bwd(gy_acc_k_2);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)2], gy_sum_resident_2);
            }
            scalar_t gy_sum_resident_3 = warp_sum_xor_lars_bwd(gy_acc_k_3);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)3], gy_sum_resident_3);
            }
            scalar_t gy_sum_resident_4 = warp_sum_xor_lars_bwd(gy_acc_k_4);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)4], gy_sum_resident_4);
            }
            scalar_t gy_sum_resident_5 = warp_sum_xor_lars_bwd(gy_acc_k_5);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)5], gy_sum_resident_5);
            }
            scalar_t gy_sum_resident_6 = warp_sum_xor_lars_bwd(gy_acc_k_6);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)6], gy_sum_resident_6);
            }
            scalar_t gy_sum_resident_7 = warp_sum_xor_lars_bwd(gy_acc_k_7);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)7], gy_sum_resident_7);
            }
            scalar_t gy_sum_resident_8 = warp_sum_xor_lars_bwd(gy_acc_k_8);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)8], gy_sum_resident_8);
            }
            scalar_t gy_sum_resident_9 = warp_sum_xor_lars_bwd(gy_acc_k_9);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)9], gy_sum_resident_9);
            }
            scalar_t gy_sum_resident_10 = warp_sum_xor_lars_bwd(gy_acc_k_10);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)10], gy_sum_resident_10);
            }
            scalar_t gy_sum_resident_11 = warp_sum_xor_lars_bwd(gy_acc_k_11);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)11], gy_sum_resident_11);
            }
            scalar_t gy_sum_resident_12 = warp_sum_xor_lars_bwd(gy_acc_k_12);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)12], gy_sum_resident_12);
            }
            scalar_t gy_sum_resident_13 = warp_sum_xor_lars_bwd(gy_acc_k_13);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)13], gy_sum_resident_13);
            }
            scalar_t gy_sum_resident_14 = warp_sum_xor_lars_bwd(gy_acc_k_14);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)14], gy_sum_resident_14);
            }
            scalar_t gy_sum_resident_15 = warp_sum_xor_lars_bwd(gy_acc_k_15);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)15], gy_sum_resident_15);
            }
        }
    }
}

static inline bool mul_fits_int32_bwd(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}

static inline bool mul3_fits_int32_bwd(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32_bwd(a, b)) return false;
    return mul_fits_int32_bwd(a * b, c);
}

static inline bool should_use_int32_index_bwd(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32_bwd(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32_bwd(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool go_ok = mul3_fits_int32_bwd(go_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && go_ok;
}

template <typename scalar_t, typename index_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd_typed(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    constexpr int kSmemAccCount = 0;
    size_t smem_bytes = (size_t)kSmemAccCount * 32u * sizeof(scalar_t);
#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
    if (smem_bytes > 49152) {
        cudaFuncSetAttribute(uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd<scalar_t, index_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
#endif
    uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd<scalar_t, index_t><<<grid, block, smem_bytes, stream>>>(
        w, x, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    constexpr bool kUseXSrc = true;
    constexpr bool kUseYSrc = false;
    constexpr bool kUseScatter = true;
    constexpr bool kModeScalarY = true;
    if (should_use_int32_index_bwd(B, WB, Iw, Ix, Ky, V, U, S,
                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd_typed<scalar_t, int32_t>(
            w, x, y, grad_out, grad_w, grad_x, grad_y,
            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd_typed<scalar_t, int64_t>(
            w, x, y, grad_out, grad_w, grad_x, grad_y,
            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd_auto<scalar_t>(
        w, x, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
}



std::vector<torch::Tensor> launcher_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd(
    torch::Tensor w,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor grad_out,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    int64_t V64)
{
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : [B,Ky,1] or [S,Ky,1]
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");

    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");

    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");


    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int)src_idx.size(0);
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");
    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");


    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd", [&] {

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    GPU_KERNEL_LAUNCH_CHECK();

    return {grad_w, grad_x, grad_y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd, "uniform1d_lars_bwd_all_inputs_accall_u224_path215_uu_u_xsrc1_ysrc0_scatter1_full_bwd backward LARS fused jit impl");
}
