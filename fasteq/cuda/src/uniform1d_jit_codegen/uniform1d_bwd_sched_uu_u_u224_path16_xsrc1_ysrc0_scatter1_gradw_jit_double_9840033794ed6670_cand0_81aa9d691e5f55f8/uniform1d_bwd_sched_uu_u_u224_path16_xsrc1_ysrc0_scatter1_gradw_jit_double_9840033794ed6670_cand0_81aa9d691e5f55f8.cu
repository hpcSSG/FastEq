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
__global__ void uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd(
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

    const index_t w_base  = (index_t)w_row * (index_t)896;
    const index_t gw_base = (index_t)w_row * (index_t)896;
    const index_t x_base  = (index_t)x_row * (index_t)224;
    const index_t gx_base = (index_t)x_row * (index_t)224;
    const index_t y_base  = (index_t)y_row * (index_t)Ky;
    const index_t gy_base = (index_t)y_row * (index_t)Ky;
    const index_t go_base = (index_t)go_row * (index_t)3584;

    for (int u_base = 0; u_base < U; u_base += 32) {
        const int u = u_base + lane;
        if (u < U) {
            scalar_t r0;
            scalar_t r1;
            scalar_t r2;
            scalar_t r3;
            scalar_t r4;
            scalar_t r5;

            scalar_t gw_acc_i_0 = scalar_t(0);
            scalar_t gw_acc_i_1 = scalar_t(0);
            scalar_t gw_acc_i_2 = scalar_t(0);
            scalar_t gw_acc_i_3 = scalar_t(0);
            scalar_t gx_acc_j_0 = scalar_t(0);
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

            // inst 0: load | global key=(0, 0, (0, 0, 0, 0, -3, 1), "('y', 9)")
            r0 = y[y_base + (index_t)9];
            // inst 1: load | global key=(0, 0, (0, 0, 1, 0, -2, 1), "('go', 9)")
            r1 = grad_out[go_base + (index_t)2016 + (index_t)u];
            // inst 2: load | global key=(0, 0, (0, 0, 2, 0, -13, 7), "('w', 3)")
            r2 = w[w_base + (index_t)672 + (index_t)u];
            // inst 3: load | global key=(1, 2, (4, 1, 9, 0, -33, 16), "('x', 0)")
            r3 = x[x_base + (index_t)0 + (index_t)u];
            // inst 4: bwd_fma_resident | path#9: gw[3] += go[9]*x[0]*y[9], gx[0] += w[3]*go[9]*y[9], gy[9] += w[3]*go[9]*x[0]
            gw_acc_i_3 += scalar_t(1.0000000000000007) * r1 * r3 * r0;
            gx_acc_j_0 += scalar_t(1.0000000000000007) * (r2 * r1) * r0;
            gy_acc_k_9 += scalar_t(1.0000000000000007) * (r2 * r1) * r3;
            // inst 5: release | last use after path#9
            // inst 6: release | last use after path#9
            // inst 7: load | global key=(0, 0, (0, 0, 5, 0, -10, 5), "('w', 2)")
            r1 = w[w_base + (index_t)448 + (index_t)u];
            // inst 8: load | global key=(0, 0, (0, 0, 3, 0, -6, 3), "('w', 1)")
            r0 = w[w_base + (index_t)224 + (index_t)u];
            // inst 9: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 8)")
            r4 = y[y_base + (index_t)8];
            // inst 10: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 8)")
            r5 = grad_out[go_base + (index_t)1792 + (index_t)u];
            // inst 11: bwd_fma_resident | path#8: gw[2] += go[8]*x[0]*y[8], gx[0] += w[2]*go[8]*y[8], gy[8] += w[2]*go[8]*x[0]
            gw_acc_i_2 += scalar_t(1.0000000000000004) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(1.0000000000000004) * (r1 * r5) * r4;
            gy_acc_k_8 += scalar_t(1.0000000000000004) * (r1 * r5) * r3;
            // inst 12: release | last use after path#8
            // inst 13: release | last use after path#8
            // inst 14: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 7)")
            r5 = y[y_base + (index_t)7];
            // inst 15: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 7)")
            r4 = grad_out[go_base + (index_t)1568 + (index_t)u];
            // inst 16: bwd_fma_resident | path#7: gw[2] += go[7]*x[0]*y[7], gx[0] += w[2]*go[7]*y[7], gy[7] += w[2]*go[7]*x[0]
            gw_acc_i_2 += scalar_t(1.0) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(1.0) * (r1 * r4) * r5;
            gy_acc_k_7 += scalar_t(1.0) * (r1 * r4) * r3;
            // inst 17: release | last use after path#7
            // inst 18: release | last use after path#7
            // inst 19: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 6)")
            r4 = y[y_base + (index_t)6];
            // inst 20: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 6)")
            r5 = grad_out[go_base + (index_t)1344 + (index_t)u];
            // inst 21: bwd_fma_resident | path#6: gw[2] += go[6]*x[0]*y[6], gx[0] += w[2]*go[6]*y[6], gy[6] += w[2]*go[6]*x[0]
            gw_acc_i_2 += scalar_t(0.9999999999999997) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(0.9999999999999997) * (r1 * r5) * r4;
            gy_acc_k_6 += scalar_t(0.9999999999999997) * (r1 * r5) * r3;
            // inst 22: release | last use after path#6
            // inst 23: release | last use after path#6
            // inst 24: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 5)")
            r5 = y[y_base + (index_t)5];
            // inst 25: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 5)")
            r4 = grad_out[go_base + (index_t)1120 + (index_t)u];
            // inst 26: bwd_fma_resident | path#5: gw[2] += go[5]*x[0]*y[5], gx[0] += w[2]*go[5]*y[5], gy[5] += w[2]*go[5]*x[0]
            gw_acc_i_2 += scalar_t(1.0000000000000002) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(1.0000000000000002) * (r1 * r4) * r5;
            gy_acc_k_5 += scalar_t(1.0000000000000002) * (r1 * r4) * r3;
            // inst 27: release | last use after path#5
            // inst 28: release | last use after path#5
            // inst 29: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 4)")
            r4 = y[y_base + (index_t)4];
            // inst 30: load | global key=(1, 3, (5, 1, 3, 0, 0, 1), "('go', 4)")
            r5 = grad_out[go_base + (index_t)896 + (index_t)u];
            // inst 31: bwd_fma_resident | path#4: gw[2] += go[4]*x[0]*y[4], gx[0] += w[2]*go[4]*y[4], gy[4] += w[2]*go[4]*x[0]
            gw_acc_i_2 += scalar_t(0.9999999999999993) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(0.9999999999999993) * (r1 * r5) * r4;
            gy_acc_k_4 += scalar_t(0.9999999999999993) * (r1 * r5) * r3;
            // inst 32: release | last use after path#4
            // inst 33: release | last use after path#4
            // inst 34: release | last use after path#4
            // inst 35: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 3)")
            r5 = y[y_base + (index_t)3];
            // inst 36: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 3)")
            r4 = grad_out[go_base + (index_t)672 + (index_t)u];
            // inst 37: bwd_fma_resident | path#3: gw[1] += go[3]*x[0]*y[3], gx[0] += w[1]*go[3]*y[3], gy[3] += w[1]*go[3]*x[0]
            gw_acc_i_1 += scalar_t(1.0000000000000002) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(1.0000000000000002) * (r0 * r4) * r5;
            gy_acc_k_3 += scalar_t(1.0000000000000002) * (r0 * r4) * r3;
            // inst 38: release | last use after path#3
            // inst 39: release | last use after path#3
            // inst 40: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 2)")
            r4 = y[y_base + (index_t)2];
            // inst 41: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 2)")
            r5 = grad_out[go_base + (index_t)448 + (index_t)u];
            // inst 42: bwd_fma_resident | path#2: gw[1] += go[2]*x[0]*y[2], gx[0] += w[1]*go[2]*y[2], gy[2] += w[1]*go[2]*x[0]
            gw_acc_i_1 += scalar_t(0.9999999999999998) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(0.9999999999999998) * (r0 * r5) * r4;
            gy_acc_k_2 += scalar_t(0.9999999999999998) * (r0 * r5) * r3;
            // inst 43: release | last use after path#2
            // inst 44: release | last use after path#2
            // inst 45: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 15)")
            r5 = y[y_base + (index_t)15];
            // inst 46: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 15)")
            r4 = grad_out[go_base + (index_t)3360 + (index_t)u];
            // inst 47: bwd_fma_resident | path#15: gw[3] += go[15]*x[0]*y[15], gx[0] += w[3]*go[15]*y[15], gy[15] += w[3]*go[15]*x[0]
            gw_acc_i_3 += scalar_t(1.0000000000000004) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(1.0000000000000004) * (r2 * r4) * r5;
            gy_acc_k_15 += scalar_t(1.0000000000000004) * (r2 * r4) * r3;
            // inst 48: release | last use after path#15
            // inst 49: release | last use after path#15
            // inst 50: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 14)")
            r4 = y[y_base + (index_t)14];
            // inst 51: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 14)")
            r5 = grad_out[go_base + (index_t)3136 + (index_t)u];
            // inst 52: bwd_fma_resident | path#14: gw[3] += go[14]*x[0]*y[14], gx[0] += w[3]*go[14]*y[14], gy[14] += w[3]*go[14]*x[0]
            gw_acc_i_3 += scalar_t(0.9999999999999998) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(0.9999999999999998) * (r2 * r5) * r4;
            gy_acc_k_14 += scalar_t(0.9999999999999998) * (r2 * r5) * r3;
            // inst 53: release | last use after path#14
            // inst 54: release | last use after path#14
            // inst 55: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 13)")
            r5 = y[y_base + (index_t)13];
            // inst 56: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 13)")
            r4 = grad_out[go_base + (index_t)2912 + (index_t)u];
            // inst 57: bwd_fma_resident | path#13: gw[3] += go[13]*x[0]*y[13], gx[0] += w[3]*go[13]*y[13], gy[13] += w[3]*go[13]*x[0]
            gw_acc_i_3 += scalar_t(0.9999999999999996) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(0.9999999999999996) * (r2 * r4) * r5;
            gy_acc_k_13 += scalar_t(0.9999999999999996) * (r2 * r4) * r3;
            // inst 58: release | last use after path#13
            // inst 59: release | last use after path#13
            // inst 60: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 12)")
            r4 = y[y_base + (index_t)12];
            // inst 61: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 12)")
            r5 = grad_out[go_base + (index_t)2688 + (index_t)u];
            // inst 62: bwd_fma_resident | path#12: gw[3] += go[12]*x[0]*y[12], gx[0] += w[3]*go[12]*y[12], gy[12] += w[3]*go[12]*x[0]
            gw_acc_i_3 += scalar_t(0.9999999999999996) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(0.9999999999999996) * (r2 * r5) * r4;
            gy_acc_k_12 += scalar_t(0.9999999999999996) * (r2 * r5) * r3;
            // inst 63: release | last use after path#12
            // inst 64: release | last use after path#12
            // inst 65: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 11)")
            r5 = y[y_base + (index_t)11];
            // inst 66: load | global key=(1, 2, (3, 1, 3, 0, 0, 1), "('go', 11)")
            r4 = grad_out[go_base + (index_t)2464 + (index_t)u];
            // inst 67: bwd_fma_resident | path#11: gw[3] += go[11]*x[0]*y[11], gx[0] += w[3]*go[11]*y[11], gy[11] += w[3]*go[11]*x[0]
            gw_acc_i_3 += scalar_t(0.9999999999999991) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(0.9999999999999991) * (r2 * r4) * r5;
            gy_acc_k_11 += scalar_t(0.9999999999999991) * (r2 * r4) * r3;
            // inst 68: release | last use after path#11
            // inst 69: release | last use after path#11
            // inst 70: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 10)")
            r4 = y[y_base + (index_t)10];
            // inst 71: load | global key=(1, 3, (5, 1, 3, 0, 0, 1), "('go', 10)")
            r5 = grad_out[go_base + (index_t)2240 + (index_t)u];
            // inst 72: bwd_fma_resident | path#10: gw[3] += go[10]*x[0]*y[10], gx[0] += w[3]*go[10]*y[10], gy[10] += w[3]*go[10]*x[0]
            gw_acc_i_3 += scalar_t(1.0000000000000004) * r5 * r3 * r4;
            gx_acc_j_0 += scalar_t(1.0000000000000004) * (r2 * r5) * r4;
            gy_acc_k_10 += scalar_t(1.0000000000000004) * (r2 * r5) * r3;
            // inst 73: release | last use after path#10
            // inst 74: release | last use after path#10
            // inst 75: release | last use after path#10
            // inst 76: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('y', 1)")
            r5 = y[y_base + (index_t)1];
            // inst 77: load | global key=(1, 3, (5, 1, 3, 0, 0, 1), "('go', 1)")
            r4 = grad_out[go_base + (index_t)224 + (index_t)u];
            // inst 78: bwd_fma_resident | path#1: gw[1] += go[1]*x[0]*y[1], gx[0] += w[1]*go[1]*y[1], gy[1] += w[1]*go[1]*x[0]
            gw_acc_i_1 += scalar_t(1.0000000000000002) * r4 * r3 * r5;
            gx_acc_j_0 += scalar_t(1.0000000000000002) * (r0 * r4) * r5;
            gy_acc_k_1 += scalar_t(1.0000000000000002) * (r0 * r4) * r3;
            // inst 79: release | last use after path#1
            // inst 80: release | last use after path#1
            // inst 81: release | last use after path#1
            // inst 82: load | global key=(0, 0, (0, 0, 1, 0, -2, 1), "('y', 0)")
            r4 = y[y_base + (index_t)0];
            // inst 83: load | global key=(0, 0, (0, 0, 2, 0, -1, 1), "('w', 0)")
            r5 = w[w_base + (index_t)0 + (index_t)u];
            // inst 84: load | global key=(1, 4, (7, 1, 3, 0, 0, 1), "('go', 0)")
            r0 = grad_out[go_base + (index_t)0 + (index_t)u];
            // inst 85: bwd_fma_resident | path#0: gw[0] += go[0]*x[0]*y[0], gx[0] += w[0]*go[0]*y[0], gy[0] += w[0]*go[0]*x[0]
            gw_acc_i_0 += scalar_t(1.0) * r0 * r3 * r4;
            gx_acc_j_0 += scalar_t(1.0) * (r5 * r0) * r4;
            gy_acc_k_0 += scalar_t(1.0) * (r5 * r0) * r3;
            // inst 86: release | last use after path#0
            // inst 87: release | last use after path#0
            // inst 88: release | last use after path#0
            // inst 89: release | last use after path#0

            // mixed register/shared-memory backward accumulator writeback
            grad_w[gw_base + (index_t)0 + (index_t)u] = gw_acc_i_0;
            grad_w[gw_base + (index_t)224 + (index_t)u] = gw_acc_i_1;
            grad_w[gw_base + (index_t)448 + (index_t)u] = gw_acc_i_2;
            grad_w[gw_base + (index_t)672 + (index_t)u] = gw_acc_i_3;
            atomicAdd(&grad_x[gx_base + (index_t)0 + (index_t)u], gx_acc_j_0);
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
void launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd_typed(
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
        cudaFuncSetAttribute(uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd<scalar_t, index_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
#endif
    uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd<scalar_t, index_t><<<grid, block, smem_bytes, stream>>>(
        w, x, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd_auto(
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
        launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd_typed<scalar_t, int32_t>(
            w, x, y, grad_out, grad_w, grad_x, grad_y,
            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd_typed<scalar_t, int64_t>(
            w, x, y, grad_out, grad_w, grad_x, grad_y,
            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd(
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
    launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd_auto<scalar_t>(
        w, x, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
}



std::vector<torch::Tensor> launcher_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd(
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

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd", [&] {

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd<scalar_t>(
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
    m.def("run", &launcher_uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd, "uniform1d_lars_bwd_all_inputs_accall_u224_path16_uu_u_xsrc1_ysrc0_scatter1_full_bwd backward LARS fused jit impl");
}
