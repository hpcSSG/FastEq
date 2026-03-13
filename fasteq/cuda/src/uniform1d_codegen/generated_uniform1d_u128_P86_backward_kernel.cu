#include <stdint.h>
#include <cuda_runtime.h>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void generated_uniform1d_u128_P86_backward_kernel(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x_all,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_w,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_y,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V, int U)
{
    int b_global = (int)blockIdx.x;
    if (b_global >= B) return;

    int b   = b_list ? b_list[b_global] : b_global;
    int src = src_idx[b];
    int dst = dst_idx[b];
    int lane = ((int)threadIdx.x) & 31;
    int warp = ((int)threadIdx.x) >> 5;
    int64_t y_base = (int64_t)b * Ky;

    // preload y(k), independent of u
    scalar_t yk_4 = y[y_base + 4];
    scalar_t yk_8 = y[y_base + 8];
    scalar_t yk_5 = y[y_base + 5];
    scalar_t yk_6 = y[y_base + 6];
    scalar_t yk_7 = y[y_base + 7];
    scalar_t yk_1 = y[y_base + 1];
    scalar_t yk_3 = y[y_base + 3];
    scalar_t yk_2 = y[y_base + 2];
    scalar_t yk_11 = y[y_base + 11];
    scalar_t yk_13 = y[y_base + 13];
    scalar_t yk_0 = y[y_base + 0];
    scalar_t yk_10 = y[y_base + 10];
    scalar_t yk_12 = y[y_base + 12];
    scalar_t yk_14 = y[y_base + 14];
    scalar_t yk_9 = y[y_base + 9];
    scalar_t yk_15 = y[y_base + 15];

    // init grad_y accumulators
    scalar_t gy_acc_k_4 = scalar_t(0);
    scalar_t gy_acc_k_8 = scalar_t(0);
    scalar_t gy_acc_k_5 = scalar_t(0);
    scalar_t gy_acc_k_6 = scalar_t(0);
    scalar_t gy_acc_k_7 = scalar_t(0);
    scalar_t gy_acc_k_1 = scalar_t(0);
    scalar_t gy_acc_k_3 = scalar_t(0);
    scalar_t gy_acc_k_2 = scalar_t(0);
    scalar_t gy_acc_k_11 = scalar_t(0);
    scalar_t gy_acc_k_13 = scalar_t(0);
    scalar_t gy_acc_k_0 = scalar_t(0);
    scalar_t gy_acc_k_10 = scalar_t(0);
    scalar_t gy_acc_k_12 = scalar_t(0);
    scalar_t gy_acc_k_14 = scalar_t(0);
    scalar_t gy_acc_k_9 = scalar_t(0);
    scalar_t gy_acc_k_15 = scalar_t(0);

    // u traversal: persistent inner loop over U
    for (int u_base = 0; u_base < U; u_base += 32) {
        int u = u_base + lane;
        if (u < U) {
            // preload w(i,u)
            scalar_t wi_7 = w[((int64_t)b * Iw + 7) * (int64_t)U + u];
            scalar_t wi_9 = w[((int64_t)b * Iw + 9) * (int64_t)U + u];
            scalar_t wi_4 = w[((int64_t)b * Iw + 4) * (int64_t)U + u];
            scalar_t wi_6 = w[((int64_t)b * Iw + 6) * (int64_t)U + u];
            scalar_t wi_8 = w[((int64_t)b * Iw + 8) * (int64_t)U + u];
            scalar_t wi_5 = w[((int64_t)b * Iw + 5) * (int64_t)U + u];
            scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
            scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
            scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];
            scalar_t wi_0 = w[((int64_t)b * Iw + 0) * (int64_t)U + u];
        
            // preload x(j,u)
            scalar_t xj_1 = x_all[((int64_t)src * Ix + 1) * (int64_t)U + u];
            scalar_t xj_3 = x_all[((int64_t)src * Ix + 3) * (int64_t)U + u];
            scalar_t xj_2 = x_all[((int64_t)src * Ix + 2) * (int64_t)U + u];
            scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];
        
            // init per-u accumulators
            scalar_t gw_acc_i_7 = scalar_t(0);
            scalar_t gw_acc_i_9 = scalar_t(0);
            scalar_t gw_acc_i_4 = scalar_t(0);
            scalar_t gw_acc_i_6 = scalar_t(0);
            scalar_t gw_acc_i_8 = scalar_t(0);
            scalar_t gw_acc_i_5 = scalar_t(0);
            scalar_t gw_acc_i_1 = scalar_t(0);
            scalar_t gw_acc_i_2 = scalar_t(0);
            scalar_t gw_acc_i_3 = scalar_t(0);
            scalar_t gw_acc_i_0 = scalar_t(0);
            scalar_t gx_acc_j_1 = scalar_t(0);
            scalar_t gx_acc_j_3 = scalar_t(0);
            scalar_t gx_acc_j_2 = scalar_t(0);
            scalar_t gx_acc_j_0 = scalar_t(0);
        
            // ---- v tile 0 ----
            scalar_t go_v_0 = grad_out[((int64_t)dst * V + 0) * (int64_t)U + u];
            scalar_t go_v_1 = grad_out[((int64_t)dst * V + 1) * (int64_t)U + u];
            scalar_t go_v_2 = grad_out[((int64_t)dst * V + 2) * (int64_t)U + u];
            scalar_t go_v_3 = grad_out[((int64_t)dst * V + 3) * (int64_t)U + u];
            scalar_t go_v_4 = grad_out[((int64_t)dst * V + 4) * (int64_t)U + u];
            scalar_t go_v_5 = grad_out[((int64_t)dst * V + 5) * (int64_t)U + u];
            scalar_t go_v_6 = grad_out[((int64_t)dst * V + 6) * (int64_t)U + u];
            scalar_t go_v_7 = grad_out[((int64_t)dst * V + 7) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
            gw_acc_i_1 += scalar_t(0.57735026918962573) * xj_1 * yk_1 * go_v_1;
            gw_acc_i_1 += scalar_t(0.57735026918962584) * xj_2 * yk_2 * go_v_1;
            gw_acc_i_1 += scalar_t(0.57735026918962573) * xj_3 * yk_3 * go_v_1;
            gw_acc_i_2 += xj_0 * yk_1 * go_v_2;
            gw_acc_i_2 += xj_0 * yk_2 * go_v_3;
            gw_acc_i_2 += xj_0 * yk_3 * go_v_4;
        
            // grad_w i-tile 4
            gw_acc_i_3 += xj_1 * yk_0 * go_v_5;
            gw_acc_i_3 += xj_2 * yk_0 * go_v_6;
            gw_acc_i_3 += xj_3 * yk_0 * go_v_7;
            gw_acc_i_0 += xj_0 * yk_0 * go_v_0;
        
            // grad_x j-tile 0
            gx_acc_j_1 += scalar_t(0.57735026918962573) * wi_1 * yk_1 * go_v_1;
            gx_acc_j_1 += wi_3 * yk_0 * go_v_5;
            gx_acc_j_3 += scalar_t(0.57735026918962573) * wi_1 * yk_3 * go_v_1;
            gx_acc_j_3 += wi_3 * yk_0 * go_v_7;
        
            // grad_x j-tile 1
            gx_acc_j_2 += scalar_t(0.57735026918962584) * wi_1 * yk_2 * go_v_1;
            gx_acc_j_2 += wi_3 * yk_0 * go_v_6;
            gx_acc_j_0 += wi_0 * yk_0 * go_v_0;
            gx_acc_j_0 += wi_2 * yk_1 * go_v_2;
            gx_acc_j_0 += wi_2 * yk_2 * go_v_3;
            gx_acc_j_0 += wi_2 * yk_3 * go_v_4;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
            gy_acc_k_1 += scalar_t(0.57735026918962573) * wi_1 * xj_1 * go_v_1;
            gy_acc_k_1 += wi_2 * xj_0 * go_v_2;
        
            // grad_y k-tile 3
            gy_acc_k_3 += scalar_t(0.57735026918962573) * wi_1 * xj_3 * go_v_1;
            gy_acc_k_3 += wi_2 * xj_0 * go_v_4;
            gy_acc_k_2 += scalar_t(0.57735026918962584) * wi_1 * xj_2 * go_v_1;
            gy_acc_k_2 += wi_2 * xj_0 * go_v_3;
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
            gy_acc_k_0 += wi_0 * xj_0 * go_v_0;
            gy_acc_k_0 += wi_3 * xj_1 * go_v_5;
            gy_acc_k_0 += wi_3 * xj_2 * go_v_6;
            gy_acc_k_0 += wi_3 * xj_3 * go_v_7;
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
        
            // ---- v tile 1 ----
            scalar_t go_v_10 = grad_out[((int64_t)dst * V + 10) * (int64_t)U + u];
            scalar_t go_v_9 = grad_out[((int64_t)dst * V + 9) * (int64_t)U + u];
            scalar_t go_v_8 = grad_out[((int64_t)dst * V + 8) * (int64_t)U + u];
            scalar_t go_v_11 = grad_out[((int64_t)dst * V + 11) * (int64_t)U + u];
            scalar_t go_v_12 = grad_out[((int64_t)dst * V + 12) * (int64_t)U + u];
            scalar_t go_v_13 = grad_out[((int64_t)dst * V + 13) * (int64_t)U + u];
            scalar_t go_v_14 = grad_out[((int64_t)dst * V + 14) * (int64_t)U + u];
            scalar_t go_v_15 = grad_out[((int64_t)dst * V + 15) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
            gw_acc_i_4 += scalar_t(-0.31622776601683805) * xj_1 * yk_6 * go_v_8;
            gw_acc_i_4 += scalar_t(-0.54772255750516607) * xj_1 * yk_8 * go_v_8;
            gw_acc_i_4 += scalar_t(0.54772255750516619) * xj_2 * yk_5 * go_v_8;
            gw_acc_i_4 += scalar_t(0.54772255750516607) * xj_3 * yk_4 * go_v_8;
            gw_acc_i_4 += scalar_t(0.54772255750516619) * xj_1 * yk_5 * go_v_9;
            gw_acc_i_4 += scalar_t(0.63245553203367622) * xj_2 * yk_6 * go_v_9;
            gw_acc_i_4 += scalar_t(0.54772255750516619) * xj_3 * yk_7 * go_v_9;
            gw_acc_i_4 += scalar_t(0.54772255750516607) * xj_1 * yk_4 * go_v_10;
            gw_acc_i_4 += scalar_t(0.54772255750516619) * xj_2 * yk_7 * go_v_10;
            gw_acc_i_4 += scalar_t(-0.31622776601683805) * xj_3 * yk_6 * go_v_10;
            gw_acc_i_4 += scalar_t(0.54772255750516607) * xj_3 * yk_8 * go_v_10;
        
            // grad_w i-tile 2
            gw_acc_i_5 += xj_0 * yk_4 * go_v_11;
            gw_acc_i_5 += xj_0 * yk_5 * go_v_12;
            gw_acc_i_5 += xj_0 * yk_6 * go_v_13;
            gw_acc_i_5 += xj_0 * yk_7 * go_v_14;
            gw_acc_i_5 += xj_0 * yk_8 * go_v_15;
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_x j-tile 0
            gx_acc_j_1 += scalar_t(-0.31622776601683805) * wi_4 * yk_6 * go_v_8;
            gx_acc_j_1 += scalar_t(-0.54772255750516607) * wi_4 * yk_8 * go_v_8;
            gx_acc_j_1 += scalar_t(0.54772255750516619) * wi_4 * yk_5 * go_v_9;
            gx_acc_j_1 += scalar_t(0.54772255750516607) * wi_4 * yk_4 * go_v_10;
            gx_acc_j_3 += scalar_t(0.54772255750516607) * wi_4 * yk_4 * go_v_8;
            gx_acc_j_3 += scalar_t(0.54772255750516619) * wi_4 * yk_7 * go_v_9;
            gx_acc_j_3 += scalar_t(-0.31622776601683805) * wi_4 * yk_6 * go_v_10;
            gx_acc_j_3 += scalar_t(0.54772255750516607) * wi_4 * yk_8 * go_v_10;
        
            // grad_x j-tile 1
            gx_acc_j_2 += scalar_t(0.54772255750516619) * wi_4 * yk_5 * go_v_8;
            gx_acc_j_2 += scalar_t(0.63245553203367622) * wi_4 * yk_6 * go_v_9;
            gx_acc_j_2 += scalar_t(0.54772255750516619) * wi_4 * yk_7 * go_v_10;
            gx_acc_j_0 += wi_5 * yk_4 * go_v_11;
            gx_acc_j_0 += wi_5 * yk_5 * go_v_12;
            gx_acc_j_0 += wi_5 * yk_6 * go_v_13;
            gx_acc_j_0 += wi_5 * yk_7 * go_v_14;
            gx_acc_j_0 += wi_5 * yk_8 * go_v_15;
        
            // grad_y k-tile 0
            gy_acc_k_4 += scalar_t(0.54772255750516607) * wi_4 * xj_3 * go_v_8;
            gy_acc_k_4 += scalar_t(0.54772255750516607) * wi_4 * xj_1 * go_v_10;
            gy_acc_k_4 += wi_5 * xj_0 * go_v_11;
            gy_acc_k_8 += scalar_t(-0.54772255750516607) * wi_4 * xj_1 * go_v_8;
            gy_acc_k_8 += scalar_t(0.54772255750516607) * wi_4 * xj_3 * go_v_10;
            gy_acc_k_8 += wi_5 * xj_0 * go_v_15;
        
            // grad_y k-tile 1
            gy_acc_k_5 += scalar_t(0.54772255750516619) * wi_4 * xj_2 * go_v_8;
            gy_acc_k_5 += scalar_t(0.54772255750516619) * wi_4 * xj_1 * go_v_9;
            gy_acc_k_5 += wi_5 * xj_0 * go_v_12;
            gy_acc_k_6 += scalar_t(-0.31622776601683805) * wi_4 * xj_1 * go_v_8;
            gy_acc_k_6 += scalar_t(0.63245553203367622) * wi_4 * xj_2 * go_v_9;
            gy_acc_k_6 += scalar_t(-0.31622776601683805) * wi_4 * xj_3 * go_v_10;
            gy_acc_k_6 += wi_5 * xj_0 * go_v_13;
        
            // grad_y k-tile 2
            gy_acc_k_7 += scalar_t(0.54772255750516619) * wi_4 * xj_3 * go_v_9;
            gy_acc_k_7 += scalar_t(0.54772255750516619) * wi_4 * xj_2 * go_v_10;
            gy_acc_k_7 += wi_5 * xj_0 * go_v_14;
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
        
            // ---- v tile 2 ----
            scalar_t go_v_18 = grad_out[((int64_t)dst * V + 18) * (int64_t)U + u];
            scalar_t go_v_20 = grad_out[((int64_t)dst * V + 20) * (int64_t)U + u];
            scalar_t go_v_17 = grad_out[((int64_t)dst * V + 17) * (int64_t)U + u];
            scalar_t go_v_16 = grad_out[((int64_t)dst * V + 16) * (int64_t)U + u];
            scalar_t go_v_19 = grad_out[((int64_t)dst * V + 19) * (int64_t)U + u];
            scalar_t go_v_25 = grad_out[((int64_t)dst * V + 25) * (int64_t)U + u];
            scalar_t go_v_24 = grad_out[((int64_t)dst * V + 24) * (int64_t)U + u];
            scalar_t go_v_23 = grad_out[((int64_t)dst * V + 23) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_7 += scalar_t(0.53452248382484879) * xj_1 * yk_11 * go_v_23;
            gw_acc_i_7 += scalar_t(0.6546536707079772) * xj_2 * yk_12 * go_v_23;
            gw_acc_i_7 += scalar_t(0.53452248382484879) * xj_3 * yk_13 * go_v_23;
            gw_acc_i_7 += scalar_t(0.48795003647426655) * xj_1 * yk_10 * go_v_24;
            gw_acc_i_7 += scalar_t(0.61721339984836765) * xj_2 * yk_13 * go_v_24;
            gw_acc_i_7 += scalar_t(-0.37796447300922731) * xj_3 * yk_12 * go_v_24;
            gw_acc_i_7 += scalar_t(0.48795003647426655) * xj_3 * yk_14 * go_v_24;
            gw_acc_i_7 += scalar_t(0.59761430466719678) * xj_1 * yk_9 * go_v_25;
            gw_acc_i_7 += scalar_t(0.15430334996209191) * xj_1 * yk_11 * go_v_25;
            gw_acc_i_7 += scalar_t(0.48795003647426666) * xj_2 * yk_14 * go_v_25;
            gw_acc_i_7 += scalar_t(-0.15430334996209191) * xj_3 * yk_13 * go_v_25;
            gw_acc_i_7 += scalar_t(0.59761430466719678) * xj_3 * yk_15 * go_v_25;
        
            // grad_w i-tile 1
            gw_acc_i_6 += scalar_t(0.70710678118654746) * xj_1 * yk_3 * go_v_16;
            gw_acc_i_6 += scalar_t(0.70710678118654746) * xj_3 * yk_1 * go_v_16;
            gw_acc_i_6 += scalar_t(0.70710678118654757) * xj_1 * yk_2 * go_v_17;
            gw_acc_i_6 += scalar_t(0.70710678118654757) * xj_2 * yk_1 * go_v_17;
            gw_acc_i_6 += scalar_t(-0.40824829046386307) * xj_1 * yk_1 * go_v_18;
            gw_acc_i_6 += scalar_t(0.81649658092772615) * xj_2 * yk_2 * go_v_18;
            gw_acc_i_6 += scalar_t(-0.40824829046386307) * xj_3 * yk_3 * go_v_18;
            gw_acc_i_6 += scalar_t(0.70710678118654757) * xj_2 * yk_3 * go_v_19;
            gw_acc_i_6 += scalar_t(0.70710678118654757) * xj_3 * yk_2 * go_v_19;
            gw_acc_i_6 += scalar_t(-0.70710678118654746) * xj_1 * yk_1 * go_v_20;
            gw_acc_i_6 += scalar_t(0.70710678118654746) * xj_3 * yk_3 * go_v_20;
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_x j-tile 0
            gx_acc_j_1 += scalar_t(0.70710678118654746) * wi_6 * yk_3 * go_v_16;
            gx_acc_j_1 += scalar_t(0.70710678118654757) * wi_6 * yk_2 * go_v_17;
            gx_acc_j_1 += scalar_t(-0.40824829046386307) * wi_6 * yk_1 * go_v_18;
            gx_acc_j_1 += scalar_t(-0.70710678118654746) * wi_6 * yk_1 * go_v_20;
            gx_acc_j_1 += scalar_t(0.53452248382484879) * wi_7 * yk_11 * go_v_23;
            gx_acc_j_1 += scalar_t(0.48795003647426655) * wi_7 * yk_10 * go_v_24;
            gx_acc_j_1 += scalar_t(0.59761430466719678) * wi_7 * yk_9 * go_v_25;
            gx_acc_j_1 += scalar_t(0.15430334996209191) * wi_7 * yk_11 * go_v_25;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_6 * yk_1 * go_v_16;
            gx_acc_j_3 += scalar_t(-0.40824829046386307) * wi_6 * yk_3 * go_v_18;
            gx_acc_j_3 += scalar_t(0.70710678118654757) * wi_6 * yk_2 * go_v_19;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_6 * yk_3 * go_v_20;
            gx_acc_j_3 += scalar_t(0.53452248382484879) * wi_7 * yk_13 * go_v_23;
            gx_acc_j_3 += scalar_t(-0.37796447300922731) * wi_7 * yk_12 * go_v_24;
            gx_acc_j_3 += scalar_t(0.48795003647426655) * wi_7 * yk_14 * go_v_24;
            gx_acc_j_3 += scalar_t(-0.15430334996209191) * wi_7 * yk_13 * go_v_25;
            gx_acc_j_3 += scalar_t(0.59761430466719678) * wi_7 * yk_15 * go_v_25;
        
            // grad_x j-tile 1
            gx_acc_j_2 += scalar_t(0.70710678118654757) * wi_6 * yk_1 * go_v_17;
            gx_acc_j_2 += scalar_t(0.81649658092772615) * wi_6 * yk_2 * go_v_18;
            gx_acc_j_2 += scalar_t(0.70710678118654757) * wi_6 * yk_3 * go_v_19;
            gx_acc_j_2 += scalar_t(0.6546536707079772) * wi_7 * yk_12 * go_v_23;
            gx_acc_j_2 += scalar_t(0.61721339984836765) * wi_7 * yk_13 * go_v_24;
            gx_acc_j_2 += scalar_t(0.48795003647426666) * wi_7 * yk_14 * go_v_25;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
            gy_acc_k_1 += scalar_t(0.70710678118654746) * wi_6 * xj_3 * go_v_16;
            gy_acc_k_1 += scalar_t(0.70710678118654757) * wi_6 * xj_2 * go_v_17;
            gy_acc_k_1 += scalar_t(-0.40824829046386307) * wi_6 * xj_1 * go_v_18;
            gy_acc_k_1 += scalar_t(-0.70710678118654746) * wi_6 * xj_1 * go_v_20;
        
            // grad_y k-tile 3
            gy_acc_k_3 += scalar_t(0.70710678118654746) * wi_6 * xj_1 * go_v_16;
            gy_acc_k_3 += scalar_t(-0.40824829046386307) * wi_6 * xj_3 * go_v_18;
            gy_acc_k_3 += scalar_t(0.70710678118654757) * wi_6 * xj_2 * go_v_19;
            gy_acc_k_3 += scalar_t(0.70710678118654746) * wi_6 * xj_3 * go_v_20;
            gy_acc_k_2 += scalar_t(0.70710678118654757) * wi_6 * xj_1 * go_v_17;
            gy_acc_k_2 += scalar_t(0.81649658092772615) * wi_6 * xj_2 * go_v_18;
            gy_acc_k_2 += scalar_t(0.70710678118654757) * wi_6 * xj_3 * go_v_19;
        
            // grad_y k-tile 4
            gy_acc_k_11 += scalar_t(0.53452248382484879) * wi_7 * xj_1 * go_v_23;
            gy_acc_k_11 += scalar_t(0.15430334996209191) * wi_7 * xj_1 * go_v_25;
            gy_acc_k_13 += scalar_t(0.53452248382484879) * wi_7 * xj_3 * go_v_23;
            gy_acc_k_13 += scalar_t(0.61721339984836765) * wi_7 * xj_2 * go_v_24;
            gy_acc_k_13 += scalar_t(-0.15430334996209191) * wi_7 * xj_3 * go_v_25;
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.48795003647426655) * wi_7 * xj_1 * go_v_24;
        
            // grad_y k-tile 6
            gy_acc_k_12 += scalar_t(0.6546536707079772) * wi_7 * xj_2 * go_v_23;
            gy_acc_k_12 += scalar_t(-0.37796447300922731) * wi_7 * xj_3 * go_v_24;
            gy_acc_k_14 += scalar_t(0.48795003647426655) * wi_7 * xj_3 * go_v_24;
            gy_acc_k_14 += scalar_t(0.48795003647426666) * wi_7 * xj_2 * go_v_25;
        
            // grad_y k-tile 7
            gy_acc_k_9 += scalar_t(0.59761430466719678) * wi_7 * xj_1 * go_v_25;
            gy_acc_k_15 += scalar_t(0.59761430466719678) * wi_7 * xj_3 * go_v_25;
        
            // ---- v tile 3 ----
            scalar_t go_v_22 = grad_out[((int64_t)dst * V + 22) * (int64_t)U + u];
            scalar_t go_v_21 = grad_out[((int64_t)dst * V + 21) * (int64_t)U + u];
            scalar_t go_v_26 = grad_out[((int64_t)dst * V + 26) * (int64_t)U + u];
            scalar_t go_v_27 = grad_out[((int64_t)dst * V + 27) * (int64_t)U + u];
            scalar_t go_v_28 = grad_out[((int64_t)dst * V + 28) * (int64_t)U + u];
            scalar_t go_v_29 = grad_out[((int64_t)dst * V + 29) * (int64_t)U + u];
            scalar_t go_v_30 = grad_out[((int64_t)dst * V + 30) * (int64_t)U + u];
            scalar_t go_v_31 = grad_out[((int64_t)dst * V + 31) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_7 += scalar_t(-0.15430334996209191) * xj_1 * yk_13 * go_v_21;
            gw_acc_i_7 += scalar_t(-0.59761430466719678) * xj_1 * yk_15 * go_v_21;
            gw_acc_i_7 += scalar_t(0.48795003647426666) * xj_2 * yk_10 * go_v_21;
            gw_acc_i_7 += scalar_t(0.59761430466719678) * xj_3 * yk_9 * go_v_21;
            gw_acc_i_7 += scalar_t(-0.15430334996209191) * xj_3 * yk_11 * go_v_21;
            gw_acc_i_7 += scalar_t(-0.37796447300922731) * xj_1 * yk_12 * go_v_22;
            gw_acc_i_7 += scalar_t(-0.48795003647426655) * xj_1 * yk_14 * go_v_22;
            gw_acc_i_7 += scalar_t(0.61721339984836765) * xj_2 * yk_11 * go_v_22;
            gw_acc_i_7 += scalar_t(0.48795003647426655) * xj_3 * yk_10 * go_v_22;
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
            gw_acc_i_8 += xj_0 * yk_9 * go_v_26;
            gw_acc_i_8 += xj_0 * yk_10 * go_v_27;
            gw_acc_i_8 += xj_0 * yk_11 * go_v_28;
            gw_acc_i_8 += xj_0 * yk_12 * go_v_29;
            gw_acc_i_8 += xj_0 * yk_13 * go_v_30;
            gw_acc_i_8 += xj_0 * yk_14 * go_v_31;
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_x j-tile 0
            gx_acc_j_1 += scalar_t(-0.15430334996209191) * wi_7 * yk_13 * go_v_21;
            gx_acc_j_1 += scalar_t(-0.59761430466719678) * wi_7 * yk_15 * go_v_21;
            gx_acc_j_1 += scalar_t(-0.37796447300922731) * wi_7 * yk_12 * go_v_22;
            gx_acc_j_1 += scalar_t(-0.48795003647426655) * wi_7 * yk_14 * go_v_22;
            gx_acc_j_3 += scalar_t(0.59761430466719678) * wi_7 * yk_9 * go_v_21;
            gx_acc_j_3 += scalar_t(-0.15430334996209191) * wi_7 * yk_11 * go_v_21;
            gx_acc_j_3 += scalar_t(0.48795003647426655) * wi_7 * yk_10 * go_v_22;
        
            // grad_x j-tile 1
            gx_acc_j_2 += scalar_t(0.48795003647426666) * wi_7 * yk_10 * go_v_21;
            gx_acc_j_2 += scalar_t(0.61721339984836765) * wi_7 * yk_11 * go_v_22;
            gx_acc_j_0 += wi_8 * yk_9 * go_v_26;
            gx_acc_j_0 += wi_8 * yk_10 * go_v_27;
            gx_acc_j_0 += wi_8 * yk_11 * go_v_28;
            gx_acc_j_0 += wi_8 * yk_12 * go_v_29;
            gx_acc_j_0 += wi_8 * yk_13 * go_v_30;
            gx_acc_j_0 += wi_8 * yk_14 * go_v_31;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
            gy_acc_k_11 += scalar_t(-0.15430334996209191) * wi_7 * xj_3 * go_v_21;
            gy_acc_k_11 += scalar_t(0.61721339984836765) * wi_7 * xj_2 * go_v_22;
            gy_acc_k_11 += wi_8 * xj_0 * go_v_28;
            gy_acc_k_13 += scalar_t(-0.15430334996209191) * wi_7 * xj_1 * go_v_21;
            gy_acc_k_13 += wi_8 * xj_0 * go_v_30;
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.48795003647426666) * wi_7 * xj_2 * go_v_21;
            gy_acc_k_10 += scalar_t(0.48795003647426655) * wi_7 * xj_3 * go_v_22;
            gy_acc_k_10 += wi_8 * xj_0 * go_v_27;
        
            // grad_y k-tile 6
            gy_acc_k_12 += scalar_t(-0.37796447300922731) * wi_7 * xj_1 * go_v_22;
            gy_acc_k_12 += wi_8 * xj_0 * go_v_29;
            gy_acc_k_14 += scalar_t(-0.48795003647426655) * wi_7 * xj_1 * go_v_22;
            gy_acc_k_14 += wi_8 * xj_0 * go_v_31;
        
            // grad_y k-tile 7
            gy_acc_k_9 += scalar_t(0.59761430466719678) * wi_7 * xj_3 * go_v_21;
            gy_acc_k_9 += wi_8 * xj_0 * go_v_26;
            gy_acc_k_15 += scalar_t(-0.59761430466719678) * wi_7 * xj_1 * go_v_21;
        
            // ---- v tile 4 ----
            scalar_t go_v_32 = grad_out[((int64_t)dst * V + 32) * (int64_t)U + u];
            scalar_t go_v_37 = grad_out[((int64_t)dst * V + 37) * (int64_t)U + u];
            scalar_t go_v_39 = grad_out[((int64_t)dst * V + 39) * (int64_t)U + u];
            scalar_t go_v_36 = grad_out[((int64_t)dst * V + 36) * (int64_t)U + u];
            scalar_t go_v_38 = grad_out[((int64_t)dst * V + 38) * (int64_t)U + u];
            scalar_t go_v_35 = grad_out[((int64_t)dst * V + 35) * (int64_t)U + u];
            scalar_t go_v_34 = grad_out[((int64_t)dst * V + 34) * (int64_t)U + u];
            scalar_t go_v_33 = grad_out[((int64_t)dst * V + 33) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_9 += scalar_t(0.70710678118654746) * xj_1 * yk_8 * go_v_33;
            gw_acc_i_9 += scalar_t(0.70710678118654746) * xj_3 * yk_4 * go_v_33;
            gw_acc_i_9 += scalar_t(0.57735026918962573) * xj_1 * yk_7 * go_v_34;
            gw_acc_i_9 += scalar_t(0.57735026918962573) * xj_2 * yk_4 * go_v_34;
            gw_acc_i_9 += scalar_t(0.57735026918962573) * xj_3 * yk_5 * go_v_34;
            gw_acc_i_9 += scalar_t(0.63245553203367588) * xj_1 * yk_6 * go_v_35;
            gw_acc_i_9 += scalar_t(0.18257418583505536) * xj_1 * yk_8 * go_v_35;
            gw_acc_i_9 += scalar_t(0.73029674334022143) * xj_2 * yk_5 * go_v_35;
            gw_acc_i_9 += scalar_t(-0.18257418583505536) * xj_3 * yk_4 * go_v_35;
            gw_acc_i_9 += scalar_t(-0.44721359549995787) * xj_1 * yk_5 * go_v_36;
            gw_acc_i_9 += scalar_t(0.7745966692414834) * xj_2 * yk_6 * go_v_36;
            gw_acc_i_9 += scalar_t(-0.44721359549995787) * xj_3 * yk_7 * go_v_36;
            gw_acc_i_9 += scalar_t(-0.18257418583505536) * xj_1 * yk_4 * go_v_37;
            gw_acc_i_9 += scalar_t(0.73029674334022143) * xj_2 * yk_7 * go_v_37;
            gw_acc_i_9 += scalar_t(0.63245553203367588) * xj_3 * yk_6 * go_v_37;
            gw_acc_i_9 += scalar_t(-0.18257418583505536) * xj_3 * yk_8 * go_v_37;
            gw_acc_i_9 += scalar_t(-0.57735026918962573) * xj_1 * yk_5 * go_v_38;
            gw_acc_i_9 += scalar_t(0.57735026918962573) * xj_2 * yk_8 * go_v_38;
            gw_acc_i_9 += scalar_t(0.57735026918962573) * xj_3 * yk_7 * go_v_38;
            gw_acc_i_9 += scalar_t(-0.70710678118654746) * xj_1 * yk_4 * go_v_39;
            gw_acc_i_9 += scalar_t(0.70710678118654746) * xj_3 * yk_8 * go_v_39;
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
            gw_acc_i_8 += xj_0 * yk_15 * go_v_32;
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_x j-tile 0
            gx_acc_j_1 += scalar_t(0.70710678118654746) * wi_9 * yk_8 * go_v_33;
            gx_acc_j_1 += scalar_t(0.57735026918962573) * wi_9 * yk_7 * go_v_34;
            gx_acc_j_1 += scalar_t(0.63245553203367588) * wi_9 * yk_6 * go_v_35;
            gx_acc_j_1 += scalar_t(0.18257418583505536) * wi_9 * yk_8 * go_v_35;
            gx_acc_j_1 += scalar_t(-0.44721359549995787) * wi_9 * yk_5 * go_v_36;
            gx_acc_j_1 += scalar_t(-0.18257418583505536) * wi_9 * yk_4 * go_v_37;
            gx_acc_j_1 += scalar_t(-0.57735026918962573) * wi_9 * yk_5 * go_v_38;
            gx_acc_j_1 += scalar_t(-0.70710678118654746) * wi_9 * yk_4 * go_v_39;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_9 * yk_4 * go_v_33;
            gx_acc_j_3 += scalar_t(0.57735026918962573) * wi_9 * yk_5 * go_v_34;
            gx_acc_j_3 += scalar_t(-0.18257418583505536) * wi_9 * yk_4 * go_v_35;
            gx_acc_j_3 += scalar_t(-0.44721359549995787) * wi_9 * yk_7 * go_v_36;
            gx_acc_j_3 += scalar_t(0.63245553203367588) * wi_9 * yk_6 * go_v_37;
            gx_acc_j_3 += scalar_t(-0.18257418583505536) * wi_9 * yk_8 * go_v_37;
            gx_acc_j_3 += scalar_t(0.57735026918962573) * wi_9 * yk_7 * go_v_38;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_9 * yk_8 * go_v_39;
        
            // grad_x j-tile 1
            gx_acc_j_2 += scalar_t(0.57735026918962573) * wi_9 * yk_4 * go_v_34;
            gx_acc_j_2 += scalar_t(0.73029674334022143) * wi_9 * yk_5 * go_v_35;
            gx_acc_j_2 += scalar_t(0.7745966692414834) * wi_9 * yk_6 * go_v_36;
            gx_acc_j_2 += scalar_t(0.73029674334022143) * wi_9 * yk_7 * go_v_37;
            gx_acc_j_2 += scalar_t(0.57735026918962573) * wi_9 * yk_8 * go_v_38;
            gx_acc_j_0 += wi_8 * yk_15 * go_v_32;
        
            // grad_y k-tile 0
            gy_acc_k_4 += scalar_t(0.70710678118654746) * wi_9 * xj_3 * go_v_33;
            gy_acc_k_4 += scalar_t(0.57735026918962573) * wi_9 * xj_2 * go_v_34;
            gy_acc_k_4 += scalar_t(-0.18257418583505536) * wi_9 * xj_3 * go_v_35;
            gy_acc_k_4 += scalar_t(-0.18257418583505536) * wi_9 * xj_1 * go_v_37;
            gy_acc_k_4 += scalar_t(-0.70710678118654746) * wi_9 * xj_1 * go_v_39;
            gy_acc_k_8 += scalar_t(0.70710678118654746) * wi_9 * xj_1 * go_v_33;
            gy_acc_k_8 += scalar_t(0.18257418583505536) * wi_9 * xj_1 * go_v_35;
            gy_acc_k_8 += scalar_t(-0.18257418583505536) * wi_9 * xj_3 * go_v_37;
            gy_acc_k_8 += scalar_t(0.57735026918962573) * wi_9 * xj_2 * go_v_38;
            gy_acc_k_8 += scalar_t(0.70710678118654746) * wi_9 * xj_3 * go_v_39;
        
            // grad_y k-tile 1
            gy_acc_k_5 += scalar_t(0.57735026918962573) * wi_9 * xj_3 * go_v_34;
            gy_acc_k_5 += scalar_t(0.73029674334022143) * wi_9 * xj_2 * go_v_35;
            gy_acc_k_5 += scalar_t(-0.44721359549995787) * wi_9 * xj_1 * go_v_36;
            gy_acc_k_5 += scalar_t(-0.57735026918962573) * wi_9 * xj_1 * go_v_38;
            gy_acc_k_6 += scalar_t(0.63245553203367588) * wi_9 * xj_1 * go_v_35;
            gy_acc_k_6 += scalar_t(0.7745966692414834) * wi_9 * xj_2 * go_v_36;
            gy_acc_k_6 += scalar_t(0.63245553203367588) * wi_9 * xj_3 * go_v_37;
        
            // grad_y k-tile 2
            gy_acc_k_7 += scalar_t(0.57735026918962573) * wi_9 * xj_1 * go_v_34;
            gy_acc_k_7 += scalar_t(-0.44721359549995787) * wi_9 * xj_3 * go_v_36;
            gy_acc_k_7 += scalar_t(0.73029674334022143) * wi_9 * xj_2 * go_v_37;
            gy_acc_k_7 += scalar_t(0.57735026918962573) * wi_9 * xj_3 * go_v_38;
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
            gy_acc_k_15 += wi_8 * xj_0 * go_v_32;
        
            // direct-store grad_w for this u
            grad_w[((int64_t)b * Iw + 7) * (int64_t)U + u] = gw_acc_i_7;
            grad_w[((int64_t)b * Iw + 9) * (int64_t)U + u] = gw_acc_i_9;
            grad_w[((int64_t)b * Iw + 4) * (int64_t)U + u] = gw_acc_i_4;
            grad_w[((int64_t)b * Iw + 6) * (int64_t)U + u] = gw_acc_i_6;
            grad_w[((int64_t)b * Iw + 8) * (int64_t)U + u] = gw_acc_i_8;
            grad_w[((int64_t)b * Iw + 5) * (int64_t)U + u] = gw_acc_i_5;
            grad_w[((int64_t)b * Iw + 1) * (int64_t)U + u] = gw_acc_i_1;
            grad_w[((int64_t)b * Iw + 2) * (int64_t)U + u] = gw_acc_i_2;
            grad_w[((int64_t)b * Iw + 3) * (int64_t)U + u] = gw_acc_i_3;
            grad_w[((int64_t)b * Iw + 0) * (int64_t)U + u] = gw_acc_i_0;
        
            // atomic grad_x for this u
            atomicAdd(&grad_x[((int64_t)src * Ix + 1) * (int64_t)U + u], gx_acc_j_1);
            atomicAdd(&grad_x[((int64_t)src * Ix + 3) * (int64_t)U + u], gx_acc_j_3);
            atomicAdd(&grad_x[((int64_t)src * Ix + 2) * (int64_t)U + u], gx_acc_j_2);
            atomicAdd(&grad_x[((int64_t)src * Ix + 0) * (int64_t)U + u], gx_acc_j_0);
        }

    }

    // tail warp-reduce grad_y
    scalar_t gy_sum_k_4 = warp_sum_xor(gy_acc_k_4);
    if (lane == 0) atomicAdd(&grad_y[y_base + 4], gy_sum_k_4);
    scalar_t gy_sum_k_8 = warp_sum_xor(gy_acc_k_8);
    if (lane == 0) atomicAdd(&grad_y[y_base + 8], gy_sum_k_8);
    scalar_t gy_sum_k_5 = warp_sum_xor(gy_acc_k_5);
    if (lane == 0) atomicAdd(&grad_y[y_base + 5], gy_sum_k_5);
    scalar_t gy_sum_k_6 = warp_sum_xor(gy_acc_k_6);
    if (lane == 0) atomicAdd(&grad_y[y_base + 6], gy_sum_k_6);
    scalar_t gy_sum_k_7 = warp_sum_xor(gy_acc_k_7);
    if (lane == 0) atomicAdd(&grad_y[y_base + 7], gy_sum_k_7);
    scalar_t gy_sum_k_1 = warp_sum_xor(gy_acc_k_1);
    if (lane == 0) atomicAdd(&grad_y[y_base + 1], gy_sum_k_1);
    scalar_t gy_sum_k_3 = warp_sum_xor(gy_acc_k_3);
    if (lane == 0) atomicAdd(&grad_y[y_base + 3], gy_sum_k_3);
    scalar_t gy_sum_k_2 = warp_sum_xor(gy_acc_k_2);
    if (lane == 0) atomicAdd(&grad_y[y_base + 2], gy_sum_k_2);
    scalar_t gy_sum_k_11 = warp_sum_xor(gy_acc_k_11);
    if (lane == 0) atomicAdd(&grad_y[y_base + 11], gy_sum_k_11);
    scalar_t gy_sum_k_13 = warp_sum_xor(gy_acc_k_13);
    if (lane == 0) atomicAdd(&grad_y[y_base + 13], gy_sum_k_13);
    scalar_t gy_sum_k_0 = warp_sum_xor(gy_acc_k_0);
    if (lane == 0) atomicAdd(&grad_y[y_base + 0], gy_sum_k_0);
    scalar_t gy_sum_k_10 = warp_sum_xor(gy_acc_k_10);
    if (lane == 0) atomicAdd(&grad_y[y_base + 10], gy_sum_k_10);
    scalar_t gy_sum_k_12 = warp_sum_xor(gy_acc_k_12);
    if (lane == 0) atomicAdd(&grad_y[y_base + 12], gy_sum_k_12);
    scalar_t gy_sum_k_14 = warp_sum_xor(gy_acc_k_14);
    if (lane == 0) atomicAdd(&grad_y[y_base + 14], gy_sum_k_14);
    scalar_t gy_sum_k_9 = warp_sum_xor(gy_acc_k_9);
    if (lane == 0) atomicAdd(&grad_y[y_base + 9], gy_sum_k_9);
    scalar_t gy_sum_k_15 = warp_sum_xor(gy_acc_k_15);
    if (lane == 0) atomicAdd(&grad_y[y_base + 15], gy_sum_k_15);

}

template <typename scalar_t>
void launch_generated_uniform1d_u128_P86_backward_kernel(
    const scalar_t* w,
    const scalar_t* x_all,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int Iw, int Ix, int Ky, int V, int U,
    cudaStream_t stream)
{
    dim3 block(32);
    dim3 grid(B, 1);
    generated_uniform1d_u128_P86_backward_kernel<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);
}
