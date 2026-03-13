#include <stdint.h>
#include <cuda_runtime.h>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void generated_uniform1d_u224_P215_backward_kernel(
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
    scalar_t yk_1 = y[y_base + 1];
    scalar_t yk_3 = y[y_base + 3];
    scalar_t yk_11 = y[y_base + 11];
    scalar_t yk_13 = y[y_base + 13];
    scalar_t yk_5 = y[y_base + 5];
    scalar_t yk_7 = y[y_base + 7];
    scalar_t yk_2 = y[y_base + 2];
    scalar_t yk_4 = y[y_base + 4];
    scalar_t yk_6 = y[y_base + 6];
    scalar_t yk_8 = y[y_base + 8];
    scalar_t yk_10 = y[y_base + 10];
    scalar_t yk_12 = y[y_base + 12];
    scalar_t yk_14 = y[y_base + 14];
    scalar_t yk_9 = y[y_base + 9];
    scalar_t yk_15 = y[y_base + 15];
    scalar_t yk_0 = y[y_base + 0];

    // init grad_y accumulators
    scalar_t gy_acc_k_1 = scalar_t(0);
    scalar_t gy_acc_k_3 = scalar_t(0);
    scalar_t gy_acc_k_11 = scalar_t(0);
    scalar_t gy_acc_k_13 = scalar_t(0);
    scalar_t gy_acc_k_5 = scalar_t(0);
    scalar_t gy_acc_k_7 = scalar_t(0);
    scalar_t gy_acc_k_2 = scalar_t(0);
    scalar_t gy_acc_k_4 = scalar_t(0);
    scalar_t gy_acc_k_6 = scalar_t(0);
    scalar_t gy_acc_k_8 = scalar_t(0);
    scalar_t gy_acc_k_10 = scalar_t(0);
    scalar_t gy_acc_k_12 = scalar_t(0);
    scalar_t gy_acc_k_14 = scalar_t(0);
    scalar_t gy_acc_k_9 = scalar_t(0);
    scalar_t gy_acc_k_15 = scalar_t(0);
    scalar_t gy_acc_k_0 = scalar_t(0);

    // u traversal: persistent inner loop over U
    for (int u_base = 0; u_base < U; u_base += 32) {
        int u = u_base + lane;
        if (u < U) {
            // preload w(i,u)
            scalar_t wi_16 = w[((int64_t)b * Iw + 16) * (int64_t)U + u];
            scalar_t wi_12 = w[((int64_t)b * Iw + 12) * (int64_t)U + u];
            scalar_t wi_7 = w[((int64_t)b * Iw + 7) * (int64_t)U + u];
            scalar_t wi_10 = w[((int64_t)b * Iw + 10) * (int64_t)U + u];
            scalar_t wi_14 = w[((int64_t)b * Iw + 14) * (int64_t)U + u];
            scalar_t wi_15 = w[((int64_t)b * Iw + 15) * (int64_t)U + u];
            scalar_t wi_5 = w[((int64_t)b * Iw + 5) * (int64_t)U + u];
            scalar_t wi_6 = w[((int64_t)b * Iw + 6) * (int64_t)U + u];
            scalar_t wi_9 = w[((int64_t)b * Iw + 9) * (int64_t)U + u];
            scalar_t wi_13 = w[((int64_t)b * Iw + 13) * (int64_t)U + u];
            scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
            scalar_t wi_8 = w[((int64_t)b * Iw + 8) * (int64_t)U + u];
            scalar_t wi_11 = w[((int64_t)b * Iw + 11) * (int64_t)U + u];
            scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
            scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];
            scalar_t wi_4 = w[((int64_t)b * Iw + 4) * (int64_t)U + u];
            scalar_t wi_0 = w[((int64_t)b * Iw + 0) * (int64_t)U + u];
        
            // preload x(j,u)
            scalar_t xj_5 = x_all[((int64_t)src * Ix + 5) * (int64_t)U + u];
            scalar_t xj_7 = x_all[((int64_t)src * Ix + 7) * (int64_t)U + u];
            scalar_t xj_1 = x_all[((int64_t)src * Ix + 1) * (int64_t)U + u];
            scalar_t xj_3 = x_all[((int64_t)src * Ix + 3) * (int64_t)U + u];
            scalar_t xj_4 = x_all[((int64_t)src * Ix + 4) * (int64_t)U + u];
            scalar_t xj_8 = x_all[((int64_t)src * Ix + 8) * (int64_t)U + u];
            scalar_t xj_6 = x_all[((int64_t)src * Ix + 6) * (int64_t)U + u];
            scalar_t xj_2 = x_all[((int64_t)src * Ix + 2) * (int64_t)U + u];
            scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];
        
            // init per-u accumulators
            scalar_t gw_acc_i_16 = scalar_t(0);
            scalar_t gw_acc_i_12 = scalar_t(0);
            scalar_t gw_acc_i_7 = scalar_t(0);
            scalar_t gw_acc_i_10 = scalar_t(0);
            scalar_t gw_acc_i_14 = scalar_t(0);
            scalar_t gw_acc_i_15 = scalar_t(0);
            scalar_t gw_acc_i_5 = scalar_t(0);
            scalar_t gw_acc_i_6 = scalar_t(0);
            scalar_t gw_acc_i_9 = scalar_t(0);
            scalar_t gw_acc_i_13 = scalar_t(0);
            scalar_t gw_acc_i_2 = scalar_t(0);
            scalar_t gw_acc_i_8 = scalar_t(0);
            scalar_t gw_acc_i_11 = scalar_t(0);
            scalar_t gw_acc_i_1 = scalar_t(0);
            scalar_t gw_acc_i_3 = scalar_t(0);
            scalar_t gw_acc_i_4 = scalar_t(0);
            scalar_t gw_acc_i_0 = scalar_t(0);
            scalar_t gx_acc_j_5 = scalar_t(0);
            scalar_t gx_acc_j_7 = scalar_t(0);
            scalar_t gx_acc_j_1 = scalar_t(0);
            scalar_t gx_acc_j_3 = scalar_t(0);
            scalar_t gx_acc_j_4 = scalar_t(0);
            scalar_t gx_acc_j_8 = scalar_t(0);
            scalar_t gx_acc_j_6 = scalar_t(0);
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
        
            // grad_w i-tile 4
        
            // grad_w i-tile 5
            gw_acc_i_2 += scalar_t(0.44721359549995798) * xj_4 * yk_4 * go_v_2;
            gw_acc_i_2 += scalar_t(0.44721359549995798) * xj_5 * yk_5 * go_v_2;
            gw_acc_i_2 += scalar_t(0.44721359549995798) * xj_6 * yk_6 * go_v_2;
            gw_acc_i_2 += scalar_t(0.44721359549995798) * xj_7 * yk_7 * go_v_2;
            gw_acc_i_2 += scalar_t(0.44721359549995798) * xj_8 * yk_8 * go_v_2;
        
            // grad_w i-tile 6
            gw_acc_i_1 += scalar_t(0.57735026918962573) * xj_1 * yk_1 * go_v_1;
            gw_acc_i_1 += scalar_t(0.57735026918962584) * xj_2 * yk_2 * go_v_1;
            gw_acc_i_1 += scalar_t(0.57735026918962573) * xj_3 * yk_3 * go_v_1;
        
            // grad_w i-tile 7
            gw_acc_i_3 += xj_0 * yk_1 * go_v_3;
            gw_acc_i_3 += xj_0 * yk_2 * go_v_4;
            gw_acc_i_3 += xj_0 * yk_3 * go_v_5;
            gw_acc_i_4 += xj_1 * yk_0 * go_v_6;
            gw_acc_i_4 += xj_2 * yk_0 * go_v_7;
        
            // grad_w i-tile 8
            gw_acc_i_0 += xj_0 * yk_0 * go_v_0;
        
            // grad_x j-tile 0
            gx_acc_j_5 += scalar_t(0.44721359549995798) * wi_2 * yk_5 * go_v_2;
            gx_acc_j_7 += scalar_t(0.44721359549995798) * wi_2 * yk_7 * go_v_2;
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(0.57735026918962573) * wi_1 * yk_1 * go_v_1;
            gx_acc_j_1 += wi_4 * yk_0 * go_v_6;
            gx_acc_j_3 += scalar_t(0.57735026918962573) * wi_1 * yk_3 * go_v_1;
        
            // grad_x j-tile 2
            gx_acc_j_4 += scalar_t(0.44721359549995798) * wi_2 * yk_4 * go_v_2;
            gx_acc_j_8 += scalar_t(0.44721359549995798) * wi_2 * yk_8 * go_v_2;
        
            // grad_x j-tile 3
            gx_acc_j_6 += scalar_t(0.44721359549995798) * wi_2 * yk_6 * go_v_2;
            gx_acc_j_2 += scalar_t(0.57735026918962584) * wi_1 * yk_2 * go_v_1;
            gx_acc_j_2 += wi_4 * yk_0 * go_v_7;
        
            // grad_x j-tile 4
            gx_acc_j_0 += wi_0 * yk_0 * go_v_0;
            gx_acc_j_0 += wi_3 * yk_1 * go_v_3;
            gx_acc_j_0 += wi_3 * yk_2 * go_v_4;
            gx_acc_j_0 += wi_3 * yk_3 * go_v_5;
        
            // grad_y k-tile 0
            gy_acc_k_1 += scalar_t(0.57735026918962573) * wi_1 * xj_1 * go_v_1;
            gy_acc_k_1 += wi_3 * xj_0 * go_v_3;
            gy_acc_k_3 += scalar_t(0.57735026918962573) * wi_1 * xj_3 * go_v_1;
            gy_acc_k_3 += wi_3 * xj_0 * go_v_5;
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
            gy_acc_k_5 += scalar_t(0.44721359549995798) * wi_2 * xj_5 * go_v_2;
            gy_acc_k_7 += scalar_t(0.44721359549995798) * wi_2 * xj_7 * go_v_2;
        
            // grad_y k-tile 3
            gy_acc_k_2 += scalar_t(0.57735026918962584) * wi_1 * xj_2 * go_v_1;
            gy_acc_k_2 += wi_3 * xj_0 * go_v_4;
            gy_acc_k_4 += scalar_t(0.44721359549995798) * wi_2 * xj_4 * go_v_2;
        
            // grad_y k-tile 4
            gy_acc_k_6 += scalar_t(0.44721359549995798) * wi_2 * xj_6 * go_v_2;
            gy_acc_k_8 += scalar_t(0.44721359549995798) * wi_2 * xj_8 * go_v_2;
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
            gy_acc_k_0 += wi_0 * xj_0 * go_v_0;
            gy_acc_k_0 += wi_4 * xj_1 * go_v_6;
            gy_acc_k_0 += wi_4 * xj_2 * go_v_7;
        
            // ---- v tile 1 ----
            scalar_t go_v_8 = grad_out[((int64_t)dst * V + 8) * (int64_t)U + u];
            scalar_t go_v_11 = grad_out[((int64_t)dst * V + 11) * (int64_t)U + u];
            scalar_t go_v_10 = grad_out[((int64_t)dst * V + 10) * (int64_t)U + u];
            scalar_t go_v_9 = grad_out[((int64_t)dst * V + 9) * (int64_t)U + u];
            scalar_t go_v_14 = grad_out[((int64_t)dst * V + 14) * (int64_t)U + u];
            scalar_t go_v_12 = grad_out[((int64_t)dst * V + 12) * (int64_t)U + u];
            scalar_t go_v_13 = grad_out[((int64_t)dst * V + 13) * (int64_t)U + u];
            scalar_t go_v_17 = grad_out[((int64_t)dst * V + 17) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
            gw_acc_i_7 += scalar_t(0.46291004988627571) * xj_4 * yk_9 * go_v_17;
            gw_acc_i_7 += scalar_t(-0.11952286093343938) * xj_4 * yk_11 * go_v_17;
            gw_acc_i_7 += scalar_t(0.3779644730092272) * xj_5 * yk_10 * go_v_17;
            gw_acc_i_7 += scalar_t(0.41403933560541256) * xj_6 * yk_13 * go_v_17;
            gw_acc_i_7 += scalar_t(-0.29277002188455992) * xj_7 * yk_12 * go_v_17;
            gw_acc_i_7 += scalar_t(0.3779644730092272) * xj_7 * yk_14 * go_v_17;
            gw_acc_i_7 += scalar_t(-0.11952286093343938) * xj_8 * yk_13 * go_v_17;
            gw_acc_i_7 += scalar_t(0.46291004988627571) * xj_8 * yk_15 * go_v_17;
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
            gw_acc_i_5 += scalar_t(-0.31622776601683805) * xj_1 * yk_6 * go_v_9;
            gw_acc_i_5 += scalar_t(-0.54772255750516607) * xj_1 * yk_8 * go_v_9;
            gw_acc_i_5 += scalar_t(0.54772255750516619) * xj_2 * yk_5 * go_v_9;
            gw_acc_i_5 += scalar_t(0.54772255750516607) * xj_3 * yk_4 * go_v_9;
            gw_acc_i_5 += scalar_t(0.54772255750516619) * xj_1 * yk_5 * go_v_10;
            gw_acc_i_5 += scalar_t(0.63245553203367622) * xj_2 * yk_6 * go_v_10;
            gw_acc_i_5 += scalar_t(0.54772255750516619) * xj_3 * yk_7 * go_v_10;
            gw_acc_i_5 += scalar_t(0.54772255750516607) * xj_1 * yk_4 * go_v_11;
            gw_acc_i_5 += scalar_t(0.54772255750516619) * xj_2 * yk_7 * go_v_11;
            gw_acc_i_5 += scalar_t(-0.31622776601683805) * xj_3 * yk_6 * go_v_11;
            gw_acc_i_5 += scalar_t(0.54772255750516607) * xj_3 * yk_8 * go_v_11;
            gw_acc_i_6 += scalar_t(0.54772255750516607) * xj_4 * yk_3 * go_v_12;
            gw_acc_i_6 += scalar_t(0.54772255750516619) * xj_5 * yk_2 * go_v_12;
            gw_acc_i_6 += scalar_t(-0.31622776601683805) * xj_6 * yk_1 * go_v_12;
            gw_acc_i_6 += scalar_t(-0.54772255750516607) * xj_8 * yk_1 * go_v_12;
            gw_acc_i_6 += scalar_t(0.54772255750516619) * xj_5 * yk_1 * go_v_13;
            gw_acc_i_6 += scalar_t(0.63245553203367622) * xj_6 * yk_2 * go_v_13;
            gw_acc_i_6 += scalar_t(0.54772255750516619) * xj_7 * yk_3 * go_v_13;
            gw_acc_i_6 += scalar_t(0.54772255750516607) * xj_4 * yk_1 * go_v_14;
            gw_acc_i_6 += scalar_t(-0.31622776601683805) * xj_6 * yk_3 * go_v_14;
            gw_acc_i_6 += scalar_t(0.54772255750516619) * xj_7 * yk_2 * go_v_14;
            gw_acc_i_6 += scalar_t(0.54772255750516607) * xj_8 * yk_3 * go_v_14;
        
            // grad_w i-tile 4
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
            gw_acc_i_4 += xj_3 * yk_0 * go_v_8;
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
            gx_acc_j_5 += scalar_t(0.54772255750516619) * wi_6 * yk_2 * go_v_12;
            gx_acc_j_5 += scalar_t(0.54772255750516619) * wi_6 * yk_1 * go_v_13;
            gx_acc_j_5 += scalar_t(0.3779644730092272) * wi_7 * yk_10 * go_v_17;
            gx_acc_j_7 += scalar_t(0.54772255750516619) * wi_6 * yk_3 * go_v_13;
            gx_acc_j_7 += scalar_t(0.54772255750516619) * wi_6 * yk_2 * go_v_14;
            gx_acc_j_7 += scalar_t(-0.29277002188455992) * wi_7 * yk_12 * go_v_17;
            gx_acc_j_7 += scalar_t(0.3779644730092272) * wi_7 * yk_14 * go_v_17;
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(-0.31622776601683805) * wi_5 * yk_6 * go_v_9;
            gx_acc_j_1 += scalar_t(-0.54772255750516607) * wi_5 * yk_8 * go_v_9;
            gx_acc_j_1 += scalar_t(0.54772255750516619) * wi_5 * yk_5 * go_v_10;
            gx_acc_j_1 += scalar_t(0.54772255750516607) * wi_5 * yk_4 * go_v_11;
            gx_acc_j_3 += wi_4 * yk_0 * go_v_8;
            gx_acc_j_3 += scalar_t(0.54772255750516607) * wi_5 * yk_4 * go_v_9;
            gx_acc_j_3 += scalar_t(0.54772255750516619) * wi_5 * yk_7 * go_v_10;
            gx_acc_j_3 += scalar_t(-0.31622776601683805) * wi_5 * yk_6 * go_v_11;
            gx_acc_j_3 += scalar_t(0.54772255750516607) * wi_5 * yk_8 * go_v_11;
        
            // grad_x j-tile 2
            gx_acc_j_4 += scalar_t(0.54772255750516607) * wi_6 * yk_3 * go_v_12;
            gx_acc_j_4 += scalar_t(0.54772255750516607) * wi_6 * yk_1 * go_v_14;
            gx_acc_j_4 += scalar_t(0.46291004988627571) * wi_7 * yk_9 * go_v_17;
            gx_acc_j_4 += scalar_t(-0.11952286093343938) * wi_7 * yk_11 * go_v_17;
            gx_acc_j_8 += scalar_t(-0.54772255750516607) * wi_6 * yk_1 * go_v_12;
            gx_acc_j_8 += scalar_t(0.54772255750516607) * wi_6 * yk_3 * go_v_14;
            gx_acc_j_8 += scalar_t(-0.11952286093343938) * wi_7 * yk_13 * go_v_17;
            gx_acc_j_8 += scalar_t(0.46291004988627571) * wi_7 * yk_15 * go_v_17;
        
            // grad_x j-tile 3
            gx_acc_j_6 += scalar_t(-0.31622776601683805) * wi_6 * yk_1 * go_v_12;
            gx_acc_j_6 += scalar_t(0.63245553203367622) * wi_6 * yk_2 * go_v_13;
            gx_acc_j_6 += scalar_t(-0.31622776601683805) * wi_6 * yk_3 * go_v_14;
            gx_acc_j_6 += scalar_t(0.41403933560541256) * wi_7 * yk_13 * go_v_17;
            gx_acc_j_2 += scalar_t(0.54772255750516619) * wi_5 * yk_5 * go_v_9;
            gx_acc_j_2 += scalar_t(0.63245553203367622) * wi_5 * yk_6 * go_v_10;
            gx_acc_j_2 += scalar_t(0.54772255750516619) * wi_5 * yk_7 * go_v_11;
        
            // grad_x j-tile 4
        
            // grad_y k-tile 0
            gy_acc_k_1 += scalar_t(-0.31622776601683805) * wi_6 * xj_6 * go_v_12;
            gy_acc_k_1 += scalar_t(-0.54772255750516607) * wi_6 * xj_8 * go_v_12;
            gy_acc_k_1 += scalar_t(0.54772255750516619) * wi_6 * xj_5 * go_v_13;
            gy_acc_k_1 += scalar_t(0.54772255750516607) * wi_6 * xj_4 * go_v_14;
            gy_acc_k_3 += scalar_t(0.54772255750516607) * wi_6 * xj_4 * go_v_12;
            gy_acc_k_3 += scalar_t(0.54772255750516619) * wi_6 * xj_7 * go_v_13;
            gy_acc_k_3 += scalar_t(-0.31622776601683805) * wi_6 * xj_6 * go_v_14;
            gy_acc_k_3 += scalar_t(0.54772255750516607) * wi_6 * xj_8 * go_v_14;
        
            // grad_y k-tile 1
            gy_acc_k_11 += scalar_t(-0.11952286093343938) * wi_7 * xj_4 * go_v_17;
            gy_acc_k_13 += scalar_t(0.41403933560541256) * wi_7 * xj_6 * go_v_17;
            gy_acc_k_13 += scalar_t(-0.11952286093343938) * wi_7 * xj_8 * go_v_17;
        
            // grad_y k-tile 2
            gy_acc_k_5 += scalar_t(0.54772255750516619) * wi_5 * xj_2 * go_v_9;
            gy_acc_k_5 += scalar_t(0.54772255750516619) * wi_5 * xj_1 * go_v_10;
            gy_acc_k_7 += scalar_t(0.54772255750516619) * wi_5 * xj_3 * go_v_10;
            gy_acc_k_7 += scalar_t(0.54772255750516619) * wi_5 * xj_2 * go_v_11;
        
            // grad_y k-tile 3
            gy_acc_k_2 += scalar_t(0.54772255750516619) * wi_6 * xj_5 * go_v_12;
            gy_acc_k_2 += scalar_t(0.63245553203367622) * wi_6 * xj_6 * go_v_13;
            gy_acc_k_2 += scalar_t(0.54772255750516619) * wi_6 * xj_7 * go_v_14;
            gy_acc_k_4 += scalar_t(0.54772255750516607) * wi_5 * xj_3 * go_v_9;
            gy_acc_k_4 += scalar_t(0.54772255750516607) * wi_5 * xj_1 * go_v_11;
        
            // grad_y k-tile 4
            gy_acc_k_6 += scalar_t(-0.31622776601683805) * wi_5 * xj_1 * go_v_9;
            gy_acc_k_6 += scalar_t(0.63245553203367622) * wi_5 * xj_2 * go_v_10;
            gy_acc_k_6 += scalar_t(-0.31622776601683805) * wi_5 * xj_3 * go_v_11;
            gy_acc_k_8 += scalar_t(-0.54772255750516607) * wi_5 * xj_1 * go_v_9;
            gy_acc_k_8 += scalar_t(0.54772255750516607) * wi_5 * xj_3 * go_v_11;
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.3779644730092272) * wi_7 * xj_5 * go_v_17;
            gy_acc_k_12 += scalar_t(-0.29277002188455992) * wi_7 * xj_7 * go_v_17;
        
            // grad_y k-tile 6
            gy_acc_k_14 += scalar_t(0.3779644730092272) * wi_7 * xj_7 * go_v_17;
            gy_acc_k_9 += scalar_t(0.46291004988627571) * wi_7 * xj_4 * go_v_17;
        
            // grad_y k-tile 7
            gy_acc_k_15 += scalar_t(0.46291004988627571) * wi_7 * xj_8 * go_v_17;
            gy_acc_k_0 += wi_4 * xj_3 * go_v_8;
        
            // ---- v tile 2 ----
            scalar_t go_v_16 = grad_out[((int64_t)dst * V + 16) * (int64_t)U + u];
            scalar_t go_v_15 = grad_out[((int64_t)dst * V + 15) * (int64_t)U + u];
            scalar_t go_v_18 = grad_out[((int64_t)dst * V + 18) * (int64_t)U + u];
            scalar_t go_v_19 = grad_out[((int64_t)dst * V + 19) * (int64_t)U + u];
            scalar_t go_v_20 = grad_out[((int64_t)dst * V + 20) * (int64_t)U + u];
            scalar_t go_v_21 = grad_out[((int64_t)dst * V + 21) * (int64_t)U + u];
            scalar_t go_v_22 = grad_out[((int64_t)dst * V + 22) * (int64_t)U + u];
            scalar_t go_v_25 = grad_out[((int64_t)dst * V + 25) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
            gw_acc_i_7 += scalar_t(-0.11952286093343938) * xj_4 * yk_13 * go_v_15;
            gw_acc_i_7 += scalar_t(-0.46291004988627571) * xj_4 * yk_15 * go_v_15;
            gw_acc_i_7 += scalar_t(-0.29277002188455992) * xj_5 * yk_12 * go_v_15;
            gw_acc_i_7 += scalar_t(-0.3779644730092272) * xj_5 * yk_14 * go_v_15;
            gw_acc_i_7 += scalar_t(0.41403933560541256) * xj_6 * yk_11 * go_v_15;
            gw_acc_i_7 += scalar_t(0.3779644730092272) * xj_7 * yk_10 * go_v_15;
            gw_acc_i_7 += scalar_t(0.46291004988627571) * xj_8 * yk_9 * go_v_15;
            gw_acc_i_7 += scalar_t(0.11952286093343938) * xj_8 * yk_11 * go_v_15;
            gw_acc_i_7 += scalar_t(0.3779644730092272) * xj_4 * yk_10 * go_v_16;
            gw_acc_i_7 += scalar_t(0.4780914437337575) * xj_5 * yk_11 * go_v_16;
            gw_acc_i_7 += scalar_t(0.50709255283711019) * xj_6 * yk_12 * go_v_16;
            gw_acc_i_7 += scalar_t(0.4780914437337575) * xj_7 * yk_13 * go_v_16;
            gw_acc_i_7 += scalar_t(0.3779644730092272) * xj_8 * yk_14 * go_v_16;
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
            gw_acc_i_9 += scalar_t(-0.40824829046386307) * xj_1 * yk_1 * go_v_25;
            gw_acc_i_9 += scalar_t(0.81649658092772615) * xj_2 * yk_2 * go_v_25;
            gw_acc_i_9 += scalar_t(-0.40824829046386307) * xj_3 * yk_3 * go_v_25;
        
            // grad_w i-tile 5
            gw_acc_i_8 += xj_0 * yk_4 * go_v_18;
            gw_acc_i_8 += xj_0 * yk_5 * go_v_19;
            gw_acc_i_8 += xj_0 * yk_6 * go_v_20;
            gw_acc_i_8 += xj_0 * yk_7 * go_v_21;
            gw_acc_i_8 += xj_0 * yk_8 * go_v_22;
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
            gx_acc_j_5 += scalar_t(-0.29277002188455992) * wi_7 * yk_12 * go_v_15;
            gx_acc_j_5 += scalar_t(-0.3779644730092272) * wi_7 * yk_14 * go_v_15;
            gx_acc_j_5 += scalar_t(0.4780914437337575) * wi_7 * yk_11 * go_v_16;
            gx_acc_j_7 += scalar_t(0.3779644730092272) * wi_7 * yk_10 * go_v_15;
            gx_acc_j_7 += scalar_t(0.4780914437337575) * wi_7 * yk_13 * go_v_16;
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(-0.40824829046386307) * wi_9 * yk_1 * go_v_25;
            gx_acc_j_3 += scalar_t(-0.40824829046386307) * wi_9 * yk_3 * go_v_25;
        
            // grad_x j-tile 2
            gx_acc_j_4 += scalar_t(-0.11952286093343938) * wi_7 * yk_13 * go_v_15;
            gx_acc_j_4 += scalar_t(-0.46291004988627571) * wi_7 * yk_15 * go_v_15;
            gx_acc_j_4 += scalar_t(0.3779644730092272) * wi_7 * yk_10 * go_v_16;
            gx_acc_j_8 += scalar_t(0.46291004988627571) * wi_7 * yk_9 * go_v_15;
            gx_acc_j_8 += scalar_t(0.11952286093343938) * wi_7 * yk_11 * go_v_15;
            gx_acc_j_8 += scalar_t(0.3779644730092272) * wi_7 * yk_14 * go_v_16;
        
            // grad_x j-tile 3
            gx_acc_j_6 += scalar_t(0.41403933560541256) * wi_7 * yk_11 * go_v_15;
            gx_acc_j_6 += scalar_t(0.50709255283711019) * wi_7 * yk_12 * go_v_16;
            gx_acc_j_2 += scalar_t(0.81649658092772615) * wi_9 * yk_2 * go_v_25;
        
            // grad_x j-tile 4
            gx_acc_j_0 += wi_8 * yk_4 * go_v_18;
            gx_acc_j_0 += wi_8 * yk_5 * go_v_19;
            gx_acc_j_0 += wi_8 * yk_6 * go_v_20;
            gx_acc_j_0 += wi_8 * yk_7 * go_v_21;
            gx_acc_j_0 += wi_8 * yk_8 * go_v_22;
        
            // grad_y k-tile 0
            gy_acc_k_1 += scalar_t(-0.40824829046386307) * wi_9 * xj_1 * go_v_25;
            gy_acc_k_3 += scalar_t(-0.40824829046386307) * wi_9 * xj_3 * go_v_25;
        
            // grad_y k-tile 1
            gy_acc_k_11 += scalar_t(0.41403933560541256) * wi_7 * xj_6 * go_v_15;
            gy_acc_k_11 += scalar_t(0.11952286093343938) * wi_7 * xj_8 * go_v_15;
            gy_acc_k_11 += scalar_t(0.4780914437337575) * wi_7 * xj_5 * go_v_16;
            gy_acc_k_13 += scalar_t(-0.11952286093343938) * wi_7 * xj_4 * go_v_15;
            gy_acc_k_13 += scalar_t(0.4780914437337575) * wi_7 * xj_7 * go_v_16;
        
            // grad_y k-tile 2
            gy_acc_k_5 += wi_8 * xj_0 * go_v_19;
            gy_acc_k_7 += wi_8 * xj_0 * go_v_21;
        
            // grad_y k-tile 3
            gy_acc_k_2 += scalar_t(0.81649658092772615) * wi_9 * xj_2 * go_v_25;
            gy_acc_k_4 += wi_8 * xj_0 * go_v_18;
        
            // grad_y k-tile 4
            gy_acc_k_6 += wi_8 * xj_0 * go_v_20;
            gy_acc_k_8 += wi_8 * xj_0 * go_v_22;
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.3779644730092272) * wi_7 * xj_7 * go_v_15;
            gy_acc_k_10 += scalar_t(0.3779644730092272) * wi_7 * xj_4 * go_v_16;
            gy_acc_k_12 += scalar_t(-0.29277002188455992) * wi_7 * xj_5 * go_v_15;
            gy_acc_k_12 += scalar_t(0.50709255283711019) * wi_7 * xj_6 * go_v_16;
        
            // grad_y k-tile 6
            gy_acc_k_14 += scalar_t(-0.3779644730092272) * wi_7 * xj_5 * go_v_15;
            gy_acc_k_14 += scalar_t(0.3779644730092272) * wi_7 * xj_8 * go_v_16;
            gy_acc_k_9 += scalar_t(0.46291004988627571) * wi_7 * xj_8 * go_v_15;
        
            // grad_y k-tile 7
            gy_acc_k_15 += scalar_t(-0.46291004988627571) * wi_7 * xj_4 * go_v_15;
        
            // ---- v tile 3 ----
            scalar_t go_v_27 = grad_out[((int64_t)dst * V + 27) * (int64_t)U + u];
            scalar_t go_v_24 = grad_out[((int64_t)dst * V + 24) * (int64_t)U + u];
            scalar_t go_v_23 = grad_out[((int64_t)dst * V + 23) * (int64_t)U + u];
            scalar_t go_v_26 = grad_out[((int64_t)dst * V + 26) * (int64_t)U + u];
            scalar_t go_v_32 = grad_out[((int64_t)dst * V + 32) * (int64_t)U + u];
            scalar_t go_v_31 = grad_out[((int64_t)dst * V + 31) * (int64_t)U + u];
            scalar_t go_v_30 = grad_out[((int64_t)dst * V + 30) * (int64_t)U + u];
            scalar_t go_v_29 = grad_out[((int64_t)dst * V + 29) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
            gw_acc_i_10 += scalar_t(-0.37796447300922731) * xj_1 * yk_12 * go_v_29;
            gw_acc_i_10 += scalar_t(-0.48795003647426655) * xj_1 * yk_14 * go_v_29;
            gw_acc_i_10 += scalar_t(0.61721339984836765) * xj_2 * yk_11 * go_v_29;
            gw_acc_i_10 += scalar_t(0.48795003647426655) * xj_3 * yk_10 * go_v_29;
            gw_acc_i_10 += scalar_t(0.53452248382484879) * xj_1 * yk_11 * go_v_30;
            gw_acc_i_10 += scalar_t(0.6546536707079772) * xj_2 * yk_12 * go_v_30;
            gw_acc_i_10 += scalar_t(0.53452248382484879) * xj_3 * yk_13 * go_v_30;
            gw_acc_i_10 += scalar_t(0.48795003647426655) * xj_1 * yk_10 * go_v_31;
            gw_acc_i_10 += scalar_t(0.61721339984836765) * xj_2 * yk_13 * go_v_31;
            gw_acc_i_10 += scalar_t(-0.37796447300922731) * xj_3 * yk_12 * go_v_31;
            gw_acc_i_10 += scalar_t(0.48795003647426655) * xj_3 * yk_14 * go_v_31;
            gw_acc_i_10 += scalar_t(0.59761430466719678) * xj_1 * yk_9 * go_v_32;
            gw_acc_i_10 += scalar_t(0.15430334996209191) * xj_1 * yk_11 * go_v_32;
            gw_acc_i_10 += scalar_t(0.48795003647426666) * xj_2 * yk_14 * go_v_32;
            gw_acc_i_10 += scalar_t(-0.15430334996209191) * xj_3 * yk_13 * go_v_32;
            gw_acc_i_10 += scalar_t(0.59761430466719678) * xj_3 * yk_15 * go_v_32;
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
            gw_acc_i_9 += scalar_t(0.70710678118654746) * xj_1 * yk_3 * go_v_23;
            gw_acc_i_9 += scalar_t(0.70710678118654746) * xj_3 * yk_1 * go_v_23;
            gw_acc_i_9 += scalar_t(0.70710678118654757) * xj_1 * yk_2 * go_v_24;
            gw_acc_i_9 += scalar_t(0.70710678118654757) * xj_2 * yk_1 * go_v_24;
            gw_acc_i_9 += scalar_t(0.70710678118654757) * xj_2 * yk_3 * go_v_26;
            gw_acc_i_9 += scalar_t(0.70710678118654757) * xj_3 * yk_2 * go_v_26;
            gw_acc_i_9 += scalar_t(-0.70710678118654746) * xj_1 * yk_1 * go_v_27;
            gw_acc_i_9 += scalar_t(0.70710678118654746) * xj_3 * yk_3 * go_v_27;
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(0.70710678118654746) * wi_9 * yk_3 * go_v_23;
            gx_acc_j_1 += scalar_t(0.70710678118654757) * wi_9 * yk_2 * go_v_24;
            gx_acc_j_1 += scalar_t(-0.70710678118654746) * wi_9 * yk_1 * go_v_27;
            gx_acc_j_1 += scalar_t(-0.37796447300922731) * wi_10 * yk_12 * go_v_29;
            gx_acc_j_1 += scalar_t(-0.48795003647426655) * wi_10 * yk_14 * go_v_29;
            gx_acc_j_1 += scalar_t(0.53452248382484879) * wi_10 * yk_11 * go_v_30;
            gx_acc_j_1 += scalar_t(0.48795003647426655) * wi_10 * yk_10 * go_v_31;
            gx_acc_j_1 += scalar_t(0.59761430466719678) * wi_10 * yk_9 * go_v_32;
            gx_acc_j_1 += scalar_t(0.15430334996209191) * wi_10 * yk_11 * go_v_32;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_9 * yk_1 * go_v_23;
            gx_acc_j_3 += scalar_t(0.70710678118654757) * wi_9 * yk_2 * go_v_26;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_9 * yk_3 * go_v_27;
            gx_acc_j_3 += scalar_t(0.48795003647426655) * wi_10 * yk_10 * go_v_29;
            gx_acc_j_3 += scalar_t(0.53452248382484879) * wi_10 * yk_13 * go_v_30;
            gx_acc_j_3 += scalar_t(-0.37796447300922731) * wi_10 * yk_12 * go_v_31;
            gx_acc_j_3 += scalar_t(0.48795003647426655) * wi_10 * yk_14 * go_v_31;
            gx_acc_j_3 += scalar_t(-0.15430334996209191) * wi_10 * yk_13 * go_v_32;
            gx_acc_j_3 += scalar_t(0.59761430466719678) * wi_10 * yk_15 * go_v_32;
        
            // grad_x j-tile 2
        
            // grad_x j-tile 3
            gx_acc_j_2 += scalar_t(0.70710678118654757) * wi_9 * yk_1 * go_v_24;
            gx_acc_j_2 += scalar_t(0.70710678118654757) * wi_9 * yk_3 * go_v_26;
            gx_acc_j_2 += scalar_t(0.61721339984836765) * wi_10 * yk_11 * go_v_29;
            gx_acc_j_2 += scalar_t(0.6546536707079772) * wi_10 * yk_12 * go_v_30;
            gx_acc_j_2 += scalar_t(0.61721339984836765) * wi_10 * yk_13 * go_v_31;
            gx_acc_j_2 += scalar_t(0.48795003647426666) * wi_10 * yk_14 * go_v_32;
        
            // grad_x j-tile 4
        
            // grad_y k-tile 0
            gy_acc_k_1 += scalar_t(0.70710678118654746) * wi_9 * xj_3 * go_v_23;
            gy_acc_k_1 += scalar_t(0.70710678118654757) * wi_9 * xj_2 * go_v_24;
            gy_acc_k_1 += scalar_t(-0.70710678118654746) * wi_9 * xj_1 * go_v_27;
            gy_acc_k_3 += scalar_t(0.70710678118654746) * wi_9 * xj_1 * go_v_23;
            gy_acc_k_3 += scalar_t(0.70710678118654757) * wi_9 * xj_2 * go_v_26;
            gy_acc_k_3 += scalar_t(0.70710678118654746) * wi_9 * xj_3 * go_v_27;
        
            // grad_y k-tile 1
            gy_acc_k_11 += scalar_t(0.61721339984836765) * wi_10 * xj_2 * go_v_29;
            gy_acc_k_11 += scalar_t(0.53452248382484879) * wi_10 * xj_1 * go_v_30;
            gy_acc_k_11 += scalar_t(0.15430334996209191) * wi_10 * xj_1 * go_v_32;
            gy_acc_k_13 += scalar_t(0.53452248382484879) * wi_10 * xj_3 * go_v_30;
            gy_acc_k_13 += scalar_t(0.61721339984836765) * wi_10 * xj_2 * go_v_31;
            gy_acc_k_13 += scalar_t(-0.15430334996209191) * wi_10 * xj_3 * go_v_32;
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
            gy_acc_k_2 += scalar_t(0.70710678118654757) * wi_9 * xj_1 * go_v_24;
            gy_acc_k_2 += scalar_t(0.70710678118654757) * wi_9 * xj_3 * go_v_26;
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.48795003647426655) * wi_10 * xj_3 * go_v_29;
            gy_acc_k_10 += scalar_t(0.48795003647426655) * wi_10 * xj_1 * go_v_31;
            gy_acc_k_12 += scalar_t(-0.37796447300922731) * wi_10 * xj_1 * go_v_29;
            gy_acc_k_12 += scalar_t(0.6546536707079772) * wi_10 * xj_2 * go_v_30;
            gy_acc_k_12 += scalar_t(-0.37796447300922731) * wi_10 * xj_3 * go_v_31;
        
            // grad_y k-tile 6
            gy_acc_k_14 += scalar_t(-0.48795003647426655) * wi_10 * xj_1 * go_v_29;
            gy_acc_k_14 += scalar_t(0.48795003647426655) * wi_10 * xj_3 * go_v_31;
            gy_acc_k_14 += scalar_t(0.48795003647426666) * wi_10 * xj_2 * go_v_32;
            gy_acc_k_9 += scalar_t(0.59761430466719678) * wi_10 * xj_1 * go_v_32;
        
            // grad_y k-tile 7
            gy_acc_k_15 += scalar_t(0.59761430466719678) * wi_10 * xj_3 * go_v_32;
        
            // ---- v tile 4 ----
            scalar_t go_v_28 = grad_out[((int64_t)dst * V + 28) * (int64_t)U + u];
            scalar_t go_v_33 = grad_out[((int64_t)dst * V + 33) * (int64_t)U + u];
            scalar_t go_v_34 = grad_out[((int64_t)dst * V + 34) * (int64_t)U + u];
            scalar_t go_v_35 = grad_out[((int64_t)dst * V + 35) * (int64_t)U + u];
            scalar_t go_v_36 = grad_out[((int64_t)dst * V + 36) * (int64_t)U + u];
            scalar_t go_v_37 = grad_out[((int64_t)dst * V + 37) * (int64_t)U + u];
            scalar_t go_v_40 = grad_out[((int64_t)dst * V + 40) * (int64_t)U + u];
            scalar_t go_v_41 = grad_out[((int64_t)dst * V + 41) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_12 += scalar_t(-0.5345224838248489) * xj_4 * yk_4 * go_v_40;
            gw_acc_i_12 += scalar_t(0.26726124191242445) * xj_5 * yk_5 * go_v_40;
            gw_acc_i_12 += scalar_t(0.53452248382484902) * xj_6 * yk_6 * go_v_40;
            gw_acc_i_12 += scalar_t(0.26726124191242445) * xj_7 * yk_7 * go_v_40;
            gw_acc_i_12 += scalar_t(-0.5345224838248489) * xj_8 * yk_8 * go_v_40;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_4 * yk_5 * go_v_41;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_5 * yk_4 * go_v_41;
            gw_acc_i_12 += scalar_t(0.26726124191242445) * xj_6 * yk_7 * go_v_41;
            gw_acc_i_12 += scalar_t(0.26726124191242445) * xj_7 * yk_6 * go_v_41;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_7 * yk_8 * go_v_41;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_8 * yk_7 * go_v_41;
        
            // grad_w i-tile 1
            gw_acc_i_10 += scalar_t(-0.15430334996209191) * xj_1 * yk_13 * go_v_28;
            gw_acc_i_10 += scalar_t(-0.59761430466719678) * xj_1 * yk_15 * go_v_28;
            gw_acc_i_10 += scalar_t(0.48795003647426666) * xj_2 * yk_10 * go_v_28;
            gw_acc_i_10 += scalar_t(0.59761430466719678) * xj_3 * yk_9 * go_v_28;
            gw_acc_i_10 += scalar_t(-0.15430334996209191) * xj_3 * yk_11 * go_v_28;
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
            gw_acc_i_11 += xj_4 * yk_0 * go_v_33;
            gw_acc_i_11 += xj_5 * yk_0 * go_v_34;
            gw_acc_i_11 += xj_6 * yk_0 * go_v_35;
            gw_acc_i_11 += xj_7 * yk_0 * go_v_36;
            gw_acc_i_11 += xj_8 * yk_0 * go_v_37;
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
            gx_acc_j_5 += wi_11 * yk_0 * go_v_34;
            gx_acc_j_5 += scalar_t(0.26726124191242445) * wi_12 * yk_5 * go_v_40;
            gx_acc_j_5 += scalar_t(0.46291004988627571) * wi_12 * yk_4 * go_v_41;
            gx_acc_j_7 += wi_11 * yk_0 * go_v_36;
            gx_acc_j_7 += scalar_t(0.26726124191242445) * wi_12 * yk_7 * go_v_40;
            gx_acc_j_7 += scalar_t(0.26726124191242445) * wi_12 * yk_6 * go_v_41;
            gx_acc_j_7 += scalar_t(0.46291004988627571) * wi_12 * yk_8 * go_v_41;
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(-0.15430334996209191) * wi_10 * yk_13 * go_v_28;
            gx_acc_j_1 += scalar_t(-0.59761430466719678) * wi_10 * yk_15 * go_v_28;
            gx_acc_j_3 += scalar_t(0.59761430466719678) * wi_10 * yk_9 * go_v_28;
            gx_acc_j_3 += scalar_t(-0.15430334996209191) * wi_10 * yk_11 * go_v_28;
        
            // grad_x j-tile 2
            gx_acc_j_4 += wi_11 * yk_0 * go_v_33;
            gx_acc_j_4 += scalar_t(-0.5345224838248489) * wi_12 * yk_4 * go_v_40;
            gx_acc_j_4 += scalar_t(0.46291004988627571) * wi_12 * yk_5 * go_v_41;
            gx_acc_j_8 += wi_11 * yk_0 * go_v_37;
            gx_acc_j_8 += scalar_t(-0.5345224838248489) * wi_12 * yk_8 * go_v_40;
            gx_acc_j_8 += scalar_t(0.46291004988627571) * wi_12 * yk_7 * go_v_41;
        
            // grad_x j-tile 3
            gx_acc_j_6 += wi_11 * yk_0 * go_v_35;
            gx_acc_j_6 += scalar_t(0.53452248382484902) * wi_12 * yk_6 * go_v_40;
            gx_acc_j_6 += scalar_t(0.26726124191242445) * wi_12 * yk_7 * go_v_41;
            gx_acc_j_2 += scalar_t(0.48795003647426666) * wi_10 * yk_10 * go_v_28;
        
            // grad_x j-tile 4
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
            gy_acc_k_11 += scalar_t(-0.15430334996209191) * wi_10 * xj_3 * go_v_28;
            gy_acc_k_13 += scalar_t(-0.15430334996209191) * wi_10 * xj_1 * go_v_28;
        
            // grad_y k-tile 2
            gy_acc_k_5 += scalar_t(0.26726124191242445) * wi_12 * xj_5 * go_v_40;
            gy_acc_k_5 += scalar_t(0.46291004988627571) * wi_12 * xj_4 * go_v_41;
            gy_acc_k_7 += scalar_t(0.26726124191242445) * wi_12 * xj_7 * go_v_40;
            gy_acc_k_7 += scalar_t(0.26726124191242445) * wi_12 * xj_6 * go_v_41;
            gy_acc_k_7 += scalar_t(0.46291004988627571) * wi_12 * xj_8 * go_v_41;
        
            // grad_y k-tile 3
            gy_acc_k_4 += scalar_t(-0.5345224838248489) * wi_12 * xj_4 * go_v_40;
            gy_acc_k_4 += scalar_t(0.46291004988627571) * wi_12 * xj_5 * go_v_41;
        
            // grad_y k-tile 4
            gy_acc_k_6 += scalar_t(0.53452248382484902) * wi_12 * xj_6 * go_v_40;
            gy_acc_k_6 += scalar_t(0.26726124191242445) * wi_12 * xj_7 * go_v_41;
            gy_acc_k_8 += scalar_t(-0.5345224838248489) * wi_12 * xj_8 * go_v_40;
            gy_acc_k_8 += scalar_t(0.46291004988627571) * wi_12 * xj_7 * go_v_41;
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.48795003647426666) * wi_10 * xj_2 * go_v_28;
        
            // grad_y k-tile 6
            gy_acc_k_9 += scalar_t(0.59761430466719678) * wi_10 * xj_3 * go_v_28;
        
            // grad_y k-tile 7
            gy_acc_k_15 += scalar_t(-0.59761430466719678) * wi_10 * xj_1 * go_v_28;
            gy_acc_k_0 += wi_11 * xj_4 * go_v_33;
            gy_acc_k_0 += wi_11 * xj_5 * go_v_34;
            gy_acc_k_0 += wi_11 * xj_6 * go_v_35;
            gy_acc_k_0 += wi_11 * xj_7 * go_v_36;
            gy_acc_k_0 += wi_11 * xj_8 * go_v_37;
        
            // ---- v tile 5 ----
            scalar_t go_v_38 = grad_out[((int64_t)dst * V + 38) * (int64_t)U + u];
            scalar_t go_v_39 = grad_out[((int64_t)dst * V + 39) * (int64_t)U + u];
            scalar_t go_v_42 = grad_out[((int64_t)dst * V + 42) * (int64_t)U + u];
            scalar_t go_v_43 = grad_out[((int64_t)dst * V + 43) * (int64_t)U + u];
            scalar_t go_v_44 = grad_out[((int64_t)dst * V + 44) * (int64_t)U + u];
            scalar_t go_v_45 = grad_out[((int64_t)dst * V + 45) * (int64_t)U + u];
            scalar_t go_v_46 = grad_out[((int64_t)dst * V + 46) * (int64_t)U + u];
            scalar_t go_v_47 = grad_out[((int64_t)dst * V + 47) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_12 += scalar_t(-0.5345224838248489) * xj_4 * yk_6 * go_v_38;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_5 * yk_7 * go_v_38;
            gw_acc_i_12 += scalar_t(-0.5345224838248489) * xj_6 * yk_4 * go_v_38;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_7 * yk_5 * go_v_38;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_4 * yk_7 * go_v_39;
            gw_acc_i_12 += scalar_t(0.26726124191242445) * xj_5 * yk_6 * go_v_39;
            gw_acc_i_12 += scalar_t(-0.46291004988627571) * xj_5 * yk_8 * go_v_39;
            gw_acc_i_12 += scalar_t(0.26726124191242445) * xj_6 * yk_5 * go_v_39;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_7 * yk_4 * go_v_39;
            gw_acc_i_12 += scalar_t(-0.46291004988627571) * xj_8 * yk_5 * go_v_39;
            gw_acc_i_12 += scalar_t(-0.46291004988627571) * xj_5 * yk_5 * go_v_42;
            gw_acc_i_12 += scalar_t(-0.5345224838248489) * xj_6 * yk_8 * go_v_42;
            gw_acc_i_12 += scalar_t(0.46291004988627571) * xj_7 * yk_7 * go_v_42;
            gw_acc_i_12 += scalar_t(-0.5345224838248489) * xj_8 * yk_6 * go_v_42;
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
            gw_acc_i_13 += xj_0 * yk_9 * go_v_43;
            gw_acc_i_13 += xj_0 * yk_10 * go_v_44;
            gw_acc_i_13 += xj_0 * yk_11 * go_v_45;
            gw_acc_i_13 += xj_0 * yk_12 * go_v_46;
            gw_acc_i_13 += xj_0 * yk_13 * go_v_47;
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
            gx_acc_j_5 += scalar_t(0.46291004988627571) * wi_12 * yk_7 * go_v_38;
            gx_acc_j_5 += scalar_t(0.26726124191242445) * wi_12 * yk_6 * go_v_39;
            gx_acc_j_5 += scalar_t(-0.46291004988627571) * wi_12 * yk_8 * go_v_39;
            gx_acc_j_5 += scalar_t(-0.46291004988627571) * wi_12 * yk_5 * go_v_42;
            gx_acc_j_7 += scalar_t(0.46291004988627571) * wi_12 * yk_5 * go_v_38;
            gx_acc_j_7 += scalar_t(0.46291004988627571) * wi_12 * yk_4 * go_v_39;
            gx_acc_j_7 += scalar_t(0.46291004988627571) * wi_12 * yk_7 * go_v_42;
        
            // grad_x j-tile 1
        
            // grad_x j-tile 2
            gx_acc_j_4 += scalar_t(-0.5345224838248489) * wi_12 * yk_6 * go_v_38;
            gx_acc_j_4 += scalar_t(0.46291004988627571) * wi_12 * yk_7 * go_v_39;
            gx_acc_j_8 += scalar_t(-0.46291004988627571) * wi_12 * yk_5 * go_v_39;
            gx_acc_j_8 += scalar_t(-0.5345224838248489) * wi_12 * yk_6 * go_v_42;
        
            // grad_x j-tile 3
            gx_acc_j_6 += scalar_t(-0.5345224838248489) * wi_12 * yk_4 * go_v_38;
            gx_acc_j_6 += scalar_t(0.26726124191242445) * wi_12 * yk_5 * go_v_39;
            gx_acc_j_6 += scalar_t(-0.5345224838248489) * wi_12 * yk_8 * go_v_42;
        
            // grad_x j-tile 4
            gx_acc_j_0 += wi_13 * yk_9 * go_v_43;
            gx_acc_j_0 += wi_13 * yk_10 * go_v_44;
            gx_acc_j_0 += wi_13 * yk_11 * go_v_45;
            gx_acc_j_0 += wi_13 * yk_12 * go_v_46;
            gx_acc_j_0 += wi_13 * yk_13 * go_v_47;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
            gy_acc_k_11 += wi_13 * xj_0 * go_v_45;
            gy_acc_k_13 += wi_13 * xj_0 * go_v_47;
        
            // grad_y k-tile 2
            gy_acc_k_5 += scalar_t(0.46291004988627571) * wi_12 * xj_7 * go_v_38;
            gy_acc_k_5 += scalar_t(0.26726124191242445) * wi_12 * xj_6 * go_v_39;
            gy_acc_k_5 += scalar_t(-0.46291004988627571) * wi_12 * xj_8 * go_v_39;
            gy_acc_k_5 += scalar_t(-0.46291004988627571) * wi_12 * xj_5 * go_v_42;
            gy_acc_k_7 += scalar_t(0.46291004988627571) * wi_12 * xj_5 * go_v_38;
            gy_acc_k_7 += scalar_t(0.46291004988627571) * wi_12 * xj_4 * go_v_39;
            gy_acc_k_7 += scalar_t(0.46291004988627571) * wi_12 * xj_7 * go_v_42;
        
            // grad_y k-tile 3
            gy_acc_k_4 += scalar_t(-0.5345224838248489) * wi_12 * xj_6 * go_v_38;
            gy_acc_k_4 += scalar_t(0.46291004988627571) * wi_12 * xj_7 * go_v_39;
        
            // grad_y k-tile 4
            gy_acc_k_6 += scalar_t(-0.5345224838248489) * wi_12 * xj_4 * go_v_38;
            gy_acc_k_6 += scalar_t(0.26726124191242445) * wi_12 * xj_5 * go_v_39;
            gy_acc_k_6 += scalar_t(-0.5345224838248489) * wi_12 * xj_8 * go_v_42;
            gy_acc_k_8 += scalar_t(-0.46291004988627571) * wi_12 * xj_5 * go_v_39;
            gy_acc_k_8 += scalar_t(-0.5345224838248489) * wi_12 * xj_6 * go_v_42;
        
            // grad_y k-tile 5
            gy_acc_k_10 += wi_13 * xj_0 * go_v_44;
            gy_acc_k_12 += wi_13 * xj_0 * go_v_46;
        
            // grad_y k-tile 6
            gy_acc_k_9 += wi_13 * xj_0 * go_v_43;
        
            // grad_y k-tile 7
        
            // ---- v tile 6 ----
            scalar_t go_v_48 = grad_out[((int64_t)dst * V + 48) * (int64_t)U + u];
            scalar_t go_v_49 = grad_out[((int64_t)dst * V + 49) * (int64_t)U + u];
            scalar_t go_v_54 = grad_out[((int64_t)dst * V + 54) * (int64_t)U + u];
            scalar_t go_v_56 = grad_out[((int64_t)dst * V + 56) * (int64_t)U + u];
            scalar_t go_v_53 = grad_out[((int64_t)dst * V + 53) * (int64_t)U + u];
            scalar_t go_v_55 = grad_out[((int64_t)dst * V + 55) * (int64_t)U + u];
            scalar_t go_v_52 = grad_out[((int64_t)dst * V + 52) * (int64_t)U + u];
            scalar_t go_v_51 = grad_out[((int64_t)dst * V + 51) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
            gw_acc_i_14 += scalar_t(0.57735026918962573) * xj_1 * yk_7 * go_v_51;
            gw_acc_i_14 += scalar_t(0.57735026918962573) * xj_2 * yk_4 * go_v_51;
            gw_acc_i_14 += scalar_t(0.57735026918962573) * xj_3 * yk_5 * go_v_51;
            gw_acc_i_14 += scalar_t(0.63245553203367588) * xj_1 * yk_6 * go_v_52;
            gw_acc_i_14 += scalar_t(0.18257418583505536) * xj_1 * yk_8 * go_v_52;
            gw_acc_i_14 += scalar_t(0.73029674334022143) * xj_2 * yk_5 * go_v_52;
            gw_acc_i_14 += scalar_t(-0.18257418583505536) * xj_3 * yk_4 * go_v_52;
            gw_acc_i_14 += scalar_t(-0.44721359549995787) * xj_1 * yk_5 * go_v_53;
            gw_acc_i_14 += scalar_t(0.7745966692414834) * xj_2 * yk_6 * go_v_53;
            gw_acc_i_14 += scalar_t(-0.44721359549995787) * xj_3 * yk_7 * go_v_53;
            gw_acc_i_14 += scalar_t(-0.18257418583505536) * xj_1 * yk_4 * go_v_54;
            gw_acc_i_14 += scalar_t(0.73029674334022143) * xj_2 * yk_7 * go_v_54;
            gw_acc_i_14 += scalar_t(0.63245553203367588) * xj_3 * yk_6 * go_v_54;
            gw_acc_i_14 += scalar_t(-0.18257418583505536) * xj_3 * yk_8 * go_v_54;
            gw_acc_i_14 += scalar_t(-0.57735026918962573) * xj_1 * yk_5 * go_v_55;
            gw_acc_i_14 += scalar_t(0.57735026918962573) * xj_2 * yk_8 * go_v_55;
            gw_acc_i_14 += scalar_t(0.57735026918962573) * xj_3 * yk_7 * go_v_55;
            gw_acc_i_14 += scalar_t(-0.70710678118654746) * xj_1 * yk_4 * go_v_56;
            gw_acc_i_14 += scalar_t(0.70710678118654746) * xj_3 * yk_8 * go_v_56;
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
            gw_acc_i_13 += xj_0 * yk_14 * go_v_48;
            gw_acc_i_13 += xj_0 * yk_15 * go_v_49;
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(0.57735026918962573) * wi_14 * yk_7 * go_v_51;
            gx_acc_j_1 += scalar_t(0.63245553203367588) * wi_14 * yk_6 * go_v_52;
            gx_acc_j_1 += scalar_t(0.18257418583505536) * wi_14 * yk_8 * go_v_52;
            gx_acc_j_1 += scalar_t(-0.44721359549995787) * wi_14 * yk_5 * go_v_53;
            gx_acc_j_1 += scalar_t(-0.18257418583505536) * wi_14 * yk_4 * go_v_54;
            gx_acc_j_1 += scalar_t(-0.57735026918962573) * wi_14 * yk_5 * go_v_55;
            gx_acc_j_1 += scalar_t(-0.70710678118654746) * wi_14 * yk_4 * go_v_56;
            gx_acc_j_3 += scalar_t(0.57735026918962573) * wi_14 * yk_5 * go_v_51;
            gx_acc_j_3 += scalar_t(-0.18257418583505536) * wi_14 * yk_4 * go_v_52;
            gx_acc_j_3 += scalar_t(-0.44721359549995787) * wi_14 * yk_7 * go_v_53;
            gx_acc_j_3 += scalar_t(0.63245553203367588) * wi_14 * yk_6 * go_v_54;
            gx_acc_j_3 += scalar_t(-0.18257418583505536) * wi_14 * yk_8 * go_v_54;
            gx_acc_j_3 += scalar_t(0.57735026918962573) * wi_14 * yk_7 * go_v_55;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_14 * yk_8 * go_v_56;
        
            // grad_x j-tile 2
        
            // grad_x j-tile 3
            gx_acc_j_2 += scalar_t(0.57735026918962573) * wi_14 * yk_4 * go_v_51;
            gx_acc_j_2 += scalar_t(0.73029674334022143) * wi_14 * yk_5 * go_v_52;
            gx_acc_j_2 += scalar_t(0.7745966692414834) * wi_14 * yk_6 * go_v_53;
            gx_acc_j_2 += scalar_t(0.73029674334022143) * wi_14 * yk_7 * go_v_54;
            gx_acc_j_2 += scalar_t(0.57735026918962573) * wi_14 * yk_8 * go_v_55;
        
            // grad_x j-tile 4
            gx_acc_j_0 += wi_13 * yk_14 * go_v_48;
            gx_acc_j_0 += wi_13 * yk_15 * go_v_49;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
            gy_acc_k_5 += scalar_t(0.57735026918962573) * wi_14 * xj_3 * go_v_51;
            gy_acc_k_5 += scalar_t(0.73029674334022143) * wi_14 * xj_2 * go_v_52;
            gy_acc_k_5 += scalar_t(-0.44721359549995787) * wi_14 * xj_1 * go_v_53;
            gy_acc_k_5 += scalar_t(-0.57735026918962573) * wi_14 * xj_1 * go_v_55;
            gy_acc_k_7 += scalar_t(0.57735026918962573) * wi_14 * xj_1 * go_v_51;
            gy_acc_k_7 += scalar_t(-0.44721359549995787) * wi_14 * xj_3 * go_v_53;
            gy_acc_k_7 += scalar_t(0.73029674334022143) * wi_14 * xj_2 * go_v_54;
            gy_acc_k_7 += scalar_t(0.57735026918962573) * wi_14 * xj_3 * go_v_55;
        
            // grad_y k-tile 3
            gy_acc_k_4 += scalar_t(0.57735026918962573) * wi_14 * xj_2 * go_v_51;
            gy_acc_k_4 += scalar_t(-0.18257418583505536) * wi_14 * xj_3 * go_v_52;
            gy_acc_k_4 += scalar_t(-0.18257418583505536) * wi_14 * xj_1 * go_v_54;
            gy_acc_k_4 += scalar_t(-0.70710678118654746) * wi_14 * xj_1 * go_v_56;
        
            // grad_y k-tile 4
            gy_acc_k_6 += scalar_t(0.63245553203367588) * wi_14 * xj_1 * go_v_52;
            gy_acc_k_6 += scalar_t(0.7745966692414834) * wi_14 * xj_2 * go_v_53;
            gy_acc_k_6 += scalar_t(0.63245553203367588) * wi_14 * xj_3 * go_v_54;
            gy_acc_k_8 += scalar_t(0.18257418583505536) * wi_14 * xj_1 * go_v_52;
            gy_acc_k_8 += scalar_t(-0.18257418583505536) * wi_14 * xj_3 * go_v_54;
            gy_acc_k_8 += scalar_t(0.57735026918962573) * wi_14 * xj_2 * go_v_55;
            gy_acc_k_8 += scalar_t(0.70710678118654746) * wi_14 * xj_3 * go_v_56;
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
            gy_acc_k_14 += wi_13 * xj_0 * go_v_48;
        
            // grad_y k-tile 7
            gy_acc_k_15 += wi_13 * xj_0 * go_v_49;
        
            // ---- v tile 7 ----
            scalar_t go_v_50 = grad_out[((int64_t)dst * V + 50) * (int64_t)U + u];
            scalar_t go_v_61 = grad_out[((int64_t)dst * V + 61) * (int64_t)U + u];
            scalar_t go_v_63 = grad_out[((int64_t)dst * V + 63) * (int64_t)U + u];
            scalar_t go_v_58 = grad_out[((int64_t)dst * V + 58) * (int64_t)U + u];
            scalar_t go_v_57 = grad_out[((int64_t)dst * V + 57) * (int64_t)U + u];
            scalar_t go_v_59 = grad_out[((int64_t)dst * V + 59) * (int64_t)U + u];
            scalar_t go_v_60 = grad_out[((int64_t)dst * V + 60) * (int64_t)U + u];
            scalar_t go_v_62 = grad_out[((int64_t)dst * V + 62) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
            gw_acc_i_14 += scalar_t(0.70710678118654746) * xj_1 * yk_8 * go_v_50;
            gw_acc_i_14 += scalar_t(0.70710678118654746) * xj_3 * yk_4 * go_v_50;
            gw_acc_i_15 += scalar_t(0.70710678118654757) * xj_4 * yk_3 * go_v_57;
            gw_acc_i_15 += scalar_t(0.70710678118654757) * xj_8 * yk_1 * go_v_57;
            gw_acc_i_15 += scalar_t(0.57735026918962584) * xj_4 * yk_2 * go_v_58;
            gw_acc_i_15 += scalar_t(0.57735026918962584) * xj_5 * yk_3 * go_v_58;
            gw_acc_i_15 += scalar_t(0.57735026918962584) * xj_7 * yk_1 * go_v_58;
            gw_acc_i_15 += scalar_t(-0.18257418583505539) * xj_4 * yk_3 * go_v_59;
            gw_acc_i_15 += scalar_t(0.73029674334022154) * xj_5 * yk_2 * go_v_59;
            gw_acc_i_15 += scalar_t(0.63245553203367599) * xj_6 * yk_1 * go_v_59;
            gw_acc_i_15 += scalar_t(0.18257418583505539) * xj_8 * yk_1 * go_v_59;
            gw_acc_i_15 += scalar_t(-0.44721359549995793) * xj_5 * yk_1 * go_v_60;
            gw_acc_i_15 += scalar_t(0.77459666924148363) * xj_6 * yk_2 * go_v_60;
            gw_acc_i_15 += scalar_t(-0.44721359549995793) * xj_7 * yk_3 * go_v_60;
            gw_acc_i_15 += scalar_t(-0.18257418583505539) * xj_4 * yk_1 * go_v_61;
            gw_acc_i_15 += scalar_t(0.63245553203367599) * xj_6 * yk_3 * go_v_61;
            gw_acc_i_15 += scalar_t(0.73029674334022154) * xj_7 * yk_2 * go_v_61;
            gw_acc_i_15 += scalar_t(-0.18257418583505539) * xj_8 * yk_3 * go_v_61;
            gw_acc_i_15 += scalar_t(-0.57735026918962584) * xj_5 * yk_1 * go_v_62;
            gw_acc_i_15 += scalar_t(0.57735026918962584) * xj_7 * yk_3 * go_v_62;
            gw_acc_i_15 += scalar_t(0.57735026918962584) * xj_8 * yk_2 * go_v_62;
            gw_acc_i_15 += scalar_t(-0.70710678118654757) * xj_4 * yk_1 * go_v_63;
            gw_acc_i_15 += scalar_t(0.70710678118654757) * xj_8 * yk_3 * go_v_63;
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
            gx_acc_j_5 += scalar_t(0.57735026918962584) * wi_15 * yk_3 * go_v_58;
            gx_acc_j_5 += scalar_t(0.73029674334022154) * wi_15 * yk_2 * go_v_59;
            gx_acc_j_5 += scalar_t(-0.44721359549995793) * wi_15 * yk_1 * go_v_60;
            gx_acc_j_5 += scalar_t(-0.57735026918962584) * wi_15 * yk_1 * go_v_62;
            gx_acc_j_7 += scalar_t(0.57735026918962584) * wi_15 * yk_1 * go_v_58;
            gx_acc_j_7 += scalar_t(-0.44721359549995793) * wi_15 * yk_3 * go_v_60;
            gx_acc_j_7 += scalar_t(0.73029674334022154) * wi_15 * yk_2 * go_v_61;
            gx_acc_j_7 += scalar_t(0.57735026918962584) * wi_15 * yk_3 * go_v_62;
        
            // grad_x j-tile 1
            gx_acc_j_1 += scalar_t(0.70710678118654746) * wi_14 * yk_8 * go_v_50;
            gx_acc_j_3 += scalar_t(0.70710678118654746) * wi_14 * yk_4 * go_v_50;
        
            // grad_x j-tile 2
            gx_acc_j_4 += scalar_t(0.70710678118654757) * wi_15 * yk_3 * go_v_57;
            gx_acc_j_4 += scalar_t(0.57735026918962584) * wi_15 * yk_2 * go_v_58;
            gx_acc_j_4 += scalar_t(-0.18257418583505539) * wi_15 * yk_3 * go_v_59;
            gx_acc_j_4 += scalar_t(-0.18257418583505539) * wi_15 * yk_1 * go_v_61;
            gx_acc_j_4 += scalar_t(-0.70710678118654757) * wi_15 * yk_1 * go_v_63;
            gx_acc_j_8 += scalar_t(0.70710678118654757) * wi_15 * yk_1 * go_v_57;
            gx_acc_j_8 += scalar_t(0.18257418583505539) * wi_15 * yk_1 * go_v_59;
            gx_acc_j_8 += scalar_t(-0.18257418583505539) * wi_15 * yk_3 * go_v_61;
            gx_acc_j_8 += scalar_t(0.57735026918962584) * wi_15 * yk_2 * go_v_62;
            gx_acc_j_8 += scalar_t(0.70710678118654757) * wi_15 * yk_3 * go_v_63;
        
            // grad_x j-tile 3
            gx_acc_j_6 += scalar_t(0.63245553203367599) * wi_15 * yk_1 * go_v_59;
            gx_acc_j_6 += scalar_t(0.77459666924148363) * wi_15 * yk_2 * go_v_60;
            gx_acc_j_6 += scalar_t(0.63245553203367599) * wi_15 * yk_3 * go_v_61;
        
            // grad_x j-tile 4
        
            // grad_y k-tile 0
            gy_acc_k_1 += scalar_t(0.70710678118654757) * wi_15 * xj_8 * go_v_57;
            gy_acc_k_1 += scalar_t(0.57735026918962584) * wi_15 * xj_7 * go_v_58;
            gy_acc_k_1 += scalar_t(0.63245553203367599) * wi_15 * xj_6 * go_v_59;
            gy_acc_k_1 += scalar_t(0.18257418583505539) * wi_15 * xj_8 * go_v_59;
            gy_acc_k_1 += scalar_t(-0.44721359549995793) * wi_15 * xj_5 * go_v_60;
            gy_acc_k_1 += scalar_t(-0.18257418583505539) * wi_15 * xj_4 * go_v_61;
            gy_acc_k_1 += scalar_t(-0.57735026918962584) * wi_15 * xj_5 * go_v_62;
            gy_acc_k_1 += scalar_t(-0.70710678118654757) * wi_15 * xj_4 * go_v_63;
            gy_acc_k_3 += scalar_t(0.70710678118654757) * wi_15 * xj_4 * go_v_57;
            gy_acc_k_3 += scalar_t(0.57735026918962584) * wi_15 * xj_5 * go_v_58;
            gy_acc_k_3 += scalar_t(-0.18257418583505539) * wi_15 * xj_4 * go_v_59;
            gy_acc_k_3 += scalar_t(-0.44721359549995793) * wi_15 * xj_7 * go_v_60;
            gy_acc_k_3 += scalar_t(0.63245553203367599) * wi_15 * xj_6 * go_v_61;
            gy_acc_k_3 += scalar_t(-0.18257418583505539) * wi_15 * xj_8 * go_v_61;
            gy_acc_k_3 += scalar_t(0.57735026918962584) * wi_15 * xj_7 * go_v_62;
            gy_acc_k_3 += scalar_t(0.70710678118654757) * wi_15 * xj_8 * go_v_63;
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
            gy_acc_k_2 += scalar_t(0.57735026918962584) * wi_15 * xj_4 * go_v_58;
            gy_acc_k_2 += scalar_t(0.73029674334022154) * wi_15 * xj_5 * go_v_59;
            gy_acc_k_2 += scalar_t(0.77459666924148363) * wi_15 * xj_6 * go_v_60;
            gy_acc_k_2 += scalar_t(0.73029674334022154) * wi_15 * xj_7 * go_v_61;
            gy_acc_k_2 += scalar_t(0.57735026918962584) * wi_15 * xj_8 * go_v_62;
            gy_acc_k_4 += scalar_t(0.70710678118654746) * wi_14 * xj_3 * go_v_50;
        
            // grad_y k-tile 4
            gy_acc_k_8 += scalar_t(0.70710678118654746) * wi_14 * xj_1 * go_v_50;
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
        
            // ---- v tile 8 ----
            scalar_t go_v_68 = grad_out[((int64_t)dst * V + 68) * (int64_t)U + u];
            scalar_t go_v_67 = grad_out[((int64_t)dst * V + 67) * (int64_t)U + u];
            scalar_t go_v_70 = grad_out[((int64_t)dst * V + 70) * (int64_t)U + u];
            scalar_t go_v_65 = grad_out[((int64_t)dst * V + 65) * (int64_t)U + u];
            scalar_t go_v_64 = grad_out[((int64_t)dst * V + 64) * (int64_t)U + u];
            scalar_t go_v_66 = grad_out[((int64_t)dst * V + 66) * (int64_t)U + u];
            scalar_t go_v_69 = grad_out[((int64_t)dst * V + 69) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_16 += scalar_t(-0.28867513459481292) * xj_4 * yk_13 * go_v_64;
            gw_acc_i_16 += scalar_t(0.45643546458763845) * xj_5 * yk_14 * go_v_64;
            gw_acc_i_16 += scalar_t(-0.64549722436790291) * xj_6 * yk_9 * go_v_64;
            gw_acc_i_16 += scalar_t(0.45643546458763845) * xj_7 * yk_10 * go_v_64;
            gw_acc_i_16 += scalar_t(-0.28867513459481292) * xj_8 * yk_11 * go_v_64;
            gw_acc_i_16 += scalar_t(-0.57735026918962584) * xj_4 * yk_12 * go_v_65;
            gw_acc_i_16 += scalar_t(0.35355339059327379) * xj_5 * yk_13 * go_v_65;
            gw_acc_i_16 += scalar_t(-0.45643546458763845) * xj_5 * yk_15 * go_v_65;
            gw_acc_i_16 += scalar_t(0.45643546458763845) * xj_7 * yk_9 * go_v_65;
            gw_acc_i_16 += scalar_t(0.35355339059327379) * xj_7 * yk_11 * go_v_65;
            gw_acc_i_16 += scalar_t(0.44721359549995798) * xj_4 * yk_13 * go_v_66;
            gw_acc_i_16 += scalar_t(0.28867513459481292) * xj_4 * yk_15 * go_v_66;
            gw_acc_i_16 += scalar_t(0.18257418583505539) * xj_5 * yk_12 * go_v_66;
            gw_acc_i_16 += scalar_t(-0.35355339059327373) * xj_5 * yk_14 * go_v_66;
            gw_acc_i_16 += scalar_t(0.38729833462074181) * xj_6 * yk_11 * go_v_66;
            gw_acc_i_16 += scalar_t(0.35355339059327373) * xj_7 * yk_10 * go_v_66;
            gw_acc_i_16 += scalar_t(-0.28867513459481292) * xj_8 * yk_9 * go_v_66;
            gw_acc_i_16 += scalar_t(-0.44721359549995798) * xj_8 * yk_11 * go_v_66;
            gw_acc_i_16 += scalar_t(-0.57735026918962584) * xj_4 * yk_10 * go_v_67;
            gw_acc_i_16 += scalar_t(0.18257418583505539) * xj_5 * yk_11 * go_v_67;
            gw_acc_i_16 += scalar_t(0.51639777949432231) * xj_6 * yk_12 * go_v_67;
            gw_acc_i_16 += scalar_t(0.18257418583505539) * xj_7 * yk_13 * go_v_67;
            gw_acc_i_16 += scalar_t(-0.57735026918962584) * xj_8 * yk_14 * go_v_67;
            gw_acc_i_16 += scalar_t(-0.28867513459481292) * xj_4 * yk_9 * go_v_68;
            gw_acc_i_16 += scalar_t(0.44721359549995798) * xj_4 * yk_11 * go_v_68;
            gw_acc_i_16 += scalar_t(0.35355339059327373) * xj_5 * yk_10 * go_v_68;
            gw_acc_i_16 += scalar_t(0.38729833462074181) * xj_6 * yk_13 * go_v_68;
            gw_acc_i_16 += scalar_t(0.18257418583505539) * xj_7 * yk_12 * go_v_68;
            gw_acc_i_16 += scalar_t(0.35355339059327373) * xj_7 * yk_14 * go_v_68;
            gw_acc_i_16 += scalar_t(0.44721359549995798) * xj_8 * yk_13 * go_v_68;
            gw_acc_i_16 += scalar_t(-0.28867513459481292) * xj_8 * yk_15 * go_v_68;
            gw_acc_i_16 += scalar_t(0.45643546458763845) * xj_5 * yk_9 * go_v_69;
            gw_acc_i_16 += scalar_t(-0.35355339059327379) * xj_5 * yk_11 * go_v_69;
            gw_acc_i_16 += scalar_t(0.35355339059327379) * xj_7 * yk_13 * go_v_69;
            gw_acc_i_16 += scalar_t(0.45643546458763845) * xj_7 * yk_15 * go_v_69;
            gw_acc_i_16 += scalar_t(-0.57735026918962584) * xj_8 * yk_12 * go_v_69;
            gw_acc_i_16 += scalar_t(0.28867513459481292) * xj_4 * yk_11 * go_v_70;
            gw_acc_i_16 += scalar_t(-0.45643546458763845) * xj_5 * yk_10 * go_v_70;
            gw_acc_i_16 += scalar_t(-0.64549722436790291) * xj_6 * yk_15 * go_v_70;
            gw_acc_i_16 += scalar_t(0.45643546458763845) * xj_7 * yk_14 * go_v_70;
            gw_acc_i_16 += scalar_t(-0.28867513459481292) * xj_8 * yk_13 * go_v_70;
        
            // grad_w i-tile 1
        
            // grad_w i-tile 2
        
            // grad_w i-tile 3
        
            // grad_w i-tile 4
        
            // grad_w i-tile 5
        
            // grad_w i-tile 6
        
            // grad_w i-tile 7
        
            // grad_w i-tile 8
        
            // grad_x j-tile 0
            gx_acc_j_5 += scalar_t(0.45643546458763845) * wi_16 * yk_14 * go_v_64;
            gx_acc_j_5 += scalar_t(0.35355339059327379) * wi_16 * yk_13 * go_v_65;
            gx_acc_j_5 += scalar_t(-0.45643546458763845) * wi_16 * yk_15 * go_v_65;
            gx_acc_j_5 += scalar_t(0.18257418583505539) * wi_16 * yk_12 * go_v_66;
            gx_acc_j_5 += scalar_t(-0.35355339059327373) * wi_16 * yk_14 * go_v_66;
            gx_acc_j_5 += scalar_t(0.18257418583505539) * wi_16 * yk_11 * go_v_67;
            gx_acc_j_5 += scalar_t(0.35355339059327373) * wi_16 * yk_10 * go_v_68;
            gx_acc_j_5 += scalar_t(0.45643546458763845) * wi_16 * yk_9 * go_v_69;
            gx_acc_j_5 += scalar_t(-0.35355339059327379) * wi_16 * yk_11 * go_v_69;
            gx_acc_j_5 += scalar_t(-0.45643546458763845) * wi_16 * yk_10 * go_v_70;
            gx_acc_j_7 += scalar_t(0.45643546458763845) * wi_16 * yk_10 * go_v_64;
            gx_acc_j_7 += scalar_t(0.45643546458763845) * wi_16 * yk_9 * go_v_65;
            gx_acc_j_7 += scalar_t(0.35355339059327379) * wi_16 * yk_11 * go_v_65;
            gx_acc_j_7 += scalar_t(0.35355339059327373) * wi_16 * yk_10 * go_v_66;
            gx_acc_j_7 += scalar_t(0.18257418583505539) * wi_16 * yk_13 * go_v_67;
            gx_acc_j_7 += scalar_t(0.18257418583505539) * wi_16 * yk_12 * go_v_68;
            gx_acc_j_7 += scalar_t(0.35355339059327373) * wi_16 * yk_14 * go_v_68;
            gx_acc_j_7 += scalar_t(0.35355339059327379) * wi_16 * yk_13 * go_v_69;
            gx_acc_j_7 += scalar_t(0.45643546458763845) * wi_16 * yk_15 * go_v_69;
            gx_acc_j_7 += scalar_t(0.45643546458763845) * wi_16 * yk_14 * go_v_70;
        
            // grad_x j-tile 1
        
            // grad_x j-tile 2
            gx_acc_j_4 += scalar_t(-0.28867513459481292) * wi_16 * yk_13 * go_v_64;
            gx_acc_j_4 += scalar_t(-0.57735026918962584) * wi_16 * yk_12 * go_v_65;
            gx_acc_j_4 += scalar_t(0.44721359549995798) * wi_16 * yk_13 * go_v_66;
            gx_acc_j_4 += scalar_t(0.28867513459481292) * wi_16 * yk_15 * go_v_66;
            gx_acc_j_4 += scalar_t(-0.57735026918962584) * wi_16 * yk_10 * go_v_67;
            gx_acc_j_4 += scalar_t(-0.28867513459481292) * wi_16 * yk_9 * go_v_68;
            gx_acc_j_4 += scalar_t(0.44721359549995798) * wi_16 * yk_11 * go_v_68;
            gx_acc_j_4 += scalar_t(0.28867513459481292) * wi_16 * yk_11 * go_v_70;
            gx_acc_j_8 += scalar_t(-0.28867513459481292) * wi_16 * yk_11 * go_v_64;
            gx_acc_j_8 += scalar_t(-0.28867513459481292) * wi_16 * yk_9 * go_v_66;
            gx_acc_j_8 += scalar_t(-0.44721359549995798) * wi_16 * yk_11 * go_v_66;
            gx_acc_j_8 += scalar_t(-0.57735026918962584) * wi_16 * yk_14 * go_v_67;
            gx_acc_j_8 += scalar_t(0.44721359549995798) * wi_16 * yk_13 * go_v_68;
            gx_acc_j_8 += scalar_t(-0.28867513459481292) * wi_16 * yk_15 * go_v_68;
            gx_acc_j_8 += scalar_t(-0.57735026918962584) * wi_16 * yk_12 * go_v_69;
            gx_acc_j_8 += scalar_t(-0.28867513459481292) * wi_16 * yk_13 * go_v_70;
        
            // grad_x j-tile 3
            gx_acc_j_6 += scalar_t(-0.64549722436790291) * wi_16 * yk_9 * go_v_64;
            gx_acc_j_6 += scalar_t(0.38729833462074181) * wi_16 * yk_11 * go_v_66;
            gx_acc_j_6 += scalar_t(0.51639777949432231) * wi_16 * yk_12 * go_v_67;
            gx_acc_j_6 += scalar_t(0.38729833462074181) * wi_16 * yk_13 * go_v_68;
            gx_acc_j_6 += scalar_t(-0.64549722436790291) * wi_16 * yk_15 * go_v_70;
        
            // grad_x j-tile 4
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
            gy_acc_k_11 += scalar_t(-0.28867513459481292) * wi_16 * xj_8 * go_v_64;
            gy_acc_k_11 += scalar_t(0.35355339059327379) * wi_16 * xj_7 * go_v_65;
            gy_acc_k_11 += scalar_t(0.38729833462074181) * wi_16 * xj_6 * go_v_66;
            gy_acc_k_11 += scalar_t(-0.44721359549995798) * wi_16 * xj_8 * go_v_66;
            gy_acc_k_11 += scalar_t(0.18257418583505539) * wi_16 * xj_5 * go_v_67;
            gy_acc_k_11 += scalar_t(0.44721359549995798) * wi_16 * xj_4 * go_v_68;
            gy_acc_k_11 += scalar_t(-0.35355339059327379) * wi_16 * xj_5 * go_v_69;
            gy_acc_k_11 += scalar_t(0.28867513459481292) * wi_16 * xj_4 * go_v_70;
            gy_acc_k_13 += scalar_t(-0.28867513459481292) * wi_16 * xj_4 * go_v_64;
            gy_acc_k_13 += scalar_t(0.35355339059327379) * wi_16 * xj_5 * go_v_65;
            gy_acc_k_13 += scalar_t(0.44721359549995798) * wi_16 * xj_4 * go_v_66;
            gy_acc_k_13 += scalar_t(0.18257418583505539) * wi_16 * xj_7 * go_v_67;
            gy_acc_k_13 += scalar_t(0.38729833462074181) * wi_16 * xj_6 * go_v_68;
            gy_acc_k_13 += scalar_t(0.44721359549995798) * wi_16 * xj_8 * go_v_68;
            gy_acc_k_13 += scalar_t(0.35355339059327379) * wi_16 * xj_7 * go_v_69;
            gy_acc_k_13 += scalar_t(-0.28867513459481292) * wi_16 * xj_8 * go_v_70;
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
            gy_acc_k_10 += scalar_t(0.45643546458763845) * wi_16 * xj_7 * go_v_64;
            gy_acc_k_10 += scalar_t(0.35355339059327373) * wi_16 * xj_7 * go_v_66;
            gy_acc_k_10 += scalar_t(-0.57735026918962584) * wi_16 * xj_4 * go_v_67;
            gy_acc_k_10 += scalar_t(0.35355339059327373) * wi_16 * xj_5 * go_v_68;
            gy_acc_k_10 += scalar_t(-0.45643546458763845) * wi_16 * xj_5 * go_v_70;
            gy_acc_k_12 += scalar_t(-0.57735026918962584) * wi_16 * xj_4 * go_v_65;
            gy_acc_k_12 += scalar_t(0.18257418583505539) * wi_16 * xj_5 * go_v_66;
            gy_acc_k_12 += scalar_t(0.51639777949432231) * wi_16 * xj_6 * go_v_67;
            gy_acc_k_12 += scalar_t(0.18257418583505539) * wi_16 * xj_7 * go_v_68;
            gy_acc_k_12 += scalar_t(-0.57735026918962584) * wi_16 * xj_8 * go_v_69;
        
            // grad_y k-tile 6
            gy_acc_k_14 += scalar_t(0.45643546458763845) * wi_16 * xj_5 * go_v_64;
            gy_acc_k_14 += scalar_t(-0.35355339059327373) * wi_16 * xj_5 * go_v_66;
            gy_acc_k_14 += scalar_t(-0.57735026918962584) * wi_16 * xj_8 * go_v_67;
            gy_acc_k_14 += scalar_t(0.35355339059327373) * wi_16 * xj_7 * go_v_68;
            gy_acc_k_14 += scalar_t(0.45643546458763845) * wi_16 * xj_7 * go_v_70;
            gy_acc_k_9 += scalar_t(-0.64549722436790291) * wi_16 * xj_6 * go_v_64;
            gy_acc_k_9 += scalar_t(0.45643546458763845) * wi_16 * xj_7 * go_v_65;
            gy_acc_k_9 += scalar_t(-0.28867513459481292) * wi_16 * xj_8 * go_v_66;
            gy_acc_k_9 += scalar_t(-0.28867513459481292) * wi_16 * xj_4 * go_v_68;
            gy_acc_k_9 += scalar_t(0.45643546458763845) * wi_16 * xj_5 * go_v_69;
        
            // grad_y k-tile 7
            gy_acc_k_15 += scalar_t(-0.45643546458763845) * wi_16 * xj_5 * go_v_65;
            gy_acc_k_15 += scalar_t(0.28867513459481292) * wi_16 * xj_4 * go_v_66;
            gy_acc_k_15 += scalar_t(-0.28867513459481292) * wi_16 * xj_8 * go_v_68;
            gy_acc_k_15 += scalar_t(0.45643546458763845) * wi_16 * xj_7 * go_v_69;
            gy_acc_k_15 += scalar_t(-0.64549722436790291) * wi_16 * xj_6 * go_v_70;
        
            // direct-store grad_w for this u
            grad_w[((int64_t)b * Iw + 16) * (int64_t)U + u] = gw_acc_i_16;
            grad_w[((int64_t)b * Iw + 12) * (int64_t)U + u] = gw_acc_i_12;
            grad_w[((int64_t)b * Iw + 7) * (int64_t)U + u] = gw_acc_i_7;
            grad_w[((int64_t)b * Iw + 10) * (int64_t)U + u] = gw_acc_i_10;
            grad_w[((int64_t)b * Iw + 14) * (int64_t)U + u] = gw_acc_i_14;
            grad_w[((int64_t)b * Iw + 15) * (int64_t)U + u] = gw_acc_i_15;
            grad_w[((int64_t)b * Iw + 5) * (int64_t)U + u] = gw_acc_i_5;
            grad_w[((int64_t)b * Iw + 6) * (int64_t)U + u] = gw_acc_i_6;
            grad_w[((int64_t)b * Iw + 9) * (int64_t)U + u] = gw_acc_i_9;
            grad_w[((int64_t)b * Iw + 13) * (int64_t)U + u] = gw_acc_i_13;
            grad_w[((int64_t)b * Iw + 2) * (int64_t)U + u] = gw_acc_i_2;
            grad_w[((int64_t)b * Iw + 8) * (int64_t)U + u] = gw_acc_i_8;
            grad_w[((int64_t)b * Iw + 11) * (int64_t)U + u] = gw_acc_i_11;
            grad_w[((int64_t)b * Iw + 1) * (int64_t)U + u] = gw_acc_i_1;
            grad_w[((int64_t)b * Iw + 3) * (int64_t)U + u] = gw_acc_i_3;
            grad_w[((int64_t)b * Iw + 4) * (int64_t)U + u] = gw_acc_i_4;
            grad_w[((int64_t)b * Iw + 0) * (int64_t)U + u] = gw_acc_i_0;
        
            // atomic grad_x for this u
            atomicAdd(&grad_x[((int64_t)src * Ix + 5) * (int64_t)U + u], gx_acc_j_5);
            atomicAdd(&grad_x[((int64_t)src * Ix + 7) * (int64_t)U + u], gx_acc_j_7);
            atomicAdd(&grad_x[((int64_t)src * Ix + 1) * (int64_t)U + u], gx_acc_j_1);
            atomicAdd(&grad_x[((int64_t)src * Ix + 3) * (int64_t)U + u], gx_acc_j_3);
            atomicAdd(&grad_x[((int64_t)src * Ix + 4) * (int64_t)U + u], gx_acc_j_4);
            atomicAdd(&grad_x[((int64_t)src * Ix + 8) * (int64_t)U + u], gx_acc_j_8);
            atomicAdd(&grad_x[((int64_t)src * Ix + 6) * (int64_t)U + u], gx_acc_j_6);
            atomicAdd(&grad_x[((int64_t)src * Ix + 2) * (int64_t)U + u], gx_acc_j_2);
            atomicAdd(&grad_x[((int64_t)src * Ix + 0) * (int64_t)U + u], gx_acc_j_0);
        }

    }

    // tail warp-reduce grad_y
    scalar_t gy_sum_k_1 = warp_sum_xor(gy_acc_k_1);
    if (lane == 0) atomicAdd(&grad_y[y_base + 1], gy_sum_k_1);
    scalar_t gy_sum_k_3 = warp_sum_xor(gy_acc_k_3);
    if (lane == 0) atomicAdd(&grad_y[y_base + 3], gy_sum_k_3);
    scalar_t gy_sum_k_11 = warp_sum_xor(gy_acc_k_11);
    if (lane == 0) atomicAdd(&grad_y[y_base + 11], gy_sum_k_11);
    scalar_t gy_sum_k_13 = warp_sum_xor(gy_acc_k_13);
    if (lane == 0) atomicAdd(&grad_y[y_base + 13], gy_sum_k_13);
    scalar_t gy_sum_k_5 = warp_sum_xor(gy_acc_k_5);
    if (lane == 0) atomicAdd(&grad_y[y_base + 5], gy_sum_k_5);
    scalar_t gy_sum_k_7 = warp_sum_xor(gy_acc_k_7);
    if (lane == 0) atomicAdd(&grad_y[y_base + 7], gy_sum_k_7);
    scalar_t gy_sum_k_2 = warp_sum_xor(gy_acc_k_2);
    if (lane == 0) atomicAdd(&grad_y[y_base + 2], gy_sum_k_2);
    scalar_t gy_sum_k_4 = warp_sum_xor(gy_acc_k_4);
    if (lane == 0) atomicAdd(&grad_y[y_base + 4], gy_sum_k_4);
    scalar_t gy_sum_k_6 = warp_sum_xor(gy_acc_k_6);
    if (lane == 0) atomicAdd(&grad_y[y_base + 6], gy_sum_k_6);
    scalar_t gy_sum_k_8 = warp_sum_xor(gy_acc_k_8);
    if (lane == 0) atomicAdd(&grad_y[y_base + 8], gy_sum_k_8);
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
    scalar_t gy_sum_k_0 = warp_sum_xor(gy_acc_k_0);
    if (lane == 0) atomicAdd(&grad_y[y_base + 0], gy_sum_k_0);

}

template <typename scalar_t>
void launch_generated_uniform1d_u224_P215_backward_kernel(
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
    generated_uniform1d_u224_P215_backward_kernel<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);
}
