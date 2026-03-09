#include <stdint.h>
#include <cuda_runtime.h>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void uniform1d_codegen_two_warp_vgroup_path215_u224_bwd(
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
    int ublk     = (int)blockIdx.y;
    if (b_global >= B) return;

    int b = b_list ? b_list[b_global] : b_global;
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= 2) return;

    int u = (ublk << 5) + lane;
    if (u >= U) return;

    int src = src_idx[b];
    int dst = dst_idx[b];

    // flattened row-major offsets with true U stride
    int64_t y_base = (int64_t)b * Ky;

    if (warp == 0) {
        // preload w(i, u)
        scalar_t wi_7 = w[((int64_t)b * Iw + 7) * (int64_t)U + u];
        scalar_t wi_16 = w[((int64_t)b * Iw + 16) * (int64_t)U + u];
        scalar_t wi_12 = w[((int64_t)b * Iw + 12) * (int64_t)U + u];
        scalar_t wi_15 = w[((int64_t)b * Iw + 15) * (int64_t)U + u];
        scalar_t wi_6 = w[((int64_t)b * Iw + 6) * (int64_t)U + u];
        scalar_t wi_11 = w[((int64_t)b * Iw + 11) * (int64_t)U + u];
        scalar_t wi_10 = w[((int64_t)b * Iw + 10) * (int64_t)U + u];
        scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
        scalar_t wi_5 = w[((int64_t)b * Iw + 5) * (int64_t)U + u];
        scalar_t wi_14 = w[((int64_t)b * Iw + 14) * (int64_t)U + u];
        scalar_t wi_9 = w[((int64_t)b * Iw + 9) * (int64_t)U + u];
        scalar_t wi_4 = w[((int64_t)b * Iw + 4) * (int64_t)U + u];
        scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];
        scalar_t wi_8 = w[((int64_t)b * Iw + 8) * (int64_t)U + u];
        scalar_t wi_13 = w[((int64_t)b * Iw + 13) * (int64_t)U + u];

        // preload x(j, u)
        scalar_t xj_4 = x_all[((int64_t)src * Ix + 4) * (int64_t)U + u];
        scalar_t xj_5 = x_all[((int64_t)src * Ix + 5) * (int64_t)U + u];
        scalar_t xj_6 = x_all[((int64_t)src * Ix + 6) * (int64_t)U + u];
        scalar_t xj_7 = x_all[((int64_t)src * Ix + 7) * (int64_t)U + u];
        scalar_t xj_8 = x_all[((int64_t)src * Ix + 8) * (int64_t)U + u];
        scalar_t xj_1 = x_all[((int64_t)src * Ix + 1) * (int64_t)U + u];
        scalar_t xj_2 = x_all[((int64_t)src * Ix + 2) * (int64_t)U + u];
        scalar_t xj_3 = x_all[((int64_t)src * Ix + 3) * (int64_t)U + u];
        scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];

        // preload y(k)
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_0 = y[y_base + 0];

        // preload grad_out(v, u)
        scalar_t go_v_15 = grad_out[((int64_t)dst * V + 15) * (int64_t)U + u];
        scalar_t go_v_16 = grad_out[((int64_t)dst * V + 16) * (int64_t)U + u];
        scalar_t go_v_66 = grad_out[((int64_t)dst * V + 66) * (int64_t)U + u];
        scalar_t go_v_64 = grad_out[((int64_t)dst * V + 64) * (int64_t)U + u];
        scalar_t go_v_41 = grad_out[((int64_t)dst * V + 41) * (int64_t)U + u];
        scalar_t go_v_40 = grad_out[((int64_t)dst * V + 40) * (int64_t)U + u];
        scalar_t go_v_38 = grad_out[((int64_t)dst * V + 38) * (int64_t)U + u];
        scalar_t go_v_59 = grad_out[((int64_t)dst * V + 59) * (int64_t)U + u];
        scalar_t go_v_14 = grad_out[((int64_t)dst * V + 14) * (int64_t)U + u];
        scalar_t go_v_60 = grad_out[((int64_t)dst * V + 60) * (int64_t)U + u];
        scalar_t go_v_58 = grad_out[((int64_t)dst * V + 58) * (int64_t)U + u];
        scalar_t go_v_65 = grad_out[((int64_t)dst * V + 65) * (int64_t)U + u];
        scalar_t go_v_37 = grad_out[((int64_t)dst * V + 37) * (int64_t)U + u];
        scalar_t go_v_36 = grad_out[((int64_t)dst * V + 36) * (int64_t)U + u];
        scalar_t go_v_34 = grad_out[((int64_t)dst * V + 34) * (int64_t)U + u];
        scalar_t go_v_32 = grad_out[((int64_t)dst * V + 32) * (int64_t)U + u];
        scalar_t go_v_29 = grad_out[((int64_t)dst * V + 29) * (int64_t)U + u];
        scalar_t go_v_30 = grad_out[((int64_t)dst * V + 30) * (int64_t)U + u];
        scalar_t go_v_1 = grad_out[((int64_t)dst * V + 1) * (int64_t)U + u];
        scalar_t go_v_9 = grad_out[((int64_t)dst * V + 9) * (int64_t)U + u];
        scalar_t go_v_53 = grad_out[((int64_t)dst * V + 53) * (int64_t)U + u];
        scalar_t go_v_52 = grad_out[((int64_t)dst * V + 52) * (int64_t)U + u];
        scalar_t go_v_51 = grad_out[((int64_t)dst * V + 51) * (int64_t)U + u];
        scalar_t go_v_50 = grad_out[((int64_t)dst * V + 50) * (int64_t)U + u];
        scalar_t go_v_27 = grad_out[((int64_t)dst * V + 27) * (int64_t)U + u];
        scalar_t go_v_24 = grad_out[((int64_t)dst * V + 24) * (int64_t)U + u];
        scalar_t go_v_8 = grad_out[((int64_t)dst * V + 8) * (int64_t)U + u];
        scalar_t go_v_6 = grad_out[((int64_t)dst * V + 6) * (int64_t)U + u];
        scalar_t go_v_3 = grad_out[((int64_t)dst * V + 3) * (int64_t)U + u];
        scalar_t go_v_5 = grad_out[((int64_t)dst * V + 5) * (int64_t)U + u];
        scalar_t go_v_19 = grad_out[((int64_t)dst * V + 19) * (int64_t)U + u];
        scalar_t go_v_21 = grad_out[((int64_t)dst * V + 21) * (int64_t)U + u];
        scalar_t go_v_43 = grad_out[((int64_t)dst * V + 43) * (int64_t)U + u];
        scalar_t go_v_45 = grad_out[((int64_t)dst * V + 45) * (int64_t)U + u];
        scalar_t go_v_47 = grad_out[((int64_t)dst * V + 47) * (int64_t)U + u];
        scalar_t go_v_49 = grad_out[((int64_t)dst * V + 49) * (int64_t)U + u];

        // grad_w accumulate by unique i
        scalar_t gw_acc_i_7 = scalar_t(0);
        gw_acc_i_7 += scalar_t(-0.119522861f) * xj_4 * yk_13 * go_v_15;
        gw_acc_i_7 += scalar_t(-0.46291005f) * xj_4 * yk_15 * go_v_15;
        gw_acc_i_7 += scalar_t(-0.292770022f) * xj_5 * yk_12 * go_v_15;
        gw_acc_i_7 += scalar_t(-0.377964473f) * xj_5 * yk_14 * go_v_15;
        gw_acc_i_7 += scalar_t(0.414039336f) * xj_6 * yk_11 * go_v_15;
        gw_acc_i_7 += scalar_t(0.377964473f) * xj_7 * yk_10 * go_v_15;
        gw_acc_i_7 += scalar_t(0.46291005f) * xj_8 * yk_9 * go_v_15;
        gw_acc_i_7 += scalar_t(0.119522861f) * xj_8 * yk_11 * go_v_15;
        gw_acc_i_7 += scalar_t(0.377964473f) * xj_4 * yk_10 * go_v_16;
        gw_acc_i_7 += scalar_t(0.478091444f) * xj_5 * yk_11 * go_v_16;
        gw_acc_i_7 += scalar_t(0.507092553f) * xj_6 * yk_12 * go_v_16;
        gw_acc_i_7 += scalar_t(0.478091444f) * xj_7 * yk_13 * go_v_16;
        gw_acc_i_7 += scalar_t(0.377964473f) * xj_8 * yk_14 * go_v_16;
        atomicAdd(&grad_w[((int64_t)b * Iw + 7) * (int64_t)U + u], gw_acc_i_7);

        scalar_t gw_acc_i_16 = scalar_t(0);
        gw_acc_i_16 += scalar_t(0.447213595f) * xj_4 * yk_13 * go_v_66;
        gw_acc_i_16 += scalar_t(0.288675135f) * xj_4 * yk_15 * go_v_66;
        gw_acc_i_16 += scalar_t(0.182574186f) * xj_5 * yk_12 * go_v_66;
        gw_acc_i_16 += scalar_t(-0.353553391f) * xj_5 * yk_14 * go_v_66;
        gw_acc_i_16 += scalar_t(0.387298335f) * xj_6 * yk_11 * go_v_66;
        gw_acc_i_16 += scalar_t(0.353553391f) * xj_7 * yk_10 * go_v_66;
        gw_acc_i_16 += scalar_t(-0.288675135f) * xj_8 * yk_9 * go_v_66;
        gw_acc_i_16 += scalar_t(-0.447213595f) * xj_8 * yk_11 * go_v_66;
        gw_acc_i_16 += scalar_t(-0.288675135f) * xj_4 * yk_13 * go_v_64;
        gw_acc_i_16 += scalar_t(0.456435465f) * xj_5 * yk_14 * go_v_64;
        gw_acc_i_16 += scalar_t(-0.645497224f) * xj_6 * yk_9 * go_v_64;
        gw_acc_i_16 += scalar_t(0.456435465f) * xj_7 * yk_10 * go_v_64;
        gw_acc_i_16 += scalar_t(-0.288675135f) * xj_8 * yk_11 * go_v_64;
        gw_acc_i_16 += scalar_t(-0.577350269f) * xj_4 * yk_12 * go_v_65;
        gw_acc_i_16 += scalar_t(0.353553391f) * xj_5 * yk_13 * go_v_65;
        gw_acc_i_16 += scalar_t(-0.456435465f) * xj_5 * yk_15 * go_v_65;
        gw_acc_i_16 += scalar_t(0.456435465f) * xj_7 * yk_9 * go_v_65;
        gw_acc_i_16 += scalar_t(0.353553391f) * xj_7 * yk_11 * go_v_65;
        atomicAdd(&grad_w[((int64_t)b * Iw + 16) * (int64_t)U + u], gw_acc_i_16);

        scalar_t gw_acc_i_12 = scalar_t(0);
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_4 * yk_5 * go_v_41;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_5 * yk_4 * go_v_41;
        gw_acc_i_12 += scalar_t(0.267261242f) * xj_6 * yk_7 * go_v_41;
        gw_acc_i_12 += scalar_t(0.267261242f) * xj_7 * yk_6 * go_v_41;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_7 * yk_8 * go_v_41;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_8 * yk_7 * go_v_41;
        gw_acc_i_12 += scalar_t(-0.534522484f) * xj_4 * yk_4 * go_v_40;
        gw_acc_i_12 += scalar_t(0.267261242f) * xj_5 * yk_5 * go_v_40;
        gw_acc_i_12 += scalar_t(0.534522484f) * xj_6 * yk_6 * go_v_40;
        gw_acc_i_12 += scalar_t(0.267261242f) * xj_7 * yk_7 * go_v_40;
        gw_acc_i_12 += scalar_t(-0.534522484f) * xj_8 * yk_8 * go_v_40;
        gw_acc_i_12 += scalar_t(-0.534522484f) * xj_4 * yk_6 * go_v_38;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_5 * yk_7 * go_v_38;
        gw_acc_i_12 += scalar_t(-0.534522484f) * xj_6 * yk_4 * go_v_38;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_7 * yk_5 * go_v_38;
        atomicAdd(&grad_w[((int64_t)b * Iw + 12) * (int64_t)U + u], gw_acc_i_12);

        scalar_t gw_acc_i_15 = scalar_t(0);
        gw_acc_i_15 += scalar_t(-0.182574186f) * xj_4 * yk_3 * go_v_59;
        gw_acc_i_15 += scalar_t(0.730296743f) * xj_5 * yk_2 * go_v_59;
        gw_acc_i_15 += scalar_t(0.632455532f) * xj_6 * yk_1 * go_v_59;
        gw_acc_i_15 += scalar_t(0.182574186f) * xj_8 * yk_1 * go_v_59;
        gw_acc_i_15 += scalar_t(-0.447213595f) * xj_5 * yk_1 * go_v_60;
        gw_acc_i_15 += scalar_t(0.774596669f) * xj_6 * yk_2 * go_v_60;
        gw_acc_i_15 += scalar_t(-0.447213595f) * xj_7 * yk_3 * go_v_60;
        gw_acc_i_15 += scalar_t(0.577350269f) * xj_4 * yk_2 * go_v_58;
        gw_acc_i_15 += scalar_t(0.577350269f) * xj_5 * yk_3 * go_v_58;
        gw_acc_i_15 += scalar_t(0.577350269f) * xj_7 * yk_1 * go_v_58;
        atomicAdd(&grad_w[((int64_t)b * Iw + 15) * (int64_t)U + u], gw_acc_i_15);

        scalar_t gw_acc_i_6 = scalar_t(0);
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_4 * yk_1 * go_v_14;
        gw_acc_i_6 += scalar_t(-0.316227766f) * xj_6 * yk_3 * go_v_14;
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_7 * yk_2 * go_v_14;
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_8 * yk_3 * go_v_14;
        atomicAdd(&grad_w[((int64_t)b * Iw + 6) * (int64_t)U + u], gw_acc_i_6);

        scalar_t gw_acc_i_11 = scalar_t(0);
        gw_acc_i_11 += xj_8 * yk_0 * go_v_37;
        gw_acc_i_11 += xj_7 * yk_0 * go_v_36;
        gw_acc_i_11 += xj_5 * yk_0 * go_v_34;
        atomicAdd(&grad_w[((int64_t)b * Iw + 11) * (int64_t)U + u], gw_acc_i_11);

        scalar_t gw_acc_i_10 = scalar_t(0);
        gw_acc_i_10 += scalar_t(0.597614305f) * xj_1 * yk_9 * go_v_32;
        gw_acc_i_10 += scalar_t(0.15430335f) * xj_1 * yk_11 * go_v_32;
        gw_acc_i_10 += scalar_t(0.487950036f) * xj_2 * yk_14 * go_v_32;
        gw_acc_i_10 += scalar_t(-0.15430335f) * xj_3 * yk_13 * go_v_32;
        gw_acc_i_10 += scalar_t(0.597614305f) * xj_3 * yk_15 * go_v_32;
        gw_acc_i_10 += scalar_t(-0.377964473f) * xj_1 * yk_12 * go_v_29;
        gw_acc_i_10 += scalar_t(-0.487950036f) * xj_1 * yk_14 * go_v_29;
        gw_acc_i_10 += scalar_t(0.6172134f) * xj_2 * yk_11 * go_v_29;
        gw_acc_i_10 += scalar_t(0.487950036f) * xj_3 * yk_10 * go_v_29;
        gw_acc_i_10 += scalar_t(0.534522484f) * xj_1 * yk_11 * go_v_30;
        gw_acc_i_10 += scalar_t(0.654653671f) * xj_2 * yk_12 * go_v_30;
        gw_acc_i_10 += scalar_t(0.534522484f) * xj_3 * yk_13 * go_v_30;
        atomicAdd(&grad_w[((int64_t)b * Iw + 10) * (int64_t)U + u], gw_acc_i_10);

        scalar_t gw_acc_i_1 = scalar_t(0);
        gw_acc_i_1 += scalar_t(0.577350269f) * xj_1 * yk_1 * go_v_1;
        gw_acc_i_1 += scalar_t(0.577350269f) * xj_2 * yk_2 * go_v_1;
        gw_acc_i_1 += scalar_t(0.577350269f) * xj_3 * yk_3 * go_v_1;
        atomicAdd(&grad_w[((int64_t)b * Iw + 1) * (int64_t)U + u], gw_acc_i_1);

        scalar_t gw_acc_i_5 = scalar_t(0);
        gw_acc_i_5 += scalar_t(-0.316227766f) * xj_1 * yk_6 * go_v_9;
        gw_acc_i_5 += scalar_t(-0.547722558f) * xj_1 * yk_8 * go_v_9;
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_2 * yk_5 * go_v_9;
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_3 * yk_4 * go_v_9;
        atomicAdd(&grad_w[((int64_t)b * Iw + 5) * (int64_t)U + u], gw_acc_i_5);

        scalar_t gw_acc_i_14 = scalar_t(0);
        gw_acc_i_14 += scalar_t(-0.447213595f) * xj_1 * yk_5 * go_v_53;
        gw_acc_i_14 += scalar_t(0.774596669f) * xj_2 * yk_6 * go_v_53;
        gw_acc_i_14 += scalar_t(-0.447213595f) * xj_3 * yk_7 * go_v_53;
        gw_acc_i_14 += scalar_t(0.632455532f) * xj_1 * yk_6 * go_v_52;
        gw_acc_i_14 += scalar_t(0.182574186f) * xj_1 * yk_8 * go_v_52;
        gw_acc_i_14 += scalar_t(0.730296743f) * xj_2 * yk_5 * go_v_52;
        gw_acc_i_14 += scalar_t(-0.182574186f) * xj_3 * yk_4 * go_v_52;
        gw_acc_i_14 += scalar_t(0.577350269f) * xj_1 * yk_7 * go_v_51;
        gw_acc_i_14 += scalar_t(0.577350269f) * xj_2 * yk_4 * go_v_51;
        gw_acc_i_14 += scalar_t(0.577350269f) * xj_3 * yk_5 * go_v_51;
        gw_acc_i_14 += scalar_t(0.707106781f) * xj_1 * yk_8 * go_v_50;
        gw_acc_i_14 += scalar_t(0.707106781f) * xj_3 * yk_4 * go_v_50;
        atomicAdd(&grad_w[((int64_t)b * Iw + 14) * (int64_t)U + u], gw_acc_i_14);

        scalar_t gw_acc_i_9 = scalar_t(0);
        gw_acc_i_9 += scalar_t(-0.707106781f) * xj_1 * yk_1 * go_v_27;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_3 * yk_3 * go_v_27;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_1 * yk_2 * go_v_24;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_2 * yk_1 * go_v_24;
        atomicAdd(&grad_w[((int64_t)b * Iw + 9) * (int64_t)U + u], gw_acc_i_9);

        scalar_t gw_acc_i_4 = scalar_t(0);
        gw_acc_i_4 += xj_3 * yk_0 * go_v_8;
        gw_acc_i_4 += xj_1 * yk_0 * go_v_6;
        atomicAdd(&grad_w[((int64_t)b * Iw + 4) * (int64_t)U + u], gw_acc_i_4);

        scalar_t gw_acc_i_3 = scalar_t(0);
        gw_acc_i_3 += xj_0 * yk_1 * go_v_3;
        gw_acc_i_3 += xj_0 * yk_3 * go_v_5;
        atomicAdd(&grad_w[((int64_t)b * Iw + 3) * (int64_t)U + u], gw_acc_i_3);

        scalar_t gw_acc_i_8 = scalar_t(0);
        gw_acc_i_8 += xj_0 * yk_5 * go_v_19;
        gw_acc_i_8 += xj_0 * yk_7 * go_v_21;
        atomicAdd(&grad_w[((int64_t)b * Iw + 8) * (int64_t)U + u], gw_acc_i_8);

        scalar_t gw_acc_i_13 = scalar_t(0);
        gw_acc_i_13 += xj_0 * yk_9 * go_v_43;
        gw_acc_i_13 += xj_0 * yk_11 * go_v_45;
        gw_acc_i_13 += xj_0 * yk_13 * go_v_47;
        gw_acc_i_13 += xj_0 * yk_15 * go_v_49;
        atomicAdd(&grad_w[((int64_t)b * Iw + 13) * (int64_t)U + u], gw_acc_i_13);

        // grad_x accumulate by unique j
        scalar_t gx_acc_j_4 = scalar_t(0);
        gx_acc_j_4 += scalar_t(-0.119522861f) * wi_7 * yk_13 * go_v_15;
        gx_acc_j_4 += scalar_t(-0.46291005f) * wi_7 * yk_15 * go_v_15;
        gx_acc_j_4 += scalar_t(0.377964473f) * wi_7 * yk_10 * go_v_16;
        gx_acc_j_4 += scalar_t(0.447213595f) * wi_16 * yk_13 * go_v_66;
        gx_acc_j_4 += scalar_t(0.288675135f) * wi_16 * yk_15 * go_v_66;
        gx_acc_j_4 += scalar_t(-0.288675135f) * wi_16 * yk_13 * go_v_64;
        gx_acc_j_4 += scalar_t(0.46291005f) * wi_12 * yk_5 * go_v_41;
        gx_acc_j_4 += scalar_t(-0.534522484f) * wi_12 * yk_4 * go_v_40;
        gx_acc_j_4 += scalar_t(-0.534522484f) * wi_12 * yk_6 * go_v_38;
        gx_acc_j_4 += scalar_t(-0.182574186f) * wi_15 * yk_3 * go_v_59;
        gx_acc_j_4 += scalar_t(0.547722558f) * wi_6 * yk_1 * go_v_14;
        gx_acc_j_4 += scalar_t(0.577350269f) * wi_15 * yk_2 * go_v_58;
        gx_acc_j_4 += scalar_t(-0.577350269f) * wi_16 * yk_12 * go_v_65;
        atomicAdd(&grad_x[((int64_t)src * Ix + 4) * (int64_t)U + u], gx_acc_j_4);

        scalar_t gx_acc_j_5 = scalar_t(0);
        gx_acc_j_5 += scalar_t(-0.292770022f) * wi_7 * yk_12 * go_v_15;
        gx_acc_j_5 += scalar_t(-0.377964473f) * wi_7 * yk_14 * go_v_15;
        gx_acc_j_5 += scalar_t(0.478091444f) * wi_7 * yk_11 * go_v_16;
        gx_acc_j_5 += scalar_t(0.182574186f) * wi_16 * yk_12 * go_v_66;
        gx_acc_j_5 += scalar_t(-0.353553391f) * wi_16 * yk_14 * go_v_66;
        gx_acc_j_5 += scalar_t(0.456435465f) * wi_16 * yk_14 * go_v_64;
        gx_acc_j_5 += scalar_t(0.46291005f) * wi_12 * yk_4 * go_v_41;
        gx_acc_j_5 += scalar_t(0.267261242f) * wi_12 * yk_5 * go_v_40;
        gx_acc_j_5 += scalar_t(0.46291005f) * wi_12 * yk_7 * go_v_38;
        gx_acc_j_5 += scalar_t(0.730296743f) * wi_15 * yk_2 * go_v_59;
        gx_acc_j_5 += scalar_t(-0.447213595f) * wi_15 * yk_1 * go_v_60;
        gx_acc_j_5 += scalar_t(0.577350269f) * wi_15 * yk_3 * go_v_58;
        gx_acc_j_5 += scalar_t(0.353553391f) * wi_16 * yk_13 * go_v_65;
        gx_acc_j_5 += scalar_t(-0.456435465f) * wi_16 * yk_15 * go_v_65;
        gx_acc_j_5 += wi_11 * yk_0 * go_v_34;
        atomicAdd(&grad_x[((int64_t)src * Ix + 5) * (int64_t)U + u], gx_acc_j_5);

        scalar_t gx_acc_j_6 = scalar_t(0);
        gx_acc_j_6 += scalar_t(0.414039336f) * wi_7 * yk_11 * go_v_15;
        gx_acc_j_6 += scalar_t(0.507092553f) * wi_7 * yk_12 * go_v_16;
        gx_acc_j_6 += scalar_t(0.387298335f) * wi_16 * yk_11 * go_v_66;
        gx_acc_j_6 += scalar_t(-0.645497224f) * wi_16 * yk_9 * go_v_64;
        gx_acc_j_6 += scalar_t(0.267261242f) * wi_12 * yk_7 * go_v_41;
        gx_acc_j_6 += scalar_t(0.534522484f) * wi_12 * yk_6 * go_v_40;
        gx_acc_j_6 += scalar_t(-0.534522484f) * wi_12 * yk_4 * go_v_38;
        gx_acc_j_6 += scalar_t(0.632455532f) * wi_15 * yk_1 * go_v_59;
        gx_acc_j_6 += scalar_t(-0.316227766f) * wi_6 * yk_3 * go_v_14;
        gx_acc_j_6 += scalar_t(0.774596669f) * wi_15 * yk_2 * go_v_60;
        atomicAdd(&grad_x[((int64_t)src * Ix + 6) * (int64_t)U + u], gx_acc_j_6);

        scalar_t gx_acc_j_7 = scalar_t(0);
        gx_acc_j_7 += scalar_t(0.377964473f) * wi_7 * yk_10 * go_v_15;
        gx_acc_j_7 += scalar_t(0.478091444f) * wi_7 * yk_13 * go_v_16;
        gx_acc_j_7 += scalar_t(0.353553391f) * wi_16 * yk_10 * go_v_66;
        gx_acc_j_7 += scalar_t(0.456435465f) * wi_16 * yk_10 * go_v_64;
        gx_acc_j_7 += scalar_t(0.267261242f) * wi_12 * yk_6 * go_v_41;
        gx_acc_j_7 += scalar_t(0.46291005f) * wi_12 * yk_8 * go_v_41;
        gx_acc_j_7 += scalar_t(0.267261242f) * wi_12 * yk_7 * go_v_40;
        gx_acc_j_7 += scalar_t(0.46291005f) * wi_12 * yk_5 * go_v_38;
        gx_acc_j_7 += scalar_t(0.547722558f) * wi_6 * yk_2 * go_v_14;
        gx_acc_j_7 += scalar_t(-0.447213595f) * wi_15 * yk_3 * go_v_60;
        gx_acc_j_7 += scalar_t(0.577350269f) * wi_15 * yk_1 * go_v_58;
        gx_acc_j_7 += scalar_t(0.456435465f) * wi_16 * yk_9 * go_v_65;
        gx_acc_j_7 += scalar_t(0.353553391f) * wi_16 * yk_11 * go_v_65;
        gx_acc_j_7 += wi_11 * yk_0 * go_v_36;
        atomicAdd(&grad_x[((int64_t)src * Ix + 7) * (int64_t)U + u], gx_acc_j_7);

        scalar_t gx_acc_j_8 = scalar_t(0);
        gx_acc_j_8 += scalar_t(0.46291005f) * wi_7 * yk_9 * go_v_15;
        gx_acc_j_8 += scalar_t(0.119522861f) * wi_7 * yk_11 * go_v_15;
        gx_acc_j_8 += scalar_t(0.377964473f) * wi_7 * yk_14 * go_v_16;
        gx_acc_j_8 += scalar_t(-0.288675135f) * wi_16 * yk_9 * go_v_66;
        gx_acc_j_8 += scalar_t(-0.447213595f) * wi_16 * yk_11 * go_v_66;
        gx_acc_j_8 += scalar_t(-0.288675135f) * wi_16 * yk_11 * go_v_64;
        gx_acc_j_8 += scalar_t(0.46291005f) * wi_12 * yk_7 * go_v_41;
        gx_acc_j_8 += scalar_t(-0.534522484f) * wi_12 * yk_8 * go_v_40;
        gx_acc_j_8 += scalar_t(0.182574186f) * wi_15 * yk_1 * go_v_59;
        gx_acc_j_8 += scalar_t(0.547722558f) * wi_6 * yk_3 * go_v_14;
        gx_acc_j_8 += wi_11 * yk_0 * go_v_37;
        atomicAdd(&grad_x[((int64_t)src * Ix + 8) * (int64_t)U + u], gx_acc_j_8);

        scalar_t gx_acc_j_1 = scalar_t(0);
        gx_acc_j_1 += scalar_t(0.597614305f) * wi_10 * yk_9 * go_v_32;
        gx_acc_j_1 += scalar_t(0.15430335f) * wi_10 * yk_11 * go_v_32;
        gx_acc_j_1 += scalar_t(-0.377964473f) * wi_10 * yk_12 * go_v_29;
        gx_acc_j_1 += scalar_t(-0.487950036f) * wi_10 * yk_14 * go_v_29;
        gx_acc_j_1 += scalar_t(0.534522484f) * wi_10 * yk_11 * go_v_30;
        gx_acc_j_1 += scalar_t(0.577350269f) * wi_1 * yk_1 * go_v_1;
        gx_acc_j_1 += scalar_t(-0.316227766f) * wi_5 * yk_6 * go_v_9;
        gx_acc_j_1 += scalar_t(-0.547722558f) * wi_5 * yk_8 * go_v_9;
        gx_acc_j_1 += scalar_t(-0.447213595f) * wi_14 * yk_5 * go_v_53;
        gx_acc_j_1 += scalar_t(0.632455532f) * wi_14 * yk_6 * go_v_52;
        gx_acc_j_1 += scalar_t(0.182574186f) * wi_14 * yk_8 * go_v_52;
        gx_acc_j_1 += scalar_t(0.577350269f) * wi_14 * yk_7 * go_v_51;
        gx_acc_j_1 += scalar_t(0.707106781f) * wi_14 * yk_8 * go_v_50;
        gx_acc_j_1 += scalar_t(-0.707106781f) * wi_9 * yk_1 * go_v_27;
        gx_acc_j_1 += scalar_t(0.707106781f) * wi_9 * yk_2 * go_v_24;
        gx_acc_j_1 += wi_4 * yk_0 * go_v_6;
        atomicAdd(&grad_x[((int64_t)src * Ix + 1) * (int64_t)U + u], gx_acc_j_1);

        scalar_t gx_acc_j_2 = scalar_t(0);
        gx_acc_j_2 += scalar_t(0.487950036f) * wi_10 * yk_14 * go_v_32;
        gx_acc_j_2 += scalar_t(0.6172134f) * wi_10 * yk_11 * go_v_29;
        gx_acc_j_2 += scalar_t(0.654653671f) * wi_10 * yk_12 * go_v_30;
        gx_acc_j_2 += scalar_t(0.577350269f) * wi_1 * yk_2 * go_v_1;
        gx_acc_j_2 += scalar_t(0.547722558f) * wi_5 * yk_5 * go_v_9;
        gx_acc_j_2 += scalar_t(0.774596669f) * wi_14 * yk_6 * go_v_53;
        gx_acc_j_2 += scalar_t(0.730296743f) * wi_14 * yk_5 * go_v_52;
        gx_acc_j_2 += scalar_t(0.577350269f) * wi_14 * yk_4 * go_v_51;
        gx_acc_j_2 += scalar_t(0.707106781f) * wi_9 * yk_1 * go_v_24;
        atomicAdd(&grad_x[((int64_t)src * Ix + 2) * (int64_t)U + u], gx_acc_j_2);

        scalar_t gx_acc_j_3 = scalar_t(0);
        gx_acc_j_3 += scalar_t(-0.15430335f) * wi_10 * yk_13 * go_v_32;
        gx_acc_j_3 += scalar_t(0.597614305f) * wi_10 * yk_15 * go_v_32;
        gx_acc_j_3 += scalar_t(0.487950036f) * wi_10 * yk_10 * go_v_29;
        gx_acc_j_3 += scalar_t(0.534522484f) * wi_10 * yk_13 * go_v_30;
        gx_acc_j_3 += scalar_t(0.577350269f) * wi_1 * yk_3 * go_v_1;
        gx_acc_j_3 += scalar_t(0.547722558f) * wi_5 * yk_4 * go_v_9;
        gx_acc_j_3 += scalar_t(-0.447213595f) * wi_14 * yk_7 * go_v_53;
        gx_acc_j_3 += scalar_t(-0.182574186f) * wi_14 * yk_4 * go_v_52;
        gx_acc_j_3 += scalar_t(0.577350269f) * wi_14 * yk_5 * go_v_51;
        gx_acc_j_3 += scalar_t(0.707106781f) * wi_14 * yk_4 * go_v_50;
        gx_acc_j_3 += scalar_t(0.707106781f) * wi_9 * yk_3 * go_v_27;
        gx_acc_j_3 += wi_4 * yk_0 * go_v_8;
        atomicAdd(&grad_x[((int64_t)src * Ix + 3) * (int64_t)U + u], gx_acc_j_3);

        scalar_t gx_acc_j_0 = scalar_t(0);
        gx_acc_j_0 += wi_3 * yk_1 * go_v_3;
        gx_acc_j_0 += wi_3 * yk_3 * go_v_5;
        gx_acc_j_0 += wi_8 * yk_5 * go_v_19;
        gx_acc_j_0 += wi_8 * yk_7 * go_v_21;
        gx_acc_j_0 += wi_13 * yk_9 * go_v_43;
        gx_acc_j_0 += wi_13 * yk_11 * go_v_45;
        gx_acc_j_0 += wi_13 * yk_13 * go_v_47;
        gx_acc_j_0 += wi_13 * yk_15 * go_v_49;
        atomicAdd(&grad_x[((int64_t)src * Ix + 0) * (int64_t)U + u], gx_acc_j_0);

        // grad_y accumulate by unique k, reduced across current 32-channel tile
        scalar_t gy_lane_k_13 = scalar_t(0);
        gy_lane_k_13 += scalar_t(-0.119522861f) * wi_7 * xj_4 * go_v_15;
        gy_lane_k_13 += scalar_t(0.478091444f) * wi_7 * xj_7 * go_v_16;
        gy_lane_k_13 += scalar_t(0.447213595f) * wi_16 * xj_4 * go_v_66;
        gy_lane_k_13 += scalar_t(-0.288675135f) * wi_16 * xj_4 * go_v_64;
        gy_lane_k_13 += scalar_t(0.353553391f) * wi_16 * xj_5 * go_v_65;
        gy_lane_k_13 += scalar_t(-0.15430335f) * wi_10 * xj_3 * go_v_32;
        gy_lane_k_13 += scalar_t(0.534522484f) * wi_10 * xj_3 * go_v_30;
        gy_lane_k_13 += wi_13 * xj_0 * go_v_47;
        scalar_t gy_sum_k_13 = warp_sum(gy_lane_k_13);
        if (lane == 0) atomicAdd(&grad_y[y_base + 13], gy_sum_k_13);

        scalar_t gy_lane_k_15 = scalar_t(0);
        gy_lane_k_15 += scalar_t(-0.46291005f) * wi_7 * xj_4 * go_v_15;
        gy_lane_k_15 += scalar_t(0.288675135f) * wi_16 * xj_4 * go_v_66;
        gy_lane_k_15 += scalar_t(-0.456435465f) * wi_16 * xj_5 * go_v_65;
        gy_lane_k_15 += scalar_t(0.597614305f) * wi_10 * xj_3 * go_v_32;
        gy_lane_k_15 += wi_13 * xj_0 * go_v_49;
        scalar_t gy_sum_k_15 = warp_sum(gy_lane_k_15);
        if (lane == 0) atomicAdd(&grad_y[y_base + 15], gy_sum_k_15);

        scalar_t gy_lane_k_12 = scalar_t(0);
        gy_lane_k_12 += scalar_t(-0.292770022f) * wi_7 * xj_5 * go_v_15;
        gy_lane_k_12 += scalar_t(0.507092553f) * wi_7 * xj_6 * go_v_16;
        gy_lane_k_12 += scalar_t(0.182574186f) * wi_16 * xj_5 * go_v_66;
        gy_lane_k_12 += scalar_t(-0.577350269f) * wi_16 * xj_4 * go_v_65;
        gy_lane_k_12 += scalar_t(-0.377964473f) * wi_10 * xj_1 * go_v_29;
        gy_lane_k_12 += scalar_t(0.654653671f) * wi_10 * xj_2 * go_v_30;
        scalar_t gy_sum_k_12 = warp_sum(gy_lane_k_12);
        if (lane == 0) atomicAdd(&grad_y[y_base + 12], gy_sum_k_12);

        scalar_t gy_lane_k_14 = scalar_t(0);
        gy_lane_k_14 += scalar_t(-0.377964473f) * wi_7 * xj_5 * go_v_15;
        gy_lane_k_14 += scalar_t(0.377964473f) * wi_7 * xj_8 * go_v_16;
        gy_lane_k_14 += scalar_t(-0.353553391f) * wi_16 * xj_5 * go_v_66;
        gy_lane_k_14 += scalar_t(0.456435465f) * wi_16 * xj_5 * go_v_64;
        gy_lane_k_14 += scalar_t(0.487950036f) * wi_10 * xj_2 * go_v_32;
        gy_lane_k_14 += scalar_t(-0.487950036f) * wi_10 * xj_1 * go_v_29;
        scalar_t gy_sum_k_14 = warp_sum(gy_lane_k_14);
        if (lane == 0) atomicAdd(&grad_y[y_base + 14], gy_sum_k_14);

        scalar_t gy_lane_k_11 = scalar_t(0);
        gy_lane_k_11 += scalar_t(0.414039336f) * wi_7 * xj_6 * go_v_15;
        gy_lane_k_11 += scalar_t(0.119522861f) * wi_7 * xj_8 * go_v_15;
        gy_lane_k_11 += scalar_t(0.478091444f) * wi_7 * xj_5 * go_v_16;
        gy_lane_k_11 += scalar_t(0.387298335f) * wi_16 * xj_6 * go_v_66;
        gy_lane_k_11 += scalar_t(-0.447213595f) * wi_16 * xj_8 * go_v_66;
        gy_lane_k_11 += scalar_t(-0.288675135f) * wi_16 * xj_8 * go_v_64;
        gy_lane_k_11 += scalar_t(0.353553391f) * wi_16 * xj_7 * go_v_65;
        gy_lane_k_11 += scalar_t(0.15430335f) * wi_10 * xj_1 * go_v_32;
        gy_lane_k_11 += scalar_t(0.6172134f) * wi_10 * xj_2 * go_v_29;
        gy_lane_k_11 += scalar_t(0.534522484f) * wi_10 * xj_1 * go_v_30;
        gy_lane_k_11 += wi_13 * xj_0 * go_v_45;
        scalar_t gy_sum_k_11 = warp_sum(gy_lane_k_11);
        if (lane == 0) atomicAdd(&grad_y[y_base + 11], gy_sum_k_11);

        scalar_t gy_lane_k_10 = scalar_t(0);
        gy_lane_k_10 += scalar_t(0.377964473f) * wi_7 * xj_7 * go_v_15;
        gy_lane_k_10 += scalar_t(0.377964473f) * wi_7 * xj_4 * go_v_16;
        gy_lane_k_10 += scalar_t(0.353553391f) * wi_16 * xj_7 * go_v_66;
        gy_lane_k_10 += scalar_t(0.456435465f) * wi_16 * xj_7 * go_v_64;
        gy_lane_k_10 += scalar_t(0.487950036f) * wi_10 * xj_3 * go_v_29;
        scalar_t gy_sum_k_10 = warp_sum(gy_lane_k_10);
        if (lane == 0) atomicAdd(&grad_y[y_base + 10], gy_sum_k_10);

        scalar_t gy_lane_k_9 = scalar_t(0);
        gy_lane_k_9 += scalar_t(0.46291005f) * wi_7 * xj_8 * go_v_15;
        gy_lane_k_9 += scalar_t(-0.288675135f) * wi_16 * xj_8 * go_v_66;
        gy_lane_k_9 += scalar_t(-0.645497224f) * wi_16 * xj_6 * go_v_64;
        gy_lane_k_9 += scalar_t(0.456435465f) * wi_16 * xj_7 * go_v_65;
        gy_lane_k_9 += scalar_t(0.597614305f) * wi_10 * xj_1 * go_v_32;
        gy_lane_k_9 += wi_13 * xj_0 * go_v_43;
        scalar_t gy_sum_k_9 = warp_sum(gy_lane_k_9);
        if (lane == 0) atomicAdd(&grad_y[y_base + 9], gy_sum_k_9);

        scalar_t gy_lane_k_5 = scalar_t(0);
        gy_lane_k_5 += scalar_t(0.46291005f) * wi_12 * xj_4 * go_v_41;
        gy_lane_k_5 += scalar_t(0.267261242f) * wi_12 * xj_5 * go_v_40;
        gy_lane_k_5 += scalar_t(0.46291005f) * wi_12 * xj_7 * go_v_38;
        gy_lane_k_5 += scalar_t(0.547722558f) * wi_5 * xj_2 * go_v_9;
        gy_lane_k_5 += scalar_t(-0.447213595f) * wi_14 * xj_1 * go_v_53;
        gy_lane_k_5 += scalar_t(0.730296743f) * wi_14 * xj_2 * go_v_52;
        gy_lane_k_5 += scalar_t(0.577350269f) * wi_14 * xj_3 * go_v_51;
        gy_lane_k_5 += wi_8 * xj_0 * go_v_19;
        scalar_t gy_sum_k_5 = warp_sum(gy_lane_k_5);
        if (lane == 0) atomicAdd(&grad_y[y_base + 5], gy_sum_k_5);

        scalar_t gy_lane_k_4 = scalar_t(0);
        gy_lane_k_4 += scalar_t(0.46291005f) * wi_12 * xj_5 * go_v_41;
        gy_lane_k_4 += scalar_t(-0.534522484f) * wi_12 * xj_4 * go_v_40;
        gy_lane_k_4 += scalar_t(-0.534522484f) * wi_12 * xj_6 * go_v_38;
        gy_lane_k_4 += scalar_t(0.547722558f) * wi_5 * xj_3 * go_v_9;
        gy_lane_k_4 += scalar_t(-0.182574186f) * wi_14 * xj_3 * go_v_52;
        gy_lane_k_4 += scalar_t(0.577350269f) * wi_14 * xj_2 * go_v_51;
        gy_lane_k_4 += scalar_t(0.707106781f) * wi_14 * xj_3 * go_v_50;
        scalar_t gy_sum_k_4 = warp_sum(gy_lane_k_4);
        if (lane == 0) atomicAdd(&grad_y[y_base + 4], gy_sum_k_4);

        scalar_t gy_lane_k_7 = scalar_t(0);
        gy_lane_k_7 += scalar_t(0.267261242f) * wi_12 * xj_6 * go_v_41;
        gy_lane_k_7 += scalar_t(0.46291005f) * wi_12 * xj_8 * go_v_41;
        gy_lane_k_7 += scalar_t(0.267261242f) * wi_12 * xj_7 * go_v_40;
        gy_lane_k_7 += scalar_t(0.46291005f) * wi_12 * xj_5 * go_v_38;
        gy_lane_k_7 += scalar_t(-0.447213595f) * wi_14 * xj_3 * go_v_53;
        gy_lane_k_7 += scalar_t(0.577350269f) * wi_14 * xj_1 * go_v_51;
        gy_lane_k_7 += wi_8 * xj_0 * go_v_21;
        scalar_t gy_sum_k_7 = warp_sum(gy_lane_k_7);
        if (lane == 0) atomicAdd(&grad_y[y_base + 7], gy_sum_k_7);

        scalar_t gy_lane_k_6 = scalar_t(0);
        gy_lane_k_6 += scalar_t(0.267261242f) * wi_12 * xj_7 * go_v_41;
        gy_lane_k_6 += scalar_t(0.534522484f) * wi_12 * xj_6 * go_v_40;
        gy_lane_k_6 += scalar_t(-0.534522484f) * wi_12 * xj_4 * go_v_38;
        gy_lane_k_6 += scalar_t(-0.316227766f) * wi_5 * xj_1 * go_v_9;
        gy_lane_k_6 += scalar_t(0.774596669f) * wi_14 * xj_2 * go_v_53;
        gy_lane_k_6 += scalar_t(0.632455532f) * wi_14 * xj_1 * go_v_52;
        scalar_t gy_sum_k_6 = warp_sum(gy_lane_k_6);
        if (lane == 0) atomicAdd(&grad_y[y_base + 6], gy_sum_k_6);

        scalar_t gy_lane_k_8 = scalar_t(0);
        gy_lane_k_8 += scalar_t(0.46291005f) * wi_12 * xj_7 * go_v_41;
        gy_lane_k_8 += scalar_t(-0.534522484f) * wi_12 * xj_8 * go_v_40;
        gy_lane_k_8 += scalar_t(-0.547722558f) * wi_5 * xj_1 * go_v_9;
        gy_lane_k_8 += scalar_t(0.182574186f) * wi_14 * xj_1 * go_v_52;
        gy_lane_k_8 += scalar_t(0.707106781f) * wi_14 * xj_1 * go_v_50;
        scalar_t gy_sum_k_8 = warp_sum(gy_lane_k_8);
        if (lane == 0) atomicAdd(&grad_y[y_base + 8], gy_sum_k_8);

        scalar_t gy_lane_k_3 = scalar_t(0);
        gy_lane_k_3 += scalar_t(-0.182574186f) * wi_15 * xj_4 * go_v_59;
        gy_lane_k_3 += scalar_t(-0.316227766f) * wi_6 * xj_6 * go_v_14;
        gy_lane_k_3 += scalar_t(0.547722558f) * wi_6 * xj_8 * go_v_14;
        gy_lane_k_3 += scalar_t(-0.447213595f) * wi_15 * xj_7 * go_v_60;
        gy_lane_k_3 += scalar_t(0.577350269f) * wi_15 * xj_5 * go_v_58;
        gy_lane_k_3 += scalar_t(0.577350269f) * wi_1 * xj_3 * go_v_1;
        gy_lane_k_3 += scalar_t(0.707106781f) * wi_9 * xj_3 * go_v_27;
        gy_lane_k_3 += wi_3 * xj_0 * go_v_5;
        scalar_t gy_sum_k_3 = warp_sum(gy_lane_k_3);
        if (lane == 0) atomicAdd(&grad_y[y_base + 3], gy_sum_k_3);

        scalar_t gy_lane_k_2 = scalar_t(0);
        gy_lane_k_2 += scalar_t(0.730296743f) * wi_15 * xj_5 * go_v_59;
        gy_lane_k_2 += scalar_t(0.547722558f) * wi_6 * xj_7 * go_v_14;
        gy_lane_k_2 += scalar_t(0.774596669f) * wi_15 * xj_6 * go_v_60;
        gy_lane_k_2 += scalar_t(0.577350269f) * wi_15 * xj_4 * go_v_58;
        gy_lane_k_2 += scalar_t(0.577350269f) * wi_1 * xj_2 * go_v_1;
        gy_lane_k_2 += scalar_t(0.707106781f) * wi_9 * xj_1 * go_v_24;
        scalar_t gy_sum_k_2 = warp_sum(gy_lane_k_2);
        if (lane == 0) atomicAdd(&grad_y[y_base + 2], gy_sum_k_2);

        scalar_t gy_lane_k_1 = scalar_t(0);
        gy_lane_k_1 += scalar_t(0.632455532f) * wi_15 * xj_6 * go_v_59;
        gy_lane_k_1 += scalar_t(0.182574186f) * wi_15 * xj_8 * go_v_59;
        gy_lane_k_1 += scalar_t(0.547722558f) * wi_6 * xj_4 * go_v_14;
        gy_lane_k_1 += scalar_t(-0.447213595f) * wi_15 * xj_5 * go_v_60;
        gy_lane_k_1 += scalar_t(0.577350269f) * wi_15 * xj_7 * go_v_58;
        gy_lane_k_1 += scalar_t(0.577350269f) * wi_1 * xj_1 * go_v_1;
        gy_lane_k_1 += scalar_t(-0.707106781f) * wi_9 * xj_1 * go_v_27;
        gy_lane_k_1 += scalar_t(0.707106781f) * wi_9 * xj_2 * go_v_24;
        gy_lane_k_1 += wi_3 * xj_0 * go_v_3;
        scalar_t gy_sum_k_1 = warp_sum(gy_lane_k_1);
        if (lane == 0) atomicAdd(&grad_y[y_base + 1], gy_sum_k_1);

        scalar_t gy_lane_k_0 = scalar_t(0);
        gy_lane_k_0 += wi_11 * xj_8 * go_v_37;
        gy_lane_k_0 += wi_11 * xj_7 * go_v_36;
        gy_lane_k_0 += wi_11 * xj_5 * go_v_34;
        gy_lane_k_0 += wi_4 * xj_3 * go_v_8;
        gy_lane_k_0 += wi_4 * xj_1 * go_v_6;
        scalar_t gy_sum_k_0 = warp_sum(gy_lane_k_0);
        if (lane == 0) atomicAdd(&grad_y[y_base + 0], gy_sum_k_0);

    }

    if (warp == 1) {
        // preload w(i, u)
        scalar_t wi_7 = w[((int64_t)b * Iw + 7) * (int64_t)U + u];
        scalar_t wi_16 = w[((int64_t)b * Iw + 16) * (int64_t)U + u];
        scalar_t wi_12 = w[((int64_t)b * Iw + 12) * (int64_t)U + u];
        scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
        scalar_t wi_6 = w[((int64_t)b * Iw + 6) * (int64_t)U + u];
        scalar_t wi_15 = w[((int64_t)b * Iw + 15) * (int64_t)U + u];
        scalar_t wi_11 = w[((int64_t)b * Iw + 11) * (int64_t)U + u];
        scalar_t wi_10 = w[((int64_t)b * Iw + 10) * (int64_t)U + u];
        scalar_t wi_9 = w[((int64_t)b * Iw + 9) * (int64_t)U + u];
        scalar_t wi_5 = w[((int64_t)b * Iw + 5) * (int64_t)U + u];
        scalar_t wi_14 = w[((int64_t)b * Iw + 14) * (int64_t)U + u];
        scalar_t wi_4 = w[((int64_t)b * Iw + 4) * (int64_t)U + u];
        scalar_t wi_0 = w[((int64_t)b * Iw + 0) * (int64_t)U + u];
        scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];
        scalar_t wi_8 = w[((int64_t)b * Iw + 8) * (int64_t)U + u];
        scalar_t wi_13 = w[((int64_t)b * Iw + 13) * (int64_t)U + u];

        // preload x(j, u)
        scalar_t xj_4 = x_all[((int64_t)src * Ix + 4) * (int64_t)U + u];
        scalar_t xj_5 = x_all[((int64_t)src * Ix + 5) * (int64_t)U + u];
        scalar_t xj_6 = x_all[((int64_t)src * Ix + 6) * (int64_t)U + u];
        scalar_t xj_7 = x_all[((int64_t)src * Ix + 7) * (int64_t)U + u];
        scalar_t xj_8 = x_all[((int64_t)src * Ix + 8) * (int64_t)U + u];
        scalar_t xj_1 = x_all[((int64_t)src * Ix + 1) * (int64_t)U + u];
        scalar_t xj_2 = x_all[((int64_t)src * Ix + 2) * (int64_t)U + u];
        scalar_t xj_3 = x_all[((int64_t)src * Ix + 3) * (int64_t)U + u];
        scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];

        // preload y(k)
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_0 = y[y_base + 0];

        // preload grad_out(v, u)
        scalar_t go_v_17 = grad_out[((int64_t)dst * V + 17) * (int64_t)U + u];
        scalar_t go_v_68 = grad_out[((int64_t)dst * V + 68) * (int64_t)U + u];
        scalar_t go_v_67 = grad_out[((int64_t)dst * V + 67) * (int64_t)U + u];
        scalar_t go_v_70 = grad_out[((int64_t)dst * V + 70) * (int64_t)U + u];
        scalar_t go_v_39 = grad_out[((int64_t)dst * V + 39) * (int64_t)U + u];
        scalar_t go_v_2 = grad_out[((int64_t)dst * V + 2) * (int64_t)U + u];
        scalar_t go_v_42 = grad_out[((int64_t)dst * V + 42) * (int64_t)U + u];
        scalar_t go_v_12 = grad_out[((int64_t)dst * V + 12) * (int64_t)U + u];
        scalar_t go_v_61 = grad_out[((int64_t)dst * V + 61) * (int64_t)U + u];
        scalar_t go_v_13 = grad_out[((int64_t)dst * V + 13) * (int64_t)U + u];
        scalar_t go_v_69 = grad_out[((int64_t)dst * V + 69) * (int64_t)U + u];
        scalar_t go_v_62 = grad_out[((int64_t)dst * V + 62) * (int64_t)U + u];
        scalar_t go_v_63 = grad_out[((int64_t)dst * V + 63) * (int64_t)U + u];
        scalar_t go_v_57 = grad_out[((int64_t)dst * V + 57) * (int64_t)U + u];
        scalar_t go_v_35 = grad_out[((int64_t)dst * V + 35) * (int64_t)U + u];
        scalar_t go_v_33 = grad_out[((int64_t)dst * V + 33) * (int64_t)U + u];
        scalar_t go_v_28 = grad_out[((int64_t)dst * V + 28) * (int64_t)U + u];
        scalar_t go_v_31 = grad_out[((int64_t)dst * V + 31) * (int64_t)U + u];
        scalar_t go_v_25 = grad_out[((int64_t)dst * V + 25) * (int64_t)U + u];
        scalar_t go_v_11 = grad_out[((int64_t)dst * V + 11) * (int64_t)U + u];
        scalar_t go_v_10 = grad_out[((int64_t)dst * V + 10) * (int64_t)U + u];
        scalar_t go_v_54 = grad_out[((int64_t)dst * V + 54) * (int64_t)U + u];
        scalar_t go_v_55 = grad_out[((int64_t)dst * V + 55) * (int64_t)U + u];
        scalar_t go_v_56 = grad_out[((int64_t)dst * V + 56) * (int64_t)U + u];
        scalar_t go_v_23 = grad_out[((int64_t)dst * V + 23) * (int64_t)U + u];
        scalar_t go_v_26 = grad_out[((int64_t)dst * V + 26) * (int64_t)U + u];
        scalar_t go_v_7 = grad_out[((int64_t)dst * V + 7) * (int64_t)U + u];
        scalar_t go_v_0 = grad_out[((int64_t)dst * V + 0) * (int64_t)U + u];
        scalar_t go_v_4 = grad_out[((int64_t)dst * V + 4) * (int64_t)U + u];
        scalar_t go_v_18 = grad_out[((int64_t)dst * V + 18) * (int64_t)U + u];
        scalar_t go_v_20 = grad_out[((int64_t)dst * V + 20) * (int64_t)U + u];
        scalar_t go_v_22 = grad_out[((int64_t)dst * V + 22) * (int64_t)U + u];
        scalar_t go_v_44 = grad_out[((int64_t)dst * V + 44) * (int64_t)U + u];
        scalar_t go_v_46 = grad_out[((int64_t)dst * V + 46) * (int64_t)U + u];
        scalar_t go_v_48 = grad_out[((int64_t)dst * V + 48) * (int64_t)U + u];

        // grad_w accumulate by unique i
        scalar_t gw_acc_i_7 = scalar_t(0);
        gw_acc_i_7 += scalar_t(0.46291005f) * xj_4 * yk_9 * go_v_17;
        gw_acc_i_7 += scalar_t(-0.119522861f) * xj_4 * yk_11 * go_v_17;
        gw_acc_i_7 += scalar_t(0.377964473f) * xj_5 * yk_10 * go_v_17;
        gw_acc_i_7 += scalar_t(0.414039336f) * xj_6 * yk_13 * go_v_17;
        gw_acc_i_7 += scalar_t(-0.292770022f) * xj_7 * yk_12 * go_v_17;
        gw_acc_i_7 += scalar_t(0.377964473f) * xj_7 * yk_14 * go_v_17;
        gw_acc_i_7 += scalar_t(-0.119522861f) * xj_8 * yk_13 * go_v_17;
        gw_acc_i_7 += scalar_t(0.46291005f) * xj_8 * yk_15 * go_v_17;
        atomicAdd(&grad_w[((int64_t)b * Iw + 7) * (int64_t)U + u], gw_acc_i_7);

        scalar_t gw_acc_i_16 = scalar_t(0);
        gw_acc_i_16 += scalar_t(-0.288675135f) * xj_4 * yk_9 * go_v_68;
        gw_acc_i_16 += scalar_t(0.447213595f) * xj_4 * yk_11 * go_v_68;
        gw_acc_i_16 += scalar_t(0.353553391f) * xj_5 * yk_10 * go_v_68;
        gw_acc_i_16 += scalar_t(0.387298335f) * xj_6 * yk_13 * go_v_68;
        gw_acc_i_16 += scalar_t(0.182574186f) * xj_7 * yk_12 * go_v_68;
        gw_acc_i_16 += scalar_t(0.353553391f) * xj_7 * yk_14 * go_v_68;
        gw_acc_i_16 += scalar_t(0.447213595f) * xj_8 * yk_13 * go_v_68;
        gw_acc_i_16 += scalar_t(-0.288675135f) * xj_8 * yk_15 * go_v_68;
        gw_acc_i_16 += scalar_t(-0.577350269f) * xj_4 * yk_10 * go_v_67;
        gw_acc_i_16 += scalar_t(0.182574186f) * xj_5 * yk_11 * go_v_67;
        gw_acc_i_16 += scalar_t(0.516397779f) * xj_6 * yk_12 * go_v_67;
        gw_acc_i_16 += scalar_t(0.182574186f) * xj_7 * yk_13 * go_v_67;
        gw_acc_i_16 += scalar_t(-0.577350269f) * xj_8 * yk_14 * go_v_67;
        gw_acc_i_16 += scalar_t(0.288675135f) * xj_4 * yk_11 * go_v_70;
        gw_acc_i_16 += scalar_t(-0.456435465f) * xj_5 * yk_10 * go_v_70;
        gw_acc_i_16 += scalar_t(-0.645497224f) * xj_6 * yk_15 * go_v_70;
        gw_acc_i_16 += scalar_t(0.456435465f) * xj_7 * yk_14 * go_v_70;
        gw_acc_i_16 += scalar_t(-0.288675135f) * xj_8 * yk_13 * go_v_70;
        gw_acc_i_16 += scalar_t(0.456435465f) * xj_5 * yk_9 * go_v_69;
        gw_acc_i_16 += scalar_t(-0.353553391f) * xj_5 * yk_11 * go_v_69;
        gw_acc_i_16 += scalar_t(0.353553391f) * xj_7 * yk_13 * go_v_69;
        gw_acc_i_16 += scalar_t(0.456435465f) * xj_7 * yk_15 * go_v_69;
        gw_acc_i_16 += scalar_t(-0.577350269f) * xj_8 * yk_12 * go_v_69;
        atomicAdd(&grad_w[((int64_t)b * Iw + 16) * (int64_t)U + u], gw_acc_i_16);

        scalar_t gw_acc_i_12 = scalar_t(0);
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_4 * yk_7 * go_v_39;
        gw_acc_i_12 += scalar_t(0.267261242f) * xj_5 * yk_6 * go_v_39;
        gw_acc_i_12 += scalar_t(-0.46291005f) * xj_5 * yk_8 * go_v_39;
        gw_acc_i_12 += scalar_t(0.267261242f) * xj_6 * yk_5 * go_v_39;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_7 * yk_4 * go_v_39;
        gw_acc_i_12 += scalar_t(-0.46291005f) * xj_8 * yk_5 * go_v_39;
        gw_acc_i_12 += scalar_t(-0.46291005f) * xj_5 * yk_5 * go_v_42;
        gw_acc_i_12 += scalar_t(-0.534522484f) * xj_6 * yk_8 * go_v_42;
        gw_acc_i_12 += scalar_t(0.46291005f) * xj_7 * yk_7 * go_v_42;
        gw_acc_i_12 += scalar_t(-0.534522484f) * xj_8 * yk_6 * go_v_42;
        atomicAdd(&grad_w[((int64_t)b * Iw + 12) * (int64_t)U + u], gw_acc_i_12);

        scalar_t gw_acc_i_2 = scalar_t(0);
        gw_acc_i_2 += scalar_t(0.447213595f) * xj_4 * yk_4 * go_v_2;
        gw_acc_i_2 += scalar_t(0.447213595f) * xj_5 * yk_5 * go_v_2;
        gw_acc_i_2 += scalar_t(0.447213595f) * xj_6 * yk_6 * go_v_2;
        gw_acc_i_2 += scalar_t(0.447213595f) * xj_7 * yk_7 * go_v_2;
        gw_acc_i_2 += scalar_t(0.447213595f) * xj_8 * yk_8 * go_v_2;
        atomicAdd(&grad_w[((int64_t)b * Iw + 2) * (int64_t)U + u], gw_acc_i_2);

        scalar_t gw_acc_i_6 = scalar_t(0);
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_4 * yk_3 * go_v_12;
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_5 * yk_2 * go_v_12;
        gw_acc_i_6 += scalar_t(-0.316227766f) * xj_6 * yk_1 * go_v_12;
        gw_acc_i_6 += scalar_t(-0.547722558f) * xj_8 * yk_1 * go_v_12;
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_5 * yk_1 * go_v_13;
        gw_acc_i_6 += scalar_t(0.632455532f) * xj_6 * yk_2 * go_v_13;
        gw_acc_i_6 += scalar_t(0.547722558f) * xj_7 * yk_3 * go_v_13;
        atomicAdd(&grad_w[((int64_t)b * Iw + 6) * (int64_t)U + u], gw_acc_i_6);

        scalar_t gw_acc_i_15 = scalar_t(0);
        gw_acc_i_15 += scalar_t(-0.182574186f) * xj_4 * yk_1 * go_v_61;
        gw_acc_i_15 += scalar_t(0.632455532f) * xj_6 * yk_3 * go_v_61;
        gw_acc_i_15 += scalar_t(0.730296743f) * xj_7 * yk_2 * go_v_61;
        gw_acc_i_15 += scalar_t(-0.182574186f) * xj_8 * yk_3 * go_v_61;
        gw_acc_i_15 += scalar_t(-0.577350269f) * xj_5 * yk_1 * go_v_62;
        gw_acc_i_15 += scalar_t(0.577350269f) * xj_7 * yk_3 * go_v_62;
        gw_acc_i_15 += scalar_t(0.577350269f) * xj_8 * yk_2 * go_v_62;
        gw_acc_i_15 += scalar_t(-0.707106781f) * xj_4 * yk_1 * go_v_63;
        gw_acc_i_15 += scalar_t(0.707106781f) * xj_8 * yk_3 * go_v_63;
        gw_acc_i_15 += scalar_t(0.707106781f) * xj_4 * yk_3 * go_v_57;
        gw_acc_i_15 += scalar_t(0.707106781f) * xj_8 * yk_1 * go_v_57;
        atomicAdd(&grad_w[((int64_t)b * Iw + 15) * (int64_t)U + u], gw_acc_i_15);

        scalar_t gw_acc_i_11 = scalar_t(0);
        gw_acc_i_11 += xj_6 * yk_0 * go_v_35;
        gw_acc_i_11 += xj_4 * yk_0 * go_v_33;
        atomicAdd(&grad_w[((int64_t)b * Iw + 11) * (int64_t)U + u], gw_acc_i_11);

        scalar_t gw_acc_i_10 = scalar_t(0);
        gw_acc_i_10 += scalar_t(-0.15430335f) * xj_1 * yk_13 * go_v_28;
        gw_acc_i_10 += scalar_t(-0.597614305f) * xj_1 * yk_15 * go_v_28;
        gw_acc_i_10 += scalar_t(0.487950036f) * xj_2 * yk_10 * go_v_28;
        gw_acc_i_10 += scalar_t(0.597614305f) * xj_3 * yk_9 * go_v_28;
        gw_acc_i_10 += scalar_t(-0.15430335f) * xj_3 * yk_11 * go_v_28;
        gw_acc_i_10 += scalar_t(0.487950036f) * xj_1 * yk_10 * go_v_31;
        gw_acc_i_10 += scalar_t(0.6172134f) * xj_2 * yk_13 * go_v_31;
        gw_acc_i_10 += scalar_t(-0.377964473f) * xj_3 * yk_12 * go_v_31;
        gw_acc_i_10 += scalar_t(0.487950036f) * xj_3 * yk_14 * go_v_31;
        atomicAdd(&grad_w[((int64_t)b * Iw + 10) * (int64_t)U + u], gw_acc_i_10);

        scalar_t gw_acc_i_9 = scalar_t(0);
        gw_acc_i_9 += scalar_t(-0.40824829f) * xj_1 * yk_1 * go_v_25;
        gw_acc_i_9 += scalar_t(0.816496581f) * xj_2 * yk_2 * go_v_25;
        gw_acc_i_9 += scalar_t(-0.40824829f) * xj_3 * yk_3 * go_v_25;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_1 * yk_3 * go_v_23;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_3 * yk_1 * go_v_23;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_2 * yk_3 * go_v_26;
        gw_acc_i_9 += scalar_t(0.707106781f) * xj_3 * yk_2 * go_v_26;
        atomicAdd(&grad_w[((int64_t)b * Iw + 9) * (int64_t)U + u], gw_acc_i_9);

        scalar_t gw_acc_i_5 = scalar_t(0);
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_1 * yk_4 * go_v_11;
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_2 * yk_7 * go_v_11;
        gw_acc_i_5 += scalar_t(-0.316227766f) * xj_3 * yk_6 * go_v_11;
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_3 * yk_8 * go_v_11;
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_1 * yk_5 * go_v_10;
        gw_acc_i_5 += scalar_t(0.632455532f) * xj_2 * yk_6 * go_v_10;
        gw_acc_i_5 += scalar_t(0.547722558f) * xj_3 * yk_7 * go_v_10;
        atomicAdd(&grad_w[((int64_t)b * Iw + 5) * (int64_t)U + u], gw_acc_i_5);

        scalar_t gw_acc_i_14 = scalar_t(0);
        gw_acc_i_14 += scalar_t(-0.182574186f) * xj_1 * yk_4 * go_v_54;
        gw_acc_i_14 += scalar_t(0.730296743f) * xj_2 * yk_7 * go_v_54;
        gw_acc_i_14 += scalar_t(0.632455532f) * xj_3 * yk_6 * go_v_54;
        gw_acc_i_14 += scalar_t(-0.182574186f) * xj_3 * yk_8 * go_v_54;
        gw_acc_i_14 += scalar_t(-0.577350269f) * xj_1 * yk_5 * go_v_55;
        gw_acc_i_14 += scalar_t(0.577350269f) * xj_2 * yk_8 * go_v_55;
        gw_acc_i_14 += scalar_t(0.577350269f) * xj_3 * yk_7 * go_v_55;
        gw_acc_i_14 += scalar_t(-0.707106781f) * xj_1 * yk_4 * go_v_56;
        gw_acc_i_14 += scalar_t(0.707106781f) * xj_3 * yk_8 * go_v_56;
        atomicAdd(&grad_w[((int64_t)b * Iw + 14) * (int64_t)U + u], gw_acc_i_14);

        scalar_t gw_acc_i_4 = scalar_t(0);
        gw_acc_i_4 += xj_2 * yk_0 * go_v_7;
        atomicAdd(&grad_w[((int64_t)b * Iw + 4) * (int64_t)U + u], gw_acc_i_4);

        scalar_t gw_acc_i_0 = scalar_t(0);
        gw_acc_i_0 += xj_0 * yk_0 * go_v_0;
        atomicAdd(&grad_w[((int64_t)b * Iw + 0) * (int64_t)U + u], gw_acc_i_0);

        scalar_t gw_acc_i_3 = scalar_t(0);
        gw_acc_i_3 += xj_0 * yk_2 * go_v_4;
        atomicAdd(&grad_w[((int64_t)b * Iw + 3) * (int64_t)U + u], gw_acc_i_3);

        scalar_t gw_acc_i_8 = scalar_t(0);
        gw_acc_i_8 += xj_0 * yk_4 * go_v_18;
        gw_acc_i_8 += xj_0 * yk_6 * go_v_20;
        gw_acc_i_8 += xj_0 * yk_8 * go_v_22;
        atomicAdd(&grad_w[((int64_t)b * Iw + 8) * (int64_t)U + u], gw_acc_i_8);

        scalar_t gw_acc_i_13 = scalar_t(0);
        gw_acc_i_13 += xj_0 * yk_10 * go_v_44;
        gw_acc_i_13 += xj_0 * yk_12 * go_v_46;
        gw_acc_i_13 += xj_0 * yk_14 * go_v_48;
        atomicAdd(&grad_w[((int64_t)b * Iw + 13) * (int64_t)U + u], gw_acc_i_13);

        // grad_x accumulate by unique j
        scalar_t gx_acc_j_4 = scalar_t(0);
        gx_acc_j_4 += scalar_t(0.46291005f) * wi_7 * yk_9 * go_v_17;
        gx_acc_j_4 += scalar_t(-0.119522861f) * wi_7 * yk_11 * go_v_17;
        gx_acc_j_4 += scalar_t(-0.288675135f) * wi_16 * yk_9 * go_v_68;
        gx_acc_j_4 += scalar_t(0.447213595f) * wi_16 * yk_11 * go_v_68;
        gx_acc_j_4 += scalar_t(-0.577350269f) * wi_16 * yk_10 * go_v_67;
        gx_acc_j_4 += scalar_t(0.288675135f) * wi_16 * yk_11 * go_v_70;
        gx_acc_j_4 += scalar_t(0.46291005f) * wi_12 * yk_7 * go_v_39;
        gx_acc_j_4 += scalar_t(0.447213595f) * wi_2 * yk_4 * go_v_2;
        gx_acc_j_4 += scalar_t(0.547722558f) * wi_6 * yk_3 * go_v_12;
        gx_acc_j_4 += scalar_t(-0.182574186f) * wi_15 * yk_1 * go_v_61;
        gx_acc_j_4 += scalar_t(-0.707106781f) * wi_15 * yk_1 * go_v_63;
        gx_acc_j_4 += scalar_t(0.707106781f) * wi_15 * yk_3 * go_v_57;
        gx_acc_j_4 += wi_11 * yk_0 * go_v_33;
        atomicAdd(&grad_x[((int64_t)src * Ix + 4) * (int64_t)U + u], gx_acc_j_4);

        scalar_t gx_acc_j_5 = scalar_t(0);
        gx_acc_j_5 += scalar_t(0.377964473f) * wi_7 * yk_10 * go_v_17;
        gx_acc_j_5 += scalar_t(0.353553391f) * wi_16 * yk_10 * go_v_68;
        gx_acc_j_5 += scalar_t(0.182574186f) * wi_16 * yk_11 * go_v_67;
        gx_acc_j_5 += scalar_t(-0.456435465f) * wi_16 * yk_10 * go_v_70;
        gx_acc_j_5 += scalar_t(0.267261242f) * wi_12 * yk_6 * go_v_39;
        gx_acc_j_5 += scalar_t(-0.46291005f) * wi_12 * yk_8 * go_v_39;
        gx_acc_j_5 += scalar_t(0.447213595f) * wi_2 * yk_5 * go_v_2;
        gx_acc_j_5 += scalar_t(-0.46291005f) * wi_12 * yk_5 * go_v_42;
        gx_acc_j_5 += scalar_t(0.547722558f) * wi_6 * yk_2 * go_v_12;
        gx_acc_j_5 += scalar_t(0.547722558f) * wi_6 * yk_1 * go_v_13;
        gx_acc_j_5 += scalar_t(0.456435465f) * wi_16 * yk_9 * go_v_69;
        gx_acc_j_5 += scalar_t(-0.353553391f) * wi_16 * yk_11 * go_v_69;
        gx_acc_j_5 += scalar_t(-0.577350269f) * wi_15 * yk_1 * go_v_62;
        atomicAdd(&grad_x[((int64_t)src * Ix + 5) * (int64_t)U + u], gx_acc_j_5);

        scalar_t gx_acc_j_6 = scalar_t(0);
        gx_acc_j_6 += scalar_t(0.414039336f) * wi_7 * yk_13 * go_v_17;
        gx_acc_j_6 += scalar_t(0.387298335f) * wi_16 * yk_13 * go_v_68;
        gx_acc_j_6 += scalar_t(0.516397779f) * wi_16 * yk_12 * go_v_67;
        gx_acc_j_6 += scalar_t(-0.645497224f) * wi_16 * yk_15 * go_v_70;
        gx_acc_j_6 += scalar_t(0.267261242f) * wi_12 * yk_5 * go_v_39;
        gx_acc_j_6 += scalar_t(0.447213595f) * wi_2 * yk_6 * go_v_2;
        gx_acc_j_6 += scalar_t(-0.534522484f) * wi_12 * yk_8 * go_v_42;
        gx_acc_j_6 += scalar_t(-0.316227766f) * wi_6 * yk_1 * go_v_12;
        gx_acc_j_6 += scalar_t(0.632455532f) * wi_15 * yk_3 * go_v_61;
        gx_acc_j_6 += scalar_t(0.632455532f) * wi_6 * yk_2 * go_v_13;
        gx_acc_j_6 += wi_11 * yk_0 * go_v_35;
        atomicAdd(&grad_x[((int64_t)src * Ix + 6) * (int64_t)U + u], gx_acc_j_6);

        scalar_t gx_acc_j_7 = scalar_t(0);
        gx_acc_j_7 += scalar_t(-0.292770022f) * wi_7 * yk_12 * go_v_17;
        gx_acc_j_7 += scalar_t(0.377964473f) * wi_7 * yk_14 * go_v_17;
        gx_acc_j_7 += scalar_t(0.182574186f) * wi_16 * yk_12 * go_v_68;
        gx_acc_j_7 += scalar_t(0.353553391f) * wi_16 * yk_14 * go_v_68;
        gx_acc_j_7 += scalar_t(0.182574186f) * wi_16 * yk_13 * go_v_67;
        gx_acc_j_7 += scalar_t(0.456435465f) * wi_16 * yk_14 * go_v_70;
        gx_acc_j_7 += scalar_t(0.46291005f) * wi_12 * yk_4 * go_v_39;
        gx_acc_j_7 += scalar_t(0.447213595f) * wi_2 * yk_7 * go_v_2;
        gx_acc_j_7 += scalar_t(0.46291005f) * wi_12 * yk_7 * go_v_42;
        gx_acc_j_7 += scalar_t(0.730296743f) * wi_15 * yk_2 * go_v_61;
        gx_acc_j_7 += scalar_t(0.547722558f) * wi_6 * yk_3 * go_v_13;
        gx_acc_j_7 += scalar_t(0.353553391f) * wi_16 * yk_13 * go_v_69;
        gx_acc_j_7 += scalar_t(0.456435465f) * wi_16 * yk_15 * go_v_69;
        gx_acc_j_7 += scalar_t(0.577350269f) * wi_15 * yk_3 * go_v_62;
        atomicAdd(&grad_x[((int64_t)src * Ix + 7) * (int64_t)U + u], gx_acc_j_7);

        scalar_t gx_acc_j_8 = scalar_t(0);
        gx_acc_j_8 += scalar_t(-0.119522861f) * wi_7 * yk_13 * go_v_17;
        gx_acc_j_8 += scalar_t(0.46291005f) * wi_7 * yk_15 * go_v_17;
        gx_acc_j_8 += scalar_t(0.447213595f) * wi_16 * yk_13 * go_v_68;
        gx_acc_j_8 += scalar_t(-0.288675135f) * wi_16 * yk_15 * go_v_68;
        gx_acc_j_8 += scalar_t(-0.577350269f) * wi_16 * yk_14 * go_v_67;
        gx_acc_j_8 += scalar_t(-0.288675135f) * wi_16 * yk_13 * go_v_70;
        gx_acc_j_8 += scalar_t(-0.46291005f) * wi_12 * yk_5 * go_v_39;
        gx_acc_j_8 += scalar_t(0.447213595f) * wi_2 * yk_8 * go_v_2;
        gx_acc_j_8 += scalar_t(-0.534522484f) * wi_12 * yk_6 * go_v_42;
        gx_acc_j_8 += scalar_t(-0.547722558f) * wi_6 * yk_1 * go_v_12;
        gx_acc_j_8 += scalar_t(-0.182574186f) * wi_15 * yk_3 * go_v_61;
        gx_acc_j_8 += scalar_t(-0.577350269f) * wi_16 * yk_12 * go_v_69;
        gx_acc_j_8 += scalar_t(0.577350269f) * wi_15 * yk_2 * go_v_62;
        gx_acc_j_8 += scalar_t(0.707106781f) * wi_15 * yk_3 * go_v_63;
        gx_acc_j_8 += scalar_t(0.707106781f) * wi_15 * yk_1 * go_v_57;
        atomicAdd(&grad_x[((int64_t)src * Ix + 8) * (int64_t)U + u], gx_acc_j_8);

        scalar_t gx_acc_j_1 = scalar_t(0);
        gx_acc_j_1 += scalar_t(-0.15430335f) * wi_10 * yk_13 * go_v_28;
        gx_acc_j_1 += scalar_t(-0.597614305f) * wi_10 * yk_15 * go_v_28;
        gx_acc_j_1 += scalar_t(0.487950036f) * wi_10 * yk_10 * go_v_31;
        gx_acc_j_1 += scalar_t(-0.40824829f) * wi_9 * yk_1 * go_v_25;
        gx_acc_j_1 += scalar_t(0.547722558f) * wi_5 * yk_4 * go_v_11;
        gx_acc_j_1 += scalar_t(0.547722558f) * wi_5 * yk_5 * go_v_10;
        gx_acc_j_1 += scalar_t(-0.182574186f) * wi_14 * yk_4 * go_v_54;
        gx_acc_j_1 += scalar_t(-0.577350269f) * wi_14 * yk_5 * go_v_55;
        gx_acc_j_1 += scalar_t(-0.707106781f) * wi_14 * yk_4 * go_v_56;
        gx_acc_j_1 += scalar_t(0.707106781f) * wi_9 * yk_3 * go_v_23;
        atomicAdd(&grad_x[((int64_t)src * Ix + 1) * (int64_t)U + u], gx_acc_j_1);

        scalar_t gx_acc_j_2 = scalar_t(0);
        gx_acc_j_2 += scalar_t(0.487950036f) * wi_10 * yk_10 * go_v_28;
        gx_acc_j_2 += scalar_t(0.6172134f) * wi_10 * yk_13 * go_v_31;
        gx_acc_j_2 += scalar_t(0.816496581f) * wi_9 * yk_2 * go_v_25;
        gx_acc_j_2 += scalar_t(0.547722558f) * wi_5 * yk_7 * go_v_11;
        gx_acc_j_2 += scalar_t(0.632455532f) * wi_5 * yk_6 * go_v_10;
        gx_acc_j_2 += scalar_t(0.730296743f) * wi_14 * yk_7 * go_v_54;
        gx_acc_j_2 += scalar_t(0.577350269f) * wi_14 * yk_8 * go_v_55;
        gx_acc_j_2 += scalar_t(0.707106781f) * wi_9 * yk_3 * go_v_26;
        gx_acc_j_2 += wi_4 * yk_0 * go_v_7;
        atomicAdd(&grad_x[((int64_t)src * Ix + 2) * (int64_t)U + u], gx_acc_j_2);

        scalar_t gx_acc_j_3 = scalar_t(0);
        gx_acc_j_3 += scalar_t(0.597614305f) * wi_10 * yk_9 * go_v_28;
        gx_acc_j_3 += scalar_t(-0.15430335f) * wi_10 * yk_11 * go_v_28;
        gx_acc_j_3 += scalar_t(-0.377964473f) * wi_10 * yk_12 * go_v_31;
        gx_acc_j_3 += scalar_t(0.487950036f) * wi_10 * yk_14 * go_v_31;
        gx_acc_j_3 += scalar_t(-0.40824829f) * wi_9 * yk_3 * go_v_25;
        gx_acc_j_3 += scalar_t(-0.316227766f) * wi_5 * yk_6 * go_v_11;
        gx_acc_j_3 += scalar_t(0.547722558f) * wi_5 * yk_8 * go_v_11;
        gx_acc_j_3 += scalar_t(0.547722558f) * wi_5 * yk_7 * go_v_10;
        gx_acc_j_3 += scalar_t(0.632455532f) * wi_14 * yk_6 * go_v_54;
        gx_acc_j_3 += scalar_t(-0.182574186f) * wi_14 * yk_8 * go_v_54;
        gx_acc_j_3 += scalar_t(0.577350269f) * wi_14 * yk_7 * go_v_55;
        gx_acc_j_3 += scalar_t(0.707106781f) * wi_14 * yk_8 * go_v_56;
        gx_acc_j_3 += scalar_t(0.707106781f) * wi_9 * yk_1 * go_v_23;
        gx_acc_j_3 += scalar_t(0.707106781f) * wi_9 * yk_2 * go_v_26;
        atomicAdd(&grad_x[((int64_t)src * Ix + 3) * (int64_t)U + u], gx_acc_j_3);

        scalar_t gx_acc_j_0 = scalar_t(0);
        gx_acc_j_0 += wi_0 * yk_0 * go_v_0;
        gx_acc_j_0 += wi_3 * yk_2 * go_v_4;
        gx_acc_j_0 += wi_8 * yk_4 * go_v_18;
        gx_acc_j_0 += wi_8 * yk_6 * go_v_20;
        gx_acc_j_0 += wi_8 * yk_8 * go_v_22;
        gx_acc_j_0 += wi_13 * yk_10 * go_v_44;
        gx_acc_j_0 += wi_13 * yk_12 * go_v_46;
        gx_acc_j_0 += wi_13 * yk_14 * go_v_48;
        atomicAdd(&grad_x[((int64_t)src * Ix + 0) * (int64_t)U + u], gx_acc_j_0);

        // grad_y accumulate by unique k, reduced across current 32-channel tile
        scalar_t gy_lane_k_9 = scalar_t(0);
        gy_lane_k_9 += scalar_t(0.46291005f) * wi_7 * xj_4 * go_v_17;
        gy_lane_k_9 += scalar_t(-0.288675135f) * wi_16 * xj_4 * go_v_68;
        gy_lane_k_9 += scalar_t(0.456435465f) * wi_16 * xj_5 * go_v_69;
        gy_lane_k_9 += scalar_t(0.597614305f) * wi_10 * xj_3 * go_v_28;
        scalar_t gy_sum_k_9 = warp_sum(gy_lane_k_9);
        if (lane == 0) atomicAdd(&grad_y[y_base + 9], gy_sum_k_9);

        scalar_t gy_lane_k_11 = scalar_t(0);
        gy_lane_k_11 += scalar_t(-0.119522861f) * wi_7 * xj_4 * go_v_17;
        gy_lane_k_11 += scalar_t(0.447213595f) * wi_16 * xj_4 * go_v_68;
        gy_lane_k_11 += scalar_t(0.182574186f) * wi_16 * xj_5 * go_v_67;
        gy_lane_k_11 += scalar_t(0.288675135f) * wi_16 * xj_4 * go_v_70;
        gy_lane_k_11 += scalar_t(-0.353553391f) * wi_16 * xj_5 * go_v_69;
        gy_lane_k_11 += scalar_t(-0.15430335f) * wi_10 * xj_3 * go_v_28;
        scalar_t gy_sum_k_11 = warp_sum(gy_lane_k_11);
        if (lane == 0) atomicAdd(&grad_y[y_base + 11], gy_sum_k_11);

        scalar_t gy_lane_k_10 = scalar_t(0);
        gy_lane_k_10 += scalar_t(0.377964473f) * wi_7 * xj_5 * go_v_17;
        gy_lane_k_10 += scalar_t(0.353553391f) * wi_16 * xj_5 * go_v_68;
        gy_lane_k_10 += scalar_t(-0.577350269f) * wi_16 * xj_4 * go_v_67;
        gy_lane_k_10 += scalar_t(-0.456435465f) * wi_16 * xj_5 * go_v_70;
        gy_lane_k_10 += scalar_t(0.487950036f) * wi_10 * xj_2 * go_v_28;
        gy_lane_k_10 += scalar_t(0.487950036f) * wi_10 * xj_1 * go_v_31;
        gy_lane_k_10 += wi_13 * xj_0 * go_v_44;
        scalar_t gy_sum_k_10 = warp_sum(gy_lane_k_10);
        if (lane == 0) atomicAdd(&grad_y[y_base + 10], gy_sum_k_10);

        scalar_t gy_lane_k_13 = scalar_t(0);
        gy_lane_k_13 += scalar_t(0.414039336f) * wi_7 * xj_6 * go_v_17;
        gy_lane_k_13 += scalar_t(-0.119522861f) * wi_7 * xj_8 * go_v_17;
        gy_lane_k_13 += scalar_t(0.387298335f) * wi_16 * xj_6 * go_v_68;
        gy_lane_k_13 += scalar_t(0.447213595f) * wi_16 * xj_8 * go_v_68;
        gy_lane_k_13 += scalar_t(0.182574186f) * wi_16 * xj_7 * go_v_67;
        gy_lane_k_13 += scalar_t(-0.288675135f) * wi_16 * xj_8 * go_v_70;
        gy_lane_k_13 += scalar_t(0.353553391f) * wi_16 * xj_7 * go_v_69;
        gy_lane_k_13 += scalar_t(-0.15430335f) * wi_10 * xj_1 * go_v_28;
        gy_lane_k_13 += scalar_t(0.6172134f) * wi_10 * xj_2 * go_v_31;
        scalar_t gy_sum_k_13 = warp_sum(gy_lane_k_13);
        if (lane == 0) atomicAdd(&grad_y[y_base + 13], gy_sum_k_13);

        scalar_t gy_lane_k_12 = scalar_t(0);
        gy_lane_k_12 += scalar_t(-0.292770022f) * wi_7 * xj_7 * go_v_17;
        gy_lane_k_12 += scalar_t(0.182574186f) * wi_16 * xj_7 * go_v_68;
        gy_lane_k_12 += scalar_t(0.516397779f) * wi_16 * xj_6 * go_v_67;
        gy_lane_k_12 += scalar_t(-0.577350269f) * wi_16 * xj_8 * go_v_69;
        gy_lane_k_12 += scalar_t(-0.377964473f) * wi_10 * xj_3 * go_v_31;
        gy_lane_k_12 += wi_13 * xj_0 * go_v_46;
        scalar_t gy_sum_k_12 = warp_sum(gy_lane_k_12);
        if (lane == 0) atomicAdd(&grad_y[y_base + 12], gy_sum_k_12);

        scalar_t gy_lane_k_14 = scalar_t(0);
        gy_lane_k_14 += scalar_t(0.377964473f) * wi_7 * xj_7 * go_v_17;
        gy_lane_k_14 += scalar_t(0.353553391f) * wi_16 * xj_7 * go_v_68;
        gy_lane_k_14 += scalar_t(-0.577350269f) * wi_16 * xj_8 * go_v_67;
        gy_lane_k_14 += scalar_t(0.456435465f) * wi_16 * xj_7 * go_v_70;
        gy_lane_k_14 += scalar_t(0.487950036f) * wi_10 * xj_3 * go_v_31;
        gy_lane_k_14 += wi_13 * xj_0 * go_v_48;
        scalar_t gy_sum_k_14 = warp_sum(gy_lane_k_14);
        if (lane == 0) atomicAdd(&grad_y[y_base + 14], gy_sum_k_14);

        scalar_t gy_lane_k_15 = scalar_t(0);
        gy_lane_k_15 += scalar_t(0.46291005f) * wi_7 * xj_8 * go_v_17;
        gy_lane_k_15 += scalar_t(-0.288675135f) * wi_16 * xj_8 * go_v_68;
        gy_lane_k_15 += scalar_t(-0.645497224f) * wi_16 * xj_6 * go_v_70;
        gy_lane_k_15 += scalar_t(0.456435465f) * wi_16 * xj_7 * go_v_69;
        gy_lane_k_15 += scalar_t(-0.597614305f) * wi_10 * xj_1 * go_v_28;
        scalar_t gy_sum_k_15 = warp_sum(gy_lane_k_15);
        if (lane == 0) atomicAdd(&grad_y[y_base + 15], gy_sum_k_15);

        scalar_t gy_lane_k_7 = scalar_t(0);
        gy_lane_k_7 += scalar_t(0.46291005f) * wi_12 * xj_4 * go_v_39;
        gy_lane_k_7 += scalar_t(0.447213595f) * wi_2 * xj_7 * go_v_2;
        gy_lane_k_7 += scalar_t(0.46291005f) * wi_12 * xj_7 * go_v_42;
        gy_lane_k_7 += scalar_t(0.547722558f) * wi_5 * xj_2 * go_v_11;
        gy_lane_k_7 += scalar_t(0.547722558f) * wi_5 * xj_3 * go_v_10;
        gy_lane_k_7 += scalar_t(0.730296743f) * wi_14 * xj_2 * go_v_54;
        gy_lane_k_7 += scalar_t(0.577350269f) * wi_14 * xj_3 * go_v_55;
        scalar_t gy_sum_k_7 = warp_sum(gy_lane_k_7);
        if (lane == 0) atomicAdd(&grad_y[y_base + 7], gy_sum_k_7);

        scalar_t gy_lane_k_6 = scalar_t(0);
        gy_lane_k_6 += scalar_t(0.267261242f) * wi_12 * xj_5 * go_v_39;
        gy_lane_k_6 += scalar_t(0.447213595f) * wi_2 * xj_6 * go_v_2;
        gy_lane_k_6 += scalar_t(-0.534522484f) * wi_12 * xj_8 * go_v_42;
        gy_lane_k_6 += scalar_t(-0.316227766f) * wi_5 * xj_3 * go_v_11;
        gy_lane_k_6 += scalar_t(0.632455532f) * wi_5 * xj_2 * go_v_10;
        gy_lane_k_6 += scalar_t(0.632455532f) * wi_14 * xj_3 * go_v_54;
        gy_lane_k_6 += wi_8 * xj_0 * go_v_20;
        scalar_t gy_sum_k_6 = warp_sum(gy_lane_k_6);
        if (lane == 0) atomicAdd(&grad_y[y_base + 6], gy_sum_k_6);

        scalar_t gy_lane_k_8 = scalar_t(0);
        gy_lane_k_8 += scalar_t(-0.46291005f) * wi_12 * xj_5 * go_v_39;
        gy_lane_k_8 += scalar_t(0.447213595f) * wi_2 * xj_8 * go_v_2;
        gy_lane_k_8 += scalar_t(-0.534522484f) * wi_12 * xj_6 * go_v_42;
        gy_lane_k_8 += scalar_t(0.547722558f) * wi_5 * xj_3 * go_v_11;
        gy_lane_k_8 += scalar_t(-0.182574186f) * wi_14 * xj_3 * go_v_54;
        gy_lane_k_8 += scalar_t(0.577350269f) * wi_14 * xj_2 * go_v_55;
        gy_lane_k_8 += scalar_t(0.707106781f) * wi_14 * xj_3 * go_v_56;
        gy_lane_k_8 += wi_8 * xj_0 * go_v_22;
        scalar_t gy_sum_k_8 = warp_sum(gy_lane_k_8);
        if (lane == 0) atomicAdd(&grad_y[y_base + 8], gy_sum_k_8);

        scalar_t gy_lane_k_5 = scalar_t(0);
        gy_lane_k_5 += scalar_t(0.267261242f) * wi_12 * xj_6 * go_v_39;
        gy_lane_k_5 += scalar_t(-0.46291005f) * wi_12 * xj_8 * go_v_39;
        gy_lane_k_5 += scalar_t(0.447213595f) * wi_2 * xj_5 * go_v_2;
        gy_lane_k_5 += scalar_t(-0.46291005f) * wi_12 * xj_5 * go_v_42;
        gy_lane_k_5 += scalar_t(0.547722558f) * wi_5 * xj_1 * go_v_10;
        gy_lane_k_5 += scalar_t(-0.577350269f) * wi_14 * xj_1 * go_v_55;
        scalar_t gy_sum_k_5 = warp_sum(gy_lane_k_5);
        if (lane == 0) atomicAdd(&grad_y[y_base + 5], gy_sum_k_5);

        scalar_t gy_lane_k_4 = scalar_t(0);
        gy_lane_k_4 += scalar_t(0.46291005f) * wi_12 * xj_7 * go_v_39;
        gy_lane_k_4 += scalar_t(0.447213595f) * wi_2 * xj_4 * go_v_2;
        gy_lane_k_4 += scalar_t(0.547722558f) * wi_5 * xj_1 * go_v_11;
        gy_lane_k_4 += scalar_t(-0.182574186f) * wi_14 * xj_1 * go_v_54;
        gy_lane_k_4 += scalar_t(-0.707106781f) * wi_14 * xj_1 * go_v_56;
        gy_lane_k_4 += wi_8 * xj_0 * go_v_18;
        scalar_t gy_sum_k_4 = warp_sum(gy_lane_k_4);
        if (lane == 0) atomicAdd(&grad_y[y_base + 4], gy_sum_k_4);

        scalar_t gy_lane_k_3 = scalar_t(0);
        gy_lane_k_3 += scalar_t(0.547722558f) * wi_6 * xj_4 * go_v_12;
        gy_lane_k_3 += scalar_t(0.632455532f) * wi_15 * xj_6 * go_v_61;
        gy_lane_k_3 += scalar_t(-0.182574186f) * wi_15 * xj_8 * go_v_61;
        gy_lane_k_3 += scalar_t(0.547722558f) * wi_6 * xj_7 * go_v_13;
        gy_lane_k_3 += scalar_t(0.577350269f) * wi_15 * xj_7 * go_v_62;
        gy_lane_k_3 += scalar_t(0.707106781f) * wi_15 * xj_8 * go_v_63;
        gy_lane_k_3 += scalar_t(0.707106781f) * wi_15 * xj_4 * go_v_57;
        gy_lane_k_3 += scalar_t(-0.40824829f) * wi_9 * xj_3 * go_v_25;
        gy_lane_k_3 += scalar_t(0.707106781f) * wi_9 * xj_1 * go_v_23;
        gy_lane_k_3 += scalar_t(0.707106781f) * wi_9 * xj_2 * go_v_26;
        scalar_t gy_sum_k_3 = warp_sum(gy_lane_k_3);
        if (lane == 0) atomicAdd(&grad_y[y_base + 3], gy_sum_k_3);

        scalar_t gy_lane_k_2 = scalar_t(0);
        gy_lane_k_2 += scalar_t(0.547722558f) * wi_6 * xj_5 * go_v_12;
        gy_lane_k_2 += scalar_t(0.730296743f) * wi_15 * xj_7 * go_v_61;
        gy_lane_k_2 += scalar_t(0.632455532f) * wi_6 * xj_6 * go_v_13;
        gy_lane_k_2 += scalar_t(0.577350269f) * wi_15 * xj_8 * go_v_62;
        gy_lane_k_2 += scalar_t(0.816496581f) * wi_9 * xj_2 * go_v_25;
        gy_lane_k_2 += scalar_t(0.707106781f) * wi_9 * xj_3 * go_v_26;
        gy_lane_k_2 += wi_3 * xj_0 * go_v_4;
        scalar_t gy_sum_k_2 = warp_sum(gy_lane_k_2);
        if (lane == 0) atomicAdd(&grad_y[y_base + 2], gy_sum_k_2);

        scalar_t gy_lane_k_1 = scalar_t(0);
        gy_lane_k_1 += scalar_t(-0.316227766f) * wi_6 * xj_6 * go_v_12;
        gy_lane_k_1 += scalar_t(-0.547722558f) * wi_6 * xj_8 * go_v_12;
        gy_lane_k_1 += scalar_t(-0.182574186f) * wi_15 * xj_4 * go_v_61;
        gy_lane_k_1 += scalar_t(0.547722558f) * wi_6 * xj_5 * go_v_13;
        gy_lane_k_1 += scalar_t(-0.577350269f) * wi_15 * xj_5 * go_v_62;
        gy_lane_k_1 += scalar_t(-0.707106781f) * wi_15 * xj_4 * go_v_63;
        gy_lane_k_1 += scalar_t(0.707106781f) * wi_15 * xj_8 * go_v_57;
        gy_lane_k_1 += scalar_t(-0.40824829f) * wi_9 * xj_1 * go_v_25;
        gy_lane_k_1 += scalar_t(0.707106781f) * wi_9 * xj_3 * go_v_23;
        scalar_t gy_sum_k_1 = warp_sum(gy_lane_k_1);
        if (lane == 0) atomicAdd(&grad_y[y_base + 1], gy_sum_k_1);

        scalar_t gy_lane_k_0 = scalar_t(0);
        gy_lane_k_0 += wi_11 * xj_6 * go_v_35;
        gy_lane_k_0 += wi_11 * xj_4 * go_v_33;
        gy_lane_k_0 += wi_4 * xj_2 * go_v_7;
        gy_lane_k_0 += wi_0 * xj_0 * go_v_0;
        scalar_t gy_sum_k_0 = warp_sum(gy_lane_k_0);
        if (lane == 0) atomicAdd(&grad_y[y_base + 0], gy_sum_k_0);

    }

}

template <typename scalar_t>
void launch_uniform1d_codegen_two_warp_vgroup_path215_u224_bwd(
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
    dim3 block(64);
    dim3 grid(B, (U + 31) / 32);
    uniform1d_codegen_two_warp_vgroup_path215_u224_bwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);
}