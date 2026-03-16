
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_bwd_u32_P22_gradw(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x_all,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ grad_w,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V)
{
    int lane = threadIdx.x;
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;
    int64_t y_base  = (int64_t)b * Ky;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gw_base = ((int64_t)b   * Iw) * 32 + lane;

    // ---- grad_w chunk 0 ----
    scalar_t acc_i_0 = scalar_t(0);
    scalar_t acc_i_1 = scalar_t(0);
    scalar_t acc_i_2 = scalar_t(0);
    scalar_t acc_i_3 = scalar_t(0);
    scalar_t acc_i_4 = scalar_t(0);
    scalar_t acc_i_5 = scalar_t(0);

    // target i = 0
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_0 = grad_out[go_base + ((int64_t)0 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_0 = fma((scalar_t)(1.0) * x_j_0, y_k_0 * go_v_0, acc_i_0);
    }

    // target i = 1
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_1 = grad_out[go_base + ((int64_t)1 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_1 = fma((scalar_t)(1.0) * x_j_1, y_k_0 * go_v_1, acc_i_1);
    }

    // target i = 2
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_2 = grad_out[go_base + ((int64_t)2 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_2 = fma((scalar_t)(1.0) * x_j_2, y_k_0 * go_v_2, acc_i_2);
    }

    // target i = 3
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_3 = grad_out[go_base + ((int64_t)3 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_3 = fma((scalar_t)(1.0) * x_j_3, y_k_0 * go_v_3, acc_i_3);
    }

    // target i = 4
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_4 = fma((scalar_t)(0.57735025882720947) * x_j_4, y_k_1 * go_v_4, acc_i_4);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_4 = fma((scalar_t)(0.57735025882720947) * x_j_6, y_k_2 * go_v_4, acc_i_4);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_4 = fma((scalar_t)(0.57735025882720947) * x_j_8, y_k_3 * go_v_4, acc_i_4);
    }

    // target i = 5
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_5 = fma((scalar_t)(0.57735025882720947) * x_j_5, y_k_1 * go_v_5, acc_i_5);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_5 = fma((scalar_t)(0.57735025882720947) * x_j_7, y_k_2 * go_v_5, acc_i_5);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_5 = fma((scalar_t)(0.57735025882720947) * x_j_9, y_k_3 * go_v_5, acc_i_5);
    }

    grad_w[gw_base + ((int64_t)0 << 5)] += acc_i_0;
    grad_w[gw_base + ((int64_t)1 << 5)] += acc_i_1;
    grad_w[gw_base + ((int64_t)2 << 5)] += acc_i_2;
    grad_w[gw_base + ((int64_t)3 << 5)] += acc_i_3;
    grad_w[gw_base + ((int64_t)4 << 5)] += acc_i_4;
    grad_w[gw_base + ((int64_t)5 << 5)] += acc_i_5;

    // ---- grad_w chunk 1 ----
    scalar_t acc_i_6 = scalar_t(0);
    scalar_t acc_i_7 = scalar_t(0);

    // target i = 6
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_6 = fma((scalar_t)(0.44721359014511108) * x_j_10, y_k_4 * go_v_6, acc_i_6);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_6 = fma((scalar_t)(0.44721359014511108) * x_j_11, y_k_5 * go_v_6, acc_i_6);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_6 = fma((scalar_t)(0.44721359014511108) * x_j_12, y_k_6 * go_v_6, acc_i_6);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_6 = fma((scalar_t)(0.44721359014511108) * x_j_13, y_k_7 * go_v_6, acc_i_6);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_6 = fma((scalar_t)(0.44721359014511108) * x_j_14, y_k_8 * go_v_6, acc_i_6);
    }

    // target i = 7
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_15, y_k_9 * go_v_7, acc_i_7);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_16, y_k_10 * go_v_7, acc_i_7);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_17, y_k_11 * go_v_7, acc_i_7);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_18, y_k_12 * go_v_7, acc_i_7);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_19, y_k_13 * go_v_7, acc_i_7);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_20, y_k_14 * go_v_7, acc_i_7);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_7 = fma((scalar_t)(0.37796446681022644) * x_j_21, y_k_15 * go_v_7, acc_i_7);
    }

    grad_w[gw_base + ((int64_t)6 << 5)] += acc_i_6;
    grad_w[gw_base + ((int64_t)7 << 5)] += acc_i_7;

}
template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_bwd_u32_P22_gradx(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ grad_x,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V)
{
    int lane = threadIdx.x;
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;
    int64_t y_base  = (int64_t)b * Ky;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gx_base = ((int64_t)src * Ix) * 32 + lane;

    // ---- grad_x chunk 0 ----
    scalar_t acc_j_0 = scalar_t(0);
    scalar_t acc_j_1 = scalar_t(0);
    scalar_t acc_j_2 = scalar_t(0);
    scalar_t acc_j_3 = scalar_t(0);
    scalar_t acc_j_4 = scalar_t(0);
    scalar_t acc_j_6 = scalar_t(0);
    scalar_t acc_j_8 = scalar_t(0);
    scalar_t acc_j_5 = scalar_t(0);

    // target j = 0
    {
        scalar_t w_i_0 = w[w_base + ((int64_t)0 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_0 = grad_out[go_base + ((int64_t)0 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_0, y_k_0 * go_v_0, acc_j_0);
    }

    // target j = 1
    {
        scalar_t w_i_1 = w[w_base + ((int64_t)1 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_1 = grad_out[go_base + ((int64_t)1 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_1, y_k_0 * go_v_1, acc_j_1);
    }

    // target j = 2
    {
        scalar_t w_i_2 = w[w_base + ((int64_t)2 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_2 = grad_out[go_base + ((int64_t)2 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_2, y_k_0 * go_v_2, acc_j_2);
    }

    // target j = 3
    {
        scalar_t w_i_3 = w[w_base + ((int64_t)3 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_3 = grad_out[go_base + ((int64_t)3 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_3, y_k_0 * go_v_3, acc_j_3);
    }

    // target j = 4
    {
        scalar_t w_i_4 = w[w_base + ((int64_t)4 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        acc_j_4 = fma((scalar_t)(0.57735025882720947) * w_i_4, y_k_1 * go_v_4, acc_j_4);
    }

    // target j = 6
    {
        scalar_t w_i_4 = w[w_base + ((int64_t)4 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        acc_j_6 = fma((scalar_t)(0.57735025882720947) * w_i_4, y_k_2 * go_v_4, acc_j_6);
    }

    // target j = 8
    {
        scalar_t w_i_4 = w[w_base + ((int64_t)4 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        acc_j_8 = fma((scalar_t)(0.57735025882720947) * w_i_4, y_k_3 * go_v_4, acc_j_8);
    }

    // target j = 5
    {
        scalar_t w_i_5 = w[w_base + ((int64_t)5 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        acc_j_5 = fma((scalar_t)(0.57735025882720947) * w_i_5, y_k_1 * go_v_5, acc_j_5);
    }

    grad_x[gx_base + ((int64_t)0 << 5)] += acc_j_0;
    grad_x[gx_base + ((int64_t)1 << 5)] += acc_j_1;
    grad_x[gx_base + ((int64_t)2 << 5)] += acc_j_2;
    grad_x[gx_base + ((int64_t)3 << 5)] += acc_j_3;
    grad_x[gx_base + ((int64_t)4 << 5)] += acc_j_4;
    grad_x[gx_base + ((int64_t)6 << 5)] += acc_j_6;
    grad_x[gx_base + ((int64_t)8 << 5)] += acc_j_8;
    grad_x[gx_base + ((int64_t)5 << 5)] += acc_j_5;

    // ---- grad_x chunk 1 ----
    scalar_t acc_j_7 = scalar_t(0);
    scalar_t acc_j_9 = scalar_t(0);
    scalar_t acc_j_10 = scalar_t(0);
    scalar_t acc_j_11 = scalar_t(0);
    scalar_t acc_j_12 = scalar_t(0);
    scalar_t acc_j_13 = scalar_t(0);
    scalar_t acc_j_14 = scalar_t(0);
    scalar_t acc_j_15 = scalar_t(0);

    // target j = 7
    {
        scalar_t w_i_5 = w[w_base + ((int64_t)5 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        acc_j_7 = fma((scalar_t)(0.57735025882720947) * w_i_5, y_k_2 * go_v_5, acc_j_7);
    }

    // target j = 9
    {
        scalar_t w_i_5 = w[w_base + ((int64_t)5 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        acc_j_9 = fma((scalar_t)(0.57735025882720947) * w_i_5, y_k_3 * go_v_5, acc_j_9);
    }

    // target j = 10
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_10 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_4 * go_v_6, acc_j_10);
    }

    // target j = 11
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_11 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_5 * go_v_6, acc_j_11);
    }

    // target j = 12
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_12 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_6 * go_v_6, acc_j_12);
    }

    // target j = 13
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_13 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_7 * go_v_6, acc_j_13);
    }

    // target j = 14
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_14 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_8 * go_v_6, acc_j_14);
    }

    // target j = 15
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_15 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_9 * go_v_7, acc_j_15);
    }

    grad_x[gx_base + ((int64_t)7 << 5)] += acc_j_7;
    grad_x[gx_base + ((int64_t)9 << 5)] += acc_j_9;
    grad_x[gx_base + ((int64_t)10 << 5)] += acc_j_10;
    grad_x[gx_base + ((int64_t)11 << 5)] += acc_j_11;
    grad_x[gx_base + ((int64_t)12 << 5)] += acc_j_12;
    grad_x[gx_base + ((int64_t)13 << 5)] += acc_j_13;
    grad_x[gx_base + ((int64_t)14 << 5)] += acc_j_14;
    grad_x[gx_base + ((int64_t)15 << 5)] += acc_j_15;

    // ---- grad_x chunk 2 ----
    scalar_t acc_j_16 = scalar_t(0);
    scalar_t acc_j_17 = scalar_t(0);
    scalar_t acc_j_18 = scalar_t(0);
    scalar_t acc_j_19 = scalar_t(0);
    scalar_t acc_j_20 = scalar_t(0);
    scalar_t acc_j_21 = scalar_t(0);

    // target j = 16
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_16 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_10 * go_v_7, acc_j_16);
    }

    // target j = 17
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_17 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_11 * go_v_7, acc_j_17);
    }

    // target j = 18
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_18 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_12 * go_v_7, acc_j_18);
    }

    // target j = 19
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_19 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_13 * go_v_7, acc_j_19);
    }

    // target j = 20
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_20 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_14 * go_v_7, acc_j_20);
    }

    // target j = 21
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_21 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_15 * go_v_7, acc_j_21);
    }

    grad_x[gx_base + ((int64_t)16 << 5)] += acc_j_16;
    grad_x[gx_base + ((int64_t)17 << 5)] += acc_j_17;
    grad_x[gx_base + ((int64_t)18 << 5)] += acc_j_18;
    grad_x[gx_base + ((int64_t)19 << 5)] += acc_j_19;
    grad_x[gx_base + ((int64_t)20 << 5)] += acc_j_20;
    grad_x[gx_base + ((int64_t)21 << 5)] += acc_j_21;

}
template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_bwd_u32_P22_grady(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x_all,
    scalar_t* __restrict__ grad_y,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V)
{
    int lane = threadIdx.x;
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;
    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gy_base = (int64_t)b * Ky;

    // ---- grad_y chunk 0 ----
    scalar_t local_k_0 = scalar_t(0);
    {
        scalar_t w_i_0  = w[w_base + ((int64_t)0 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_0 = grad_out[go_base + ((int64_t)0 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_0, x_j_0 * go_v_0, local_k_0);
    }
    {
        scalar_t w_i_1  = w[w_base + ((int64_t)1 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_1 = grad_out[go_base + ((int64_t)1 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_1, x_j_1 * go_v_1, local_k_0);
    }
    {
        scalar_t w_i_2  = w[w_base + ((int64_t)2 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_2 = grad_out[go_base + ((int64_t)2 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_2, x_j_2 * go_v_2, local_k_0);
    }
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_3 = grad_out[go_base + ((int64_t)3 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_3, x_j_3 * go_v_3, local_k_0);
    }
    scalar_t sum_k_0 = warp_sum(local_k_0);
    if (lane == 0) grad_y[gy_base + 0] += sum_k_0;

    scalar_t local_k_1 = scalar_t(0);
    {
        scalar_t w_i_4  = w[w_base + ((int64_t)4 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        local_k_1 = fma((scalar_t)(0.57735025882720947) * w_i_4, x_j_4 * go_v_4, local_k_1);
    }
    {
        scalar_t w_i_5  = w[w_base + ((int64_t)5 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        local_k_1 = fma((scalar_t)(0.57735025882720947) * w_i_5, x_j_5 * go_v_5, local_k_1);
    }
    scalar_t sum_k_1 = warp_sum(local_k_1);
    if (lane == 0) grad_y[gy_base + 1] += sum_k_1;

    scalar_t local_k_2 = scalar_t(0);
    {
        scalar_t w_i_4  = w[w_base + ((int64_t)4 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        local_k_2 = fma((scalar_t)(0.57735025882720947) * w_i_4, x_j_6 * go_v_4, local_k_2);
    }
    {
        scalar_t w_i_5  = w[w_base + ((int64_t)5 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        local_k_2 = fma((scalar_t)(0.57735025882720947) * w_i_5, x_j_7 * go_v_5, local_k_2);
    }
    scalar_t sum_k_2 = warp_sum(local_k_2);
    if (lane == 0) grad_y[gy_base + 2] += sum_k_2;

    scalar_t local_k_3 = scalar_t(0);
    {
        scalar_t w_i_4  = w[w_base + ((int64_t)4 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        local_k_3 = fma((scalar_t)(0.57735025882720947) * w_i_4, x_j_8 * go_v_4, local_k_3);
    }
    {
        scalar_t w_i_5  = w[w_base + ((int64_t)5 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        local_k_3 = fma((scalar_t)(0.57735025882720947) * w_i_5, x_j_9 * go_v_5, local_k_3);
    }
    scalar_t sum_k_3 = warp_sum(local_k_3);
    if (lane == 0) grad_y[gy_base + 3] += sum_k_3;

    scalar_t local_k_4 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_4 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_10 * go_v_6, local_k_4);
    }
    scalar_t sum_k_4 = warp_sum(local_k_4);
    if (lane == 0) grad_y[gy_base + 4] += sum_k_4;

    scalar_t local_k_5 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_5 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_11 * go_v_6, local_k_5);
    }
    scalar_t sum_k_5 = warp_sum(local_k_5);
    if (lane == 0) grad_y[gy_base + 5] += sum_k_5;

    // ---- grad_y chunk 1 ----
    scalar_t local_k_6 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_6 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_12 * go_v_6, local_k_6);
    }
    scalar_t sum_k_6 = warp_sum(local_k_6);
    if (lane == 0) grad_y[gy_base + 6] += sum_k_6;

    scalar_t local_k_7 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_7 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_13 * go_v_6, local_k_7);
    }
    scalar_t sum_k_7 = warp_sum(local_k_7);
    if (lane == 0) grad_y[gy_base + 7] += sum_k_7;

    scalar_t local_k_8 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_8 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_14 * go_v_6, local_k_8);
    }
    scalar_t sum_k_8 = warp_sum(local_k_8);
    if (lane == 0) grad_y[gy_base + 8] += sum_k_8;

    scalar_t local_k_9 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_9 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_15 * go_v_7, local_k_9);
    }
    scalar_t sum_k_9 = warp_sum(local_k_9);
    if (lane == 0) grad_y[gy_base + 9] += sum_k_9;

    scalar_t local_k_10 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_10 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_16 * go_v_7, local_k_10);
    }
    scalar_t sum_k_10 = warp_sum(local_k_10);
    if (lane == 0) grad_y[gy_base + 10] += sum_k_10;

    scalar_t local_k_11 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_11 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_17 * go_v_7, local_k_11);
    }
    scalar_t sum_k_11 = warp_sum(local_k_11);
    if (lane == 0) grad_y[gy_base + 11] += sum_k_11;

    // ---- grad_y chunk 2 ----
    scalar_t local_k_12 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_12 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_18 * go_v_7, local_k_12);
    }
    scalar_t sum_k_12 = warp_sum(local_k_12);
    if (lane == 0) grad_y[gy_base + 12] += sum_k_12;

    scalar_t local_k_13 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_13 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_19 * go_v_7, local_k_13);
    }
    scalar_t sum_k_13 = warp_sum(local_k_13);
    if (lane == 0) grad_y[gy_base + 13] += sum_k_13;

    scalar_t local_k_14 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_14 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_20 * go_v_7, local_k_14);
    }
    scalar_t sum_k_14 = warp_sum(local_k_14);
    if (lane == 0) grad_y[gy_base + 14] += sum_k_14;

    scalar_t local_k_15 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_15 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_21 * go_v_7, local_k_15);
    }
    scalar_t sum_k_15 = warp_sum(local_k_15);
    if (lane == 0) grad_y[gy_base + 15] += sum_k_15;

}

std::vector<torch::Tensor> uniform1d_split_bwd_u32_P22(
    torch::Tensor grad_out,
    torch::Tensor w,
    torch::Tensor x_all,
    torch::Tensor y,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    torch::Tensor b_list,
    int64_t Iw,
    int64_t Ix,
    int64_t Ky,
    int64_t V)
{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x_all.is_cuda(), "x_all must be CUDA");
    TORCH_CHECK(y.is_cuda(), "y must be CUDA");

    auto B = b_list.numel() > 0 ? (int)b_list.numel() : (int)w.size(0);

    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x_all);
    auto grad_y = torch::zeros_like(y);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(32);
    dim3 grid(B);

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_split_bwd_u32_P22", [&] {
        uniform1d_split_bwd_u32_P22_gradw<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        uniform1d_split_bwd_u32_P22_gradx<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        uniform1d_split_bwd_u32_P22_grady<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);
    });

    return {grad_w, grad_x, grad_y};
}

TORCH_LIBRARY(uniform1d_split_bwd_u32_p22, m)
{
    m.def("run", &uniform1d_split_bwd_u32_P22);
}