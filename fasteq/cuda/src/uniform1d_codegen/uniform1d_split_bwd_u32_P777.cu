
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_bwd_u32_P777_gradw(
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
    scalar_t acc_i_8 = scalar_t(0);

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

    // target i = 8
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_8 = grad_out[go_base + ((int64_t)8 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_8 = fma((scalar_t)(1.0) * x_j_0, y_k_1 * go_v_8, acc_i_8);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_12 = grad_out[go_base + ((int64_t)12 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_8 = fma((scalar_t)(1.0) * x_j_0, y_k_2 * go_v_12, acc_i_8);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_16 = grad_out[go_base + ((int64_t)16 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_8 = fma((scalar_t)(1.0) * x_j_0, y_k_3 * go_v_16, acc_i_8);
    }

    grad_w[gw_base + ((int64_t)6 << 5)] += acc_i_6;
    grad_w[gw_base + ((int64_t)7 << 5)] += acc_i_7;
    grad_w[gw_base + ((int64_t)8 << 5)] += acc_i_8;

    // ---- grad_w chunk 2 ----
    scalar_t acc_i_9 = scalar_t(0);
    scalar_t acc_i_10 = scalar_t(0);
    scalar_t acc_i_11 = scalar_t(0);
    scalar_t acc_i_12 = scalar_t(0);
    scalar_t acc_i_13 = scalar_t(0);

    // target i = 9
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_9 = grad_out[go_base + ((int64_t)9 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_9 = fma((scalar_t)(1.0) * x_j_1, y_k_1 * go_v_9, acc_i_9);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_13 = grad_out[go_base + ((int64_t)13 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_9 = fma((scalar_t)(1.0) * x_j_1, y_k_2 * go_v_13, acc_i_9);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_17 = grad_out[go_base + ((int64_t)17 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_9 = fma((scalar_t)(1.0) * x_j_1, y_k_3 * go_v_17, acc_i_9);
    }

    // target i = 10
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_10 = grad_out[go_base + ((int64_t)10 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_10 = fma((scalar_t)(1.0) * x_j_2, y_k_1 * go_v_10, acc_i_10);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_14 = grad_out[go_base + ((int64_t)14 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_10 = fma((scalar_t)(1.0) * x_j_2, y_k_2 * go_v_14, acc_i_10);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_18 = grad_out[go_base + ((int64_t)18 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_10 = fma((scalar_t)(1.0) * x_j_2, y_k_3 * go_v_18, acc_i_10);
    }

    // target i = 11
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_11 = grad_out[go_base + ((int64_t)11 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_11 = fma((scalar_t)(1.0) * x_j_3, y_k_1 * go_v_11, acc_i_11);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_15 = grad_out[go_base + ((int64_t)15 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_11 = fma((scalar_t)(1.0) * x_j_3, y_k_2 * go_v_15, acc_i_11);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_19 = grad_out[go_base + ((int64_t)19 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_11 = fma((scalar_t)(1.0) * x_j_3, y_k_3 * go_v_19, acc_i_11);
    }

    // target i = 12
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_20 = grad_out[go_base + ((int64_t)20 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_12 = fma((scalar_t)(1.0) * x_j_4, y_k_0 * go_v_20, acc_i_12);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_22 = grad_out[go_base + ((int64_t)22 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_12 = fma((scalar_t)(1.0) * x_j_6, y_k_0 * go_v_22, acc_i_12);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_24 = grad_out[go_base + ((int64_t)24 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_12 = fma((scalar_t)(1.0) * x_j_8, y_k_0 * go_v_24, acc_i_12);
    }

    // target i = 13
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_21 = grad_out[go_base + ((int64_t)21 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_13 = fma((scalar_t)(1.0) * x_j_5, y_k_0 * go_v_21, acc_i_13);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_23 = grad_out[go_base + ((int64_t)23 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_13 = fma((scalar_t)(1.0) * x_j_7, y_k_0 * go_v_23, acc_i_13);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_25 = grad_out[go_base + ((int64_t)25 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_13 = fma((scalar_t)(1.0) * x_j_9, y_k_0 * go_v_25, acc_i_13);
    }

    grad_w[gw_base + ((int64_t)9 << 5)] += acc_i_9;
    grad_w[gw_base + ((int64_t)10 << 5)] += acc_i_10;
    grad_w[gw_base + ((int64_t)11 << 5)] += acc_i_11;
    grad_w[gw_base + ((int64_t)12 << 5)] += acc_i_12;
    grad_w[gw_base + ((int64_t)13 << 5)] += acc_i_13;

    // ---- grad_w chunk 3 ----
    scalar_t acc_i_14 = scalar_t(0);
    scalar_t acc_i_15 = scalar_t(0);
    scalar_t acc_i_16 = scalar_t(0);

    // target i = 14
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_14 = fma((scalar_t)(-0.31622776389122009) * x_j_4, y_k_6 * go_v_26, acc_i_14);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_14 = fma((scalar_t)(-0.54772257804870605) * x_j_4, y_k_8 * go_v_26, acc_i_14);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_6, y_k_5 * go_v_26, acc_i_14);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_8, y_k_4 * go_v_26, acc_i_14);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_4, y_k_5 * go_v_28, acc_i_14);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_14 = fma((scalar_t)(0.63245552778244019) * x_j_6, y_k_6 * go_v_28, acc_i_14);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_8, y_k_7 * go_v_28, acc_i_14);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_4, y_k_4 * go_v_30, acc_i_14);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_6, y_k_7 * go_v_30, acc_i_14);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_14 = fma((scalar_t)(-0.31622776389122009) * x_j_8, y_k_6 * go_v_30, acc_i_14);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_14 = fma((scalar_t)(0.54772257804870605) * x_j_8, y_k_8 * go_v_30, acc_i_14);
    }

    // target i = 15
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_15 = fma((scalar_t)(-0.31622776389122009) * x_j_5, y_k_6 * go_v_27, acc_i_15);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_15 = fma((scalar_t)(-0.54772257804870605) * x_j_5, y_k_8 * go_v_27, acc_i_15);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_7, y_k_5 * go_v_27, acc_i_15);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_9, y_k_4 * go_v_27, acc_i_15);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_5, y_k_5 * go_v_29, acc_i_15);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_15 = fma((scalar_t)(0.63245552778244019) * x_j_7, y_k_6 * go_v_29, acc_i_15);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_9, y_k_7 * go_v_29, acc_i_15);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_5, y_k_4 * go_v_31, acc_i_15);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_7, y_k_7 * go_v_31, acc_i_15);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_15 = fma((scalar_t)(-0.31622776389122009) * x_j_9, y_k_6 * go_v_31, acc_i_15);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_15 = fma((scalar_t)(0.54772257804870605) * x_j_9, y_k_8 * go_v_31, acc_i_15);
    }

    // target i = 16
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_10, y_k_3 * go_v_32, acc_i_16);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_11, y_k_2 * go_v_32, acc_i_16);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_16 = fma((scalar_t)(-0.31622776389122009) * x_j_12, y_k_1 * go_v_32, acc_i_16);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_16 = fma((scalar_t)(-0.54772257804870605) * x_j_14, y_k_1 * go_v_32, acc_i_16);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_11, y_k_1 * go_v_33, acc_i_16);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_16 = fma((scalar_t)(0.63245552778244019) * x_j_12, y_k_2 * go_v_33, acc_i_16);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_13, y_k_3 * go_v_33, acc_i_16);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_10, y_k_1 * go_v_34, acc_i_16);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_16 = fma((scalar_t)(-0.31622776389122009) * x_j_12, y_k_3 * go_v_34, acc_i_16);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_14, y_k_3 * go_v_34, acc_i_16);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_16 = fma((scalar_t)(0.54772257804870605) * x_j_13, y_k_2 * go_v_34, acc_i_16);
    }

    grad_w[gw_base + ((int64_t)14 << 5)] += acc_i_14;
    grad_w[gw_base + ((int64_t)15 << 5)] += acc_i_15;
    grad_w[gw_base + ((int64_t)16 << 5)] += acc_i_16;

    // ---- grad_w chunk 4 ----
    scalar_t acc_i_17 = scalar_t(0);
    scalar_t acc_i_18 = scalar_t(0);

    // target i = 17
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_17 = fma((scalar_t)(-0.11952286213636398) * x_j_10, y_k_13 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_17 = fma((scalar_t)(-0.46291005611419678) * x_j_10, y_k_15 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_17 = fma((scalar_t)(-0.29276999831199646) * x_j_11, y_k_12 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_17 = fma((scalar_t)(-0.37796446681022644) * x_j_11, y_k_14 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_17 = fma((scalar_t)(0.41403934359550476) * x_j_12, y_k_11 * go_v_35, acc_i_17);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_17 = fma((scalar_t)(0.11952286213636398) * x_j_14, y_k_11 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_17 = fma((scalar_t)(0.37796446681022644) * x_j_13, y_k_10 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_17 = fma((scalar_t)(0.46291005611419678) * x_j_14, y_k_9 * go_v_35, acc_i_17);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_17 = fma((scalar_t)(0.37796446681022644) * x_j_10, y_k_10 * go_v_36, acc_i_17);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_17 = fma((scalar_t)(0.47809144854545593) * x_j_11, y_k_11 * go_v_36, acc_i_17);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_17 = fma((scalar_t)(0.50709253549575806) * x_j_12, y_k_12 * go_v_36, acc_i_17);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_17 = fma((scalar_t)(0.47809144854545593) * x_j_13, y_k_13 * go_v_36, acc_i_17);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_17 = fma((scalar_t)(0.37796446681022644) * x_j_14, y_k_14 * go_v_36, acc_i_17);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_17 = fma((scalar_t)(0.46291005611419678) * x_j_10, y_k_9 * go_v_37, acc_i_17);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_17 = fma((scalar_t)(-0.11952286213636398) * x_j_10, y_k_11 * go_v_37, acc_i_17);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_17 = fma((scalar_t)(0.37796446681022644) * x_j_11, y_k_10 * go_v_37, acc_i_17);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_17 = fma((scalar_t)(0.41403934359550476) * x_j_12, y_k_13 * go_v_37, acc_i_17);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_17 = fma((scalar_t)(-0.11952286213636398) * x_j_14, y_k_13 * go_v_37, acc_i_17);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_17 = fma((scalar_t)(-0.29276999831199646) * x_j_13, y_k_12 * go_v_37, acc_i_17);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_17 = fma((scalar_t)(0.37796446681022644) * x_j_13, y_k_14 * go_v_37, acc_i_17);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_17 = fma((scalar_t)(0.46291005611419678) * x_j_14, y_k_15 * go_v_37, acc_i_17);
    }

    // target i = 18
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_18 = fma((scalar_t)(0.46291005611419678) * x_j_15, y_k_8 * go_v_38, acc_i_18);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_18 = fma((scalar_t)(0.11952286213636398) * x_j_17, y_k_8 * go_v_38, acc_i_18);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_18 = fma((scalar_t)(0.37796446681022644) * x_j_16, y_k_7 * go_v_38, acc_i_18);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_18 = fma((scalar_t)(0.41403934359550476) * x_j_17, y_k_6 * go_v_38, acc_i_18);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_18 = fma((scalar_t)(-0.29276999831199646) * x_j_18, y_k_5 * go_v_38, acc_i_18);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_18 = fma((scalar_t)(-0.37796446681022644) * x_j_20, y_k_5 * go_v_38, acc_i_18);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_18 = fma((scalar_t)(-0.11952286213636398) * x_j_19, y_k_4 * go_v_38, acc_i_18);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_18 = fma((scalar_t)(-0.46291005611419678) * x_j_21, y_k_4 * go_v_38, acc_i_18);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_18 = fma((scalar_t)(0.37796446681022644) * x_j_16, y_k_4 * go_v_39, acc_i_18);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_18 = fma((scalar_t)(0.47809144854545593) * x_j_17, y_k_5 * go_v_39, acc_i_18);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_18 = fma((scalar_t)(0.50709253549575806) * x_j_18, y_k_6 * go_v_39, acc_i_18);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_18 = fma((scalar_t)(0.47809144854545593) * x_j_19, y_k_7 * go_v_39, acc_i_18);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_18 = fma((scalar_t)(0.37796446681022644) * x_j_20, y_k_8 * go_v_39, acc_i_18);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_18 = fma((scalar_t)(0.46291005611419678) * x_j_15, y_k_4 * go_v_40, acc_i_18);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_18 = fma((scalar_t)(-0.11952286213636398) * x_j_17, y_k_4 * go_v_40, acc_i_18);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_18 = fma((scalar_t)(0.37796446681022644) * x_j_16, y_k_5 * go_v_40, acc_i_18);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_18 = fma((scalar_t)(-0.29276999831199646) * x_j_18, y_k_7 * go_v_40, acc_i_18);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_18 = fma((scalar_t)(0.37796446681022644) * x_j_20, y_k_7 * go_v_40, acc_i_18);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_18 = fma((scalar_t)(0.41403934359550476) * x_j_19, y_k_6 * go_v_40, acc_i_18);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_18 = fma((scalar_t)(-0.11952286213636398) * x_j_19, y_k_8 * go_v_40, acc_i_18);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_18 = fma((scalar_t)(0.46291005611419678) * x_j_21, y_k_8 * go_v_40, acc_i_18);
    }

    grad_w[gw_base + ((int64_t)17 << 5)] += acc_i_17;
    grad_w[gw_base + ((int64_t)18 << 5)] += acc_i_18;

    // ---- grad_w chunk 5 ----
    scalar_t acc_i_19 = scalar_t(0);
    scalar_t acc_i_20 = scalar_t(0);
    scalar_t acc_i_21 = scalar_t(0);

    // target i = 19
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_41 = grad_out[go_base + ((int64_t)41 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_19 = fma((scalar_t)(0.70710676908493042) * x_j_6, y_k_3 * go_v_41, acc_i_19);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_41 = grad_out[go_base + ((int64_t)41 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_19 = fma((scalar_t)(-0.70710676908493042) * x_j_8, y_k_2 * go_v_41, acc_i_19);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_43 = grad_out[go_base + ((int64_t)43 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_19 = fma((scalar_t)(-0.70710676908493042) * x_j_4, y_k_3 * go_v_43, acc_i_19);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_43 = grad_out[go_base + ((int64_t)43 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_19 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_1 * go_v_43, acc_i_19);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_45 = grad_out[go_base + ((int64_t)45 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_19 = fma((scalar_t)(0.70710676908493042) * x_j_4, y_k_2 * go_v_45, acc_i_19);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_45 = grad_out[go_base + ((int64_t)45 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_19 = fma((scalar_t)(-0.70710676908493042) * x_j_6, y_k_1 * go_v_45, acc_i_19);
    }

    // target i = 20
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_42 = grad_out[go_base + ((int64_t)42 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_20 = fma((scalar_t)(0.70710676908493042) * x_j_7, y_k_3 * go_v_42, acc_i_20);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_42 = grad_out[go_base + ((int64_t)42 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_20 = fma((scalar_t)(-0.70710676908493042) * x_j_9, y_k_2 * go_v_42, acc_i_20);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_44 = grad_out[go_base + ((int64_t)44 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_20 = fma((scalar_t)(-0.70710676908493042) * x_j_5, y_k_3 * go_v_44, acc_i_20);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_44 = grad_out[go_base + ((int64_t)44 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_20 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_1 * go_v_44, acc_i_20);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_46 = grad_out[go_base + ((int64_t)46 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_20 = fma((scalar_t)(0.70710676908493042) * x_j_5, y_k_2 * go_v_46, acc_i_20);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_46 = grad_out[go_base + ((int64_t)46 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_20 = fma((scalar_t)(-0.70710676908493042) * x_j_7, y_k_1 * go_v_46, acc_i_20);
    }

    // target i = 21
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_21 = fma((scalar_t)(-0.31622776389122009) * x_j_10, y_k_5 * go_v_47, acc_i_21);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_21 = fma((scalar_t)(0.31622776389122009) * x_j_11, y_k_4 * go_v_47, acc_i_21);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_21 = fma((scalar_t)(0.54772257804870605) * x_j_12, y_k_7 * go_v_47, acc_i_21);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_21 = fma((scalar_t)(-0.31622776389122009) * x_j_14, y_k_7 * go_v_47, acc_i_21);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_21 = fma((scalar_t)(-0.54772257804870605) * x_j_13, y_k_6 * go_v_47, acc_i_21);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_21 = fma((scalar_t)(0.31622776389122009) * x_j_13, y_k_8 * go_v_47, acc_i_21);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_21 = fma((scalar_t)(-0.63245552778244019) * x_j_10, y_k_8 * go_v_48, acc_i_21);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_21 = fma((scalar_t)(-0.31622776389122009) * x_j_11, y_k_7 * go_v_48, acc_i_21);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_21 = fma((scalar_t)(0.31622776389122009) * x_j_13, y_k_5 * go_v_48, acc_i_21);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_21 = fma((scalar_t)(0.63245552778244019) * x_j_14, y_k_4 * go_v_48, acc_i_21);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_21 = fma((scalar_t)(0.31622776389122009) * x_j_10, y_k_7 * go_v_49, acc_i_21);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_21 = fma((scalar_t)(0.54772257804870605) * x_j_11, y_k_6 * go_v_49, acc_i_21);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_21 = fma((scalar_t)(0.31622776389122009) * x_j_11, y_k_8 * go_v_49, acc_i_21);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_21 = fma((scalar_t)(-0.54772257804870605) * x_j_12, y_k_5 * go_v_49, acc_i_21);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_21 = fma((scalar_t)(-0.31622776389122009) * x_j_14, y_k_5 * go_v_49, acc_i_21);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_21 = fma((scalar_t)(-0.31622776389122009) * x_j_13, y_k_4 * go_v_49, acc_i_21);
    }

    grad_w[gw_base + ((int64_t)19 << 5)] += acc_i_19;
    grad_w[gw_base + ((int64_t)20 << 5)] += acc_i_20;
    grad_w[gw_base + ((int64_t)21 << 5)] += acc_i_21;

    // ---- grad_w chunk 6 ----
    scalar_t acc_i_22 = scalar_t(0);
    scalar_t acc_i_23 = scalar_t(0);

    // target i = 22
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_22 = fma((scalar_t)(-0.23145502805709839) * x_j_15, y_k_10 * go_v_50, acc_i_22);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_22 = fma((scalar_t)(0.29880714416503906) * x_j_17, y_k_10 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_22 = fma((scalar_t)(0.23145502805709839) * x_j_16, y_k_9 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_22 = fma((scalar_t)(-0.29880714416503906) * x_j_16, y_k_11 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_22 = fma((scalar_t)(0.46291005611419678) * x_j_18, y_k_13 * go_v_50, acc_i_22);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_22 = fma((scalar_t)(-0.29880714416503906) * x_j_20, y_k_13 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_22 = fma((scalar_t)(-0.46291005611419678) * x_j_19, y_k_12 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_22 = fma((scalar_t)(0.29880714416503906) * x_j_19, y_k_14 * go_v_50, acc_i_22);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_22 = fma((scalar_t)(-0.23145502805709839) * x_j_21, y_k_14 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_22 = fma((scalar_t)(0.23145502805709839) * x_j_20, y_k_15 * go_v_50, acc_i_22);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_22 = fma((scalar_t)(-0.56694668531417847) * x_j_15, y_k_15 * go_v_51, acc_i_22);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_22 = fma((scalar_t)(-0.37796449661254883) * x_j_16, y_k_14 * go_v_51, acc_i_22);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_22 = fma((scalar_t)(-0.18898224830627441) * x_j_17, y_k_13 * go_v_51, acc_i_22);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_22 = fma((scalar_t)(0.18898224830627441) * x_j_19, y_k_11 * go_v_51, acc_i_22);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_22 = fma((scalar_t)(0.37796449661254883) * x_j_20, y_k_10 * go_v_51, acc_i_22);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_22 = fma((scalar_t)(0.56694668531417847) * x_j_21, y_k_9 * go_v_51, acc_i_22);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_22 = fma((scalar_t)(0.23145502805709839) * x_j_15, y_k_14 * go_v_52, acc_i_22);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_22 = fma((scalar_t)(0.29880714416503906) * x_j_17, y_k_14 * go_v_52, acc_i_22);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_22 = fma((scalar_t)(0.29880714416503906) * x_j_16, y_k_13 * go_v_52, acc_i_22);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_22 = fma((scalar_t)(0.23145502805709839) * x_j_16, y_k_15 * go_v_52, acc_i_22);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_22 = fma((scalar_t)(0.46291005611419678) * x_j_17, y_k_12 * go_v_52, acc_i_22);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_22 = fma((scalar_t)(-0.46291005611419678) * x_j_18, y_k_11 * go_v_52, acc_i_22);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_22 = fma((scalar_t)(-0.29880714416503906) * x_j_20, y_k_11 * go_v_52, acc_i_22);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_22 = fma((scalar_t)(-0.29880714416503906) * x_j_19, y_k_10 * go_v_52, acc_i_22);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_22 = fma((scalar_t)(-0.23145502805709839) * x_j_21, y_k_10 * go_v_52, acc_i_22);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_22 = fma((scalar_t)(-0.23145502805709839) * x_j_20, y_k_9 * go_v_52, acc_i_22);
    }

    // target i = 23
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_23 = fma((scalar_t)(0.40824827551841736) * x_j_4, y_k_5 * go_v_53, acc_i_23);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_23 = fma((scalar_t)(0.81649655103683472) * x_j_6, y_k_8 * go_v_53, acc_i_23);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_23 = fma((scalar_t)(-0.40824827551841736) * x_j_8, y_k_7 * go_v_53, acc_i_23);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_23 = fma((scalar_t)(-0.40824827551841736) * x_j_4, y_k_4 * go_v_55, acc_i_23);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_23 = fma((scalar_t)(0.40824827551841736) * x_j_6, y_k_7 * go_v_55, acc_i_23);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_23 = fma((scalar_t)(-0.70710676908493042) * x_j_8, y_k_6 * go_v_55, acc_i_23);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_23 = fma((scalar_t)(-0.40824827551841736) * x_j_8, y_k_8 * go_v_55, acc_i_23);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_57 = grad_out[go_base + ((int64_t)57 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_23 = fma((scalar_t)(-0.70710676908493042) * x_j_4, y_k_7 * go_v_57, acc_i_23);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_57 = grad_out[go_base + ((int64_t)57 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_23 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_5 * go_v_57, acc_i_23);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_23 = fma((scalar_t)(0.70710676908493042) * x_j_4, y_k_6 * go_v_59, acc_i_23);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_23 = fma((scalar_t)(-0.40824827551841736) * x_j_4, y_k_8 * go_v_59, acc_i_23);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_23 = fma((scalar_t)(-0.40824827551841736) * x_j_6, y_k_5 * go_v_59, acc_i_23);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_23 = fma((scalar_t)(0.40824827551841736) * x_j_8, y_k_4 * go_v_59, acc_i_23);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_23 = fma((scalar_t)(0.40824827551841736) * x_j_4, y_k_7 * go_v_61, acc_i_23);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_23 = fma((scalar_t)(-0.81649655103683472) * x_j_6, y_k_4 * go_v_61, acc_i_23);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_23 = fma((scalar_t)(0.40824827551841736) * x_j_8, y_k_5 * go_v_61, acc_i_23);
    }

    grad_w[gw_base + ((int64_t)22 << 5)] += acc_i_22;
    grad_w[gw_base + ((int64_t)23 << 5)] += acc_i_23;

    // ---- grad_w chunk 7 ----
    scalar_t acc_i_24 = scalar_t(0);
    scalar_t acc_i_25 = scalar_t(0);

    // target i = 24
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_24 = fma((scalar_t)(0.40824827551841736) * x_j_5, y_k_5 * go_v_54, acc_i_24);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_24 = fma((scalar_t)(0.81649655103683472) * x_j_7, y_k_8 * go_v_54, acc_i_24);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_24 = fma((scalar_t)(-0.40824827551841736) * x_j_9, y_k_7 * go_v_54, acc_i_24);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_24 = fma((scalar_t)(-0.40824827551841736) * x_j_5, y_k_4 * go_v_56, acc_i_24);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_24 = fma((scalar_t)(0.40824827551841736) * x_j_7, y_k_7 * go_v_56, acc_i_24);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_24 = fma((scalar_t)(-0.70710676908493042) * x_j_9, y_k_6 * go_v_56, acc_i_24);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_24 = fma((scalar_t)(-0.40824827551841736) * x_j_9, y_k_8 * go_v_56, acc_i_24);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_58 = grad_out[go_base + ((int64_t)58 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_24 = fma((scalar_t)(-0.70710676908493042) * x_j_5, y_k_7 * go_v_58, acc_i_24);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_58 = grad_out[go_base + ((int64_t)58 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_24 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_5 * go_v_58, acc_i_24);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_24 = fma((scalar_t)(0.70710676908493042) * x_j_5, y_k_6 * go_v_60, acc_i_24);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_24 = fma((scalar_t)(-0.40824827551841736) * x_j_5, y_k_8 * go_v_60, acc_i_24);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_24 = fma((scalar_t)(-0.40824827551841736) * x_j_7, y_k_5 * go_v_60, acc_i_24);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_24 = fma((scalar_t)(0.40824827551841736) * x_j_9, y_k_4 * go_v_60, acc_i_24);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_24 = fma((scalar_t)(0.40824827551841736) * x_j_5, y_k_7 * go_v_62, acc_i_24);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_24 = fma((scalar_t)(-0.81649655103683472) * x_j_7, y_k_4 * go_v_62, acc_i_24);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_24 = fma((scalar_t)(0.40824827551841736) * x_j_9, y_k_5 * go_v_62, acc_i_24);
    }

    // target i = 25
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_25 = fma((scalar_t)(-0.40824827551841736) * x_j_11, y_k_1 * go_v_63, acc_i_25);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_25 = fma((scalar_t)(0.40824827551841736) * x_j_13, y_k_3 * go_v_63, acc_i_25);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_25 = fma((scalar_t)(-0.81649655103683472) * x_j_14, y_k_2 * go_v_63, acc_i_25);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_25 = fma((scalar_t)(0.40824827551841736) * x_j_10, y_k_1 * go_v_64, acc_i_25);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_25 = fma((scalar_t)(0.70710676908493042) * x_j_12, y_k_3 * go_v_64, acc_i_25);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_25 = fma((scalar_t)(0.40824827551841736) * x_j_14, y_k_3 * go_v_64, acc_i_25);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_25 = fma((scalar_t)(-0.40824827551841736) * x_j_13, y_k_2 * go_v_64, acc_i_25);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_65 = grad_out[go_base + ((int64_t)65 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_25 = fma((scalar_t)(-0.70710676908493042) * x_j_11, y_k_3 * go_v_65, acc_i_25);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_65 = grad_out[go_base + ((int64_t)65 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_25 = fma((scalar_t)(0.70710676908493042) * x_j_13, y_k_1 * go_v_65, acc_i_25);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_25 = fma((scalar_t)(-0.40824827551841736) * x_j_10, y_k_3 * go_v_66, acc_i_25);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_25 = fma((scalar_t)(0.40824827551841736) * x_j_11, y_k_2 * go_v_66, acc_i_25);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_25 = fma((scalar_t)(-0.70710676908493042) * x_j_12, y_k_1 * go_v_66, acc_i_25);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_25 = fma((scalar_t)(0.40824827551841736) * x_j_14, y_k_1 * go_v_66, acc_i_25);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_25 = fma((scalar_t)(0.81649655103683472) * x_j_10, y_k_2 * go_v_67, acc_i_25);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_25 = fma((scalar_t)(-0.40824827551841736) * x_j_11, y_k_3 * go_v_67, acc_i_25);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_25 = fma((scalar_t)(-0.40824827551841736) * x_j_13, y_k_1 * go_v_67, acc_i_25);
    }

    grad_w[gw_base + ((int64_t)24 << 5)] += acc_i_24;
    grad_w[gw_base + ((int64_t)25 << 5)] += acc_i_25;

    // ---- grad_w chunk 8 ----
    scalar_t acc_i_26 = scalar_t(0);
    scalar_t acc_i_27 = scalar_t(0);

    // target i = 26
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_26 = fma((scalar_t)(0.42257711291313171) * x_j_11, y_k_9 * go_v_68, acc_i_26);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_26 = fma((scalar_t)(0.32732683420181274) * x_j_11, y_k_11 * go_v_68, acc_i_26);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_26 = fma((scalar_t)(0.59761428833007812) * x_j_12, y_k_14 * go_v_68, acc_i_26);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_26 = fma((scalar_t)(-0.32732683420181274) * x_j_13, y_k_13 * go_v_68, acc_i_26);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_26 = fma((scalar_t)(0.42257711291313171) * x_j_13, y_k_15 * go_v_68, acc_i_26);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_26 = fma((scalar_t)(0.26726123690605164) * x_j_14, y_k_12 * go_v_68, acc_i_26);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_26 = fma((scalar_t)(-0.42257711291313171) * x_j_10, y_k_9 * go_v_69, acc_i_26);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_26 = fma((scalar_t)(-0.32732683420181274) * x_j_10, y_k_11 * go_v_69, acc_i_26);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_26 = fma((scalar_t)(0.37796446681022644) * x_j_12, y_k_13 * go_v_69, acc_i_26);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_26 = fma((scalar_t)(-0.32732683420181274) * x_j_14, y_k_13 * go_v_69, acc_i_26);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_26 = fma((scalar_t)(-0.53452247381210327) * x_j_13, y_k_12 * go_v_69, acc_i_26);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_26 = fma((scalar_t)(-0.42257711291313171) * x_j_14, y_k_15 * go_v_69, acc_i_26);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_26 = fma((scalar_t)(-0.59761428833007812) * x_j_10, y_k_14 * go_v_70, acc_i_26);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_26 = fma((scalar_t)(-0.37796446681022644) * x_j_11, y_k_13 * go_v_70, acc_i_26);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_26 = fma((scalar_t)(0.37796446681022644) * x_j_13, y_k_11 * go_v_70, acc_i_26);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_26 = fma((scalar_t)(0.59761428833007812) * x_j_14, y_k_10 * go_v_70, acc_i_26);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_26 = fma((scalar_t)(0.32732683420181274) * x_j_10, y_k_13 * go_v_71, acc_i_26);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_26 = fma((scalar_t)(-0.42257711291313171) * x_j_10, y_k_15 * go_v_71, acc_i_26);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_26 = fma((scalar_t)(0.53452247381210327) * x_j_11, y_k_12 * go_v_71, acc_i_26);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_26 = fma((scalar_t)(-0.37796446681022644) * x_j_12, y_k_11 * go_v_71, acc_i_26);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_26 = fma((scalar_t)(-0.32732683420181274) * x_j_14, y_k_11 * go_v_71, acc_i_26);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_26 = fma((scalar_t)(0.42257711291313171) * x_j_14, y_k_9 * go_v_71, acc_i_26);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_26 = fma((scalar_t)(-0.26726123690605164) * x_j_10, y_k_12 * go_v_72, acc_i_26);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_26 = fma((scalar_t)(0.32732683420181274) * x_j_11, y_k_13 * go_v_72, acc_i_26);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_26 = fma((scalar_t)(0.42257711291313171) * x_j_11, y_k_15 * go_v_72, acc_i_26);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_26 = fma((scalar_t)(-0.59761428833007812) * x_j_12, y_k_10 * go_v_72, acc_i_26);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_26 = fma((scalar_t)(-0.42257711291313171) * x_j_13, y_k_9 * go_v_72, acc_i_26);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_26 = fma((scalar_t)(0.32732683420181274) * x_j_13, y_k_11 * go_v_72, acc_i_26);
    }

    // target i = 27
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_27 = fma((scalar_t)(-0.42257711291313171) * x_j_15, y_k_5 * go_v_73, acc_i_27);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_27 = fma((scalar_t)(-0.32732683420181274) * x_j_17, y_k_5 * go_v_73, acc_i_27);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_27 = fma((scalar_t)(-0.26726123690605164) * x_j_18, y_k_8 * go_v_73, acc_i_27);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_27 = fma((scalar_t)(0.32732683420181274) * x_j_19, y_k_7 * go_v_73, acc_i_27);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_27 = fma((scalar_t)(-0.42257711291313171) * x_j_21, y_k_7 * go_v_73, acc_i_27);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_27 = fma((scalar_t)(-0.59761428833007812) * x_j_20, y_k_6 * go_v_73, acc_i_27);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_27 = fma((scalar_t)(0.42257711291313171) * x_j_15, y_k_4 * go_v_74, acc_i_27);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_27 = fma((scalar_t)(0.32732683420181274) * x_j_17, y_k_4 * go_v_74, acc_i_27);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_27 = fma((scalar_t)(0.53452247381210327) * x_j_18, y_k_7 * go_v_74, acc_i_27);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_27 = fma((scalar_t)(-0.37796446681022644) * x_j_19, y_k_6 * go_v_74, acc_i_27);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_27 = fma((scalar_t)(0.32732683420181274) * x_j_19, y_k_8 * go_v_74, acc_i_27);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_27 = fma((scalar_t)(0.42257711291313171) * x_j_21, y_k_8 * go_v_74, acc_i_27);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_27 = fma((scalar_t)(-0.59761428833007812) * x_j_16, y_k_8 * go_v_75, acc_i_27);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_27 = fma((scalar_t)(-0.37796446681022644) * x_j_17, y_k_7 * go_v_75, acc_i_27);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_27 = fma((scalar_t)(0.37796446681022644) * x_j_19, y_k_5 * go_v_75, acc_i_27);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_27 = fma((scalar_t)(0.59761428833007812) * x_j_20, y_k_4 * go_v_75, acc_i_27);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_27 = fma((scalar_t)(-0.42257711291313171) * x_j_15, y_k_8 * go_v_76, acc_i_27);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_27 = fma((scalar_t)(0.32732683420181274) * x_j_17, y_k_8 * go_v_76, acc_i_27);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_27 = fma((scalar_t)(0.37796446681022644) * x_j_17, y_k_6 * go_v_76, acc_i_27);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_27 = fma((scalar_t)(-0.53452247381210327) * x_j_18, y_k_5 * go_v_76, acc_i_27);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_27 = fma((scalar_t)(-0.32732683420181274) * x_j_19, y_k_4 * go_v_76, acc_i_27);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_27 = fma((scalar_t)(0.42257711291313171) * x_j_21, y_k_4 * go_v_76, acc_i_27);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_27 = fma((scalar_t)(0.42257711291313171) * x_j_15, y_k_7 * go_v_77, acc_i_27);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_27 = fma((scalar_t)(-0.32732683420181274) * x_j_17, y_k_7 * go_v_77, acc_i_27);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_27 = fma((scalar_t)(0.59761428833007812) * x_j_16, y_k_6 * go_v_77, acc_i_27);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_27 = fma((scalar_t)(0.26726123690605164) * x_j_18, y_k_4 * go_v_77, acc_i_27);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_27 = fma((scalar_t)(-0.32732683420181274) * x_j_19, y_k_5 * go_v_77, acc_i_27);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_27 = fma((scalar_t)(-0.42257711291313171) * x_j_21, y_k_5 * go_v_77, acc_i_27);
    }

    grad_w[gw_base + ((int64_t)26 << 5)] += acc_i_26;
    grad_w[gw_base + ((int64_t)27 << 5)] += acc_i_27;

    // ---- grad_w chunk 9 ----
    scalar_t acc_i_28 = scalar_t(0);
    scalar_t acc_i_29 = scalar_t(0);
    scalar_t acc_i_30 = scalar_t(0);
    scalar_t acc_i_31 = scalar_t(0);

    // target i = 28
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_78 = grad_out[go_base + ((int64_t)78 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_28 = fma((scalar_t)(1.0) * x_j_0, y_k_4 * go_v_78, acc_i_28);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_82 = grad_out[go_base + ((int64_t)82 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_28 = fma((scalar_t)(1.0) * x_j_0, y_k_5 * go_v_82, acc_i_28);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_86 = grad_out[go_base + ((int64_t)86 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_28 = fma((scalar_t)(1.0) * x_j_0, y_k_6 * go_v_86, acc_i_28);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_90 = grad_out[go_base + ((int64_t)90 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_28 = fma((scalar_t)(1.0) * x_j_0, y_k_7 * go_v_90, acc_i_28);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_94 = grad_out[go_base + ((int64_t)94 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_28 = fma((scalar_t)(1.0) * x_j_0, y_k_8 * go_v_94, acc_i_28);
    }

    // target i = 29
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_79 = grad_out[go_base + ((int64_t)79 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_29 = fma((scalar_t)(1.0) * x_j_1, y_k_4 * go_v_79, acc_i_29);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_83 = grad_out[go_base + ((int64_t)83 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_29 = fma((scalar_t)(1.0) * x_j_1, y_k_5 * go_v_83, acc_i_29);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_87 = grad_out[go_base + ((int64_t)87 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_29 = fma((scalar_t)(1.0) * x_j_1, y_k_6 * go_v_87, acc_i_29);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_91 = grad_out[go_base + ((int64_t)91 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_29 = fma((scalar_t)(1.0) * x_j_1, y_k_7 * go_v_91, acc_i_29);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_95 = grad_out[go_base + ((int64_t)95 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_29 = fma((scalar_t)(1.0) * x_j_1, y_k_8 * go_v_95, acc_i_29);
    }

    // target i = 30
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_80 = grad_out[go_base + ((int64_t)80 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_30 = fma((scalar_t)(1.0) * x_j_2, y_k_4 * go_v_80, acc_i_30);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_84 = grad_out[go_base + ((int64_t)84 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_30 = fma((scalar_t)(1.0) * x_j_2, y_k_5 * go_v_84, acc_i_30);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_88 = grad_out[go_base + ((int64_t)88 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_30 = fma((scalar_t)(1.0) * x_j_2, y_k_6 * go_v_88, acc_i_30);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_92 = grad_out[go_base + ((int64_t)92 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_30 = fma((scalar_t)(1.0) * x_j_2, y_k_7 * go_v_92, acc_i_30);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_96 = grad_out[go_base + ((int64_t)96 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_30 = fma((scalar_t)(1.0) * x_j_2, y_k_8 * go_v_96, acc_i_30);
    }

    // target i = 31
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_81 = grad_out[go_base + ((int64_t)81 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_31 = fma((scalar_t)(1.0) * x_j_3, y_k_4 * go_v_81, acc_i_31);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_85 = grad_out[go_base + ((int64_t)85 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_31 = fma((scalar_t)(1.0) * x_j_3, y_k_5 * go_v_85, acc_i_31);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_89 = grad_out[go_base + ((int64_t)89 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_31 = fma((scalar_t)(1.0) * x_j_3, y_k_6 * go_v_89, acc_i_31);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_93 = grad_out[go_base + ((int64_t)93 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_31 = fma((scalar_t)(1.0) * x_j_3, y_k_7 * go_v_93, acc_i_31);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_97 = grad_out[go_base + ((int64_t)97 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_31 = fma((scalar_t)(1.0) * x_j_3, y_k_8 * go_v_97, acc_i_31);
    }

    grad_w[gw_base + ((int64_t)28 << 5)] += acc_i_28;
    grad_w[gw_base + ((int64_t)29 << 5)] += acc_i_29;
    grad_w[gw_base + ((int64_t)30 << 5)] += acc_i_30;
    grad_w[gw_base + ((int64_t)31 << 5)] += acc_i_31;

    // ---- grad_w chunk 10 ----
    scalar_t acc_i_32 = scalar_t(0);
    scalar_t acc_i_33 = scalar_t(0);
    scalar_t acc_i_34 = scalar_t(0);

    // target i = 32
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_98 = grad_out[go_base + ((int64_t)98 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_4, y_k_3 * go_v_98, acc_i_32);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_98 = grad_out[go_base + ((int64_t)98 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_1 * go_v_98, acc_i_32);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_100 = grad_out[go_base + ((int64_t)100 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_4, y_k_2 * go_v_100, acc_i_32);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_100 = grad_out[go_base + ((int64_t)100 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_6, y_k_1 * go_v_100, acc_i_32);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_32 = fma((scalar_t)(-0.40824827551841736) * x_j_4, y_k_1 * go_v_102, acc_i_32);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_32 = fma((scalar_t)(0.81649655103683472) * x_j_6, y_k_2 * go_v_102, acc_i_32);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_32 = fma((scalar_t)(-0.40824827551841736) * x_j_8, y_k_3 * go_v_102, acc_i_32);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_104 = grad_out[go_base + ((int64_t)104 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_6, y_k_3 * go_v_104, acc_i_32);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_104 = grad_out[go_base + ((int64_t)104 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_2 * go_v_104, acc_i_32);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_106 = grad_out[go_base + ((int64_t)106 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_32 = fma((scalar_t)(-0.70710676908493042) * x_j_4, y_k_1 * go_v_106, acc_i_32);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_106 = grad_out[go_base + ((int64_t)106 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_32 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_3 * go_v_106, acc_i_32);
    }

    // target i = 33
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_99 = grad_out[go_base + ((int64_t)99 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_5, y_k_3 * go_v_99, acc_i_33);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_99 = grad_out[go_base + ((int64_t)99 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_1 * go_v_99, acc_i_33);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_101 = grad_out[go_base + ((int64_t)101 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_5, y_k_2 * go_v_101, acc_i_33);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_101 = grad_out[go_base + ((int64_t)101 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_7, y_k_1 * go_v_101, acc_i_33);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_33 = fma((scalar_t)(-0.40824827551841736) * x_j_5, y_k_1 * go_v_103, acc_i_33);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_33 = fma((scalar_t)(0.81649655103683472) * x_j_7, y_k_2 * go_v_103, acc_i_33);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_33 = fma((scalar_t)(-0.40824827551841736) * x_j_9, y_k_3 * go_v_103, acc_i_33);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_105 = grad_out[go_base + ((int64_t)105 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_7, y_k_3 * go_v_105, acc_i_33);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_105 = grad_out[go_base + ((int64_t)105 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_2 * go_v_105, acc_i_33);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_107 = grad_out[go_base + ((int64_t)107 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_33 = fma((scalar_t)(-0.70710676908493042) * x_j_5, y_k_1 * go_v_107, acc_i_33);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_107 = grad_out[go_base + ((int64_t)107 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_33 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_3 * go_v_107, acc_i_33);
    }

    // target i = 34
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(-0.15430334210395813) * x_j_4, y_k_13 * go_v_108, acc_i_34);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(-0.59761428833007812) * x_j_4, y_k_15 * go_v_108, acc_i_34);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_34 = fma((scalar_t)(0.48795005679130554) * x_j_6, y_k_10 * go_v_108, acc_i_34);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(0.59761428833007812) * x_j_8, y_k_9 * go_v_108, acc_i_34);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(-0.15430334210395813) * x_j_8, y_k_11 * go_v_108, acc_i_34);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(-0.37796446681022644) * x_j_4, y_k_12 * go_v_110, acc_i_34);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(-0.48795005679130554) * x_j_4, y_k_14 * go_v_110, acc_i_34);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_34 = fma((scalar_t)(0.61721336841583252) * x_j_6, y_k_11 * go_v_110, acc_i_34);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(0.48795005679130554) * x_j_8, y_k_10 * go_v_110, acc_i_34);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(0.53452247381210327) * x_j_4, y_k_11 * go_v_112, acc_i_34);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_34 = fma((scalar_t)(0.65465366840362549) * x_j_6, y_k_12 * go_v_112, acc_i_34);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(0.53452247381210327) * x_j_8, y_k_13 * go_v_112, acc_i_34);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(0.48795005679130554) * x_j_4, y_k_10 * go_v_114, acc_i_34);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_34 = fma((scalar_t)(0.61721336841583252) * x_j_6, y_k_13 * go_v_114, acc_i_34);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(-0.37796446681022644) * x_j_8, y_k_12 * go_v_114, acc_i_34);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(0.48795005679130554) * x_j_8, y_k_14 * go_v_114, acc_i_34);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(0.59761428833007812) * x_j_4, y_k_9 * go_v_116, acc_i_34);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_34 = fma((scalar_t)(0.15430334210395813) * x_j_4, y_k_11 * go_v_116, acc_i_34);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_34 = fma((scalar_t)(0.48795005679130554) * x_j_6, y_k_14 * go_v_116, acc_i_34);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(-0.15430334210395813) * x_j_8, y_k_13 * go_v_116, acc_i_34);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_34 = fma((scalar_t)(0.59761428833007812) * x_j_8, y_k_15 * go_v_116, acc_i_34);
    }

    grad_w[gw_base + ((int64_t)32 << 5)] += acc_i_32;
    grad_w[gw_base + ((int64_t)33 << 5)] += acc_i_33;
    grad_w[gw_base + ((int64_t)34 << 5)] += acc_i_34;

    // ---- grad_w chunk 11 ----
    scalar_t acc_i_35 = scalar_t(0);
    scalar_t acc_i_36 = scalar_t(0);

    // target i = 35
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(-0.15430334210395813) * x_j_5, y_k_13 * go_v_109, acc_i_35);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(-0.59761428833007812) * x_j_5, y_k_15 * go_v_109, acc_i_35);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_35 = fma((scalar_t)(0.48795005679130554) * x_j_7, y_k_10 * go_v_109, acc_i_35);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(0.59761428833007812) * x_j_9, y_k_9 * go_v_109, acc_i_35);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(-0.15430334210395813) * x_j_9, y_k_11 * go_v_109, acc_i_35);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(-0.37796446681022644) * x_j_5, y_k_12 * go_v_111, acc_i_35);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(-0.48795005679130554) * x_j_5, y_k_14 * go_v_111, acc_i_35);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_35 = fma((scalar_t)(0.61721336841583252) * x_j_7, y_k_11 * go_v_111, acc_i_35);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(0.48795005679130554) * x_j_9, y_k_10 * go_v_111, acc_i_35);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(0.53452247381210327) * x_j_5, y_k_11 * go_v_113, acc_i_35);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_35 = fma((scalar_t)(0.65465366840362549) * x_j_7, y_k_12 * go_v_113, acc_i_35);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(0.53452247381210327) * x_j_9, y_k_13 * go_v_113, acc_i_35);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(0.48795005679130554) * x_j_5, y_k_10 * go_v_115, acc_i_35);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_35 = fma((scalar_t)(0.61721336841583252) * x_j_7, y_k_13 * go_v_115, acc_i_35);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(-0.37796446681022644) * x_j_9, y_k_12 * go_v_115, acc_i_35);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(0.48795005679130554) * x_j_9, y_k_14 * go_v_115, acc_i_35);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(0.59761428833007812) * x_j_5, y_k_9 * go_v_117, acc_i_35);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_35 = fma((scalar_t)(0.15430334210395813) * x_j_5, y_k_11 * go_v_117, acc_i_35);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_35 = fma((scalar_t)(0.48795005679130554) * x_j_7, y_k_14 * go_v_117, acc_i_35);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(-0.15430334210395813) * x_j_9, y_k_13 * go_v_117, acc_i_35);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_35 = fma((scalar_t)(0.59761428833007812) * x_j_9, y_k_15 * go_v_117, acc_i_35);
    }

    // target i = 36
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_118 = grad_out[go_base + ((int64_t)118 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_36 = fma((scalar_t)(1.0) * x_j_10, y_k_0 * go_v_118, acc_i_36);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_119 = grad_out[go_base + ((int64_t)119 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_36 = fma((scalar_t)(1.0) * x_j_11, y_k_0 * go_v_119, acc_i_36);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_120 = grad_out[go_base + ((int64_t)120 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_36 = fma((scalar_t)(1.0) * x_j_12, y_k_0 * go_v_120, acc_i_36);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_121 = grad_out[go_base + ((int64_t)121 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_36 = fma((scalar_t)(1.0) * x_j_13, y_k_0 * go_v_121, acc_i_36);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_122 = grad_out[go_base + ((int64_t)122 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_36 = fma((scalar_t)(1.0) * x_j_14, y_k_0 * go_v_122, acc_i_36);
    }

    grad_w[gw_base + ((int64_t)35 << 5)] += acc_i_35;
    grad_w[gw_base + ((int64_t)36 << 5)] += acc_i_36;

    // ---- grad_w chunk 12 ----
    scalar_t acc_i_37 = scalar_t(0);
    scalar_t acc_i_38 = scalar_t(0);

    // target i = 37
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_37 = fma((scalar_t)(-0.53452247381210327) * x_j_10, y_k_6 * go_v_123, acc_i_37);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_11, y_k_7 * go_v_123, acc_i_37);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_37 = fma((scalar_t)(-0.53452247381210327) * x_j_12, y_k_4 * go_v_123, acc_i_37);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_13, y_k_5 * go_v_123, acc_i_37);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_10, y_k_7 * go_v_124, acc_i_37);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_37 = fma((scalar_t)(0.26726123690605164) * x_j_11, y_k_6 * go_v_124, acc_i_37);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_37 = fma((scalar_t)(-0.46291005611419678) * x_j_11, y_k_8 * go_v_124, acc_i_37);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_37 = fma((scalar_t)(0.26726123690605164) * x_j_12, y_k_5 * go_v_124, acc_i_37);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_37 = fma((scalar_t)(-0.46291005611419678) * x_j_14, y_k_5 * go_v_124, acc_i_37);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_13, y_k_4 * go_v_124, acc_i_37);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_37 = fma((scalar_t)(-0.53452247381210327) * x_j_10, y_k_4 * go_v_125, acc_i_37);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_37 = fma((scalar_t)(0.26726123690605164) * x_j_11, y_k_5 * go_v_125, acc_i_37);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_37 = fma((scalar_t)(0.53452247381210327) * x_j_12, y_k_6 * go_v_125, acc_i_37);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_37 = fma((scalar_t)(0.26726123690605164) * x_j_13, y_k_7 * go_v_125, acc_i_37);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_37 = fma((scalar_t)(-0.53452247381210327) * x_j_14, y_k_8 * go_v_125, acc_i_37);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_10, y_k_5 * go_v_126, acc_i_37);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_11, y_k_4 * go_v_126, acc_i_37);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_37 = fma((scalar_t)(0.26726123690605164) * x_j_12, y_k_7 * go_v_126, acc_i_37);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_14, y_k_7 * go_v_126, acc_i_37);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_37 = fma((scalar_t)(0.26726123690605164) * x_j_13, y_k_6 * go_v_126, acc_i_37);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_13, y_k_8 * go_v_126, acc_i_37);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_37 = fma((scalar_t)(-0.46291005611419678) * x_j_11, y_k_5 * go_v_127, acc_i_37);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_37 = fma((scalar_t)(-0.53452247381210327) * x_j_12, y_k_8 * go_v_127, acc_i_37);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_37 = fma((scalar_t)(0.46291005611419678) * x_j_13, y_k_7 * go_v_127, acc_i_37);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_37 = fma((scalar_t)(-0.53452247381210327) * x_j_14, y_k_6 * go_v_127, acc_i_37);
    }

    // target i = 38
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_38 = fma((scalar_t)(0.59761428833007812) * x_j_15, y_k_3 * go_v_128, acc_i_38);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_38 = fma((scalar_t)(-0.15430334210395813) * x_j_17, y_k_3 * go_v_128, acc_i_38);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_38 = fma((scalar_t)(0.48795005679130554) * x_j_16, y_k_2 * go_v_128, acc_i_38);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_38 = fma((scalar_t)(-0.15430334210395813) * x_j_19, y_k_1 * go_v_128, acc_i_38);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_38 = fma((scalar_t)(-0.59761428833007812) * x_j_21, y_k_1 * go_v_128, acc_i_38);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_38 = fma((scalar_t)(0.48795005679130554) * x_j_16, y_k_3 * go_v_129, acc_i_38);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_38 = fma((scalar_t)(0.61721336841583252) * x_j_17, y_k_2 * go_v_129, acc_i_38);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_38 = fma((scalar_t)(-0.37796446681022644) * x_j_18, y_k_1 * go_v_129, acc_i_38);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_38 = fma((scalar_t)(-0.48795005679130554) * x_j_20, y_k_1 * go_v_129, acc_i_38);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_38 = fma((scalar_t)(0.53452247381210327) * x_j_17, y_k_1 * go_v_130, acc_i_38);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_38 = fma((scalar_t)(0.65465366840362549) * x_j_18, y_k_2 * go_v_130, acc_i_38);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_38 = fma((scalar_t)(0.53452247381210327) * x_j_19, y_k_3 * go_v_130, acc_i_38);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_38 = fma((scalar_t)(0.48795005679130554) * x_j_16, y_k_1 * go_v_131, acc_i_38);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_38 = fma((scalar_t)(-0.37796446681022644) * x_j_18, y_k_3 * go_v_131, acc_i_38);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_38 = fma((scalar_t)(0.48795005679130554) * x_j_20, y_k_3 * go_v_131, acc_i_38);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_38 = fma((scalar_t)(0.61721336841583252) * x_j_19, y_k_2 * go_v_131, acc_i_38);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_38 = fma((scalar_t)(0.59761428833007812) * x_j_15, y_k_1 * go_v_132, acc_i_38);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_38 = fma((scalar_t)(0.15430334210395813) * x_j_17, y_k_1 * go_v_132, acc_i_38);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_38 = fma((scalar_t)(-0.15430334210395813) * x_j_19, y_k_3 * go_v_132, acc_i_38);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_38 = fma((scalar_t)(0.59761428833007812) * x_j_21, y_k_3 * go_v_132, acc_i_38);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_38 = fma((scalar_t)(0.48795005679130554) * x_j_20, y_k_2 * go_v_132, acc_i_38);
    }

    grad_w[gw_base + ((int64_t)37 << 5)] += acc_i_37;
    grad_w[gw_base + ((int64_t)38 << 5)] += acc_i_38;

    // ---- grad_w chunk 13 ----
    scalar_t acc_i_39 = scalar_t(0);
    scalar_t acc_i_40 = scalar_t(0);

    // target i = 39
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_39 = fma((scalar_t)(-0.24397502839565277) * x_j_15, y_k_13 * go_v_133, acc_i_39);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(0.37796446681022644) * x_j_17, y_k_13 * go_v_133, acc_i_39);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_39 = fma((scalar_t)(-0.48795005679130554) * x_j_16, y_k_12 * go_v_133, acc_i_39);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(0.24397502839565277) * x_j_17, y_k_15 * go_v_133, acc_i_39);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_39 = fma((scalar_t)(-0.48795005679130554) * x_j_18, y_k_10 * go_v_133, acc_i_39);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(-0.24397502839565277) * x_j_19, y_k_9 * go_v_133, acc_i_39);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(0.37796446681022644) * x_j_19, y_k_11 * go_v_133, acc_i_39);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_39 = fma((scalar_t)(0.24397502839565277) * x_j_21, y_k_11 * go_v_133, acc_i_39);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_39 = fma((scalar_t)(0.38575837016105652) * x_j_15, y_k_14 * go_v_134, acc_i_39);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(-0.29880714416503906) * x_j_17, y_k_14 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_39 = fma((scalar_t)(0.29880714416503906) * x_j_16, y_k_13 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_39 = fma((scalar_t)(-0.38575837016105652) * x_j_16, y_k_15 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(0.15430334210395813) * x_j_17, y_k_12 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_39 = fma((scalar_t)(0.15430334210395813) * x_j_18, y_k_11 * go_v_134, acc_i_39);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_39 = fma((scalar_t)(-0.29880714416503906) * x_j_20, y_k_11 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(0.29880714416503906) * x_j_19, y_k_10 * go_v_134, acc_i_39);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_39 = fma((scalar_t)(-0.38575837016105652) * x_j_21, y_k_10 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_39 = fma((scalar_t)(0.38575837016105652) * x_j_20, y_k_9 * go_v_134, acc_i_39);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_39 = fma((scalar_t)(-0.54554474353790283) * x_j_15, y_k_9 * go_v_135, acc_i_39);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(0.32732683420181274) * x_j_17, y_k_11 * go_v_135, acc_i_39);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_39 = fma((scalar_t)(0.43643578886985779) * x_j_18, y_k_12 * go_v_135, acc_i_39);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(0.32732683420181274) * x_j_19, y_k_13 * go_v_135, acc_i_39);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_39 = fma((scalar_t)(-0.54554474353790283) * x_j_21, y_k_15 * go_v_135, acc_i_39);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_39 = fma((scalar_t)(0.38575837016105652) * x_j_15, y_k_10 * go_v_136, acc_i_39);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(0.29880714416503906) * x_j_17, y_k_10 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_39 = fma((scalar_t)(0.38575837016105652) * x_j_16, y_k_9 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_39 = fma((scalar_t)(0.29880714416503906) * x_j_16, y_k_11 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_39 = fma((scalar_t)(0.15430334210395813) * x_j_18, y_k_13 * go_v_136, acc_i_39);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_39 = fma((scalar_t)(0.29880714416503906) * x_j_20, y_k_13 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(0.15430334210395813) * x_j_19, y_k_12 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(0.29880714416503906) * x_j_19, y_k_14 * go_v_136, acc_i_39);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_39 = fma((scalar_t)(0.38575837016105652) * x_j_21, y_k_14 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_39 = fma((scalar_t)(0.38575837016105652) * x_j_20, y_k_15 * go_v_136, acc_i_39);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_39 = fma((scalar_t)(-0.24397502839565277) * x_j_15, y_k_11 * go_v_137, acc_i_39);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(-0.37796446681022644) * x_j_17, y_k_11 * go_v_137, acc_i_39);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_39 = fma((scalar_t)(-0.24397502839565277) * x_j_17, y_k_9 * go_v_137, acc_i_39);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_39 = fma((scalar_t)(-0.48795005679130554) * x_j_18, y_k_14 * go_v_137, acc_i_39);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(0.37796446681022644) * x_j_19, y_k_13 * go_v_137, acc_i_39);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_39 = fma((scalar_t)(-0.24397502839565277) * x_j_21, y_k_13 * go_v_137, acc_i_39);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_39 = fma((scalar_t)(-0.24397502839565277) * x_j_19, y_k_15 * go_v_137, acc_i_39);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_39 = fma((scalar_t)(-0.48795005679130554) * x_j_20, y_k_12 * go_v_137, acc_i_39);
    }

    // target i = 40
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_138 = grad_out[go_base + ((int64_t)138 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_9 * go_v_138, acc_i_40);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_142 = grad_out[go_base + ((int64_t)142 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_10 * go_v_142, acc_i_40);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_146 = grad_out[go_base + ((int64_t)146 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_11 * go_v_146, acc_i_40);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_150 = grad_out[go_base + ((int64_t)150 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_12 * go_v_150, acc_i_40);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_154 = grad_out[go_base + ((int64_t)154 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_13 * go_v_154, acc_i_40);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_158 = grad_out[go_base + ((int64_t)158 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_14 * go_v_158, acc_i_40);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_162 = grad_out[go_base + ((int64_t)162 << 5)];
        scalar_t x_j_0 = x_all[x_base + ((int64_t)0 << 5)];
        acc_i_40 = fma((scalar_t)(1.0) * x_j_0, y_k_15 * go_v_162, acc_i_40);
    }

    grad_w[gw_base + ((int64_t)39 << 5)] += acc_i_39;
    grad_w[gw_base + ((int64_t)40 << 5)] += acc_i_40;

    // ---- grad_w chunk 14 ----
    scalar_t acc_i_41 = scalar_t(0);
    scalar_t acc_i_42 = scalar_t(0);
    scalar_t acc_i_43 = scalar_t(0);

    // target i = 41
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_139 = grad_out[go_base + ((int64_t)139 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_9 * go_v_139, acc_i_41);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_143 = grad_out[go_base + ((int64_t)143 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_10 * go_v_143, acc_i_41);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_147 = grad_out[go_base + ((int64_t)147 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_11 * go_v_147, acc_i_41);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_151 = grad_out[go_base + ((int64_t)151 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_12 * go_v_151, acc_i_41);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_155 = grad_out[go_base + ((int64_t)155 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_13 * go_v_155, acc_i_41);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_159 = grad_out[go_base + ((int64_t)159 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_14 * go_v_159, acc_i_41);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_163 = grad_out[go_base + ((int64_t)163 << 5)];
        scalar_t x_j_1 = x_all[x_base + ((int64_t)1 << 5)];
        acc_i_41 = fma((scalar_t)(1.0) * x_j_1, y_k_15 * go_v_163, acc_i_41);
    }

    // target i = 42
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_140 = grad_out[go_base + ((int64_t)140 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_9 * go_v_140, acc_i_42);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_144 = grad_out[go_base + ((int64_t)144 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_10 * go_v_144, acc_i_42);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_148 = grad_out[go_base + ((int64_t)148 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_11 * go_v_148, acc_i_42);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_152 = grad_out[go_base + ((int64_t)152 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_12 * go_v_152, acc_i_42);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_156 = grad_out[go_base + ((int64_t)156 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_13 * go_v_156, acc_i_42);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_160 = grad_out[go_base + ((int64_t)160 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_14 * go_v_160, acc_i_42);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_164 = grad_out[go_base + ((int64_t)164 << 5)];
        scalar_t x_j_2 = x_all[x_base + ((int64_t)2 << 5)];
        acc_i_42 = fma((scalar_t)(1.0) * x_j_2, y_k_15 * go_v_164, acc_i_42);
    }

    // target i = 43
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_141 = grad_out[go_base + ((int64_t)141 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_9 * go_v_141, acc_i_43);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_145 = grad_out[go_base + ((int64_t)145 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_10 * go_v_145, acc_i_43);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_149 = grad_out[go_base + ((int64_t)149 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_11 * go_v_149, acc_i_43);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_153 = grad_out[go_base + ((int64_t)153 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_12 * go_v_153, acc_i_43);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_157 = grad_out[go_base + ((int64_t)157 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_13 * go_v_157, acc_i_43);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_161 = grad_out[go_base + ((int64_t)161 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_14 * go_v_161, acc_i_43);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_165 = grad_out[go_base + ((int64_t)165 << 5)];
        scalar_t x_j_3 = x_all[x_base + ((int64_t)3 << 5)];
        acc_i_43 = fma((scalar_t)(1.0) * x_j_3, y_k_15 * go_v_165, acc_i_43);
    }

    grad_w[gw_base + ((int64_t)41 << 5)] += acc_i_41;
    grad_w[gw_base + ((int64_t)42 << 5)] += acc_i_42;
    grad_w[gw_base + ((int64_t)43 << 5)] += acc_i_43;

    // ---- grad_w chunk 15 ----
    scalar_t acc_i_44 = scalar_t(0);
    scalar_t acc_i_45 = scalar_t(0);

    // target i = 44
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_166 = grad_out[go_base + ((int64_t)166 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(0.70710676908493042) * x_j_4, y_k_8 * go_v_166, acc_i_44);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_166 = grad_out[go_base + ((int64_t)166 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_4 * go_v_166, acc_i_44);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(0.57735025882720947) * x_j_4, y_k_7 * go_v_168, acc_i_44);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_44 = fma((scalar_t)(0.57735025882720947) * x_j_6, y_k_4 * go_v_168, acc_i_44);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(0.57735025882720947) * x_j_8, y_k_5 * go_v_168, acc_i_44);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(0.63245552778244019) * x_j_4, y_k_6 * go_v_170, acc_i_44);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(0.18257418274879456) * x_j_4, y_k_8 * go_v_170, acc_i_44);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_44 = fma((scalar_t)(0.73029673099517822) * x_j_6, y_k_5 * go_v_170, acc_i_44);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(-0.18257418274879456) * x_j_8, y_k_4 * go_v_170, acc_i_44);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(-0.44721359014511108) * x_j_4, y_k_5 * go_v_172, acc_i_44);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_44 = fma((scalar_t)(0.7745966911315918) * x_j_6, y_k_6 * go_v_172, acc_i_44);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(-0.44721359014511108) * x_j_8, y_k_7 * go_v_172, acc_i_44);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(-0.18257418274879456) * x_j_4, y_k_4 * go_v_174, acc_i_44);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_44 = fma((scalar_t)(0.73029673099517822) * x_j_6, y_k_7 * go_v_174, acc_i_44);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(0.63245552778244019) * x_j_8, y_k_6 * go_v_174, acc_i_44);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(-0.18257418274879456) * x_j_8, y_k_8 * go_v_174, acc_i_44);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(-0.57735025882720947) * x_j_4, y_k_5 * go_v_176, acc_i_44);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_44 = fma((scalar_t)(0.57735025882720947) * x_j_6, y_k_8 * go_v_176, acc_i_44);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(0.57735025882720947) * x_j_8, y_k_7 * go_v_176, acc_i_44);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_178 = grad_out[go_base + ((int64_t)178 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_44 = fma((scalar_t)(-0.70710676908493042) * x_j_4, y_k_4 * go_v_178, acc_i_44);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_178 = grad_out[go_base + ((int64_t)178 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_44 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_8 * go_v_178, acc_i_44);
    }

    // target i = 45
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_167 = grad_out[go_base + ((int64_t)167 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(0.70710676908493042) * x_j_5, y_k_8 * go_v_167, acc_i_45);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_167 = grad_out[go_base + ((int64_t)167 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_4 * go_v_167, acc_i_45);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(0.57735025882720947) * x_j_5, y_k_7 * go_v_169, acc_i_45);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_45 = fma((scalar_t)(0.57735025882720947) * x_j_7, y_k_4 * go_v_169, acc_i_45);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(0.57735025882720947) * x_j_9, y_k_5 * go_v_169, acc_i_45);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(0.63245552778244019) * x_j_5, y_k_6 * go_v_171, acc_i_45);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(0.18257418274879456) * x_j_5, y_k_8 * go_v_171, acc_i_45);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_45 = fma((scalar_t)(0.73029673099517822) * x_j_7, y_k_5 * go_v_171, acc_i_45);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(-0.18257418274879456) * x_j_9, y_k_4 * go_v_171, acc_i_45);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(-0.44721359014511108) * x_j_5, y_k_5 * go_v_173, acc_i_45);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_45 = fma((scalar_t)(0.7745966911315918) * x_j_7, y_k_6 * go_v_173, acc_i_45);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(-0.44721359014511108) * x_j_9, y_k_7 * go_v_173, acc_i_45);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(-0.18257418274879456) * x_j_5, y_k_4 * go_v_175, acc_i_45);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_45 = fma((scalar_t)(0.73029673099517822) * x_j_7, y_k_7 * go_v_175, acc_i_45);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(0.63245552778244019) * x_j_9, y_k_6 * go_v_175, acc_i_45);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(-0.18257418274879456) * x_j_9, y_k_8 * go_v_175, acc_i_45);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(-0.57735025882720947) * x_j_5, y_k_5 * go_v_177, acc_i_45);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_45 = fma((scalar_t)(0.57735025882720947) * x_j_7, y_k_8 * go_v_177, acc_i_45);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(0.57735025882720947) * x_j_9, y_k_7 * go_v_177, acc_i_45);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_179 = grad_out[go_base + ((int64_t)179 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_45 = fma((scalar_t)(-0.70710676908493042) * x_j_5, y_k_4 * go_v_179, acc_i_45);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_179 = grad_out[go_base + ((int64_t)179 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_45 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_8 * go_v_179, acc_i_45);
    }

    grad_w[gw_base + ((int64_t)44 << 5)] += acc_i_44;
    grad_w[gw_base + ((int64_t)45 << 5)] += acc_i_45;

    // ---- grad_w chunk 16 ----
    scalar_t acc_i_46 = scalar_t(0);
    scalar_t acc_i_47 = scalar_t(0);

    // target i = 46
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_180 = grad_out[go_base + ((int64_t)180 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_46 = fma((scalar_t)(0.70710676908493042) * x_j_10, y_k_3 * go_v_180, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_180 = grad_out[go_base + ((int64_t)180 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_46 = fma((scalar_t)(0.70710676908493042) * x_j_14, y_k_1 * go_v_180, acc_i_46);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_46 = fma((scalar_t)(0.57735025882720947) * x_j_10, y_k_2 * go_v_181, acc_i_46);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_46 = fma((scalar_t)(0.57735025882720947) * x_j_11, y_k_3 * go_v_181, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_46 = fma((scalar_t)(0.57735025882720947) * x_j_13, y_k_1 * go_v_181, acc_i_46);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_46 = fma((scalar_t)(-0.18257418274879456) * x_j_10, y_k_3 * go_v_182, acc_i_46);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_46 = fma((scalar_t)(0.73029673099517822) * x_j_11, y_k_2 * go_v_182, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_46 = fma((scalar_t)(0.63245552778244019) * x_j_12, y_k_1 * go_v_182, acc_i_46);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_46 = fma((scalar_t)(0.18257418274879456) * x_j_14, y_k_1 * go_v_182, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_46 = fma((scalar_t)(-0.44721359014511108) * x_j_11, y_k_1 * go_v_183, acc_i_46);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_46 = fma((scalar_t)(0.7745966911315918) * x_j_12, y_k_2 * go_v_183, acc_i_46);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_46 = fma((scalar_t)(-0.44721359014511108) * x_j_13, y_k_3 * go_v_183, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_46 = fma((scalar_t)(-0.18257418274879456) * x_j_10, y_k_1 * go_v_184, acc_i_46);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_46 = fma((scalar_t)(0.63245552778244019) * x_j_12, y_k_3 * go_v_184, acc_i_46);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_46 = fma((scalar_t)(-0.18257418274879456) * x_j_14, y_k_3 * go_v_184, acc_i_46);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_46 = fma((scalar_t)(0.73029673099517822) * x_j_13, y_k_2 * go_v_184, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_46 = fma((scalar_t)(-0.57735025882720947) * x_j_11, y_k_1 * go_v_185, acc_i_46);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_46 = fma((scalar_t)(0.57735025882720947) * x_j_13, y_k_3 * go_v_185, acc_i_46);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_46 = fma((scalar_t)(0.57735025882720947) * x_j_14, y_k_2 * go_v_185, acc_i_46);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_186 = grad_out[go_base + ((int64_t)186 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_46 = fma((scalar_t)(-0.70710676908493042) * x_j_10, y_k_1 * go_v_186, acc_i_46);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_186 = grad_out[go_base + ((int64_t)186 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_46 = fma((scalar_t)(0.70710676908493042) * x_j_14, y_k_3 * go_v_186, acc_i_46);
    }

    // target i = 47
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(-0.28867512941360474) * x_j_10, y_k_13 * go_v_187, acc_i_47);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(0.45643547177314758) * x_j_11, y_k_14 * go_v_187, acc_i_47);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_47 = fma((scalar_t)(-0.64549720287322998) * x_j_12, y_k_9 * go_v_187, acc_i_47);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.45643547177314758) * x_j_13, y_k_10 * go_v_187, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.28867512941360474) * x_j_14, y_k_11 * go_v_187, acc_i_47);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(-0.57735025882720947) * x_j_10, y_k_12 * go_v_188, acc_i_47);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(0.35355338454246521) * x_j_11, y_k_13 * go_v_188, acc_i_47);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(-0.45643547177314758) * x_j_11, y_k_15 * go_v_188, acc_i_47);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.45643547177314758) * x_j_13, y_k_9 * go_v_188, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.35355338454246521) * x_j_13, y_k_11 * go_v_188, acc_i_47);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(0.44721359014511108) * x_j_10, y_k_13 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(0.28867512941360474) * x_j_10, y_k_15 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(0.18257418274879456) * x_j_11, y_k_12 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(-0.35355338454246521) * x_j_11, y_k_14 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_47 = fma((scalar_t)(0.3872983455657959) * x_j_12, y_k_11 * go_v_189, acc_i_47);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.44721359014511108) * x_j_14, y_k_11 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.35355338454246521) * x_j_13, y_k_10 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.28867512941360474) * x_j_14, y_k_9 * go_v_189, acc_i_47);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(-0.57735025882720947) * x_j_10, y_k_10 * go_v_190, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(0.18257418274879456) * x_j_11, y_k_11 * go_v_190, acc_i_47);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_47 = fma((scalar_t)(0.51639777421951294) * x_j_12, y_k_12 * go_v_190, acc_i_47);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.18257418274879456) * x_j_13, y_k_13 * go_v_190, acc_i_47);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.57735025882720947) * x_j_14, y_k_14 * go_v_190, acc_i_47);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(-0.28867512941360474) * x_j_10, y_k_9 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(0.44721359014511108) * x_j_10, y_k_11 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(0.35355338454246521) * x_j_11, y_k_10 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_47 = fma((scalar_t)(0.3872983455657959) * x_j_12, y_k_13 * go_v_191, acc_i_47);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(0.44721359014511108) * x_j_14, y_k_13 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.18257418274879456) * x_j_13, y_k_12 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.35355338454246521) * x_j_13, y_k_14 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.28867512941360474) * x_j_14, y_k_15 * go_v_191, acc_i_47);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(0.45643547177314758) * x_j_11, y_k_9 * go_v_192, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(-0.35355338454246521) * x_j_11, y_k_11 * go_v_192, acc_i_47);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.35355338454246521) * x_j_13, y_k_13 * go_v_192, acc_i_47);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.45643547177314758) * x_j_13, y_k_15 * go_v_192, acc_i_47);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.57735025882720947) * x_j_14, y_k_12 * go_v_192, acc_i_47);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_47 = fma((scalar_t)(0.28867512941360474) * x_j_10, y_k_11 * go_v_193, acc_i_47);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_47 = fma((scalar_t)(-0.45643547177314758) * x_j_11, y_k_10 * go_v_193, acc_i_47);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_47 = fma((scalar_t)(-0.64549720287322998) * x_j_12, y_k_15 * go_v_193, acc_i_47);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_47 = fma((scalar_t)(0.45643547177314758) * x_j_13, y_k_14 * go_v_193, acc_i_47);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_47 = fma((scalar_t)(-0.28867512941360474) * x_j_14, y_k_13 * go_v_193, acc_i_47);
    }

    grad_w[gw_base + ((int64_t)46 << 5)] += acc_i_46;
    grad_w[gw_base + ((int64_t)47 << 5)] += acc_i_47;

    // ---- grad_w chunk 17 ----
    scalar_t acc_i_48 = scalar_t(0);
    scalar_t acc_i_49 = scalar_t(0);

    // target i = 48
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_194 = grad_out[go_base + ((int64_t)194 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_15, y_k_0 * go_v_194, acc_i_48);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_195 = grad_out[go_base + ((int64_t)195 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_16, y_k_0 * go_v_195, acc_i_48);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_196 = grad_out[go_base + ((int64_t)196 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_17, y_k_0 * go_v_196, acc_i_48);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_197 = grad_out[go_base + ((int64_t)197 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_18, y_k_0 * go_v_197, acc_i_48);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_198 = grad_out[go_base + ((int64_t)198 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_19, y_k_0 * go_v_198, acc_i_48);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_199 = grad_out[go_base + ((int64_t)199 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_20, y_k_0 * go_v_199, acc_i_48);
    }
    {
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_200 = grad_out[go_base + ((int64_t)200 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_48 = fma((scalar_t)(1.0) * x_j_21, y_k_0 * go_v_200, acc_i_48);
    }

    // target i = 49
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_49 = fma((scalar_t)(-0.64549720287322998) * x_j_15, y_k_6 * go_v_201, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_49 = fma((scalar_t)(0.45643547177314758) * x_j_16, y_k_7 * go_v_201, acc_i_49);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(-0.28867512941360474) * x_j_17, y_k_8 * go_v_201, acc_i_49);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(-0.28867512941360474) * x_j_19, y_k_4 * go_v_201, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_49 = fma((scalar_t)(0.45643547177314758) * x_j_20, y_k_5 * go_v_201, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_49 = fma((scalar_t)(0.45643547177314758) * x_j_15, y_k_7 * go_v_202, acc_i_49);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(0.35355338454246521) * x_j_17, y_k_7 * go_v_202, acc_i_49);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_49 = fma((scalar_t)(-0.57735025882720947) * x_j_18, y_k_4 * go_v_202, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(0.35355338454246521) * x_j_19, y_k_5 * go_v_202, acc_i_49);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_49 = fma((scalar_t)(-0.45643547177314758) * x_j_21, y_k_5 * go_v_202, acc_i_49);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_49 = fma((scalar_t)(-0.28867512941360474) * x_j_15, y_k_8 * go_v_203, acc_i_49);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(-0.44721359014511108) * x_j_17, y_k_8 * go_v_203, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_49 = fma((scalar_t)(0.35355338454246521) * x_j_16, y_k_7 * go_v_203, acc_i_49);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(0.3872983455657959) * x_j_17, y_k_6 * go_v_203, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_49 = fma((scalar_t)(0.18257418274879456) * x_j_18, y_k_5 * go_v_203, acc_i_49);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_49 = fma((scalar_t)(-0.35355338454246521) * x_j_20, y_k_5 * go_v_203, acc_i_49);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(0.44721359014511108) * x_j_19, y_k_4 * go_v_203, acc_i_49);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_49 = fma((scalar_t)(0.28867512941360474) * x_j_21, y_k_4 * go_v_203, acc_i_49);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_49 = fma((scalar_t)(-0.57735025882720947) * x_j_16, y_k_4 * go_v_204, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(0.18257418274879456) * x_j_17, y_k_5 * go_v_204, acc_i_49);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_49 = fma((scalar_t)(0.51639777421951294) * x_j_18, y_k_6 * go_v_204, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(0.18257418274879456) * x_j_19, y_k_7 * go_v_204, acc_i_49);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_49 = fma((scalar_t)(-0.57735025882720947) * x_j_20, y_k_8 * go_v_204, acc_i_49);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_49 = fma((scalar_t)(-0.28867512941360474) * x_j_15, y_k_4 * go_v_205, acc_i_49);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(0.44721359014511108) * x_j_17, y_k_4 * go_v_205, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_49 = fma((scalar_t)(0.35355338454246521) * x_j_16, y_k_5 * go_v_205, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_49 = fma((scalar_t)(0.18257418274879456) * x_j_18, y_k_7 * go_v_205, acc_i_49);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_49 = fma((scalar_t)(0.35355338454246521) * x_j_20, y_k_7 * go_v_205, acc_i_49);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(0.3872983455657959) * x_j_19, y_k_6 * go_v_205, acc_i_49);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(0.44721359014511108) * x_j_19, y_k_8 * go_v_205, acc_i_49);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_49 = fma((scalar_t)(-0.28867512941360474) * x_j_21, y_k_8 * go_v_205, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_49 = fma((scalar_t)(0.45643547177314758) * x_j_15, y_k_5 * go_v_206, acc_i_49);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(-0.35355338454246521) * x_j_17, y_k_5 * go_v_206, acc_i_49);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_49 = fma((scalar_t)(-0.57735025882720947) * x_j_18, y_k_8 * go_v_206, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(0.35355338454246521) * x_j_19, y_k_7 * go_v_206, acc_i_49);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_49 = fma((scalar_t)(0.45643547177314758) * x_j_21, y_k_7 * go_v_206, acc_i_49);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_49 = fma((scalar_t)(-0.45643547177314758) * x_j_16, y_k_5 * go_v_207, acc_i_49);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_49 = fma((scalar_t)(0.28867512941360474) * x_j_17, y_k_4 * go_v_207, acc_i_49);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_49 = fma((scalar_t)(-0.28867512941360474) * x_j_19, y_k_8 * go_v_207, acc_i_49);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_49 = fma((scalar_t)(0.45643547177314758) * x_j_20, y_k_7 * go_v_207, acc_i_49);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_49 = fma((scalar_t)(-0.64549720287322998) * x_j_21, y_k_6 * go_v_207, acc_i_49);
    }

    grad_w[gw_base + ((int64_t)48 << 5)] += acc_i_48;
    grad_w[gw_base + ((int64_t)49 << 5)] += acc_i_49;

    // ---- grad_w chunk 18 ----
    scalar_t acc_i_50 = scalar_t(0);
    scalar_t acc_i_51 = scalar_t(0);

    // target i = 50
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(0.35355338454246521) * x_j_4, y_k_10 * go_v_208, acc_i_50);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_50 = fma((scalar_t)(0.86602538824081421) * x_j_6, y_k_15 * go_v_208, acc_i_50);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(-0.35355338454246521) * x_j_8, y_k_14 * go_v_208, acc_i_50);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(-0.35355338454246521) * x_j_4, y_k_9 * go_v_210, acc_i_50);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(0.45643547177314758) * x_j_4, y_k_11 * go_v_210, acc_i_50);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_50 = fma((scalar_t)(0.57735025882720947) * x_j_6, y_k_14 * go_v_210, acc_i_50);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(-0.45643547177314758) * x_j_8, y_k_13 * go_v_210, acc_i_50);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(-0.35355338454246521) * x_j_8, y_k_15 * go_v_210, acc_i_50);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(-0.45643547177314758) * x_j_4, y_k_10 * go_v_212, acc_i_50);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_50 = fma((scalar_t)(0.28867512941360474) * x_j_6, y_k_13 * go_v_212, acc_i_50);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(-0.70710676908493042) * x_j_8, y_k_12 * go_v_212, acc_i_50);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(-0.45643547177314758) * x_j_8, y_k_14 * go_v_212, acc_i_50);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_214 = grad_out[go_base + ((int64_t)214 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(-0.70710676908493042) * x_j_4, y_k_13 * go_v_214, acc_i_50);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_214 = grad_out[go_base + ((int64_t)214 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(0.70710676908493042) * x_j_8, y_k_11 * go_v_214, acc_i_50);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(0.70710676908493042) * x_j_4, y_k_12 * go_v_216, acc_i_50);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(-0.45643547177314758) * x_j_4, y_k_14 * go_v_216, acc_i_50);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_50 = fma((scalar_t)(-0.28867512941360474) * x_j_6, y_k_11 * go_v_216, acc_i_50);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(0.45643547177314758) * x_j_8, y_k_10 * go_v_216, acc_i_50);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(0.45643547177314758) * x_j_4, y_k_13 * go_v_218, acc_i_50);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(-0.35355338454246521) * x_j_4, y_k_15 * go_v_218, acc_i_50);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_50 = fma((scalar_t)(-0.57735025882720947) * x_j_6, y_k_10 * go_v_218, acc_i_50);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(0.35355338454246521) * x_j_8, y_k_9 * go_v_218, acc_i_50);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(0.45643547177314758) * x_j_8, y_k_11 * go_v_218, acc_i_50);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        scalar_t x_j_4 = x_all[x_base + ((int64_t)4 << 5)];
        acc_i_50 = fma((scalar_t)(0.35355338454246521) * x_j_4, y_k_14 * go_v_220, acc_i_50);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        scalar_t x_j_6 = x_all[x_base + ((int64_t)6 << 5)];
        acc_i_50 = fma((scalar_t)(-0.86602538824081421) * x_j_6, y_k_9 * go_v_220, acc_i_50);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        scalar_t x_j_8 = x_all[x_base + ((int64_t)8 << 5)];
        acc_i_50 = fma((scalar_t)(0.35355338454246521) * x_j_8, y_k_10 * go_v_220, acc_i_50);
    }

    // target i = 51
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(0.35355338454246521) * x_j_5, y_k_10 * go_v_209, acc_i_51);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_51 = fma((scalar_t)(0.86602538824081421) * x_j_7, y_k_15 * go_v_209, acc_i_51);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(-0.35355338454246521) * x_j_9, y_k_14 * go_v_209, acc_i_51);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(-0.35355338454246521) * x_j_5, y_k_9 * go_v_211, acc_i_51);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(0.45643547177314758) * x_j_5, y_k_11 * go_v_211, acc_i_51);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_51 = fma((scalar_t)(0.57735025882720947) * x_j_7, y_k_14 * go_v_211, acc_i_51);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(-0.45643547177314758) * x_j_9, y_k_13 * go_v_211, acc_i_51);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(-0.35355338454246521) * x_j_9, y_k_15 * go_v_211, acc_i_51);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(-0.45643547177314758) * x_j_5, y_k_10 * go_v_213, acc_i_51);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_51 = fma((scalar_t)(0.28867512941360474) * x_j_7, y_k_13 * go_v_213, acc_i_51);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(-0.70710676908493042) * x_j_9, y_k_12 * go_v_213, acc_i_51);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(-0.45643547177314758) * x_j_9, y_k_14 * go_v_213, acc_i_51);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_215 = grad_out[go_base + ((int64_t)215 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(-0.70710676908493042) * x_j_5, y_k_13 * go_v_215, acc_i_51);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_215 = grad_out[go_base + ((int64_t)215 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(0.70710676908493042) * x_j_9, y_k_11 * go_v_215, acc_i_51);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(0.70710676908493042) * x_j_5, y_k_12 * go_v_217, acc_i_51);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(-0.45643547177314758) * x_j_5, y_k_14 * go_v_217, acc_i_51);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_51 = fma((scalar_t)(-0.28867512941360474) * x_j_7, y_k_11 * go_v_217, acc_i_51);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(0.45643547177314758) * x_j_9, y_k_10 * go_v_217, acc_i_51);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(0.45643547177314758) * x_j_5, y_k_13 * go_v_219, acc_i_51);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(-0.35355338454246521) * x_j_5, y_k_15 * go_v_219, acc_i_51);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_51 = fma((scalar_t)(-0.57735025882720947) * x_j_7, y_k_10 * go_v_219, acc_i_51);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(0.35355338454246521) * x_j_9, y_k_9 * go_v_219, acc_i_51);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(0.45643547177314758) * x_j_9, y_k_11 * go_v_219, acc_i_51);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        scalar_t x_j_5 = x_all[x_base + ((int64_t)5 << 5)];
        acc_i_51 = fma((scalar_t)(0.35355338454246521) * x_j_5, y_k_14 * go_v_221, acc_i_51);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        scalar_t x_j_7 = x_all[x_base + ((int64_t)7 << 5)];
        acc_i_51 = fma((scalar_t)(-0.86602538824081421) * x_j_7, y_k_9 * go_v_221, acc_i_51);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        scalar_t x_j_9 = x_all[x_base + ((int64_t)9 << 5)];
        acc_i_51 = fma((scalar_t)(0.35355338454246521) * x_j_9, y_k_10 * go_v_221, acc_i_51);
    }

    grad_w[gw_base + ((int64_t)50 << 5)] += acc_i_50;
    grad_w[gw_base + ((int64_t)51 << 5)] += acc_i_51;

    // ---- grad_w chunk 19 ----
    scalar_t acc_i_52 = scalar_t(0);
    scalar_t acc_i_53 = scalar_t(0);

    // target i = 52
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_52 = fma((scalar_t)(0.5) * x_j_10, y_k_5 * go_v_222, acc_i_52);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_52 = fma((scalar_t)(-0.5) * x_j_11, y_k_4 * go_v_222, acc_i_52);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_52 = fma((scalar_t)(0.5) * x_j_13, y_k_8 * go_v_222, acc_i_52);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_52 = fma((scalar_t)(-0.5) * x_j_14, y_k_7 * go_v_222, acc_i_52);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_223 = grad_out[go_base + ((int64_t)223 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_52 = fma((scalar_t)(0.70710676908493042) * x_j_12, y_k_8 * go_v_223, acc_i_52);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_223 = grad_out[go_base + ((int64_t)223 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_52 = fma((scalar_t)(-0.70710676908493042) * x_j_14, y_k_6 * go_v_223, acc_i_52);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_52 = fma((scalar_t)(0.3872983455657959) * x_j_10, y_k_5 * go_v_224, acc_i_52);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_52 = fma((scalar_t)(-0.3872983455657959) * x_j_11, y_k_4 * go_v_224, acc_i_52);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_52 = fma((scalar_t)(0.44721359014511108) * x_j_12, y_k_7 * go_v_224, acc_i_52);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_52 = fma((scalar_t)(0.3872983455657959) * x_j_14, y_k_7 * go_v_224, acc_i_52);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_52 = fma((scalar_t)(-0.44721359014511108) * x_j_13, y_k_6 * go_v_224, acc_i_52);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_52 = fma((scalar_t)(-0.3872983455657959) * x_j_13, y_k_8 * go_v_224, acc_i_52);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_52 = fma((scalar_t)(0.31622776389122009) * x_j_10, y_k_8 * go_v_225, acc_i_52);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_52 = fma((scalar_t)(-0.63245552778244019) * x_j_11, y_k_7 * go_v_225, acc_i_52);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_52 = fma((scalar_t)(0.63245552778244019) * x_j_13, y_k_5 * go_v_225, acc_i_52);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_52 = fma((scalar_t)(-0.31622776389122009) * x_j_14, y_k_4 * go_v_225, acc_i_52);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_52 = fma((scalar_t)(-0.3872983455657959) * x_j_10, y_k_7 * go_v_226, acc_i_52);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_52 = fma((scalar_t)(0.44721359014511108) * x_j_11, y_k_6 * go_v_226, acc_i_52);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_52 = fma((scalar_t)(-0.3872983455657959) * x_j_11, y_k_8 * go_v_226, acc_i_52);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_52 = fma((scalar_t)(-0.44721359014511108) * x_j_12, y_k_5 * go_v_226, acc_i_52);
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_52 = fma((scalar_t)(0.3872983455657959) * x_j_14, y_k_5 * go_v_226, acc_i_52);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_52 = fma((scalar_t)(0.3872983455657959) * x_j_13, y_k_4 * go_v_226, acc_i_52);
    }
    {
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_227 = grad_out[go_base + ((int64_t)227 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_52 = fma((scalar_t)(0.70710676908493042) * x_j_10, y_k_6 * go_v_227, acc_i_52);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_227 = grad_out[go_base + ((int64_t)227 << 5)];
        scalar_t x_j_12 = x_all[x_base + ((int64_t)12 << 5)];
        acc_i_52 = fma((scalar_t)(-0.70710676908493042) * x_j_12, y_k_4 * go_v_227, acc_i_52);
    }
    {
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        scalar_t x_j_10 = x_all[x_base + ((int64_t)10 << 5)];
        acc_i_52 = fma((scalar_t)(0.5) * x_j_10, y_k_7 * go_v_228, acc_i_52);
    }
    {
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        scalar_t x_j_11 = x_all[x_base + ((int64_t)11 << 5)];
        acc_i_52 = fma((scalar_t)(-0.5) * x_j_11, y_k_8 * go_v_228, acc_i_52);
    }
    {
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        scalar_t x_j_13 = x_all[x_base + ((int64_t)13 << 5)];
        acc_i_52 = fma((scalar_t)(-0.5) * x_j_13, y_k_4 * go_v_228, acc_i_52);
    }
    {
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        scalar_t x_j_14 = x_all[x_base + ((int64_t)14 << 5)];
        acc_i_52 = fma((scalar_t)(0.5) * x_j_14, y_k_5 * go_v_228, acc_i_52);
    }

    // target i = 53
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_53 = fma((scalar_t)(-0.35355338454246521) * x_j_16, y_k_1 * go_v_229, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_53 = fma((scalar_t)(0.35355338454246521) * x_j_20, y_k_3 * go_v_229, acc_i_53);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_53 = fma((scalar_t)(-0.86602538824081421) * x_j_21, y_k_2 * go_v_229, acc_i_53);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_53 = fma((scalar_t)(0.35355338454246521) * x_j_15, y_k_1 * go_v_230, acc_i_53);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_53 = fma((scalar_t)(-0.45643547177314758) * x_j_17, y_k_1 * go_v_230, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_53 = fma((scalar_t)(0.45643547177314758) * x_j_19, y_k_3 * go_v_230, acc_i_53);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_53 = fma((scalar_t)(0.35355338454246521) * x_j_21, y_k_3 * go_v_230, acc_i_53);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_53 = fma((scalar_t)(-0.57735025882720947) * x_j_20, y_k_2 * go_v_230, acc_i_53);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_53 = fma((scalar_t)(0.45643547177314758) * x_j_16, y_k_1 * go_v_231, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_53 = fma((scalar_t)(0.70710676908493042) * x_j_18, y_k_3 * go_v_231, acc_i_53);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_53 = fma((scalar_t)(0.45643547177314758) * x_j_20, y_k_3 * go_v_231, acc_i_53);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_53 = fma((scalar_t)(-0.28867512941360474) * x_j_19, y_k_2 * go_v_231, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_232 = grad_out[go_base + ((int64_t)232 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_53 = fma((scalar_t)(-0.70710676908493042) * x_j_17, y_k_3 * go_v_232, acc_i_53);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_232 = grad_out[go_base + ((int64_t)232 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_53 = fma((scalar_t)(0.70710676908493042) * x_j_19, y_k_1 * go_v_232, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_53 = fma((scalar_t)(-0.45643547177314758) * x_j_16, y_k_3 * go_v_233, acc_i_53);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_53 = fma((scalar_t)(0.28867512941360474) * x_j_17, y_k_2 * go_v_233, acc_i_53);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_53 = fma((scalar_t)(-0.70710676908493042) * x_j_18, y_k_1 * go_v_233, acc_i_53);
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_53 = fma((scalar_t)(0.45643547177314758) * x_j_20, y_k_1 * go_v_233, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_53 = fma((scalar_t)(-0.35355338454246521) * x_j_15, y_k_3 * go_v_234, acc_i_53);
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_53 = fma((scalar_t)(-0.45643547177314758) * x_j_17, y_k_3 * go_v_234, acc_i_53);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_53 = fma((scalar_t)(0.57735025882720947) * x_j_16, y_k_2 * go_v_234, acc_i_53);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_53 = fma((scalar_t)(-0.45643547177314758) * x_j_19, y_k_1 * go_v_234, acc_i_53);
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_53 = fma((scalar_t)(0.35355338454246521) * x_j_21, y_k_1 * go_v_234, acc_i_53);
    }
    {
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_53 = fma((scalar_t)(0.86602538824081421) * x_j_15, y_k_2 * go_v_235, acc_i_53);
    }
    {
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_53 = fma((scalar_t)(-0.35355338454246521) * x_j_16, y_k_3 * go_v_235, acc_i_53);
    }
    {
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_53 = fma((scalar_t)(-0.35355338454246521) * x_j_20, y_k_1 * go_v_235, acc_i_53);
    }

    grad_w[gw_base + ((int64_t)52 << 5)] += acc_i_52;
    grad_w[gw_base + ((int64_t)53 << 5)] += acc_i_53;

    // ---- grad_w chunk 20 ----
    scalar_t acc_i_54 = scalar_t(0);

    // target i = 54
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_16, y_k_11 * go_v_236, acc_i_54);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_17, y_k_10 * go_v_236, acc_i_54);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_18, y_k_15 * go_v_236, acc_i_54);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_19, y_k_14 * go_v_236, acc_i_54);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_20, y_k_13 * go_v_236, acc_i_54);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_21, y_k_12 * go_v_236, acc_i_54);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_15, y_k_11 * go_v_237, acc_i_54);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_17, y_k_9 * go_v_237, acc_i_54);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_18, y_k_14 * go_v_237, acc_i_54);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_19, y_k_15 * go_v_237, acc_i_54);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_20, y_k_12 * go_v_237, acc_i_54);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_21, y_k_13 * go_v_237, acc_i_54);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_15, y_k_10 * go_v_238, acc_i_54);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_16, y_k_9 * go_v_238, acc_i_54);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_18, y_k_13 * go_v_238, acc_i_54);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_19, y_k_12 * go_v_238, acc_i_54);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_20, y_k_15 * go_v_238, acc_i_54);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_21, y_k_14 * go_v_238, acc_i_54);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_15, y_k_15 * go_v_239, acc_i_54);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_16, y_k_14 * go_v_239, acc_i_54);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_17, y_k_13 * go_v_239, acc_i_54);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_19, y_k_11 * go_v_239, acc_i_54);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_20, y_k_10 * go_v_239, acc_i_54);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_21, y_k_9 * go_v_239, acc_i_54);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_15, y_k_14 * go_v_240, acc_i_54);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_16, y_k_15 * go_v_240, acc_i_54);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_17, y_k_12 * go_v_240, acc_i_54);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_18, y_k_11 * go_v_240, acc_i_54);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_20, y_k_9 * go_v_240, acc_i_54);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_21, y_k_10 * go_v_240, acc_i_54);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_15, y_k_13 * go_v_241, acc_i_54);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_16, y_k_12 * go_v_241, acc_i_54);
    }
    {
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_17, y_k_15 * go_v_241, acc_i_54);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_18, y_k_10 * go_v_241, acc_i_54);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_19, y_k_9 * go_v_241, acc_i_54);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        scalar_t x_j_21 = x_all[x_base + ((int64_t)21 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_21, y_k_11 * go_v_241, acc_i_54);
    }
    {
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        scalar_t x_j_15 = x_all[x_base + ((int64_t)15 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_15, y_k_12 * go_v_242, acc_i_54);
    }
    {
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        scalar_t x_j_16 = x_all[x_base + ((int64_t)16 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_16, y_k_13 * go_v_242, acc_i_54);
    }
    {
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        scalar_t x_j_17 = x_all[x_base + ((int64_t)17 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_17, y_k_14 * go_v_242, acc_i_54);
    }
    {
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        scalar_t x_j_18 = x_all[x_base + ((int64_t)18 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_18, y_k_9 * go_v_242, acc_i_54);
    }
    {
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        scalar_t x_j_19 = x_all[x_base + ((int64_t)19 << 5)];
        acc_i_54 = fma((scalar_t)(-0.40824830532073975) * x_j_19, y_k_10 * go_v_242, acc_i_54);
    }
    {
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        scalar_t x_j_20 = x_all[x_base + ((int64_t)20 << 5)];
        acc_i_54 = fma((scalar_t)(0.40824830532073975) * x_j_20, y_k_11 * go_v_242, acc_i_54);
    }

    grad_w[gw_base + ((int64_t)54 << 5)] += acc_i_54;

}
template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_bwd_u32_P777_gradx(
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

    // target j = 0
    {
        scalar_t w_i_0 = w[w_base + ((int64_t)0 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_0 = grad_out[go_base + ((int64_t)0 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_0, y_k_0 * go_v_0, acc_j_0);
    }
    {
        scalar_t w_i_8 = w[w_base + ((int64_t)8 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_8 = grad_out[go_base + ((int64_t)8 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_8, y_k_1 * go_v_8, acc_j_0);
    }
    {
        scalar_t w_i_8 = w[w_base + ((int64_t)8 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_12 = grad_out[go_base + ((int64_t)12 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_8, y_k_2 * go_v_12, acc_j_0);
    }
    {
        scalar_t w_i_8 = w[w_base + ((int64_t)8 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_16 = grad_out[go_base + ((int64_t)16 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_8, y_k_3 * go_v_16, acc_j_0);
    }
    {
        scalar_t w_i_28 = w[w_base + ((int64_t)28 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_78 = grad_out[go_base + ((int64_t)78 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_28, y_k_4 * go_v_78, acc_j_0);
    }
    {
        scalar_t w_i_28 = w[w_base + ((int64_t)28 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_82 = grad_out[go_base + ((int64_t)82 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_28, y_k_5 * go_v_82, acc_j_0);
    }
    {
        scalar_t w_i_28 = w[w_base + ((int64_t)28 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_86 = grad_out[go_base + ((int64_t)86 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_28, y_k_6 * go_v_86, acc_j_0);
    }
    {
        scalar_t w_i_28 = w[w_base + ((int64_t)28 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_90 = grad_out[go_base + ((int64_t)90 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_28, y_k_7 * go_v_90, acc_j_0);
    }
    {
        scalar_t w_i_28 = w[w_base + ((int64_t)28 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_94 = grad_out[go_base + ((int64_t)94 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_28, y_k_8 * go_v_94, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_138 = grad_out[go_base + ((int64_t)138 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_9 * go_v_138, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_142 = grad_out[go_base + ((int64_t)142 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_10 * go_v_142, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_146 = grad_out[go_base + ((int64_t)146 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_11 * go_v_146, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_150 = grad_out[go_base + ((int64_t)150 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_12 * go_v_150, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_154 = grad_out[go_base + ((int64_t)154 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_13 * go_v_154, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_158 = grad_out[go_base + ((int64_t)158 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_14 * go_v_158, acc_j_0);
    }
    {
        scalar_t w_i_40 = w[w_base + ((int64_t)40 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_162 = grad_out[go_base + ((int64_t)162 << 5)];
        acc_j_0 = fma((scalar_t)(1.0) * w_i_40, y_k_15 * go_v_162, acc_j_0);
    }

    grad_x[gx_base + ((int64_t)0 << 5)] += acc_j_0;

    // ---- grad_x chunk 1 ----
    scalar_t acc_j_1 = scalar_t(0);

    // target j = 1
    {
        scalar_t w_i_1 = w[w_base + ((int64_t)1 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_1 = grad_out[go_base + ((int64_t)1 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_1, y_k_0 * go_v_1, acc_j_1);
    }
    {
        scalar_t w_i_9 = w[w_base + ((int64_t)9 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_9 = grad_out[go_base + ((int64_t)9 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_9, y_k_1 * go_v_9, acc_j_1);
    }
    {
        scalar_t w_i_9 = w[w_base + ((int64_t)9 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_13 = grad_out[go_base + ((int64_t)13 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_9, y_k_2 * go_v_13, acc_j_1);
    }
    {
        scalar_t w_i_9 = w[w_base + ((int64_t)9 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_17 = grad_out[go_base + ((int64_t)17 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_9, y_k_3 * go_v_17, acc_j_1);
    }
    {
        scalar_t w_i_29 = w[w_base + ((int64_t)29 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_79 = grad_out[go_base + ((int64_t)79 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_29, y_k_4 * go_v_79, acc_j_1);
    }
    {
        scalar_t w_i_29 = w[w_base + ((int64_t)29 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_83 = grad_out[go_base + ((int64_t)83 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_29, y_k_5 * go_v_83, acc_j_1);
    }
    {
        scalar_t w_i_29 = w[w_base + ((int64_t)29 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_87 = grad_out[go_base + ((int64_t)87 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_29, y_k_6 * go_v_87, acc_j_1);
    }
    {
        scalar_t w_i_29 = w[w_base + ((int64_t)29 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_91 = grad_out[go_base + ((int64_t)91 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_29, y_k_7 * go_v_91, acc_j_1);
    }
    {
        scalar_t w_i_29 = w[w_base + ((int64_t)29 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_95 = grad_out[go_base + ((int64_t)95 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_29, y_k_8 * go_v_95, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_139 = grad_out[go_base + ((int64_t)139 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_9 * go_v_139, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_143 = grad_out[go_base + ((int64_t)143 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_10 * go_v_143, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_147 = grad_out[go_base + ((int64_t)147 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_11 * go_v_147, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_151 = grad_out[go_base + ((int64_t)151 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_12 * go_v_151, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_155 = grad_out[go_base + ((int64_t)155 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_13 * go_v_155, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_159 = grad_out[go_base + ((int64_t)159 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_14 * go_v_159, acc_j_1);
    }
    {
        scalar_t w_i_41 = w[w_base + ((int64_t)41 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_163 = grad_out[go_base + ((int64_t)163 << 5)];
        acc_j_1 = fma((scalar_t)(1.0) * w_i_41, y_k_15 * go_v_163, acc_j_1);
    }

    grad_x[gx_base + ((int64_t)1 << 5)] += acc_j_1;

    // ---- grad_x chunk 2 ----
    scalar_t acc_j_2 = scalar_t(0);

    // target j = 2
    {
        scalar_t w_i_2 = w[w_base + ((int64_t)2 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_2 = grad_out[go_base + ((int64_t)2 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_2, y_k_0 * go_v_2, acc_j_2);
    }
    {
        scalar_t w_i_10 = w[w_base + ((int64_t)10 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_10 = grad_out[go_base + ((int64_t)10 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_10, y_k_1 * go_v_10, acc_j_2);
    }
    {
        scalar_t w_i_10 = w[w_base + ((int64_t)10 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_14 = grad_out[go_base + ((int64_t)14 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_10, y_k_2 * go_v_14, acc_j_2);
    }
    {
        scalar_t w_i_10 = w[w_base + ((int64_t)10 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_18 = grad_out[go_base + ((int64_t)18 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_10, y_k_3 * go_v_18, acc_j_2);
    }
    {
        scalar_t w_i_30 = w[w_base + ((int64_t)30 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_80 = grad_out[go_base + ((int64_t)80 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_30, y_k_4 * go_v_80, acc_j_2);
    }
    {
        scalar_t w_i_30 = w[w_base + ((int64_t)30 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_84 = grad_out[go_base + ((int64_t)84 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_30, y_k_5 * go_v_84, acc_j_2);
    }
    {
        scalar_t w_i_30 = w[w_base + ((int64_t)30 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_88 = grad_out[go_base + ((int64_t)88 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_30, y_k_6 * go_v_88, acc_j_2);
    }
    {
        scalar_t w_i_30 = w[w_base + ((int64_t)30 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_92 = grad_out[go_base + ((int64_t)92 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_30, y_k_7 * go_v_92, acc_j_2);
    }
    {
        scalar_t w_i_30 = w[w_base + ((int64_t)30 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_96 = grad_out[go_base + ((int64_t)96 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_30, y_k_8 * go_v_96, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_140 = grad_out[go_base + ((int64_t)140 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_9 * go_v_140, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_144 = grad_out[go_base + ((int64_t)144 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_10 * go_v_144, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_148 = grad_out[go_base + ((int64_t)148 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_11 * go_v_148, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_152 = grad_out[go_base + ((int64_t)152 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_12 * go_v_152, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_156 = grad_out[go_base + ((int64_t)156 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_13 * go_v_156, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_160 = grad_out[go_base + ((int64_t)160 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_14 * go_v_160, acc_j_2);
    }
    {
        scalar_t w_i_42 = w[w_base + ((int64_t)42 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_164 = grad_out[go_base + ((int64_t)164 << 5)];
        acc_j_2 = fma((scalar_t)(1.0) * w_i_42, y_k_15 * go_v_164, acc_j_2);
    }

    grad_x[gx_base + ((int64_t)2 << 5)] += acc_j_2;

    // ---- grad_x chunk 3 ----
    scalar_t acc_j_3 = scalar_t(0);

    // target j = 3
    {
        scalar_t w_i_3 = w[w_base + ((int64_t)3 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_3 = grad_out[go_base + ((int64_t)3 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_3, y_k_0 * go_v_3, acc_j_3);
    }
    {
        scalar_t w_i_11 = w[w_base + ((int64_t)11 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_11 = grad_out[go_base + ((int64_t)11 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_11, y_k_1 * go_v_11, acc_j_3);
    }
    {
        scalar_t w_i_11 = w[w_base + ((int64_t)11 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_15 = grad_out[go_base + ((int64_t)15 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_11, y_k_2 * go_v_15, acc_j_3);
    }
    {
        scalar_t w_i_11 = w[w_base + ((int64_t)11 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_19 = grad_out[go_base + ((int64_t)19 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_11, y_k_3 * go_v_19, acc_j_3);
    }
    {
        scalar_t w_i_31 = w[w_base + ((int64_t)31 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_81 = grad_out[go_base + ((int64_t)81 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_31, y_k_4 * go_v_81, acc_j_3);
    }
    {
        scalar_t w_i_31 = w[w_base + ((int64_t)31 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_85 = grad_out[go_base + ((int64_t)85 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_31, y_k_5 * go_v_85, acc_j_3);
    }
    {
        scalar_t w_i_31 = w[w_base + ((int64_t)31 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_89 = grad_out[go_base + ((int64_t)89 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_31, y_k_6 * go_v_89, acc_j_3);
    }
    {
        scalar_t w_i_31 = w[w_base + ((int64_t)31 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_93 = grad_out[go_base + ((int64_t)93 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_31, y_k_7 * go_v_93, acc_j_3);
    }
    {
        scalar_t w_i_31 = w[w_base + ((int64_t)31 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_97 = grad_out[go_base + ((int64_t)97 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_31, y_k_8 * go_v_97, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_141 = grad_out[go_base + ((int64_t)141 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_9 * go_v_141, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_145 = grad_out[go_base + ((int64_t)145 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_10 * go_v_145, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_149 = grad_out[go_base + ((int64_t)149 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_11 * go_v_149, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_153 = grad_out[go_base + ((int64_t)153 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_12 * go_v_153, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_157 = grad_out[go_base + ((int64_t)157 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_13 * go_v_157, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_161 = grad_out[go_base + ((int64_t)161 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_14 * go_v_161, acc_j_3);
    }
    {
        scalar_t w_i_43 = w[w_base + ((int64_t)43 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_165 = grad_out[go_base + ((int64_t)165 << 5)];
        acc_j_3 = fma((scalar_t)(1.0) * w_i_43, y_k_15 * go_v_165, acc_j_3);
    }

    grad_x[gx_base + ((int64_t)3 << 5)] += acc_j_3;

    // ---- grad_x chunk 4 ----
    scalar_t acc_j_4 = scalar_t(0);

    // target j = 4
    {
        scalar_t w_i_4 = w[w_base + ((int64_t)4 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        acc_j_4 = fma((scalar_t)(0.57735025882720947) * w_i_4, y_k_1 * go_v_4, acc_j_4);
    }
    {
        scalar_t w_i_12 = w[w_base + ((int64_t)12 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_20 = grad_out[go_base + ((int64_t)20 << 5)];
        acc_j_4 = fma((scalar_t)(1.0) * w_i_12, y_k_0 * go_v_20, acc_j_4);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        acc_j_4 = fma((scalar_t)(-0.31622776389122009) * w_i_14, y_k_6 * go_v_26, acc_j_4);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        acc_j_4 = fma((scalar_t)(-0.54772257804870605) * w_i_14, y_k_8 * go_v_26, acc_j_4);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        acc_j_4 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_5 * go_v_28, acc_j_4);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        acc_j_4 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_4 * go_v_30, acc_j_4);
    }
    {
        scalar_t w_i_19 = w[w_base + ((int64_t)19 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_43 = grad_out[go_base + ((int64_t)43 << 5)];
        acc_j_4 = fma((scalar_t)(-0.70710676908493042) * w_i_19, y_k_3 * go_v_43, acc_j_4);
    }
    {
        scalar_t w_i_19 = w[w_base + ((int64_t)19 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_45 = grad_out[go_base + ((int64_t)45 << 5)];
        acc_j_4 = fma((scalar_t)(0.70710676908493042) * w_i_19, y_k_2 * go_v_45, acc_j_4);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        acc_j_4 = fma((scalar_t)(0.40824827551841736) * w_i_23, y_k_5 * go_v_53, acc_j_4);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        acc_j_4 = fma((scalar_t)(-0.40824827551841736) * w_i_23, y_k_4 * go_v_55, acc_j_4);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_57 = grad_out[go_base + ((int64_t)57 << 5)];
        acc_j_4 = fma((scalar_t)(-0.70710676908493042) * w_i_23, y_k_7 * go_v_57, acc_j_4);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        acc_j_4 = fma((scalar_t)(0.70710676908493042) * w_i_23, y_k_6 * go_v_59, acc_j_4);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        acc_j_4 = fma((scalar_t)(-0.40824827551841736) * w_i_23, y_k_8 * go_v_59, acc_j_4);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        acc_j_4 = fma((scalar_t)(0.40824827551841736) * w_i_23, y_k_7 * go_v_61, acc_j_4);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_98 = grad_out[go_base + ((int64_t)98 << 5)];
        acc_j_4 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_3 * go_v_98, acc_j_4);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_100 = grad_out[go_base + ((int64_t)100 << 5)];
        acc_j_4 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_2 * go_v_100, acc_j_4);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        acc_j_4 = fma((scalar_t)(-0.40824827551841736) * w_i_32, y_k_1 * go_v_102, acc_j_4);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_106 = grad_out[go_base + ((int64_t)106 << 5)];
        acc_j_4 = fma((scalar_t)(-0.70710676908493042) * w_i_32, y_k_1 * go_v_106, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        acc_j_4 = fma((scalar_t)(-0.15430334210395813) * w_i_34, y_k_13 * go_v_108, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        acc_j_4 = fma((scalar_t)(-0.59761428833007812) * w_i_34, y_k_15 * go_v_108, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        acc_j_4 = fma((scalar_t)(-0.37796446681022644) * w_i_34, y_k_12 * go_v_110, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        acc_j_4 = fma((scalar_t)(-0.48795005679130554) * w_i_34, y_k_14 * go_v_110, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        acc_j_4 = fma((scalar_t)(0.53452247381210327) * w_i_34, y_k_11 * go_v_112, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        acc_j_4 = fma((scalar_t)(0.48795005679130554) * w_i_34, y_k_10 * go_v_114, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        acc_j_4 = fma((scalar_t)(0.59761428833007812) * w_i_34, y_k_9 * go_v_116, acc_j_4);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        acc_j_4 = fma((scalar_t)(0.15430334210395813) * w_i_34, y_k_11 * go_v_116, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_166 = grad_out[go_base + ((int64_t)166 << 5)];
        acc_j_4 = fma((scalar_t)(0.70710676908493042) * w_i_44, y_k_8 * go_v_166, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        acc_j_4 = fma((scalar_t)(0.57735025882720947) * w_i_44, y_k_7 * go_v_168, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        acc_j_4 = fma((scalar_t)(0.63245552778244019) * w_i_44, y_k_6 * go_v_170, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        acc_j_4 = fma((scalar_t)(0.18257418274879456) * w_i_44, y_k_8 * go_v_170, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        acc_j_4 = fma((scalar_t)(-0.44721359014511108) * w_i_44, y_k_5 * go_v_172, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        acc_j_4 = fma((scalar_t)(-0.18257418274879456) * w_i_44, y_k_4 * go_v_174, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        acc_j_4 = fma((scalar_t)(-0.57735025882720947) * w_i_44, y_k_5 * go_v_176, acc_j_4);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_178 = grad_out[go_base + ((int64_t)178 << 5)];
        acc_j_4 = fma((scalar_t)(-0.70710676908493042) * w_i_44, y_k_4 * go_v_178, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        acc_j_4 = fma((scalar_t)(0.35355338454246521) * w_i_50, y_k_10 * go_v_208, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        acc_j_4 = fma((scalar_t)(-0.35355338454246521) * w_i_50, y_k_9 * go_v_210, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        acc_j_4 = fma((scalar_t)(0.45643547177314758) * w_i_50, y_k_11 * go_v_210, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        acc_j_4 = fma((scalar_t)(-0.45643547177314758) * w_i_50, y_k_10 * go_v_212, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_214 = grad_out[go_base + ((int64_t)214 << 5)];
        acc_j_4 = fma((scalar_t)(-0.70710676908493042) * w_i_50, y_k_13 * go_v_214, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        acc_j_4 = fma((scalar_t)(0.70710676908493042) * w_i_50, y_k_12 * go_v_216, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        acc_j_4 = fma((scalar_t)(-0.45643547177314758) * w_i_50, y_k_14 * go_v_216, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        acc_j_4 = fma((scalar_t)(0.45643547177314758) * w_i_50, y_k_13 * go_v_218, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        acc_j_4 = fma((scalar_t)(-0.35355338454246521) * w_i_50, y_k_15 * go_v_218, acc_j_4);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        acc_j_4 = fma((scalar_t)(0.35355338454246521) * w_i_50, y_k_14 * go_v_220, acc_j_4);
    }

    grad_x[gx_base + ((int64_t)4 << 5)] += acc_j_4;

    // ---- grad_x chunk 5 ----
    scalar_t acc_j_6 = scalar_t(0);

    // target j = 6
    {
        scalar_t w_i_4 = w[w_base + ((int64_t)4 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        acc_j_6 = fma((scalar_t)(0.57735025882720947) * w_i_4, y_k_2 * go_v_4, acc_j_6);
    }
    {
        scalar_t w_i_12 = w[w_base + ((int64_t)12 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_22 = grad_out[go_base + ((int64_t)22 << 5)];
        acc_j_6 = fma((scalar_t)(1.0) * w_i_12, y_k_0 * go_v_22, acc_j_6);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        acc_j_6 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_5 * go_v_26, acc_j_6);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        acc_j_6 = fma((scalar_t)(0.63245552778244019) * w_i_14, y_k_6 * go_v_28, acc_j_6);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        acc_j_6 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_7 * go_v_30, acc_j_6);
    }
    {
        scalar_t w_i_19 = w[w_base + ((int64_t)19 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_41 = grad_out[go_base + ((int64_t)41 << 5)];
        acc_j_6 = fma((scalar_t)(0.70710676908493042) * w_i_19, y_k_3 * go_v_41, acc_j_6);
    }
    {
        scalar_t w_i_19 = w[w_base + ((int64_t)19 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_45 = grad_out[go_base + ((int64_t)45 << 5)];
        acc_j_6 = fma((scalar_t)(-0.70710676908493042) * w_i_19, y_k_1 * go_v_45, acc_j_6);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        acc_j_6 = fma((scalar_t)(0.81649655103683472) * w_i_23, y_k_8 * go_v_53, acc_j_6);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        acc_j_6 = fma((scalar_t)(0.40824827551841736) * w_i_23, y_k_7 * go_v_55, acc_j_6);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        acc_j_6 = fma((scalar_t)(-0.40824827551841736) * w_i_23, y_k_5 * go_v_59, acc_j_6);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        acc_j_6 = fma((scalar_t)(-0.81649655103683472) * w_i_23, y_k_4 * go_v_61, acc_j_6);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_100 = grad_out[go_base + ((int64_t)100 << 5)];
        acc_j_6 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_1 * go_v_100, acc_j_6);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        acc_j_6 = fma((scalar_t)(0.81649655103683472) * w_i_32, y_k_2 * go_v_102, acc_j_6);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_104 = grad_out[go_base + ((int64_t)104 << 5)];
        acc_j_6 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_3 * go_v_104, acc_j_6);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        acc_j_6 = fma((scalar_t)(0.48795005679130554) * w_i_34, y_k_10 * go_v_108, acc_j_6);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        acc_j_6 = fma((scalar_t)(0.61721336841583252) * w_i_34, y_k_11 * go_v_110, acc_j_6);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        acc_j_6 = fma((scalar_t)(0.65465366840362549) * w_i_34, y_k_12 * go_v_112, acc_j_6);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        acc_j_6 = fma((scalar_t)(0.61721336841583252) * w_i_34, y_k_13 * go_v_114, acc_j_6);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        acc_j_6 = fma((scalar_t)(0.48795005679130554) * w_i_34, y_k_14 * go_v_116, acc_j_6);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        acc_j_6 = fma((scalar_t)(0.57735025882720947) * w_i_44, y_k_4 * go_v_168, acc_j_6);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        acc_j_6 = fma((scalar_t)(0.73029673099517822) * w_i_44, y_k_5 * go_v_170, acc_j_6);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        acc_j_6 = fma((scalar_t)(0.7745966911315918) * w_i_44, y_k_6 * go_v_172, acc_j_6);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        acc_j_6 = fma((scalar_t)(0.73029673099517822) * w_i_44, y_k_7 * go_v_174, acc_j_6);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        acc_j_6 = fma((scalar_t)(0.57735025882720947) * w_i_44, y_k_8 * go_v_176, acc_j_6);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        acc_j_6 = fma((scalar_t)(0.86602538824081421) * w_i_50, y_k_15 * go_v_208, acc_j_6);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        acc_j_6 = fma((scalar_t)(0.57735025882720947) * w_i_50, y_k_14 * go_v_210, acc_j_6);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        acc_j_6 = fma((scalar_t)(0.28867512941360474) * w_i_50, y_k_13 * go_v_212, acc_j_6);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        acc_j_6 = fma((scalar_t)(-0.28867512941360474) * w_i_50, y_k_11 * go_v_216, acc_j_6);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        acc_j_6 = fma((scalar_t)(-0.57735025882720947) * w_i_50, y_k_10 * go_v_218, acc_j_6);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        acc_j_6 = fma((scalar_t)(-0.86602538824081421) * w_i_50, y_k_9 * go_v_220, acc_j_6);
    }

    grad_x[gx_base + ((int64_t)6 << 5)] += acc_j_6;

    // ---- grad_x chunk 6 ----
    scalar_t acc_j_8 = scalar_t(0);

    // target j = 8
    {
        scalar_t w_i_4 = w[w_base + ((int64_t)4 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        acc_j_8 = fma((scalar_t)(0.57735025882720947) * w_i_4, y_k_3 * go_v_4, acc_j_8);
    }
    {
        scalar_t w_i_12 = w[w_base + ((int64_t)12 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_24 = grad_out[go_base + ((int64_t)24 << 5)];
        acc_j_8 = fma((scalar_t)(1.0) * w_i_12, y_k_0 * go_v_24, acc_j_8);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        acc_j_8 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_4 * go_v_26, acc_j_8);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        acc_j_8 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_7 * go_v_28, acc_j_8);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        acc_j_8 = fma((scalar_t)(-0.31622776389122009) * w_i_14, y_k_6 * go_v_30, acc_j_8);
    }
    {
        scalar_t w_i_14 = w[w_base + ((int64_t)14 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        acc_j_8 = fma((scalar_t)(0.54772257804870605) * w_i_14, y_k_8 * go_v_30, acc_j_8);
    }
    {
        scalar_t w_i_19 = w[w_base + ((int64_t)19 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_41 = grad_out[go_base + ((int64_t)41 << 5)];
        acc_j_8 = fma((scalar_t)(-0.70710676908493042) * w_i_19, y_k_2 * go_v_41, acc_j_8);
    }
    {
        scalar_t w_i_19 = w[w_base + ((int64_t)19 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_43 = grad_out[go_base + ((int64_t)43 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_19, y_k_1 * go_v_43, acc_j_8);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        acc_j_8 = fma((scalar_t)(-0.40824827551841736) * w_i_23, y_k_7 * go_v_53, acc_j_8);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        acc_j_8 = fma((scalar_t)(-0.70710676908493042) * w_i_23, y_k_6 * go_v_55, acc_j_8);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        acc_j_8 = fma((scalar_t)(-0.40824827551841736) * w_i_23, y_k_8 * go_v_55, acc_j_8);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_57 = grad_out[go_base + ((int64_t)57 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_23, y_k_5 * go_v_57, acc_j_8);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        acc_j_8 = fma((scalar_t)(0.40824827551841736) * w_i_23, y_k_4 * go_v_59, acc_j_8);
    }
    {
        scalar_t w_i_23 = w[w_base + ((int64_t)23 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        acc_j_8 = fma((scalar_t)(0.40824827551841736) * w_i_23, y_k_5 * go_v_61, acc_j_8);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_98 = grad_out[go_base + ((int64_t)98 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_1 * go_v_98, acc_j_8);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        acc_j_8 = fma((scalar_t)(-0.40824827551841736) * w_i_32, y_k_3 * go_v_102, acc_j_8);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_104 = grad_out[go_base + ((int64_t)104 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_2 * go_v_104, acc_j_8);
    }
    {
        scalar_t w_i_32 = w[w_base + ((int64_t)32 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_106 = grad_out[go_base + ((int64_t)106 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_32, y_k_3 * go_v_106, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        acc_j_8 = fma((scalar_t)(0.59761428833007812) * w_i_34, y_k_9 * go_v_108, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        acc_j_8 = fma((scalar_t)(-0.15430334210395813) * w_i_34, y_k_11 * go_v_108, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        acc_j_8 = fma((scalar_t)(0.48795005679130554) * w_i_34, y_k_10 * go_v_110, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        acc_j_8 = fma((scalar_t)(0.53452247381210327) * w_i_34, y_k_13 * go_v_112, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        acc_j_8 = fma((scalar_t)(-0.37796446681022644) * w_i_34, y_k_12 * go_v_114, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        acc_j_8 = fma((scalar_t)(0.48795005679130554) * w_i_34, y_k_14 * go_v_114, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        acc_j_8 = fma((scalar_t)(-0.15430334210395813) * w_i_34, y_k_13 * go_v_116, acc_j_8);
    }
    {
        scalar_t w_i_34 = w[w_base + ((int64_t)34 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        acc_j_8 = fma((scalar_t)(0.59761428833007812) * w_i_34, y_k_15 * go_v_116, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_166 = grad_out[go_base + ((int64_t)166 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_44, y_k_4 * go_v_166, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        acc_j_8 = fma((scalar_t)(0.57735025882720947) * w_i_44, y_k_5 * go_v_168, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        acc_j_8 = fma((scalar_t)(-0.18257418274879456) * w_i_44, y_k_4 * go_v_170, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        acc_j_8 = fma((scalar_t)(-0.44721359014511108) * w_i_44, y_k_7 * go_v_172, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        acc_j_8 = fma((scalar_t)(0.63245552778244019) * w_i_44, y_k_6 * go_v_174, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        acc_j_8 = fma((scalar_t)(-0.18257418274879456) * w_i_44, y_k_8 * go_v_174, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        acc_j_8 = fma((scalar_t)(0.57735025882720947) * w_i_44, y_k_7 * go_v_176, acc_j_8);
    }
    {
        scalar_t w_i_44 = w[w_base + ((int64_t)44 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_178 = grad_out[go_base + ((int64_t)178 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_44, y_k_8 * go_v_178, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        acc_j_8 = fma((scalar_t)(-0.35355338454246521) * w_i_50, y_k_14 * go_v_208, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        acc_j_8 = fma((scalar_t)(-0.45643547177314758) * w_i_50, y_k_13 * go_v_210, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        acc_j_8 = fma((scalar_t)(-0.35355338454246521) * w_i_50, y_k_15 * go_v_210, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        acc_j_8 = fma((scalar_t)(-0.70710676908493042) * w_i_50, y_k_12 * go_v_212, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        acc_j_8 = fma((scalar_t)(-0.45643547177314758) * w_i_50, y_k_14 * go_v_212, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_214 = grad_out[go_base + ((int64_t)214 << 5)];
        acc_j_8 = fma((scalar_t)(0.70710676908493042) * w_i_50, y_k_11 * go_v_214, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        acc_j_8 = fma((scalar_t)(0.45643547177314758) * w_i_50, y_k_10 * go_v_216, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        acc_j_8 = fma((scalar_t)(0.35355338454246521) * w_i_50, y_k_9 * go_v_218, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        acc_j_8 = fma((scalar_t)(0.45643547177314758) * w_i_50, y_k_11 * go_v_218, acc_j_8);
    }
    {
        scalar_t w_i_50 = w[w_base + ((int64_t)50 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        acc_j_8 = fma((scalar_t)(0.35355338454246521) * w_i_50, y_k_10 * go_v_220, acc_j_8);
    }

    grad_x[gx_base + ((int64_t)8 << 5)] += acc_j_8;

    // ---- grad_x chunk 7 ----
    scalar_t acc_j_5 = scalar_t(0);

    // target j = 5
    {
        scalar_t w_i_5 = w[w_base + ((int64_t)5 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        acc_j_5 = fma((scalar_t)(0.57735025882720947) * w_i_5, y_k_1 * go_v_5, acc_j_5);
    }
    {
        scalar_t w_i_13 = w[w_base + ((int64_t)13 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_21 = grad_out[go_base + ((int64_t)21 << 5)];
        acc_j_5 = fma((scalar_t)(1.0) * w_i_13, y_k_0 * go_v_21, acc_j_5);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        acc_j_5 = fma((scalar_t)(-0.31622776389122009) * w_i_15, y_k_6 * go_v_27, acc_j_5);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        acc_j_5 = fma((scalar_t)(-0.54772257804870605) * w_i_15, y_k_8 * go_v_27, acc_j_5);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        acc_j_5 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_5 * go_v_29, acc_j_5);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        acc_j_5 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_4 * go_v_31, acc_j_5);
    }
    {
        scalar_t w_i_20 = w[w_base + ((int64_t)20 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_44 = grad_out[go_base + ((int64_t)44 << 5)];
        acc_j_5 = fma((scalar_t)(-0.70710676908493042) * w_i_20, y_k_3 * go_v_44, acc_j_5);
    }
    {
        scalar_t w_i_20 = w[w_base + ((int64_t)20 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_46 = grad_out[go_base + ((int64_t)46 << 5)];
        acc_j_5 = fma((scalar_t)(0.70710676908493042) * w_i_20, y_k_2 * go_v_46, acc_j_5);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        acc_j_5 = fma((scalar_t)(0.40824827551841736) * w_i_24, y_k_5 * go_v_54, acc_j_5);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        acc_j_5 = fma((scalar_t)(-0.40824827551841736) * w_i_24, y_k_4 * go_v_56, acc_j_5);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_58 = grad_out[go_base + ((int64_t)58 << 5)];
        acc_j_5 = fma((scalar_t)(-0.70710676908493042) * w_i_24, y_k_7 * go_v_58, acc_j_5);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        acc_j_5 = fma((scalar_t)(0.70710676908493042) * w_i_24, y_k_6 * go_v_60, acc_j_5);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        acc_j_5 = fma((scalar_t)(-0.40824827551841736) * w_i_24, y_k_8 * go_v_60, acc_j_5);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        acc_j_5 = fma((scalar_t)(0.40824827551841736) * w_i_24, y_k_7 * go_v_62, acc_j_5);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_99 = grad_out[go_base + ((int64_t)99 << 5)];
        acc_j_5 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_3 * go_v_99, acc_j_5);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_101 = grad_out[go_base + ((int64_t)101 << 5)];
        acc_j_5 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_2 * go_v_101, acc_j_5);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        acc_j_5 = fma((scalar_t)(-0.40824827551841736) * w_i_33, y_k_1 * go_v_103, acc_j_5);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_107 = grad_out[go_base + ((int64_t)107 << 5)];
        acc_j_5 = fma((scalar_t)(-0.70710676908493042) * w_i_33, y_k_1 * go_v_107, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        acc_j_5 = fma((scalar_t)(-0.15430334210395813) * w_i_35, y_k_13 * go_v_109, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        acc_j_5 = fma((scalar_t)(-0.59761428833007812) * w_i_35, y_k_15 * go_v_109, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        acc_j_5 = fma((scalar_t)(-0.37796446681022644) * w_i_35, y_k_12 * go_v_111, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        acc_j_5 = fma((scalar_t)(-0.48795005679130554) * w_i_35, y_k_14 * go_v_111, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        acc_j_5 = fma((scalar_t)(0.53452247381210327) * w_i_35, y_k_11 * go_v_113, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        acc_j_5 = fma((scalar_t)(0.48795005679130554) * w_i_35, y_k_10 * go_v_115, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        acc_j_5 = fma((scalar_t)(0.59761428833007812) * w_i_35, y_k_9 * go_v_117, acc_j_5);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        acc_j_5 = fma((scalar_t)(0.15430334210395813) * w_i_35, y_k_11 * go_v_117, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_167 = grad_out[go_base + ((int64_t)167 << 5)];
        acc_j_5 = fma((scalar_t)(0.70710676908493042) * w_i_45, y_k_8 * go_v_167, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        acc_j_5 = fma((scalar_t)(0.57735025882720947) * w_i_45, y_k_7 * go_v_169, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        acc_j_5 = fma((scalar_t)(0.63245552778244019) * w_i_45, y_k_6 * go_v_171, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        acc_j_5 = fma((scalar_t)(0.18257418274879456) * w_i_45, y_k_8 * go_v_171, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        acc_j_5 = fma((scalar_t)(-0.44721359014511108) * w_i_45, y_k_5 * go_v_173, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        acc_j_5 = fma((scalar_t)(-0.18257418274879456) * w_i_45, y_k_4 * go_v_175, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        acc_j_5 = fma((scalar_t)(-0.57735025882720947) * w_i_45, y_k_5 * go_v_177, acc_j_5);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_179 = grad_out[go_base + ((int64_t)179 << 5)];
        acc_j_5 = fma((scalar_t)(-0.70710676908493042) * w_i_45, y_k_4 * go_v_179, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        acc_j_5 = fma((scalar_t)(0.35355338454246521) * w_i_51, y_k_10 * go_v_209, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        acc_j_5 = fma((scalar_t)(-0.35355338454246521) * w_i_51, y_k_9 * go_v_211, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        acc_j_5 = fma((scalar_t)(0.45643547177314758) * w_i_51, y_k_11 * go_v_211, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        acc_j_5 = fma((scalar_t)(-0.45643547177314758) * w_i_51, y_k_10 * go_v_213, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_215 = grad_out[go_base + ((int64_t)215 << 5)];
        acc_j_5 = fma((scalar_t)(-0.70710676908493042) * w_i_51, y_k_13 * go_v_215, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        acc_j_5 = fma((scalar_t)(0.70710676908493042) * w_i_51, y_k_12 * go_v_217, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        acc_j_5 = fma((scalar_t)(-0.45643547177314758) * w_i_51, y_k_14 * go_v_217, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        acc_j_5 = fma((scalar_t)(0.45643547177314758) * w_i_51, y_k_13 * go_v_219, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        acc_j_5 = fma((scalar_t)(-0.35355338454246521) * w_i_51, y_k_15 * go_v_219, acc_j_5);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        acc_j_5 = fma((scalar_t)(0.35355338454246521) * w_i_51, y_k_14 * go_v_221, acc_j_5);
    }

    grad_x[gx_base + ((int64_t)5 << 5)] += acc_j_5;

    // ---- grad_x chunk 8 ----
    scalar_t acc_j_7 = scalar_t(0);

    // target j = 7
    {
        scalar_t w_i_5 = w[w_base + ((int64_t)5 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        acc_j_7 = fma((scalar_t)(0.57735025882720947) * w_i_5, y_k_2 * go_v_5, acc_j_7);
    }
    {
        scalar_t w_i_13 = w[w_base + ((int64_t)13 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_23 = grad_out[go_base + ((int64_t)23 << 5)];
        acc_j_7 = fma((scalar_t)(1.0) * w_i_13, y_k_0 * go_v_23, acc_j_7);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        acc_j_7 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_5 * go_v_27, acc_j_7);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        acc_j_7 = fma((scalar_t)(0.63245552778244019) * w_i_15, y_k_6 * go_v_29, acc_j_7);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        acc_j_7 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_7 * go_v_31, acc_j_7);
    }
    {
        scalar_t w_i_20 = w[w_base + ((int64_t)20 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_42 = grad_out[go_base + ((int64_t)42 << 5)];
        acc_j_7 = fma((scalar_t)(0.70710676908493042) * w_i_20, y_k_3 * go_v_42, acc_j_7);
    }
    {
        scalar_t w_i_20 = w[w_base + ((int64_t)20 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_46 = grad_out[go_base + ((int64_t)46 << 5)];
        acc_j_7 = fma((scalar_t)(-0.70710676908493042) * w_i_20, y_k_1 * go_v_46, acc_j_7);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        acc_j_7 = fma((scalar_t)(0.81649655103683472) * w_i_24, y_k_8 * go_v_54, acc_j_7);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        acc_j_7 = fma((scalar_t)(0.40824827551841736) * w_i_24, y_k_7 * go_v_56, acc_j_7);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        acc_j_7 = fma((scalar_t)(-0.40824827551841736) * w_i_24, y_k_5 * go_v_60, acc_j_7);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        acc_j_7 = fma((scalar_t)(-0.81649655103683472) * w_i_24, y_k_4 * go_v_62, acc_j_7);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_101 = grad_out[go_base + ((int64_t)101 << 5)];
        acc_j_7 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_1 * go_v_101, acc_j_7);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        acc_j_7 = fma((scalar_t)(0.81649655103683472) * w_i_33, y_k_2 * go_v_103, acc_j_7);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_105 = grad_out[go_base + ((int64_t)105 << 5)];
        acc_j_7 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_3 * go_v_105, acc_j_7);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        acc_j_7 = fma((scalar_t)(0.48795005679130554) * w_i_35, y_k_10 * go_v_109, acc_j_7);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        acc_j_7 = fma((scalar_t)(0.61721336841583252) * w_i_35, y_k_11 * go_v_111, acc_j_7);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        acc_j_7 = fma((scalar_t)(0.65465366840362549) * w_i_35, y_k_12 * go_v_113, acc_j_7);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        acc_j_7 = fma((scalar_t)(0.61721336841583252) * w_i_35, y_k_13 * go_v_115, acc_j_7);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        acc_j_7 = fma((scalar_t)(0.48795005679130554) * w_i_35, y_k_14 * go_v_117, acc_j_7);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        acc_j_7 = fma((scalar_t)(0.57735025882720947) * w_i_45, y_k_4 * go_v_169, acc_j_7);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        acc_j_7 = fma((scalar_t)(0.73029673099517822) * w_i_45, y_k_5 * go_v_171, acc_j_7);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        acc_j_7 = fma((scalar_t)(0.7745966911315918) * w_i_45, y_k_6 * go_v_173, acc_j_7);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        acc_j_7 = fma((scalar_t)(0.73029673099517822) * w_i_45, y_k_7 * go_v_175, acc_j_7);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        acc_j_7 = fma((scalar_t)(0.57735025882720947) * w_i_45, y_k_8 * go_v_177, acc_j_7);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        acc_j_7 = fma((scalar_t)(0.86602538824081421) * w_i_51, y_k_15 * go_v_209, acc_j_7);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        acc_j_7 = fma((scalar_t)(0.57735025882720947) * w_i_51, y_k_14 * go_v_211, acc_j_7);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        acc_j_7 = fma((scalar_t)(0.28867512941360474) * w_i_51, y_k_13 * go_v_213, acc_j_7);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        acc_j_7 = fma((scalar_t)(-0.28867512941360474) * w_i_51, y_k_11 * go_v_217, acc_j_7);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        acc_j_7 = fma((scalar_t)(-0.57735025882720947) * w_i_51, y_k_10 * go_v_219, acc_j_7);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        acc_j_7 = fma((scalar_t)(-0.86602538824081421) * w_i_51, y_k_9 * go_v_221, acc_j_7);
    }

    grad_x[gx_base + ((int64_t)7 << 5)] += acc_j_7;

    // ---- grad_x chunk 9 ----
    scalar_t acc_j_9 = scalar_t(0);

    // target j = 9
    {
        scalar_t w_i_5 = w[w_base + ((int64_t)5 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        acc_j_9 = fma((scalar_t)(0.57735025882720947) * w_i_5, y_k_3 * go_v_5, acc_j_9);
    }
    {
        scalar_t w_i_13 = w[w_base + ((int64_t)13 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_25 = grad_out[go_base + ((int64_t)25 << 5)];
        acc_j_9 = fma((scalar_t)(1.0) * w_i_13, y_k_0 * go_v_25, acc_j_9);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        acc_j_9 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_4 * go_v_27, acc_j_9);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        acc_j_9 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_7 * go_v_29, acc_j_9);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        acc_j_9 = fma((scalar_t)(-0.31622776389122009) * w_i_15, y_k_6 * go_v_31, acc_j_9);
    }
    {
        scalar_t w_i_15 = w[w_base + ((int64_t)15 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        acc_j_9 = fma((scalar_t)(0.54772257804870605) * w_i_15, y_k_8 * go_v_31, acc_j_9);
    }
    {
        scalar_t w_i_20 = w[w_base + ((int64_t)20 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_42 = grad_out[go_base + ((int64_t)42 << 5)];
        acc_j_9 = fma((scalar_t)(-0.70710676908493042) * w_i_20, y_k_2 * go_v_42, acc_j_9);
    }
    {
        scalar_t w_i_20 = w[w_base + ((int64_t)20 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_44 = grad_out[go_base + ((int64_t)44 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_20, y_k_1 * go_v_44, acc_j_9);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        acc_j_9 = fma((scalar_t)(-0.40824827551841736) * w_i_24, y_k_7 * go_v_54, acc_j_9);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        acc_j_9 = fma((scalar_t)(-0.70710676908493042) * w_i_24, y_k_6 * go_v_56, acc_j_9);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        acc_j_9 = fma((scalar_t)(-0.40824827551841736) * w_i_24, y_k_8 * go_v_56, acc_j_9);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_58 = grad_out[go_base + ((int64_t)58 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_24, y_k_5 * go_v_58, acc_j_9);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        acc_j_9 = fma((scalar_t)(0.40824827551841736) * w_i_24, y_k_4 * go_v_60, acc_j_9);
    }
    {
        scalar_t w_i_24 = w[w_base + ((int64_t)24 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        acc_j_9 = fma((scalar_t)(0.40824827551841736) * w_i_24, y_k_5 * go_v_62, acc_j_9);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_99 = grad_out[go_base + ((int64_t)99 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_1 * go_v_99, acc_j_9);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        acc_j_9 = fma((scalar_t)(-0.40824827551841736) * w_i_33, y_k_3 * go_v_103, acc_j_9);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_105 = grad_out[go_base + ((int64_t)105 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_2 * go_v_105, acc_j_9);
    }
    {
        scalar_t w_i_33 = w[w_base + ((int64_t)33 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_107 = grad_out[go_base + ((int64_t)107 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_33, y_k_3 * go_v_107, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        acc_j_9 = fma((scalar_t)(0.59761428833007812) * w_i_35, y_k_9 * go_v_109, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        acc_j_9 = fma((scalar_t)(-0.15430334210395813) * w_i_35, y_k_11 * go_v_109, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        acc_j_9 = fma((scalar_t)(0.48795005679130554) * w_i_35, y_k_10 * go_v_111, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        acc_j_9 = fma((scalar_t)(0.53452247381210327) * w_i_35, y_k_13 * go_v_113, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        acc_j_9 = fma((scalar_t)(-0.37796446681022644) * w_i_35, y_k_12 * go_v_115, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        acc_j_9 = fma((scalar_t)(0.48795005679130554) * w_i_35, y_k_14 * go_v_115, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        acc_j_9 = fma((scalar_t)(-0.15430334210395813) * w_i_35, y_k_13 * go_v_117, acc_j_9);
    }
    {
        scalar_t w_i_35 = w[w_base + ((int64_t)35 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        acc_j_9 = fma((scalar_t)(0.59761428833007812) * w_i_35, y_k_15 * go_v_117, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_167 = grad_out[go_base + ((int64_t)167 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_45, y_k_4 * go_v_167, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        acc_j_9 = fma((scalar_t)(0.57735025882720947) * w_i_45, y_k_5 * go_v_169, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        acc_j_9 = fma((scalar_t)(-0.18257418274879456) * w_i_45, y_k_4 * go_v_171, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        acc_j_9 = fma((scalar_t)(-0.44721359014511108) * w_i_45, y_k_7 * go_v_173, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        acc_j_9 = fma((scalar_t)(0.63245552778244019) * w_i_45, y_k_6 * go_v_175, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        acc_j_9 = fma((scalar_t)(-0.18257418274879456) * w_i_45, y_k_8 * go_v_175, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        acc_j_9 = fma((scalar_t)(0.57735025882720947) * w_i_45, y_k_7 * go_v_177, acc_j_9);
    }
    {
        scalar_t w_i_45 = w[w_base + ((int64_t)45 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_179 = grad_out[go_base + ((int64_t)179 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_45, y_k_8 * go_v_179, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        acc_j_9 = fma((scalar_t)(-0.35355338454246521) * w_i_51, y_k_14 * go_v_209, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        acc_j_9 = fma((scalar_t)(-0.45643547177314758) * w_i_51, y_k_13 * go_v_211, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        acc_j_9 = fma((scalar_t)(-0.35355338454246521) * w_i_51, y_k_15 * go_v_211, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        acc_j_9 = fma((scalar_t)(-0.70710676908493042) * w_i_51, y_k_12 * go_v_213, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        acc_j_9 = fma((scalar_t)(-0.45643547177314758) * w_i_51, y_k_14 * go_v_213, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_215 = grad_out[go_base + ((int64_t)215 << 5)];
        acc_j_9 = fma((scalar_t)(0.70710676908493042) * w_i_51, y_k_11 * go_v_215, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        acc_j_9 = fma((scalar_t)(0.45643547177314758) * w_i_51, y_k_10 * go_v_217, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        acc_j_9 = fma((scalar_t)(0.35355338454246521) * w_i_51, y_k_9 * go_v_219, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        acc_j_9 = fma((scalar_t)(0.45643547177314758) * w_i_51, y_k_11 * go_v_219, acc_j_9);
    }
    {
        scalar_t w_i_51 = w[w_base + ((int64_t)51 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        acc_j_9 = fma((scalar_t)(0.35355338454246521) * w_i_51, y_k_10 * go_v_221, acc_j_9);
    }

    grad_x[gx_base + ((int64_t)9 << 5)] += acc_j_9;

    // ---- grad_x chunk 10 ----
    scalar_t acc_j_10 = scalar_t(0);

    // target j = 10
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_10 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_4 * go_v_6, acc_j_10);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        acc_j_10 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_3 * go_v_32, acc_j_10);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        acc_j_10 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_1 * go_v_34, acc_j_10);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_10 = fma((scalar_t)(-0.11952286213636398) * w_i_17, y_k_13 * go_v_35, acc_j_10);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_10 = fma((scalar_t)(-0.46291005611419678) * w_i_17, y_k_15 * go_v_35, acc_j_10);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        acc_j_10 = fma((scalar_t)(0.37796446681022644) * w_i_17, y_k_10 * go_v_36, acc_j_10);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_10 = fma((scalar_t)(0.46291005611419678) * w_i_17, y_k_9 * go_v_37, acc_j_10);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_10 = fma((scalar_t)(-0.11952286213636398) * w_i_17, y_k_11 * go_v_37, acc_j_10);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        acc_j_10 = fma((scalar_t)(-0.31622776389122009) * w_i_21, y_k_5 * go_v_47, acc_j_10);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        acc_j_10 = fma((scalar_t)(-0.63245552778244019) * w_i_21, y_k_8 * go_v_48, acc_j_10);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        acc_j_10 = fma((scalar_t)(0.31622776389122009) * w_i_21, y_k_7 * go_v_49, acc_j_10);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        acc_j_10 = fma((scalar_t)(0.40824827551841736) * w_i_25, y_k_1 * go_v_64, acc_j_10);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        acc_j_10 = fma((scalar_t)(-0.40824827551841736) * w_i_25, y_k_3 * go_v_66, acc_j_10);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        acc_j_10 = fma((scalar_t)(0.81649655103683472) * w_i_25, y_k_2 * go_v_67, acc_j_10);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        acc_j_10 = fma((scalar_t)(-0.42257711291313171) * w_i_26, y_k_9 * go_v_69, acc_j_10);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        acc_j_10 = fma((scalar_t)(-0.32732683420181274) * w_i_26, y_k_11 * go_v_69, acc_j_10);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        acc_j_10 = fma((scalar_t)(-0.59761428833007812) * w_i_26, y_k_14 * go_v_70, acc_j_10);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        acc_j_10 = fma((scalar_t)(0.32732683420181274) * w_i_26, y_k_13 * go_v_71, acc_j_10);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        acc_j_10 = fma((scalar_t)(-0.42257711291313171) * w_i_26, y_k_15 * go_v_71, acc_j_10);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        acc_j_10 = fma((scalar_t)(-0.26726123690605164) * w_i_26, y_k_12 * go_v_72, acc_j_10);
    }
    {
        scalar_t w_i_36 = w[w_base + ((int64_t)36 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_118 = grad_out[go_base + ((int64_t)118 << 5)];
        acc_j_10 = fma((scalar_t)(1.0) * w_i_36, y_k_0 * go_v_118, acc_j_10);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        acc_j_10 = fma((scalar_t)(-0.53452247381210327) * w_i_37, y_k_6 * go_v_123, acc_j_10);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        acc_j_10 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_7 * go_v_124, acc_j_10);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        acc_j_10 = fma((scalar_t)(-0.53452247381210327) * w_i_37, y_k_4 * go_v_125, acc_j_10);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        acc_j_10 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_5 * go_v_126, acc_j_10);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_180 = grad_out[go_base + ((int64_t)180 << 5)];
        acc_j_10 = fma((scalar_t)(0.70710676908493042) * w_i_46, y_k_3 * go_v_180, acc_j_10);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        acc_j_10 = fma((scalar_t)(0.57735025882720947) * w_i_46, y_k_2 * go_v_181, acc_j_10);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        acc_j_10 = fma((scalar_t)(-0.18257418274879456) * w_i_46, y_k_3 * go_v_182, acc_j_10);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        acc_j_10 = fma((scalar_t)(-0.18257418274879456) * w_i_46, y_k_1 * go_v_184, acc_j_10);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_186 = grad_out[go_base + ((int64_t)186 << 5)];
        acc_j_10 = fma((scalar_t)(-0.70710676908493042) * w_i_46, y_k_1 * go_v_186, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        acc_j_10 = fma((scalar_t)(-0.28867512941360474) * w_i_47, y_k_13 * go_v_187, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        acc_j_10 = fma((scalar_t)(-0.57735025882720947) * w_i_47, y_k_12 * go_v_188, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_10 = fma((scalar_t)(0.44721359014511108) * w_i_47, y_k_13 * go_v_189, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_10 = fma((scalar_t)(0.28867512941360474) * w_i_47, y_k_15 * go_v_189, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        acc_j_10 = fma((scalar_t)(-0.57735025882720947) * w_i_47, y_k_10 * go_v_190, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_10 = fma((scalar_t)(-0.28867512941360474) * w_i_47, y_k_9 * go_v_191, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_10 = fma((scalar_t)(0.44721359014511108) * w_i_47, y_k_11 * go_v_191, acc_j_10);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        acc_j_10 = fma((scalar_t)(0.28867512941360474) * w_i_47, y_k_11 * go_v_193, acc_j_10);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        acc_j_10 = fma((scalar_t)(0.5) * w_i_52, y_k_5 * go_v_222, acc_j_10);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        acc_j_10 = fma((scalar_t)(0.3872983455657959) * w_i_52, y_k_5 * go_v_224, acc_j_10);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        acc_j_10 = fma((scalar_t)(0.31622776389122009) * w_i_52, y_k_8 * go_v_225, acc_j_10);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        acc_j_10 = fma((scalar_t)(-0.3872983455657959) * w_i_52, y_k_7 * go_v_226, acc_j_10);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_227 = grad_out[go_base + ((int64_t)227 << 5)];
        acc_j_10 = fma((scalar_t)(0.70710676908493042) * w_i_52, y_k_6 * go_v_227, acc_j_10);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        acc_j_10 = fma((scalar_t)(0.5) * w_i_52, y_k_7 * go_v_228, acc_j_10);
    }

    grad_x[gx_base + ((int64_t)10 << 5)] += acc_j_10;

    // ---- grad_x chunk 11 ----
    scalar_t acc_j_11 = scalar_t(0);

    // target j = 11
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_11 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_5 * go_v_6, acc_j_11);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        acc_j_11 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_2 * go_v_32, acc_j_11);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        acc_j_11 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_1 * go_v_33, acc_j_11);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_11 = fma((scalar_t)(-0.29276999831199646) * w_i_17, y_k_12 * go_v_35, acc_j_11);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_11 = fma((scalar_t)(-0.37796446681022644) * w_i_17, y_k_14 * go_v_35, acc_j_11);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        acc_j_11 = fma((scalar_t)(0.47809144854545593) * w_i_17, y_k_11 * go_v_36, acc_j_11);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_11 = fma((scalar_t)(0.37796446681022644) * w_i_17, y_k_10 * go_v_37, acc_j_11);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        acc_j_11 = fma((scalar_t)(0.31622776389122009) * w_i_21, y_k_4 * go_v_47, acc_j_11);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        acc_j_11 = fma((scalar_t)(-0.31622776389122009) * w_i_21, y_k_7 * go_v_48, acc_j_11);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        acc_j_11 = fma((scalar_t)(0.54772257804870605) * w_i_21, y_k_6 * go_v_49, acc_j_11);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        acc_j_11 = fma((scalar_t)(0.31622776389122009) * w_i_21, y_k_8 * go_v_49, acc_j_11);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        acc_j_11 = fma((scalar_t)(-0.40824827551841736) * w_i_25, y_k_1 * go_v_63, acc_j_11);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_65 = grad_out[go_base + ((int64_t)65 << 5)];
        acc_j_11 = fma((scalar_t)(-0.70710676908493042) * w_i_25, y_k_3 * go_v_65, acc_j_11);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        acc_j_11 = fma((scalar_t)(0.40824827551841736) * w_i_25, y_k_2 * go_v_66, acc_j_11);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        acc_j_11 = fma((scalar_t)(-0.40824827551841736) * w_i_25, y_k_3 * go_v_67, acc_j_11);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        acc_j_11 = fma((scalar_t)(0.42257711291313171) * w_i_26, y_k_9 * go_v_68, acc_j_11);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        acc_j_11 = fma((scalar_t)(0.32732683420181274) * w_i_26, y_k_11 * go_v_68, acc_j_11);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        acc_j_11 = fma((scalar_t)(-0.37796446681022644) * w_i_26, y_k_13 * go_v_70, acc_j_11);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        acc_j_11 = fma((scalar_t)(0.53452247381210327) * w_i_26, y_k_12 * go_v_71, acc_j_11);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        acc_j_11 = fma((scalar_t)(0.32732683420181274) * w_i_26, y_k_13 * go_v_72, acc_j_11);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        acc_j_11 = fma((scalar_t)(0.42257711291313171) * w_i_26, y_k_15 * go_v_72, acc_j_11);
    }
    {
        scalar_t w_i_36 = w[w_base + ((int64_t)36 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_119 = grad_out[go_base + ((int64_t)119 << 5)];
        acc_j_11 = fma((scalar_t)(1.0) * w_i_36, y_k_0 * go_v_119, acc_j_11);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        acc_j_11 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_7 * go_v_123, acc_j_11);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        acc_j_11 = fma((scalar_t)(0.26726123690605164) * w_i_37, y_k_6 * go_v_124, acc_j_11);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        acc_j_11 = fma((scalar_t)(-0.46291005611419678) * w_i_37, y_k_8 * go_v_124, acc_j_11);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        acc_j_11 = fma((scalar_t)(0.26726123690605164) * w_i_37, y_k_5 * go_v_125, acc_j_11);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        acc_j_11 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_4 * go_v_126, acc_j_11);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        acc_j_11 = fma((scalar_t)(-0.46291005611419678) * w_i_37, y_k_5 * go_v_127, acc_j_11);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        acc_j_11 = fma((scalar_t)(0.57735025882720947) * w_i_46, y_k_3 * go_v_181, acc_j_11);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        acc_j_11 = fma((scalar_t)(0.73029673099517822) * w_i_46, y_k_2 * go_v_182, acc_j_11);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        acc_j_11 = fma((scalar_t)(-0.44721359014511108) * w_i_46, y_k_1 * go_v_183, acc_j_11);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        acc_j_11 = fma((scalar_t)(-0.57735025882720947) * w_i_46, y_k_1 * go_v_185, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        acc_j_11 = fma((scalar_t)(0.45643547177314758) * w_i_47, y_k_14 * go_v_187, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        acc_j_11 = fma((scalar_t)(0.35355338454246521) * w_i_47, y_k_13 * go_v_188, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        acc_j_11 = fma((scalar_t)(-0.45643547177314758) * w_i_47, y_k_15 * go_v_188, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_11 = fma((scalar_t)(0.18257418274879456) * w_i_47, y_k_12 * go_v_189, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_11 = fma((scalar_t)(-0.35355338454246521) * w_i_47, y_k_14 * go_v_189, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        acc_j_11 = fma((scalar_t)(0.18257418274879456) * w_i_47, y_k_11 * go_v_190, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_11 = fma((scalar_t)(0.35355338454246521) * w_i_47, y_k_10 * go_v_191, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        acc_j_11 = fma((scalar_t)(0.45643547177314758) * w_i_47, y_k_9 * go_v_192, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        acc_j_11 = fma((scalar_t)(-0.35355338454246521) * w_i_47, y_k_11 * go_v_192, acc_j_11);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        acc_j_11 = fma((scalar_t)(-0.45643547177314758) * w_i_47, y_k_10 * go_v_193, acc_j_11);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        acc_j_11 = fma((scalar_t)(-0.5) * w_i_52, y_k_4 * go_v_222, acc_j_11);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        acc_j_11 = fma((scalar_t)(-0.3872983455657959) * w_i_52, y_k_4 * go_v_224, acc_j_11);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        acc_j_11 = fma((scalar_t)(-0.63245552778244019) * w_i_52, y_k_7 * go_v_225, acc_j_11);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        acc_j_11 = fma((scalar_t)(0.44721359014511108) * w_i_52, y_k_6 * go_v_226, acc_j_11);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        acc_j_11 = fma((scalar_t)(-0.3872983455657959) * w_i_52, y_k_8 * go_v_226, acc_j_11);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        acc_j_11 = fma((scalar_t)(-0.5) * w_i_52, y_k_8 * go_v_228, acc_j_11);
    }

    grad_x[gx_base + ((int64_t)11 << 5)] += acc_j_11;

    // ---- grad_x chunk 12 ----
    scalar_t acc_j_12 = scalar_t(0);

    // target j = 12
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_12 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_6 * go_v_6, acc_j_12);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        acc_j_12 = fma((scalar_t)(-0.31622776389122009) * w_i_16, y_k_1 * go_v_32, acc_j_12);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        acc_j_12 = fma((scalar_t)(0.63245552778244019) * w_i_16, y_k_2 * go_v_33, acc_j_12);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        acc_j_12 = fma((scalar_t)(-0.31622776389122009) * w_i_16, y_k_3 * go_v_34, acc_j_12);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_12 = fma((scalar_t)(0.41403934359550476) * w_i_17, y_k_11 * go_v_35, acc_j_12);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        acc_j_12 = fma((scalar_t)(0.50709253549575806) * w_i_17, y_k_12 * go_v_36, acc_j_12);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_12 = fma((scalar_t)(0.41403934359550476) * w_i_17, y_k_13 * go_v_37, acc_j_12);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        acc_j_12 = fma((scalar_t)(0.54772257804870605) * w_i_21, y_k_7 * go_v_47, acc_j_12);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        acc_j_12 = fma((scalar_t)(-0.54772257804870605) * w_i_21, y_k_5 * go_v_49, acc_j_12);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        acc_j_12 = fma((scalar_t)(0.70710676908493042) * w_i_25, y_k_3 * go_v_64, acc_j_12);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        acc_j_12 = fma((scalar_t)(-0.70710676908493042) * w_i_25, y_k_1 * go_v_66, acc_j_12);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        acc_j_12 = fma((scalar_t)(0.59761428833007812) * w_i_26, y_k_14 * go_v_68, acc_j_12);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        acc_j_12 = fma((scalar_t)(0.37796446681022644) * w_i_26, y_k_13 * go_v_69, acc_j_12);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        acc_j_12 = fma((scalar_t)(-0.37796446681022644) * w_i_26, y_k_11 * go_v_71, acc_j_12);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        acc_j_12 = fma((scalar_t)(-0.59761428833007812) * w_i_26, y_k_10 * go_v_72, acc_j_12);
    }
    {
        scalar_t w_i_36 = w[w_base + ((int64_t)36 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_120 = grad_out[go_base + ((int64_t)120 << 5)];
        acc_j_12 = fma((scalar_t)(1.0) * w_i_36, y_k_0 * go_v_120, acc_j_12);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        acc_j_12 = fma((scalar_t)(-0.53452247381210327) * w_i_37, y_k_4 * go_v_123, acc_j_12);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        acc_j_12 = fma((scalar_t)(0.26726123690605164) * w_i_37, y_k_5 * go_v_124, acc_j_12);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        acc_j_12 = fma((scalar_t)(0.53452247381210327) * w_i_37, y_k_6 * go_v_125, acc_j_12);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        acc_j_12 = fma((scalar_t)(0.26726123690605164) * w_i_37, y_k_7 * go_v_126, acc_j_12);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        acc_j_12 = fma((scalar_t)(-0.53452247381210327) * w_i_37, y_k_8 * go_v_127, acc_j_12);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        acc_j_12 = fma((scalar_t)(0.63245552778244019) * w_i_46, y_k_1 * go_v_182, acc_j_12);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        acc_j_12 = fma((scalar_t)(0.7745966911315918) * w_i_46, y_k_2 * go_v_183, acc_j_12);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        acc_j_12 = fma((scalar_t)(0.63245552778244019) * w_i_46, y_k_3 * go_v_184, acc_j_12);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        acc_j_12 = fma((scalar_t)(-0.64549720287322998) * w_i_47, y_k_9 * go_v_187, acc_j_12);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_12 = fma((scalar_t)(0.3872983455657959) * w_i_47, y_k_11 * go_v_189, acc_j_12);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        acc_j_12 = fma((scalar_t)(0.51639777421951294) * w_i_47, y_k_12 * go_v_190, acc_j_12);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_12 = fma((scalar_t)(0.3872983455657959) * w_i_47, y_k_13 * go_v_191, acc_j_12);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        acc_j_12 = fma((scalar_t)(-0.64549720287322998) * w_i_47, y_k_15 * go_v_193, acc_j_12);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_223 = grad_out[go_base + ((int64_t)223 << 5)];
        acc_j_12 = fma((scalar_t)(0.70710676908493042) * w_i_52, y_k_8 * go_v_223, acc_j_12);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        acc_j_12 = fma((scalar_t)(0.44721359014511108) * w_i_52, y_k_7 * go_v_224, acc_j_12);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        acc_j_12 = fma((scalar_t)(-0.44721359014511108) * w_i_52, y_k_5 * go_v_226, acc_j_12);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_227 = grad_out[go_base + ((int64_t)227 << 5)];
        acc_j_12 = fma((scalar_t)(-0.70710676908493042) * w_i_52, y_k_4 * go_v_227, acc_j_12);
    }

    grad_x[gx_base + ((int64_t)12 << 5)] += acc_j_12;

    // ---- grad_x chunk 13 ----
    scalar_t acc_j_13 = scalar_t(0);

    // target j = 13
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_13 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_7 * go_v_6, acc_j_13);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        acc_j_13 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_3 * go_v_33, acc_j_13);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        acc_j_13 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_2 * go_v_34, acc_j_13);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_13 = fma((scalar_t)(0.37796446681022644) * w_i_17, y_k_10 * go_v_35, acc_j_13);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        acc_j_13 = fma((scalar_t)(0.47809144854545593) * w_i_17, y_k_13 * go_v_36, acc_j_13);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_13 = fma((scalar_t)(-0.29276999831199646) * w_i_17, y_k_12 * go_v_37, acc_j_13);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_13 = fma((scalar_t)(0.37796446681022644) * w_i_17, y_k_14 * go_v_37, acc_j_13);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        acc_j_13 = fma((scalar_t)(-0.54772257804870605) * w_i_21, y_k_6 * go_v_47, acc_j_13);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        acc_j_13 = fma((scalar_t)(0.31622776389122009) * w_i_21, y_k_8 * go_v_47, acc_j_13);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        acc_j_13 = fma((scalar_t)(0.31622776389122009) * w_i_21, y_k_5 * go_v_48, acc_j_13);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        acc_j_13 = fma((scalar_t)(-0.31622776389122009) * w_i_21, y_k_4 * go_v_49, acc_j_13);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        acc_j_13 = fma((scalar_t)(0.40824827551841736) * w_i_25, y_k_3 * go_v_63, acc_j_13);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        acc_j_13 = fma((scalar_t)(-0.40824827551841736) * w_i_25, y_k_2 * go_v_64, acc_j_13);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_65 = grad_out[go_base + ((int64_t)65 << 5)];
        acc_j_13 = fma((scalar_t)(0.70710676908493042) * w_i_25, y_k_1 * go_v_65, acc_j_13);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        acc_j_13 = fma((scalar_t)(-0.40824827551841736) * w_i_25, y_k_1 * go_v_67, acc_j_13);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        acc_j_13 = fma((scalar_t)(-0.32732683420181274) * w_i_26, y_k_13 * go_v_68, acc_j_13);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        acc_j_13 = fma((scalar_t)(0.42257711291313171) * w_i_26, y_k_15 * go_v_68, acc_j_13);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        acc_j_13 = fma((scalar_t)(-0.53452247381210327) * w_i_26, y_k_12 * go_v_69, acc_j_13);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        acc_j_13 = fma((scalar_t)(0.37796446681022644) * w_i_26, y_k_11 * go_v_70, acc_j_13);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        acc_j_13 = fma((scalar_t)(-0.42257711291313171) * w_i_26, y_k_9 * go_v_72, acc_j_13);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        acc_j_13 = fma((scalar_t)(0.32732683420181274) * w_i_26, y_k_11 * go_v_72, acc_j_13);
    }
    {
        scalar_t w_i_36 = w[w_base + ((int64_t)36 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_121 = grad_out[go_base + ((int64_t)121 << 5)];
        acc_j_13 = fma((scalar_t)(1.0) * w_i_36, y_k_0 * go_v_121, acc_j_13);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        acc_j_13 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_5 * go_v_123, acc_j_13);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        acc_j_13 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_4 * go_v_124, acc_j_13);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        acc_j_13 = fma((scalar_t)(0.26726123690605164) * w_i_37, y_k_7 * go_v_125, acc_j_13);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        acc_j_13 = fma((scalar_t)(0.26726123690605164) * w_i_37, y_k_6 * go_v_126, acc_j_13);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        acc_j_13 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_8 * go_v_126, acc_j_13);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        acc_j_13 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_7 * go_v_127, acc_j_13);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        acc_j_13 = fma((scalar_t)(0.57735025882720947) * w_i_46, y_k_1 * go_v_181, acc_j_13);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        acc_j_13 = fma((scalar_t)(-0.44721359014511108) * w_i_46, y_k_3 * go_v_183, acc_j_13);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        acc_j_13 = fma((scalar_t)(0.73029673099517822) * w_i_46, y_k_2 * go_v_184, acc_j_13);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        acc_j_13 = fma((scalar_t)(0.57735025882720947) * w_i_46, y_k_3 * go_v_185, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        acc_j_13 = fma((scalar_t)(0.45643547177314758) * w_i_47, y_k_10 * go_v_187, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        acc_j_13 = fma((scalar_t)(0.45643547177314758) * w_i_47, y_k_9 * go_v_188, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        acc_j_13 = fma((scalar_t)(0.35355338454246521) * w_i_47, y_k_11 * go_v_188, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_13 = fma((scalar_t)(0.35355338454246521) * w_i_47, y_k_10 * go_v_189, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        acc_j_13 = fma((scalar_t)(0.18257418274879456) * w_i_47, y_k_13 * go_v_190, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_13 = fma((scalar_t)(0.18257418274879456) * w_i_47, y_k_12 * go_v_191, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_13 = fma((scalar_t)(0.35355338454246521) * w_i_47, y_k_14 * go_v_191, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        acc_j_13 = fma((scalar_t)(0.35355338454246521) * w_i_47, y_k_13 * go_v_192, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        acc_j_13 = fma((scalar_t)(0.45643547177314758) * w_i_47, y_k_15 * go_v_192, acc_j_13);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        acc_j_13 = fma((scalar_t)(0.45643547177314758) * w_i_47, y_k_14 * go_v_193, acc_j_13);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        acc_j_13 = fma((scalar_t)(0.5) * w_i_52, y_k_8 * go_v_222, acc_j_13);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        acc_j_13 = fma((scalar_t)(-0.44721359014511108) * w_i_52, y_k_6 * go_v_224, acc_j_13);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        acc_j_13 = fma((scalar_t)(-0.3872983455657959) * w_i_52, y_k_8 * go_v_224, acc_j_13);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        acc_j_13 = fma((scalar_t)(0.63245552778244019) * w_i_52, y_k_5 * go_v_225, acc_j_13);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        acc_j_13 = fma((scalar_t)(0.3872983455657959) * w_i_52, y_k_4 * go_v_226, acc_j_13);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        acc_j_13 = fma((scalar_t)(-0.5) * w_i_52, y_k_4 * go_v_228, acc_j_13);
    }

    grad_x[gx_base + ((int64_t)13 << 5)] += acc_j_13;

    // ---- grad_x chunk 14 ----
    scalar_t acc_j_14 = scalar_t(0);

    // target j = 14
    {
        scalar_t w_i_6 = w[w_base + ((int64_t)6 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        acc_j_14 = fma((scalar_t)(0.44721359014511108) * w_i_6, y_k_8 * go_v_6, acc_j_14);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        acc_j_14 = fma((scalar_t)(-0.54772257804870605) * w_i_16, y_k_1 * go_v_32, acc_j_14);
    }
    {
        scalar_t w_i_16 = w[w_base + ((int64_t)16 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        acc_j_14 = fma((scalar_t)(0.54772257804870605) * w_i_16, y_k_3 * go_v_34, acc_j_14);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_14 = fma((scalar_t)(0.46291005611419678) * w_i_17, y_k_9 * go_v_35, acc_j_14);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        acc_j_14 = fma((scalar_t)(0.11952286213636398) * w_i_17, y_k_11 * go_v_35, acc_j_14);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        acc_j_14 = fma((scalar_t)(0.37796446681022644) * w_i_17, y_k_14 * go_v_36, acc_j_14);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_14 = fma((scalar_t)(-0.11952286213636398) * w_i_17, y_k_13 * go_v_37, acc_j_14);
    }
    {
        scalar_t w_i_17 = w[w_base + ((int64_t)17 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        acc_j_14 = fma((scalar_t)(0.46291005611419678) * w_i_17, y_k_15 * go_v_37, acc_j_14);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        acc_j_14 = fma((scalar_t)(-0.31622776389122009) * w_i_21, y_k_7 * go_v_47, acc_j_14);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        acc_j_14 = fma((scalar_t)(0.63245552778244019) * w_i_21, y_k_4 * go_v_48, acc_j_14);
    }
    {
        scalar_t w_i_21 = w[w_base + ((int64_t)21 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        acc_j_14 = fma((scalar_t)(-0.31622776389122009) * w_i_21, y_k_5 * go_v_49, acc_j_14);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        acc_j_14 = fma((scalar_t)(-0.81649655103683472) * w_i_25, y_k_2 * go_v_63, acc_j_14);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        acc_j_14 = fma((scalar_t)(0.40824827551841736) * w_i_25, y_k_3 * go_v_64, acc_j_14);
    }
    {
        scalar_t w_i_25 = w[w_base + ((int64_t)25 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        acc_j_14 = fma((scalar_t)(0.40824827551841736) * w_i_25, y_k_1 * go_v_66, acc_j_14);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        acc_j_14 = fma((scalar_t)(0.26726123690605164) * w_i_26, y_k_12 * go_v_68, acc_j_14);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        acc_j_14 = fma((scalar_t)(-0.32732683420181274) * w_i_26, y_k_13 * go_v_69, acc_j_14);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        acc_j_14 = fma((scalar_t)(-0.42257711291313171) * w_i_26, y_k_15 * go_v_69, acc_j_14);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        acc_j_14 = fma((scalar_t)(0.59761428833007812) * w_i_26, y_k_10 * go_v_70, acc_j_14);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        acc_j_14 = fma((scalar_t)(0.42257711291313171) * w_i_26, y_k_9 * go_v_71, acc_j_14);
    }
    {
        scalar_t w_i_26 = w[w_base + ((int64_t)26 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        acc_j_14 = fma((scalar_t)(-0.32732683420181274) * w_i_26, y_k_11 * go_v_71, acc_j_14);
    }
    {
        scalar_t w_i_36 = w[w_base + ((int64_t)36 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_122 = grad_out[go_base + ((int64_t)122 << 5)];
        acc_j_14 = fma((scalar_t)(1.0) * w_i_36, y_k_0 * go_v_122, acc_j_14);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        acc_j_14 = fma((scalar_t)(-0.46291005611419678) * w_i_37, y_k_5 * go_v_124, acc_j_14);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        acc_j_14 = fma((scalar_t)(-0.53452247381210327) * w_i_37, y_k_8 * go_v_125, acc_j_14);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        acc_j_14 = fma((scalar_t)(0.46291005611419678) * w_i_37, y_k_7 * go_v_126, acc_j_14);
    }
    {
        scalar_t w_i_37 = w[w_base + ((int64_t)37 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        acc_j_14 = fma((scalar_t)(-0.53452247381210327) * w_i_37, y_k_6 * go_v_127, acc_j_14);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_180 = grad_out[go_base + ((int64_t)180 << 5)];
        acc_j_14 = fma((scalar_t)(0.70710676908493042) * w_i_46, y_k_1 * go_v_180, acc_j_14);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        acc_j_14 = fma((scalar_t)(0.18257418274879456) * w_i_46, y_k_1 * go_v_182, acc_j_14);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        acc_j_14 = fma((scalar_t)(-0.18257418274879456) * w_i_46, y_k_3 * go_v_184, acc_j_14);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        acc_j_14 = fma((scalar_t)(0.57735025882720947) * w_i_46, y_k_2 * go_v_185, acc_j_14);
    }
    {
        scalar_t w_i_46 = w[w_base + ((int64_t)46 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_186 = grad_out[go_base + ((int64_t)186 << 5)];
        acc_j_14 = fma((scalar_t)(0.70710676908493042) * w_i_46, y_k_3 * go_v_186, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        acc_j_14 = fma((scalar_t)(-0.28867512941360474) * w_i_47, y_k_11 * go_v_187, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_14 = fma((scalar_t)(-0.28867512941360474) * w_i_47, y_k_9 * go_v_189, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        acc_j_14 = fma((scalar_t)(-0.44721359014511108) * w_i_47, y_k_11 * go_v_189, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        acc_j_14 = fma((scalar_t)(-0.57735025882720947) * w_i_47, y_k_14 * go_v_190, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_14 = fma((scalar_t)(0.44721359014511108) * w_i_47, y_k_13 * go_v_191, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        acc_j_14 = fma((scalar_t)(-0.28867512941360474) * w_i_47, y_k_15 * go_v_191, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        acc_j_14 = fma((scalar_t)(-0.57735025882720947) * w_i_47, y_k_12 * go_v_192, acc_j_14);
    }
    {
        scalar_t w_i_47 = w[w_base + ((int64_t)47 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        acc_j_14 = fma((scalar_t)(-0.28867512941360474) * w_i_47, y_k_13 * go_v_193, acc_j_14);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        acc_j_14 = fma((scalar_t)(-0.5) * w_i_52, y_k_7 * go_v_222, acc_j_14);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_223 = grad_out[go_base + ((int64_t)223 << 5)];
        acc_j_14 = fma((scalar_t)(-0.70710676908493042) * w_i_52, y_k_6 * go_v_223, acc_j_14);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        acc_j_14 = fma((scalar_t)(0.3872983455657959) * w_i_52, y_k_7 * go_v_224, acc_j_14);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        acc_j_14 = fma((scalar_t)(-0.31622776389122009) * w_i_52, y_k_4 * go_v_225, acc_j_14);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        acc_j_14 = fma((scalar_t)(0.3872983455657959) * w_i_52, y_k_5 * go_v_226, acc_j_14);
    }
    {
        scalar_t w_i_52 = w[w_base + ((int64_t)52 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        acc_j_14 = fma((scalar_t)(0.5) * w_i_52, y_k_5 * go_v_228, acc_j_14);
    }

    grad_x[gx_base + ((int64_t)14 << 5)] += acc_j_14;

    // ---- grad_x chunk 15 ----
    scalar_t acc_j_15 = scalar_t(0);

    // target j = 15
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_15 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_9 * go_v_7, acc_j_15);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_15 = fma((scalar_t)(0.46291005611419678) * w_i_18, y_k_8 * go_v_38, acc_j_15);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_15 = fma((scalar_t)(0.46291005611419678) * w_i_18, y_k_4 * go_v_40, acc_j_15);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_15 = fma((scalar_t)(-0.23145502805709839) * w_i_22, y_k_10 * go_v_50, acc_j_15);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        acc_j_15 = fma((scalar_t)(-0.56694668531417847) * w_i_22, y_k_15 * go_v_51, acc_j_15);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_15 = fma((scalar_t)(0.23145502805709839) * w_i_22, y_k_14 * go_v_52, acc_j_15);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        acc_j_15 = fma((scalar_t)(-0.42257711291313171) * w_i_27, y_k_5 * go_v_73, acc_j_15);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        acc_j_15 = fma((scalar_t)(0.42257711291313171) * w_i_27, y_k_4 * go_v_74, acc_j_15);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        acc_j_15 = fma((scalar_t)(-0.42257711291313171) * w_i_27, y_k_8 * go_v_76, acc_j_15);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        acc_j_15 = fma((scalar_t)(0.42257711291313171) * w_i_27, y_k_7 * go_v_77, acc_j_15);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        acc_j_15 = fma((scalar_t)(0.59761428833007812) * w_i_38, y_k_3 * go_v_128, acc_j_15);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        acc_j_15 = fma((scalar_t)(0.59761428833007812) * w_i_38, y_k_1 * go_v_132, acc_j_15);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_15 = fma((scalar_t)(-0.24397502839565277) * w_i_39, y_k_13 * go_v_133, acc_j_15);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_15 = fma((scalar_t)(0.38575837016105652) * w_i_39, y_k_14 * go_v_134, acc_j_15);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        acc_j_15 = fma((scalar_t)(-0.54554474353790283) * w_i_39, y_k_9 * go_v_135, acc_j_15);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_15 = fma((scalar_t)(0.38575837016105652) * w_i_39, y_k_10 * go_v_136, acc_j_15);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_15 = fma((scalar_t)(-0.24397502839565277) * w_i_39, y_k_11 * go_v_137, acc_j_15);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_194 = grad_out[go_base + ((int64_t)194 << 5)];
        acc_j_15 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_194, acc_j_15);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        acc_j_15 = fma((scalar_t)(-0.64549720287322998) * w_i_49, y_k_6 * go_v_201, acc_j_15);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        acc_j_15 = fma((scalar_t)(0.45643547177314758) * w_i_49, y_k_7 * go_v_202, acc_j_15);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_15 = fma((scalar_t)(-0.28867512941360474) * w_i_49, y_k_8 * go_v_203, acc_j_15);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_15 = fma((scalar_t)(-0.28867512941360474) * w_i_49, y_k_4 * go_v_205, acc_j_15);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        acc_j_15 = fma((scalar_t)(0.45643547177314758) * w_i_49, y_k_5 * go_v_206, acc_j_15);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        acc_j_15 = fma((scalar_t)(0.35355338454246521) * w_i_53, y_k_1 * go_v_230, acc_j_15);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        acc_j_15 = fma((scalar_t)(-0.35355338454246521) * w_i_53, y_k_3 * go_v_234, acc_j_15);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        acc_j_15 = fma((scalar_t)(0.86602538824081421) * w_i_53, y_k_2 * go_v_235, acc_j_15);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        acc_j_15 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_11 * go_v_237, acc_j_15);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        acc_j_15 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_10 * go_v_238, acc_j_15);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        acc_j_15 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_15 * go_v_239, acc_j_15);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        acc_j_15 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_14 * go_v_240, acc_j_15);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        acc_j_15 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_13 * go_v_241, acc_j_15);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        acc_j_15 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_12 * go_v_242, acc_j_15);
    }

    grad_x[gx_base + ((int64_t)15 << 5)] += acc_j_15;

    // ---- grad_x chunk 16 ----
    scalar_t acc_j_16 = scalar_t(0);

    // target j = 16
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_16 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_10 * go_v_7, acc_j_16);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_16 = fma((scalar_t)(0.37796446681022644) * w_i_18, y_k_7 * go_v_38, acc_j_16);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        acc_j_16 = fma((scalar_t)(0.37796446681022644) * w_i_18, y_k_4 * go_v_39, acc_j_16);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_16 = fma((scalar_t)(0.37796446681022644) * w_i_18, y_k_5 * go_v_40, acc_j_16);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_16 = fma((scalar_t)(0.23145502805709839) * w_i_22, y_k_9 * go_v_50, acc_j_16);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_16 = fma((scalar_t)(-0.29880714416503906) * w_i_22, y_k_11 * go_v_50, acc_j_16);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        acc_j_16 = fma((scalar_t)(-0.37796449661254883) * w_i_22, y_k_14 * go_v_51, acc_j_16);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_16 = fma((scalar_t)(0.29880714416503906) * w_i_22, y_k_13 * go_v_52, acc_j_16);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_16 = fma((scalar_t)(0.23145502805709839) * w_i_22, y_k_15 * go_v_52, acc_j_16);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        acc_j_16 = fma((scalar_t)(-0.59761428833007812) * w_i_27, y_k_8 * go_v_75, acc_j_16);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        acc_j_16 = fma((scalar_t)(0.59761428833007812) * w_i_27, y_k_6 * go_v_77, acc_j_16);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        acc_j_16 = fma((scalar_t)(0.48795005679130554) * w_i_38, y_k_2 * go_v_128, acc_j_16);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        acc_j_16 = fma((scalar_t)(0.48795005679130554) * w_i_38, y_k_3 * go_v_129, acc_j_16);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        acc_j_16 = fma((scalar_t)(0.48795005679130554) * w_i_38, y_k_1 * go_v_131, acc_j_16);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_16 = fma((scalar_t)(-0.48795005679130554) * w_i_39, y_k_12 * go_v_133, acc_j_16);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_16 = fma((scalar_t)(0.29880714416503906) * w_i_39, y_k_13 * go_v_134, acc_j_16);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_16 = fma((scalar_t)(-0.38575837016105652) * w_i_39, y_k_15 * go_v_134, acc_j_16);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_16 = fma((scalar_t)(0.38575837016105652) * w_i_39, y_k_9 * go_v_136, acc_j_16);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_16 = fma((scalar_t)(0.29880714416503906) * w_i_39, y_k_11 * go_v_136, acc_j_16);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_195 = grad_out[go_base + ((int64_t)195 << 5)];
        acc_j_16 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_195, acc_j_16);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        acc_j_16 = fma((scalar_t)(0.45643547177314758) * w_i_49, y_k_7 * go_v_201, acc_j_16);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_16 = fma((scalar_t)(0.35355338454246521) * w_i_49, y_k_7 * go_v_203, acc_j_16);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        acc_j_16 = fma((scalar_t)(-0.57735025882720947) * w_i_49, y_k_4 * go_v_204, acc_j_16);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_16 = fma((scalar_t)(0.35355338454246521) * w_i_49, y_k_5 * go_v_205, acc_j_16);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        acc_j_16 = fma((scalar_t)(-0.45643547177314758) * w_i_49, y_k_5 * go_v_207, acc_j_16);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        acc_j_16 = fma((scalar_t)(-0.35355338454246521) * w_i_53, y_k_1 * go_v_229, acc_j_16);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        acc_j_16 = fma((scalar_t)(0.45643547177314758) * w_i_53, y_k_1 * go_v_231, acc_j_16);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        acc_j_16 = fma((scalar_t)(-0.45643547177314758) * w_i_53, y_k_3 * go_v_233, acc_j_16);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        acc_j_16 = fma((scalar_t)(0.57735025882720947) * w_i_53, y_k_2 * go_v_234, acc_j_16);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        acc_j_16 = fma((scalar_t)(-0.35355338454246521) * w_i_53, y_k_3 * go_v_235, acc_j_16);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        acc_j_16 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_11 * go_v_236, acc_j_16);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        acc_j_16 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_9 * go_v_238, acc_j_16);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        acc_j_16 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_14 * go_v_239, acc_j_16);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        acc_j_16 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_15 * go_v_240, acc_j_16);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        acc_j_16 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_12 * go_v_241, acc_j_16);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        acc_j_16 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_13 * go_v_242, acc_j_16);
    }

    grad_x[gx_base + ((int64_t)16 << 5)] += acc_j_16;

    // ---- grad_x chunk 17 ----
    scalar_t acc_j_17 = scalar_t(0);

    // target j = 17
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_17 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_11 * go_v_7, acc_j_17);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_17 = fma((scalar_t)(0.41403934359550476) * w_i_18, y_k_6 * go_v_38, acc_j_17);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_17 = fma((scalar_t)(0.11952286213636398) * w_i_18, y_k_8 * go_v_38, acc_j_17);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        acc_j_17 = fma((scalar_t)(0.47809144854545593) * w_i_18, y_k_5 * go_v_39, acc_j_17);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_17 = fma((scalar_t)(-0.11952286213636398) * w_i_18, y_k_4 * go_v_40, acc_j_17);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_17 = fma((scalar_t)(0.29880714416503906) * w_i_22, y_k_10 * go_v_50, acc_j_17);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        acc_j_17 = fma((scalar_t)(-0.18898224830627441) * w_i_22, y_k_13 * go_v_51, acc_j_17);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_17 = fma((scalar_t)(0.46291005611419678) * w_i_22, y_k_12 * go_v_52, acc_j_17);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_17 = fma((scalar_t)(0.29880714416503906) * w_i_22, y_k_14 * go_v_52, acc_j_17);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        acc_j_17 = fma((scalar_t)(-0.32732683420181274) * w_i_27, y_k_5 * go_v_73, acc_j_17);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        acc_j_17 = fma((scalar_t)(0.32732683420181274) * w_i_27, y_k_4 * go_v_74, acc_j_17);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        acc_j_17 = fma((scalar_t)(-0.37796446681022644) * w_i_27, y_k_7 * go_v_75, acc_j_17);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        acc_j_17 = fma((scalar_t)(0.37796446681022644) * w_i_27, y_k_6 * go_v_76, acc_j_17);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        acc_j_17 = fma((scalar_t)(0.32732683420181274) * w_i_27, y_k_8 * go_v_76, acc_j_17);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        acc_j_17 = fma((scalar_t)(-0.32732683420181274) * w_i_27, y_k_7 * go_v_77, acc_j_17);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        acc_j_17 = fma((scalar_t)(-0.15430334210395813) * w_i_38, y_k_3 * go_v_128, acc_j_17);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        acc_j_17 = fma((scalar_t)(0.61721336841583252) * w_i_38, y_k_2 * go_v_129, acc_j_17);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        acc_j_17 = fma((scalar_t)(0.53452247381210327) * w_i_38, y_k_1 * go_v_130, acc_j_17);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        acc_j_17 = fma((scalar_t)(0.15430334210395813) * w_i_38, y_k_1 * go_v_132, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_17 = fma((scalar_t)(0.37796446681022644) * w_i_39, y_k_13 * go_v_133, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_17 = fma((scalar_t)(0.24397502839565277) * w_i_39, y_k_15 * go_v_133, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_17 = fma((scalar_t)(0.15430334210395813) * w_i_39, y_k_12 * go_v_134, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_17 = fma((scalar_t)(-0.29880714416503906) * w_i_39, y_k_14 * go_v_134, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        acc_j_17 = fma((scalar_t)(0.32732683420181274) * w_i_39, y_k_11 * go_v_135, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_17 = fma((scalar_t)(0.29880714416503906) * w_i_39, y_k_10 * go_v_136, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_17 = fma((scalar_t)(-0.24397502839565277) * w_i_39, y_k_9 * go_v_137, acc_j_17);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_17 = fma((scalar_t)(-0.37796446681022644) * w_i_39, y_k_11 * go_v_137, acc_j_17);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_196 = grad_out[go_base + ((int64_t)196 << 5)];
        acc_j_17 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_196, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        acc_j_17 = fma((scalar_t)(-0.28867512941360474) * w_i_49, y_k_8 * go_v_201, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        acc_j_17 = fma((scalar_t)(0.35355338454246521) * w_i_49, y_k_7 * go_v_202, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_17 = fma((scalar_t)(0.3872983455657959) * w_i_49, y_k_6 * go_v_203, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_17 = fma((scalar_t)(-0.44721359014511108) * w_i_49, y_k_8 * go_v_203, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        acc_j_17 = fma((scalar_t)(0.18257418274879456) * w_i_49, y_k_5 * go_v_204, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_17 = fma((scalar_t)(0.44721359014511108) * w_i_49, y_k_4 * go_v_205, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        acc_j_17 = fma((scalar_t)(-0.35355338454246521) * w_i_49, y_k_5 * go_v_206, acc_j_17);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        acc_j_17 = fma((scalar_t)(0.28867512941360474) * w_i_49, y_k_4 * go_v_207, acc_j_17);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        acc_j_17 = fma((scalar_t)(-0.45643547177314758) * w_i_53, y_k_1 * go_v_230, acc_j_17);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_232 = grad_out[go_base + ((int64_t)232 << 5)];
        acc_j_17 = fma((scalar_t)(-0.70710676908493042) * w_i_53, y_k_3 * go_v_232, acc_j_17);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        acc_j_17 = fma((scalar_t)(0.28867512941360474) * w_i_53, y_k_2 * go_v_233, acc_j_17);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        acc_j_17 = fma((scalar_t)(-0.45643547177314758) * w_i_53, y_k_3 * go_v_234, acc_j_17);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        acc_j_17 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_10 * go_v_236, acc_j_17);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        acc_j_17 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_9 * go_v_237, acc_j_17);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        acc_j_17 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_13 * go_v_239, acc_j_17);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        acc_j_17 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_12 * go_v_240, acc_j_17);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        acc_j_17 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_15 * go_v_241, acc_j_17);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        acc_j_17 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_14 * go_v_242, acc_j_17);
    }

    grad_x[gx_base + ((int64_t)17 << 5)] += acc_j_17;

    // ---- grad_x chunk 18 ----
    scalar_t acc_j_18 = scalar_t(0);

    // target j = 18
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_18 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_12 * go_v_7, acc_j_18);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_18 = fma((scalar_t)(-0.29276999831199646) * w_i_18, y_k_5 * go_v_38, acc_j_18);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        acc_j_18 = fma((scalar_t)(0.50709253549575806) * w_i_18, y_k_6 * go_v_39, acc_j_18);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_18 = fma((scalar_t)(-0.29276999831199646) * w_i_18, y_k_7 * go_v_40, acc_j_18);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_18 = fma((scalar_t)(0.46291005611419678) * w_i_22, y_k_13 * go_v_50, acc_j_18);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_18 = fma((scalar_t)(-0.46291005611419678) * w_i_22, y_k_11 * go_v_52, acc_j_18);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        acc_j_18 = fma((scalar_t)(-0.26726123690605164) * w_i_27, y_k_8 * go_v_73, acc_j_18);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        acc_j_18 = fma((scalar_t)(0.53452247381210327) * w_i_27, y_k_7 * go_v_74, acc_j_18);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        acc_j_18 = fma((scalar_t)(-0.53452247381210327) * w_i_27, y_k_5 * go_v_76, acc_j_18);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        acc_j_18 = fma((scalar_t)(0.26726123690605164) * w_i_27, y_k_4 * go_v_77, acc_j_18);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        acc_j_18 = fma((scalar_t)(-0.37796446681022644) * w_i_38, y_k_1 * go_v_129, acc_j_18);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        acc_j_18 = fma((scalar_t)(0.65465366840362549) * w_i_38, y_k_2 * go_v_130, acc_j_18);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        acc_j_18 = fma((scalar_t)(-0.37796446681022644) * w_i_38, y_k_3 * go_v_131, acc_j_18);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_18 = fma((scalar_t)(-0.48795005679130554) * w_i_39, y_k_10 * go_v_133, acc_j_18);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_18 = fma((scalar_t)(0.15430334210395813) * w_i_39, y_k_11 * go_v_134, acc_j_18);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        acc_j_18 = fma((scalar_t)(0.43643578886985779) * w_i_39, y_k_12 * go_v_135, acc_j_18);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_18 = fma((scalar_t)(0.15430334210395813) * w_i_39, y_k_13 * go_v_136, acc_j_18);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_18 = fma((scalar_t)(-0.48795005679130554) * w_i_39, y_k_14 * go_v_137, acc_j_18);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_197 = grad_out[go_base + ((int64_t)197 << 5)];
        acc_j_18 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_197, acc_j_18);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        acc_j_18 = fma((scalar_t)(-0.57735025882720947) * w_i_49, y_k_4 * go_v_202, acc_j_18);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_18 = fma((scalar_t)(0.18257418274879456) * w_i_49, y_k_5 * go_v_203, acc_j_18);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        acc_j_18 = fma((scalar_t)(0.51639777421951294) * w_i_49, y_k_6 * go_v_204, acc_j_18);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_18 = fma((scalar_t)(0.18257418274879456) * w_i_49, y_k_7 * go_v_205, acc_j_18);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        acc_j_18 = fma((scalar_t)(-0.57735025882720947) * w_i_49, y_k_8 * go_v_206, acc_j_18);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        acc_j_18 = fma((scalar_t)(0.70710676908493042) * w_i_53, y_k_3 * go_v_231, acc_j_18);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        acc_j_18 = fma((scalar_t)(-0.70710676908493042) * w_i_53, y_k_1 * go_v_233, acc_j_18);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        acc_j_18 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_15 * go_v_236, acc_j_18);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        acc_j_18 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_14 * go_v_237, acc_j_18);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        acc_j_18 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_13 * go_v_238, acc_j_18);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        acc_j_18 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_11 * go_v_240, acc_j_18);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        acc_j_18 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_10 * go_v_241, acc_j_18);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        acc_j_18 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_9 * go_v_242, acc_j_18);
    }

    grad_x[gx_base + ((int64_t)18 << 5)] += acc_j_18;

    // ---- grad_x chunk 19 ----
    scalar_t acc_j_19 = scalar_t(0);

    // target j = 19
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_19 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_13 * go_v_7, acc_j_19);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_19 = fma((scalar_t)(-0.11952286213636398) * w_i_18, y_k_4 * go_v_38, acc_j_19);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        acc_j_19 = fma((scalar_t)(0.47809144854545593) * w_i_18, y_k_7 * go_v_39, acc_j_19);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_19 = fma((scalar_t)(0.41403934359550476) * w_i_18, y_k_6 * go_v_40, acc_j_19);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_19 = fma((scalar_t)(-0.11952286213636398) * w_i_18, y_k_8 * go_v_40, acc_j_19);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_19 = fma((scalar_t)(-0.46291005611419678) * w_i_22, y_k_12 * go_v_50, acc_j_19);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_19 = fma((scalar_t)(0.29880714416503906) * w_i_22, y_k_14 * go_v_50, acc_j_19);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        acc_j_19 = fma((scalar_t)(0.18898224830627441) * w_i_22, y_k_11 * go_v_51, acc_j_19);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_19 = fma((scalar_t)(-0.29880714416503906) * w_i_22, y_k_10 * go_v_52, acc_j_19);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        acc_j_19 = fma((scalar_t)(0.32732683420181274) * w_i_27, y_k_7 * go_v_73, acc_j_19);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        acc_j_19 = fma((scalar_t)(-0.37796446681022644) * w_i_27, y_k_6 * go_v_74, acc_j_19);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        acc_j_19 = fma((scalar_t)(0.32732683420181274) * w_i_27, y_k_8 * go_v_74, acc_j_19);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        acc_j_19 = fma((scalar_t)(0.37796446681022644) * w_i_27, y_k_5 * go_v_75, acc_j_19);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        acc_j_19 = fma((scalar_t)(-0.32732683420181274) * w_i_27, y_k_4 * go_v_76, acc_j_19);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        acc_j_19 = fma((scalar_t)(-0.32732683420181274) * w_i_27, y_k_5 * go_v_77, acc_j_19);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        acc_j_19 = fma((scalar_t)(-0.15430334210395813) * w_i_38, y_k_1 * go_v_128, acc_j_19);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        acc_j_19 = fma((scalar_t)(0.53452247381210327) * w_i_38, y_k_3 * go_v_130, acc_j_19);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        acc_j_19 = fma((scalar_t)(0.61721336841583252) * w_i_38, y_k_2 * go_v_131, acc_j_19);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        acc_j_19 = fma((scalar_t)(-0.15430334210395813) * w_i_38, y_k_3 * go_v_132, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_19 = fma((scalar_t)(-0.24397502839565277) * w_i_39, y_k_9 * go_v_133, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_19 = fma((scalar_t)(0.37796446681022644) * w_i_39, y_k_11 * go_v_133, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_19 = fma((scalar_t)(0.29880714416503906) * w_i_39, y_k_10 * go_v_134, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        acc_j_19 = fma((scalar_t)(0.32732683420181274) * w_i_39, y_k_13 * go_v_135, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_19 = fma((scalar_t)(0.15430334210395813) * w_i_39, y_k_12 * go_v_136, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_19 = fma((scalar_t)(0.29880714416503906) * w_i_39, y_k_14 * go_v_136, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_19 = fma((scalar_t)(0.37796446681022644) * w_i_39, y_k_13 * go_v_137, acc_j_19);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_19 = fma((scalar_t)(-0.24397502839565277) * w_i_39, y_k_15 * go_v_137, acc_j_19);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_198 = grad_out[go_base + ((int64_t)198 << 5)];
        acc_j_19 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_198, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        acc_j_19 = fma((scalar_t)(-0.28867512941360474) * w_i_49, y_k_4 * go_v_201, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        acc_j_19 = fma((scalar_t)(0.35355338454246521) * w_i_49, y_k_5 * go_v_202, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_19 = fma((scalar_t)(0.44721359014511108) * w_i_49, y_k_4 * go_v_203, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        acc_j_19 = fma((scalar_t)(0.18257418274879456) * w_i_49, y_k_7 * go_v_204, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_19 = fma((scalar_t)(0.3872983455657959) * w_i_49, y_k_6 * go_v_205, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_19 = fma((scalar_t)(0.44721359014511108) * w_i_49, y_k_8 * go_v_205, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        acc_j_19 = fma((scalar_t)(0.35355338454246521) * w_i_49, y_k_7 * go_v_206, acc_j_19);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        acc_j_19 = fma((scalar_t)(-0.28867512941360474) * w_i_49, y_k_8 * go_v_207, acc_j_19);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        acc_j_19 = fma((scalar_t)(0.45643547177314758) * w_i_53, y_k_3 * go_v_230, acc_j_19);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        acc_j_19 = fma((scalar_t)(-0.28867512941360474) * w_i_53, y_k_2 * go_v_231, acc_j_19);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_232 = grad_out[go_base + ((int64_t)232 << 5)];
        acc_j_19 = fma((scalar_t)(0.70710676908493042) * w_i_53, y_k_1 * go_v_232, acc_j_19);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        acc_j_19 = fma((scalar_t)(-0.45643547177314758) * w_i_53, y_k_1 * go_v_234, acc_j_19);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        acc_j_19 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_14 * go_v_236, acc_j_19);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        acc_j_19 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_15 * go_v_237, acc_j_19);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        acc_j_19 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_12 * go_v_238, acc_j_19);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        acc_j_19 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_11 * go_v_239, acc_j_19);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        acc_j_19 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_9 * go_v_241, acc_j_19);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        acc_j_19 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_10 * go_v_242, acc_j_19);
    }

    grad_x[gx_base + ((int64_t)19 << 5)] += acc_j_19;

    // ---- grad_x chunk 20 ----
    scalar_t acc_j_20 = scalar_t(0);

    // target j = 20
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_20 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_14 * go_v_7, acc_j_20);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_20 = fma((scalar_t)(-0.37796446681022644) * w_i_18, y_k_5 * go_v_38, acc_j_20);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        acc_j_20 = fma((scalar_t)(0.37796446681022644) * w_i_18, y_k_8 * go_v_39, acc_j_20);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_20 = fma((scalar_t)(0.37796446681022644) * w_i_18, y_k_7 * go_v_40, acc_j_20);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_20 = fma((scalar_t)(-0.29880714416503906) * w_i_22, y_k_13 * go_v_50, acc_j_20);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_20 = fma((scalar_t)(0.23145502805709839) * w_i_22, y_k_15 * go_v_50, acc_j_20);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        acc_j_20 = fma((scalar_t)(0.37796449661254883) * w_i_22, y_k_10 * go_v_51, acc_j_20);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_20 = fma((scalar_t)(-0.23145502805709839) * w_i_22, y_k_9 * go_v_52, acc_j_20);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_20 = fma((scalar_t)(-0.29880714416503906) * w_i_22, y_k_11 * go_v_52, acc_j_20);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        acc_j_20 = fma((scalar_t)(-0.59761428833007812) * w_i_27, y_k_6 * go_v_73, acc_j_20);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        acc_j_20 = fma((scalar_t)(0.59761428833007812) * w_i_27, y_k_4 * go_v_75, acc_j_20);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        acc_j_20 = fma((scalar_t)(-0.48795005679130554) * w_i_38, y_k_1 * go_v_129, acc_j_20);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        acc_j_20 = fma((scalar_t)(0.48795005679130554) * w_i_38, y_k_3 * go_v_131, acc_j_20);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        acc_j_20 = fma((scalar_t)(0.48795005679130554) * w_i_38, y_k_2 * go_v_132, acc_j_20);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_20 = fma((scalar_t)(0.38575837016105652) * w_i_39, y_k_9 * go_v_134, acc_j_20);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_20 = fma((scalar_t)(-0.29880714416503906) * w_i_39, y_k_11 * go_v_134, acc_j_20);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_20 = fma((scalar_t)(0.29880714416503906) * w_i_39, y_k_13 * go_v_136, acc_j_20);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_20 = fma((scalar_t)(0.38575837016105652) * w_i_39, y_k_15 * go_v_136, acc_j_20);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_20 = fma((scalar_t)(-0.48795005679130554) * w_i_39, y_k_12 * go_v_137, acc_j_20);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_199 = grad_out[go_base + ((int64_t)199 << 5)];
        acc_j_20 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_199, acc_j_20);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        acc_j_20 = fma((scalar_t)(0.45643547177314758) * w_i_49, y_k_5 * go_v_201, acc_j_20);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_20 = fma((scalar_t)(-0.35355338454246521) * w_i_49, y_k_5 * go_v_203, acc_j_20);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        acc_j_20 = fma((scalar_t)(-0.57735025882720947) * w_i_49, y_k_8 * go_v_204, acc_j_20);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_20 = fma((scalar_t)(0.35355338454246521) * w_i_49, y_k_7 * go_v_205, acc_j_20);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        acc_j_20 = fma((scalar_t)(0.45643547177314758) * w_i_49, y_k_7 * go_v_207, acc_j_20);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        acc_j_20 = fma((scalar_t)(0.35355338454246521) * w_i_53, y_k_3 * go_v_229, acc_j_20);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        acc_j_20 = fma((scalar_t)(-0.57735025882720947) * w_i_53, y_k_2 * go_v_230, acc_j_20);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        acc_j_20 = fma((scalar_t)(0.45643547177314758) * w_i_53, y_k_3 * go_v_231, acc_j_20);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        acc_j_20 = fma((scalar_t)(0.45643547177314758) * w_i_53, y_k_1 * go_v_233, acc_j_20);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        acc_j_20 = fma((scalar_t)(-0.35355338454246521) * w_i_53, y_k_1 * go_v_235, acc_j_20);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        acc_j_20 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_13 * go_v_236, acc_j_20);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        acc_j_20 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_12 * go_v_237, acc_j_20);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        acc_j_20 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_15 * go_v_238, acc_j_20);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        acc_j_20 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_10 * go_v_239, acc_j_20);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        acc_j_20 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_9 * go_v_240, acc_j_20);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        acc_j_20 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_11 * go_v_242, acc_j_20);
    }

    grad_x[gx_base + ((int64_t)20 << 5)] += acc_j_20;

    // ---- grad_x chunk 21 ----
    scalar_t acc_j_21 = scalar_t(0);

    // target j = 21
    {
        scalar_t w_i_7 = w[w_base + ((int64_t)7 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        acc_j_21 = fma((scalar_t)(0.37796446681022644) * w_i_7, y_k_15 * go_v_7, acc_j_21);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        acc_j_21 = fma((scalar_t)(-0.46291005611419678) * w_i_18, y_k_4 * go_v_38, acc_j_21);
    }
    {
        scalar_t w_i_18 = w[w_base + ((int64_t)18 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        acc_j_21 = fma((scalar_t)(0.46291005611419678) * w_i_18, y_k_8 * go_v_40, acc_j_21);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        acc_j_21 = fma((scalar_t)(-0.23145502805709839) * w_i_22, y_k_14 * go_v_50, acc_j_21);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        acc_j_21 = fma((scalar_t)(0.56694668531417847) * w_i_22, y_k_9 * go_v_51, acc_j_21);
    }
    {
        scalar_t w_i_22 = w[w_base + ((int64_t)22 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        acc_j_21 = fma((scalar_t)(-0.23145502805709839) * w_i_22, y_k_10 * go_v_52, acc_j_21);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        acc_j_21 = fma((scalar_t)(-0.42257711291313171) * w_i_27, y_k_7 * go_v_73, acc_j_21);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        acc_j_21 = fma((scalar_t)(0.42257711291313171) * w_i_27, y_k_8 * go_v_74, acc_j_21);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        acc_j_21 = fma((scalar_t)(0.42257711291313171) * w_i_27, y_k_4 * go_v_76, acc_j_21);
    }
    {
        scalar_t w_i_27 = w[w_base + ((int64_t)27 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        acc_j_21 = fma((scalar_t)(-0.42257711291313171) * w_i_27, y_k_5 * go_v_77, acc_j_21);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        acc_j_21 = fma((scalar_t)(-0.59761428833007812) * w_i_38, y_k_1 * go_v_128, acc_j_21);
    }
    {
        scalar_t w_i_38 = w[w_base + ((int64_t)38 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        acc_j_21 = fma((scalar_t)(0.59761428833007812) * w_i_38, y_k_3 * go_v_132, acc_j_21);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        acc_j_21 = fma((scalar_t)(0.24397502839565277) * w_i_39, y_k_11 * go_v_133, acc_j_21);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        acc_j_21 = fma((scalar_t)(-0.38575837016105652) * w_i_39, y_k_10 * go_v_134, acc_j_21);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_15 = y[y_base + 15];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        acc_j_21 = fma((scalar_t)(-0.54554474353790283) * w_i_39, y_k_15 * go_v_135, acc_j_21);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        acc_j_21 = fma((scalar_t)(0.38575837016105652) * w_i_39, y_k_14 * go_v_136, acc_j_21);
    }
    {
        scalar_t w_i_39 = w[w_base + ((int64_t)39 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        acc_j_21 = fma((scalar_t)(-0.24397502839565277) * w_i_39, y_k_13 * go_v_137, acc_j_21);
    }
    {
        scalar_t w_i_48 = w[w_base + ((int64_t)48 << 5)];
        scalar_t y_k_0 = y[y_base + 0];
        scalar_t go_v_200 = grad_out[go_base + ((int64_t)200 << 5)];
        acc_j_21 = fma((scalar_t)(1.0) * w_i_48, y_k_0 * go_v_200, acc_j_21);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_5 = y[y_base + 5];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        acc_j_21 = fma((scalar_t)(-0.45643547177314758) * w_i_49, y_k_5 * go_v_202, acc_j_21);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_4 = y[y_base + 4];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        acc_j_21 = fma((scalar_t)(0.28867512941360474) * w_i_49, y_k_4 * go_v_203, acc_j_21);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_8 = y[y_base + 8];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        acc_j_21 = fma((scalar_t)(-0.28867512941360474) * w_i_49, y_k_8 * go_v_205, acc_j_21);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_7 = y[y_base + 7];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        acc_j_21 = fma((scalar_t)(0.45643547177314758) * w_i_49, y_k_7 * go_v_206, acc_j_21);
    }
    {
        scalar_t w_i_49 = w[w_base + ((int64_t)49 << 5)];
        scalar_t y_k_6 = y[y_base + 6];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        acc_j_21 = fma((scalar_t)(-0.64549720287322998) * w_i_49, y_k_6 * go_v_207, acc_j_21);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_2 = y[y_base + 2];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        acc_j_21 = fma((scalar_t)(-0.86602538824081421) * w_i_53, y_k_2 * go_v_229, acc_j_21);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_3 = y[y_base + 3];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        acc_j_21 = fma((scalar_t)(0.35355338454246521) * w_i_53, y_k_3 * go_v_230, acc_j_21);
    }
    {
        scalar_t w_i_53 = w[w_base + ((int64_t)53 << 5)];
        scalar_t y_k_1 = y[y_base + 1];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        acc_j_21 = fma((scalar_t)(0.35355338454246521) * w_i_53, y_k_1 * go_v_234, acc_j_21);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_12 = y[y_base + 12];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        acc_j_21 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_12 * go_v_236, acc_j_21);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_13 = y[y_base + 13];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        acc_j_21 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_13 * go_v_237, acc_j_21);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_14 = y[y_base + 14];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        acc_j_21 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_14 * go_v_238, acc_j_21);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_9 = y[y_base + 9];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        acc_j_21 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_9 * go_v_239, acc_j_21);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_10 = y[y_base + 10];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        acc_j_21 = fma((scalar_t)(0.40824830532073975) * w_i_54, y_k_10 * go_v_240, acc_j_21);
    }
    {
        scalar_t w_i_54 = w[w_base + ((int64_t)54 << 5)];
        scalar_t y_k_11 = y[y_base + 11];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        acc_j_21 = fma((scalar_t)(-0.40824830532073975) * w_i_54, y_k_11 * go_v_241, acc_j_21);
    }

    grad_x[gx_base + ((int64_t)21 << 5)] += acc_j_21;

}
template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_bwd_u32_P777_grady(
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
    {
        scalar_t w_i_12  = w[w_base + ((int64_t)12 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_20 = grad_out[go_base + ((int64_t)20 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_12, x_j_4 * go_v_20, local_k_0);
    }
    {
        scalar_t w_i_13  = w[w_base + ((int64_t)13 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_21 = grad_out[go_base + ((int64_t)21 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_13, x_j_5 * go_v_21, local_k_0);
    }
    {
        scalar_t w_i_12  = w[w_base + ((int64_t)12 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_22 = grad_out[go_base + ((int64_t)22 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_12, x_j_6 * go_v_22, local_k_0);
    }
    {
        scalar_t w_i_13  = w[w_base + ((int64_t)13 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_23 = grad_out[go_base + ((int64_t)23 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_13, x_j_7 * go_v_23, local_k_0);
    }
    {
        scalar_t w_i_12  = w[w_base + ((int64_t)12 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_24 = grad_out[go_base + ((int64_t)24 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_12, x_j_8 * go_v_24, local_k_0);
    }
    {
        scalar_t w_i_13  = w[w_base + ((int64_t)13 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_25 = grad_out[go_base + ((int64_t)25 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_13, x_j_9 * go_v_25, local_k_0);
    }
    {
        scalar_t w_i_36  = w[w_base + ((int64_t)36 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_118 = grad_out[go_base + ((int64_t)118 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_36, x_j_10 * go_v_118, local_k_0);
    }
    {
        scalar_t w_i_36  = w[w_base + ((int64_t)36 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_119 = grad_out[go_base + ((int64_t)119 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_36, x_j_11 * go_v_119, local_k_0);
    }
    {
        scalar_t w_i_36  = w[w_base + ((int64_t)36 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_120 = grad_out[go_base + ((int64_t)120 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_36, x_j_12 * go_v_120, local_k_0);
    }
    {
        scalar_t w_i_36  = w[w_base + ((int64_t)36 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_121 = grad_out[go_base + ((int64_t)121 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_36, x_j_13 * go_v_121, local_k_0);
    }
    {
        scalar_t w_i_36  = w[w_base + ((int64_t)36 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_122 = grad_out[go_base + ((int64_t)122 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_36, x_j_14 * go_v_122, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_194 = grad_out[go_base + ((int64_t)194 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_15 * go_v_194, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_195 = grad_out[go_base + ((int64_t)195 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_16 * go_v_195, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_196 = grad_out[go_base + ((int64_t)196 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_17 * go_v_196, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_197 = grad_out[go_base + ((int64_t)197 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_18 * go_v_197, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_198 = grad_out[go_base + ((int64_t)198 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_19 * go_v_198, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_199 = grad_out[go_base + ((int64_t)199 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_20 * go_v_199, local_k_0);
    }
    {
        scalar_t w_i_48  = w[w_base + ((int64_t)48 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_200 = grad_out[go_base + ((int64_t)200 << 5)];
        local_k_0 = fma((scalar_t)(1.0) * w_i_48, x_j_21 * go_v_200, local_k_0);
    }
    scalar_t sum_k_0 = warp_sum(local_k_0);
    if (lane == 0) grad_y[gy_base + 0] += sum_k_0;

    // ---- grad_y chunk 1 ----
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
    {
        scalar_t w_i_8  = w[w_base + ((int64_t)8 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_8 = grad_out[go_base + ((int64_t)8 << 5)];
        local_k_1 = fma((scalar_t)(1.0) * w_i_8, x_j_0 * go_v_8, local_k_1);
    }
    {
        scalar_t w_i_9  = w[w_base + ((int64_t)9 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_9 = grad_out[go_base + ((int64_t)9 << 5)];
        local_k_1 = fma((scalar_t)(1.0) * w_i_9, x_j_1 * go_v_9, local_k_1);
    }
    {
        scalar_t w_i_10  = w[w_base + ((int64_t)10 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_10 = grad_out[go_base + ((int64_t)10 << 5)];
        local_k_1 = fma((scalar_t)(1.0) * w_i_10, x_j_2 * go_v_10, local_k_1);
    }
    {
        scalar_t w_i_11  = w[w_base + ((int64_t)11 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_11 = grad_out[go_base + ((int64_t)11 << 5)];
        local_k_1 = fma((scalar_t)(1.0) * w_i_11, x_j_3 * go_v_11, local_k_1);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        local_k_1 = fma((scalar_t)(-0.31622776389122009) * w_i_16, x_j_12 * go_v_32, local_k_1);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        local_k_1 = fma((scalar_t)(-0.54772257804870605) * w_i_16, x_j_14 * go_v_32, local_k_1);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        local_k_1 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_11 * go_v_33, local_k_1);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        local_k_1 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_10 * go_v_34, local_k_1);
    }
    {
        scalar_t w_i_19  = w[w_base + ((int64_t)19 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_43 = grad_out[go_base + ((int64_t)43 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_19, x_j_8 * go_v_43, local_k_1);
    }
    {
        scalar_t w_i_20  = w[w_base + ((int64_t)20 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_44 = grad_out[go_base + ((int64_t)44 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_20, x_j_9 * go_v_44, local_k_1);
    }
    {
        scalar_t w_i_19  = w[w_base + ((int64_t)19 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_45 = grad_out[go_base + ((int64_t)45 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_19, x_j_6 * go_v_45, local_k_1);
    }
    {
        scalar_t w_i_20  = w[w_base + ((int64_t)20 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_46 = grad_out[go_base + ((int64_t)46 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_20, x_j_7 * go_v_46, local_k_1);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        local_k_1 = fma((scalar_t)(-0.40824827551841736) * w_i_25, x_j_11 * go_v_63, local_k_1);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        local_k_1 = fma((scalar_t)(0.40824827551841736) * w_i_25, x_j_10 * go_v_64, local_k_1);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_65 = grad_out[go_base + ((int64_t)65 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_25, x_j_13 * go_v_65, local_k_1);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_25, x_j_12 * go_v_66, local_k_1);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        local_k_1 = fma((scalar_t)(0.40824827551841736) * w_i_25, x_j_14 * go_v_66, local_k_1);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        local_k_1 = fma((scalar_t)(-0.40824827551841736) * w_i_25, x_j_13 * go_v_67, local_k_1);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_98 = grad_out[go_base + ((int64_t)98 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_8 * go_v_98, local_k_1);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_99 = grad_out[go_base + ((int64_t)99 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_9 * go_v_99, local_k_1);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_100 = grad_out[go_base + ((int64_t)100 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_6 * go_v_100, local_k_1);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_101 = grad_out[go_base + ((int64_t)101 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_7 * go_v_101, local_k_1);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        local_k_1 = fma((scalar_t)(-0.40824827551841736) * w_i_32, x_j_4 * go_v_102, local_k_1);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        local_k_1 = fma((scalar_t)(-0.40824827551841736) * w_i_33, x_j_5 * go_v_103, local_k_1);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_106 = grad_out[go_base + ((int64_t)106 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_32, x_j_4 * go_v_106, local_k_1);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_107 = grad_out[go_base + ((int64_t)107 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_33, x_j_5 * go_v_107, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        local_k_1 = fma((scalar_t)(-0.15430334210395813) * w_i_38, x_j_19 * go_v_128, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        local_k_1 = fma((scalar_t)(-0.59761428833007812) * w_i_38, x_j_21 * go_v_128, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        local_k_1 = fma((scalar_t)(-0.37796446681022644) * w_i_38, x_j_18 * go_v_129, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        local_k_1 = fma((scalar_t)(-0.48795005679130554) * w_i_38, x_j_20 * go_v_129, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        local_k_1 = fma((scalar_t)(0.53452247381210327) * w_i_38, x_j_17 * go_v_130, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        local_k_1 = fma((scalar_t)(0.48795005679130554) * w_i_38, x_j_16 * go_v_131, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        local_k_1 = fma((scalar_t)(0.59761428833007812) * w_i_38, x_j_15 * go_v_132, local_k_1);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        local_k_1 = fma((scalar_t)(0.15430334210395813) * w_i_38, x_j_17 * go_v_132, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_180 = grad_out[go_base + ((int64_t)180 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_46, x_j_14 * go_v_180, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        local_k_1 = fma((scalar_t)(0.57735025882720947) * w_i_46, x_j_13 * go_v_181, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        local_k_1 = fma((scalar_t)(0.63245552778244019) * w_i_46, x_j_12 * go_v_182, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        local_k_1 = fma((scalar_t)(0.18257418274879456) * w_i_46, x_j_14 * go_v_182, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        local_k_1 = fma((scalar_t)(-0.44721359014511108) * w_i_46, x_j_11 * go_v_183, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        local_k_1 = fma((scalar_t)(-0.18257418274879456) * w_i_46, x_j_10 * go_v_184, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        local_k_1 = fma((scalar_t)(-0.57735025882720947) * w_i_46, x_j_11 * go_v_185, local_k_1);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_186 = grad_out[go_base + ((int64_t)186 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_46, x_j_10 * go_v_186, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        local_k_1 = fma((scalar_t)(-0.35355338454246521) * w_i_53, x_j_16 * go_v_229, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        local_k_1 = fma((scalar_t)(0.35355338454246521) * w_i_53, x_j_15 * go_v_230, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        local_k_1 = fma((scalar_t)(-0.45643547177314758) * w_i_53, x_j_17 * go_v_230, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        local_k_1 = fma((scalar_t)(0.45643547177314758) * w_i_53, x_j_16 * go_v_231, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_232 = grad_out[go_base + ((int64_t)232 << 5)];
        local_k_1 = fma((scalar_t)(0.70710676908493042) * w_i_53, x_j_19 * go_v_232, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        local_k_1 = fma((scalar_t)(-0.70710676908493042) * w_i_53, x_j_18 * go_v_233, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        local_k_1 = fma((scalar_t)(0.45643547177314758) * w_i_53, x_j_20 * go_v_233, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        local_k_1 = fma((scalar_t)(-0.45643547177314758) * w_i_53, x_j_19 * go_v_234, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        local_k_1 = fma((scalar_t)(0.35355338454246521) * w_i_53, x_j_21 * go_v_234, local_k_1);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        local_k_1 = fma((scalar_t)(-0.35355338454246521) * w_i_53, x_j_20 * go_v_235, local_k_1);
    }
    scalar_t sum_k_1 = warp_sum(local_k_1);
    if (lane == 0) grad_y[gy_base + 1] += sum_k_1;

    // ---- grad_y chunk 2 ----
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
    {
        scalar_t w_i_8  = w[w_base + ((int64_t)8 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_12 = grad_out[go_base + ((int64_t)12 << 5)];
        local_k_2 = fma((scalar_t)(1.0) * w_i_8, x_j_0 * go_v_12, local_k_2);
    }
    {
        scalar_t w_i_9  = w[w_base + ((int64_t)9 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_13 = grad_out[go_base + ((int64_t)13 << 5)];
        local_k_2 = fma((scalar_t)(1.0) * w_i_9, x_j_1 * go_v_13, local_k_2);
    }
    {
        scalar_t w_i_10  = w[w_base + ((int64_t)10 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_14 = grad_out[go_base + ((int64_t)14 << 5)];
        local_k_2 = fma((scalar_t)(1.0) * w_i_10, x_j_2 * go_v_14, local_k_2);
    }
    {
        scalar_t w_i_11  = w[w_base + ((int64_t)11 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_15 = grad_out[go_base + ((int64_t)15 << 5)];
        local_k_2 = fma((scalar_t)(1.0) * w_i_11, x_j_3 * go_v_15, local_k_2);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        local_k_2 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_11 * go_v_32, local_k_2);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        local_k_2 = fma((scalar_t)(0.63245552778244019) * w_i_16, x_j_12 * go_v_33, local_k_2);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        local_k_2 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_13 * go_v_34, local_k_2);
    }
    {
        scalar_t w_i_19  = w[w_base + ((int64_t)19 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_41 = grad_out[go_base + ((int64_t)41 << 5)];
        local_k_2 = fma((scalar_t)(-0.70710676908493042) * w_i_19, x_j_8 * go_v_41, local_k_2);
    }
    {
        scalar_t w_i_20  = w[w_base + ((int64_t)20 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_42 = grad_out[go_base + ((int64_t)42 << 5)];
        local_k_2 = fma((scalar_t)(-0.70710676908493042) * w_i_20, x_j_9 * go_v_42, local_k_2);
    }
    {
        scalar_t w_i_19  = w[w_base + ((int64_t)19 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_45 = grad_out[go_base + ((int64_t)45 << 5)];
        local_k_2 = fma((scalar_t)(0.70710676908493042) * w_i_19, x_j_4 * go_v_45, local_k_2);
    }
    {
        scalar_t w_i_20  = w[w_base + ((int64_t)20 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_46 = grad_out[go_base + ((int64_t)46 << 5)];
        local_k_2 = fma((scalar_t)(0.70710676908493042) * w_i_20, x_j_5 * go_v_46, local_k_2);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        local_k_2 = fma((scalar_t)(-0.81649655103683472) * w_i_25, x_j_14 * go_v_63, local_k_2);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        local_k_2 = fma((scalar_t)(-0.40824827551841736) * w_i_25, x_j_13 * go_v_64, local_k_2);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        local_k_2 = fma((scalar_t)(0.40824827551841736) * w_i_25, x_j_11 * go_v_66, local_k_2);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        local_k_2 = fma((scalar_t)(0.81649655103683472) * w_i_25, x_j_10 * go_v_67, local_k_2);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_100 = grad_out[go_base + ((int64_t)100 << 5)];
        local_k_2 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_4 * go_v_100, local_k_2);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_101 = grad_out[go_base + ((int64_t)101 << 5)];
        local_k_2 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_5 * go_v_101, local_k_2);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        local_k_2 = fma((scalar_t)(0.81649655103683472) * w_i_32, x_j_6 * go_v_102, local_k_2);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        local_k_2 = fma((scalar_t)(0.81649655103683472) * w_i_33, x_j_7 * go_v_103, local_k_2);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_104 = grad_out[go_base + ((int64_t)104 << 5)];
        local_k_2 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_8 * go_v_104, local_k_2);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_105 = grad_out[go_base + ((int64_t)105 << 5)];
        local_k_2 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_9 * go_v_105, local_k_2);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        local_k_2 = fma((scalar_t)(0.48795005679130554) * w_i_38, x_j_16 * go_v_128, local_k_2);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        local_k_2 = fma((scalar_t)(0.61721336841583252) * w_i_38, x_j_17 * go_v_129, local_k_2);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        local_k_2 = fma((scalar_t)(0.65465366840362549) * w_i_38, x_j_18 * go_v_130, local_k_2);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        local_k_2 = fma((scalar_t)(0.61721336841583252) * w_i_38, x_j_19 * go_v_131, local_k_2);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        local_k_2 = fma((scalar_t)(0.48795005679130554) * w_i_38, x_j_20 * go_v_132, local_k_2);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        local_k_2 = fma((scalar_t)(0.57735025882720947) * w_i_46, x_j_10 * go_v_181, local_k_2);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        local_k_2 = fma((scalar_t)(0.73029673099517822) * w_i_46, x_j_11 * go_v_182, local_k_2);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        local_k_2 = fma((scalar_t)(0.7745966911315918) * w_i_46, x_j_12 * go_v_183, local_k_2);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        local_k_2 = fma((scalar_t)(0.73029673099517822) * w_i_46, x_j_13 * go_v_184, local_k_2);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        local_k_2 = fma((scalar_t)(0.57735025882720947) * w_i_46, x_j_14 * go_v_185, local_k_2);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        local_k_2 = fma((scalar_t)(-0.86602538824081421) * w_i_53, x_j_21 * go_v_229, local_k_2);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        local_k_2 = fma((scalar_t)(-0.57735025882720947) * w_i_53, x_j_20 * go_v_230, local_k_2);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        local_k_2 = fma((scalar_t)(-0.28867512941360474) * w_i_53, x_j_19 * go_v_231, local_k_2);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        local_k_2 = fma((scalar_t)(0.28867512941360474) * w_i_53, x_j_17 * go_v_233, local_k_2);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        local_k_2 = fma((scalar_t)(0.57735025882720947) * w_i_53, x_j_16 * go_v_234, local_k_2);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        local_k_2 = fma((scalar_t)(0.86602538824081421) * w_i_53, x_j_15 * go_v_235, local_k_2);
    }
    scalar_t sum_k_2 = warp_sum(local_k_2);
    if (lane == 0) grad_y[gy_base + 2] += sum_k_2;

    // ---- grad_y chunk 3 ----
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
    {
        scalar_t w_i_8  = w[w_base + ((int64_t)8 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_16 = grad_out[go_base + ((int64_t)16 << 5)];
        local_k_3 = fma((scalar_t)(1.0) * w_i_8, x_j_0 * go_v_16, local_k_3);
    }
    {
        scalar_t w_i_9  = w[w_base + ((int64_t)9 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_17 = grad_out[go_base + ((int64_t)17 << 5)];
        local_k_3 = fma((scalar_t)(1.0) * w_i_9, x_j_1 * go_v_17, local_k_3);
    }
    {
        scalar_t w_i_10  = w[w_base + ((int64_t)10 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_18 = grad_out[go_base + ((int64_t)18 << 5)];
        local_k_3 = fma((scalar_t)(1.0) * w_i_10, x_j_2 * go_v_18, local_k_3);
    }
    {
        scalar_t w_i_11  = w[w_base + ((int64_t)11 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_19 = grad_out[go_base + ((int64_t)19 << 5)];
        local_k_3 = fma((scalar_t)(1.0) * w_i_11, x_j_3 * go_v_19, local_k_3);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_32 = grad_out[go_base + ((int64_t)32 << 5)];
        local_k_3 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_10 * go_v_32, local_k_3);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_33 = grad_out[go_base + ((int64_t)33 << 5)];
        local_k_3 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_13 * go_v_33, local_k_3);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        local_k_3 = fma((scalar_t)(-0.31622776389122009) * w_i_16, x_j_12 * go_v_34, local_k_3);
    }
    {
        scalar_t w_i_16  = w[w_base + ((int64_t)16 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_34 = grad_out[go_base + ((int64_t)34 << 5)];
        local_k_3 = fma((scalar_t)(0.54772257804870605) * w_i_16, x_j_14 * go_v_34, local_k_3);
    }
    {
        scalar_t w_i_19  = w[w_base + ((int64_t)19 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_41 = grad_out[go_base + ((int64_t)41 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_19, x_j_6 * go_v_41, local_k_3);
    }
    {
        scalar_t w_i_20  = w[w_base + ((int64_t)20 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_42 = grad_out[go_base + ((int64_t)42 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_20, x_j_7 * go_v_42, local_k_3);
    }
    {
        scalar_t w_i_19  = w[w_base + ((int64_t)19 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_43 = grad_out[go_base + ((int64_t)43 << 5)];
        local_k_3 = fma((scalar_t)(-0.70710676908493042) * w_i_19, x_j_4 * go_v_43, local_k_3);
    }
    {
        scalar_t w_i_20  = w[w_base + ((int64_t)20 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_44 = grad_out[go_base + ((int64_t)44 << 5)];
        local_k_3 = fma((scalar_t)(-0.70710676908493042) * w_i_20, x_j_5 * go_v_44, local_k_3);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_63 = grad_out[go_base + ((int64_t)63 << 5)];
        local_k_3 = fma((scalar_t)(0.40824827551841736) * w_i_25, x_j_13 * go_v_63, local_k_3);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_25, x_j_12 * go_v_64, local_k_3);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_64 = grad_out[go_base + ((int64_t)64 << 5)];
        local_k_3 = fma((scalar_t)(0.40824827551841736) * w_i_25, x_j_14 * go_v_64, local_k_3);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_65 = grad_out[go_base + ((int64_t)65 << 5)];
        local_k_3 = fma((scalar_t)(-0.70710676908493042) * w_i_25, x_j_11 * go_v_65, local_k_3);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_66 = grad_out[go_base + ((int64_t)66 << 5)];
        local_k_3 = fma((scalar_t)(-0.40824827551841736) * w_i_25, x_j_10 * go_v_66, local_k_3);
    }
    {
        scalar_t w_i_25  = w[w_base + ((int64_t)25 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_67 = grad_out[go_base + ((int64_t)67 << 5)];
        local_k_3 = fma((scalar_t)(-0.40824827551841736) * w_i_25, x_j_11 * go_v_67, local_k_3);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_98 = grad_out[go_base + ((int64_t)98 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_4 * go_v_98, local_k_3);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_99 = grad_out[go_base + ((int64_t)99 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_5 * go_v_99, local_k_3);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_102 = grad_out[go_base + ((int64_t)102 << 5)];
        local_k_3 = fma((scalar_t)(-0.40824827551841736) * w_i_32, x_j_8 * go_v_102, local_k_3);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_103 = grad_out[go_base + ((int64_t)103 << 5)];
        local_k_3 = fma((scalar_t)(-0.40824827551841736) * w_i_33, x_j_9 * go_v_103, local_k_3);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_104 = grad_out[go_base + ((int64_t)104 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_6 * go_v_104, local_k_3);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_105 = grad_out[go_base + ((int64_t)105 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_7 * go_v_105, local_k_3);
    }
    {
        scalar_t w_i_32  = w[w_base + ((int64_t)32 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_106 = grad_out[go_base + ((int64_t)106 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_32, x_j_8 * go_v_106, local_k_3);
    }
    {
        scalar_t w_i_33  = w[w_base + ((int64_t)33 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_107 = grad_out[go_base + ((int64_t)107 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_33, x_j_9 * go_v_107, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        local_k_3 = fma((scalar_t)(0.59761428833007812) * w_i_38, x_j_15 * go_v_128, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_128 = grad_out[go_base + ((int64_t)128 << 5)];
        local_k_3 = fma((scalar_t)(-0.15430334210395813) * w_i_38, x_j_17 * go_v_128, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_129 = grad_out[go_base + ((int64_t)129 << 5)];
        local_k_3 = fma((scalar_t)(0.48795005679130554) * w_i_38, x_j_16 * go_v_129, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_130 = grad_out[go_base + ((int64_t)130 << 5)];
        local_k_3 = fma((scalar_t)(0.53452247381210327) * w_i_38, x_j_19 * go_v_130, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        local_k_3 = fma((scalar_t)(-0.37796446681022644) * w_i_38, x_j_18 * go_v_131, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_131 = grad_out[go_base + ((int64_t)131 << 5)];
        local_k_3 = fma((scalar_t)(0.48795005679130554) * w_i_38, x_j_20 * go_v_131, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        local_k_3 = fma((scalar_t)(-0.15430334210395813) * w_i_38, x_j_19 * go_v_132, local_k_3);
    }
    {
        scalar_t w_i_38  = w[w_base + ((int64_t)38 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_132 = grad_out[go_base + ((int64_t)132 << 5)];
        local_k_3 = fma((scalar_t)(0.59761428833007812) * w_i_38, x_j_21 * go_v_132, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_180 = grad_out[go_base + ((int64_t)180 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_46, x_j_10 * go_v_180, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_181 = grad_out[go_base + ((int64_t)181 << 5)];
        local_k_3 = fma((scalar_t)(0.57735025882720947) * w_i_46, x_j_11 * go_v_181, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_182 = grad_out[go_base + ((int64_t)182 << 5)];
        local_k_3 = fma((scalar_t)(-0.18257418274879456) * w_i_46, x_j_10 * go_v_182, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_183 = grad_out[go_base + ((int64_t)183 << 5)];
        local_k_3 = fma((scalar_t)(-0.44721359014511108) * w_i_46, x_j_13 * go_v_183, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        local_k_3 = fma((scalar_t)(0.63245552778244019) * w_i_46, x_j_12 * go_v_184, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_184 = grad_out[go_base + ((int64_t)184 << 5)];
        local_k_3 = fma((scalar_t)(-0.18257418274879456) * w_i_46, x_j_14 * go_v_184, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_185 = grad_out[go_base + ((int64_t)185 << 5)];
        local_k_3 = fma((scalar_t)(0.57735025882720947) * w_i_46, x_j_13 * go_v_185, local_k_3);
    }
    {
        scalar_t w_i_46  = w[w_base + ((int64_t)46 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_186 = grad_out[go_base + ((int64_t)186 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_46, x_j_14 * go_v_186, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_229 = grad_out[go_base + ((int64_t)229 << 5)];
        local_k_3 = fma((scalar_t)(0.35355338454246521) * w_i_53, x_j_20 * go_v_229, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        local_k_3 = fma((scalar_t)(0.45643547177314758) * w_i_53, x_j_19 * go_v_230, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_230 = grad_out[go_base + ((int64_t)230 << 5)];
        local_k_3 = fma((scalar_t)(0.35355338454246521) * w_i_53, x_j_21 * go_v_230, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        local_k_3 = fma((scalar_t)(0.70710676908493042) * w_i_53, x_j_18 * go_v_231, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_231 = grad_out[go_base + ((int64_t)231 << 5)];
        local_k_3 = fma((scalar_t)(0.45643547177314758) * w_i_53, x_j_20 * go_v_231, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_232 = grad_out[go_base + ((int64_t)232 << 5)];
        local_k_3 = fma((scalar_t)(-0.70710676908493042) * w_i_53, x_j_17 * go_v_232, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_233 = grad_out[go_base + ((int64_t)233 << 5)];
        local_k_3 = fma((scalar_t)(-0.45643547177314758) * w_i_53, x_j_16 * go_v_233, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        local_k_3 = fma((scalar_t)(-0.35355338454246521) * w_i_53, x_j_15 * go_v_234, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_234 = grad_out[go_base + ((int64_t)234 << 5)];
        local_k_3 = fma((scalar_t)(-0.45643547177314758) * w_i_53, x_j_17 * go_v_234, local_k_3);
    }
    {
        scalar_t w_i_53  = w[w_base + ((int64_t)53 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_235 = grad_out[go_base + ((int64_t)235 << 5)];
        local_k_3 = fma((scalar_t)(-0.35355338454246521) * w_i_53, x_j_16 * go_v_235, local_k_3);
    }
    scalar_t sum_k_3 = warp_sum(local_k_3);
    if (lane == 0) grad_y[gy_base + 3] += sum_k_3;

    // ---- grad_y chunk 4 ----
    scalar_t local_k_4 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_4 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_10 * go_v_6, local_k_4);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        local_k_4 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_8 * go_v_26, local_k_4);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        local_k_4 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_9 * go_v_27, local_k_4);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        local_k_4 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_4 * go_v_30, local_k_4);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        local_k_4 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_5 * go_v_31, local_k_4);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_4 = fma((scalar_t)(-0.11952286213636398) * w_i_18, x_j_19 * go_v_38, local_k_4);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_4 = fma((scalar_t)(-0.46291005611419678) * w_i_18, x_j_21 * go_v_38, local_k_4);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        local_k_4 = fma((scalar_t)(0.37796446681022644) * w_i_18, x_j_16 * go_v_39, local_k_4);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_4 = fma((scalar_t)(0.46291005611419678) * w_i_18, x_j_15 * go_v_40, local_k_4);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_4 = fma((scalar_t)(-0.11952286213636398) * w_i_18, x_j_17 * go_v_40, local_k_4);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        local_k_4 = fma((scalar_t)(0.31622776389122009) * w_i_21, x_j_11 * go_v_47, local_k_4);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        local_k_4 = fma((scalar_t)(0.63245552778244019) * w_i_21, x_j_14 * go_v_48, local_k_4);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        local_k_4 = fma((scalar_t)(-0.31622776389122009) * w_i_21, x_j_13 * go_v_49, local_k_4);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        local_k_4 = fma((scalar_t)(-0.40824827551841736) * w_i_23, x_j_4 * go_v_55, local_k_4);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        local_k_4 = fma((scalar_t)(-0.40824827551841736) * w_i_24, x_j_5 * go_v_56, local_k_4);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        local_k_4 = fma((scalar_t)(0.40824827551841736) * w_i_23, x_j_8 * go_v_59, local_k_4);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        local_k_4 = fma((scalar_t)(0.40824827551841736) * w_i_24, x_j_9 * go_v_60, local_k_4);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        local_k_4 = fma((scalar_t)(-0.81649655103683472) * w_i_23, x_j_6 * go_v_61, local_k_4);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        local_k_4 = fma((scalar_t)(-0.81649655103683472) * w_i_24, x_j_7 * go_v_62, local_k_4);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        local_k_4 = fma((scalar_t)(0.42257711291313171) * w_i_27, x_j_15 * go_v_74, local_k_4);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        local_k_4 = fma((scalar_t)(0.32732683420181274) * w_i_27, x_j_17 * go_v_74, local_k_4);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        local_k_4 = fma((scalar_t)(0.59761428833007812) * w_i_27, x_j_20 * go_v_75, local_k_4);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        local_k_4 = fma((scalar_t)(-0.32732683420181274) * w_i_27, x_j_19 * go_v_76, local_k_4);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        local_k_4 = fma((scalar_t)(0.42257711291313171) * w_i_27, x_j_21 * go_v_76, local_k_4);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        local_k_4 = fma((scalar_t)(0.26726123690605164) * w_i_27, x_j_18 * go_v_77, local_k_4);
    }
    {
        scalar_t w_i_28  = w[w_base + ((int64_t)28 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_78 = grad_out[go_base + ((int64_t)78 << 5)];
        local_k_4 = fma((scalar_t)(1.0) * w_i_28, x_j_0 * go_v_78, local_k_4);
    }
    {
        scalar_t w_i_29  = w[w_base + ((int64_t)29 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_79 = grad_out[go_base + ((int64_t)79 << 5)];
        local_k_4 = fma((scalar_t)(1.0) * w_i_29, x_j_1 * go_v_79, local_k_4);
    }
    {
        scalar_t w_i_30  = w[w_base + ((int64_t)30 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_80 = grad_out[go_base + ((int64_t)80 << 5)];
        local_k_4 = fma((scalar_t)(1.0) * w_i_30, x_j_2 * go_v_80, local_k_4);
    }
    {
        scalar_t w_i_31  = w[w_base + ((int64_t)31 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_81 = grad_out[go_base + ((int64_t)81 << 5)];
        local_k_4 = fma((scalar_t)(1.0) * w_i_31, x_j_3 * go_v_81, local_k_4);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        local_k_4 = fma((scalar_t)(-0.53452247381210327) * w_i_37, x_j_12 * go_v_123, local_k_4);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        local_k_4 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_13 * go_v_124, local_k_4);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        local_k_4 = fma((scalar_t)(-0.53452247381210327) * w_i_37, x_j_10 * go_v_125, local_k_4);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        local_k_4 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_11 * go_v_126, local_k_4);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_166 = grad_out[go_base + ((int64_t)166 << 5)];
        local_k_4 = fma((scalar_t)(0.70710676908493042) * w_i_44, x_j_8 * go_v_166, local_k_4);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_167 = grad_out[go_base + ((int64_t)167 << 5)];
        local_k_4 = fma((scalar_t)(0.70710676908493042) * w_i_45, x_j_9 * go_v_167, local_k_4);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        local_k_4 = fma((scalar_t)(0.57735025882720947) * w_i_44, x_j_6 * go_v_168, local_k_4);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        local_k_4 = fma((scalar_t)(0.57735025882720947) * w_i_45, x_j_7 * go_v_169, local_k_4);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        local_k_4 = fma((scalar_t)(-0.18257418274879456) * w_i_44, x_j_8 * go_v_170, local_k_4);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        local_k_4 = fma((scalar_t)(-0.18257418274879456) * w_i_45, x_j_9 * go_v_171, local_k_4);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        local_k_4 = fma((scalar_t)(-0.18257418274879456) * w_i_44, x_j_4 * go_v_174, local_k_4);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        local_k_4 = fma((scalar_t)(-0.18257418274879456) * w_i_45, x_j_5 * go_v_175, local_k_4);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_178 = grad_out[go_base + ((int64_t)178 << 5)];
        local_k_4 = fma((scalar_t)(-0.70710676908493042) * w_i_44, x_j_4 * go_v_178, local_k_4);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_179 = grad_out[go_base + ((int64_t)179 << 5)];
        local_k_4 = fma((scalar_t)(-0.70710676908493042) * w_i_45, x_j_5 * go_v_179, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        local_k_4 = fma((scalar_t)(-0.28867512941360474) * w_i_49, x_j_19 * go_v_201, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        local_k_4 = fma((scalar_t)(-0.57735025882720947) * w_i_49, x_j_18 * go_v_202, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_4 = fma((scalar_t)(0.44721359014511108) * w_i_49, x_j_19 * go_v_203, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_4 = fma((scalar_t)(0.28867512941360474) * w_i_49, x_j_21 * go_v_203, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        local_k_4 = fma((scalar_t)(-0.57735025882720947) * w_i_49, x_j_16 * go_v_204, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_4 = fma((scalar_t)(-0.28867512941360474) * w_i_49, x_j_15 * go_v_205, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_4 = fma((scalar_t)(0.44721359014511108) * w_i_49, x_j_17 * go_v_205, local_k_4);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        local_k_4 = fma((scalar_t)(0.28867512941360474) * w_i_49, x_j_17 * go_v_207, local_k_4);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        local_k_4 = fma((scalar_t)(-0.5) * w_i_52, x_j_11 * go_v_222, local_k_4);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        local_k_4 = fma((scalar_t)(-0.3872983455657959) * w_i_52, x_j_11 * go_v_224, local_k_4);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        local_k_4 = fma((scalar_t)(-0.31622776389122009) * w_i_52, x_j_14 * go_v_225, local_k_4);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        local_k_4 = fma((scalar_t)(0.3872983455657959) * w_i_52, x_j_13 * go_v_226, local_k_4);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_227 = grad_out[go_base + ((int64_t)227 << 5)];
        local_k_4 = fma((scalar_t)(-0.70710676908493042) * w_i_52, x_j_12 * go_v_227, local_k_4);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        local_k_4 = fma((scalar_t)(-0.5) * w_i_52, x_j_13 * go_v_228, local_k_4);
    }
    scalar_t sum_k_4 = warp_sum(local_k_4);
    if (lane == 0) grad_y[gy_base + 4] += sum_k_4;

    // ---- grad_y chunk 5 ----
    scalar_t local_k_5 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_5 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_11 * go_v_6, local_k_5);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        local_k_5 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_6 * go_v_26, local_k_5);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        local_k_5 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_7 * go_v_27, local_k_5);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        local_k_5 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_4 * go_v_28, local_k_5);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        local_k_5 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_5 * go_v_29, local_k_5);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_5 = fma((scalar_t)(-0.29276999831199646) * w_i_18, x_j_18 * go_v_38, local_k_5);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_5 = fma((scalar_t)(-0.37796446681022644) * w_i_18, x_j_20 * go_v_38, local_k_5);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        local_k_5 = fma((scalar_t)(0.47809144854545593) * w_i_18, x_j_17 * go_v_39, local_k_5);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_5 = fma((scalar_t)(0.37796446681022644) * w_i_18, x_j_16 * go_v_40, local_k_5);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        local_k_5 = fma((scalar_t)(-0.31622776389122009) * w_i_21, x_j_10 * go_v_47, local_k_5);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        local_k_5 = fma((scalar_t)(0.31622776389122009) * w_i_21, x_j_13 * go_v_48, local_k_5);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        local_k_5 = fma((scalar_t)(-0.54772257804870605) * w_i_21, x_j_12 * go_v_49, local_k_5);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        local_k_5 = fma((scalar_t)(-0.31622776389122009) * w_i_21, x_j_14 * go_v_49, local_k_5);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        local_k_5 = fma((scalar_t)(0.40824827551841736) * w_i_23, x_j_4 * go_v_53, local_k_5);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        local_k_5 = fma((scalar_t)(0.40824827551841736) * w_i_24, x_j_5 * go_v_54, local_k_5);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_57 = grad_out[go_base + ((int64_t)57 << 5)];
        local_k_5 = fma((scalar_t)(0.70710676908493042) * w_i_23, x_j_8 * go_v_57, local_k_5);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_58 = grad_out[go_base + ((int64_t)58 << 5)];
        local_k_5 = fma((scalar_t)(0.70710676908493042) * w_i_24, x_j_9 * go_v_58, local_k_5);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        local_k_5 = fma((scalar_t)(-0.40824827551841736) * w_i_23, x_j_6 * go_v_59, local_k_5);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        local_k_5 = fma((scalar_t)(-0.40824827551841736) * w_i_24, x_j_7 * go_v_60, local_k_5);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        local_k_5 = fma((scalar_t)(0.40824827551841736) * w_i_23, x_j_8 * go_v_61, local_k_5);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        local_k_5 = fma((scalar_t)(0.40824827551841736) * w_i_24, x_j_9 * go_v_62, local_k_5);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        local_k_5 = fma((scalar_t)(-0.42257711291313171) * w_i_27, x_j_15 * go_v_73, local_k_5);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        local_k_5 = fma((scalar_t)(-0.32732683420181274) * w_i_27, x_j_17 * go_v_73, local_k_5);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        local_k_5 = fma((scalar_t)(0.37796446681022644) * w_i_27, x_j_19 * go_v_75, local_k_5);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        local_k_5 = fma((scalar_t)(-0.53452247381210327) * w_i_27, x_j_18 * go_v_76, local_k_5);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        local_k_5 = fma((scalar_t)(-0.32732683420181274) * w_i_27, x_j_19 * go_v_77, local_k_5);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        local_k_5 = fma((scalar_t)(-0.42257711291313171) * w_i_27, x_j_21 * go_v_77, local_k_5);
    }
    {
        scalar_t w_i_28  = w[w_base + ((int64_t)28 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_82 = grad_out[go_base + ((int64_t)82 << 5)];
        local_k_5 = fma((scalar_t)(1.0) * w_i_28, x_j_0 * go_v_82, local_k_5);
    }
    {
        scalar_t w_i_29  = w[w_base + ((int64_t)29 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_83 = grad_out[go_base + ((int64_t)83 << 5)];
        local_k_5 = fma((scalar_t)(1.0) * w_i_29, x_j_1 * go_v_83, local_k_5);
    }
    {
        scalar_t w_i_30  = w[w_base + ((int64_t)30 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_84 = grad_out[go_base + ((int64_t)84 << 5)];
        local_k_5 = fma((scalar_t)(1.0) * w_i_30, x_j_2 * go_v_84, local_k_5);
    }
    {
        scalar_t w_i_31  = w[w_base + ((int64_t)31 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_85 = grad_out[go_base + ((int64_t)85 << 5)];
        local_k_5 = fma((scalar_t)(1.0) * w_i_31, x_j_3 * go_v_85, local_k_5);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        local_k_5 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_13 * go_v_123, local_k_5);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        local_k_5 = fma((scalar_t)(0.26726123690605164) * w_i_37, x_j_12 * go_v_124, local_k_5);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        local_k_5 = fma((scalar_t)(-0.46291005611419678) * w_i_37, x_j_14 * go_v_124, local_k_5);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        local_k_5 = fma((scalar_t)(0.26726123690605164) * w_i_37, x_j_11 * go_v_125, local_k_5);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        local_k_5 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_10 * go_v_126, local_k_5);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        local_k_5 = fma((scalar_t)(-0.46291005611419678) * w_i_37, x_j_11 * go_v_127, local_k_5);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        local_k_5 = fma((scalar_t)(0.57735025882720947) * w_i_44, x_j_8 * go_v_168, local_k_5);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        local_k_5 = fma((scalar_t)(0.57735025882720947) * w_i_45, x_j_9 * go_v_169, local_k_5);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        local_k_5 = fma((scalar_t)(0.73029673099517822) * w_i_44, x_j_6 * go_v_170, local_k_5);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        local_k_5 = fma((scalar_t)(0.73029673099517822) * w_i_45, x_j_7 * go_v_171, local_k_5);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        local_k_5 = fma((scalar_t)(-0.44721359014511108) * w_i_44, x_j_4 * go_v_172, local_k_5);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        local_k_5 = fma((scalar_t)(-0.44721359014511108) * w_i_45, x_j_5 * go_v_173, local_k_5);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        local_k_5 = fma((scalar_t)(-0.57735025882720947) * w_i_44, x_j_4 * go_v_176, local_k_5);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        local_k_5 = fma((scalar_t)(-0.57735025882720947) * w_i_45, x_j_5 * go_v_177, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        local_k_5 = fma((scalar_t)(0.45643547177314758) * w_i_49, x_j_20 * go_v_201, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        local_k_5 = fma((scalar_t)(0.35355338454246521) * w_i_49, x_j_19 * go_v_202, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        local_k_5 = fma((scalar_t)(-0.45643547177314758) * w_i_49, x_j_21 * go_v_202, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_5 = fma((scalar_t)(0.18257418274879456) * w_i_49, x_j_18 * go_v_203, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_5 = fma((scalar_t)(-0.35355338454246521) * w_i_49, x_j_20 * go_v_203, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        local_k_5 = fma((scalar_t)(0.18257418274879456) * w_i_49, x_j_17 * go_v_204, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_5 = fma((scalar_t)(0.35355338454246521) * w_i_49, x_j_16 * go_v_205, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        local_k_5 = fma((scalar_t)(0.45643547177314758) * w_i_49, x_j_15 * go_v_206, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        local_k_5 = fma((scalar_t)(-0.35355338454246521) * w_i_49, x_j_17 * go_v_206, local_k_5);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        local_k_5 = fma((scalar_t)(-0.45643547177314758) * w_i_49, x_j_16 * go_v_207, local_k_5);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        local_k_5 = fma((scalar_t)(0.5) * w_i_52, x_j_10 * go_v_222, local_k_5);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        local_k_5 = fma((scalar_t)(0.3872983455657959) * w_i_52, x_j_10 * go_v_224, local_k_5);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        local_k_5 = fma((scalar_t)(0.63245552778244019) * w_i_52, x_j_13 * go_v_225, local_k_5);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        local_k_5 = fma((scalar_t)(-0.44721359014511108) * w_i_52, x_j_12 * go_v_226, local_k_5);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        local_k_5 = fma((scalar_t)(0.3872983455657959) * w_i_52, x_j_14 * go_v_226, local_k_5);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        local_k_5 = fma((scalar_t)(0.5) * w_i_52, x_j_14 * go_v_228, local_k_5);
    }
    scalar_t sum_k_5 = warp_sum(local_k_5);
    if (lane == 0) grad_y[gy_base + 5] += sum_k_5;

    // ---- grad_y chunk 6 ----
    scalar_t local_k_6 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_6 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_12 * go_v_6, local_k_6);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        local_k_6 = fma((scalar_t)(-0.31622776389122009) * w_i_14, x_j_4 * go_v_26, local_k_6);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        local_k_6 = fma((scalar_t)(-0.31622776389122009) * w_i_15, x_j_5 * go_v_27, local_k_6);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        local_k_6 = fma((scalar_t)(0.63245552778244019) * w_i_14, x_j_6 * go_v_28, local_k_6);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        local_k_6 = fma((scalar_t)(0.63245552778244019) * w_i_15, x_j_7 * go_v_29, local_k_6);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        local_k_6 = fma((scalar_t)(-0.31622776389122009) * w_i_14, x_j_8 * go_v_30, local_k_6);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        local_k_6 = fma((scalar_t)(-0.31622776389122009) * w_i_15, x_j_9 * go_v_31, local_k_6);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_6 = fma((scalar_t)(0.41403934359550476) * w_i_18, x_j_17 * go_v_38, local_k_6);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        local_k_6 = fma((scalar_t)(0.50709253549575806) * w_i_18, x_j_18 * go_v_39, local_k_6);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_6 = fma((scalar_t)(0.41403934359550476) * w_i_18, x_j_19 * go_v_40, local_k_6);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        local_k_6 = fma((scalar_t)(-0.54772257804870605) * w_i_21, x_j_13 * go_v_47, local_k_6);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        local_k_6 = fma((scalar_t)(0.54772257804870605) * w_i_21, x_j_11 * go_v_49, local_k_6);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        local_k_6 = fma((scalar_t)(-0.70710676908493042) * w_i_23, x_j_8 * go_v_55, local_k_6);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        local_k_6 = fma((scalar_t)(-0.70710676908493042) * w_i_24, x_j_9 * go_v_56, local_k_6);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        local_k_6 = fma((scalar_t)(0.70710676908493042) * w_i_23, x_j_4 * go_v_59, local_k_6);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        local_k_6 = fma((scalar_t)(0.70710676908493042) * w_i_24, x_j_5 * go_v_60, local_k_6);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        local_k_6 = fma((scalar_t)(-0.59761428833007812) * w_i_27, x_j_20 * go_v_73, local_k_6);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        local_k_6 = fma((scalar_t)(-0.37796446681022644) * w_i_27, x_j_19 * go_v_74, local_k_6);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        local_k_6 = fma((scalar_t)(0.37796446681022644) * w_i_27, x_j_17 * go_v_76, local_k_6);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        local_k_6 = fma((scalar_t)(0.59761428833007812) * w_i_27, x_j_16 * go_v_77, local_k_6);
    }
    {
        scalar_t w_i_28  = w[w_base + ((int64_t)28 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_86 = grad_out[go_base + ((int64_t)86 << 5)];
        local_k_6 = fma((scalar_t)(1.0) * w_i_28, x_j_0 * go_v_86, local_k_6);
    }
    {
        scalar_t w_i_29  = w[w_base + ((int64_t)29 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_87 = grad_out[go_base + ((int64_t)87 << 5)];
        local_k_6 = fma((scalar_t)(1.0) * w_i_29, x_j_1 * go_v_87, local_k_6);
    }
    {
        scalar_t w_i_30  = w[w_base + ((int64_t)30 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_88 = grad_out[go_base + ((int64_t)88 << 5)];
        local_k_6 = fma((scalar_t)(1.0) * w_i_30, x_j_2 * go_v_88, local_k_6);
    }
    {
        scalar_t w_i_31  = w[w_base + ((int64_t)31 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_89 = grad_out[go_base + ((int64_t)89 << 5)];
        local_k_6 = fma((scalar_t)(1.0) * w_i_31, x_j_3 * go_v_89, local_k_6);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        local_k_6 = fma((scalar_t)(-0.53452247381210327) * w_i_37, x_j_10 * go_v_123, local_k_6);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        local_k_6 = fma((scalar_t)(0.26726123690605164) * w_i_37, x_j_11 * go_v_124, local_k_6);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        local_k_6 = fma((scalar_t)(0.53452247381210327) * w_i_37, x_j_12 * go_v_125, local_k_6);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        local_k_6 = fma((scalar_t)(0.26726123690605164) * w_i_37, x_j_13 * go_v_126, local_k_6);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        local_k_6 = fma((scalar_t)(-0.53452247381210327) * w_i_37, x_j_14 * go_v_127, local_k_6);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        local_k_6 = fma((scalar_t)(0.63245552778244019) * w_i_44, x_j_4 * go_v_170, local_k_6);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        local_k_6 = fma((scalar_t)(0.63245552778244019) * w_i_45, x_j_5 * go_v_171, local_k_6);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        local_k_6 = fma((scalar_t)(0.7745966911315918) * w_i_44, x_j_6 * go_v_172, local_k_6);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        local_k_6 = fma((scalar_t)(0.7745966911315918) * w_i_45, x_j_7 * go_v_173, local_k_6);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        local_k_6 = fma((scalar_t)(0.63245552778244019) * w_i_44, x_j_8 * go_v_174, local_k_6);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        local_k_6 = fma((scalar_t)(0.63245552778244019) * w_i_45, x_j_9 * go_v_175, local_k_6);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        local_k_6 = fma((scalar_t)(-0.64549720287322998) * w_i_49, x_j_15 * go_v_201, local_k_6);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_6 = fma((scalar_t)(0.3872983455657959) * w_i_49, x_j_17 * go_v_203, local_k_6);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        local_k_6 = fma((scalar_t)(0.51639777421951294) * w_i_49, x_j_18 * go_v_204, local_k_6);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_6 = fma((scalar_t)(0.3872983455657959) * w_i_49, x_j_19 * go_v_205, local_k_6);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        local_k_6 = fma((scalar_t)(-0.64549720287322998) * w_i_49, x_j_21 * go_v_207, local_k_6);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_223 = grad_out[go_base + ((int64_t)223 << 5)];
        local_k_6 = fma((scalar_t)(-0.70710676908493042) * w_i_52, x_j_14 * go_v_223, local_k_6);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        local_k_6 = fma((scalar_t)(-0.44721359014511108) * w_i_52, x_j_13 * go_v_224, local_k_6);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        local_k_6 = fma((scalar_t)(0.44721359014511108) * w_i_52, x_j_11 * go_v_226, local_k_6);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_227 = grad_out[go_base + ((int64_t)227 << 5)];
        local_k_6 = fma((scalar_t)(0.70710676908493042) * w_i_52, x_j_10 * go_v_227, local_k_6);
    }
    scalar_t sum_k_6 = warp_sum(local_k_6);
    if (lane == 0) grad_y[gy_base + 6] += sum_k_6;

    // ---- grad_y chunk 7 ----
    scalar_t local_k_7 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_7 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_13 * go_v_6, local_k_7);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_28 = grad_out[go_base + ((int64_t)28 << 5)];
        local_k_7 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_8 * go_v_28, local_k_7);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_29 = grad_out[go_base + ((int64_t)29 << 5)];
        local_k_7 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_9 * go_v_29, local_k_7);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        local_k_7 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_6 * go_v_30, local_k_7);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        local_k_7 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_7 * go_v_31, local_k_7);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_7 = fma((scalar_t)(0.37796446681022644) * w_i_18, x_j_16 * go_v_38, local_k_7);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        local_k_7 = fma((scalar_t)(0.47809144854545593) * w_i_18, x_j_19 * go_v_39, local_k_7);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_7 = fma((scalar_t)(-0.29276999831199646) * w_i_18, x_j_18 * go_v_40, local_k_7);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_7 = fma((scalar_t)(0.37796446681022644) * w_i_18, x_j_20 * go_v_40, local_k_7);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        local_k_7 = fma((scalar_t)(0.54772257804870605) * w_i_21, x_j_12 * go_v_47, local_k_7);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        local_k_7 = fma((scalar_t)(-0.31622776389122009) * w_i_21, x_j_14 * go_v_47, local_k_7);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        local_k_7 = fma((scalar_t)(-0.31622776389122009) * w_i_21, x_j_11 * go_v_48, local_k_7);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        local_k_7 = fma((scalar_t)(0.31622776389122009) * w_i_21, x_j_10 * go_v_49, local_k_7);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        local_k_7 = fma((scalar_t)(-0.40824827551841736) * w_i_23, x_j_8 * go_v_53, local_k_7);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        local_k_7 = fma((scalar_t)(-0.40824827551841736) * w_i_24, x_j_9 * go_v_54, local_k_7);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        local_k_7 = fma((scalar_t)(0.40824827551841736) * w_i_23, x_j_6 * go_v_55, local_k_7);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        local_k_7 = fma((scalar_t)(0.40824827551841736) * w_i_24, x_j_7 * go_v_56, local_k_7);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_57 = grad_out[go_base + ((int64_t)57 << 5)];
        local_k_7 = fma((scalar_t)(-0.70710676908493042) * w_i_23, x_j_4 * go_v_57, local_k_7);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_58 = grad_out[go_base + ((int64_t)58 << 5)];
        local_k_7 = fma((scalar_t)(-0.70710676908493042) * w_i_24, x_j_5 * go_v_58, local_k_7);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_61 = grad_out[go_base + ((int64_t)61 << 5)];
        local_k_7 = fma((scalar_t)(0.40824827551841736) * w_i_23, x_j_4 * go_v_61, local_k_7);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_62 = grad_out[go_base + ((int64_t)62 << 5)];
        local_k_7 = fma((scalar_t)(0.40824827551841736) * w_i_24, x_j_5 * go_v_62, local_k_7);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        local_k_7 = fma((scalar_t)(0.32732683420181274) * w_i_27, x_j_19 * go_v_73, local_k_7);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        local_k_7 = fma((scalar_t)(-0.42257711291313171) * w_i_27, x_j_21 * go_v_73, local_k_7);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        local_k_7 = fma((scalar_t)(0.53452247381210327) * w_i_27, x_j_18 * go_v_74, local_k_7);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        local_k_7 = fma((scalar_t)(-0.37796446681022644) * w_i_27, x_j_17 * go_v_75, local_k_7);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        local_k_7 = fma((scalar_t)(0.42257711291313171) * w_i_27, x_j_15 * go_v_77, local_k_7);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_77 = grad_out[go_base + ((int64_t)77 << 5)];
        local_k_7 = fma((scalar_t)(-0.32732683420181274) * w_i_27, x_j_17 * go_v_77, local_k_7);
    }
    {
        scalar_t w_i_28  = w[w_base + ((int64_t)28 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_90 = grad_out[go_base + ((int64_t)90 << 5)];
        local_k_7 = fma((scalar_t)(1.0) * w_i_28, x_j_0 * go_v_90, local_k_7);
    }
    {
        scalar_t w_i_29  = w[w_base + ((int64_t)29 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_91 = grad_out[go_base + ((int64_t)91 << 5)];
        local_k_7 = fma((scalar_t)(1.0) * w_i_29, x_j_1 * go_v_91, local_k_7);
    }
    {
        scalar_t w_i_30  = w[w_base + ((int64_t)30 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_92 = grad_out[go_base + ((int64_t)92 << 5)];
        local_k_7 = fma((scalar_t)(1.0) * w_i_30, x_j_2 * go_v_92, local_k_7);
    }
    {
        scalar_t w_i_31  = w[w_base + ((int64_t)31 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_93 = grad_out[go_base + ((int64_t)93 << 5)];
        local_k_7 = fma((scalar_t)(1.0) * w_i_31, x_j_3 * go_v_93, local_k_7);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_123 = grad_out[go_base + ((int64_t)123 << 5)];
        local_k_7 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_11 * go_v_123, local_k_7);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        local_k_7 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_10 * go_v_124, local_k_7);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        local_k_7 = fma((scalar_t)(0.26726123690605164) * w_i_37, x_j_13 * go_v_125, local_k_7);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        local_k_7 = fma((scalar_t)(0.26726123690605164) * w_i_37, x_j_12 * go_v_126, local_k_7);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        local_k_7 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_14 * go_v_126, local_k_7);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        local_k_7 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_13 * go_v_127, local_k_7);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_168 = grad_out[go_base + ((int64_t)168 << 5)];
        local_k_7 = fma((scalar_t)(0.57735025882720947) * w_i_44, x_j_4 * go_v_168, local_k_7);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_169 = grad_out[go_base + ((int64_t)169 << 5)];
        local_k_7 = fma((scalar_t)(0.57735025882720947) * w_i_45, x_j_5 * go_v_169, local_k_7);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_172 = grad_out[go_base + ((int64_t)172 << 5)];
        local_k_7 = fma((scalar_t)(-0.44721359014511108) * w_i_44, x_j_8 * go_v_172, local_k_7);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_173 = grad_out[go_base + ((int64_t)173 << 5)];
        local_k_7 = fma((scalar_t)(-0.44721359014511108) * w_i_45, x_j_9 * go_v_173, local_k_7);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        local_k_7 = fma((scalar_t)(0.73029673099517822) * w_i_44, x_j_6 * go_v_174, local_k_7);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        local_k_7 = fma((scalar_t)(0.73029673099517822) * w_i_45, x_j_7 * go_v_175, local_k_7);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        local_k_7 = fma((scalar_t)(0.57735025882720947) * w_i_44, x_j_8 * go_v_176, local_k_7);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        local_k_7 = fma((scalar_t)(0.57735025882720947) * w_i_45, x_j_9 * go_v_177, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        local_k_7 = fma((scalar_t)(0.45643547177314758) * w_i_49, x_j_16 * go_v_201, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        local_k_7 = fma((scalar_t)(0.45643547177314758) * w_i_49, x_j_15 * go_v_202, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_202 = grad_out[go_base + ((int64_t)202 << 5)];
        local_k_7 = fma((scalar_t)(0.35355338454246521) * w_i_49, x_j_17 * go_v_202, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_7 = fma((scalar_t)(0.35355338454246521) * w_i_49, x_j_16 * go_v_203, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        local_k_7 = fma((scalar_t)(0.18257418274879456) * w_i_49, x_j_19 * go_v_204, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_7 = fma((scalar_t)(0.18257418274879456) * w_i_49, x_j_18 * go_v_205, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_7 = fma((scalar_t)(0.35355338454246521) * w_i_49, x_j_20 * go_v_205, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        local_k_7 = fma((scalar_t)(0.35355338454246521) * w_i_49, x_j_19 * go_v_206, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        local_k_7 = fma((scalar_t)(0.45643547177314758) * w_i_49, x_j_21 * go_v_206, local_k_7);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        local_k_7 = fma((scalar_t)(0.45643547177314758) * w_i_49, x_j_20 * go_v_207, local_k_7);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        local_k_7 = fma((scalar_t)(-0.5) * w_i_52, x_j_14 * go_v_222, local_k_7);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        local_k_7 = fma((scalar_t)(0.44721359014511108) * w_i_52, x_j_12 * go_v_224, local_k_7);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        local_k_7 = fma((scalar_t)(0.3872983455657959) * w_i_52, x_j_14 * go_v_224, local_k_7);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        local_k_7 = fma((scalar_t)(-0.63245552778244019) * w_i_52, x_j_11 * go_v_225, local_k_7);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        local_k_7 = fma((scalar_t)(-0.3872983455657959) * w_i_52, x_j_10 * go_v_226, local_k_7);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        local_k_7 = fma((scalar_t)(0.5) * w_i_52, x_j_10 * go_v_228, local_k_7);
    }
    scalar_t sum_k_7 = warp_sum(local_k_7);
    if (lane == 0) grad_y[gy_base + 7] += sum_k_7;

    // ---- grad_y chunk 8 ----
    scalar_t local_k_8 = scalar_t(0);
    {
        scalar_t w_i_6  = w[w_base + ((int64_t)6 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_8 = fma((scalar_t)(0.44721359014511108) * w_i_6, x_j_14 * go_v_6, local_k_8);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_26 = grad_out[go_base + ((int64_t)26 << 5)];
        local_k_8 = fma((scalar_t)(-0.54772257804870605) * w_i_14, x_j_4 * go_v_26, local_k_8);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_27 = grad_out[go_base + ((int64_t)27 << 5)];
        local_k_8 = fma((scalar_t)(-0.54772257804870605) * w_i_15, x_j_5 * go_v_27, local_k_8);
    }
    {
        scalar_t w_i_14  = w[w_base + ((int64_t)14 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_30 = grad_out[go_base + ((int64_t)30 << 5)];
        local_k_8 = fma((scalar_t)(0.54772257804870605) * w_i_14, x_j_8 * go_v_30, local_k_8);
    }
    {
        scalar_t w_i_15  = w[w_base + ((int64_t)15 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_31 = grad_out[go_base + ((int64_t)31 << 5)];
        local_k_8 = fma((scalar_t)(0.54772257804870605) * w_i_15, x_j_9 * go_v_31, local_k_8);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_8 = fma((scalar_t)(0.46291005611419678) * w_i_18, x_j_15 * go_v_38, local_k_8);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_38 = grad_out[go_base + ((int64_t)38 << 5)];
        local_k_8 = fma((scalar_t)(0.11952286213636398) * w_i_18, x_j_17 * go_v_38, local_k_8);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_39 = grad_out[go_base + ((int64_t)39 << 5)];
        local_k_8 = fma((scalar_t)(0.37796446681022644) * w_i_18, x_j_20 * go_v_39, local_k_8);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_8 = fma((scalar_t)(-0.11952286213636398) * w_i_18, x_j_19 * go_v_40, local_k_8);
    }
    {
        scalar_t w_i_18  = w[w_base + ((int64_t)18 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_40 = grad_out[go_base + ((int64_t)40 << 5)];
        local_k_8 = fma((scalar_t)(0.46291005611419678) * w_i_18, x_j_21 * go_v_40, local_k_8);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_47 = grad_out[go_base + ((int64_t)47 << 5)];
        local_k_8 = fma((scalar_t)(0.31622776389122009) * w_i_21, x_j_13 * go_v_47, local_k_8);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_48 = grad_out[go_base + ((int64_t)48 << 5)];
        local_k_8 = fma((scalar_t)(-0.63245552778244019) * w_i_21, x_j_10 * go_v_48, local_k_8);
    }
    {
        scalar_t w_i_21  = w[w_base + ((int64_t)21 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_49 = grad_out[go_base + ((int64_t)49 << 5)];
        local_k_8 = fma((scalar_t)(0.31622776389122009) * w_i_21, x_j_11 * go_v_49, local_k_8);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_53 = grad_out[go_base + ((int64_t)53 << 5)];
        local_k_8 = fma((scalar_t)(0.81649655103683472) * w_i_23, x_j_6 * go_v_53, local_k_8);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_54 = grad_out[go_base + ((int64_t)54 << 5)];
        local_k_8 = fma((scalar_t)(0.81649655103683472) * w_i_24, x_j_7 * go_v_54, local_k_8);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_55 = grad_out[go_base + ((int64_t)55 << 5)];
        local_k_8 = fma((scalar_t)(-0.40824827551841736) * w_i_23, x_j_8 * go_v_55, local_k_8);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_56 = grad_out[go_base + ((int64_t)56 << 5)];
        local_k_8 = fma((scalar_t)(-0.40824827551841736) * w_i_24, x_j_9 * go_v_56, local_k_8);
    }
    {
        scalar_t w_i_23  = w[w_base + ((int64_t)23 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_59 = grad_out[go_base + ((int64_t)59 << 5)];
        local_k_8 = fma((scalar_t)(-0.40824827551841736) * w_i_23, x_j_4 * go_v_59, local_k_8);
    }
    {
        scalar_t w_i_24  = w[w_base + ((int64_t)24 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_60 = grad_out[go_base + ((int64_t)60 << 5)];
        local_k_8 = fma((scalar_t)(-0.40824827551841736) * w_i_24, x_j_5 * go_v_60, local_k_8);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_73 = grad_out[go_base + ((int64_t)73 << 5)];
        local_k_8 = fma((scalar_t)(-0.26726123690605164) * w_i_27, x_j_18 * go_v_73, local_k_8);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        local_k_8 = fma((scalar_t)(0.32732683420181274) * w_i_27, x_j_19 * go_v_74, local_k_8);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_74 = grad_out[go_base + ((int64_t)74 << 5)];
        local_k_8 = fma((scalar_t)(0.42257711291313171) * w_i_27, x_j_21 * go_v_74, local_k_8);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_75 = grad_out[go_base + ((int64_t)75 << 5)];
        local_k_8 = fma((scalar_t)(-0.59761428833007812) * w_i_27, x_j_16 * go_v_75, local_k_8);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        local_k_8 = fma((scalar_t)(-0.42257711291313171) * w_i_27, x_j_15 * go_v_76, local_k_8);
    }
    {
        scalar_t w_i_27  = w[w_base + ((int64_t)27 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_76 = grad_out[go_base + ((int64_t)76 << 5)];
        local_k_8 = fma((scalar_t)(0.32732683420181274) * w_i_27, x_j_17 * go_v_76, local_k_8);
    }
    {
        scalar_t w_i_28  = w[w_base + ((int64_t)28 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_94 = grad_out[go_base + ((int64_t)94 << 5)];
        local_k_8 = fma((scalar_t)(1.0) * w_i_28, x_j_0 * go_v_94, local_k_8);
    }
    {
        scalar_t w_i_29  = w[w_base + ((int64_t)29 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_95 = grad_out[go_base + ((int64_t)95 << 5)];
        local_k_8 = fma((scalar_t)(1.0) * w_i_29, x_j_1 * go_v_95, local_k_8);
    }
    {
        scalar_t w_i_30  = w[w_base + ((int64_t)30 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_96 = grad_out[go_base + ((int64_t)96 << 5)];
        local_k_8 = fma((scalar_t)(1.0) * w_i_30, x_j_2 * go_v_96, local_k_8);
    }
    {
        scalar_t w_i_31  = w[w_base + ((int64_t)31 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_97 = grad_out[go_base + ((int64_t)97 << 5)];
        local_k_8 = fma((scalar_t)(1.0) * w_i_31, x_j_3 * go_v_97, local_k_8);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_124 = grad_out[go_base + ((int64_t)124 << 5)];
        local_k_8 = fma((scalar_t)(-0.46291005611419678) * w_i_37, x_j_11 * go_v_124, local_k_8);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_125 = grad_out[go_base + ((int64_t)125 << 5)];
        local_k_8 = fma((scalar_t)(-0.53452247381210327) * w_i_37, x_j_14 * go_v_125, local_k_8);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_126 = grad_out[go_base + ((int64_t)126 << 5)];
        local_k_8 = fma((scalar_t)(0.46291005611419678) * w_i_37, x_j_13 * go_v_126, local_k_8);
    }
    {
        scalar_t w_i_37  = w[w_base + ((int64_t)37 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_127 = grad_out[go_base + ((int64_t)127 << 5)];
        local_k_8 = fma((scalar_t)(-0.53452247381210327) * w_i_37, x_j_12 * go_v_127, local_k_8);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_166 = grad_out[go_base + ((int64_t)166 << 5)];
        local_k_8 = fma((scalar_t)(0.70710676908493042) * w_i_44, x_j_4 * go_v_166, local_k_8);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_167 = grad_out[go_base + ((int64_t)167 << 5)];
        local_k_8 = fma((scalar_t)(0.70710676908493042) * w_i_45, x_j_5 * go_v_167, local_k_8);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_170 = grad_out[go_base + ((int64_t)170 << 5)];
        local_k_8 = fma((scalar_t)(0.18257418274879456) * w_i_44, x_j_4 * go_v_170, local_k_8);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_171 = grad_out[go_base + ((int64_t)171 << 5)];
        local_k_8 = fma((scalar_t)(0.18257418274879456) * w_i_45, x_j_5 * go_v_171, local_k_8);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_174 = grad_out[go_base + ((int64_t)174 << 5)];
        local_k_8 = fma((scalar_t)(-0.18257418274879456) * w_i_44, x_j_8 * go_v_174, local_k_8);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_175 = grad_out[go_base + ((int64_t)175 << 5)];
        local_k_8 = fma((scalar_t)(-0.18257418274879456) * w_i_45, x_j_9 * go_v_175, local_k_8);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_176 = grad_out[go_base + ((int64_t)176 << 5)];
        local_k_8 = fma((scalar_t)(0.57735025882720947) * w_i_44, x_j_6 * go_v_176, local_k_8);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_177 = grad_out[go_base + ((int64_t)177 << 5)];
        local_k_8 = fma((scalar_t)(0.57735025882720947) * w_i_45, x_j_7 * go_v_177, local_k_8);
    }
    {
        scalar_t w_i_44  = w[w_base + ((int64_t)44 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_178 = grad_out[go_base + ((int64_t)178 << 5)];
        local_k_8 = fma((scalar_t)(0.70710676908493042) * w_i_44, x_j_8 * go_v_178, local_k_8);
    }
    {
        scalar_t w_i_45  = w[w_base + ((int64_t)45 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_179 = grad_out[go_base + ((int64_t)179 << 5)];
        local_k_8 = fma((scalar_t)(0.70710676908493042) * w_i_45, x_j_9 * go_v_179, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_201 = grad_out[go_base + ((int64_t)201 << 5)];
        local_k_8 = fma((scalar_t)(-0.28867512941360474) * w_i_49, x_j_17 * go_v_201, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_8 = fma((scalar_t)(-0.28867512941360474) * w_i_49, x_j_15 * go_v_203, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_203 = grad_out[go_base + ((int64_t)203 << 5)];
        local_k_8 = fma((scalar_t)(-0.44721359014511108) * w_i_49, x_j_17 * go_v_203, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_204 = grad_out[go_base + ((int64_t)204 << 5)];
        local_k_8 = fma((scalar_t)(-0.57735025882720947) * w_i_49, x_j_20 * go_v_204, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_8 = fma((scalar_t)(0.44721359014511108) * w_i_49, x_j_19 * go_v_205, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_205 = grad_out[go_base + ((int64_t)205 << 5)];
        local_k_8 = fma((scalar_t)(-0.28867512941360474) * w_i_49, x_j_21 * go_v_205, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_206 = grad_out[go_base + ((int64_t)206 << 5)];
        local_k_8 = fma((scalar_t)(-0.57735025882720947) * w_i_49, x_j_18 * go_v_206, local_k_8);
    }
    {
        scalar_t w_i_49  = w[w_base + ((int64_t)49 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_207 = grad_out[go_base + ((int64_t)207 << 5)];
        local_k_8 = fma((scalar_t)(-0.28867512941360474) * w_i_49, x_j_19 * go_v_207, local_k_8);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_222 = grad_out[go_base + ((int64_t)222 << 5)];
        local_k_8 = fma((scalar_t)(0.5) * w_i_52, x_j_13 * go_v_222, local_k_8);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_223 = grad_out[go_base + ((int64_t)223 << 5)];
        local_k_8 = fma((scalar_t)(0.70710676908493042) * w_i_52, x_j_12 * go_v_223, local_k_8);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_224 = grad_out[go_base + ((int64_t)224 << 5)];
        local_k_8 = fma((scalar_t)(-0.3872983455657959) * w_i_52, x_j_13 * go_v_224, local_k_8);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_225 = grad_out[go_base + ((int64_t)225 << 5)];
        local_k_8 = fma((scalar_t)(0.31622776389122009) * w_i_52, x_j_10 * go_v_225, local_k_8);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_226 = grad_out[go_base + ((int64_t)226 << 5)];
        local_k_8 = fma((scalar_t)(-0.3872983455657959) * w_i_52, x_j_11 * go_v_226, local_k_8);
    }
    {
        scalar_t w_i_52  = w[w_base + ((int64_t)52 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_228 = grad_out[go_base + ((int64_t)228 << 5)];
        local_k_8 = fma((scalar_t)(-0.5) * w_i_52, x_j_11 * go_v_228, local_k_8);
    }
    scalar_t sum_k_8 = warp_sum(local_k_8);
    if (lane == 0) grad_y[gy_base + 8] += sum_k_8;

    // ---- grad_y chunk 9 ----
    scalar_t local_k_9 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_9 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_15 * go_v_7, local_k_9);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_9 = fma((scalar_t)(0.46291005611419678) * w_i_17, x_j_14 * go_v_35, local_k_9);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_9 = fma((scalar_t)(0.46291005611419678) * w_i_17, x_j_10 * go_v_37, local_k_9);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_9 = fma((scalar_t)(0.23145502805709839) * w_i_22, x_j_16 * go_v_50, local_k_9);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        local_k_9 = fma((scalar_t)(0.56694668531417847) * w_i_22, x_j_21 * go_v_51, local_k_9);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_9 = fma((scalar_t)(-0.23145502805709839) * w_i_22, x_j_20 * go_v_52, local_k_9);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        local_k_9 = fma((scalar_t)(0.42257711291313171) * w_i_26, x_j_11 * go_v_68, local_k_9);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        local_k_9 = fma((scalar_t)(-0.42257711291313171) * w_i_26, x_j_10 * go_v_69, local_k_9);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        local_k_9 = fma((scalar_t)(0.42257711291313171) * w_i_26, x_j_14 * go_v_71, local_k_9);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        local_k_9 = fma((scalar_t)(-0.42257711291313171) * w_i_26, x_j_13 * go_v_72, local_k_9);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        local_k_9 = fma((scalar_t)(0.59761428833007812) * w_i_34, x_j_8 * go_v_108, local_k_9);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        local_k_9 = fma((scalar_t)(0.59761428833007812) * w_i_35, x_j_9 * go_v_109, local_k_9);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        local_k_9 = fma((scalar_t)(0.59761428833007812) * w_i_34, x_j_4 * go_v_116, local_k_9);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        local_k_9 = fma((scalar_t)(0.59761428833007812) * w_i_35, x_j_5 * go_v_117, local_k_9);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_9 = fma((scalar_t)(-0.24397502839565277) * w_i_39, x_j_19 * go_v_133, local_k_9);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_9 = fma((scalar_t)(0.38575837016105652) * w_i_39, x_j_20 * go_v_134, local_k_9);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        local_k_9 = fma((scalar_t)(-0.54554474353790283) * w_i_39, x_j_15 * go_v_135, local_k_9);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_9 = fma((scalar_t)(0.38575837016105652) * w_i_39, x_j_16 * go_v_136, local_k_9);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_9 = fma((scalar_t)(-0.24397502839565277) * w_i_39, x_j_17 * go_v_137, local_k_9);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_138 = grad_out[go_base + ((int64_t)138 << 5)];
        local_k_9 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_138, local_k_9);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_139 = grad_out[go_base + ((int64_t)139 << 5)];
        local_k_9 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_139, local_k_9);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_140 = grad_out[go_base + ((int64_t)140 << 5)];
        local_k_9 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_140, local_k_9);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_141 = grad_out[go_base + ((int64_t)141 << 5)];
        local_k_9 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_141, local_k_9);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        local_k_9 = fma((scalar_t)(-0.64549720287322998) * w_i_47, x_j_12 * go_v_187, local_k_9);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        local_k_9 = fma((scalar_t)(0.45643547177314758) * w_i_47, x_j_13 * go_v_188, local_k_9);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_9 = fma((scalar_t)(-0.28867512941360474) * w_i_47, x_j_14 * go_v_189, local_k_9);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_9 = fma((scalar_t)(-0.28867512941360474) * w_i_47, x_j_10 * go_v_191, local_k_9);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        local_k_9 = fma((scalar_t)(0.45643547177314758) * w_i_47, x_j_11 * go_v_192, local_k_9);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        local_k_9 = fma((scalar_t)(-0.35355338454246521) * w_i_50, x_j_4 * go_v_210, local_k_9);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        local_k_9 = fma((scalar_t)(-0.35355338454246521) * w_i_51, x_j_5 * go_v_211, local_k_9);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        local_k_9 = fma((scalar_t)(0.35355338454246521) * w_i_50, x_j_8 * go_v_218, local_k_9);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        local_k_9 = fma((scalar_t)(0.35355338454246521) * w_i_51, x_j_9 * go_v_219, local_k_9);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        local_k_9 = fma((scalar_t)(-0.86602538824081421) * w_i_50, x_j_6 * go_v_220, local_k_9);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        local_k_9 = fma((scalar_t)(-0.86602538824081421) * w_i_51, x_j_7 * go_v_221, local_k_9);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        local_k_9 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_17 * go_v_237, local_k_9);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        local_k_9 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_16 * go_v_238, local_k_9);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        local_k_9 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_21 * go_v_239, local_k_9);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        local_k_9 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_20 * go_v_240, local_k_9);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        local_k_9 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_19 * go_v_241, local_k_9);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        local_k_9 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_18 * go_v_242, local_k_9);
    }
    scalar_t sum_k_9 = warp_sum(local_k_9);
    if (lane == 0) grad_y[gy_base + 9] += sum_k_9;

    // ---- grad_y chunk 10 ----
    scalar_t local_k_10 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_10 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_16 * go_v_7, local_k_10);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_10 = fma((scalar_t)(0.37796446681022644) * w_i_17, x_j_13 * go_v_35, local_k_10);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        local_k_10 = fma((scalar_t)(0.37796446681022644) * w_i_17, x_j_10 * go_v_36, local_k_10);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_10 = fma((scalar_t)(0.37796446681022644) * w_i_17, x_j_11 * go_v_37, local_k_10);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_10 = fma((scalar_t)(-0.23145502805709839) * w_i_22, x_j_15 * go_v_50, local_k_10);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_10 = fma((scalar_t)(0.29880714416503906) * w_i_22, x_j_17 * go_v_50, local_k_10);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        local_k_10 = fma((scalar_t)(0.37796449661254883) * w_i_22, x_j_20 * go_v_51, local_k_10);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_10 = fma((scalar_t)(-0.29880714416503906) * w_i_22, x_j_19 * go_v_52, local_k_10);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_10 = fma((scalar_t)(-0.23145502805709839) * w_i_22, x_j_21 * go_v_52, local_k_10);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        local_k_10 = fma((scalar_t)(0.59761428833007812) * w_i_26, x_j_14 * go_v_70, local_k_10);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        local_k_10 = fma((scalar_t)(-0.59761428833007812) * w_i_26, x_j_12 * go_v_72, local_k_10);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        local_k_10 = fma((scalar_t)(0.48795005679130554) * w_i_34, x_j_6 * go_v_108, local_k_10);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        local_k_10 = fma((scalar_t)(0.48795005679130554) * w_i_35, x_j_7 * go_v_109, local_k_10);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        local_k_10 = fma((scalar_t)(0.48795005679130554) * w_i_34, x_j_8 * go_v_110, local_k_10);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        local_k_10 = fma((scalar_t)(0.48795005679130554) * w_i_35, x_j_9 * go_v_111, local_k_10);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        local_k_10 = fma((scalar_t)(0.48795005679130554) * w_i_34, x_j_4 * go_v_114, local_k_10);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        local_k_10 = fma((scalar_t)(0.48795005679130554) * w_i_35, x_j_5 * go_v_115, local_k_10);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_10 = fma((scalar_t)(-0.48795005679130554) * w_i_39, x_j_18 * go_v_133, local_k_10);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_10 = fma((scalar_t)(0.29880714416503906) * w_i_39, x_j_19 * go_v_134, local_k_10);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_10 = fma((scalar_t)(-0.38575837016105652) * w_i_39, x_j_21 * go_v_134, local_k_10);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_10 = fma((scalar_t)(0.38575837016105652) * w_i_39, x_j_15 * go_v_136, local_k_10);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_10 = fma((scalar_t)(0.29880714416503906) * w_i_39, x_j_17 * go_v_136, local_k_10);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_142 = grad_out[go_base + ((int64_t)142 << 5)];
        local_k_10 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_142, local_k_10);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_143 = grad_out[go_base + ((int64_t)143 << 5)];
        local_k_10 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_143, local_k_10);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_144 = grad_out[go_base + ((int64_t)144 << 5)];
        local_k_10 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_144, local_k_10);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_145 = grad_out[go_base + ((int64_t)145 << 5)];
        local_k_10 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_145, local_k_10);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        local_k_10 = fma((scalar_t)(0.45643547177314758) * w_i_47, x_j_13 * go_v_187, local_k_10);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_10 = fma((scalar_t)(0.35355338454246521) * w_i_47, x_j_13 * go_v_189, local_k_10);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        local_k_10 = fma((scalar_t)(-0.57735025882720947) * w_i_47, x_j_10 * go_v_190, local_k_10);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_10 = fma((scalar_t)(0.35355338454246521) * w_i_47, x_j_11 * go_v_191, local_k_10);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        local_k_10 = fma((scalar_t)(-0.45643547177314758) * w_i_47, x_j_11 * go_v_193, local_k_10);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        local_k_10 = fma((scalar_t)(0.35355338454246521) * w_i_50, x_j_4 * go_v_208, local_k_10);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        local_k_10 = fma((scalar_t)(0.35355338454246521) * w_i_51, x_j_5 * go_v_209, local_k_10);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        local_k_10 = fma((scalar_t)(-0.45643547177314758) * w_i_50, x_j_4 * go_v_212, local_k_10);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        local_k_10 = fma((scalar_t)(-0.45643547177314758) * w_i_51, x_j_5 * go_v_213, local_k_10);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        local_k_10 = fma((scalar_t)(0.45643547177314758) * w_i_50, x_j_8 * go_v_216, local_k_10);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        local_k_10 = fma((scalar_t)(0.45643547177314758) * w_i_51, x_j_9 * go_v_217, local_k_10);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        local_k_10 = fma((scalar_t)(-0.57735025882720947) * w_i_50, x_j_6 * go_v_218, local_k_10);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        local_k_10 = fma((scalar_t)(-0.57735025882720947) * w_i_51, x_j_7 * go_v_219, local_k_10);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        local_k_10 = fma((scalar_t)(0.35355338454246521) * w_i_50, x_j_8 * go_v_220, local_k_10);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        local_k_10 = fma((scalar_t)(0.35355338454246521) * w_i_51, x_j_9 * go_v_221, local_k_10);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        local_k_10 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_17 * go_v_236, local_k_10);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        local_k_10 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_15 * go_v_238, local_k_10);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        local_k_10 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_20 * go_v_239, local_k_10);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        local_k_10 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_21 * go_v_240, local_k_10);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        local_k_10 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_18 * go_v_241, local_k_10);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        local_k_10 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_19 * go_v_242, local_k_10);
    }
    scalar_t sum_k_10 = warp_sum(local_k_10);
    if (lane == 0) grad_y[gy_base + 10] += sum_k_10;

    // ---- grad_y chunk 11 ----
    scalar_t local_k_11 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_11 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_17 * go_v_7, local_k_11);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_11 = fma((scalar_t)(0.41403934359550476) * w_i_17, x_j_12 * go_v_35, local_k_11);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_11 = fma((scalar_t)(0.11952286213636398) * w_i_17, x_j_14 * go_v_35, local_k_11);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        local_k_11 = fma((scalar_t)(0.47809144854545593) * w_i_17, x_j_11 * go_v_36, local_k_11);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_11 = fma((scalar_t)(-0.11952286213636398) * w_i_17, x_j_10 * go_v_37, local_k_11);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_11 = fma((scalar_t)(-0.29880714416503906) * w_i_22, x_j_16 * go_v_50, local_k_11);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        local_k_11 = fma((scalar_t)(0.18898224830627441) * w_i_22, x_j_19 * go_v_51, local_k_11);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_11 = fma((scalar_t)(-0.46291005611419678) * w_i_22, x_j_18 * go_v_52, local_k_11);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_11 = fma((scalar_t)(-0.29880714416503906) * w_i_22, x_j_20 * go_v_52, local_k_11);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        local_k_11 = fma((scalar_t)(0.32732683420181274) * w_i_26, x_j_11 * go_v_68, local_k_11);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        local_k_11 = fma((scalar_t)(-0.32732683420181274) * w_i_26, x_j_10 * go_v_69, local_k_11);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        local_k_11 = fma((scalar_t)(0.37796446681022644) * w_i_26, x_j_13 * go_v_70, local_k_11);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        local_k_11 = fma((scalar_t)(-0.37796446681022644) * w_i_26, x_j_12 * go_v_71, local_k_11);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        local_k_11 = fma((scalar_t)(-0.32732683420181274) * w_i_26, x_j_14 * go_v_71, local_k_11);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        local_k_11 = fma((scalar_t)(0.32732683420181274) * w_i_26, x_j_13 * go_v_72, local_k_11);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        local_k_11 = fma((scalar_t)(-0.15430334210395813) * w_i_34, x_j_8 * go_v_108, local_k_11);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        local_k_11 = fma((scalar_t)(-0.15430334210395813) * w_i_35, x_j_9 * go_v_109, local_k_11);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        local_k_11 = fma((scalar_t)(0.61721336841583252) * w_i_34, x_j_6 * go_v_110, local_k_11);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        local_k_11 = fma((scalar_t)(0.61721336841583252) * w_i_35, x_j_7 * go_v_111, local_k_11);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        local_k_11 = fma((scalar_t)(0.53452247381210327) * w_i_34, x_j_4 * go_v_112, local_k_11);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        local_k_11 = fma((scalar_t)(0.53452247381210327) * w_i_35, x_j_5 * go_v_113, local_k_11);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        local_k_11 = fma((scalar_t)(0.15430334210395813) * w_i_34, x_j_4 * go_v_116, local_k_11);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        local_k_11 = fma((scalar_t)(0.15430334210395813) * w_i_35, x_j_5 * go_v_117, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_11 = fma((scalar_t)(0.37796446681022644) * w_i_39, x_j_19 * go_v_133, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_11 = fma((scalar_t)(0.24397502839565277) * w_i_39, x_j_21 * go_v_133, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_11 = fma((scalar_t)(0.15430334210395813) * w_i_39, x_j_18 * go_v_134, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_11 = fma((scalar_t)(-0.29880714416503906) * w_i_39, x_j_20 * go_v_134, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        local_k_11 = fma((scalar_t)(0.32732683420181274) * w_i_39, x_j_17 * go_v_135, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_11 = fma((scalar_t)(0.29880714416503906) * w_i_39, x_j_16 * go_v_136, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_11 = fma((scalar_t)(-0.24397502839565277) * w_i_39, x_j_15 * go_v_137, local_k_11);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_11 = fma((scalar_t)(-0.37796446681022644) * w_i_39, x_j_17 * go_v_137, local_k_11);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_146 = grad_out[go_base + ((int64_t)146 << 5)];
        local_k_11 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_146, local_k_11);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_147 = grad_out[go_base + ((int64_t)147 << 5)];
        local_k_11 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_147, local_k_11);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_148 = grad_out[go_base + ((int64_t)148 << 5)];
        local_k_11 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_148, local_k_11);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_149 = grad_out[go_base + ((int64_t)149 << 5)];
        local_k_11 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_149, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        local_k_11 = fma((scalar_t)(-0.28867512941360474) * w_i_47, x_j_14 * go_v_187, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        local_k_11 = fma((scalar_t)(0.35355338454246521) * w_i_47, x_j_13 * go_v_188, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_11 = fma((scalar_t)(0.3872983455657959) * w_i_47, x_j_12 * go_v_189, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_11 = fma((scalar_t)(-0.44721359014511108) * w_i_47, x_j_14 * go_v_189, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        local_k_11 = fma((scalar_t)(0.18257418274879456) * w_i_47, x_j_11 * go_v_190, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_11 = fma((scalar_t)(0.44721359014511108) * w_i_47, x_j_10 * go_v_191, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        local_k_11 = fma((scalar_t)(-0.35355338454246521) * w_i_47, x_j_11 * go_v_192, local_k_11);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        local_k_11 = fma((scalar_t)(0.28867512941360474) * w_i_47, x_j_10 * go_v_193, local_k_11);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        local_k_11 = fma((scalar_t)(0.45643547177314758) * w_i_50, x_j_4 * go_v_210, local_k_11);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        local_k_11 = fma((scalar_t)(0.45643547177314758) * w_i_51, x_j_5 * go_v_211, local_k_11);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_214 = grad_out[go_base + ((int64_t)214 << 5)];
        local_k_11 = fma((scalar_t)(0.70710676908493042) * w_i_50, x_j_8 * go_v_214, local_k_11);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_215 = grad_out[go_base + ((int64_t)215 << 5)];
        local_k_11 = fma((scalar_t)(0.70710676908493042) * w_i_51, x_j_9 * go_v_215, local_k_11);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        local_k_11 = fma((scalar_t)(-0.28867512941360474) * w_i_50, x_j_6 * go_v_216, local_k_11);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        local_k_11 = fma((scalar_t)(-0.28867512941360474) * w_i_51, x_j_7 * go_v_217, local_k_11);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        local_k_11 = fma((scalar_t)(0.45643547177314758) * w_i_50, x_j_8 * go_v_218, local_k_11);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        local_k_11 = fma((scalar_t)(0.45643547177314758) * w_i_51, x_j_9 * go_v_219, local_k_11);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        local_k_11 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_16 * go_v_236, local_k_11);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        local_k_11 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_15 * go_v_237, local_k_11);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        local_k_11 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_19 * go_v_239, local_k_11);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        local_k_11 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_18 * go_v_240, local_k_11);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        local_k_11 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_21 * go_v_241, local_k_11);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        local_k_11 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_20 * go_v_242, local_k_11);
    }
    scalar_t sum_k_11 = warp_sum(local_k_11);
    if (lane == 0) grad_y[gy_base + 11] += sum_k_11;

    // ---- grad_y chunk 12 ----
    scalar_t local_k_12 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_12 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_18 * go_v_7, local_k_12);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_12 = fma((scalar_t)(-0.29276999831199646) * w_i_17, x_j_11 * go_v_35, local_k_12);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        local_k_12 = fma((scalar_t)(0.50709253549575806) * w_i_17, x_j_12 * go_v_36, local_k_12);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_12 = fma((scalar_t)(-0.29276999831199646) * w_i_17, x_j_13 * go_v_37, local_k_12);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_12 = fma((scalar_t)(-0.46291005611419678) * w_i_22, x_j_19 * go_v_50, local_k_12);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_12 = fma((scalar_t)(0.46291005611419678) * w_i_22, x_j_17 * go_v_52, local_k_12);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        local_k_12 = fma((scalar_t)(0.26726123690605164) * w_i_26, x_j_14 * go_v_68, local_k_12);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        local_k_12 = fma((scalar_t)(-0.53452247381210327) * w_i_26, x_j_13 * go_v_69, local_k_12);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        local_k_12 = fma((scalar_t)(0.53452247381210327) * w_i_26, x_j_11 * go_v_71, local_k_12);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        local_k_12 = fma((scalar_t)(-0.26726123690605164) * w_i_26, x_j_10 * go_v_72, local_k_12);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        local_k_12 = fma((scalar_t)(-0.37796446681022644) * w_i_34, x_j_4 * go_v_110, local_k_12);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        local_k_12 = fma((scalar_t)(-0.37796446681022644) * w_i_35, x_j_5 * go_v_111, local_k_12);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        local_k_12 = fma((scalar_t)(0.65465366840362549) * w_i_34, x_j_6 * go_v_112, local_k_12);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        local_k_12 = fma((scalar_t)(0.65465366840362549) * w_i_35, x_j_7 * go_v_113, local_k_12);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        local_k_12 = fma((scalar_t)(-0.37796446681022644) * w_i_34, x_j_8 * go_v_114, local_k_12);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        local_k_12 = fma((scalar_t)(-0.37796446681022644) * w_i_35, x_j_9 * go_v_115, local_k_12);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_12 = fma((scalar_t)(-0.48795005679130554) * w_i_39, x_j_16 * go_v_133, local_k_12);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_12 = fma((scalar_t)(0.15430334210395813) * w_i_39, x_j_17 * go_v_134, local_k_12);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        local_k_12 = fma((scalar_t)(0.43643578886985779) * w_i_39, x_j_18 * go_v_135, local_k_12);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_12 = fma((scalar_t)(0.15430334210395813) * w_i_39, x_j_19 * go_v_136, local_k_12);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_12 = fma((scalar_t)(-0.48795005679130554) * w_i_39, x_j_20 * go_v_137, local_k_12);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_150 = grad_out[go_base + ((int64_t)150 << 5)];
        local_k_12 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_150, local_k_12);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_151 = grad_out[go_base + ((int64_t)151 << 5)];
        local_k_12 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_151, local_k_12);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_152 = grad_out[go_base + ((int64_t)152 << 5)];
        local_k_12 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_152, local_k_12);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_153 = grad_out[go_base + ((int64_t)153 << 5)];
        local_k_12 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_153, local_k_12);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        local_k_12 = fma((scalar_t)(-0.57735025882720947) * w_i_47, x_j_10 * go_v_188, local_k_12);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_12 = fma((scalar_t)(0.18257418274879456) * w_i_47, x_j_11 * go_v_189, local_k_12);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        local_k_12 = fma((scalar_t)(0.51639777421951294) * w_i_47, x_j_12 * go_v_190, local_k_12);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_12 = fma((scalar_t)(0.18257418274879456) * w_i_47, x_j_13 * go_v_191, local_k_12);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        local_k_12 = fma((scalar_t)(-0.57735025882720947) * w_i_47, x_j_14 * go_v_192, local_k_12);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        local_k_12 = fma((scalar_t)(-0.70710676908493042) * w_i_50, x_j_8 * go_v_212, local_k_12);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        local_k_12 = fma((scalar_t)(-0.70710676908493042) * w_i_51, x_j_9 * go_v_213, local_k_12);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        local_k_12 = fma((scalar_t)(0.70710676908493042) * w_i_50, x_j_4 * go_v_216, local_k_12);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        local_k_12 = fma((scalar_t)(0.70710676908493042) * w_i_51, x_j_5 * go_v_217, local_k_12);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        local_k_12 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_21 * go_v_236, local_k_12);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        local_k_12 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_20 * go_v_237, local_k_12);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        local_k_12 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_19 * go_v_238, local_k_12);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        local_k_12 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_17 * go_v_240, local_k_12);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        local_k_12 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_16 * go_v_241, local_k_12);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        local_k_12 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_15 * go_v_242, local_k_12);
    }
    scalar_t sum_k_12 = warp_sum(local_k_12);
    if (lane == 0) grad_y[gy_base + 12] += sum_k_12;

    // ---- grad_y chunk 13 ----
    scalar_t local_k_13 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_13 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_19 * go_v_7, local_k_13);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_13 = fma((scalar_t)(-0.11952286213636398) * w_i_17, x_j_10 * go_v_35, local_k_13);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        local_k_13 = fma((scalar_t)(0.47809144854545593) * w_i_17, x_j_13 * go_v_36, local_k_13);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_13 = fma((scalar_t)(0.41403934359550476) * w_i_17, x_j_12 * go_v_37, local_k_13);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_13 = fma((scalar_t)(-0.11952286213636398) * w_i_17, x_j_14 * go_v_37, local_k_13);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_13 = fma((scalar_t)(0.46291005611419678) * w_i_22, x_j_18 * go_v_50, local_k_13);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_13 = fma((scalar_t)(-0.29880714416503906) * w_i_22, x_j_20 * go_v_50, local_k_13);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        local_k_13 = fma((scalar_t)(-0.18898224830627441) * w_i_22, x_j_17 * go_v_51, local_k_13);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_13 = fma((scalar_t)(0.29880714416503906) * w_i_22, x_j_16 * go_v_52, local_k_13);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        local_k_13 = fma((scalar_t)(-0.32732683420181274) * w_i_26, x_j_13 * go_v_68, local_k_13);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        local_k_13 = fma((scalar_t)(0.37796446681022644) * w_i_26, x_j_12 * go_v_69, local_k_13);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        local_k_13 = fma((scalar_t)(-0.32732683420181274) * w_i_26, x_j_14 * go_v_69, local_k_13);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        local_k_13 = fma((scalar_t)(-0.37796446681022644) * w_i_26, x_j_11 * go_v_70, local_k_13);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        local_k_13 = fma((scalar_t)(0.32732683420181274) * w_i_26, x_j_10 * go_v_71, local_k_13);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        local_k_13 = fma((scalar_t)(0.32732683420181274) * w_i_26, x_j_11 * go_v_72, local_k_13);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        local_k_13 = fma((scalar_t)(-0.15430334210395813) * w_i_34, x_j_4 * go_v_108, local_k_13);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        local_k_13 = fma((scalar_t)(-0.15430334210395813) * w_i_35, x_j_5 * go_v_109, local_k_13);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_112 = grad_out[go_base + ((int64_t)112 << 5)];
        local_k_13 = fma((scalar_t)(0.53452247381210327) * w_i_34, x_j_8 * go_v_112, local_k_13);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_113 = grad_out[go_base + ((int64_t)113 << 5)];
        local_k_13 = fma((scalar_t)(0.53452247381210327) * w_i_35, x_j_9 * go_v_113, local_k_13);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        local_k_13 = fma((scalar_t)(0.61721336841583252) * w_i_34, x_j_6 * go_v_114, local_k_13);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        local_k_13 = fma((scalar_t)(0.61721336841583252) * w_i_35, x_j_7 * go_v_115, local_k_13);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        local_k_13 = fma((scalar_t)(-0.15430334210395813) * w_i_34, x_j_8 * go_v_116, local_k_13);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        local_k_13 = fma((scalar_t)(-0.15430334210395813) * w_i_35, x_j_9 * go_v_117, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_13 = fma((scalar_t)(-0.24397502839565277) * w_i_39, x_j_15 * go_v_133, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_13 = fma((scalar_t)(0.37796446681022644) * w_i_39, x_j_17 * go_v_133, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_13 = fma((scalar_t)(0.29880714416503906) * w_i_39, x_j_16 * go_v_134, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        local_k_13 = fma((scalar_t)(0.32732683420181274) * w_i_39, x_j_19 * go_v_135, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_13 = fma((scalar_t)(0.15430334210395813) * w_i_39, x_j_18 * go_v_136, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_13 = fma((scalar_t)(0.29880714416503906) * w_i_39, x_j_20 * go_v_136, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_13 = fma((scalar_t)(0.37796446681022644) * w_i_39, x_j_19 * go_v_137, local_k_13);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_13 = fma((scalar_t)(-0.24397502839565277) * w_i_39, x_j_21 * go_v_137, local_k_13);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_154 = grad_out[go_base + ((int64_t)154 << 5)];
        local_k_13 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_154, local_k_13);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_155 = grad_out[go_base + ((int64_t)155 << 5)];
        local_k_13 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_155, local_k_13);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_156 = grad_out[go_base + ((int64_t)156 << 5)];
        local_k_13 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_156, local_k_13);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_157 = grad_out[go_base + ((int64_t)157 << 5)];
        local_k_13 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_157, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        local_k_13 = fma((scalar_t)(-0.28867512941360474) * w_i_47, x_j_10 * go_v_187, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        local_k_13 = fma((scalar_t)(0.35355338454246521) * w_i_47, x_j_11 * go_v_188, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_13 = fma((scalar_t)(0.44721359014511108) * w_i_47, x_j_10 * go_v_189, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        local_k_13 = fma((scalar_t)(0.18257418274879456) * w_i_47, x_j_13 * go_v_190, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_13 = fma((scalar_t)(0.3872983455657959) * w_i_47, x_j_12 * go_v_191, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_13 = fma((scalar_t)(0.44721359014511108) * w_i_47, x_j_14 * go_v_191, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        local_k_13 = fma((scalar_t)(0.35355338454246521) * w_i_47, x_j_13 * go_v_192, local_k_13);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        local_k_13 = fma((scalar_t)(-0.28867512941360474) * w_i_47, x_j_14 * go_v_193, local_k_13);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        local_k_13 = fma((scalar_t)(-0.45643547177314758) * w_i_50, x_j_8 * go_v_210, local_k_13);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        local_k_13 = fma((scalar_t)(-0.45643547177314758) * w_i_51, x_j_9 * go_v_211, local_k_13);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        local_k_13 = fma((scalar_t)(0.28867512941360474) * w_i_50, x_j_6 * go_v_212, local_k_13);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        local_k_13 = fma((scalar_t)(0.28867512941360474) * w_i_51, x_j_7 * go_v_213, local_k_13);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_214 = grad_out[go_base + ((int64_t)214 << 5)];
        local_k_13 = fma((scalar_t)(-0.70710676908493042) * w_i_50, x_j_4 * go_v_214, local_k_13);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_215 = grad_out[go_base + ((int64_t)215 << 5)];
        local_k_13 = fma((scalar_t)(-0.70710676908493042) * w_i_51, x_j_5 * go_v_215, local_k_13);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        local_k_13 = fma((scalar_t)(0.45643547177314758) * w_i_50, x_j_4 * go_v_218, local_k_13);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        local_k_13 = fma((scalar_t)(0.45643547177314758) * w_i_51, x_j_5 * go_v_219, local_k_13);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        local_k_13 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_20 * go_v_236, local_k_13);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        local_k_13 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_21 * go_v_237, local_k_13);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        local_k_13 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_18 * go_v_238, local_k_13);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        local_k_13 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_17 * go_v_239, local_k_13);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        local_k_13 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_15 * go_v_241, local_k_13);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        local_k_13 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_16 * go_v_242, local_k_13);
    }
    scalar_t sum_k_13 = warp_sum(local_k_13);
    if (lane == 0) grad_y[gy_base + 13] += sum_k_13;

    // ---- grad_y chunk 14 ----
    scalar_t local_k_14 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_14 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_20 * go_v_7, local_k_14);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_14 = fma((scalar_t)(-0.37796446681022644) * w_i_17, x_j_11 * go_v_35, local_k_14);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_36 = grad_out[go_base + ((int64_t)36 << 5)];
        local_k_14 = fma((scalar_t)(0.37796446681022644) * w_i_17, x_j_14 * go_v_36, local_k_14);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_14 = fma((scalar_t)(0.37796446681022644) * w_i_17, x_j_13 * go_v_37, local_k_14);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_14 = fma((scalar_t)(0.29880714416503906) * w_i_22, x_j_19 * go_v_50, local_k_14);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_14 = fma((scalar_t)(-0.23145502805709839) * w_i_22, x_j_21 * go_v_50, local_k_14);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        local_k_14 = fma((scalar_t)(-0.37796449661254883) * w_i_22, x_j_16 * go_v_51, local_k_14);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_14 = fma((scalar_t)(0.23145502805709839) * w_i_22, x_j_15 * go_v_52, local_k_14);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_14 = fma((scalar_t)(0.29880714416503906) * w_i_22, x_j_17 * go_v_52, local_k_14);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        local_k_14 = fma((scalar_t)(0.59761428833007812) * w_i_26, x_j_12 * go_v_68, local_k_14);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_70 = grad_out[go_base + ((int64_t)70 << 5)];
        local_k_14 = fma((scalar_t)(-0.59761428833007812) * w_i_26, x_j_10 * go_v_70, local_k_14);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_110 = grad_out[go_base + ((int64_t)110 << 5)];
        local_k_14 = fma((scalar_t)(-0.48795005679130554) * w_i_34, x_j_4 * go_v_110, local_k_14);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_111 = grad_out[go_base + ((int64_t)111 << 5)];
        local_k_14 = fma((scalar_t)(-0.48795005679130554) * w_i_35, x_j_5 * go_v_111, local_k_14);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_114 = grad_out[go_base + ((int64_t)114 << 5)];
        local_k_14 = fma((scalar_t)(0.48795005679130554) * w_i_34, x_j_8 * go_v_114, local_k_14);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_115 = grad_out[go_base + ((int64_t)115 << 5)];
        local_k_14 = fma((scalar_t)(0.48795005679130554) * w_i_35, x_j_9 * go_v_115, local_k_14);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        local_k_14 = fma((scalar_t)(0.48795005679130554) * w_i_34, x_j_6 * go_v_116, local_k_14);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        local_k_14 = fma((scalar_t)(0.48795005679130554) * w_i_35, x_j_7 * go_v_117, local_k_14);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_14 = fma((scalar_t)(0.38575837016105652) * w_i_39, x_j_15 * go_v_134, local_k_14);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_14 = fma((scalar_t)(-0.29880714416503906) * w_i_39, x_j_17 * go_v_134, local_k_14);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_14 = fma((scalar_t)(0.29880714416503906) * w_i_39, x_j_19 * go_v_136, local_k_14);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_14 = fma((scalar_t)(0.38575837016105652) * w_i_39, x_j_21 * go_v_136, local_k_14);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_14 = fma((scalar_t)(-0.48795005679130554) * w_i_39, x_j_18 * go_v_137, local_k_14);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_158 = grad_out[go_base + ((int64_t)158 << 5)];
        local_k_14 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_158, local_k_14);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_159 = grad_out[go_base + ((int64_t)159 << 5)];
        local_k_14 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_159, local_k_14);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_160 = grad_out[go_base + ((int64_t)160 << 5)];
        local_k_14 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_160, local_k_14);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_161 = grad_out[go_base + ((int64_t)161 << 5)];
        local_k_14 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_161, local_k_14);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_187 = grad_out[go_base + ((int64_t)187 << 5)];
        local_k_14 = fma((scalar_t)(0.45643547177314758) * w_i_47, x_j_11 * go_v_187, local_k_14);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_14 = fma((scalar_t)(-0.35355338454246521) * w_i_47, x_j_11 * go_v_189, local_k_14);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_190 = grad_out[go_base + ((int64_t)190 << 5)];
        local_k_14 = fma((scalar_t)(-0.57735025882720947) * w_i_47, x_j_14 * go_v_190, local_k_14);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_14 = fma((scalar_t)(0.35355338454246521) * w_i_47, x_j_13 * go_v_191, local_k_14);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        local_k_14 = fma((scalar_t)(0.45643547177314758) * w_i_47, x_j_13 * go_v_193, local_k_14);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        local_k_14 = fma((scalar_t)(-0.35355338454246521) * w_i_50, x_j_8 * go_v_208, local_k_14);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        local_k_14 = fma((scalar_t)(-0.35355338454246521) * w_i_51, x_j_9 * go_v_209, local_k_14);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        local_k_14 = fma((scalar_t)(0.57735025882720947) * w_i_50, x_j_6 * go_v_210, local_k_14);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        local_k_14 = fma((scalar_t)(0.57735025882720947) * w_i_51, x_j_7 * go_v_211, local_k_14);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_212 = grad_out[go_base + ((int64_t)212 << 5)];
        local_k_14 = fma((scalar_t)(-0.45643547177314758) * w_i_50, x_j_8 * go_v_212, local_k_14);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_213 = grad_out[go_base + ((int64_t)213 << 5)];
        local_k_14 = fma((scalar_t)(-0.45643547177314758) * w_i_51, x_j_9 * go_v_213, local_k_14);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_216 = grad_out[go_base + ((int64_t)216 << 5)];
        local_k_14 = fma((scalar_t)(-0.45643547177314758) * w_i_50, x_j_4 * go_v_216, local_k_14);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_217 = grad_out[go_base + ((int64_t)217 << 5)];
        local_k_14 = fma((scalar_t)(-0.45643547177314758) * w_i_51, x_j_5 * go_v_217, local_k_14);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_220 = grad_out[go_base + ((int64_t)220 << 5)];
        local_k_14 = fma((scalar_t)(0.35355338454246521) * w_i_50, x_j_4 * go_v_220, local_k_14);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_221 = grad_out[go_base + ((int64_t)221 << 5)];
        local_k_14 = fma((scalar_t)(0.35355338454246521) * w_i_51, x_j_5 * go_v_221, local_k_14);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        local_k_14 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_19 * go_v_236, local_k_14);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        local_k_14 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_18 * go_v_237, local_k_14);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        local_k_14 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_21 * go_v_238, local_k_14);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        local_k_14 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_16 * go_v_239, local_k_14);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        local_k_14 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_15 * go_v_240, local_k_14);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_242 = grad_out[go_base + ((int64_t)242 << 5)];
        local_k_14 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_17 * go_v_242, local_k_14);
    }
    scalar_t sum_k_14 = warp_sum(local_k_14);
    if (lane == 0) grad_y[gy_base + 14] += sum_k_14;

    // ---- grad_y chunk 15 ----
    scalar_t local_k_15 = scalar_t(0);
    {
        scalar_t w_i_7  = w[w_base + ((int64_t)7 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_15 = fma((scalar_t)(0.37796446681022644) * w_i_7, x_j_21 * go_v_7, local_k_15);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_35 = grad_out[go_base + ((int64_t)35 << 5)];
        local_k_15 = fma((scalar_t)(-0.46291005611419678) * w_i_17, x_j_10 * go_v_35, local_k_15);
    }
    {
        scalar_t w_i_17  = w[w_base + ((int64_t)17 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_37 = grad_out[go_base + ((int64_t)37 << 5)];
        local_k_15 = fma((scalar_t)(0.46291005611419678) * w_i_17, x_j_14 * go_v_37, local_k_15);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_50 = grad_out[go_base + ((int64_t)50 << 5)];
        local_k_15 = fma((scalar_t)(0.23145502805709839) * w_i_22, x_j_20 * go_v_50, local_k_15);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_51 = grad_out[go_base + ((int64_t)51 << 5)];
        local_k_15 = fma((scalar_t)(-0.56694668531417847) * w_i_22, x_j_15 * go_v_51, local_k_15);
    }
    {
        scalar_t w_i_22  = w[w_base + ((int64_t)22 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_52 = grad_out[go_base + ((int64_t)52 << 5)];
        local_k_15 = fma((scalar_t)(0.23145502805709839) * w_i_22, x_j_16 * go_v_52, local_k_15);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_68 = grad_out[go_base + ((int64_t)68 << 5)];
        local_k_15 = fma((scalar_t)(0.42257711291313171) * w_i_26, x_j_13 * go_v_68, local_k_15);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_69 = grad_out[go_base + ((int64_t)69 << 5)];
        local_k_15 = fma((scalar_t)(-0.42257711291313171) * w_i_26, x_j_14 * go_v_69, local_k_15);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_71 = grad_out[go_base + ((int64_t)71 << 5)];
        local_k_15 = fma((scalar_t)(-0.42257711291313171) * w_i_26, x_j_10 * go_v_71, local_k_15);
    }
    {
        scalar_t w_i_26  = w[w_base + ((int64_t)26 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_72 = grad_out[go_base + ((int64_t)72 << 5)];
        local_k_15 = fma((scalar_t)(0.42257711291313171) * w_i_26, x_j_11 * go_v_72, local_k_15);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_108 = grad_out[go_base + ((int64_t)108 << 5)];
        local_k_15 = fma((scalar_t)(-0.59761428833007812) * w_i_34, x_j_4 * go_v_108, local_k_15);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_109 = grad_out[go_base + ((int64_t)109 << 5)];
        local_k_15 = fma((scalar_t)(-0.59761428833007812) * w_i_35, x_j_5 * go_v_109, local_k_15);
    }
    {
        scalar_t w_i_34  = w[w_base + ((int64_t)34 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_116 = grad_out[go_base + ((int64_t)116 << 5)];
        local_k_15 = fma((scalar_t)(0.59761428833007812) * w_i_34, x_j_8 * go_v_116, local_k_15);
    }
    {
        scalar_t w_i_35  = w[w_base + ((int64_t)35 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_117 = grad_out[go_base + ((int64_t)117 << 5)];
        local_k_15 = fma((scalar_t)(0.59761428833007812) * w_i_35, x_j_9 * go_v_117, local_k_15);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_133 = grad_out[go_base + ((int64_t)133 << 5)];
        local_k_15 = fma((scalar_t)(0.24397502839565277) * w_i_39, x_j_17 * go_v_133, local_k_15);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_134 = grad_out[go_base + ((int64_t)134 << 5)];
        local_k_15 = fma((scalar_t)(-0.38575837016105652) * w_i_39, x_j_16 * go_v_134, local_k_15);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_21  = x_all[x_base + ((int64_t)21 << 5)];
        scalar_t go_v_135 = grad_out[go_base + ((int64_t)135 << 5)];
        local_k_15 = fma((scalar_t)(-0.54554474353790283) * w_i_39, x_j_21 * go_v_135, local_k_15);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_136 = grad_out[go_base + ((int64_t)136 << 5)];
        local_k_15 = fma((scalar_t)(0.38575837016105652) * w_i_39, x_j_20 * go_v_136, local_k_15);
    }
    {
        scalar_t w_i_39  = w[w_base + ((int64_t)39 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_137 = grad_out[go_base + ((int64_t)137 << 5)];
        local_k_15 = fma((scalar_t)(-0.24397502839565277) * w_i_39, x_j_19 * go_v_137, local_k_15);
    }
    {
        scalar_t w_i_40  = w[w_base + ((int64_t)40 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_162 = grad_out[go_base + ((int64_t)162 << 5)];
        local_k_15 = fma((scalar_t)(1.0) * w_i_40, x_j_0 * go_v_162, local_k_15);
    }
    {
        scalar_t w_i_41  = w[w_base + ((int64_t)41 << 5)];
        scalar_t x_j_1  = x_all[x_base + ((int64_t)1 << 5)];
        scalar_t go_v_163 = grad_out[go_base + ((int64_t)163 << 5)];
        local_k_15 = fma((scalar_t)(1.0) * w_i_41, x_j_1 * go_v_163, local_k_15);
    }
    {
        scalar_t w_i_42  = w[w_base + ((int64_t)42 << 5)];
        scalar_t x_j_2  = x_all[x_base + ((int64_t)2 << 5)];
        scalar_t go_v_164 = grad_out[go_base + ((int64_t)164 << 5)];
        local_k_15 = fma((scalar_t)(1.0) * w_i_42, x_j_2 * go_v_164, local_k_15);
    }
    {
        scalar_t w_i_43  = w[w_base + ((int64_t)43 << 5)];
        scalar_t x_j_3  = x_all[x_base + ((int64_t)3 << 5)];
        scalar_t go_v_165 = grad_out[go_base + ((int64_t)165 << 5)];
        local_k_15 = fma((scalar_t)(1.0) * w_i_43, x_j_3 * go_v_165, local_k_15);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_11  = x_all[x_base + ((int64_t)11 << 5)];
        scalar_t go_v_188 = grad_out[go_base + ((int64_t)188 << 5)];
        local_k_15 = fma((scalar_t)(-0.45643547177314758) * w_i_47, x_j_11 * go_v_188, local_k_15);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_10  = x_all[x_base + ((int64_t)10 << 5)];
        scalar_t go_v_189 = grad_out[go_base + ((int64_t)189 << 5)];
        local_k_15 = fma((scalar_t)(0.28867512941360474) * w_i_47, x_j_10 * go_v_189, local_k_15);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_14  = x_all[x_base + ((int64_t)14 << 5)];
        scalar_t go_v_191 = grad_out[go_base + ((int64_t)191 << 5)];
        local_k_15 = fma((scalar_t)(-0.28867512941360474) * w_i_47, x_j_14 * go_v_191, local_k_15);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_13  = x_all[x_base + ((int64_t)13 << 5)];
        scalar_t go_v_192 = grad_out[go_base + ((int64_t)192 << 5)];
        local_k_15 = fma((scalar_t)(0.45643547177314758) * w_i_47, x_j_13 * go_v_192, local_k_15);
    }
    {
        scalar_t w_i_47  = w[w_base + ((int64_t)47 << 5)];
        scalar_t x_j_12  = x_all[x_base + ((int64_t)12 << 5)];
        scalar_t go_v_193 = grad_out[go_base + ((int64_t)193 << 5)];
        local_k_15 = fma((scalar_t)(-0.64549720287322998) * w_i_47, x_j_12 * go_v_193, local_k_15);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_6  = x_all[x_base + ((int64_t)6 << 5)];
        scalar_t go_v_208 = grad_out[go_base + ((int64_t)208 << 5)];
        local_k_15 = fma((scalar_t)(0.86602538824081421) * w_i_50, x_j_6 * go_v_208, local_k_15);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_7  = x_all[x_base + ((int64_t)7 << 5)];
        scalar_t go_v_209 = grad_out[go_base + ((int64_t)209 << 5)];
        local_k_15 = fma((scalar_t)(0.86602538824081421) * w_i_51, x_j_7 * go_v_209, local_k_15);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_8  = x_all[x_base + ((int64_t)8 << 5)];
        scalar_t go_v_210 = grad_out[go_base + ((int64_t)210 << 5)];
        local_k_15 = fma((scalar_t)(-0.35355338454246521) * w_i_50, x_j_8 * go_v_210, local_k_15);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_9  = x_all[x_base + ((int64_t)9 << 5)];
        scalar_t go_v_211 = grad_out[go_base + ((int64_t)211 << 5)];
        local_k_15 = fma((scalar_t)(-0.35355338454246521) * w_i_51, x_j_9 * go_v_211, local_k_15);
    }
    {
        scalar_t w_i_50  = w[w_base + ((int64_t)50 << 5)];
        scalar_t x_j_4  = x_all[x_base + ((int64_t)4 << 5)];
        scalar_t go_v_218 = grad_out[go_base + ((int64_t)218 << 5)];
        local_k_15 = fma((scalar_t)(-0.35355338454246521) * w_i_50, x_j_4 * go_v_218, local_k_15);
    }
    {
        scalar_t w_i_51  = w[w_base + ((int64_t)51 << 5)];
        scalar_t x_j_5  = x_all[x_base + ((int64_t)5 << 5)];
        scalar_t go_v_219 = grad_out[go_base + ((int64_t)219 << 5)];
        local_k_15 = fma((scalar_t)(-0.35355338454246521) * w_i_51, x_j_5 * go_v_219, local_k_15);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_18  = x_all[x_base + ((int64_t)18 << 5)];
        scalar_t go_v_236 = grad_out[go_base + ((int64_t)236 << 5)];
        local_k_15 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_18 * go_v_236, local_k_15);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_19  = x_all[x_base + ((int64_t)19 << 5)];
        scalar_t go_v_237 = grad_out[go_base + ((int64_t)237 << 5)];
        local_k_15 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_19 * go_v_237, local_k_15);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_20  = x_all[x_base + ((int64_t)20 << 5)];
        scalar_t go_v_238 = grad_out[go_base + ((int64_t)238 << 5)];
        local_k_15 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_20 * go_v_238, local_k_15);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_15  = x_all[x_base + ((int64_t)15 << 5)];
        scalar_t go_v_239 = grad_out[go_base + ((int64_t)239 << 5)];
        local_k_15 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_15 * go_v_239, local_k_15);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_16  = x_all[x_base + ((int64_t)16 << 5)];
        scalar_t go_v_240 = grad_out[go_base + ((int64_t)240 << 5)];
        local_k_15 = fma((scalar_t)(-0.40824830532073975) * w_i_54, x_j_16 * go_v_240, local_k_15);
    }
    {
        scalar_t w_i_54  = w[w_base + ((int64_t)54 << 5)];
        scalar_t x_j_17  = x_all[x_base + ((int64_t)17 << 5)];
        scalar_t go_v_241 = grad_out[go_base + ((int64_t)241 << 5)];
        local_k_15 = fma((scalar_t)(0.40824830532073975) * w_i_54, x_j_17 * go_v_241, local_k_15);
    }
    scalar_t sum_k_15 = warp_sum(local_k_15);
    if (lane == 0) grad_y[gy_base + 15] += sum_k_15;

}

std::vector<torch::Tensor> uniform1d_split_bwd_u32_P777(
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

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_split_bwd_u32_P777", [&] {
        uniform1d_split_bwd_u32_P777_gradw<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        uniform1d_split_bwd_u32_P777_gradx<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        uniform1d_split_bwd_u32_P777_grady<scalar_t><<<grid, block, 0, stream>>>(
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

TORCH_LIBRARY(uniform1d_split_bwd_u32_p777, m)
{
    m.def("run", &uniform1d_split_bwd_u32_P777);
}