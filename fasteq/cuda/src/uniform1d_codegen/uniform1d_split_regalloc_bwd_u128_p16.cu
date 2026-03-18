
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void uniform1d_split_regalloc_bwd_u128_p16_gradw(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x_all,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ grad_w,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V)
{
    int lane = (int)threadIdx.x;
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;
    int64_t y_base  = (int64_t)b * Ky;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gw_base = ((int64_t)b   * Iw) * 32 + lane;

    // ============================================================
    // segment 0: targets = [0, 1, 2, 3]
    // ============================================================
    {
        scalar_t acc0 = scalar_t(0);
        scalar_t acc1 = scalar_t(0);
        scalar_t acc2 = scalar_t(0);
        scalar_t acc3 = scalar_t(0);

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 0];
            scalar_t gov = grad_out[go_base + ((int64_t)0 << 5)];
            acc0 = fma((scalar_t)(1.0), xv * (yv * gov), acc0);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 1];
            scalar_t gov = grad_out[go_base + ((int64_t)1 << 5)];
            acc1 = fma((scalar_t)(1.0), xv * (yv * gov), acc1);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 2];
            scalar_t gov = grad_out[go_base + ((int64_t)2 << 5)];
            acc1 = fma((scalar_t)(1.0), xv * (yv * gov), acc1);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 3];
            scalar_t gov = grad_out[go_base + ((int64_t)3 << 5)];
            acc1 = fma((scalar_t)(1.0), xv * (yv * gov), acc1);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 4];
            scalar_t gov = grad_out[go_base + ((int64_t)4 << 5)];
            acc2 = fma((scalar_t)(1.0), xv * (yv * gov), acc2);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 5];
            scalar_t gov = grad_out[go_base + ((int64_t)5 << 5)];
            acc2 = fma((scalar_t)(1.0), xv * (yv * gov), acc2);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 6];
            scalar_t gov = grad_out[go_base + ((int64_t)6 << 5)];
            acc2 = fma((scalar_t)(1.0), xv * (yv * gov), acc2);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 7];
            scalar_t gov = grad_out[go_base + ((int64_t)7 << 5)];
            acc2 = fma((scalar_t)(1.0), xv * (yv * gov), acc2);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 8];
            scalar_t gov = grad_out[go_base + ((int64_t)8 << 5)];
            acc2 = fma((scalar_t)(1.0), xv * (yv * gov), acc2);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 9];
            scalar_t gov = grad_out[go_base + ((int64_t)9 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 10];
            scalar_t gov = grad_out[go_base + ((int64_t)10 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 11];
            scalar_t gov = grad_out[go_base + ((int64_t)11 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 12];
            scalar_t gov = grad_out[go_base + ((int64_t)12 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 13];
            scalar_t gov = grad_out[go_base + ((int64_t)13 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 14];
            scalar_t gov = grad_out[go_base + ((int64_t)14 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        {
            scalar_t xv  = x_all[x_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 15];
            scalar_t gov = grad_out[go_base + ((int64_t)15 << 5)];
            acc3 = fma((scalar_t)(1.0), xv * (yv * gov), acc3);
        }

        grad_w[gw_base + ((int64_t)0 << 5)] = acc0;
        grad_w[gw_base + ((int64_t)1 << 5)] = acc1;
        grad_w[gw_base + ((int64_t)2 << 5)] = acc2;
        grad_w[gw_base + ((int64_t)3 << 5)] = acc3;
    }

}
#include <stdint.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void uniform1d_split_regalloc_bwd_u128_p16_gradx(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ grad_x,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V)
{
    int lane = (int)threadIdx.x;
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;
    int64_t y_base  = (int64_t)b * Ky;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gx_base = ((int64_t)src * Ix) * 32 + lane;

    // ============================================================
    // grad_x segment 0: targets = [0]
    // ============================================================
    {
        scalar_t acc0 = scalar_t(0);

        {
            scalar_t wv  = w[w_base + ((int64_t)0 << 5)];
            scalar_t yv  = y[y_base + 0];
            scalar_t gov = grad_out[go_base + ((int64_t)0 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)1 << 5)];
            scalar_t yv  = y[y_base + 1];
            scalar_t gov = grad_out[go_base + ((int64_t)1 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)1 << 5)];
            scalar_t yv  = y[y_base + 2];
            scalar_t gov = grad_out[go_base + ((int64_t)2 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)1 << 5)];
            scalar_t yv  = y[y_base + 3];
            scalar_t gov = grad_out[go_base + ((int64_t)3 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)2 << 5)];
            scalar_t yv  = y[y_base + 4];
            scalar_t gov = grad_out[go_base + ((int64_t)4 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)2 << 5)];
            scalar_t yv  = y[y_base + 5];
            scalar_t gov = grad_out[go_base + ((int64_t)5 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)2 << 5)];
            scalar_t yv  = y[y_base + 6];
            scalar_t gov = grad_out[go_base + ((int64_t)6 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)2 << 5)];
            scalar_t yv  = y[y_base + 7];
            scalar_t gov = grad_out[go_base + ((int64_t)7 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)2 << 5)];
            scalar_t yv  = y[y_base + 8];
            scalar_t gov = grad_out[go_base + ((int64_t)8 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 9];
            scalar_t gov = grad_out[go_base + ((int64_t)9 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 10];
            scalar_t gov = grad_out[go_base + ((int64_t)10 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 11];
            scalar_t gov = grad_out[go_base + ((int64_t)11 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 12];
            scalar_t gov = grad_out[go_base + ((int64_t)12 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 13];
            scalar_t gov = grad_out[go_base + ((int64_t)13 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 14];
            scalar_t gov = grad_out[go_base + ((int64_t)14 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        {
            scalar_t wv  = w[w_base + ((int64_t)3 << 5)];
            scalar_t yv  = y[y_base + 15];
            scalar_t gov = grad_out[go_base + ((int64_t)15 << 5)];
            acc0 = fma((scalar_t)(1.0), wv * (yv * gov), acc0);
        }

        atomicAdd(&grad_x[gx_base + ((int64_t)0 << 5)], acc0);
    }

}
template <typename scalar_t>
__launch_bounds__(32, 8) __global__ void uniform1d_split_regalloc_bwd_u128_p16_grady(
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
    scalar_t sum_k_0 = warp_sum(local_k_0);
    if (lane == 0) grad_y[gy_base + 0] += sum_k_0;

    scalar_t local_k_1 = scalar_t(0);
    {
        scalar_t w_i_1  = w[w_base + ((int64_t)1 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_1 = grad_out[go_base + ((int64_t)1 << 5)];
        local_k_1 = fma((scalar_t)(1.0) * w_i_1, x_j_0 * go_v_1, local_k_1);
    }
    scalar_t sum_k_1 = warp_sum(local_k_1);
    if (lane == 0) grad_y[gy_base + 1] += sum_k_1;

    scalar_t local_k_2 = scalar_t(0);
    {
        scalar_t w_i_1  = w[w_base + ((int64_t)1 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_2 = grad_out[go_base + ((int64_t)2 << 5)];
        local_k_2 = fma((scalar_t)(1.0) * w_i_1, x_j_0 * go_v_2, local_k_2);
    }
    scalar_t sum_k_2 = warp_sum(local_k_2);
    if (lane == 0) grad_y[gy_base + 2] += sum_k_2;

    scalar_t local_k_3 = scalar_t(0);
    {
        scalar_t w_i_1  = w[w_base + ((int64_t)1 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_3 = grad_out[go_base + ((int64_t)3 << 5)];
        local_k_3 = fma((scalar_t)(1.0) * w_i_1, x_j_0 * go_v_3, local_k_3);
    }
    scalar_t sum_k_3 = warp_sum(local_k_3);
    if (lane == 0) grad_y[gy_base + 3] += sum_k_3;

    // ---- grad_y chunk 1 ----
    scalar_t local_k_4 = scalar_t(0);
    {
        scalar_t w_i_2  = w[w_base + ((int64_t)2 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_4 = grad_out[go_base + ((int64_t)4 << 5)];
        local_k_4 = fma((scalar_t)(1.0) * w_i_2, x_j_0 * go_v_4, local_k_4);
    }
    scalar_t sum_k_4 = warp_sum(local_k_4);
    if (lane == 0) grad_y[gy_base + 4] += sum_k_4;

    scalar_t local_k_5 = scalar_t(0);
    {
        scalar_t w_i_2  = w[w_base + ((int64_t)2 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_5 = grad_out[go_base + ((int64_t)5 << 5)];
        local_k_5 = fma((scalar_t)(1.0) * w_i_2, x_j_0 * go_v_5, local_k_5);
    }
    scalar_t sum_k_5 = warp_sum(local_k_5);
    if (lane == 0) grad_y[gy_base + 5] += sum_k_5;

    scalar_t local_k_6 = scalar_t(0);
    {
        scalar_t w_i_2  = w[w_base + ((int64_t)2 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_6 = grad_out[go_base + ((int64_t)6 << 5)];
        local_k_6 = fma((scalar_t)(1.0) * w_i_2, x_j_0 * go_v_6, local_k_6);
    }
    scalar_t sum_k_6 = warp_sum(local_k_6);
    if (lane == 0) grad_y[gy_base + 6] += sum_k_6;

    scalar_t local_k_7 = scalar_t(0);
    {
        scalar_t w_i_2  = w[w_base + ((int64_t)2 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_7 = grad_out[go_base + ((int64_t)7 << 5)];
        local_k_7 = fma((scalar_t)(1.0) * w_i_2, x_j_0 * go_v_7, local_k_7);
    }
    scalar_t sum_k_7 = warp_sum(local_k_7);
    if (lane == 0) grad_y[gy_base + 7] += sum_k_7;

    // ---- grad_y chunk 2 ----
    scalar_t local_k_8 = scalar_t(0);
    {
        scalar_t w_i_2  = w[w_base + ((int64_t)2 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_8 = grad_out[go_base + ((int64_t)8 << 5)];
        local_k_8 = fma((scalar_t)(1.0) * w_i_2, x_j_0 * go_v_8, local_k_8);
    }
    scalar_t sum_k_8 = warp_sum(local_k_8);
    if (lane == 0) grad_y[gy_base + 8] += sum_k_8;

    scalar_t local_k_9 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_9 = grad_out[go_base + ((int64_t)9 << 5)];
        local_k_9 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_9, local_k_9);
    }
    scalar_t sum_k_9 = warp_sum(local_k_9);
    if (lane == 0) grad_y[gy_base + 9] += sum_k_9;

    scalar_t local_k_10 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_10 = grad_out[go_base + ((int64_t)10 << 5)];
        local_k_10 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_10, local_k_10);
    }
    scalar_t sum_k_10 = warp_sum(local_k_10);
    if (lane == 0) grad_y[gy_base + 10] += sum_k_10;

    scalar_t local_k_11 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_11 = grad_out[go_base + ((int64_t)11 << 5)];
        local_k_11 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_11, local_k_11);
    }
    scalar_t sum_k_11 = warp_sum(local_k_11);
    if (lane == 0) grad_y[gy_base + 11] += sum_k_11;

    // ---- grad_y chunk 3 ----
    scalar_t local_k_12 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_12 = grad_out[go_base + ((int64_t)12 << 5)];
        local_k_12 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_12, local_k_12);
    }
    scalar_t sum_k_12 = warp_sum(local_k_12);
    if (lane == 0) grad_y[gy_base + 12] += sum_k_12;

    scalar_t local_k_13 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_13 = grad_out[go_base + ((int64_t)13 << 5)];
        local_k_13 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_13, local_k_13);
    }
    scalar_t sum_k_13 = warp_sum(local_k_13);
    if (lane == 0) grad_y[gy_base + 13] += sum_k_13;

    scalar_t local_k_14 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_14 = grad_out[go_base + ((int64_t)14 << 5)];
        local_k_14 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_14, local_k_14);
    }
    scalar_t sum_k_14 = warp_sum(local_k_14);
    if (lane == 0) grad_y[gy_base + 14] += sum_k_14;

    scalar_t local_k_15 = scalar_t(0);
    {
        scalar_t w_i_3  = w[w_base + ((int64_t)3 << 5)];
        scalar_t x_j_0  = x_all[x_base + ((int64_t)0 << 5)];
        scalar_t go_v_15 = grad_out[go_base + ((int64_t)15 << 5)];
        local_k_15 = fma((scalar_t)(1.0) * w_i_3, x_j_0 * go_v_15, local_k_15);
    }
    scalar_t sum_k_15 = warp_sum(local_k_15);
    if (lane == 0) grad_y[gy_base + 15] += sum_k_15;

}

std::vector<torch::Tensor> uniform1d_split_regalloc_bwd_u128_p16(
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

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_split_regalloc_bwd_u128_p16", [&] {
        uniform1d_split_regalloc_bwd_u128_p16_gradw<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        uniform1d_split_regalloc_bwd_u128_p16_gradx<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        uniform1d_split_regalloc_bwd_u128_p16_grady<scalar_t><<<grid, block, 0, stream>>>(
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

TORCH_LIBRARY(uniform1d_split_regalloc_bwd_u128_p16_codegen, m) {
    m.def("run", &uniform1d_split_regalloc_bwd_u128_p16);
}
