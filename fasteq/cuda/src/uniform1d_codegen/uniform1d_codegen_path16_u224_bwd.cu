#include <stdint.h>
#include <cuda_runtime.h>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void uniform1d_codegen_blocku_vgroup_path16_u224_bwd(
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
    int tid = (int)threadIdx.x;
    if (tid >= U) return;

    int lane    = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (U + 31) >> 5;
    int u = tid;

    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t y_base = (int64_t)b * Ky;

    extern __shared__ char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
    // layout: [num_unique_k][num_warps]

    // preload w(i, u)
    scalar_t wi_0 = w[((int64_t)b * Iw + 0) * (int64_t)U + u];
    scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
    scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
    scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];

    // preload x(j, u)
    scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];

    // preload y(k)
    scalar_t yk_0 = y[y_base + 0];
    scalar_t yk_1 = y[y_base + 1];
    scalar_t yk_2 = y[y_base + 2];
    scalar_t yk_3 = y[y_base + 3];
    scalar_t yk_4 = y[y_base + 4];
    scalar_t yk_5 = y[y_base + 5];
    scalar_t yk_6 = y[y_base + 6];
    scalar_t yk_7 = y[y_base + 7];
    scalar_t yk_8 = y[y_base + 8];
    scalar_t yk_9 = y[y_base + 9];
    scalar_t yk_10 = y[y_base + 10];
    scalar_t yk_11 = y[y_base + 11];
    scalar_t yk_12 = y[y_base + 12];
    scalar_t yk_13 = y[y_base + 13];
    scalar_t yk_14 = y[y_base + 14];
    scalar_t yk_15 = y[y_base + 15];

    // preload grad_out(v, u)
    scalar_t go_v_0 = grad_out[((int64_t)dst * V + 0) * (int64_t)U + u];
    scalar_t go_v_1 = grad_out[((int64_t)dst * V + 1) * (int64_t)U + u];
    scalar_t go_v_2 = grad_out[((int64_t)dst * V + 2) * (int64_t)U + u];
    scalar_t go_v_3 = grad_out[((int64_t)dst * V + 3) * (int64_t)U + u];
    scalar_t go_v_4 = grad_out[((int64_t)dst * V + 4) * (int64_t)U + u];
    scalar_t go_v_5 = grad_out[((int64_t)dst * V + 5) * (int64_t)U + u];
    scalar_t go_v_6 = grad_out[((int64_t)dst * V + 6) * (int64_t)U + u];
    scalar_t go_v_7 = grad_out[((int64_t)dst * V + 7) * (int64_t)U + u];
    scalar_t go_v_8 = grad_out[((int64_t)dst * V + 8) * (int64_t)U + u];
    scalar_t go_v_9 = grad_out[((int64_t)dst * V + 9) * (int64_t)U + u];
    scalar_t go_v_10 = grad_out[((int64_t)dst * V + 10) * (int64_t)U + u];
    scalar_t go_v_11 = grad_out[((int64_t)dst * V + 11) * (int64_t)U + u];
    scalar_t go_v_12 = grad_out[((int64_t)dst * V + 12) * (int64_t)U + u];
    scalar_t go_v_13 = grad_out[((int64_t)dst * V + 13) * (int64_t)U + u];
    scalar_t go_v_14 = grad_out[((int64_t)dst * V + 14) * (int64_t)U + u];
    scalar_t go_v_15 = grad_out[((int64_t)dst * V + 15) * (int64_t)U + u];

    // grad_w accumulate by unique i (no atomic: one thread owns one (b, i, u))
    scalar_t gw_acc_i_0 = scalar_t(0);
    gw_acc_i_0 += xj_0 * yk_0 * go_v_0;
    grad_w[((int64_t)b * Iw + 0) * (int64_t)U + u] = gw_acc_i_0;

    scalar_t gw_acc_i_1 = scalar_t(0);
    gw_acc_i_1 += xj_0 * yk_1 * go_v_1;
    gw_acc_i_1 += xj_0 * yk_2 * go_v_2;
    gw_acc_i_1 += xj_0 * yk_3 * go_v_3;
    grad_w[((int64_t)b * Iw + 1) * (int64_t)U + u] = gw_acc_i_1;

    scalar_t gw_acc_i_2 = scalar_t(0);
    gw_acc_i_2 += xj_0 * yk_4 * go_v_4;
    gw_acc_i_2 += xj_0 * yk_5 * go_v_5;
    gw_acc_i_2 += xj_0 * yk_6 * go_v_6;
    gw_acc_i_2 += xj_0 * yk_7 * go_v_7;
    gw_acc_i_2 += xj_0 * yk_8 * go_v_8;
    grad_w[((int64_t)b * Iw + 2) * (int64_t)U + u] = gw_acc_i_2;

    scalar_t gw_acc_i_3 = scalar_t(0);
    gw_acc_i_3 += xj_0 * yk_9 * go_v_9;
    gw_acc_i_3 += xj_0 * yk_10 * go_v_10;
    gw_acc_i_3 += xj_0 * yk_11 * go_v_11;
    gw_acc_i_3 += xj_0 * yk_12 * go_v_12;
    gw_acc_i_3 += xj_0 * yk_13 * go_v_13;
    gw_acc_i_3 += xj_0 * yk_14 * go_v_14;
    gw_acc_i_3 += xj_0 * yk_15 * go_v_15;
    grad_w[((int64_t)b * Iw + 3) * (int64_t)U + u] = gw_acc_i_3;

    // grad_x accumulate by unique j
    scalar_t gx_acc_j_0 = scalar_t(0);
    gx_acc_j_0 += wi_0 * yk_0 * go_v_0;
    gx_acc_j_0 += wi_1 * yk_1 * go_v_1;
    gx_acc_j_0 += wi_1 * yk_2 * go_v_2;
    gx_acc_j_0 += wi_1 * yk_3 * go_v_3;
    gx_acc_j_0 += wi_2 * yk_4 * go_v_4;
    gx_acc_j_0 += wi_2 * yk_5 * go_v_5;
    gx_acc_j_0 += wi_2 * yk_6 * go_v_6;
    gx_acc_j_0 += wi_2 * yk_7 * go_v_7;
    gx_acc_j_0 += wi_2 * yk_8 * go_v_8;
    gx_acc_j_0 += wi_3 * yk_9 * go_v_9;
    gx_acc_j_0 += wi_3 * yk_10 * go_v_10;
    gx_acc_j_0 += wi_3 * yk_11 * go_v_11;
    gx_acc_j_0 += wi_3 * yk_12 * go_v_12;
    gx_acc_j_0 += wi_3 * yk_13 * go_v_13;
    gx_acc_j_0 += wi_3 * yk_14 * go_v_14;
    gx_acc_j_0 += wi_3 * yk_15 * go_v_15;
    atomicAdd(&grad_x[((int64_t)src * Ix + 0) * (int64_t)U + u], gx_acc_j_0);

    // grad_y accumulate by unique k, reduced across full block (all U channels)
    scalar_t gy_lane_k_0 = scalar_t(0);
    gy_lane_k_0 += wi_0 * xj_0 * go_v_0;
    scalar_t gy_warp_k_0 = warp_sum(gy_lane_k_0);
    if (lane == 0) smem[0 * num_warps + warp_id] = gy_warp_k_0;

    scalar_t gy_lane_k_1 = scalar_t(0);
    gy_lane_k_1 += wi_1 * xj_0 * go_v_1;
    scalar_t gy_warp_k_1 = warp_sum(gy_lane_k_1);
    if (lane == 0) smem[1 * num_warps + warp_id] = gy_warp_k_1;

    scalar_t gy_lane_k_2 = scalar_t(0);
    gy_lane_k_2 += wi_1 * xj_0 * go_v_2;
    scalar_t gy_warp_k_2 = warp_sum(gy_lane_k_2);
    if (lane == 0) smem[2 * num_warps + warp_id] = gy_warp_k_2;

    scalar_t gy_lane_k_3 = scalar_t(0);
    gy_lane_k_3 += wi_1 * xj_0 * go_v_3;
    scalar_t gy_warp_k_3 = warp_sum(gy_lane_k_3);
    if (lane == 0) smem[3 * num_warps + warp_id] = gy_warp_k_3;

    scalar_t gy_lane_k_4 = scalar_t(0);
    gy_lane_k_4 += wi_2 * xj_0 * go_v_4;
    scalar_t gy_warp_k_4 = warp_sum(gy_lane_k_4);
    if (lane == 0) smem[4 * num_warps + warp_id] = gy_warp_k_4;

    scalar_t gy_lane_k_5 = scalar_t(0);
    gy_lane_k_5 += wi_2 * xj_0 * go_v_5;
    scalar_t gy_warp_k_5 = warp_sum(gy_lane_k_5);
    if (lane == 0) smem[5 * num_warps + warp_id] = gy_warp_k_5;

    scalar_t gy_lane_k_6 = scalar_t(0);
    gy_lane_k_6 += wi_2 * xj_0 * go_v_6;
    scalar_t gy_warp_k_6 = warp_sum(gy_lane_k_6);
    if (lane == 0) smem[6 * num_warps + warp_id] = gy_warp_k_6;

    scalar_t gy_lane_k_7 = scalar_t(0);
    gy_lane_k_7 += wi_2 * xj_0 * go_v_7;
    scalar_t gy_warp_k_7 = warp_sum(gy_lane_k_7);
    if (lane == 0) smem[7 * num_warps + warp_id] = gy_warp_k_7;

    scalar_t gy_lane_k_8 = scalar_t(0);
    gy_lane_k_8 += wi_2 * xj_0 * go_v_8;
    scalar_t gy_warp_k_8 = warp_sum(gy_lane_k_8);
    if (lane == 0) smem[8 * num_warps + warp_id] = gy_warp_k_8;

    scalar_t gy_lane_k_9 = scalar_t(0);
    gy_lane_k_9 += wi_3 * xj_0 * go_v_9;
    scalar_t gy_warp_k_9 = warp_sum(gy_lane_k_9);
    if (lane == 0) smem[9 * num_warps + warp_id] = gy_warp_k_9;

    scalar_t gy_lane_k_10 = scalar_t(0);
    gy_lane_k_10 += wi_3 * xj_0 * go_v_10;
    scalar_t gy_warp_k_10 = warp_sum(gy_lane_k_10);
    if (lane == 0) smem[10 * num_warps + warp_id] = gy_warp_k_10;

    scalar_t gy_lane_k_11 = scalar_t(0);
    gy_lane_k_11 += wi_3 * xj_0 * go_v_11;
    scalar_t gy_warp_k_11 = warp_sum(gy_lane_k_11);
    if (lane == 0) smem[11 * num_warps + warp_id] = gy_warp_k_11;

    scalar_t gy_lane_k_12 = scalar_t(0);
    gy_lane_k_12 += wi_3 * xj_0 * go_v_12;
    scalar_t gy_warp_k_12 = warp_sum(gy_lane_k_12);
    if (lane == 0) smem[12 * num_warps + warp_id] = gy_warp_k_12;

    scalar_t gy_lane_k_13 = scalar_t(0);
    gy_lane_k_13 += wi_3 * xj_0 * go_v_13;
    scalar_t gy_warp_k_13 = warp_sum(gy_lane_k_13);
    if (lane == 0) smem[13 * num_warps + warp_id] = gy_warp_k_13;

    scalar_t gy_lane_k_14 = scalar_t(0);
    gy_lane_k_14 += wi_3 * xj_0 * go_v_14;
    scalar_t gy_warp_k_14 = warp_sum(gy_lane_k_14);
    if (lane == 0) smem[14 * num_warps + warp_id] = gy_warp_k_14;

    scalar_t gy_lane_k_15 = scalar_t(0);
    gy_lane_k_15 += wi_3 * xj_0 * go_v_15;
    scalar_t gy_warp_k_15 = warp_sum(gy_lane_k_15);
    if (lane == 0) smem[15 * num_warps + warp_id] = gy_warp_k_15;

    __syncthreads();

    if (warp_id == 0) {
        int red_lane = lane;
        scalar_t block_sum_k_0 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_0 = smem[0 * num_warps + red_lane];
        block_sum_k_0 = warp_sum(block_sum_k_0);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 0], block_sum_k_0);

        scalar_t block_sum_k_1 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_1 = smem[1 * num_warps + red_lane];
        block_sum_k_1 = warp_sum(block_sum_k_1);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 1], block_sum_k_1);

        scalar_t block_sum_k_2 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_2 = smem[2 * num_warps + red_lane];
        block_sum_k_2 = warp_sum(block_sum_k_2);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 2], block_sum_k_2);

        scalar_t block_sum_k_3 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_3 = smem[3 * num_warps + red_lane];
        block_sum_k_3 = warp_sum(block_sum_k_3);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 3], block_sum_k_3);

        scalar_t block_sum_k_4 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_4 = smem[4 * num_warps + red_lane];
        block_sum_k_4 = warp_sum(block_sum_k_4);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 4], block_sum_k_4);

        scalar_t block_sum_k_5 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_5 = smem[5 * num_warps + red_lane];
        block_sum_k_5 = warp_sum(block_sum_k_5);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 5], block_sum_k_5);

        scalar_t block_sum_k_6 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_6 = smem[6 * num_warps + red_lane];
        block_sum_k_6 = warp_sum(block_sum_k_6);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 6], block_sum_k_6);

        scalar_t block_sum_k_7 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_7 = smem[7 * num_warps + red_lane];
        block_sum_k_7 = warp_sum(block_sum_k_7);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 7], block_sum_k_7);

        scalar_t block_sum_k_8 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_8 = smem[8 * num_warps + red_lane];
        block_sum_k_8 = warp_sum(block_sum_k_8);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 8], block_sum_k_8);

        scalar_t block_sum_k_9 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_9 = smem[9 * num_warps + red_lane];
        block_sum_k_9 = warp_sum(block_sum_k_9);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 9], block_sum_k_9);

        scalar_t block_sum_k_10 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_10 = smem[10 * num_warps + red_lane];
        block_sum_k_10 = warp_sum(block_sum_k_10);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 10], block_sum_k_10);

        scalar_t block_sum_k_11 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_11 = smem[11 * num_warps + red_lane];
        block_sum_k_11 = warp_sum(block_sum_k_11);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 11], block_sum_k_11);

        scalar_t block_sum_k_12 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_12 = smem[12 * num_warps + red_lane];
        block_sum_k_12 = warp_sum(block_sum_k_12);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 12], block_sum_k_12);

        scalar_t block_sum_k_13 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_13 = smem[13 * num_warps + red_lane];
        block_sum_k_13 = warp_sum(block_sum_k_13);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 13], block_sum_k_13);

        scalar_t block_sum_k_14 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_14 = smem[14 * num_warps + red_lane];
        block_sum_k_14 = warp_sum(block_sum_k_14);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 14], block_sum_k_14);

        scalar_t block_sum_k_15 = scalar_t(0);
        if (red_lane < num_warps) block_sum_k_15 = smem[15 * num_warps + red_lane];
        block_sum_k_15 = warp_sum(block_sum_k_15);
        if (red_lane == 0) atomicAdd(&grad_y[y_base + 15], block_sum_k_15);

    }
}

template <typename scalar_t>
void launch_uniform1d_codegen_blocku_vgroup_path16_u224_bwd(
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
    // 要求 U <= 1024
    size_t smem_bytes = sizeof(scalar_t) * 16 * ((U + 31) / 32);
    dim3 block(U);
    dim3 grid(B);
    uniform1d_codegen_blocku_vgroup_path16_u224_bwd<scalar_t><<<grid, block, smem_bytes, stream>>>(
        w, x_all, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);
}
