#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cstdint>
#include "cuda_utils.hpp"

template <typename scalar_t, typename index_t>
__global__ void uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_w,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_y,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)
{
    const int e_local = (int)blockIdx.x;
    if (e_local >= B) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    constexpr int U_CONST = 128;
    (void)U_CONST;
    const int e_orig = b_list ? b_list[e_local] : e_local;
    const int w_row  = (WB == 1 ? 0 : e_orig);

    const int src = src_idx[e_orig];
    const int dst = dst_idx[e_orig];

    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;
    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;
    const index_t x_base  = (index_t)src * (index_t)Ix * (index_t)U;
    const index_t gx_base = (index_t)src * (index_t)Ix * (index_t)U;
    const index_t y_base  = (index_t)e_orig * (index_t)Ky;
    const index_t gy_base = (index_t)e_orig * (index_t)Ky;
    const index_t go_base = (index_t)dst * (index_t)V * (index_t)U;

    // full-resident accumulators across all phases/subphases
    scalar_t gw_acc_i_0;
    scalar_t gw_acc_i_1;
    scalar_t gw_acc_i_2;
    scalar_t gw_acc_i_3;
    scalar_t gx_acc_j_0;
    scalar_t gy_acc_k_0;
    scalar_t gy_acc_k_1;
    scalar_t gy_acc_k_2;
    scalar_t gy_acc_k_3;
    scalar_t gy_acc_k_4;
    scalar_t gy_acc_k_5;
    scalar_t gy_acc_k_6;
    scalar_t gy_acc_k_7;
    scalar_t gy_acc_k_8;
    scalar_t gy_acc_k_9;
    scalar_t gy_acc_k_10;
    scalar_t gy_acc_k_11;
    scalar_t gy_acc_k_12;
    scalar_t gy_acc_k_13;
    scalar_t gy_acc_k_14;
    scalar_t gy_acc_k_15;

    // subphase-local slot names (static upper bounds across all subphases)
    scalar_t wi_slot_0;
    scalar_t wi_slot_1;
    scalar_t wi_slot_2;
    scalar_t wi_slot_3;
    scalar_t go_slot_0;
    scalar_t go_slot_1;
    scalar_t go_slot_2;
    scalar_t go_slot_3;
    scalar_t go_slot_4;
    scalar_t go_slot_5;
    scalar_t go_slot_6;
    scalar_t go_slot_7;
    scalar_t go_slot_8;
    scalar_t go_slot_9;
    scalar_t go_slot_10;
    scalar_t go_slot_11;
    scalar_t go_slot_12;
    scalar_t go_slot_13;
    scalar_t go_slot_14;
    scalar_t go_slot_15;
    scalar_t x_slot_0;
    scalar_t y_slot_0;
    scalar_t y_slot_1;
    scalar_t y_slot_2;
    scalar_t y_slot_3;
    scalar_t y_slot_4;
    scalar_t y_slot_5;
    scalar_t y_slot_6;
    scalar_t y_slot_7;
    scalar_t y_slot_8;
    scalar_t y_slot_9;
    scalar_t y_slot_10;
    scalar_t y_slot_11;
    scalar_t y_slot_12;
    scalar_t y_slot_13;
    scalar_t y_slot_14;
    scalar_t y_slot_15;

    for (int u_base = 0; u_base < U; u_base += 32) {
        int u = u_base + lane;
        if (u < U) {

            // reset full-resident accumulators
            gw_acc_i_0 = scalar_t(0);
            gw_acc_i_1 = scalar_t(0);
            gw_acc_i_2 = scalar_t(0);
            gw_acc_i_3 = scalar_t(0);
            gx_acc_j_0 = scalar_t(0);
            gy_acc_k_0 = scalar_t(0);
            gy_acc_k_1 = scalar_t(0);
            gy_acc_k_2 = scalar_t(0);
            gy_acc_k_3 = scalar_t(0);
            gy_acc_k_4 = scalar_t(0);
            gy_acc_k_5 = scalar_t(0);
            gy_acc_k_6 = scalar_t(0);
            gy_acc_k_7 = scalar_t(0);
            gy_acc_k_8 = scalar_t(0);
            gy_acc_k_9 = scalar_t(0);
            gy_acc_k_10 = scalar_t(0);
            gy_acc_k_11 = scalar_t(0);
            gy_acc_k_12 = scalar_t(0);
            gy_acc_k_13 = scalar_t(0);
            gy_acc_k_14 = scalar_t(0);
            gy_acc_k_15 = scalar_t(0);

            // ===== phase 0: pivot=x[0] =====
            // ---- subphase 0 ----
            {
                // preload wi slots for subphase 0
                wi_slot_0 = w[w_base + (index_t)0 + (index_t)u];
                wi_slot_1 = w[w_base + (index_t)128 + (index_t)u];
                wi_slot_2 = w[w_base + (index_t)256 + (index_t)u];
                wi_slot_3 = w[w_base + (index_t)384 + (index_t)u];
                // preload go slots for subphase 0
                go_slot_0 = grad_out[go_base + (index_t)0 + (index_t)u];
                go_slot_1 = grad_out[go_base + (index_t)128 + (index_t)u];
                go_slot_2 = grad_out[go_base + (index_t)256 + (index_t)u];
                go_slot_3 = grad_out[go_base + (index_t)384 + (index_t)u];
                go_slot_4 = grad_out[go_base + (index_t)512 + (index_t)u];
                go_slot_5 = grad_out[go_base + (index_t)640 + (index_t)u];
                go_slot_6 = grad_out[go_base + (index_t)768 + (index_t)u];
                go_slot_7 = grad_out[go_base + (index_t)896 + (index_t)u];
                go_slot_8 = grad_out[go_base + (index_t)1024 + (index_t)u];
                go_slot_9 = grad_out[go_base + (index_t)1152 + (index_t)u];
                go_slot_10 = grad_out[go_base + (index_t)1280 + (index_t)u];
                go_slot_11 = grad_out[go_base + (index_t)1408 + (index_t)u];
                go_slot_12 = grad_out[go_base + (index_t)1536 + (index_t)u];
                go_slot_13 = grad_out[go_base + (index_t)1664 + (index_t)u];
                go_slot_14 = grad_out[go_base + (index_t)1792 + (index_t)u];
                go_slot_15 = grad_out[go_base + (index_t)1920 + (index_t)u];
                // preload x slots for subphase 0
                x_slot_0 = x[x_base + (index_t)0 + (index_t)u];
                // preload y slots for subphase 0
                y_slot_0 = y[y_base + (index_t)0];
                y_slot_1 = y[y_base + (index_t)1];
                y_slot_2 = y[y_base + (index_t)2];
                y_slot_3 = y[y_base + (index_t)3];
                y_slot_4 = y[y_base + (index_t)4];
                y_slot_5 = y[y_base + (index_t)5];
                y_slot_6 = y[y_base + (index_t)6];
                y_slot_7 = y[y_base + (index_t)7];
                y_slot_8 = y[y_base + (index_t)8];
                y_slot_9 = y[y_base + (index_t)9];
                y_slot_10 = y[y_base + (index_t)10];
                y_slot_11 = y[y_base + (index_t)11];
                y_slot_12 = y[y_base + (index_t)12];
                y_slot_13 = y[y_base + (index_t)13];
                y_slot_14 = y[y_base + (index_t)14];
                y_slot_15 = y[y_base + (index_t)15];
                // accumulate ops for subphase 0 (sorted by wg)
                gx_acc_j_0 += scalar_t(1.0) * wi_slot_0 * go_slot_0 * y_slot_0;
                gy_acc_k_0 += scalar_t(1.0) * wi_slot_0 * go_slot_0 * x_slot_0;
                gw_acc_i_0 += scalar_t(1.0) * go_slot_0 * x_slot_0 * y_slot_0;
                gx_acc_j_0 += scalar_t(1.0000000000000002) * wi_slot_1 * go_slot_1 * y_slot_1;
                gy_acc_k_1 += scalar_t(1.0000000000000002) * wi_slot_1 * go_slot_1 * x_slot_0;
                gw_acc_i_1 += scalar_t(1.0000000000000002) * go_slot_1 * x_slot_0 * y_slot_1;
                gx_acc_j_0 += scalar_t(0.9999999999999998) * wi_slot_1 * go_slot_2 * y_slot_2;
                gy_acc_k_2 += scalar_t(0.9999999999999998) * wi_slot_1 * go_slot_2 * x_slot_0;
                gw_acc_i_1 += scalar_t(0.9999999999999998) * go_slot_2 * x_slot_0 * y_slot_2;
                gx_acc_j_0 += scalar_t(1.0000000000000002) * wi_slot_1 * go_slot_3 * y_slot_3;
                gy_acc_k_3 += scalar_t(1.0000000000000002) * wi_slot_1 * go_slot_3 * x_slot_0;
                gw_acc_i_1 += scalar_t(1.0000000000000002) * go_slot_3 * x_slot_0 * y_slot_3;
                gx_acc_j_0 += scalar_t(0.9999999999999993) * wi_slot_2 * go_slot_4 * y_slot_4;
                gy_acc_k_4 += scalar_t(0.9999999999999993) * wi_slot_2 * go_slot_4 * x_slot_0;
                gw_acc_i_2 += scalar_t(0.9999999999999993) * go_slot_4 * x_slot_0 * y_slot_4;
                gx_acc_j_0 += scalar_t(1.0000000000000002) * wi_slot_2 * go_slot_5 * y_slot_5;
                gy_acc_k_5 += scalar_t(1.0000000000000002) * wi_slot_2 * go_slot_5 * x_slot_0;
                gw_acc_i_2 += scalar_t(1.0000000000000002) * go_slot_5 * x_slot_0 * y_slot_5;
                gx_acc_j_0 += scalar_t(0.9999999999999997) * wi_slot_2 * go_slot_6 * y_slot_6;
                gy_acc_k_6 += scalar_t(0.9999999999999997) * wi_slot_2 * go_slot_6 * x_slot_0;
                gw_acc_i_2 += scalar_t(0.9999999999999997) * go_slot_6 * x_slot_0 * y_slot_6;
                gx_acc_j_0 += scalar_t(1.0) * wi_slot_2 * go_slot_7 * y_slot_7;
                gy_acc_k_7 += scalar_t(1.0) * wi_slot_2 * go_slot_7 * x_slot_0;
                gw_acc_i_2 += scalar_t(1.0) * go_slot_7 * x_slot_0 * y_slot_7;
                gx_acc_j_0 += scalar_t(1.0000000000000004) * wi_slot_2 * go_slot_8 * y_slot_8;
                gy_acc_k_8 += scalar_t(1.0000000000000004) * wi_slot_2 * go_slot_8 * x_slot_0;
                gw_acc_i_2 += scalar_t(1.0000000000000004) * go_slot_8 * x_slot_0 * y_slot_8;
                gx_acc_j_0 += scalar_t(1.0000000000000007) * wi_slot_3 * go_slot_9 * y_slot_9;
                gy_acc_k_9 += scalar_t(1.0000000000000007) * wi_slot_3 * go_slot_9 * x_slot_0;
                gw_acc_i_3 += scalar_t(1.0000000000000007) * go_slot_9 * x_slot_0 * y_slot_9;
                gx_acc_j_0 += scalar_t(1.0000000000000004) * wi_slot_3 * go_slot_10 * y_slot_10;
                gy_acc_k_10 += scalar_t(1.0000000000000004) * wi_slot_3 * go_slot_10 * x_slot_0;
                gw_acc_i_3 += scalar_t(1.0000000000000004) * go_slot_10 * x_slot_0 * y_slot_10;
                gx_acc_j_0 += scalar_t(0.9999999999999991) * wi_slot_3 * go_slot_11 * y_slot_11;
                gy_acc_k_11 += scalar_t(0.9999999999999991) * wi_slot_3 * go_slot_11 * x_slot_0;
                gw_acc_i_3 += scalar_t(0.9999999999999991) * go_slot_11 * x_slot_0 * y_slot_11;
                gx_acc_j_0 += scalar_t(0.9999999999999996) * wi_slot_3 * go_slot_12 * y_slot_12;
                gy_acc_k_12 += scalar_t(0.9999999999999996) * wi_slot_3 * go_slot_12 * x_slot_0;
                gw_acc_i_3 += scalar_t(0.9999999999999996) * go_slot_12 * x_slot_0 * y_slot_12;
                gx_acc_j_0 += scalar_t(0.9999999999999996) * wi_slot_3 * go_slot_13 * y_slot_13;
                gy_acc_k_13 += scalar_t(0.9999999999999996) * wi_slot_3 * go_slot_13 * x_slot_0;
                gw_acc_i_3 += scalar_t(0.9999999999999996) * go_slot_13 * x_slot_0 * y_slot_13;
                gx_acc_j_0 += scalar_t(0.9999999999999998) * wi_slot_3 * go_slot_14 * y_slot_14;
                gy_acc_k_14 += scalar_t(0.9999999999999998) * wi_slot_3 * go_slot_14 * x_slot_0;
                gw_acc_i_3 += scalar_t(0.9999999999999998) * go_slot_14 * x_slot_0 * y_slot_14;
                gx_acc_j_0 += scalar_t(1.0000000000000004) * wi_slot_3 * go_slot_15 * y_slot_15;
                gy_acc_k_15 += scalar_t(1.0000000000000004) * wi_slot_3 * go_slot_15 * x_slot_0;
                gw_acc_i_3 += scalar_t(1.0000000000000004) * go_slot_15 * x_slot_0 * y_slot_15;
            }

            // final writeback after all phases/subphases
            grad_w[gw_base + (index_t)0 + (index_t)u] = gw_acc_i_0;
            grad_w[gw_base + (index_t)128 + (index_t)u] = gw_acc_i_1;
            grad_w[gw_base + (index_t)256 + (index_t)u] = gw_acc_i_2;
            grad_w[gw_base + (index_t)384 + (index_t)u] = gw_acc_i_3;

            // grad_x writeback
            atomicAdd(&grad_x[gx_base + (index_t)0 + (index_t)u], gx_acc_j_0);

            // grad_y writeback
            scalar_t gy_sum_0 = warp_sum_xor(gy_acc_k_0);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)0], gy_sum_0);
            }
            scalar_t gy_sum_1 = warp_sum_xor(gy_acc_k_1);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)1], gy_sum_1);
            }
            scalar_t gy_sum_2 = warp_sum_xor(gy_acc_k_2);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)2], gy_sum_2);
            }
            scalar_t gy_sum_3 = warp_sum_xor(gy_acc_k_3);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)3], gy_sum_3);
            }
            scalar_t gy_sum_4 = warp_sum_xor(gy_acc_k_4);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)4], gy_sum_4);
            }
            scalar_t gy_sum_5 = warp_sum_xor(gy_acc_k_5);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)5], gy_sum_5);
            }
            scalar_t gy_sum_6 = warp_sum_xor(gy_acc_k_6);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)6], gy_sum_6);
            }
            scalar_t gy_sum_7 = warp_sum_xor(gy_acc_k_7);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)7], gy_sum_7);
            }
            scalar_t gy_sum_8 = warp_sum_xor(gy_acc_k_8);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)8], gy_sum_8);
            }
            scalar_t gy_sum_9 = warp_sum_xor(gy_acc_k_9);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)9], gy_sum_9);
            }
            scalar_t gy_sum_10 = warp_sum_xor(gy_acc_k_10);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)10], gy_sum_10);
            }
            scalar_t gy_sum_11 = warp_sum_xor(gy_acc_k_11);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)11], gy_sum_11);
            }
            scalar_t gy_sum_12 = warp_sum_xor(gy_acc_k_12);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)12], gy_sum_12);
            }
            scalar_t gy_sum_13 = warp_sum_xor(gy_acc_k_13);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)13], gy_sum_13);
            }
            scalar_t gy_sum_14 = warp_sum_xor(gy_acc_k_14);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)14], gy_sum_14);
            }
            scalar_t gy_sum_15 = warp_sum_xor(gy_acc_k_15);
            if (lane == 0) {
                atomicAdd(&grad_y[gy_base + (index_t)15], gy_sum_15);
            }
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

static inline bool should_use_int32_index(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);
    bool y_ok = false;
    if (mode_scalar_y) {
        y_ok = mul_fits_int32((int64_t)B, (int64_t)Ky);
    } else {
        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
        y_ok = mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);
    }
    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool go_ok = mul3_fits_int32(go_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && go_ok;
}

template <typename scalar_t, typename index_t>
void launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused_typed(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    cudaStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused<scalar_t, index_t><<<grid, block, 0, stream>>>(
        w, x, y, grad_out,
        grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    cudaStream_t stream)
{
    constexpr bool kUseXSrc = true;
    constexpr bool kUseYSrc = false;
    constexpr bool kUseScatter = true;
    constexpr bool kModeScalarY = true;
    if (should_use_int32_index(B, WB, Iw, Ix, Ky, V, U, S,
                               kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused_typed<scalar_t, int32_t>(
            w, x, y, grad_out, grad_w, grad_x, grad_y,
            src_idx, dst_idx, b_list,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused_typed<scalar_t, int64_t>(
            w, x, y, grad_out, grad_w, grad_x, grad_y,
            src_idx, dst_idx, b_list,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_w,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    cudaStream_t stream)
{
    launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused_auto<scalar_t>(
        w, x, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list,
        B, WB, Iw, Ix, Ky, V, U, S, stream);
}



std::vector<torch::Tensor> launcher_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused(
    torch::Tensor w,           // [WB,Iw,U]
    torch::Tensor x,       // [S,Ix,U]
    torch::Tensor y,           // [B,Ky,1] or [S,Ky,1]
    torch::Tensor grad_out,    // [B,V,U] or [S,V,U]
    torch::Tensor src_idx,    // [?] int32
    torch::Tensor dst_idx,    // [?] int32
    torch::Tensor b_list,    // [B] int32 optional
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");

    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");

    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U]");
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

    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");
    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");


    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }

    auto grad_x = torch::zeros_like(x);
    auto grad_w = torch::zeros_like(w);
    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused", [&] {
        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (scalar_t*)grad_w.data_ptr<scalar_t>(),
            (scalar_t*)grad_x.data_ptr<scalar_t>(),
            (scalar_t*)grad_y.data_ptr<scalar_t>(),
            (const int32_t*)src_idx.data_ptr<int32_t>(),
            (const int32_t*)dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_w, grad_x, grad_y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused, "uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_bwd_fused backward fused jit impl");
}
