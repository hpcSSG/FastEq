#include <stdint.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void uniform1d_combine_u224_p16_bwd(
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

    // init grad_y accumulators
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

    // u traversal: persistent inner loop over U
    for (int u_base = 0; u_base < U; u_base += 32) {
        int u = u_base + lane;
        if (u < U) {
            // preload w(i,u)
            scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];
            scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
            scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
            scalar_t wi_0 = w[((int64_t)b * Iw + 0) * (int64_t)U + u];
        
            // preload x(j,u)
            scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];
        
            // init per-u accumulators
            scalar_t gw_acc_i_3 = scalar_t(0);
            scalar_t gw_acc_i_2 = scalar_t(0);
            scalar_t gw_acc_i_1 = scalar_t(0);
            scalar_t gw_acc_i_0 = scalar_t(0);
            scalar_t gx_acc_j_0 = scalar_t(0);
        
            // ---- v tile 0 ----
            scalar_t go_v_0 = grad_out[((int64_t)dst * V + 0) * (int64_t)U + u];
            scalar_t go_v_1 = grad_out[((int64_t)dst * V + 1) * (int64_t)U + u];
            scalar_t go_v_2 = grad_out[((int64_t)dst * V + 2) * (int64_t)U + u];
            scalar_t go_v_3 = grad_out[((int64_t)dst * V + 3) * (int64_t)U + u];
        
            // grad_w i-tile 0
        
            // grad_w i-tile 1
            gw_acc_i_1 += xj_0 * yk_1 * go_v_1;
            gw_acc_i_1 += xj_0 * yk_2 * go_v_2;
            gw_acc_i_1 += xj_0 * yk_3 * go_v_3;
            gw_acc_i_0 += xj_0 * yk_0 * go_v_0;
        
            // grad_x j-tile 0
            gx_acc_j_0 += wi_0 * yk_0 * go_v_0;
            gx_acc_j_0 += wi_1 * yk_1 * go_v_1;
            gx_acc_j_0 += wi_1 * yk_2 * go_v_2;
            gx_acc_j_0 += wi_1 * yk_3 * go_v_3;
        
            // grad_y k-tile 0
            gy_acc_k_0 += wi_0 * xj_0 * go_v_0;
            gy_acc_k_1 += wi_1 * xj_0 * go_v_1;
        
            // grad_y k-tile 1
            gy_acc_k_2 += wi_1 * xj_0 * go_v_2;
            gy_acc_k_3 += wi_1 * xj_0 * go_v_3;
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
        
            // ---- v tile 1 ----
            scalar_t go_v_4 = grad_out[((int64_t)dst * V + 4) * (int64_t)U + u];
            scalar_t go_v_5 = grad_out[((int64_t)dst * V + 5) * (int64_t)U + u];
            scalar_t go_v_6 = grad_out[((int64_t)dst * V + 6) * (int64_t)U + u];
            scalar_t go_v_7 = grad_out[((int64_t)dst * V + 7) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_2 += xj_0 * yk_4 * go_v_4;
            gw_acc_i_2 += xj_0 * yk_5 * go_v_5;
            gw_acc_i_2 += xj_0 * yk_6 * go_v_6;
            gw_acc_i_2 += xj_0 * yk_7 * go_v_7;
        
            // grad_w i-tile 1
        
            // grad_x j-tile 0
            gx_acc_j_0 += wi_2 * yk_4 * go_v_4;
            gx_acc_j_0 += wi_2 * yk_5 * go_v_5;
            gx_acc_j_0 += wi_2 * yk_6 * go_v_6;
            gx_acc_j_0 += wi_2 * yk_7 * go_v_7;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
            gy_acc_k_4 += wi_2 * xj_0 * go_v_4;
            gy_acc_k_5 += wi_2 * xj_0 * go_v_5;
        
            // grad_y k-tile 3
            gy_acc_k_6 += wi_2 * xj_0 * go_v_6;
            gy_acc_k_7 += wi_2 * xj_0 * go_v_7;
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
        
            // ---- v tile 2 ----
            scalar_t go_v_8 = grad_out[((int64_t)dst * V + 8) * (int64_t)U + u];
            scalar_t go_v_9 = grad_out[((int64_t)dst * V + 9) * (int64_t)U + u];
            scalar_t go_v_10 = grad_out[((int64_t)dst * V + 10) * (int64_t)U + u];
            scalar_t go_v_11 = grad_out[((int64_t)dst * V + 11) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_3 += xj_0 * yk_9 * go_v_9;
            gw_acc_i_3 += xj_0 * yk_10 * go_v_10;
            gw_acc_i_3 += xj_0 * yk_11 * go_v_11;
            gw_acc_i_2 += xj_0 * yk_8 * go_v_8;
        
            // grad_w i-tile 1
        
            // grad_x j-tile 0
            gx_acc_j_0 += wi_2 * yk_8 * go_v_8;
            gx_acc_j_0 += wi_3 * yk_9 * go_v_9;
            gx_acc_j_0 += wi_3 * yk_10 * go_v_10;
            gx_acc_j_0 += wi_3 * yk_11 * go_v_11;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
            gy_acc_k_8 += wi_2 * xj_0 * go_v_8;
            gy_acc_k_9 += wi_3 * xj_0 * go_v_9;
        
            // grad_y k-tile 5
            gy_acc_k_10 += wi_3 * xj_0 * go_v_10;
            gy_acc_k_11 += wi_3 * xj_0 * go_v_11;
        
            // grad_y k-tile 6
        
            // grad_y k-tile 7
        
            // ---- v tile 3 ----
            scalar_t go_v_12 = grad_out[((int64_t)dst * V + 12) * (int64_t)U + u];
            scalar_t go_v_13 = grad_out[((int64_t)dst * V + 13) * (int64_t)U + u];
            scalar_t go_v_14 = grad_out[((int64_t)dst * V + 14) * (int64_t)U + u];
            scalar_t go_v_15 = grad_out[((int64_t)dst * V + 15) * (int64_t)U + u];
        
            // grad_w i-tile 0
            gw_acc_i_3 += xj_0 * yk_12 * go_v_12;
            gw_acc_i_3 += xj_0 * yk_13 * go_v_13;
            gw_acc_i_3 += xj_0 * yk_14 * go_v_14;
            gw_acc_i_3 += xj_0 * yk_15 * go_v_15;
        
            // grad_w i-tile 1
        
            // grad_x j-tile 0
            gx_acc_j_0 += wi_3 * yk_12 * go_v_12;
            gx_acc_j_0 += wi_3 * yk_13 * go_v_13;
            gx_acc_j_0 += wi_3 * yk_14 * go_v_14;
            gx_acc_j_0 += wi_3 * yk_15 * go_v_15;
        
            // grad_y k-tile 0
        
            // grad_y k-tile 1
        
            // grad_y k-tile 2
        
            // grad_y k-tile 3
        
            // grad_y k-tile 4
        
            // grad_y k-tile 5
        
            // grad_y k-tile 6
            gy_acc_k_12 += wi_3 * xj_0 * go_v_12;
            gy_acc_k_13 += wi_3 * xj_0 * go_v_13;
        
            // grad_y k-tile 7
            gy_acc_k_14 += wi_3 * xj_0 * go_v_14;
            gy_acc_k_15 += wi_3 * xj_0 * go_v_15;
        
            // direct-store grad_w for this u
            grad_w[((int64_t)b * Iw + 3) * (int64_t)U + u] = gw_acc_i_3;
            grad_w[((int64_t)b * Iw + 2) * (int64_t)U + u] = gw_acc_i_2;
            grad_w[((int64_t)b * Iw + 1) * (int64_t)U + u] = gw_acc_i_1;
            grad_w[((int64_t)b * Iw + 0) * (int64_t)U + u] = gw_acc_i_0;
        
            // atomic grad_x for this u
            atomicAdd(&grad_x[((int64_t)src * Ix + 0) * (int64_t)U + u], gx_acc_j_0);
        }

    }

    // tail warp-reduce grad_y
    scalar_t gy_sum_k_0 = warp_sum_xor(gy_acc_k_0);
    if (lane == 0) atomicAdd(&grad_y[y_base + 0], gy_sum_k_0);
    scalar_t gy_sum_k_1 = warp_sum_xor(gy_acc_k_1);
    if (lane == 0) atomicAdd(&grad_y[y_base + 1], gy_sum_k_1);
    scalar_t gy_sum_k_2 = warp_sum_xor(gy_acc_k_2);
    if (lane == 0) atomicAdd(&grad_y[y_base + 2], gy_sum_k_2);
    scalar_t gy_sum_k_3 = warp_sum_xor(gy_acc_k_3);
    if (lane == 0) atomicAdd(&grad_y[y_base + 3], gy_sum_k_3);
    scalar_t gy_sum_k_4 = warp_sum_xor(gy_acc_k_4);
    if (lane == 0) atomicAdd(&grad_y[y_base + 4], gy_sum_k_4);
    scalar_t gy_sum_k_5 = warp_sum_xor(gy_acc_k_5);
    if (lane == 0) atomicAdd(&grad_y[y_base + 5], gy_sum_k_5);
    scalar_t gy_sum_k_6 = warp_sum_xor(gy_acc_k_6);
    if (lane == 0) atomicAdd(&grad_y[y_base + 6], gy_sum_k_6);
    scalar_t gy_sum_k_7 = warp_sum_xor(gy_acc_k_7);
    if (lane == 0) atomicAdd(&grad_y[y_base + 7], gy_sum_k_7);
    scalar_t gy_sum_k_8 = warp_sum_xor(gy_acc_k_8);
    if (lane == 0) atomicAdd(&grad_y[y_base + 8], gy_sum_k_8);
    scalar_t gy_sum_k_9 = warp_sum_xor(gy_acc_k_9);
    if (lane == 0) atomicAdd(&grad_y[y_base + 9], gy_sum_k_9);
    scalar_t gy_sum_k_10 = warp_sum_xor(gy_acc_k_10);
    if (lane == 0) atomicAdd(&grad_y[y_base + 10], gy_sum_k_10);
    scalar_t gy_sum_k_11 = warp_sum_xor(gy_acc_k_11);
    if (lane == 0) atomicAdd(&grad_y[y_base + 11], gy_sum_k_11);
    scalar_t gy_sum_k_12 = warp_sum_xor(gy_acc_k_12);
    if (lane == 0) atomicAdd(&grad_y[y_base + 12], gy_sum_k_12);
    scalar_t gy_sum_k_13 = warp_sum_xor(gy_acc_k_13);
    if (lane == 0) atomicAdd(&grad_y[y_base + 13], gy_sum_k_13);
    scalar_t gy_sum_k_14 = warp_sum_xor(gy_acc_k_14);
    if (lane == 0) atomicAdd(&grad_y[y_base + 14], gy_sum_k_14);
    scalar_t gy_sum_k_15 = warp_sum_xor(gy_acc_k_15);
    if (lane == 0) atomicAdd(&grad_y[y_base + 15], gy_sum_k_15);

}

template <typename scalar_t>
void launch_uniform1d_combine_u224_p16_bwd(
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
    uniform1d_combine_u224_p16_bwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, grad_out, grad_w, grad_x, grad_y,
        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);
}



std::vector<torch::Tensor> launcher_uniform1d_combine_u224_p16_bwd(
    torch::Tensor grad_out,     // [S,V,U]
    torch::Tensor w,            // [B,Iw,U]
    torch::Tensor x_all,        // [S,Ix,U]
    torch::Tensor y,            // [B,Ky,1]
    torch::Tensor src_idx,      // [B] int32
    torch::Tensor dst_idx,      // [B] int32
    torch::Tensor b_list,       // [B] int32
    int64_t V64)
{

    TORCH_CHECK(w.scalar_type() == grad_out.scalar_type(),
                "w and grad_out must have the same dtype");
    TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                "x_all dtype must match w");
    TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                "y dtype must match w");

    TORCH_CHECK(w.dim() == 3, "w must be [B,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be [B,Ky,1]");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be [S,V,U]");

    const int B  = (int)w.size(0);
    const int Iw = (int)w.size(1);
    const int U  = (int)w.size(2);

    const int S  = (int)x_all.size(0);
    const int Ix = (int)x_all.size(1);

    const int Ky = (int)y.size(1);
    const int V  = (int)V64;

    TORCH_CHECK((int)x_all.size(0) == S, "internal shape error for x_all");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B, "y B mismatch");
    TORCH_CHECK((int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)grad_out.size(0) == S, "grad_out S mismatch");
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");

    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "dst_idx must be [B]");
    TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");

    TORCH_CHECK(U > 0, "U must be > 0");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    TORCH_CHECK(B >= 0 && S >= 0 && Iw >= 0 && Ix >= 0 && Ky >= 0 && V >= 0,
                "invalid negative shape");

    c10::cuda::CUDAGuard device_guard(w.device());

    auto grad_w = torch::zeros_like(w);      // [B,Iw,U]
    auto grad_x = torch::zeros_like(x_all);  // [S,Ix,U]
    auto grad_y = torch::zeros_like(y);      // [B,Ky,1]

    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_combine_u224_p16_bwd", [&] {

        launch_uniform1d_combine_u224_p16_bwd<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x_all.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (scalar_t*)grad_w.data_ptr<scalar_t>(),
            (scalar_t*)grad_x.data_ptr<scalar_t>(),
            (scalar_t*)grad_y.data_ptr<scalar_t>(),
            (const int32_t*)src_idx.data_ptr<int32_t>(),
            (const int32_t*)dst_idx.data_ptr<int32_t>(),
            (const int32_t*)b_list.data_ptr<int32_t>(),
            B, Iw, Ix, Ky, V, U, stream);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_w, grad_x, grad_y};
}

TORCH_LIBRARY(uniform1d_combine_u224_p16_bwd_codegen, m) {
    m.def("run", &launcher_uniform1d_combine_u224_p16_bwd);
}
