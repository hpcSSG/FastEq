#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cstdint>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void uniform1d_combine_u128_path16_fwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x_all,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V, int U)
{
    int b_global = (int)blockIdx.x;
    int ublk     = (int)blockIdx.y;
    if (b_global >= B) return;

    int b = b_list ? b_list[b_global] : b_global;

    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= 2) return;

    int u = (ublk << 5) + lane;
    if (u >= U) return;

    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base = (int64_t)b   * (int64_t)Iw * (int64_t)U;
    int64_t x_base = (int64_t)src * (int64_t)Ix * (int64_t)U;
    int64_t y_base = (int64_t)b   * (int64_t)Ky;

    if (warp == 0) {
        // preload w(i)
        scalar_t wi_0 = w[w_base + (int64_t)0 * (int64_t)U + u];
        scalar_t wi_1 = w[w_base + (int64_t)1 * (int64_t)U + u];
        scalar_t wi_2 = w[w_base + (int64_t)2 * (int64_t)U + u];
        scalar_t wi_3 = w[w_base + (int64_t)3 * (int64_t)U + u];

        // preload x(j)
        scalar_t xj_0 = x_all[x_base + (int64_t)0 * (int64_t)U + u];

        // preload y(k)
        scalar_t yk_0 = y[y_base + 0];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];

        // per-v accumulation
        scalar_t sum_v_0 = scalar_t(0);
        sum_v_0 += wi_0 * xj_0 * yk_0;

        scalar_t sum_v_2 = scalar_t(0);
        sum_v_2 += wi_1 * xj_0 * yk_2;

        scalar_t sum_v_4 = scalar_t(0);
        sum_v_4 += wi_2 * xj_0 * yk_4;

        scalar_t sum_v_6 = scalar_t(0);
        sum_v_6 += wi_2 * xj_0 * yk_6;

        scalar_t sum_v_8 = scalar_t(0);
        sum_v_8 += wi_2 * xj_0 * yk_8;

        scalar_t sum_v_10 = scalar_t(0);
        sum_v_10 += wi_3 * xj_0 * yk_10;

        scalar_t sum_v_12 = scalar_t(0);
        sum_v_12 += wi_3 * xj_0 * yk_12;

        scalar_t sum_v_14 = scalar_t(0);
        sum_v_14 += wi_3 * xj_0 * yk_14;

        // writeback
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)0) * (int64_t)U + u], sum_v_0);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)2) * (int64_t)U + u], sum_v_2);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)4) * (int64_t)U + u], sum_v_4);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)6) * (int64_t)U + u], sum_v_6);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)8) * (int64_t)U + u], sum_v_8);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)10) * (int64_t)U + u], sum_v_10);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)12) * (int64_t)U + u], sum_v_12);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)14) * (int64_t)U + u], sum_v_14);
    }

    if (warp == 1) {
        // preload w(i)
        scalar_t wi_1 = w[w_base + (int64_t)1 * (int64_t)U + u];
        scalar_t wi_2 = w[w_base + (int64_t)2 * (int64_t)U + u];
        scalar_t wi_3 = w[w_base + (int64_t)3 * (int64_t)U + u];

        // preload x(j)
        scalar_t xj_0 = x_all[x_base + (int64_t)0 * (int64_t)U + u];

        // preload y(k)
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_15 = y[y_base + 15];

        // per-v accumulation
        scalar_t sum_v_1 = scalar_t(0);
        sum_v_1 += wi_1 * xj_0 * yk_1;

        scalar_t sum_v_3 = scalar_t(0);
        sum_v_3 += wi_1 * xj_0 * yk_3;

        scalar_t sum_v_5 = scalar_t(0);
        sum_v_5 += wi_2 * xj_0 * yk_5;

        scalar_t sum_v_7 = scalar_t(0);
        sum_v_7 += wi_2 * xj_0 * yk_7;

        scalar_t sum_v_9 = scalar_t(0);
        sum_v_9 += wi_3 * xj_0 * yk_9;

        scalar_t sum_v_11 = scalar_t(0);
        sum_v_11 += wi_3 * xj_0 * yk_11;

        scalar_t sum_v_13 = scalar_t(0);
        sum_v_13 += wi_3 * xj_0 * yk_13;

        scalar_t sum_v_15 = scalar_t(0);
        sum_v_15 += wi_3 * xj_0 * yk_15;

        // writeback
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)1) * (int64_t)U + u], sum_v_1);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)3) * (int64_t)U + u], sum_v_3);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)5) * (int64_t)U + u], sum_v_5);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)7) * (int64_t)U + u], sum_v_7);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)9) * (int64_t)U + u], sum_v_9);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)11) * (int64_t)U + u], sum_v_11);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)13) * (int64_t)U + u], sum_v_13);
        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t)15) * (int64_t)U + u], sum_v_15);
    }

}

// launcher helper
template <typename scalar_t>
void launch_uniform1d_combine_u128_path16_fwd(
    const scalar_t* w,
    const scalar_t* x_all,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int Iw, int Ix, int Ky, int V, int U,
    cudaStream_t stream)
{
    dim3 block(64);
    dim3 grid(B, (U + 31) / 32);
    uniform1d_combine_u128_path16_fwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, out, src_idx, dst_idx, b_list,
        B, Iw, Ix, Ky, V, U);
}


torch::Tensor launcher_uniform1d_combine_u128_path16_fwd(
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // [B,Ky,1]
    torch::Tensor src_idx,    // [B] int32
    torch::Tensor dst_idx,    // [B] int32
    torch::Tensor b_list,     // [B] int32
    int64_t V64)
{

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(dst_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "w/x_all/y must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");

    int B  = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);

    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto out = torch::zeros({S, V, U}, w.options());

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_combine_u128_path16_fwd", [&] {

        launch_uniform1d_combine_u128_path16_fwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U, stream);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

TORCH_LIBRARY(uniform1d_combine_u128_path16_fwd_codegen, m) {
    m.def("run", &launcher_uniform1d_combine_u128_path16_fwd);
}
