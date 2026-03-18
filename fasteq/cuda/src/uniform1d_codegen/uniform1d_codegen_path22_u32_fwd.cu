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
__global__ void uniform1d_codegen_path22_u32_fwd(
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

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= 2) return;

    int u = (ublk << 5) + lane;
    if (u >= U) return;

    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base = (int64_t)b   * Iw * 32;
    int64_t x_base = (int64_t)src * Ix * 32;
    int64_t y_base = (int64_t)b   * Ky;
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    if (warp == 0) {
        // preload w(i)
        scalar_t wi_7 = w[w_base + 7LL * 32 + lane];
        scalar_t wi_4 = w[w_base + 4LL * 32 + lane];
        scalar_t wi_1 = w[w_base + 1LL * 32 + lane];

        // preload x(j)
        scalar_t xj_15 = x_all[x_base + 15LL * 32 + lane];
        scalar_t xj_16 = x_all[x_base + 16LL * 32 + lane];
        scalar_t xj_17 = x_all[x_base + 17LL * 32 + lane];
        scalar_t xj_18 = x_all[x_base + 18LL * 32 + lane];
        scalar_t xj_19 = x_all[x_base + 19LL * 32 + lane];
        scalar_t xj_20 = x_all[x_base + 20LL * 32 + lane];
        scalar_t xj_21 = x_all[x_base + 21LL * 32 + lane];
        scalar_t xj_4 = x_all[x_base + 4LL * 32 + lane];
        scalar_t xj_6 = x_all[x_base + 6LL * 32 + lane];
        scalar_t xj_8 = x_all[x_base + 8LL * 32 + lane];
        scalar_t xj_1 = x_all[x_base + 1LL * 32 + lane];

        // preload y(k)
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_7 = scalar_t(0);
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_15 * yk_9;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_16 * yk_10;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_17 * yk_11;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_18 * yk_12;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_19 * yk_13;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_20 * yk_14;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_21 * yk_15;

        scalar_t sum_v_4 = scalar_t(0);
        sum_v_4 += scalar_t(0.577350259f) * wi_4 * xj_4 * yk_1;
        sum_v_4 += scalar_t(0.577350259f) * wi_4 * xj_6 * yk_2;
        sum_v_4 += scalar_t(0.577350259f) * wi_4 * xj_8 * yk_3;

        scalar_t sum_v_1 = scalar_t(0);
        sum_v_1 += wi_1 * xj_1 * yk_0;

        // writeback
        atomicAdd(&out[o_base + (7LL << 5)], sum_v_7);
        atomicAdd(&out[o_base + (4LL << 5)], sum_v_4);
        atomicAdd(&out[o_base + (1LL << 5)], sum_v_1);
    }

    if (warp == 1) {
        // preload w(i)
        scalar_t wi_6 = w[w_base + 6LL * 32 + lane];
        scalar_t wi_5 = w[w_base + 5LL * 32 + lane];
        scalar_t wi_3 = w[w_base + 3LL * 32 + lane];
        scalar_t wi_2 = w[w_base + 2LL * 32 + lane];
        scalar_t wi_0 = w[w_base + 0LL * 32 + lane];

        // preload x(j)
        scalar_t xj_10 = x_all[x_base + 10LL * 32 + lane];
        scalar_t xj_11 = x_all[x_base + 11LL * 32 + lane];
        scalar_t xj_12 = x_all[x_base + 12LL * 32 + lane];
        scalar_t xj_13 = x_all[x_base + 13LL * 32 + lane];
        scalar_t xj_14 = x_all[x_base + 14LL * 32 + lane];
        scalar_t xj_5 = x_all[x_base + 5LL * 32 + lane];
        scalar_t xj_7 = x_all[x_base + 7LL * 32 + lane];
        scalar_t xj_9 = x_all[x_base + 9LL * 32 + lane];
        scalar_t xj_3 = x_all[x_base + 3LL * 32 + lane];
        scalar_t xj_2 = x_all[x_base + 2LL * 32 + lane];
        scalar_t xj_0 = x_all[x_base + 0LL * 32 + lane];

        // preload y(k)
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_6 = scalar_t(0);
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_10 * yk_4;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_11 * yk_5;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_12 * yk_6;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_13 * yk_7;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_14 * yk_8;

        scalar_t sum_v_5 = scalar_t(0);
        sum_v_5 += scalar_t(0.577350259f) * wi_5 * xj_5 * yk_1;
        sum_v_5 += scalar_t(0.577350259f) * wi_5 * xj_7 * yk_2;
        sum_v_5 += scalar_t(0.577350259f) * wi_5 * xj_9 * yk_3;

        scalar_t sum_v_3 = scalar_t(0);
        sum_v_3 += wi_3 * xj_3 * yk_0;

        scalar_t sum_v_2 = scalar_t(0);
        sum_v_2 += wi_2 * xj_2 * yk_0;

        scalar_t sum_v_0 = scalar_t(0);
        sum_v_0 += wi_0 * xj_0 * yk_0;

        // writeback
        atomicAdd(&out[o_base + (6LL << 5)], sum_v_6);
        atomicAdd(&out[o_base + (5LL << 5)], sum_v_5);
        atomicAdd(&out[o_base + (3LL << 5)], sum_v_3);
        atomicAdd(&out[o_base + (2LL << 5)], sum_v_2);
        atomicAdd(&out[o_base + (0LL << 5)], sum_v_0);
    }

}

// launcher helper
template <typename scalar_t>
void launch_uniform1d_codegen_path22_u32_fwd(
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
    dim3 block(64);  // 2 warps
    dim3 grid(B, (U + 31) / 32);
    uniform1d_codegen_path22_u32_fwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, out, src_idx, dst_idx, b_list,
        B, Iw, Ix, Ky, V, U);
}


torch::Tensor launcher_uniform1d_codegen_path22_u32_fwd(
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

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_codegen_path22_u32_fwd", [&] {

        launch_uniform1d_codegen_path22_u32_fwd<scalar_t>(
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

TORCH_LIBRARY(uniform1d_codegen_path22_u32_fwd_codegen, m) {
    m.def("run", &launcher_uniform1d_codegen_path22_u32_fwd);
}
