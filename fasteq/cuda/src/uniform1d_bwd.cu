#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include "cuda_utils.hpp"

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void u1d_groupout_bwd(
    const scalar_t* __restrict__ w,        // [B, Iw, U]
    const scalar_t* __restrict__ x,        // [B, Ix, U]
    const scalar_t* __restrict__ y,        // [B, Ky, 1]
    const scalar_t* __restrict__ grad_out, // [B, V,  U]
    scalar_t* __restrict__ grad_w,         // [B, Iw, U]
    scalar_t* __restrict__ grad_x,         // [B, Ix, U]
    scalar_t* __restrict__ grad_y,         // [B, Ky, 1]
    const int32_t* __restrict__ i_list,    // [P]
    const int32_t* __restrict__ j_list,    // [P]
    const int32_t* __restrict__ k_list,    // [P]
    const scalar_t* __restrict__ coeff_list,// [P]
    const int32_t* __restrict__ v_offsets, // [V+1]
    int B, int Iw, int Ix, int Ky, int V, int U)
{
    int b = (int)blockIdx.x;
    int v = (int)blockIdx.y;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;     // 0..WARPS_PER_BLOCK-1
    int lane = tid &  31;    // 0..31

    int tiles = U >> 5; // U/32
    int u_tile = (int)blockIdx.z * WARPS_PER_BLOCK + warp;
    if (u_tile >= tiles) return;

    int u = (u_tile << 5) + lane; // u_tile*32 + lane

    int start = v_offsets[v];
    int end   = v_offsets[v + 1];

    // grad_out for this (b,v,u)
    scalar_t go = grad_out[((b * V + v) * U) + u];

    // 遍历该 v 的所有 path
    for (int t = start; t < end; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        scalar_t c = coeff_list[t];

        // y 是标量：warp 内广播
        scalar_t yval;
        if (lane == 0) {
            yval = y[(b * Ky + k) * 1 + 0];
        }
        yval = __shfl_sync(0xffffffff, yval, 0);

        scalar_t wval = w[((b * Iw + i) * U) + u];
        scalar_t xval = x[((b * Ix + j) * U) + u];

        // dw += c * x * y * go
        atomicAdd(&grad_w[((b * Iw + i) * U) + u], c * xval * yval * go);

        // dx += c * w * y * go
        atomicAdd(&grad_x[((b * Ix + j) * U) + u], c * wval * yval * go);

        // dy += c * sum_u (w * x * go)
        scalar_t dy_lane = c * wval * xval * go;
        scalar_t dy_sum  = warp_sum(dy_lane);
        if (lane == 0) {
            atomicAdd(&grad_y[(b * Ky + k) * 1 + 0], dy_sum);
        }
    }
}

std::vector<torch::Tensor> uniform1d_bwd_launch(
    torch::Tensor w,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor grad_out,
    torch::Tensor i_list,
    torch::Tensor j_list,
    torch::Tensor k_list,
    torch::Tensor coeff_list,
    torch::Tensor v_offsets,
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");

    TORCH_CHECK(i_list.scalar_type() == torch::kInt32, "i_list must be int32");
    TORCH_CHECK(j_list.scalar_type() == torch::kInt32, "j_list must be int32");
    TORCH_CHECK(k_list.scalar_type() == torch::kInt32, "k_list must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");

    TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w dtype");
    TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w dtype");
    TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w dtype");
    TORCH_CHECK(coeff_list.scalar_type() == w.scalar_type(), "coeff_list dtype must match w dtype");

    int B  = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK((int)x.size(0) == B && (int)x.size(2) == U, "x shape mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(2) == 1, "y shape mismatch (expect [B,Ky,1])");
    TORCH_CHECK((int)grad_out.size(0) == B && (int)grad_out.size(1) == V && (int)grad_out.size(2) == U,
                "grad_out shape mismatch (expect [B,V,U])");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros_like(y);

    int tiles = U / 32;
    int warps = (tiles >= 8) ? 8 : (tiles >= 4) ? 4 : (tiles >= 2) ? 2 : 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (warps == 8) {
        dim3 block(32 * 8);
        dim3 grid(B, V, ceil_div_int(tiles, 8));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_bwd_8", [&]{
            u1d_groupout_bwd<scalar_t, 8><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U
            );
        });
    } else if (warps == 4) {
        dim3 block(32 * 4);
        dim3 grid(B, V, ceil_div_int(tiles, 4));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_bwd_4", [&]{
            u1d_groupout_bwd<scalar_t, 4><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U
            );
        });
    } else if (warps == 2) {
        dim3 block(32 * 2);
        dim3 grid(B, V, ceil_div_int(tiles, 2));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_bwd_2", [&]{
            u1d_groupout_bwd<scalar_t, 2><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U
            );
        });
    } else {
        dim3 block(32);
        dim3 grid(B, V, tiles);
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_bwd_1", [&]{
            u1d_groupout_bwd<scalar_t, 1><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U
            );
        });
    }

    
    return {grad_w, grad_x, grad_y}; // <- 不推荐这种 stack（形状不同会报错）
}


TORCH_LIBRARY(u1d_bwd, m)
{
    m.def("backward", &uniform1d_bwd_launch);
}
