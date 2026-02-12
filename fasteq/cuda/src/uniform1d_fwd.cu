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
__global__ void u1d_groupout_fwd(
    const scalar_t* __restrict__ w,   // [B, Iw, U]
    const scalar_t* __restrict__ x,   // [B, Ix, U]
    const scalar_t* __restrict__ y,   // [B, Ky, 1]
    scalar_t* __restrict__ out,       // [B, V, U]
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    const int32_t* __restrict__ v_offsets,   // [V+1]
    int B, int Iw, int Ix, int Ky, int V, int U)
{
    int b = (int)blockIdx.x;
    int v = (int)blockIdx.y;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid &  31;

    int u_tile = (int)blockIdx.z * WARPS_PER_BLOCK + warp;
    int u = (u_tile << 5) + lane;
    if (u >= U) return;

    int start = v_offsets[v];
    int end   = v_offsets[v + 1];

    scalar_t acc = (scalar_t)0;

    for (int t = start; t < end; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        scalar_t c = coeff_list[t];

        // y[b,k,0] 对该 path 的所有 u 都是同一个标量：每个 warp 用 lane0 读一次，然后 warp 内广播
        scalar_t yval;
        if (lane == 0) {
            yval = y[(b * Ky + k) * 1 + 0];
        }
        yval = __shfl_sync(0xffffffff, yval, 0);

        scalar_t wval = w[((b * Iw + i) * U) + u];
        scalar_t xval = x[((b * Ix + j) * U) + u];

        acc += c * wval * xval * yval;
    }

    out[((b * V + v) * U) + u] += acc;
}

torch::Tensor uniform1d_fwd_launch(
    torch::Tensor w,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor i_list,
    torch::Tensor j_list,
    torch::Tensor k_list,
    torch::Tensor coeff_list,
    torch::Tensor v_offsets,
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous(), "w/x/y must be contiguous");
    TORCH_CHECK(i_list.scalar_type() == torch::kInt32, "i_list must be int32");
    TORCH_CHECK(j_list.scalar_type() == torch::kInt32, "j_list must be int32");
    TORCH_CHECK(k_list.scalar_type() == torch::kInt32, "k_list must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");

    int B  = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(x.size(0) == w.size(0) && x.size(2) == w.size(2), "x shape mismatch");
    TORCH_CHECK(y.size(0) == w.size(0) && y.size(2) == 1, "y shape mismatch (expect [B,Ky,1])");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto out = torch::zeros({B, V, U}, w.options());

    int tiles = U / 32;
    int warps = (tiles >= 8) ? 8 : (tiles >= 4) ? 4 : (tiles >= 2) ? 2 : 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (warps == 8) {
        dim3 block(32 * 8);
        dim3 grid(B, V, ceil_div_int(tiles, 8));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_fwd", [&]{
            u1d_groupout_fwd<scalar_t, 8><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_fwd", [&]{
            u1d_groupout_fwd<scalar_t, 4><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_fwd", [&]{
            u1d_groupout_fwd<scalar_t, 2><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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
        dim3 grid(B, V, tiles); // 每个 block 一个 warp 一个 tile
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_groupout_fwd", [&]{
            u1d_groupout_fwd<scalar_t, 1><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U
            );
        });
    }

    return out;
}

TORCH_LIBRARY(u1d_fwd, m)
{
    m.def("forward", &uniform1d_fwd_launch);
}
