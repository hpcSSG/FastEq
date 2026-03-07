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
__global__ void u1d_groupout_bwd_gather_scatterB_csr(
    const scalar_t* __restrict__ grad_out, // [S, V, U]
    const scalar_t* __restrict__ w,        // [B, Iw, U]
    const scalar_t* __restrict__ x_all,    // [S, Ix, U]
    const scalar_t* __restrict__ y,        // [B, Ky, 1] (flattened)
    scalar_t* __restrict__ grad_w,         // [B, Iw, U]
    scalar_t* __restrict__ grad_x_all,     // [S, Ix, U]
    scalar_t* __restrict__ grad_y,         // [B, Ky, 1] (flattened)
    const int32_t* __restrict__ src_idx,   // [B] in [0..S-1]
    const int32_t* __restrict__ b_list,    // [B] b indices sorted by class
    const int32_t* __restrict__ cls_offsets,// [S+1]
    const int32_t* __restrict__ i_list,    // [P]
    const int32_t* __restrict__ j_list,    // [P]
    const int32_t* __restrict__ k_list,    // [P]
    const scalar_t* __restrict__ coeff_list,// [P]
    const int32_t* __restrict__ v_offsets, // [V+1]
    int B, int S, int Iw, int Ix, int Ky, int V, int U)
{
    int cls = (int)blockIdx.x;  // 0..S-1
    int v   = (int)blockIdx.y;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;        // 0..WARPS_PER_BLOCK-1
    int lane = tid & 31;

    int u_tile = (int)blockIdx.z * WARPS_PER_BLOCK + warp;
    int u = (u_tile << 5) + lane;
    if (u >= U) return;

    // paths for this v
    int t0 = v_offsets[v];
    int t1 = v_offsets[v + 1];

    // b bucket for this cls
    int bi0 = cls_offsets[cls];
    int bi1 = cls_offsets[cls + 1];

    // upstream grad for this output element
    scalar_t go = grad_out[((cls * V + v) * U) + u];

    // reduce over b in this cls
    for (int bi = bi0; bi < bi1; ++bi) {
        int b = b_list[bi];
        int src = src_idx[b];

        // loop paths in this v bucket
        for (int t = t0; t < t1; ++t) {
            int i = i_list[t];
            int j = j_list[t];
            int k = k_list[t];
            scalar_t c = coeff_list[t];

            // y[b,k,0] scalar, broadcast within warp
            scalar_t yval;
            if (lane == 0) {
                yval = y[(b * Ky + k)]; // flatten [B,Ky,1]
            }
            yval = __shfl_sync(0xffffffff, yval, 0);

            // forward values
            scalar_t wval = w[((b   * Iw + i)   * U) + u];
            scalar_t xval = x_all[((src * Ix + j) * U) + u];

            // dw += go * c * x * y
            scalar_t dw = go * c * xval * yval;
            atomicAdd(&grad_w[((b * Iw + i) * U) + u], dw);

            // dx += go * c * w * y
            scalar_t dx = go * c * wval * yval;
            atomicAdd(&grad_x_all[((src * Ix + j) * U) + u], dx);

            // dy += sum_u go * c * w * x   (scalar per (b,k))
            scalar_t dy_lane = go * c * wval * xval;
            scalar_t dy_sum = warp_sum(dy_lane);
            if (lane == 0) {
                atomicAdd(&grad_y[(b * Ky + k)], dy_sum);
            }
        }
    }
}

std::vector<torch::Tensor> u1d_fused_bwd_launch(
    torch::Tensor grad_out,   // [S,V,U]
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // [B,Ky,1]
    torch::Tensor src_idx,    // [B] int32
    torch::Tensor b_list,     // [B] int32
    torch::Tensor cls_offsets,// [S+1] int32
    torch::Tensor i_list,     // [P] int32
    torch::Tensor j_list,     // [P] int32
    torch::Tensor k_list,     // [P] int32
    torch::Tensor coeff_list, // [P] same dtype
    torch::Tensor v_offsets,  // [V+1] int32
    int64_t Iw64, int64_t Ix64, int64_t Ky64, int64_t V64, int64_t U64)
{
    TORCH_CHECK(grad_out.is_cuda() && w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "all must be CUDA");
    TORCH_CHECK(grad_out.is_contiguous() && w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx int32");
    TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list int32");
    TORCH_CHECK(cls_offsets.scalar_type() == torch::kInt32, "cls_offsets int32");
    TORCH_CHECK(i_list.scalar_type() == torch::kInt32, "i_list int32");
    TORCH_CHECK(j_list.scalar_type() == torch::kInt32, "j_list int32");
    TORCH_CHECK(k_list.scalar_type() == torch::kInt32, "k_list int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets int32");
    TORCH_CHECK(coeff_list.scalar_type() == w.scalar_type(), "coeff dtype must match w");
    TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

    int B  = (int)y.size(0);
    int Iw = (int)Iw64;
    int Ix = (int)Ix64;
    int Ky = (int)Ky64;
    int V  = (int)V64;
    int U  = (int)U64;
    int S  = (int)x_all.size(0);

    TORCH_CHECK(U % 32 == 0, "U must be multiple of 32");
    TORCH_CHECK((int)grad_out.size(0) == S && (int)grad_out.size(1) == V && (int)grad_out.size(2) == U, "grad_out shape mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(1) == Ky && (int)y.size(2) == 1, "y must be [B,Ky,1]");

    auto grad_w     = torch::zeros_like(w);
    auto grad_x_all = torch::zeros_like(x_all);
    auto grad_y     = torch::zeros_like(y);

    int tiles = U / 32;
    int warps = (tiles >= 8) ? 8 : (tiles >= 4) ? 4 : (tiles >= 2) ? 2 : 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (warps == 8) {
        dim3 block(32 * 8);
        dim3 grid(S, V, ceil_div_int(tiles, 8));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_bwd_w8", [&]{
            u1d_groupout_bwd_gather_scatterB_csr<scalar_t, 8><<<grid, block, 0, stream>>>(
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x_all.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                (const int32_t*)cls_offsets.data_ptr<int32_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, S, Iw, Ix, Ky, V, U
            );
        });
    } else if (warps == 4) {
        dim3 block(32 * 4);
        dim3 grid(S, V, ceil_div_int(tiles, 4));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_bwd_w4", [&]{
            u1d_groupout_bwd_gather_scatterB_csr<scalar_t, 4><<<grid, block, 0, stream>>>(
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x_all.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                (const int32_t*)cls_offsets.data_ptr<int32_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, S, Iw, Ix, Ky, V, U
            );
        });
    } else if (warps == 2) {
        dim3 block(32 * 2);
        dim3 grid(S, V, ceil_div_int(tiles, 2));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_bwd_w2", [&]{
            u1d_groupout_bwd_gather_scatterB_csr<scalar_t, 2><<<grid, block, 0, stream>>>(
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x_all.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                (const int32_t*)cls_offsets.data_ptr<int32_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, S, Iw, Ix, Ky, V, U
            );
        });
    } else {
        dim3 block(32);
        dim3 grid(S, V, tiles);
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_bwd_w1", [&]{
            u1d_groupout_bwd_gather_scatterB_csr<scalar_t, 1><<<grid, block, 0, stream>>>(
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x_all.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                (const int32_t*)cls_offsets.data_ptr<int32_t>(),
                (const int32_t*)i_list.data_ptr<int32_t>(),
                (const int32_t*)j_list.data_ptr<int32_t>(),
                (const int32_t*)k_list.data_ptr<int32_t>(),
                (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, S, Iw, Ix, Ky, V, U
            );
        });
    }

    return {grad_w.flatten(), grad_x_all.flatten(), grad_y.flatten()};
}

TORCH_LIBRARY(u1d_fused_bwd, m)
{
    m.def("backward", &u1d_fused_bwd_launch);
}
