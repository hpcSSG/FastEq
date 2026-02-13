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
__global__ void u1d_groupout_fwd_gather_scatter_csr(
    const scalar_t* __restrict__ w,       // [B, Iw, U]
    const scalar_t* __restrict__ x_all,   // [S, Ix, U]
    const scalar_t* __restrict__ y,       // [B, Ky, 1]  (last dim = 1)
    scalar_t* __restrict__ out,           // [S, V, U]   (NO B)
    const int32_t* __restrict__ src_idx,  // [B]  in [0..S-1]   (input_indices[1])
    const int32_t* __restrict__ b_list,   // [B]  b indices sorted by class
    const int32_t* __restrict__ cls_offsets, // [S+1] CSR offsets
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    const int32_t* __restrict__ v_offsets,   // [V+1]
    int B, int S, int Iw, int Ix, int Ky, int V, int U)
{
    int cls = (int)blockIdx.x;
    int v   = (int)blockIdx.y;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int u_tile = (int)blockIdx.z * WARPS_PER_BLOCK + warp;
    int u = (u_tile << 5) + lane;
    if (u >= U) return;

    int t0 = v_offsets[v];
    int t1 = v_offsets[v + 1];

    int bi0 = cls_offsets[cls];
    int bi1 = cls_offsets[cls + 1];

    scalar_t acc = (scalar_t)0;

    // reduce over b in this class bucket
    for (int bi = bi0; bi < bi1; ++bi) {
        int b = b_list[bi];
        int src = src_idx[b];

        for (int t = t0; t < t1; ++t) {
            int i = i_list[t];
            int j = j_list[t];
            int k = k_list[t];
            scalar_t c = coeff_list[t];

            // y[b,k,0] broadcast within warp
            scalar_t yval;
            if (lane == 0) {
                yval = y[(b * Ky + k)];
            }
            yval = __shfl_sync(0xffffffff, yval, 0);

            scalar_t wval = w[((b   * Iw + i)   * U) + u];
            scalar_t xval = x_all[((src * Ix + j) * U) + u];

            acc += c * wval * xval * yval;
        }
    }

    // write no atomic
    out[((cls * V + v) * U) + u] = acc;
}


torch::Tensor u1d_fused_fwd_launch(
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // [B,Ky,1]
    torch::Tensor src_idx,    // [B] int32
    torch::Tensor b_list,     // [B] int32
    torch::Tensor cls_offsets,// [S+1] int32
    torch::Tensor i_list,     // [P] int32
    torch::Tensor j_list,     // [P] int32
    torch::Tensor k_list,     // [P] int32
    torch::Tensor coeff_list, // [P] same dtype as w
    torch::Tensor v_offsets,  // [V+1] int32
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda() && b_list.is_cuda() && cls_offsets.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "w/x_all/y must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
    TORCH_CHECK(cls_offsets.scalar_type() == torch::kInt32, "cls_offsets must be int32");
    TORCH_CHECK(i_list.scalar_type() == torch::kInt32, "i_list must be int32");
    TORCH_CHECK(j_list.scalar_type() == torch::kInt32, "j_list must be int32");
    TORCH_CHECK(k_list.scalar_type() == torch::kInt32, "k_list must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");
    TORCH_CHECK(coeff_list.scalar_type() == w.scalar_type(), "coeff_list dtype must match w");

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
    TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
    TORCH_CHECK((int)cls_offsets.numel() == S + 1, "cls_offsets must be [S+1]");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto out = torch::zeros({S, V, U}, w.options());

    int tiles = U / 32;
    int warps = (tiles >= 8) ? 8 : (tiles >= 4) ? 4 : (tiles >= 2) ? 2 : 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (warps == 8) {
        dim3 block(32 * 8);
        dim3 grid(S, V, ceil_div_int(tiles, 8));
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_w8", [&]{
            u1d_groupout_fwd_gather_scatter_csr<scalar_t, 8><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_w4", [&]{
            u1d_groupout_fwd_gather_scatter_csr<scalar_t, 4><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_w2", [&]{
            u1d_groupout_fwd_gather_scatter_csr<scalar_t, 2><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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
        AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_w1", [&]{
            u1d_groupout_fwd_gather_scatter_csr<scalar_t, 1><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
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

    return out; // [S, V, U]
}


TORCH_LIBRARY(u1d_fused_fwd, m)
{
    m.def("forward", &u1d_fused_fwd_launch);
}
