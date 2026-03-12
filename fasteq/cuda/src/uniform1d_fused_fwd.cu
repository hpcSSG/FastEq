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
__global__ void u1d_gather_scatter_csr_node_parallel(
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
            
            int64_t w_off = ((int64_t)b * Iw + i) * (int64_t)U + u;
            int64_t x_off = ((int64_t)src * Ix + j) * (int64_t)U + u;
            int64_t y_off = (int64_t)b * Ky + k;
            int64_t o_off = ((int64_t)cls * V + v) * (int64_t)U + u;

            scalar_t wval = w[w_off];
            scalar_t xval = x_all[x_off];
            if (lane == 0) yval = y[y_off];
            yval = __shfl_sync(0xffffffff, yval, 0);

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
            u1d_gather_scatter_csr_node_parallel<scalar_t, 8><<<grid, block, 0, stream>>>(
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
            u1d_gather_scatter_csr_node_parallel<scalar_t, 4><<<grid, block, 0, stream>>>(
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
            u1d_gather_scatter_csr_node_parallel<scalar_t, 2><<<grid, block, 0, stream>>>(
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
            u1d_gather_scatter_csr_node_parallel<scalar_t, 1><<<grid, block, 0, stream>>>(
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


template <typename scalar_t>
struct __align__(32) Path32;

template <>
struct __align__(32) Path32<float> {
    int32_t i, j, k, pad0;       // 16B
    float   c;                   // 4B
    float   pad1, pad2, pad3;    // 12B -> 32B
};

template <>
struct __align__(32) Path32<double> {
    int32_t i, j, k, pad0;       // 16B
    double  c;                   // 8B
    double  pad1;                // 8B -> 32B
};

template <typename scalar_t>
__global__ void u1d_gather_scatter_csr_packed_node_parallel(
    const scalar_t* __restrict__ w,       // [B, Iw, U]
    const scalar_t* __restrict__ x_all,   // [S, Ix, U]
    const scalar_t* __restrict__ y,       // [B, Ky, 1] (treat as [B,Ky])
    scalar_t* __restrict__ out,           // [S, V, U]
    const int32_t* __restrict__ src_idx,  // [B]
    const int32_t* __restrict__ b_list,   // [B]
    const int32_t* __restrict__ cls_offsets, // [S+1]
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    const Path32<scalar_t>* __restrict__ paths, // [P] packed (i,j,k,c)
    const int32_t* __restrict__ v_offsets,      // [V+1]
    int B, int S, int Iw, int Ix, int Ky, int V, int U)
{
    int cls = (int)blockIdx.x;
    int v   = (int)blockIdx.y;

    int tid  = (int)threadIdx.x;
    int u = tid;

    int t0  = v_offsets[v];
    int t1  = v_offsets[v + 1];

    int bi0 = cls_offsets[cls];
    int bi1 = cls_offsets[cls + 1];

    scalar_t acc = (scalar_t)0;

    for (int bi = bi0; bi < bi1; ++bi) {
        int b   = b_list[bi];
        int src = src_idx[b];

        for (int t = t0; t < t1; ++t) {
            Path32<scalar_t> p = paths[t];
            int i = p.i;
            int j = p.j;
            int k = p.k;
            scalar_t c = (scalar_t)p.c;

            scalar_t yval;
            int64_t y_off = (int64_t)b * Ky + k;
            if (tid == 0) yval = y[y_off];
            yval = __shfl_sync(0xffffffff, yval, 0);

            int64_t w_off = ((int64_t)b   * Iw + i) * (int64_t)U + u;
            int64_t x_off = ((int64_t)src * Ix + j) * (int64_t)U + u;

            scalar_t wval = w[w_off];
            scalar_t xval = x_all[x_off];

            acc += c * wval * xval * yval;
        }
    }

    out[((cls * V + v) * U) + u] = acc;
}

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void u1d_gather_scatter_csr_packed_node_parallel_warpbi(
    const scalar_t* __restrict__ w,       // [B, Iw, U]
    const scalar_t* __restrict__ x_all,   // [S, Ix, U]
    const scalar_t* __restrict__ y,       // [B, Ky, 1]
    scalar_t* __restrict__ out,           // [S, V, U]
    const int32_t* __restrict__ src_idx,  // [B]
    const int32_t* __restrict__ b_list,   // [B]
    const int32_t* __restrict__ cls_offsets, // [S+1]
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    const Path32<scalar_t>* __restrict__ paths, // [P] packed (i,j,k,c)
    const int32_t* __restrict__ v_offsets,      // [V+1]
    int B, int S, int Iw, int Ix, int Ky, int V, int U)
{

    const int cls = (int)blockIdx.x;
    const int v   = (int)blockIdx.y;

    const int tid   = (int)threadIdx.x;
    const int lane  = tid & 31;           // 0..31
    const int warp  = tid >> 5;           // 0..WARPS_PER_BLOCK-1
    const int u = lane;

    const int t0 = v_offsets[v];
    const int t1 = v_offsets[v + 1];

    const int bi0 = cls_offsets[cls];
    const int bi1 = cls_offsets[cls + 1];

    scalar_t acc = (scalar_t)0;

    for (int bi = bi0; bi < bi1; bi += WARPS_PER_BLOCK) {
        const int b   = b_list[bi];
        const int src = src_idx[b];

        const int64_t b_base_w = (int64_t)b   * (int64_t)Iw * (int64_t)U;   // b*Iw*U
        const int64_t s_base_x = (int64_t)src * (int64_t)Ix * (int64_t)U;   // src*Ix*U
        const int64_t b_base_y = (int64_t)b   * (int64_t)Ky;               // b*Ky

        for (int t = t0; t < t1; ++t) {
            const Path32<scalar_t> p = paths[t];
            const int i = (int)p.i;
            const int j = (int)p.j;
            const int k = (int)p.k;
            const scalar_t c = (scalar_t)p.c;
            

            const int64_t w_off = b_base_w + (int64_t)i * (int64_t)U + u;
            const int64_t x_off = s_base_x + (int64_t)j * (int64_t)U + u;
            const int64_t y_off = b_base_y + (int64_t)k;

            scalar_t wval = __ldg(&w[w_off]);
            scalar_t xval = __ldg(&x_all[x_off]);
            scalar_t yval = __ldg(&y[y_off]);

            acc += c * wval * xval * yval;
        }
    }

    out[((cls * V + v) * U) + u] = acc;
}


torch::Tensor u1d_fused_fwd_np_launch(
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
    torch::Tensor packed_paths,
    torch::Tensor v_offsets,  // [V+1] int32
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda() && b_list.is_cuda() && cls_offsets.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "w/x_all/y must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
    TORCH_CHECK(cls_offsets.scalar_type() == torch::kInt32, "cls_offsets must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");

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

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_w1", [&]{
        const auto* paths = reinterpret_cast<const Path32<scalar_t>*>(
            packed_paths.data_ptr<uint8_t>());
        /*
        dim3 block(32, 1, 1);
        dim3 grid(S, V, 1);
        
        u1d_gather_scatter_csr_packed_node_parallel<scalar_t><<<grid, block, 0, stream>>>(
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
            paths,
            (const int32_t*)v_offsets.data_ptr<int32_t>(),
            B, S, Iw, Ix, Ky, V, U
        );
        */

        constexpr int WARPS_PER_BLOCK = 1;
        dim3 block(32 * WARPS_PER_BLOCK, 1, 1);
        dim3 grid(S, V, 1);

        u1d_gather_scatter_csr_packed_node_parallel_warpbi<scalar_t, WARPS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(
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
                paths,
                (const int32_t*)v_offsets.data_ptr<int32_t>(),
                B, S, Iw, Ix, Ky, V, U
            );

    });

    return out; // [S, V, U]
}

template <typename scalar_t>
__global__ void u1d_edge_parallel_atomic(
    const scalar_t* __restrict__ w,       // [B, Iw, U]
    const scalar_t* __restrict__ x_all,   // [S, Ix, U]
    const scalar_t* __restrict__ y,       // [B, Ky]
    scalar_t* __restrict__ out,           // [S_cls, V, U]
    const int32_t* __restrict__ src_idx,  // [B]
    const int32_t* __restrict__ cls_idx,  // [B]
    const Path32<scalar_t>* __restrict__ paths, // [P] packed (i,j,k,c)
    const int32_t* __restrict__ v_offsets,// [V+1]
    int B, int S, int Iw, int Ix, int Ky, int V, int U)
{
    int b = (int)blockIdx.x;     // 0..B-1
    int u = (int)threadIdx.x;    // 0..63

    if (b >= B || u >= U) return;

    int cls = (int)cls_idx[b];
    int src = (int)src_idx[b];

    for (int v = 0; v < V; ++v) {
        int t0 = v_offsets[v];
        int t1 = v_offsets[v + 1];

        scalar_t acc = (scalar_t)0;

        // 对该 v 的所有路径项 t
        for (int t = t0; t < t1; ++t) {
            
            //int i = i_list[t];
            //int j = j_list[t];
            //int k = k_list[t];
            //scalar_t c = coeff_list[t];

            Path32<scalar_t> p = paths[t];
            int i = p.i;
            int j = p.j;
            int k = p.k;
            scalar_t c = (scalar_t)p.c;

            int64_t w_off = ((int64_t)b   * Iw + i) * (int64_t)U + u;
            int64_t x_off = ((int64_t)src * Ix + j) * (int64_t)U + u;
            int64_t y_off = (int64_t)b * Ky + k;

            scalar_t wval = __ldg(&w[w_off]);
            scalar_t xval = __ldg(&x_all[x_off]);
            scalar_t yval = __ldg(&y[y_off]);

            acc += c * wval * xval * yval;
        }

        int64_t o_off = ((int64_t)cls * V + v) * (int64_t)U + u;
        atomicAdd(&out[o_off], acc);
    }
}

template <typename scalar_t, int E_TILE>
__global__ void u1d_cls_edge_tile_atomic_u32(
    const scalar_t* __restrict__ w,       // [B, Iw, U]
    const scalar_t* __restrict__ x_all,   // [S, Ix, U]
    const scalar_t* __restrict__ y,       // [B, Ky]
    scalar_t* __restrict__ out,           // [S_cls, V, U]
    const int32_t* __restrict__ src_idx,  // [B]
    const int32_t* __restrict__ b_list,   // edges grouped by cls
    const int32_t* __restrict__ cls_offsets, // [S_cls+1]
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    const Path32<scalar_t>* __restrict__ paths,
    const int32_t* __restrict__ v_offsets,// [V+1]
    int B, int S, int Iw, int Ix, int Ky, int V, int U)
{
    int cls   = (int)blockIdx.x;
    int etile = (int)blockIdx.y;
    int u     = (int)threadIdx.x; // 0..31

    int bi0 = cls_offsets[cls];
    int bi1 = cls_offsets[cls + 1];

    int base = bi0 + etile * E_TILE;
    if (base >= bi1) return;

    // 对每个 v：先合并 E_TILE 条边，再 atomic 一次
    for (int v = 0; v < V; ++v) {
        int t0 = v_offsets[v];
        int t1 = v_offsets[v + 1];
        int vcnt = t1 - t0;

        scalar_t acc = (scalar_t)0;

        #pragma unroll
        for (int e = 0; e < E_TILE; ++e) {
            int bi = base + e;
            if (bi >= bi1) break;

            int b   = b_list[bi];
            int src = src_idx[b];

            // paths loop
            for (int t = t0; t < t1; ++t) {
                /* Path32<scalar_t> p = paths[t];
                int i = p.i, j = p.j, k = p.k;
                scalar_t c = (scalar_t)p.c; */

                int i = i_list[t];
                int j = j_list[t];
                int k = k_list[t];
                scalar_t c = coeff_list[t];

                int64_t w_off = ((int64_t)b   * Iw + i) * 32 + u;
                int64_t x_off = ((int64_t)src * Ix + j) * 32 + u;
                int64_t y_off = (int64_t)b * Ky + k;

                scalar_t wval = __ldg(&w[w_off]);
                scalar_t xval = __ldg(&x_all[x_off]);
                scalar_t yval = __ldg(&y[y_off]);

                acc += c * wval * xval * yval;
            }
        }

        int64_t o_off = ((int64_t)cls * V + v) * 32 + u;
        atomicAdd(&out[o_off], acc);
    }
}

template <typename scalar_t, int E_TILE, int WARPS_PER_BLOCK>
__global__ void stp_path_first_kernel(
    const scalar_t* __restrict__ w,        // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,    // [S, Ix, 32]
    const scalar_t* __restrict__ y,        // [B, Ky]
    scalar_t* __restrict__ out,            // [S_cls, V, 32]
    const int32_t* __restrict__ src_idx,   // [B]
    const int32_t* __restrict__ b_list,
    const int32_t* __restrict__ cls_offsets,
    const int32_t* __restrict__ i_list,
    const int32_t* __restrict__ j_list,
    const int32_t* __restrict__ k_list,
    const int32_t* __restrict__ v_list,
    const scalar_t* __restrict__ coeff_list,
    int B, int S, int Iw, int Ix, int Ky, int V, int U, int P)
{
    int cls   = blockIdx.x;
    int etile = blockIdx.y;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; 

    int bi0 = cls_offsets[cls];
    int bi1 = cls_offsets[cls + 1];

    int base = bi0 + etile * E_TILE;
    if (base >= bi1) return;

    // warp-stride over edges inside tile
    for (int e0 = warp; e0 < E_TILE; e0 += WARPS_PER_BLOCK) {
        int bi = base + e0;
        if (bi >= bi1) break;

        int b   = b_list[bi];
        int src = src_idx[b];

        int64_t w_base = ((int64_t)b   * Iw) * 32;
        int64_t x_base = ((int64_t)src * Ix) * 32;

        // ----------------------------
        // v-change triggered flush
        // ----------------------------
        int v_prev = v_list[0];
        int64_t o_base = ((int64_t)cls * V) * 32 + lane;
        int64_t o_off  = o_base + ((int64_t)v_prev << 5);
        scalar_t sum = scalar_t(0);

        // path loop
        #pragma unroll 2
        for (int t = 0; t < P; t++) {

            int i = i_list[t];
            int j = j_list[t];
            int k = k_list[t];
            int v = v_list[t];
            scalar_t c = coeff_list[t];
            
            int64_t w_off = ((int64_t)b   * Iw + i) * 32 + lane;
            int64_t x_off = ((int64_t)src * Ix + j) * 32 + lane;
            int64_t y_off = (int64_t)b * Ky + k;

            scalar_t wval = w[w_off];
            scalar_t xval = __ldg(&x_all[x_off]);
            scalar_t yval = y[y_off];
            
            scalar_t acc = c * wval * xval * yval;

            if (v != v_prev) {
                atomicAdd(&out[o_off], sum);
                sum = scalar_t(0);
                v_prev = v;
                o_off = o_base + ((int64_t)v_prev << 5);
            }
            sum += acc;
        }

        atomicAdd(&out[o_off], sum);
    }
}

template <typename scalar_t, int E_TILE, int WARPS_PER_BLOCK>
__global__ void stp_path_first_kernel_shm(
    const scalar_t* __restrict__ w,        // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,    // [S, Ix, 32]
    const scalar_t* __restrict__ y,        // [B, Ky]
    scalar_t* __restrict__ out,            // [S_cls, V, 32]
    const int32_t* __restrict__ src_idx,   // [B]
    const int32_t* __restrict__ b_list,
    const int32_t* __restrict__ cls_offsets,
    const int32_t* __restrict__ i_list,
    const int32_t* __restrict__ j_list,
    const int32_t* __restrict__ k_list,
    const int32_t* __restrict__ v_list,
    const scalar_t* __restrict__ coeff_list,
    int B, int S, int Iw, int Ix, int Ky, int V, int U, int P)
{
    int cls   = blockIdx.x;
    int etile = blockIdx.y;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; 

    int bi0 = cls_offsets[cls];
    int bi1 = cls_offsets[cls + 1];

    int base = bi0 + etile * E_TILE;
    if (base >= bi1) return;

    // ----------------------------
    // shared: paths + y-cache
    // ----------------------------
    extern __shared__ unsigned char smem[];
    int32_t* s_i = (int32_t*)smem;
    int32_t* s_j = s_i + P;
    int32_t* s_k = s_j + P;
    int32_t* s_v = s_k + P;
    scalar_t* s_c = (scalar_t*)(s_v + P);

    scalar_t* s_y = (scalar_t*)(s_c + P);

    // cooperative load paths
    for (int t = tid; t < P; t += blockDim.x) {
        s_i[t] = i_list[t];
        s_j[t] = j_list[t];
        s_k[t] = k_list[t];
        s_v[t] = v_list[t];
        s_c[t] = coeff_list[t];
    }
    __syncthreads();

    // warp-stride over edges inside tile
    for (int e0 = warp; e0 < E_TILE; e0 += WARPS_PER_BLOCK) {
        int bi = base + e0;
        if (bi >= bi1) break;

        int b   = b_list[bi];
        int src = src_idx[b];

        int64_t w_base = ((int64_t)b   * Iw) * 32;
        int64_t x_base = ((int64_t)src * Ix) * 32;

        // lane0 load y[b, :]
        if (lane == 0) {
            #pragma unroll
            for (int kk = 0; kk < 16; ++kk) { //Ky 写死 16
                s_y[warp * Ky + kk] = y[(int64_t)b * Ky + kk];
            }
        }
        __syncwarp();

        // ----------------------------
        // v-change triggered flush
        // ----------------------------
        int v_prev = (int)s_v[0];
        int64_t o_base = ((int64_t)cls * V) * 32 + lane;
        int64_t o_off  = o_base + ((int64_t)v_prev << 5);
        scalar_t sum = scalar_t(0);

        // path loop
        #pragma unroll 8
        for (int t = 0; t < P; t++) {
            int i = s_i[t];
            int j = s_j[t];
            int k = s_k[t];
            int v = s_v[t];
            scalar_t c = s_c[t];
            

            scalar_t wval = __ldg(&w[w_base + (int64_t)i * 32 + lane]);
            scalar_t xval = __ldg(&x_all[x_base + (int64_t)j * 32 + lane]);
            scalar_t yval = s_y[warp * Ky + k];

            scalar_t acc = c * wval * xval * yval;

            if (v != v_prev) {
                atomicAdd(&out[o_off], sum);
                sum = scalar_t(0);
                v_prev = v;
                o_off = o_base + ((int64_t)v_prev << 5);
            }
            sum += acc;
        }

        atomicAdd(&out[o_off], sum);
    }
}


template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void stp_edge_parallel_kernel(
    const scalar_t* __restrict__ w,        // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,    // [S, Ix, 32]
    const scalar_t* __restrict__ y,        // [B, Ky]
    scalar_t* __restrict__ out,            // [S_cls, V, 32]
    const int32_t* __restrict__ src_idx,   // [B]
    const int32_t* __restrict__ dst_idx,   // [B]
    const int32_t* __restrict__ b_list,
    const int32_t* __restrict__ cls_offsets,
    const int32_t* __restrict__ i_list,
    const int32_t* __restrict__ j_list,
    const int32_t* __restrict__ k_list,
    const int32_t* __restrict__ v_list,
    const scalar_t* __restrict__ coeff_list,
    int B, int S, int Iw, int Ix, int Ky, int V, int U, int P)
    

{
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int warp_global = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (warp_global >= B) return;

    int b = b_list ? b_list[warp_global] : warp_global;
    int src = src_idx[b];
    
    int64_t w_base = ((int64_t)b   * Iw) * 32;
    int64_t x_base = ((int64_t)src * Ix) * 32;
    int64_t k_base = (int64_t)b * Ky;

    int dst = dst_idx[b];
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    
    for (int t = 0; t < P; t++) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        scalar_t c = coeff_list[t];

        scalar_t wval = w[w_base + (int64_t)i * 32 + lane];
        scalar_t xval = x_all[x_base + (int64_t)j * 32 + lane];
        scalar_t yval = y[k_base + k];

        scalar_t sum_acc = c * wval * xval * yval;

        int64_t o_off = o_base + ((int64_t)v << 5);
        atomicAdd(&out[o_off], sum_acc);
    }

}

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void stp_edge_vec4_float(
    const scalar_t* __restrict__ w,        // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,    // [S, Ix, 32]
    const scalar_t* __restrict__ y,        // [B, Ky]
    scalar_t* __restrict__ out,            // [S_cls, V, 32]
    const int32_t* __restrict__ src_idx,   // [B]
    const int32_t* __restrict__ dst_idx,   // [B]
    const int32_t* __restrict__ b_list,
    const int32_t* __restrict__ cls_offsets,
    const int32_t* __restrict__ i_list,
    const int32_t* __restrict__ j_list,
    const int32_t* __restrict__ k_list,
    const int32_t* __restrict__ v_list,
    const scalar_t* __restrict__ coeff_list,
    int B, int S, int Iw, int Ix, int Ky, int V, int U, int P)
{
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int warp_global = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (warp_global >= B) return;

    int b   = b_list ? b_list[warp_global] : warp_global;
    int src = src_idx[b];

    int64_t w_base = (int64_t)b   * Iw * 32;
    int64_t x_base = (int64_t)src * Ix * 32;
    int64_t y_base = (int64_t)b   * Ky;

    int dst = dst_idx[b];
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    // v-run flush init
    int v_prev = v_list[0];
    int64_t o_off = o_base + ((int64_t)v_prev << 5);
    float sum = 0.0f;

    int k_prev = -1;
    float y_prev = 0.0f;

    // vectorized size
    int P4 = P & ~3;

    // only for float32
    for (int t = 0; t < P4; t += 4) {
        int4 ii = *reinterpret_cast<const int4*>(&i_list[t]);
        int4 jj = *reinterpret_cast<const int4*>(&j_list[t]);
        int4 kk = *reinterpret_cast<const int4*>(&k_list[t]);
        int4 vv = *reinterpret_cast<const int4*>(&v_list[t]);
        float4 cc = *reinterpret_cast<const float4*>(&coeff_list[t]);

        // vectorized load + compute + v-change flush
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int i = (&ii.x)[r];
            int j = (&jj.x)[r];
            int k = (&kk.x)[r];
            int v = (&vv.x)[r];
            float c = (&cc.x)[r];

            float wval = w[w_base + (int64_t)i * 32 + lane];
            float xval = x_all[x_base + (int64_t)j * 32 + lane];

            float yval;
            if (k == k_prev) {
                yval = y_prev;
            } else {
                yval = y[y_base + k];
                k_prev = k;
                y_prev = yval;
            }

            float acc = c * wval * xval * yval;

            if (v != v_prev) {
                atomicAdd(&out[o_off], sum);
                sum = 0.0f;
                v_prev = v;
                o_off = o_base + ((int64_t)v_prev << 5);
            }
            sum += acc;
        }
    }

    for (int t = P4; t < P; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        float c = coeff_list[t];

        float wval = w[w_base + (int64_t)i * 32 + lane];
        float xval = x_all[x_base + (int64_t)j * 32 + lane];

        float yval;
        if (k == k_prev) yval = y_prev;
        else { yval = y[y_base + k]; k_prev = k; y_prev = yval; }

        float acc = c * wval * xval * yval;

        if (v != v_prev) {
            atomicAdd(&out[o_off], sum);
            sum = 0.0f;
            v_prev = v;
            o_off = o_base + ((int64_t)v_prev << 5);
        }
        sum += acc;
    }

    atomicAdd(&out[o_off], sum);
}


template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void stp_edge_splitP_multiwarp(
    const scalar_t* __restrict__ w,        // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,    // [S, Ix, 32]
    const scalar_t* __restrict__ y,        // [B, Ky]
    scalar_t* __restrict__ out,            // [S_cls, V, 32]
    const int32_t* __restrict__ src_idx,   // [B]
    const int32_t* __restrict__ dst_idx,   // [B]
    const int32_t* __restrict__ b_list,
    const int32_t* __restrict__ cls_offsets,
    const int32_t* __restrict__ i_list,
    const int32_t* __restrict__ j_list,
    const int32_t* __restrict__ k_list,
    const int32_t* __restrict__ v_list,
    const scalar_t* __restrict__ coeff_list,
    int B, int S, int Iw, int Ix, int Ky, int V, int U, int P)
{
    int b_global = (int)blockIdx.x;
    if (b_global >= B) return;
    int b = b_list ? b_list[b_global] : b_global;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= WARPS_PER_BLOCK) return;

    int src = src_idx[b];
    int64_t w_base = (int64_t)b   * Iw * 32;
    int64_t x_base = (int64_t)src * Ix * 32;
    int64_t y_base = (int64_t)b   * Ky;
    int dst = dst_idx[b];
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    int chunk = (P + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int t0 = warp * chunk;
    int t1 = min(P, t0 + chunk);
    if (t0 >= t1) return;

    int v_prev = v_list[t0];
    int64_t o_off = o_base + ((int64_t)v_prev << 5);
    scalar_t sum = scalar_t(0);

    for (int t = t0; t < t1; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        scalar_t c = coeff_list[t];

        scalar_t wval = w[w_base + (int64_t)i * 32 + lane];
        scalar_t xval = x_all[x_base + (int64_t)j * 32 + lane];
        scalar_t yval = y[y_base + k];

        scalar_t acc = c * wval * xval * yval;

        if (v != v_prev) {
            atomicAdd(&out[o_off], sum);
            sum = scalar_t(0);
            v_prev = v;
            o_off = o_base + ((int64_t)v_prev << 5);
        }
        sum += acc;
    }
    atomicAdd(&out[o_off], sum);
}

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void stp_edge_splitP_multiwarp_vec4_float(
    const scalar_t* __restrict__ w,        // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,    // [S, Ix, 32]
    const scalar_t* __restrict__ y,        // [B, Ky]
    scalar_t* __restrict__ out,            // [S_cls, V, 32]
    const int32_t* __restrict__ src_idx,   // [B]
    const int32_t* __restrict__ dst_idx,   // [B]
    const int32_t* __restrict__ b_list,
    const int32_t* __restrict__ cls_offsets,
    const int32_t* __restrict__ i_list,
    const int32_t* __restrict__ j_list,
    const int32_t* __restrict__ k_list,
    const int32_t* __restrict__ v_list,
    const scalar_t* __restrict__ coeff_list,
    int B, int S, int Iw, int Ix, int Ky, int V, int U, int P)
{
    int b_global = (int)blockIdx.x;
    if (b_global >= B) return;
    int b = b_list ? b_list[b_global] : b_global;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= WARPS_PER_BLOCK) return;

    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base = (int64_t)b   * Iw * 32;
    int64_t x_base = (int64_t)src * Ix * 32;
    int64_t y_base = (int64_t)b   * Ky;
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    // split P into contiguous chunks per warp
    int chunk = (P + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int t0 = warp * chunk;
    int t1 = min(P, t0 + chunk);
    if (t0 >= t1) return;

    int v_prev = v_list[t0];
    int64_t o_off = o_base + ((int64_t)v_prev << 5);
    float sum = 0.0f;

    int k_prev = -1;
    float y_prev = 0.0f;

    // ---- scalar prefix until t is 4-aligned ----
    int t = t0;
    for (; t < t1 && (t & 3); ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        float c = coeff_list[t];

        float wval = w[w_base + (int64_t)i * 32 + lane];
        float xval = x_all[x_base + (int64_t)j * 32 + lane];

        float yval;
        if (k == k_prev) yval = y_prev;
        else { yval = y[y_base + k]; k_prev = k; y_prev = yval; }

        float acc = c * wval * xval * yval;

        if (v != v_prev) {
            atomicAdd(&out[o_off], sum);
            sum = 0.0f;
            v_prev = v;
            o_off = o_base + ((int64_t)v_prev << 5);
        }
        sum += acc;
    }

    // ---- vector loop on [t, t1) with step 4 ----
    int t_vec_end = t1 & ~3;
    for (; t < t_vec_end; t += 4) {
        // vectorized loads (4 paths)
        int4 ii = *reinterpret_cast<const int4*>(&i_list[t]);
        int4 jj = *reinterpret_cast<const int4*>(&j_list[t]);
        int4 kk = *reinterpret_cast<const int4*>(&k_list[t]);
        int4 vv = *reinterpret_cast<const int4*>(&v_list[t]);
        float4 cc = *reinterpret_cast<const float4*>(&coeff_list[t]);

        #pragma unroll 4
        for (int r = 0; r < 4; ++r) {
            int i = (&ii.x)[r];
            int j = (&jj.x)[r];
            int k = (&kk.x)[r];
            int v = (&vv.x)[r];
            float c = (&cc.x)[r];

            float wval = w[w_base + (int64_t)i * 32 + lane];
            float xval = x_all[x_base + (int64_t)j * 32 + lane];

            float yval;
            if (k == k_prev) yval = y_prev;
            else { yval = y[y_base + k]; k_prev = k; y_prev = yval; }

            float acc = c * wval * xval * yval;

            if (v != v_prev) {
                atomicAdd(&out[o_off], sum);
                sum = 0.0f;
                v_prev = v;
                o_off = o_base + ((int64_t)v_prev << 5);
            }
            sum += acc;
        }
    }

    // ---- scalar tail ----
    for (; t < t1; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        float c = coeff_list[t];

        float wval = w[w_base + (int64_t)i * 32 + lane];
        float xval = x_all[x_base + (int64_t)j * 32 + lane];

        float yval;
        if (k == k_prev) yval = y_prev;
        else { yval = y[y_base + k]; k_prev = k; y_prev = yval; }

        float acc = c * wval * xval * yval;

        if (v != v_prev) {
            atomicAdd(&out[o_off], sum);
            sum = 0.0f;
            v_prev = v;
            o_off = o_base + ((int64_t)v_prev << 5);
        }
        sum += acc;
    }

    atomicAdd(&out[o_off], sum);
}

torch::Tensor u1d_fused_fwd_ep_launch(
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // [B,Ky,1]
    torch::Tensor src_idx,    // [B] int32
    torch::Tensor dst_idx,    // [B] int32
    torch::Tensor b_list,     // [B] int32
    torch::Tensor cls_offsets,// [S+1] int32
    torch::Tensor i_list,     // [P] int32
    torch::Tensor j_list,     // [P] int32
    torch::Tensor k_list,     // [P] int32
    torch::Tensor v_list,     // [P] int32
    torch::Tensor coeff_list, // [P] same dtype as w
    torch::Tensor packed_paths,
    torch::Tensor v_offsets,  // [V+1] int32
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(dst_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "w/x_all/y must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");
    TORCH_CHECK(coeff_list.scalar_type() == w.scalar_type(), "coeff_list dtype must match w");

    int B  = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);

    int Ky = (int)y.size(1);
    int V  = (int)V64;

    const int P = (int)i_list.size(0);

    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto out = torch::zeros({S, V, U}, w.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_atomic", [&]{
        
        
        /*

        const auto* paths = reinterpret_cast<const Path32<scalar_t>*>(
                packed_paths.data_ptr<uint8_t>());

        dim3 block(32);
        dim3 grid(B, 1, 1);
        u1d_edge_parallel_atomic<scalar_t><<<grid, block, 0, stream>>>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x_all.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (scalar_t*)out.data_ptr<scalar_t>(),
            (const int32_t*)src_idx.data_ptr<int32_t>(),
            (const int32_t*)dst_idx.data_ptr<int32_t>(),
            paths,
            (const int32_t*)v_offsets.data_ptr<int32_t>(),
            B, S, Iw, Ix, Ky, V, U
        );
        */
        
        /*
        const int E_TILE = 2;
        int max_deg = 116;
        int gridy  = (max_deg + E_TILE - 1) / E_TILE;
        dim3 grid(S, gridy, 1);
        dim3 block(32,1,1);

        u1d_cls_edge_tile_atomic_u32<scalar_t, E_TILE><<<grid, block, 0, stream>>>(
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
            paths,
            (const int32_t*)v_offsets.data_ptr<int32_t>(),
            B, S, Iw, Ix, Ky, V, U
        );
        */

        /*
        constexpr int E_TILE = 2;
        int max_deg = 116;
        int gridy  = (max_deg + E_TILE - 1) / E_TILE;
        constexpr int WARPS  = 2;
        dim3 grid(S, gridy, 1);
        dim3 block(32*WARPS,1);

        size_t smem_bytes =
            (size_t)(4 * P) * sizeof(int32_t) + // i,j,k,v 四个数组索引
            (size_t)P * sizeof(scalar_t) +
            (size_t)(WARPS * Ky) * sizeof(scalar_t);

        stp_path_first_kernel_shm<scalar_t, E_TILE, WARPS><<<grid, block, smem_bytes, stream>>>(
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
            (const int32_t*)v_list.data_ptr<int32_t>(),
            (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
            B, S, Iw, Ix, Ky, V, U, P
        );
        */

        /*
        constexpr int E_TILE = 64;
        int max_deg = 116;
        int gridy  = (max_deg + E_TILE - 1) / E_TILE;
        constexpr int WARPS  = 2;
        dim3 grid(S, gridy, 1);
        dim3 block(32*WARPS,1);

        stp_path_first_kernel<scalar_t, E_TILE, WARPS><<<grid, block, 0, stream>>>(
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
            (const int32_t*)v_list.data_ptr<int32_t>(),
            (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
            B, S, Iw, Ix, Ky, V, U, P
        );
        */

        /*
        constexpr int WARPS = 2;
        dim3 block(32 * WARPS, 1, 1);
        dim3 grid((B + WARPS - 1) / WARPS, 1, 1);

        stp_edge_vec4_float<scalar_t, WARPS><<<grid, block, 0, stream>>>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x_all.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (scalar_t*)out.data_ptr<scalar_t>(),
            (const int32_t*)src_idx.data_ptr<int32_t>(),
            (const int32_t*)dst_idx.data_ptr<int32_t>(),
            (const int32_t*)b_list.data_ptr<int32_t>(),
            (const int32_t*)cls_offsets.data_ptr<int32_t>(),
            (const int32_t*)i_list.data_ptr<int32_t>(),
            (const int32_t*)j_list.data_ptr<int32_t>(),
            (const int32_t*)k_list.data_ptr<int32_t>(),
            (const int32_t*)v_list.data_ptr<int32_t>(),
            (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
            B, S, Iw, Ix, Ky, V, U, P
        );
        */
        
        constexpr int WARPS = 8;
        dim3 block(32 * WARPS, 1, 1);
        dim3 grid(B, 1, 1);

        stp_edge_splitP_multiwarp<scalar_t, WARPS><<<grid, block, 0, stream>>>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x_all.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (scalar_t*)out.data_ptr<scalar_t>(),
            (const int32_t*)src_idx.data_ptr<int32_t>(),
            (const int32_t*)dst_idx.data_ptr<int32_t>(),
            (const int32_t*)b_list.data_ptr<int32_t>(),
            (const int32_t*)cls_offsets.data_ptr<int32_t>(),
            (const int32_t*)i_list.data_ptr<int32_t>(),
            (const int32_t*)j_list.data_ptr<int32_t>(),
            (const int32_t*)k_list.data_ptr<int32_t>(),
            (const int32_t*)v_list.data_ptr<int32_t>(),
            (const scalar_t*)coeff_list.data_ptr<scalar_t>(),
            B, S, Iw, Ix, Ky, V, U, P
        );

        
    });
    
    
    return out; // [S, V, U]
}

TORCH_LIBRARY(u1d_fused_fwd, m)
{
    m.def("forward", &u1d_fused_fwd_launch);
    m.def("forward_np", &u1d_fused_fwd_np_launch);
    m.def("forward_ep", &u1d_fused_fwd_ep_launch);
}
