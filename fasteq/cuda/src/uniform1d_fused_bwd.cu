#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include "cuda_utils.hpp"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


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


// ------------------------------------------------------------
// grad_w kernel
//   grid.x = B
//   block.x = 32
//   each block handles one b, one lane handles one u
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void stp_edge_parallel_grad_w_kernel(
    const scalar_t* __restrict__ grad_out,   // [S_cls, V, 32]
    const scalar_t* __restrict__ x_all,      // [S, Ix, 32]
    const scalar_t* __restrict__ y,          // [B, Ky]
    scalar_t* __restrict__ grad_w,           // [B, Iw, 32]
    const int32_t* __restrict__ src_idx,     // [B]
    const int32_t* __restrict__ dst_idx,     // [B]
    const int32_t* __restrict__ b_list,      // [B] or nullptr
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const int32_t* __restrict__ v_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    int B, int Iw, int Ix, int Ky, int V, int P)
{
    int lane = threadIdx.x;   // u in [0, 31]
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t x_base  = ((int64_t)src * Ix) * 32;
    int64_t y_base  = (int64_t)b * Ky;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gw_base = ((int64_t)b   * Iw) * 32 + lane;

    for (int t = 0; t < P; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        scalar_t c = coeff_list[t];

        scalar_t xval = x_all[x_base + (int64_t)j * 32 + lane];
        scalar_t yval = y[y_base + k];
        scalar_t goval = grad_out[go_base + ((int64_t)v << 5)];

        scalar_t contrib = c * xval * yval * goval;

        atomicAdd(&grad_w[gw_base + (int64_t)i * 32], contrib);
    }
}

// ------------------------------------------------------------
// grad_x kernel
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void stp_edge_parallel_grad_x_kernel(
    const scalar_t* __restrict__ grad_out,   // [S_cls, V, 32]
    const scalar_t* __restrict__ w,          // [B, Iw, 32]
    const scalar_t* __restrict__ y,          // [B, Ky]
    scalar_t* __restrict__ grad_x,           // [S, Ix, 32]
    const int32_t* __restrict__ src_idx,     // [B]
    const int32_t* __restrict__ dst_idx,     // [B]
    const int32_t* __restrict__ b_list,      // [B] or nullptr
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const int32_t* __restrict__ v_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    int B, int Iw, int Ix, int Ky, int V, int P)
{
    int lane = threadIdx.x;   // u in [0, 31]
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base  = ((int64_t)b   * Iw) * 32;
    int64_t y_base  = (int64_t)b * Ky;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gx_base = ((int64_t)src * Ix) * 32 + lane;

    for (int t = 0; t < P; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        scalar_t c = coeff_list[t];

        scalar_t wval = w[w_base + (int64_t)i * 32 + lane];
        scalar_t yval = y[y_base + k];
        scalar_t goval = grad_out[go_base + ((int64_t)v << 5)];

        scalar_t contrib = c * wval * yval * goval;

        atomicAdd(&grad_x[gx_base + (int64_t)j * 32], contrib);
    }
}

// ------------------------------------------------------------
// grad_y kernel
//   for each path, each lane computes one u contribution
//   then warp-reduce over 32 lanes
//   lane 0 atomicAdd into grad_y[b, k]
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void stp_edge_parallel_grad_y_kernel(
    const scalar_t* __restrict__ grad_out,   // [S_cls, V, 32]
    const scalar_t* __restrict__ w,          // [B, Iw, 32]
    const scalar_t* __restrict__ x_all,      // [S, Ix, 32]
    scalar_t* __restrict__ grad_y,           // [B, Ky]
    const int32_t* __restrict__ src_idx,     // [B]
    const int32_t* __restrict__ dst_idx,     // [B]
    const int32_t* __restrict__ b_list,      // [B] or nullptr
    const int32_t* __restrict__ i_list,      // [P]
    const int32_t* __restrict__ j_list,      // [P]
    const int32_t* __restrict__ k_list,      // [P]
    const int32_t* __restrict__ v_list,      // [P]
    const scalar_t* __restrict__ coeff_list, // [P]
    int B, int Iw, int Ix, int Ky, int V, int P)
{
    int lane = threadIdx.x;   // u in [0, 31]
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    int b   = b_list ? b_list[bidx] : bidx;
    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base  = ((int64_t)b   * Iw) * 32;
    int64_t x_base  = ((int64_t)src * Ix) * 32;
    int64_t go_base = ((int64_t)dst * V) * 32 + lane;
    int64_t gy_base = (int64_t)b * Ky;

    for (int t = 0; t < P; ++t) {
        int i = i_list[t];
        int j = j_list[t];
        int k = k_list[t];
        int v = v_list[t];
        scalar_t c = coeff_list[t];

        scalar_t wval = w[w_base + (int64_t)i * 32 + lane];
        scalar_t xval = x_all[x_base + (int64_t)j * 32 + lane];
        scalar_t goval = grad_out[go_base + ((int64_t)v << 5)];

        scalar_t local = c * wval * xval * goval;
        scalar_t sum = warp_sum(local);

        if (lane == 0) {
            atomicAdd(&grad_y[gy_base + k], sum);
        }
    }
}


std::vector<torch::Tensor> stp_edge_parallel_backward_launch(
    torch::Tensor grad_out,    // [S_cls, V, 32]
    torch::Tensor w,           // [B, Iw, 32]
    torch::Tensor x_all,       // [S, Ix, 32]
    torch::Tensor y,           // [B, Ky]
    torch::Tensor src_idx,     // [B] int32
    torch::Tensor dst_idx,     // [B] int32
    torch::Tensor b_list,      // [B] int32, can be empty tensor
    torch::Tensor i_list,      // [P] int32
    torch::Tensor j_list,      // [P] int32
    torch::Tensor k_list,      // [P] int32
    torch::Tensor v_list,      // [P] int32
    torch::Tensor coeff_list   // [P] float/double
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(w);
    CHECK_INPUT(x_all);
    CHECK_INPUT(y);
    CHECK_INPUT(src_idx);
    CHECK_INPUT(dst_idx);
    CHECK_INPUT(i_list);
    CHECK_INPUT(j_list);
    CHECK_INPUT(k_list);
    CHECK_INPUT(v_list);
    CHECK_INPUT(coeff_list);
    if (b_list.defined() && b_list.numel() > 0) CHECK_INPUT(b_list);

    TORCH_CHECK(w.dim() == 3, "w must be [B, Iw, 32]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S, Ix, 32]");
    TORCH_CHECK(y.dim() == 3, "y must be [B, Ky, 1]");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be [S_cls, V, 32]");

    const int64_t B  = w.size(0);
    const int64_t Iw = w.size(1);
    const int64_t U  = w.size(2);
    const int64_t S  = x_all.size(0);
    const int64_t Ix = x_all.size(1);
    const int64_t Ux = x_all.size(2);
    const int64_t By = y.size(0);
    const int64_t Ky = y.size(1);
    const int64_t V  = grad_out.size(1);
    const int64_t Ug = grad_out.size(2);
    const int64_t P  = i_list.numel();

    TORCH_CHECK(U == 32, "Only U=32 is supported");
    TORCH_CHECK(Ux == 32, "x_all.size(2) must be 32");
    TORCH_CHECK(Ug == 32, "grad_out.size(2) must be 32");
    TORCH_CHECK(By == B, "y.size(0) must equal B");

    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
    TORCH_CHECK(i_list.scalar_type() == torch::kInt32, "i_list must be int32");
    TORCH_CHECK(j_list.scalar_type() == torch::kInt32, "j_list must be int32");
    TORCH_CHECK(k_list.scalar_type() == torch::kInt32, "k_list must be int32");
    TORCH_CHECK(v_list.scalar_type() == torch::kInt32, "v_list must be int32");
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK(b_list.numel() == B, "b_list.numel() must equal B");
    }

    TORCH_CHECK(w.scalar_type() == grad_out.scalar_type(), "w and grad_out dtype mismatch");
    TORCH_CHECK(x_all.scalar_type() == grad_out.scalar_type(), "x_all and grad_out dtype mismatch");
    TORCH_CHECK(y.scalar_type() == grad_out.scalar_type(), "y and grad_out dtype mismatch");
    TORCH_CHECK(coeff_list.scalar_type() == grad_out.scalar_type(), "coeff_list and grad_out dtype mismatch");

    TORCH_CHECK(
        grad_out.scalar_type() == torch::kFloat || grad_out.scalar_type() == torch::kDouble,
        "Only float32/float64 are supported");

    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x_all);
    auto grad_y = torch::zeros_like(y);

    const int threads = 32;
    const dim3 block(threads);
    const dim3 grid((unsigned int)B);

    auto stream = at::cuda::getDefaultCUDAStream();

    const int32_t* b_list_ptr =
        (b_list.defined() && b_list.numel() > 0) ? b_list.data_ptr<int32_t>() : nullptr;

    if (grad_out.scalar_type() == torch::kFloat) {
        using scalar_t = float;

        stp_edge_parallel_grad_w_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            i_list.data_ptr<int32_t>(),
            j_list.data_ptr<int32_t>(),
            k_list.data_ptr<int32_t>(),
            v_list.data_ptr<int32_t>(),
            coeff_list.data_ptr<scalar_t>(),
            (int)B, (int)Iw, (int)Ix, (int)Ky, (int)V, (int)P
        );

        stp_edge_parallel_grad_x_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            i_list.data_ptr<int32_t>(),
            j_list.data_ptr<int32_t>(),
            k_list.data_ptr<int32_t>(),
            v_list.data_ptr<int32_t>(),
            coeff_list.data_ptr<scalar_t>(),
            (int)B, (int)Iw, (int)Ix, (int)Ky, (int)V, (int)P
        );

        stp_edge_parallel_grad_y_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            i_list.data_ptr<int32_t>(),
            j_list.data_ptr<int32_t>(),
            k_list.data_ptr<int32_t>(),
            v_list.data_ptr<int32_t>(),
            coeff_list.data_ptr<scalar_t>(),
            (int)B, (int)Iw, (int)Ix, (int)Ky, (int)V, (int)P
        );
    } else {
        using scalar_t = double;

        stp_edge_parallel_grad_w_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            i_list.data_ptr<int32_t>(),
            j_list.data_ptr<int32_t>(),
            k_list.data_ptr<int32_t>(),
            v_list.data_ptr<int32_t>(),
            coeff_list.data_ptr<scalar_t>(),
            (int)B, (int)Iw, (int)Ix, (int)Ky, (int)V, (int)P
        );

        stp_edge_parallel_grad_x_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            i_list.data_ptr<int32_t>(),
            j_list.data_ptr<int32_t>(),
            k_list.data_ptr<int32_t>(),
            v_list.data_ptr<int32_t>(),
            coeff_list.data_ptr<scalar_t>(),
            (int)B, (int)Iw, (int)Ix, (int)Ky, (int)V, (int)P
        );

        stp_edge_parallel_grad_y_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list_ptr,
            i_list.data_ptr<int32_t>(),
            j_list.data_ptr<int32_t>(),
            k_list.data_ptr<int32_t>(),
            v_list.data_ptr<int32_t>(),
            coeff_list.data_ptr<scalar_t>(),
            (int)B, (int)Iw, (int)Ix, (int)Ky, (int)V, (int)P
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_w, grad_x, grad_y};
}


TORCH_LIBRARY(u1d_fused_bwd, m)
{
    m.def("backward", &u1d_fused_bwd_launch);
    m.def("backward_ep", &stp_edge_parallel_backward_launch);
}
