#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

template<typename scalar_t, int MAX_D>
__device__ inline void flush_group_scalar_impl(
    int rcv, int d, int o,
    int u, int U, int DIM_SUM,
    scalar_t acc[MAX_D],
    scalar_t* __restrict__ out_nodes
){
    if (rcv < 0 || u >= U) return;
    const size_t out_base =
        ((size_t)rcv * (size_t)DIM_SUM + (size_t)o) * (size_t)U + (size_t)u;

    #pragma unroll
    for (int j = 0; j < MAX_D; ++j) {
        if (j >= d) break;
        atomicAdd(&out_nodes[out_base + (size_t)j * (size_t)U], acc[j]);
        acc[j] = scalar_t(0);
    }
}

// ============================================================================
// Backward kernel: 对 node_feats, edge_attrs, tp_weights 求导
// ============================================================================
template<int TileU, int MAX_D>
__global__ void fused_mp_warp_streaming_kernel_groupflush_scalar_backward(
    const double* __restrict__ node_feats,       // [N, U]
    const double* __restrict__ edge_attrs,       // [E, DIM_SUM]
    const double* __restrict__ tp_weights,       // [E, P, U]
    const int32_t* __restrict__ receiver,        // [E]
    const int32_t* __restrict__ start_idx,       // [N]
    const int32_t* __restrict__ end_idx,         // [N]
    const int32_t* __restrict__ dim_list,        // [P]
    const int32_t* __restrict__ offs,            // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,

    const double* __restrict__ grad_out_nodes,   // [N, DIM_SUM, U]
    double* __restrict__ grad_node_feats,        // [N, U]
    double* __restrict__ grad_edge_attrs,        // [E, DIM_SUM]
    double* __restrict__ grad_tp_weights         // [E, P, U]
){
    constexpr int W = WARP_SIZE;

    const int lane            = threadIdx.x & (W - 1);  // 0..31
    const int warp_in_block   = threadIdx.x / W;
    const int warps_per_block = blockDim.x / W;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    const int num_u_groups = (U + TileU - 1) / TileU;
    const int sender_idx   = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (sender_idx >= N) return;

    const int32_t st = start_idx[sender_idx];
    const int32_t ed = end_idx[sender_idx];
    if (st >= ed) return;

    const int u_block_start = u_group * TileU;
    const int u_block_end   = min(u_block_start + TileU, U);

    for (int u = u_block_start + lane; u < u_block_end; u += W) {
        if (u >= U) continue;

        const size_t x_idx =
            (size_t)sender_idx * (size_t)U + (size_t)u;
        const double x_val = node_feats[x_idx];

        double gx = 0.0;

        #pragma unroll
        for (int p = 0; p < MAX_D; ++p) {
            if (p >= P) break;

            const int d = dim_list[p];
            const int o = offs[p];

            for (int e = st; e < ed; ++e) {
                const int rcv = receiver[e];

                const size_t w_idx =
                    ((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u;
                const double w_val = tp_weights[w_idx];

                const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;
                const size_t gout_base =
                    ((size_t)rcv * (size_t)DIM_SUM + (size_t)o) * (size_t)U
                    + (size_t)u;

                double gw = 0.0;

                #pragma unroll
                for (int j = 0; j < MAX_D; ++j) {
                    if (j >= d) break;

                    const double yv   = edge_attrs[y_base + j];
                    const double gout =
                        grad_out_nodes[gout_base + (size_t)j * (size_t)U];

                    // 对 x 的梯度
                    gx += yv * w_val * gout;

                    // 对 w 的梯度
                    gw += x_val * yv * gout;

                    // 对 y 的梯度 (多个 u 会并发写同一个 y[e,o+j])
                    const double gy = x_val * w_val * gout;
                    atomicAdd(&grad_edge_attrs[y_base + j], gy);
                }

                grad_tp_weights[w_idx] = gw;
            } // e
        } // p

        grad_node_feats[x_idx] = gx;
    } // u-loop
}


template<typename T>
__inline__ __device__ T warp_reduce_sum(T v, unsigned mask) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

template<int TileU, int MAX_D = 8, typename scalar_t>
__global__ void fused_mp_warp_streaming_kernel_groupflush_scalar_backward_warp_reduce(
    const scalar_t* __restrict__ node_feats,       // [N, U]
    const scalar_t* __restrict__ edge_attrs,       // [E, DIM_SUM]
    const scalar_t* __restrict__ tp_weights,       // [E, P, U]
    const int32_t* __restrict__ receiver,          // [E]
    const int32_t* __restrict__ start_idx,         // [N]
    const int32_t* __restrict__ end_idx,           // [N]
    const int32_t* __restrict__ dim_list,          // [P]
    const int32_t* __restrict__ offs,              // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,

    const scalar_t* __restrict__ grad_out_nodes,   // [N, DIM_SUM, U]
    scalar_t* __restrict__ grad_node_feats,        // [N, U]
    scalar_t* __restrict__ grad_edge_attrs,        // [E, DIM_SUM]
    scalar_t* __restrict__ grad_tp_weights         // [E, P, U]
){
    constexpr int W = WARP_SIZE;

    const int lane            = threadIdx.x & (W - 1);  // 0..31
    const int warp_in_block   = threadIdx.x / W;
    const int warps_per_block = blockDim.x / W;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    const int num_u_groups = (U + TileU - 1) / TileU;
    const int sender_idx   = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (sender_idx >= N) return;

    const int32_t st = start_idx[sender_idx];
    const int32_t ed = end_idx[sender_idx];
    if (st >= ed) return;

    const int u_block_start = u_group * TileU;
    const int u_block_end   = min(u_block_start + TileU, U);

    for (int u = u_block_start + lane; u < u_block_end; u += W) {
        if (u >= U) continue;

        const size_t x_idx =
            (size_t)sender_idx * (size_t)U + (size_t)u;
        const scalar_t x_val = node_feats[x_idx];

        scalar_t gx = scalar_t(0);

        #pragma unroll
        for (int p = 0; p < MAX_D; ++p) {
            if (p >= P) break;

            const int d = dim_list[p];
            const int o = offs[p];

            for (int e = st; e < ed; ++e) {
                const int rcv = receiver[e];

                const size_t w_idx =
                    ((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u;
                const scalar_t w_val = tp_weights[w_idx];

                const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;
                const size_t gout_base =
                    ((size_t)rcv * (size_t)DIM_SUM + (size_t)o) * (size_t)U
                    + (size_t)u;

                scalar_t gw = scalar_t(0);

                #pragma unroll
                for (int j = 0; j < MAX_D; ++j) {
                    if (j >= d) break;

                    const scalar_t yv   = edge_attrs[y_base + j];
                    const scalar_t gout =
                        grad_out_nodes[gout_base + (size_t)j * (size_t)U];

                    // 对 x 的梯度
                    gx += yv * w_val * gout;

                    // 对 w 的梯度
                    gw += x_val * yv * gout;

                    // 对 y 的梯度（ warp-level reduction）
                    scalar_t gy_local = x_val * w_val * gout;

                    unsigned mask = __activemask();
                    scalar_t gy_sum = warp_reduce_sum(gy_local, mask);
                    // 如果你的 warp_reduce_sum 现在是 double 版本，
                    // 把它改成模板 <typename T> 再用这里的 scalar_t 就行。

                    // 当前 warp 内这个 (e, o+j) 的所有 u，只由 lane 0 做一次 atomicAdd
                    if (lane == 0) {
                        atomicAdd(&grad_edge_attrs[y_base + j], gy_sum);
                    }
                } // j

                // 写入 dL/d tp_weights[e,p,u]
                grad_tp_weights[w_idx] = gw;
            } // e
        } // p

        // 写入 dL/d node_feats[sender_idx, u]
        grad_node_feats[x_idx] = gx;
    } // u-loop
}


std::vector<torch::Tensor> fused_mp_backward(
    torch::Tensor grad_out_nodes, // [N, DIM_SUM, U]
    torch::Tensor node_feats,     // [N, U]
    torch::Tensor edge_attrs,     // [E, DIM_SUM]
    torch::Tensor tp_weights,     // [E, P, U]
    torch::Tensor receiver,       // [E]
    torch::Tensor start_idx,      // [N]
    torch::Tensor end_idx,        // [N]
    torch::Tensor dim_list,       // [P]
    torch::Tensor offs)           // [P]
{
    constexpr int kTileU = 32;

    TORCH_CHECK(grad_out_nodes.is_cuda(), "grad_out_nodes must be CUDA");
    TORCH_CHECK(node_feats.is_cuda(),     "node_feats must be CUDA");
    TORCH_CHECK(edge_attrs.is_cuda(),     "edge_attrs must be CUDA");
    TORCH_CHECK(tp_weights.is_cuda(),     "tp_weights must be CUDA");
    TORCH_CHECK(receiver.is_cuda(),       "receiver must be CUDA");
    TORCH_CHECK(start_idx.is_cuda(),      "start_idx must be CUDA");
    TORCH_CHECK(end_idx.is_cuda(),        "end_idx must be CUDA");
    TORCH_CHECK(dim_list.is_cuda(),       "dim_list must be CUDA");
    TORCH_CHECK(offs.is_cuda(),           "offs must be CUDA");

    // dtype 检查：所有浮点张量必须同 dtype，且是 float32 或 float64
    auto dtype = node_feats.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "node_feats must be float32 or float64");
    TORCH_CHECK(edge_attrs.scalar_type() == dtype,
                "edge_attrs must have same dtype as node_feats");
    TORCH_CHECK(tp_weights.scalar_type() == dtype,
                "tp_weights must have same dtype as node_feats");
    TORCH_CHECK(grad_out_nodes.scalar_type() == dtype,
                "grad_out_nodes must have same dtype as node_feats");

    TORCH_CHECK(dim_list.scalar_type() == at::kInt,
                "dim_list must be int32");
    TORCH_CHECK(offs.scalar_type() == at::kInt,
                "offs must be int32");

    grad_out_nodes = grad_out_nodes.contiguous();
    node_feats     = node_feats.contiguous();
    edge_attrs     = edge_attrs.contiguous();
    tp_weights     = tp_weights.contiguous();
    start_idx      = start_idx.contiguous();
    end_idx        = end_idx.contiguous();
    dim_list       = dim_list.contiguous();
    offs           = offs.contiguous();

    torch::Tensor receiver_i32 = (receiver.dtype() == at::kInt)
        ? receiver.contiguous()
        : receiver.to(at::kInt, /*non_blocking=*/true).contiguous();

    const int64_t N       = node_feats.size(0);
    const int64_t U       = node_feats.size(1);
    const int64_t E       = edge_attrs.size(0);
    const int64_t DIM_SUM = edge_attrs.size(1);
    const int64_t P       = dim_list.size(0);

    grad_out_nodes = grad_out_nodes.view({N, DIM_SUM, U});

    auto grad_node_feats = torch::zeros_like(node_feats);
    auto grad_edge_attrs = torch::zeros_like(edge_attrs);
    auto grad_tp_weights = torch::zeros_like(tp_weights);

    const int num_u_groups = (int)((U + kTileU - 1) / kTileU);
    const int total_warps  = (int)(N * num_u_groups);

    const int warps_per_block = 4;
    dim3 block(warps_per_block * WARP_SIZE);
    dim3 grid((total_warps + warps_per_block - 1) / warps_per_block);

    // grad_edge_attrs 用 atomicAdd 累加，初始为 0
    // grad_node_feats / grad_tp_weights 直接覆盖

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_mp_backward", [&] {
        using scalar_t = scalar_t;

        fused_mp_warp_streaming_kernel_groupflush_scalar_backward_warp_reduce<
            kTileU, 8, scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            node_feats.data_ptr<scalar_t>(),
            edge_attrs.data_ptr<scalar_t>(),
            tp_weights.data_ptr<scalar_t>(),
            receiver_i32.data_ptr<int32_t>(),
            start_idx.data_ptr<int32_t>(),
            end_idx.data_ptr<int32_t>(),
            dim_list.data_ptr<int32_t>(),
            offs.data_ptr<int32_t>(),
            (int32_t)N, (int32_t)E, (int32_t)U, (int32_t)P, (int32_t)DIM_SUM,
            grad_out_nodes.data_ptr<scalar_t>(),
            grad_node_feats.data_ptr<scalar_t>(),
            grad_edge_attrs.data_ptr<scalar_t>(),
            grad_tp_weights.data_ptr<scalar_t>());
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_node_feats, grad_edge_attrs, grad_tp_weights};
}


TORCH_LIBRARY(fused_mp_bwd, m)
{
    m.def("backward", &fused_mp_backward);
}