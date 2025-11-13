// fused_mp_warp_streaming.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void k_fill_zero_i32(int32_t* arr, int32_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) arr[i] = 0;
}

// sender 必须非降序（已按 sender 排序）
// 对每条边 i 并行：检测 run 起点/终点，无需原子
__global__ void k_runs_from_sorted_sender(
    const int32_t* __restrict__ sender, // [E], sorted
    int32_t E,
    int32_t* __restrict__ start_idx,    // [N]
    int32_t* __restrict__ end_idx       // [N]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;

    int32_t s = sender[i];

    // run start
    if (i == 0 || sender[i - 1] != s) {
        start_idx[s] = i;
    }
    // run end  (end is exclusive: i+1)
    if (i == E - 1 || sender[i + 1] != s) {
        end_idx[s] = i + 1;
    }
}

extern "C" void compute_sender_runs_sorted_cuda(
    const int32_t* sender,  // [E], int32, sorted, cuda
    int32_t E,
    int32_t N,
    int32_t* start_idx,     // [N], int32, cuda (output)
    int32_t* end_idx,       // [N], int32, cuda (output)
    cudaStream_t stream
){
    const int threads = 256;
    const int blocksN = (N + threads - 1) / threads;
    const int blocksE = (E + threads - 1) / threads;

    k_fill_zero_i32<<<blocksN, threads, 0, stream>>>(start_idx, N);
    k_fill_zero_i32<<<blocksN, threads, 0, stream>>>(end_idx, N);

    // 单 pass 根据排序好的 sender 写入每个节点的起止索引
    k_runs_from_sorted_sender<<<blocksE, threads, 0, stream>>>(
        sender, E, start_idx, end_idx
    );
}


std::vector<torch::Tensor> compute_sender_runs_sorted_binding(
    torch::Tensor sender, // [E], int32 或 int64, CUDA, 且已排序
    int64_t nnodes
){
    TORCH_CHECK(sender.is_cuda(), "sender must be CUDA tensor");
    TORCH_CHECK(sender.dtype() == at::kInt || sender.dtype() == at::kLong,
                "sender must be int32 or int64");
    TORCH_CHECK(nnodes >= 0, "nnodes must be non-negative");

    auto dev = sender.device();
    int64_t E64 = sender.numel();
    int32_t N = static_cast<int32_t>(nnodes);
    int32_t E = static_cast<int32_t>(E64);
    TORCH_CHECK((int64_t)N == nnodes && (int64_t)E == E64, "size too large for int32");

    // 若是 int64 转 int32（假定 sender 值域 < 2^31）
    torch::Tensor sender_i32 = (sender.dtype() == at::kInt)
        ? sender.contiguous()
        : sender.to(at::kInt, /*non_blocking=*/true).contiguous();

    auto opts_i32 = torch::TensorOptions().dtype(at::kInt).device(dev);
    auto start_idx = torch::empty({nnodes}, opts_i32);
    auto end_idx   = torch::empty({nnodes}, opts_i32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compute_sender_runs_sorted_cuda(
        sender_i32.data_ptr<int32_t>(),
        E, N,
        start_idx.data_ptr<int32_t>(),
        end_idx.data_ptr<int32_t>(),
        stream
    );

    return {start_idx, end_idx};
}

// ======================================================================
// 可选：receiver-major CSR 构建 (recv_row_ptr)
// 假设 edges 已经按 receiver 非降序排序：
//   receiver_sorted[e] ∈ [0, N-1]
// 则：
//   recv_row_ptr[r]   = 第一条 receiver==r 的边的索引
//   recv_row_ptr[r+1] = 最后一条 receiver==r 的边索引 + 1
// ======================================================================

__global__ void build_recv_row_ptr_kernel(
    const int32_t* __restrict__ receiver_sorted, // [E]
    int32_t N,
    int32_t E,
    int32_t* __restrict__ recv_row_ptr          // [N+1]
){
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    const int32_t r = receiver_sorted[e];

    if (e == 0) {
        // 第一条边
        recv_row_ptr[0] = 0;
    }

    // 边界检测：receiver 有变化的地方
    if (e == 0 || receiver_sorted[e-1] != r) {
        // r 是新的一段的开始
        recv_row_ptr[r] = e;
    }
    if (e == E-1) {
        // 最后一条边，补上尾部
        recv_row_ptr[r+1] = E;
        // 对于末尾没有出现过 edge 的 receiver，要在 host 端或额外 kernel 填补：
        // 例如在 host 端做一次 prefix fill，确保每个 r 都有 recv_row_ptr[r] <= recv_row_ptr[r+1]
    }
}

template<int TileU>
__global__ void fused_mp_warp_streaming_kernel(
    const double* __restrict__ node_feats,     // [N, U]
    const double* __restrict__ edge_attrs,     // [E, DIM_SUM]
    const double* __restrict__ tp_weights,     // [E, P, U]
    const int32_t* __restrict__ receiver,      // [E]
    const int32_t* __restrict__ start_idx,     // [N]
    const int32_t* __restrict__ end_idx,       // [N]
    const int32_t* __restrict__ dim_list,      // [P]
    const int32_t* __restrict__ offs,          // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    double* __restrict__ out_nodes             // [N, DIM_SUM, U]
){
    const int lane = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    const int num_u_groups = (U + TileU - 1) / TileU;
    const int sender_idx   = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (sender_idx >= N) return;

    const int32_t st = start_idx[sender_idx];
    const int32_t ed = end_idx[sender_idx];
    if (st >= ed) return;

    const int u0 = u_group * TileU;
    const int u  = u0 + lane;

    // 预取 x_s[u]
    double x_val = 0.0;
    if (u < U) x_val = node_feats[(size_t)sender_idx * (size_t)U + (size_t)u];

    // 主循环
    #pragma unroll
    for (int p = 0; p < /* small P, e.g. 4 */ 8; ++p) {
        if (p >= P) break;
        const int d = dim_list[p];
        const int o = offs[p];

        for (int e = st; e < ed; ++e) {
            double base_u = 0.0;
            if (u < U) {
                base_u = x_val * tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u];
            }

            const int rcv = receiver[e];
            if (u < U) {
                const size_t out_base_rcv = ((size_t)rcv * (size_t)DIM_SUM + (size_t)o) * (size_t)U + (size_t)u;
                const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j >= d) break;
                    const double y_val = edge_attrs[y_base + j];
                    const double msg   = y_val * base_u;
                    atomicAdd(&out_nodes[out_base_rcv + (size_t)j * (size_t)U], msg);
                }
            }
        }
    }
}

template<int MAX_D>
__device__ inline void flush_group_scalar_impl(
    int rcv, int d, int o,
    int u, int U, int DIM_SUM,
    double acc[MAX_D],
    double* __restrict__ out_nodes
){
    if (rcv < 0 || u >= U) return;
    const size_t out_base =
        ((size_t)rcv * (size_t)DIM_SUM + (size_t)o) * (size_t)U + (size_t)u;

    #pragma unroll
    for (int j = 0; j < MAX_D; ++j) {
        if (j >= d) break;
        atomicAdd(&out_nodes[out_base + (size_t)j * (size_t)U], acc[j]);
        acc[j] = 0.0;
    }
}

// ============================
//  标量版 fused_mp + group-flush + u 内层循环
//
//  映射：warp -> (sender_idx, u_block)
//  一个 warp 负责某个 sender 的一个 U-block：
//     u_block = (warp_u_group * TileU) .. min(u_block+TileU, U)
//  同一 warp 内通过 u-loop 支持 TileU >= / <= WARP_SIZE，
//  所以 TileU 可以是 32 / 64 / 96 / 128 / … 任意正数。
//  在 sender 段内，若 receiver 已按升序排列，
//  则相同 receiver 的边是连续的，可以 group-flush，大幅减少 atomic 次数。
// ============================

template<int TileU, int MAX_D = 8>
__global__ void fused_mp_warp_streaming_kernel_groupflush_scalar(
    const double* __restrict__ node_feats,     // [N, U]
    const double* __restrict__ edge_attrs,     // [E, DIM_SUM]
    const double* __restrict__ tp_weights,     // [E, P, U]
    const int32_t* __restrict__ receiver,      // [E]
    const int32_t* __restrict__ start_idx,     // [N]
    const int32_t* __restrict__ end_idx,       // [N]
    const int32_t* __restrict__ dim_list,      // [P]
    const int32_t* __restrict__ offs,          // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    double* __restrict__ out_nodes             // [N, DIM_SUM, U]
){
    constexpr int W = WARP_SIZE;

    const int lane            = threadIdx.x & (W - 1);  // 0..31
    const int warp_in_block   = threadIdx.x / W;
    const int warps_per_block = blockDim.x / W;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    // 2D 映射：warp -> (sender_idx, u_group)
    // 每个 warp 只负责一个 sender 的一个 U-block（长度约为 TileU）
    const int num_u_groups = (U + TileU - 1) / TileU;
    const int sender_idx   = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (sender_idx >= N) return;

    const int32_t st = start_idx[sender_idx];
    const int32_t ed = end_idx[sender_idx];
    if (st >= ed) return; // 无出边

    const int u_block_start = u_group * TileU;
    const int u_block_end   = min(u_block_start + TileU, U);

    // 遍历本 warp 负责的 U-block，中每个 lane 以步长 W 遍历自己的 u
    for (int u = u_block_start + lane; u < u_block_end; u += W) {
        if (u >= U) continue;

        // 读取 x[sender, u]
        const double x_val = node_feats[(size_t)sender_idx * (size_t)U + (size_t)u];

        // 遍历每个 path（小 d），对该 u 做 sender 段的 group-flush
        #pragma unroll
        for (int p = 0; p < MAX_D; ++p) {
            if (p >= P) break;
            const int d = dim_list[p];
            const int o = offs[p];

            int    cur_rcv = -1;
            double acc[MAX_D];
            #pragma unroll
            for (int j = 0; j < MAX_D; ++j) acc[j] = 0.0;

            // 扫描 sender 段 e=st..ed-1（如果段内 receiver 升序，group-flush 效果最好）
            for (int e = st; e < ed; ++e) {
                double base_u = 0.0;
                if (u < U) {
                    base_u = x_val *
                        tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u];
                }

                const int rcv = receiver[e];

                // 新的 receiver 组：flush 旧组
                if (rcv != cur_rcv) {
                    flush_group_scalar_impl<MAX_D>(
                        cur_rcv, d, o, u, U, DIM_SUM, acc, out_nodes);
                    cur_rcv = rcv;
                }

                if (u < U) {
                    const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;
                    #pragma unroll
                    for (int j = 0; j < MAX_D; ++j) {
                        if (j >= d) break;
                        const double yv = edge_attrs[y_base + j];
                        acc[j] += yv * base_u;
                    }
                }
            } // e

            // 段尾 flush 最后一组
            flush_group_scalar_impl<MAX_D>(
                cur_rcv, d, o, u, U, DIM_SUM, acc, out_nodes);
        } // p
    } // u-loop
}


extern "C" void fused_mp_warp_streaming_launch(
    const double* node_feats,     // [N, U]
    const double* edge_attrs,     // [E, DIM_SUM]
    const double* tp_weights,     // [E, P, U]
    const int32_t* receiver,      // [E]
    const int32_t* start_idx,     // [N]
    const int32_t* end_idx,       // [N]
    const int32_t* dim_list,      // [P]
    const int32_t* offs,          // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    double* out_nodes,            // [N, DIM_SUM, U]
    cudaStream_t stream,
    int TileU,
    int warps_per_block
){
    
    const int num_u_groups = (U + TileU - 1) / TileU;
    const int total_warps  = N * num_u_groups;
    const int threads_per_block = warps_per_block * 32;   // try 8 or 16
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    

    switch (TileU) {
        case 16:
            fused_mp_warp_streaming_kernel<16><<<blocks, threads_per_block, 0, stream>>>(
                node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
            break;
        case 32:
            fused_mp_warp_streaming_kernel<32><<<blocks, threads_per_block, 0, stream>>>(
                node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
            break;
        case 64:
            fused_mp_warp_streaming_kernel<64><<<blocks, threads_per_block, 0, stream>>>(
                node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
            break;
        default:
            fused_mp_warp_streaming_kernel<32><<<blocks, threads_per_block, 0, stream>>>(
                node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
    }
#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
#endif
}



extern "C" void fused_mp_warp_streaming_scalar_launch(
    const double* node_feats,     // [N, U]
    const double* edge_attrs,     // [E, DIM_SUM]
    const double* tp_weights,     // [E, P, U]
    const int32_t* receiver,      // [E]
    const int32_t* start_idx,     // [N]
    const int32_t* end_idx,       // [N]
    const int32_t* dim_list,      // [P]
    const int32_t* offs,          // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    double* out_nodes,            // [N, DIM_SUM, U]
    cudaStream_t stream,
    int TileU,                    // 建议 32
    int warps_per_block           // 建议 8 或 16
){
    if (TileU <= 0) TileU = 32;

    const int num_u_groups = (U + TileU - 1) / TileU;
    const int total_warps  = N * num_u_groups;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    switch (TileU) {
        case 16:  fused_mp_warp_streaming_kernel_groupflush_scalar<16><<<blocks, threads_per_block, 0, stream>>>(
                    node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                    dim_list, offs, N, E, U, P, DIM_SUM, out_nodes); break;
        case 32:  fused_mp_warp_streaming_kernel_groupflush_scalar<32><<<blocks, threads_per_block, 0, stream>>>(
                    node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                    dim_list, offs, N, E, U, P, DIM_SUM, out_nodes); break;
        case 64:  fused_mp_warp_streaming_kernel_groupflush_scalar<64><<<blocks, threads_per_block, 0, stream>>>(
                    node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                    dim_list, offs, N, E, U, P, DIM_SUM, out_nodes); break;
        case 96:  fused_mp_warp_streaming_kernel_groupflush_scalar<96><<<blocks, threads_per_block, 0, stream>>>(
                    node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                    dim_list, offs, N, E, U, P, DIM_SUM, out_nodes); break;
        default:  fused_mp_warp_streaming_kernel_groupflush_scalar<32><<<blocks, threads_per_block, 0, stream>>>(
                    node_feats, edge_attrs, tp_weights, receiver, start_idx, end_idx,
                    dim_list, offs, N, E, U, P, DIM_SUM, out_nodes); break;
    }

#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }
#endif
}

torch::Tensor fused_mp_forward(
    torch::Tensor node_feats,   // [N,U], float64, cuda, contiguous
    torch::Tensor edge_attrs,   // [E,DIM_SUM], float64, cuda, contiguous
    torch::Tensor tp_weights,   // [E,P,U], float64, cuda, contiguous
    torch::Tensor receiver,     // [E], int32, cuda, contiguous
    torch::Tensor start_idx,    // [N], int32, cuda, contiguous
    torch::Tensor end_idx,      // [N], int32, cuda, contiguous
    torch::Tensor dim_list,     // [P], int32, cuda/CPU (we’ll move to cuda if needed)
    torch::Tensor offs,         // [P], int32, cuda/CPU
    int TileU = 32,
    int warps_per_block = 4
){
    TORCH_CHECK(node_feats.is_cuda() && edge_attrs.is_cuda() &&
                tp_weights.is_cuda() && receiver.is_cuda() &&
                start_idx.is_cuda()  && end_idx.is_cuda(),
                "All main tensors must be CUDA");
    TORCH_CHECK(node_feats.scalar_type() == at::kDouble, "node_feats must be float64");
    TORCH_CHECK(edge_attrs.scalar_type() == at::kDouble, "edge_attrs must be float64");
    TORCH_CHECK(tp_weights.scalar_type() == at::kDouble, "tp_weights must be float64");
    TORCH_CHECK(receiver.scalar_type() == at::kInt, "receiver must be int32");
    TORCH_CHECK(start_idx.scalar_type() == at::kInt, "start_idx must be int32");
    TORCH_CHECK(end_idx.scalar_type() == at::kInt, "end_idx must be int32");
    TORCH_CHECK(dim_list.scalar_type() == at::kInt, "dim_list must be int32");
    TORCH_CHECK(offs.scalar_type() == at::kInt, "offs must be int32");

    auto node_feats_c = node_feats.contiguous();
    auto edge_attrs_c = edge_attrs.contiguous();
    auto tp_weights_c = tp_weights.contiguous();
    auto receiver_c   = receiver.contiguous();
    auto start_idx_c  = start_idx.contiguous();
    auto end_idx_c    = end_idx.contiguous();

    // dim_list/offs 若在 CPU，搬到 GPU（很小的数组）
    if (!dim_list.is_cuda()) dim_list = dim_list.to(node_feats.device(), /*non_blocking=*/true);
    if (!offs.is_cuda())     offs     = offs.to(node_feats.device(), /*non_blocking=*/true);
    auto dim_list_c = dim_list.contiguous();
    auto offs_c     = offs.contiguous();

    const auto N = static_cast<int32_t>(node_feats_c.size(0));
    const auto U = static_cast<int32_t>(node_feats_c.size(1));
    const auto E = static_cast<int32_t>(edge_attrs_c.size(0));
    const auto DIM_SUM = static_cast<int32_t>(edge_attrs_c.size(1));
    const auto P = static_cast<int32_t>(tp_weights_c.size(1));

    TORCH_CHECK(tp_weights_c.size(0) == E && tp_weights_c.size(2) == U, "tp_weights shape mismatch");
    TORCH_CHECK(receiver_c.size(0) == E, "receiver size mismatch");
    TORCH_CHECK(start_idx_c.size(0) == N && end_idx_c.size(0) == N, "start/end size mismatch");
    TORCH_CHECK(dim_list_c.size(0) == P && offs_c.size(0) == P, "dim_list/offs size mismatch");

    auto out_nodes = torch::zeros({(int64_t)N, (int64_t)DIM_SUM, (int64_t)U},
                                  node_feats_c.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    
    //fused_mp_warp_streaming_launch(
    fused_mp_warp_streaming_scalar_launch(
        node_feats_c.data_ptr<double>(),
        edge_attrs_c.data_ptr<double>(),
        tp_weights_c.data_ptr<double>(),
        receiver_c.data_ptr<int32_t>(),
        start_idx_c.data_ptr<int32_t>(),
        end_idx_c.data_ptr<int32_t>(),
        dim_list_c.data_ptr<int32_t>(),
        offs_c.data_ptr<int32_t>(),
        N, E, U, P, DIM_SUM,
        out_nodes.data_ptr<double>(),
        stream,
        TileU,
        warps_per_block
    );

    return out_nodes;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sender_runs_sorted", &compute_sender_runs_sorted_binding,
        "Compute [start_idx,end_idx] for pre-sorted sender on GPU");
    m.def("forward", &fused_mp_forward, "Fused MP (warp streaming) forward");
}