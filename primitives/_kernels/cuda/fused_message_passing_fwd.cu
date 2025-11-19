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

// 将sender、receiver 的排序数组从COO变为CSR格式
__global__ void k_runs_from_sorted_sender_receiver(
    const int32_t* __restrict__ sender_receiver, // [E], sorted
    int32_t E,
    int32_t* __restrict__ start_idx,    // [N]
    int32_t* __restrict__ end_idx       // [N]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;

    int32_t s = sender_receiver[i];

    // run start
    if (i == 0 || sender_receiver[i - 1] != s) {
        start_idx[s] = i;
    }
    // run end  (end is exclusive: i+1)
    if (i == E - 1 || sender_receiver[i + 1] != s) {
        end_idx[s] = i + 1;
    }
}

extern "C" void compute_sender_receiver_csr(
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
    k_runs_from_sorted_sender_receiver<<<blocksE, threads, 0, stream>>>(
        sender, E, start_idx, end_idx
    );
}


std::vector<torch::Tensor> compute_sender_receiver_csr_binding(
    torch::Tensor sender_receiver, // [E], int32 或 int64, CUDA, 且已排序
    int64_t nnodes
){
    TORCH_CHECK(sender_receiver.is_cuda(), "sender must be CUDA tensor");
    TORCH_CHECK(sender_receiver.dtype() == at::kInt || sender_receiver.dtype() == at::kLong,
                "sender must be int32 or int64");
    TORCH_CHECK(nnodes >= 0, "nnodes must be non-negative");

    auto dev = sender_receiver.device();
    int64_t E64 = sender_receiver.numel();
    int32_t N = static_cast<int32_t>(nnodes);
    int32_t E = static_cast<int32_t>(E64);
    TORCH_CHECK((int64_t)N == nnodes && (int64_t)E == E64, "size too large for int32");

    // 若是 int64 转 int32（假定 sender 值域 < 2^31）
    torch::Tensor sender_receiver_i32 = (sender_receiver.dtype() == at::kInt)
        ? sender_receiver.contiguous()
        : sender_receiver.to(at::kInt, /*non_blocking=*/true).contiguous();

    auto opts_i32 = torch::TensorOptions().dtype(at::kInt).device(dev);
    auto start_idx = torch::empty({nnodes}, opts_i32);
    auto end_idx   = torch::empty({nnodes}, opts_i32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compute_sender_receiver_csr(
        sender_receiver_i32.data_ptr<int32_t>(),
        E, N,
        start_idx.data_ptr<int32_t>(),
        end_idx.data_ptr<int32_t>(),
        stream
    );

    return {start_idx, end_idx};
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
//  sender-major 设计， 适合sender 出度很大的情况。
//  对于拥有大量出边的 sender，这个 x_val 在 warp 内可以被多次复用
//  映射：warp -> (sender_idx, u_block)
//  一个 warp 负责某个 sender 的一个 U-block
//  在 sender 段内，若 receiver 已按升序排列，
// MAX_D=8 是因为dim_list = {1, 3, 5, 7}; 
// ============================

template<int TileU = 32, int MAX_D = 8>
__global__ void fused_mp_warp_sender_major_allpaths(
    const double* __restrict__ node_feats,     // [N, U]
    const double* __restrict__ edge_attrs,     // [E, DIM_SUM]
    const double* __restrict__ tp_weights,     // [E, P, U]
    const int32_t* __restrict__ sender,        // [E]
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

            // 扫描 sender 段 e=st..ed-1
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


// ============================
//  receiver-major 设计， 减少L2 cache的原子开销
// ============================

template<int TileU, int MAX_D = 8>
__global__ void fused_mp_warp_receiver_major_allpaths(
    const double* __restrict__ node_feats,      // [N, U]
    const double* __restrict__ edge_attrs,      // [E, DIM_SUM]  // 已按 receiver 排序
    const double* __restrict__ tp_weights,      // [E, P, U]     // 已按 receiver 排序
    const int32_t* __restrict__ sender_sorted,  // [E]           // 对应每条边的 sender
    const int32_t* __restrict__ receiver_sorted,// [E] (其实 kernel 内只用 CSR 就够)
    const int32_t* __restrict__ recv_start,     // [N]，按 receiver 的 CSR start
    const int32_t* __restrict__ recv_end,       // [N]，按 receiver 的 CSR end
    const int32_t* __restrict__ dim_list,       // [P]
    const int32_t* __restrict__ offs,           // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    double* __restrict__ out_nodes              // [N, DIM_SUM, U]
){
    constexpr int W = WARP_SIZE; // 32

    const int lane            = threadIdx.x & (W - 1);    // 0..31
    const int warp_in_block   = threadIdx.x / W;          // 0..(blockDim.x/W-1)
    const int warps_per_block = blockDim.x / W;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    // 2D 映射：warp -> (receiver_idx, u_group)
    const int num_u_groups = (U + TileU - 1) / TileU;
    const int receiver_idx = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (receiver_idx >= N) return;

    const int st = recv_start[receiver_idx];
    const int ed = recv_end[receiver_idx];
    if (st >= ed) {
        // 该 receiver 没有入边
        return;
    }

    const int u_block_start = u_group * TileU;
    const int u_block_end   = min(u_block_start + TileU, U);

    // 遍历这个 warp 负责的 U-block
    for (int u = u_block_start + lane; u < u_block_end; u += W) {
        if (u >= U) continue;

        // 这个 warp 中每个 lane 负责一个 u（stride 32）
        // 对这个 (receiver_idx, u)，遍历所有 path p
        for (int p = 0; p < P; ++p) {
            const int d = dim_list[p];
            const int o = offs[p];

            // 局部累加缓冲（小 d 方向，最大 MAX_D）
            double acc[MAX_D];
            #pragma unroll
            for (int j = 0; j < MAX_D; ++j) {
                acc[j] = 0.0;
            }

            // 扫描该 receiver 的所有入边 e ∈ [st, ed)
            for (int e = st; e < ed; ++e) {
                const int s = sender_sorted[e]; // 注意：这里用按 receiver 排序后的 sender

                // 读取 x[sender, u]
                const double x_val = node_feats[(size_t)s * (size_t)U + (size_t)u];

                // 对应 path p 的权重
                const double w_val =
                    tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u];

                const double base_u = x_val * w_val;

                // edge_attrs 段 [o, o+d)
                const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;

                #pragma unroll
                for (int j = 0; j < MAX_D; ++j) {
                    if (j >= d) break;
                    const double yv = edge_attrs[y_base + j];
                    acc[j] += yv * base_u;
                }
            } // e-loop

            // 写回 out_nodes[receiver_idx, o .. o+d-1, u]
            // 注意：receiver-major 下，每个 (receiver_idx, u) 只由一个 warp 负责，
            // 不需要 atomicAdd，只要保证 out_nodes 事先清零即可。
            const size_t out_base =
                ((size_t)receiver_idx * (size_t)DIM_SUM + (size_t)o) * (size_t)U +
                (size_t)u;

            #pragma unroll
            for (int j = 0; j < MAX_D; ++j) {
                if (j >= d) break;
                out_nodes[out_base + (size_t)j * (size_t)U] += acc[j];
            }
        } // p-loop
    } // u-loop
}


extern "C" void fused_mp_launch(
    const double* node_feats,     // [N, U]
    const double* edge_attrs,     // [E, DIM_SUM]
    const double* tp_weights,     // [E, P, U]
    const int32_t* sender,        // [E]
    const int32_t* receiver,      // [E]
    const int32_t* start_idx,     // [N]
    const int32_t* end_idx,       // [N]
    const int32_t* dim_list,      // [P]
    const int32_t* offs,          // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    double* out_nodes,            // [N, DIM_SUM, U]
    cudaStream_t stream,
    bool receiver_major = false
) {

    const int warps_per_block = 8;
    const int TileU = 32;
    const int num_u_groups = (U + TileU - 1) / TileU;
    const int total_warps  = N * num_u_groups;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    if (receiver_major) {
        // Receiver Major
        fused_mp_warp_receiver_major_allpaths<32><<<blocks, threads_per_block, 0, stream>>>(
            node_feats, edge_attrs, tp_weights, sender, receiver, start_idx, end_idx,
            dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
    } else {
        // Sender Major
        fused_mp_warp_sender_major_allpaths<32><<<blocks, threads_per_block, 0, stream>>>(
            node_feats, edge_attrs, tp_weights, sender, receiver, start_idx, end_idx,
            dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
    }

#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }
#endif
}

std::vector<torch::Tensor> fused_mp_forward(
    torch::Tensor node_feats,   // [N,U], float64, cuda, contiguous
    torch::Tensor edge_attrs,   // [E,DIM_SUM], float64, cuda, contiguous
    torch::Tensor tp_weights,   // [E,P,U], float64, cuda, contiguous
    torch::Tensor sender,       // [E], int32, cuda, contiguous
    torch::Tensor receiver,     // [E], int32, cuda, contiguous
    torch::Tensor dim_list,     // [P], int32, cuda/CPU
    torch::Tensor offs,         // [P], int32, cuda/CPU
    bool receiver_major = false
){
    
    TORCH_CHECK(node_feats.is_cuda() && edge_attrs.is_cuda() &&
                tp_weights.is_cuda() && sender.is_cuda() && receiver.is_cuda(),
                "All main tensors must be CUDA");
    TORCH_CHECK(node_feats.scalar_type() == at::kDouble, "node_feats must be float64");
    TORCH_CHECK(edge_attrs.scalar_type() == at::kDouble, "edge_attrs must be float64");
    TORCH_CHECK(tp_weights.scalar_type() == at::kDouble, "tp_weights must be float64");
    TORCH_CHECK(dim_list.scalar_type() == at::kInt, "dim_list must be int32");
    TORCH_CHECK(offs.scalar_type() == at::kInt, "offs must be int32");

    auto node_feats_c = node_feats.contiguous();
    auto edge_attrs_c = edge_attrs.contiguous();
    auto tp_weights_c = tp_weights.contiguous();


    if (!dim_list.is_cuda()) dim_list = dim_list.to(node_feats.device(), /*non_blocking=*/true);
    if (!offs.is_cuda())     offs     = offs.to(node_feats.device(), /*non_blocking=*/true);

    const auto N = static_cast<int32_t>(node_feats_c.size(0));
    const auto U = static_cast<int32_t>(node_feats_c.size(1));
    const auto E = static_cast<int32_t>(edge_attrs_c.size(0));
    const auto DIM_SUM = static_cast<int32_t>(edge_attrs_c.size(1));
    const auto P = static_cast<int32_t>(offs.size(0));

    tp_weights_c = tp_weights_c.view({E, P, U});

    TORCH_CHECK(tp_weights_c.size(0) == E && tp_weights_c.size(2) == U, "tp_weights shape mismatch");
    TORCH_CHECK(receiver.size(0) == E, "receiver size mismatch");
    TORCH_CHECK(dim_list.size(0) == offs.size(0), "dim_list/offs size mismatch");

    torch::Tensor sender_i32 = (sender.dtype() == at::kInt)
        ? sender.contiguous()
        : sender.to(at::kInt, /*non_blocking=*/true).contiguous();
    
    torch::Tensor receiver_i32 = (receiver.dtype() == at::kInt)
    ? receiver.contiguous()
    : receiver.to(at::kInt, /*non_blocking=*/true).contiguous();

    auto opts_i32 = torch::TensorOptions().dtype(at::kInt).device(node_feats.device());
    auto start_idx = torch::empty({N}, opts_i32);
    auto end_idx   = torch::empty({N}, opts_i32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (receiver_major) {
        compute_sender_receiver_csr(
            receiver_i32.data_ptr<int32_t>(),
            E, N,
            start_idx.data_ptr<int32_t>(),
            end_idx.data_ptr<int32_t>(),
            stream
        );
    } else {
        compute_sender_receiver_csr(
            sender_i32.data_ptr<int32_t>(),
            E, N,
            start_idx.data_ptr<int32_t>(),
            end_idx.data_ptr<int32_t>(),
            stream
        );
    }
    
    auto out_nodes = torch::zeros({N, DIM_SUM, U},
                                  node_feats_c.options());

    fused_mp_launch(
        node_feats_c.data_ptr<double>(),
        edge_attrs_c.data_ptr<double>(),
        tp_weights_c.data_ptr<double>(),
        sender_i32.data_ptr<int32_t>(),
        receiver_i32.data_ptr<int32_t>(),
        start_idx.data_ptr<int32_t>(),
        end_idx.data_ptr<int32_t>(),
        dim_list.data_ptr<int32_t>(),
        offs.data_ptr<int32_t>(),
        N, E, U, P, DIM_SUM,
        out_nodes.data_ptr<double>(),
        stream,
        receiver_major
    );
    out_nodes = out_nodes.view({N, DIM_SUM * U});
    return {out_nodes, start_idx, end_idx};
}


TORCH_LIBRARY(fused_mp_fwd, m)
{
    m.def("forward", &fused_mp_forward);
}