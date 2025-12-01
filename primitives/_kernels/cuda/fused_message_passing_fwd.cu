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
    const int32_t* sender_receiver,  // [E], int32, sorted, cuda
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
        sender_receiver, E, start_idx, end_idx
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

template<int TileU, int MAX_D = 8, typename scalar_t>
__global__ void fused_mp_warp_receiver_major_allpaths(
    const scalar_t* __restrict__ node_feats,       // [N, U]
    const scalar_t* __restrict__ edge_attrs,       // [E, DIM_SUM]
    const scalar_t* __restrict__ tp_weights,       // [E, P, U]
    const int32_t* __restrict__ sender_sorted,     // [E]
    const int32_t* __restrict__ receiver_sorted,   // [E] (这里只是保持接口一致，不用也可)
    const int32_t* __restrict__ recv_start,        // [N]
    const int32_t* __restrict__ recv_end,          // [N]
    const int32_t* __restrict__ dim_list,          // [P]
    const int32_t* __restrict__ offs,              // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    scalar_t* __restrict__ out_nodes               // [N, DIM_SUM, U]
){
    constexpr int W = WARP_SIZE;

    const int lane            = threadIdx.x & (W - 1);
    const int warp_in_block   = threadIdx.x / W;
    const int warps_per_block = blockDim.x / W;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    const int num_u_groups = (U + TileU - 1) / TileU;
    const int receiver_idx = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (receiver_idx >= N) return;

    const int st = recv_start[receiver_idx];
    const int ed = recv_end[receiver_idx];
    if (st >= ed) return;

    const int u_block_start = u_group * TileU;
    const int u_block_end   = min(u_block_start + TileU, U);

    for (int u = u_block_start + lane; u < u_block_end; u += W) {
        if (u >= U) continue;

        for (int p = 0; p < P; ++p) {
            const int d = dim_list[p];
            const int o = offs[p];

            scalar_t acc[MAX_D];
            #pragma unroll
            for (int j = 0; j < MAX_D; ++j) {
                acc[j] = scalar_t(0);
            }

            for (int e = st; e < ed; ++e) {
                const int s = sender_sorted[e];

                const scalar_t x_val =
                    node_feats[(size_t)s * (size_t)U + (size_t)u];

                const scalar_t w_val =
                    tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u];

                const scalar_t base_u = x_val * w_val;

                const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;

                #pragma unroll
                for (int j = 0; j < MAX_D; ++j) {
                    if (j >= d) break;
                    const scalar_t yv = edge_attrs[y_base + j];
                    acc[j] += yv * base_u;
                }
            } // e-loop

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



template<int TileU = 32, int MAX_D = 8, typename scalar_t>
__global__ void fused_mp_warp_sender_major_allpaths(
    const scalar_t* __restrict__ node_feats,     // [N, U]
    const scalar_t* __restrict__ edge_attrs,     // [E, DIM_SUM]
    const scalar_t* __restrict__ tp_weights,     // [E, P, U]
    const int32_t* __restrict__ sender,          // [E]
    const int32_t* __restrict__ receiver,        // [E]
    const int32_t* __restrict__ start_idx,       // [N]
    const int32_t* __restrict__ end_idx,         // [N]
    const int32_t* __restrict__ dim_list,        // [P]
    const int32_t* __restrict__ offs,            // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    scalar_t* __restrict__ out_nodes             // [N, DIM_SUM, U]
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

        const scalar_t x_val = node_feats[(size_t)sender_idx * (size_t)U + (size_t)u];

        #pragma unroll
        for (int p = 0; p < MAX_D; ++p) {
            if (p >= P) break;
            const int d = dim_list[p];
            const int o = offs[p];

            int       cur_rcv = -1;
            scalar_t  acc[MAX_D];
            #pragma unroll
            for (int j = 0; j < MAX_D; ++j) acc[j] = scalar_t(0);

            for (int e = st; e < ed; ++e) {
                scalar_t base_u = scalar_t(0);
                if (u < U) {
                    base_u = x_val *
                        tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u];
                }

                const int rcv = receiver[e];

                if (rcv != cur_rcv) {
                    flush_group_scalar_impl<scalar_t, MAX_D>(
                        cur_rcv, d, o, u, U, DIM_SUM, acc, out_nodes);
                    cur_rcv = rcv;
                }

                if (u < U) {
                    const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;
                    #pragma unroll
                    for (int j = 0; j < MAX_D; ++j) {
                        if (j >= d) break;
                        const scalar_t yv = edge_attrs[y_base + j];
                        acc[j] += yv * base_u;
                    }
                }
            } // e

            flush_group_scalar_impl<scalar_t, MAX_D>(
                cur_rcv, d, o, u, U, DIM_SUM, acc, out_nodes);
        } // p
    } // u-loop
}

template<int TileU = 32, int MAX_D = 8, typename scalar_t>
__global__ void fused_mp_warp_sender_major_allpaths_v2(
    const scalar_t* __restrict__ node_feats,     // [N, U]
    const scalar_t* __restrict__ edge_attrs,     // [E, DIM_SUM]
    const scalar_t* __restrict__ tp_weights,     // [E, P, U]
    const int32_t* __restrict__ sender,          // [E] （实际上 kernel 里用的是 start_idx/end_idx）
    const int32_t* __restrict__ receiver,        // [E]
    const int32_t* __restrict__ start_idx,       // [N]
    const int32_t* __restrict__ end_idx,         // [N]
    const int32_t* __restrict__ dim_list,        // [P]
    const int32_t* __restrict__ offs,            // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    scalar_t* __restrict__ out_nodes             // [N, DIM_SUM, U]
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

    // --------------------------------------
    // shared memory：每个 warp 私有一小段 [MAX_D]，缓存 edge_attrs[e, o:o+d]
    // layout: smem_y[warps_per_block][MAX_D]
    // --------------------------------------

    extern __shared__ char smem_raw[]; // 全 block 共享的动态 shared memory
    scalar_t* smem_y_all = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* smem_y     = smem_y_all + warp_in_block * MAX_D;

    // u-loop：每个 warp 负责一个 TileU 的 u 范围，lane 负责其中一部分
    for (int u = u_block_start + lane; u < u_block_end; u += W) {
        if (u >= U) continue;

        // x_val 放在 register，后面所有 e / p 循环复用
        const scalar_t x_val = node_feats[(size_t)sender_idx * (size_t)U + (size_t)u];

        // 对每个 path p
        #pragma unroll
        for (int p = 0; p < MAX_D; ++p) {
            if (p >= P) break;
            const int d = dim_list[p];   // 该 path 的 dim 长度（<= MAX_D, <= DIM_SUM）
            const int o = offs[p];       // 在 DIM_SUM 上的 offset

            int       cur_rcv = -1;
            scalar_t  acc[MAX_D];
            #pragma unroll
            for (int j = 0; j < MAX_D; ++j) acc[j] = scalar_t(0);

            // 遍历此 sender 的所有 edges
            for (int e = st; e < ed; ++e) {

                // -------------------------
                // 1) 预取 edge_attrs[e, o:o+d] 到 shared memory
                //    仅由 warp 内前 d 个 lane 负责加载，然后所有 lane 使用 smem_y[j]
                // -------------------------
                const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;

                // 【标量版本】所有类型通用
                if (lane < d) {
                    smem_y[lane] = edge_attrs[y_base + lane];
                }
                // 若想在 float + d==4/8 情况下使用 vec4，可以改成：
                // if constexpr (std::is_same<scalar_t,float>::value) {
                //   if (d == 4 && lane == 0) {
                //      auto v = *reinterpret_cast<const float4*>(edge_attrs + y_base);
                //      smem_y[0] = v.x; smem_y[1] = v.y; smem_y[2] = v.z; smem_y[3] = v.w;
                //   } else if (d == 8 && lane < 2) {
                //      auto v = reinterpret_cast<const float4*>(edge_attrs + y_base)[lane];
                //      smem_y[lane*4 + 0] = v.x;
                //      smem_y[lane*4 + 1] = v.y;
                //      smem_y[lane*4 + 2] = v.z;
                //      smem_y[lane*4 + 3] = v.w;
                //   }
                // } else {
                //   if (lane < d) smem_y[lane] = edge_attrs[y_base + lane];
                // }

                __syncwarp();

                // -------------------------
                // 2) 计算 base_u（依赖 u），用 register 存储
                // -------------------------
                scalar_t base_u = scalar_t(0);
                if (u < U) {
                    base_u = x_val *
                        tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u];
                }

                const int rcv = receiver[e];

                // -------------------------
                // 3) run-length by receiver：切换 receiver 时 flush 上一个 acc
                // -------------------------
                if (rcv != cur_rcv) {
                    flush_group_scalar_impl<scalar_t, MAX_D>(
                        cur_rcv, d, o, u, U, DIM_SUM, acc, out_nodes);
                    cur_rcv = rcv;
                }

                // -------------------------
                // 4) 累加：acc[j] += smem_y[j] * base_u
                //    注意：此时 y[j] 已经在 smem_y 里，所有 lane 读的是同一缓存
                // -------------------------
                if (u < U) {
                    #pragma unroll
                    for (int j = 0; j < MAX_D; ++j) {
                        if (j >= d) break;
                        const scalar_t yv = smem_y[j];
                        acc[j] += yv * base_u;
                    }
                }

                __syncwarp(); // 确保下一个 e 使用 smem_y 之前已经用完当前的
            } // e-loop

            // flush 最后一段 run
            flush_group_scalar_impl<scalar_t, MAX_D>(
                cur_rcv, d, o, u, U, DIM_SUM, acc, out_nodes);
        } // p-loop
    } // u-loop
}

template<int TileU = 64, int MAX_D = 8, typename scalar_t>
__global__ void fused_mp_warp_sender_major_allpaths_v3(
    const scalar_t* __restrict__ node_feats,     // [N, U]
    const scalar_t* __restrict__ edge_attrs,     // [E, DIM_SUM]
    const scalar_t* __restrict__ tp_weights,     // [E, P, U]
    const int32_t* __restrict__ sender,          // [E]  // 未用，但保持接口
    const int32_t* __restrict__ receiver,        // [E]
    const int32_t* __restrict__ start_idx,       // [N]
    const int32_t* __restrict__ end_idx,         // [N]
    const int32_t* __restrict__ dim_list,        // [P]
    const int32_t* __restrict__ offs,            // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    scalar_t* __restrict__ out_nodes             // [N, DIM_SUM, U]
){
    constexpr int W = WARP_SIZE;

    const int lane            = threadIdx.x & (W - 1);  // 0..31
    const int warp_in_block   = threadIdx.x / W;
    const int warps_per_block = blockDim.x / W;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    // -------- warp 映射：sender_idx, u_group (TileU=64) --------
    const int num_u_groups = (U + TileU - 1) / TileU;   // U=96 -> 2
    const int sender_idx   = warp_global / num_u_groups;
    const int u_group      = warp_global % num_u_groups;
    if (sender_idx >= N) return;

    const int32_t st = start_idx[sender_idx];
    const int32_t ed = end_idx[sender_idx];
    if (st >= ed) return;

    const int u_block_start = u_group * TileU;
    const int u_block_end   = min(u_block_start + TileU, U);

    // 每个 lane 负责两个 u：u0, u1 = u0 + 32
    const int u0 = u_block_start + lane;
    const int u1 = u0 + W;  // W=32

    const bool valid_u0 = (u0 < U);
    const bool valid_u1 = (u1 < U);

    if (!valid_u0 && !valid_u1) {
        return;
    }

    // ---- 动态 shared memory: [warps_per_block][MAX_D] ----
    extern __shared__ char smem_raw[];
    scalar_t* smem_y_all = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* smem_y     = smem_y_all + warp_in_block * MAX_D;

    // 预先 load x0/x1
    scalar_t x0 = scalar_t(0);
    scalar_t x1 = scalar_t(0);
    if (valid_u0) {
        x0 = node_feats[(size_t)sender_idx * (size_t)U + (size_t)u0];
    }
    if (valid_u1) {
        x1 = node_feats[(size_t)sender_idx * (size_t)U + (size_t)u1];
    }

    // -------- 遍历所有 path p --------
    #pragma unroll
    for (int p = 0; p < MAX_D; ++p) {
        if (p >= P) break;
        const int d = dim_list[p];   // ∈ {1,3,5,7}
        const int o = offs[p];

        int       cur_rcv = -1;
        scalar_t  acc0[MAX_D];
        scalar_t  acc1[MAX_D];

        #pragma unroll
        for (int j = 0; j < MAX_D; ++j) {
            acc0[j] = scalar_t(0);
            acc1[j] = scalar_t(0);
        }

        // -------- 遍历此 sender 的所有边 --------
        for (int e = st; e < ed; ++e) {
            const int rcv = receiver[e];

            // run-length by receiver: receiver 变了就 flush
            if (rcv != cur_rcv) {
                if (cur_rcv >= 0) {
                    if (valid_u0) {
                        flush_group_scalar_impl<scalar_t, MAX_D>(
                            cur_rcv, d, o, u0, U, DIM_SUM, acc0, out_nodes);
                    }
                    if (valid_u1) {
                        flush_group_scalar_impl<scalar_t, MAX_D>(
                            cur_rcv, d, o, u1, U, DIM_SUM, acc1, out_nodes);
                    }
                }
            }

            cur_rcv = rcv;

            // ---- 1) 预取 edge_attrs[e, o:o+d] 到 shared memory ----
            const size_t y_base = (size_t)e * (size_t)DIM_SUM + (size_t)o;

            if (lane < d) {
                smem_y[lane] = edge_attrs[y_base + lane];
            }
            __syncwarp();

            // ---- 2) base_u0 / base_u1 ----
            scalar_t base0 = scalar_t(0);
            scalar_t base1 = scalar_t(0);

            if (valid_u0) {
                base0 = x0 *
                    tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u0];
            }
            if (valid_u1) {
                base1 = x1 *
                    tp_weights[((size_t)e * (size_t)P + (size_t)p) * (size_t)U + (size_t)u1];
            }

            // ---- 3) 累加 acc0 / acc1 ----
            if (valid_u0 || valid_u1) {
                #pragma unroll
                for (int j = 0; j < MAX_D; ++j) {
                    if (j >= d) break;
                    const scalar_t yv = smem_y[j];
                    if (valid_u0) {
                        acc0[j] += yv * base0;
                    }
                    if (valid_u1) {
                        acc1[j] += yv * base1;
                    }
                }
            }

            __syncwarp();
        } // e-loop

        // flush 最后一段 run
        if (cur_rcv >= 0) {
            if (valid_u0) {
                flush_group_scalar_impl<scalar_t, MAX_D>(
                    cur_rcv, d, o, u0, U, DIM_SUM, acc0, out_nodes);
            }
            if (valid_u1) {
                flush_group_scalar_impl<scalar_t, MAX_D>(
                    cur_rcv, d, o, u1, U, DIM_SUM, acc1, out_nodes);
            }
        }

    } // p-loop
}


template<typename scalar_t>
void fused_mp_launch_t(
    const scalar_t* node_feats,     // [N, U]
    const scalar_t* edge_attrs,     // [E, DIM_SUM]
    const scalar_t* tp_weights,     // [E, P, U]
    const int32_t* sender,          // [E]
    const int32_t* receiver,        // [E]
    const int32_t* start_idx,       // [N]
    const int32_t* end_idx,         // [N]
    const int32_t* dim_list,        // [P]
    const int32_t* offs,            // [P]
    int32_t N, int32_t E, int32_t U, int32_t P, int32_t DIM_SUM,
    scalar_t* out_nodes,            // [N, DIM_SUM, U]
    cudaStream_t stream,
    bool receiver_major = false
) {
    const int warps_per_block   = 4; // Tuning parameter
    const int TileU             = 64;
    const int num_u_groups      = (U + TileU - 1) / TileU;
    const int total_warps       = N * num_u_groups;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int blocks            = (total_warps + warps_per_block - 1) / warps_per_block;
    constexpr int MAX_D = 8;

    size_t smem_bytes = warps_per_block * MAX_D * sizeof(scalar_t);

    if (receiver_major) {
        fused_mp_warp_receiver_major_allpaths<TileU, MAX_D, scalar_t>
            <<<blocks, threads_per_block, 0, stream>>>(
                node_feats, edge_attrs, tp_weights,
                sender, receiver, start_idx, end_idx,
                dim_list, offs, N, E, U, P, DIM_SUM, out_nodes);
    } else {
        fused_mp_warp_sender_major_allpaths_v3<TileU, MAX_D, scalar_t>
            <<<blocks, threads_per_block, smem_bytes, stream>>>(
                node_feats, edge_attrs, tp_weights,
                sender, receiver, start_idx, end_idx,
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
    torch::Tensor node_feats,   // [N,U], float32/float64, cuda, contiguous
    torch::Tensor edge_attrs,   // [E,DIM_SUM], same dtype
    torch::Tensor tp_weights,   // [E,P,U], same dtype
    torch::Tensor sender,       // [E], int32, cuda, contiguous
    torch::Tensor receiver,     // [E], int32, cuda, contiguous
    torch::Tensor dim_list,     // [P], int32, cuda/CPU
    torch::Tensor offs,         // [P], int32, cuda/CPU
    bool receiver_major = false
){
    TORCH_CHECK(node_feats.is_cuda() && edge_attrs.is_cuda() &&
                tp_weights.is_cuda() && sender.is_cuda() && receiver.is_cuda(),
                "All main tensors must be CUDA");

    auto scalar_type = node_feats.scalar_type();
    TORCH_CHECK(
        (scalar_type == at::kDouble) || (scalar_type == at::kFloat),
        "node_feats must be float32 or float64");
    TORCH_CHECK(edge_attrs.scalar_type() == scalar_type,
                "edge_attrs must have same dtype as node_feats");
    TORCH_CHECK(tp_weights.scalar_type() == scalar_type,
                "tp_weights must have same dtype as node_feats");

    TORCH_CHECK(dim_list.scalar_type() == at::kInt, "dim_list must be int32");
    TORCH_CHECK(offs.scalar_type() == at::kInt, "offs must be int32");

    auto node_feats_c = node_feats.contiguous();
    auto edge_attrs_c = edge_attrs.contiguous();
    auto tp_weights_c = tp_weights.contiguous();

    if (!dim_list.is_cuda()) dim_list = dim_list.to(node_feats.device(), /*non_blocking=*/true);
    if (!offs.is_cuda())     offs     = offs.to(node_feats.device(), /*non_blocking=*/true);

    const auto N       = static_cast<int32_t>(node_feats_c.size(0));
    const auto U       = static_cast<int32_t>(node_feats_c.size(1));
    const auto E       = static_cast<int32_t>(edge_attrs_c.size(0));
    const auto DIM_SUM = static_cast<int32_t>(edge_attrs_c.size(1));
    const auto P       = static_cast<int32_t>(offs.size(0));

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

    auto opts_i32   = torch::TensorOptions().dtype(at::kInt).device(node_feats.device());
    auto start_idx  = torch::empty({N}, opts_i32);
    auto end_idx    = torch::empty({N}, opts_i32);

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

    if (scalar_type == at::kDouble) {
        fused_mp_launch_t<double>(
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
    } else { // float32
        fused_mp_launch_t<float>(
            node_feats_c.data_ptr<float>(),
            edge_attrs_c.data_ptr<float>(),
            tp_weights_c.data_ptr<float>(),
            sender_i32.data_ptr<int32_t>(),
            receiver_i32.data_ptr<int32_t>(),
            start_idx.data_ptr<int32_t>(),
            end_idx.data_ptr<int32_t>(),
            dim_list.data_ptr<int32_t>(),
            offs.data_ptr<int32_t>(),
            N, E, U, P, DIM_SUM,
            out_nodes.data_ptr<float>(),
            stream,
            receiver_major
        );
    }

    out_nodes = out_nodes.view({N, DIM_SUM * U});
    return {out_nodes, start_idx, end_idx};
}


TORCH_LIBRARY(fused_mp_fwd, m)
{
    m.def("forward", &fused_mp_forward);
}