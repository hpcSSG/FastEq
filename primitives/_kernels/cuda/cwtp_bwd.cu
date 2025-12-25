#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_pipeline_primitives.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#include <vector>
#include <algorithm>
#include <cstdint>

#define CUDA_CHECK(expr)                                                   \
    do {                                                                   \
        cudaError_t _err = (expr);                                         \
        TORCH_CHECK(_err == cudaSuccess,                                   \
                    "CUDA error: ", cudaGetErrorString(_err),              \
                    " (error code ", static_cast<int>(_err), ")");         \
    } while (0)


// 对齐的 async copy（4/8/16 字节）
template<typename T>
__device__ __forceinline__ void async_copy_one(T* dst, const T* src) {
    if constexpr (sizeof(T) == 4) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src)
        );
    } else if constexpr (sizeof(T) == 8) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src)
        );
    } else if constexpr (sizeof(T) == 2) {
        // fp16/bf16: 拷贝2字节
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 2;\n"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src)
        );
    }
}

// cp.async commit and wait
__device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

template<int N>
__device__ __forceinline__ void async_wait_prior() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}


/*
constexpr int P  = 4;    // paths
constexpr int KS = 16;   // dim_sum = 16
constexpr int U_FIXED = 96; // U=96, 可被4整除
constexpr int WARP_SIZE = 32;

// 计算 2D 网格下的起始 b 与步长
__device__ __forceinline__ int grid_b0() {
    return blockIdx.x + blockIdx.y * gridDim.x;
}
__device__ __forceinline__ int grid_bstride() {
    return gridDim.x * gridDim.y;
}

// 支持 B>65535 的 2D-grid + grid-stride 版本
__global__ void bwd_kernel_vec4_warp32_grouped_db_bstride(
    const double* __restrict__ grad_out, // [B, KS, U]
    const double* __restrict__ x,        // [B, U]
    const double* __restrict__ y,        // [B, KS]
    const double* __restrict__ w,        // [B, P, U]
    const double* __restrict__ b_buf,    // [B, P, U]
    double* __restrict__ gx,             // [B, U]
    double* __restrict__ gy,             // [B, KS]
    double* __restrict__ gw,             // [B, P, U]
    int B, int U)
{
    const int lane  = threadIdx.x & 31;     // 单 warp
    const int slots = U / 4;                // 96/4=24
    if (lane >= slots) return;              // 多余 lane 直接退出

    // 2D grid + stride 遍历 b
    for (int b = grid_b0(); b < B; b += grid_bstride()) {

        // y[b,:] 放 shared（128B）
        __shared__ double y_sh[KS];
        if (lane < KS) y_sh[lane] = y[size_t(b)*KS + lane];
        __syncwarp();

        const int u = lane * 4;
        const size_t off_u     = size_t(u);
        const size_t off_bu    = size_t(b) * U + off_u;
        const size_t off_bk0   = size_t(b) * KS * U;

        // 只读一次 x4
        const double4 x4 = *reinterpret_cast<const double4*>(&x[off_bu]);

        // 必要累加器
        double4 gx4 = make_double4(0,0,0,0);
        double4 gw0 = make_double4(0,0,0,0);
        double4 gw1 = make_double4(0,0,0,0);
        double4 gw2 = make_double4(0,0,0,0);
        double4 gw3 = make_double4(0,0,0,0);

        // ---- path 0: k={0}（完全展开）----
        {
            const size_t off_pw = (size_t(b)*P + 0) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            const int k = 0;
            const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k)*U + off_u]);
            const double   yk = y_sh[k];

            // gy 现场 warp 归约并写回
            double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[size_t(b)*KS + k] = v;

            const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
            gw0.x += t0*x4.x; gw0.y += t1*x4.y; gw0.z += t2*x4.z; gw0.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
        }

        // ---- path 1: k={1,2,3}（短段：展开）----
        {
            const size_t off_pw = (size_t(b)*P + 1) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            #pragma unroll
            for (int k = 1; k <= 3; ++k) {
                const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k)*U + off_u]);
                const double   yk = y_sh[k];

                double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
                unsigned mask = 0xffffffffu;
                v += __shfl_down_sync(mask, v, 16);
                v += __shfl_down_sync(mask, v, 8);
                v += __shfl_down_sync(mask, v, 4);
                v += __shfl_down_sync(mask, v, 2);
                v += __shfl_down_sync(mask, v, 1);
                if (lane == 0) gy[size_t(b)*KS + k] = v;

                const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
                gw1.x += t0*x4.x; gw1.y += t1*x4.y; gw1.z += t2*x4.z; gw1.w += t3*x4.w;
                gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
            }
        }

        // ---- path 2: k={4..8}（长段：双缓冲 + 禁止展开）----
        {
            const size_t off_pw = (size_t(b)*P + 2) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            double4 g4_cur = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(4)*U + off_u]);
            double  yk_cur = y_sh[4];

            #pragma unroll 1
            for (int k = 4; k <= 8; ++k) {
                double4 g4_next; double yk_next = 0.0;
                if (k < 8) {
                    g4_next = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k+1)*U + off_u]);
                    yk_next = y_sh[k+1];
                }

                double v = g4_cur.x*bp.x + g4_cur.y*bp.y + g4_cur.z*bp.z + g4_cur.w*bp.w;
                unsigned mask = 0xffffffffu;
                v += __shfl_down_sync(mask, v, 16);
                v += __shfl_down_sync(mask, v, 8);
                v += __shfl_down_sync(mask, v, 4);
                v += __shfl_down_sync(mask, v, 2);
                v += __shfl_down_sync(mask, v, 1);
                if (lane == 0) gy[size_t(b)*KS + k] = v;

                const double t0 = g4_cur.x * yk_cur, t1 = g4_cur.y * yk_cur;
                const double t2 = g4_cur.z * yk_cur, t3 = g4_cur.w * yk_cur;
                gw2.x += t0*x4.x; gw2.y += t1*x4.y; gw2.z += t2*x4.z; gw2.w += t3*x4.w;
                gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;

                g4_cur = g4_next; yk_cur = yk_next;
            }
        }

        // ---- path 3: k={9..15}（长段：双缓冲 + 禁止展开）----
        {
            const size_t off_pw = (size_t(b)*P + 3) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            double4 g4_cur = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(9)*U + off_u]);
            double  yk_cur = y_sh[9];

            #pragma unroll 1
            for (int k = 9; k <= 15; ++k) {
                double4 g4_next; double yk_next = 0.0;
                if (k < 15) {
                    g4_next = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k+1)*U + off_u]);
                    yk_next = y_sh[k+1];
                }

                double v = g4_cur.x*bp.x + g4_cur.y*bp.y + g4_cur.z*bp.z + g4_cur.w*bp.w;
                unsigned mask = 0xffffffffu;
                v += __shfl_down_sync(mask, v, 16);
                v += __shfl_down_sync(mask, v, 8);
                v += __shfl_down_sync(mask, v, 4);
                v += __shfl_down_sync(mask, v, 2);
                v += __shfl_down_sync(mask, v, 1);
                if (lane == 0) gy[size_t(b)*KS + k] = v;

                const double t0 = g4_cur.x * yk_cur, t1 = g4_cur.y * yk_cur;
                const double t2 = g4_cur.z * yk_cur, t3 = g4_cur.w * yk_cur;
                gw3.x += t0*x4.x; gw3.y += t1*x4.y; gw3.z += t2*x4.z; gw3.w += t3*x4.w;
                gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;

                g4_cur = g4_next; yk_cur = yk_next;
            }
        }

        // 向量写回
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 0)*U + off_u]) = gw0;
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 1)*U + off_u]) = gw1;
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 2)*U + off_u]) = gw2;
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 3)*U + off_u]) = gw3;
        *reinterpret_cast<double4*>(&gx[off_bu])                      = gx4;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cwtp_small_backward(
    const at::Tensor& grad_out, // [B, KS, U]
    const at::Tensor& x,        // [B, U]
    const at::Tensor& y,        // [B, KS]
    const at::Tensor& w,        // [B, P, U]
    const at::Tensor& b_buf     // [B, P, U]
){
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && y.is_cuda() && w.is_cuda() && b_buf.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(grad_out.scalar_type()==at::kDouble && x.scalar_type()==at::kDouble &&
                y.scalar_type()==at::kDouble && w.scalar_type()==at::kDouble && b_buf.scalar_type()==at::kDouble,
                "expect double dtype");
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() && y.is_contiguous()
                && w.is_contiguous() && b_buf.is_contiguous(), "expect contiguous");

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);

    TORCH_CHECK(U == U_FIXED, "U must be 96");
    TORCH_CHECK(grad_out.sizes() == at::IntArrayRef({B, KS * U}), "grad_out must be [B,16,96]");
    TORCH_CHECK(w.sizes()        == at::IntArrayRef({B, P * U}),   "w must be [B,4,96]");
    TORCH_CHECK(b_buf.sizes()    == at::IntArrayRef({B, P, U}),   "b_buf must be [B,4,96]");

    auto gx = at::empty_like(x);
    auto gy = at::empty_like(y);
    auto gw = at::empty_like(w);

    // 计算 2D 网格：grid.x <= 65535, grid.y <= 65535；多余用 stride 覆盖
    const int max_xy = 65535;
    int gx_dim = (B > max_xy) ? max_xy : static_cast<int>(B);
    int gy_dim = static_cast<int>((B + gx_dim - 1) / gx_dim);
    if (gy_dim > max_xy) gy_dim = max_xy; // 极大 B 时由 stride 继续覆盖

    dim3 grid(gx_dim, gy_dim); // 最多 ≈ 4.29e9 blocks 覆盖能力
    dim3 block(32);            // 单 warp

    bwd_kernel_vec4_warp32_grouped_db_bstride<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<double>(),
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        b_buf.data_ptr<double>(),
        gx.data_ptr<double>(),
        gy.data_ptr<double>(),
        gw.data_ptr<double>(),
        (int)B, (int)U
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, gy, gw};
}
*/

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v, unsigned mask) {
    // 默认 32-lane warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
    unsigned mask = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(mask, v, d);
    return v;
}

// 1. 分组规约减少global write back & 2. warp reduce 减少原子写操作
template <typename scalar_t, int MAX_K_DIM>
__global__ void tp_channel_wise_sparse_groupk_warpreduce_bwd_kernel(
    const scalar_t* __restrict__ x_uv,          // [Z, UV_TOTAL]
    const scalar_t* __restrict__ x_iu,          // [Z, IU_TOTAL]
    const scalar_t* __restrict__ x_jv,          // [Z, JV_TOTAL]

    const int32_t* __restrict__ path_indices,   // [num_paths, 4] : (uv_idx, iu_idx, jv_idx, kv_idx)
    const int32_t* __restrict__ k_dims,         // [num_paths]
    const int32_t* __restrict__ iu_seg_offsets, // [iu_seg_count]
    const int32_t* __restrict__ jv_seg_offsets, // [jv_seg_count]
    const int32_t* __restrict__ kv_k_offsets,   // [kv_seg_count]

    const int32_t* __restrict__ nnz_per_path,   // [num_paths]
    const int32_t* __restrict__ nnz_offsets,    // [num_paths]

    const int32_t* __restrict__ nnz_k_offsets,  // [num_paths * MAX_K_DIM]
    const int32_t* __restrict__ nnz_k_counts,   // [num_paths * MAX_K_DIM]

    // 稀疏 CG 系数（global memory）
    const uint8_t* __restrict__ cg_i_all,       // [nnz_total]
    const uint8_t* __restrict__ cg_j_all,       // [nnz_total]
    const scalar_t* __restrict__ cg_val_all,    // [nnz_total]

    // grad_out: dL/d out
    const scalar_t* __restrict__ grad_out,      // [Z, K_TOTAL, U, V]

    // 输出梯度
    scalar_t* __restrict__ grad_x_uv,           // [Z, UV_TOTAL]
    scalar_t* __restrict__ grad_x_iu,           // [Z, IU_TOTAL]
    scalar_t* __restrict__ grad_x_jv,           // [Z, JV_TOTAL]

    int Z,
    int UV_TOTAL,
    int IU_TOTAL,
    int JV_TOTAL,
    int K_TOTAL,
    int U,
    int V,
    int num_paths
) {
    int z = blockIdx.x;   // 一个 block 一个 batch
    if (z >= Z) return;

    int u = threadIdx.x;  // 每个 thread 一个 u
    if (u >= U) return;

    extern __shared__ unsigned char smem_raw[];
    scalar_t* s_iu = reinterpret_cast<scalar_t*>(smem_raw);          // [IU_TOTAL]
    scalar_t* s_jv = s_iu + IU_TOTAL;                                // [JV_TOTAL]

    const scalar_t* x_iu_z = x_iu + (size_t)z * IU_TOTAL;
    const scalar_t* x_jv_z = x_jv + (size_t)z * JV_TOTAL;

    int threads_in_block = blockDim.x;

    // 1. 把 x_iu[z,:], x_jv[z,:] 搬到 shared
    for (int idx = u; idx < IU_TOTAL; idx += threads_in_block) {
        s_iu[idx] = x_iu_z[idx];
    }
    for (int idx = u; idx < JV_TOTAL; idx += threads_in_block) {
        s_jv[idx] = x_jv_z[idx];
    }
    __syncthreads();

    const scalar_t* x_uv_z     = x_uv     + (size_t)z * UV_TOTAL;
    const scalar_t* grad_out_z = grad_out + (size_t)z * (K_TOTAL * U * V);

    scalar_t* grad_x_uv_z = grad_x_uv + (size_t)z * UV_TOTAL;
    scalar_t* grad_x_iu_z = grad_x_iu + (size_t)z * IU_TOTAL;
    scalar_t* grad_x_jv_z = grad_x_jv + (size_t)z * JV_TOTAL;

    // 2. 遍历所有 path
    for (int p = 0; p < num_paths; ++p) {
        int uv_idx = path_indices[p * 4 + 0];
        int iu_idx = path_indices[p * 4 + 1];
        int jv_idx = path_indices[p * 4 + 2];
        int kv_idx = path_indices[p * 4 + 3];

        int k_dim   = k_dims[p];
        int nnz     = nnz_per_path[p];
        int nnz_off = nnz_offsets[p];

        if (k_dim <= 0 || nnz <= 0) {
            continue;
        }
        if (k_dim > MAX_K_DIM) {
            return;
        }

        int uv_base = uv_idx * (U * V);         // 该 uv seg 在 x_uv[z,:] 中的起点
        int iu_base = iu_seg_offsets[iu_idx];   // 该 iu seg 在 x_iu[z,:] 中的起点
        int jv_base = jv_seg_offsets[jv_idx];   // 该 jv seg 在 x_jv[z,:] 中的起点
        int k_base  = kv_k_offsets[kv_idx];     // 该 kv seg 在 K 维的起点

        for (int v_idx = 0; v_idx < V; ++v_idx) {
            int xuv_index = uv_base + u * V + v_idx;
            scalar_t xuv_uv = x_uv_z[xuv_index];

            // 按 k 分组
            for (int k_local = 0; k_local < k_dim; ++k_local) {
                int meta_idx    = p * MAX_K_DIM + k_local;
                int local_off   = nnz_k_offsets[meta_idx];
                int local_count = nnz_k_counts[meta_idx];

                if (local_count <= 0)
                    continue;

                int global_k  = k_base + k_local;
                int out_index = (global_k * U + u) * V + v_idx;
                scalar_t g = grad_out_z[out_index];  // dL/d out(z,global_k,u,v)

                if (g == static_cast<scalar_t>(0)) {
                    continue;
                }

                // 用于 d(x_uv) 的 sum_{(i,j) in nnz} c * x_iu * x_jv
                scalar_t sum_all_ij = static_cast<scalar_t>(0);

                // 遍历这个 k 的所有 nnz
                for (int tt = 0; tt < local_count; ++tt) {
                    int t   = local_off + tt;
                    int idx = nnz_off + t; // global nnz index

                    int i = static_cast<int>(cg_i_all[idx]);
                    int j = static_cast<int>(cg_j_all[idx]);
                    scalar_t c = cg_val_all[idx];

                    // x_iu[z, iu_seg][i, u]
                    int xiu_index = iu_base + i * U + u;
                    scalar_t xiu_iu = s_iu[xiu_index];

                    // x_jv[z, jv_seg][j, v]
                    int xjv_index = jv_base + j * V + v_idx;
                    scalar_t xjv_jv = s_jv[xjv_index];

                    // --- dL/d x_iu[i,u] += g * c * x_jv * x_uv ---
                    scalar_t d_xiu = g * c * xjv_jv * xuv_uv;
                    grad_x_iu_z[xiu_index] += d_xiu;

                    // --- dL/d x_jv[j,v] += g * c * x_iu * x_uv ---
                    scalar_t d_xjv = g * c * xiu_iu * xuv_uv;
                    // warp 内归约：同一个 (p,k_local,v_idx,tt) 的 xjv_index 对所有 u 都相同
                    unsigned mask = __activemask();                 // 当前 warp 活跃线程掩码
                    scalar_t warp_sum = warp_reduce_sum(d_xjv, mask);
                    if ((threadIdx.x & 31) == 0) {                  // lane0
                        atomicAdd(&grad_x_jv_z[xjv_index], warp_sum);
                    }

                    // --- 对 x_uv 的贡献项累加 ---
                    sum_all_ij += c * xiu_iu * xjv_jv;
                }

                // --- dL/d x_uv[u,v] += g * sum_{(i,j)} c * x_iu * x_jv ---
                scalar_t d_xuv = g * sum_all_ij;
                grad_x_uv_z[xuv_index] += d_xuv;
            }
        }
    }
}

std::vector<torch::Tensor> tp_channel_wise_groupk_bwd_launch(
    torch::Tensor x_uv,            // [Z, UV_TOTAL]
    torch::Tensor x_iu,            // [Z, IU_TOTAL]
    torch::Tensor x_jv,            // [Z, JV_TOTAL]

    torch::Tensor path_indices,    // [num_paths, 4], int32
    torch::Tensor k_dims,          // [num_paths], int32
    torch::Tensor iu_seg_offsets,  // [iu_seg_count], int32
    torch::Tensor jv_seg_offsets,  // [jv_seg_count], int32
    torch::Tensor kv_k_offsets,    // [kv_seg_count], int32

    // 稀疏 CG
    torch::Tensor nnz_per_path,    // [num_paths], int32
    torch::Tensor nnz_offsets,     // [num_paths], int32
    torch::Tensor nnz_k_offsets,   // [num_paths * MAX_K_DIM], int32
    torch::Tensor nnz_k_counts,    // [num_paths * MAX_K_DIM], int32
    torch::Tensor cg_i_all,        // [nnz_total], uint8
    torch::Tensor cg_j_all,        // [nnz_total], uint8
    torch::Tensor cg_val_all,      // [nnz_total], same dtype as x_uv

    torch::Tensor grad_out,        // [Z, K_TOTAL, U, V]

    const int64_t U,
    const int64_t V,
    const int64_t K_TOTAL
) {
    TORCH_CHECK(x_uv.is_cuda(), "x_uv must be CUDA");
    TORCH_CHECK(x_iu.is_cuda(), "x_iu must be CUDA");
    TORCH_CHECK(x_jv.is_cuda(), "x_jv must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(cg_val_all.is_cuda(), "cg_val_all must be CUDA");

    x_uv = x_uv.contiguous();
    x_iu = x_iu.contiguous();
    x_jv = x_jv.contiguous();
    path_indices   = path_indices.contiguous();
    k_dims         = k_dims.contiguous();
    iu_seg_offsets = iu_seg_offsets.contiguous();
    jv_seg_offsets = jv_seg_offsets.contiguous();
    kv_k_offsets   = kv_k_offsets.contiguous();
    nnz_per_path   = nnz_per_path.contiguous();
    nnz_offsets    = nnz_offsets.contiguous();
    nnz_k_offsets  = nnz_k_offsets.contiguous();
    nnz_k_counts   = nnz_k_counts.contiguous();
    cg_i_all       = cg_i_all.contiguous();
    cg_j_all       = cg_j_all.contiguous();
    cg_val_all     = cg_val_all.contiguous();

    auto Z        = x_uv.size(0);
    auto UV_TOTAL = x_uv.size(1);
    auto IU_TOTAL = x_iu.size(1);
    auto JV_TOTAL = x_jv.size(1);

    grad_out = grad_out.view({Z, K_TOTAL, U, V}).contiguous();

    TORCH_CHECK(x_iu.size(0) == Z && x_jv.size(0) == Z, "batch dim mismatch");
    TORCH_CHECK(path_indices.dim() == 2 && path_indices.size(1) == 4,
                "path_indices must be [num_paths,4]");
    TORCH_CHECK(grad_out.size(0) == Z &&
                grad_out.size(1) == K_TOTAL &&
                grad_out.size(2) == U &&
                grad_out.size(3) == V,
                "grad_out shape mismatch");

    int num_paths = path_indices.size(0);

    constexpr int MAX_K_DIM = 8;
    auto grad_x_uv = torch::zeros_like(x_uv);
    auto grad_x_iu = torch::zeros_like(x_iu);
    auto grad_x_jv = torch::zeros_like(x_jv);

    int threads = static_cast<int>(U);
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;

    int blocks = static_cast<int>(Z);
    size_t smem_bytes = (IU_TOTAL + JV_TOTAL) * x_uv.element_size();
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(),
                               "tp_channel_wise_sparse_groupk_warpreduce_bwd_kernel",
                               [&] {
        tp_channel_wise_sparse_groupk_warpreduce_bwd_kernel<scalar_t, MAX_K_DIM>
            <<<blocks, threads, smem_bytes, stream>>>(
                x_uv.data_ptr<scalar_t>(),
                x_iu.data_ptr<scalar_t>(),
                x_jv.data_ptr<scalar_t>(),
                path_indices.data_ptr<int32_t>(),
                k_dims.data_ptr<int32_t>(),
                iu_seg_offsets.data_ptr<int32_t>(),
                jv_seg_offsets.data_ptr<int32_t>(),
                kv_k_offsets.data_ptr<int32_t>(),
                nnz_per_path.data_ptr<int32_t>(),
                nnz_offsets.data_ptr<int32_t>(),
                nnz_k_offsets.data_ptr<int32_t>(),
                nnz_k_counts.data_ptr<int32_t>(),
                cg_i_all.data_ptr<uint8_t>(),
                cg_j_all.data_ptr<uint8_t>(),
                cg_val_all.data_ptr<scalar_t>(),
                grad_out.data_ptr<scalar_t>(),
                grad_x_uv.data_ptr<scalar_t>(),
                grad_x_iu.data_ptr<scalar_t>(),
                grad_x_jv.data_ptr<scalar_t>(),
                (int)Z,
                (int)UV_TOTAL,
                (int)IU_TOTAL,
                (int)JV_TOTAL,
                (int)K_TOTAL,
                (int)U,
                (int)V,
                num_paths
            );
        CUDA_CHECK(cudaGetLastError());
    });

    return {grad_x_uv, grad_x_iu, grad_x_jv};
}

template <typename T>
__device__ __forceinline__ T ld_g(const T* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}


template<int P, int UV, int IU, int JV, int KV, int I, int J, int K, int JB, typename T>
__device__ __forceinline__ void tp17_path_eval_sharedc(
    int z, int u, int lane,
    const T* __restrict__ grad_out,  // [Z, K_TOTAL, U]
    const T* __restrict__ x_uv,      // [Z, UV_TOTAL]
    const T* __restrict__ x_iu,      // [Z, IU_TOTAL]
    const T* __restrict__ x_jv,      // [Z, JV_TOTAL]
    const T* __restrict__ c_s,       // shared c base (C_TOTAL)
    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ smem_j,          // shared j partial [U*17]
    int STRIDE,                      // 17

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets, // [18]

    int32_t U_runtime,
    int32_t K_TOTAL,
    int32_t UV_TOTAL, int32_t IU_TOTAL, int32_t JV_TOTAL
) {
  int uv_base = (int)uv_seg_offsets[UV];
  int iu_base = (int)iu_seg_offsets[IU];
  int jv_base = (int)jv_seg_offsets[JV];
  int k_base  = (int)kv_k_offsets[KV];

  const T* c_ptr = c_s + (int)c_offsets[P]; // now shared

  // xuv
  T xuv = ld_g(x_uv + (int64_t)z * UV_TOTAL + (uv_base + u));

  // xj[J] : warp broadcast
  T xj[J];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    T v = (lane == 0) ? ld_g(x_jv + (int64_t)z * JV_TOTAL + (jv_base + j)) : (T)0;
    xj[j] = __shfl_sync(0xffffffff, v, 0);
  }

  // xiu[I]
  T xiu[I];
#pragma unroll
  for (int i = 0; i < I; ++i) {
    xiu[i] = ld_g(x_iu + (int64_t)z * IU_TOTAL + (iu_base + i * U_runtime + u));
  }

  // accum uv/iu
  T acc_uv = (T)0;
  T acc_iu[I];
#pragma unroll
  for (int i = 0; i < I; ++i) acc_iu[i] = (T)0;

  // jv partial: register accumulate, then one shared add at end
  T jtmp[J];
#pragma unroll
  for (int j = 0; j < J; ++j) jtmp[j] = (T)0;

#pragma unroll
  for (int kk = 0; kk < K; ++kk) {
    T go = ld_g(grad_out + (((int64_t)z * K_TOTAL + (k_base + kk)) * (int64_t)U_runtime + u));;

    // uv/iu
#pragma unroll
    for (int i = 0; i < I; ++i) {
      T s = (T)0;
#pragma unroll
      for (int j = 0; j < J; ++j) {
        T c = c_ptr[((i * J + j) * K + kk)];
        s = fma(c, xj[j], s);
      }
      acc_uv    = fma(go, s * xiu[i], acc_uv);
      acc_iu[i] = fma(go, xuv * s,    acc_iu[i]);
    }

    // jv partial
#pragma unroll
    for (int j = 0; j < J; ++j) {
      T tj = (T)0;
#pragma unroll
      for (int i = 0; i < I; ++i) {
        T c = c_ptr[((i * J + j) * K + kk)];
        tj = fma(c, xiu[i], tj);
      }
      jtmp[j] = fma(go, xuv * tj, jtmp[j]);
    }
  }

  // write back uv/iu
  grad_x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)] += acc_uv;
#pragma unroll
  for (int i = 0; i < I; ++i) {
    grad_x_iu[(int64_t)z * IU_TOTAL + (iu_base + i * U_runtime + u)] += acc_iu[i];
  }

  // one-time shared add for jv partial
#pragma unroll
  for (int j = 0; j < J; ++j) {
    int jj = JB + j;
    smem_j[u * STRIDE + jj] += jtmp[j];
  }
}


template<typename T, int JB>
__device__ __forceinline__ void tp17_eval_single(
    int z, int u,
    const T* __restrict__ grad_out,
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const T* __restrict__ smem_c,
    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ smem_j, int STRIDE,
    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,
    int U, int K_TOTAL, int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int iu_base = (int)iu_seg_offsets[IU_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)];
  const T xiu = x_iu[(int64_t)z * IU_TOTAL + (iu_base + 0*U + u)];
  const T xjv = x_jv[(int64_t)z * JV_TOTAL + (jv_base + 0)];
  const T c   = smem_c[c_base + 0];
  const T gk  = grad_out[(int64_t)z * (int64_t)K_TOTAL * U + (int64_t)(k_base + 0) * U + u];

  const T gc  = gk * c;

  grad_x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)] += gc * xiu * xjv;
  grad_x_iu[(int64_t)z * IU_TOTAL + (iu_base + 0*U + u)] += gc * xuv * xjv;
  smem_j[u * STRIDE + JB + 0] += gc * xuv * xiu;
}

template<typename T, int D, int JB>
__device__ __forceinline__ void tp17_eval_diag_jk_i0(
    int z, int u,
    const T* __restrict__ grad_out,
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const T* __restrict__ smem_c,
    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ smem_j, int STRIDE,
    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,
    int U, int K_TOTAL, int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int iu_base = (int)iu_seg_offsets[IU_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)];
  const T xiu = x_iu[(int64_t)z * IU_TOTAL + (iu_base + 0*U + u)];

#pragma unroll
  for (int t = 0; t < D; ++t) {
    const T xjv = x_jv[(int64_t)z * JV_TOTAL + (jv_base + t)];
    const T c   = smem_c[c_base + t * (D + 1)];         // (0,t,t)
    const T gk  = grad_out[(int64_t)z * (int64_t)K_TOTAL * U + (int64_t)(k_base + t) * U + u];

    const T gc = gk * c;
    grad_x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)] += gc * xiu * xjv;
    grad_x_iu[(int64_t)z * IU_TOTAL + (iu_base + 0*U + u)] += gc * xuv * xjv;
    smem_j[u * STRIDE + (JB + t)] += gc * xuv * xiu;
  }
}

template<typename T, int D, int JB>
__device__ __forceinline__ void tp17_eval_diag_ik_j0(
    int z, int u,
    const T* __restrict__ grad_out,
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const T* __restrict__ smem_c,
    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ smem_j, int STRIDE,
    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,
    int U, int K_TOTAL, int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int iu_base = (int)iu_seg_offsets[IU_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)];
  const T xjv = x_jv[(int64_t)z * JV_TOTAL + (jv_base + 0)]; // j=0 only

#pragma unroll
  for (int t = 0; t < D; ++t) {
    const T xiu = x_iu[(int64_t)z * IU_TOTAL + (iu_base + t*U + u)];
    const T c   = smem_c[c_base + t * (D + 1)];             // (t,0,t)
    const T gk  = grad_out[(int64_t)z * (int64_t)K_TOTAL * U + (int64_t)(k_base + t) * U + u];

    const T gc = gk * c;
    grad_x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)] += gc * xiu * xjv;
    grad_x_iu[(int64_t)z * IU_TOTAL + (iu_base + t*U + u)] += gc * xuv * xjv;
    smem_j[u * STRIDE + JB + 0] += gc * xuv * xiu;          // j only 0 -> global jj=JB
  }
}

template<typename T, int D, int JB>
__device__ __forceinline__ void tp17_eval_diag_ij_k0(
    int z, int u,
    const T* __restrict__ grad_out,
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const T* __restrict__ smem_c,
    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ smem_j, int STRIDE,
    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,
    int U, int K_TOTAL, int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int iu_base = (int)iu_seg_offsets[IU_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)];
  const T gk  = grad_out[(int64_t)z * (int64_t)K_TOTAL * U + (int64_t)(k_base + 0) * U + u];

#pragma unroll
  for (int t = 0; t < D; ++t) {
    const T xiu = x_iu[(int64_t)z * IU_TOTAL + (iu_base + t*U + u)];
    const T xjv = x_jv[(int64_t)z * JV_TOTAL + (jv_base + t)];
    const T c   = smem_c[c_base + t * (D + 1)];             // (t,t,0) in D×D×1 -> idx=t*(D+1)
    const T gc  = gk * c;

    grad_x_uv[(int64_t)z * UV_TOTAL + (uv_base + u)] += gc * xiu * xjv;
    grad_x_iu[(int64_t)z * IU_TOTAL + (iu_base + t*U + u)] += gc * xuv * xjv;
    smem_j[u * STRIDE + (JB + t)] += gc * xuv * xiu;
  }
}


template <typename T>
__global__ void tp17_bwd_fused_kernel_sharedc(
    const T* __restrict__ grad_out,
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const T* __restrict__ c_all,

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets, // [18]

    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ grad_x_jv,

    int32_t Z, int32_t U,
    int32_t K_TOTAL,
    int32_t UV_TOTAL, int32_t IU_TOTAL, int32_t JV_TOTAL
) {
  int z = (int)blockIdx.x;
  int tid = (int)threadIdx.x;
  if (z >= Z) return;

  bool active = (tid < U);
  int u = tid;

  constexpr int JJ = 16;
  constexpr int PAD = 1;
  constexpr int STRIDE = JJ + PAD; // 17

  int lane = tid & 31;
  int warp = tid >> 5;

  // ---- shared layout: [j-partial][c-total] ----
  extern __shared__ unsigned char smem_raw[];
  T* smem_j = reinterpret_cast<T*>(smem_raw);          // size: U*17
  T* smem_c = smem_j + (int)(U * STRIDE);              // size: c_offsets[17] (==C_TOTAL)

  // 1) init smem_j
  if (active) {
#pragma unroll
    for (int jj = 0; jj < JJ; ++jj) smem_j[u * STRIDE + jj] = (T)0;
  }

  // 2) cooperative load c_all -> smem_c (per block once)
  // C_TOTAL = c_offsets[17]
  int C_TOTAL = (int)c_offsets[17];
  for (int idx = tid; idx < C_TOTAL; idx += blockDim.x) {
    smem_c[idx] = ld_g(c_all + idx);
  }

  __syncthreads();

  // 3) 17 specialized paths (only active threads)
  if (active) {
    // (P, UV, IU, JV, KV, I, J, K, JB)

    // path0: (1,1,1), JB=0, UV=0 IU=0 JV=0 KV=0 P=0
    tp17_eval_single<T, 0>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                            uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                            U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 0,0,0,0, 0);

    // path1: (1,3,3) j=k diag, JB=1, UV=3 IU=0 JV=1 KV=3 P=1
    tp17_eval_diag_jk_i0<T, 3, 1>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 3,0,1,3, 1);

    // path2: (1,5,5) JB=4, UV=8 IU=0 JV=2 KV=8 P=2
    tp17_eval_diag_jk_i0<T, 5, 4>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 8,0,2,8, 2);

    // path3: (1,7,7) JB=9, UV=13 IU=0 JV=3 KV=13 P=3
    tp17_eval_diag_jk_i0<T, 7, 9>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 13,0,3,13, 3);

    // path4: (3,1,3) i=k, j=0, JB=0, UV=4 IU=1 JV=0 KV=4 P=4
    tp17_eval_diag_ik_j0<T, 3, 0>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 4,1,0,4, 4);

    // path5: (3,3,1) i=j, k=0, JB=1, UV=1 IU=1 JV=1 KV=1 P=5
    tp17_eval_diag_ij_k0<T, 3, 1>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 1,1,1,1, 5);

    tp17_path_eval_sharedc< 6,  9,1,1, 9, 3,3,5, 1>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
    
    tp17_path_eval_sharedc< 7,  5,1,2, 5, 3,5,3, 4>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
    

    tp17_path_eval_sharedc< 8, 14,1,2,14, 3,5,7, 4>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);

    tp17_path_eval_sharedc< 9, 10,1,3,10, 3,7,5, 9>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
       
    // path10: (5,1,5) i=k, JB=0, UV=11 IU=2 JV=0 KV=11 P=10
    tp17_eval_diag_ik_j0<T, 5, 0>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 11,2,0,11, 10);
    
    
    tp17_path_eval_sharedc<11,  6,2,1, 6, 5,3,3, 1>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
    
    tp17_path_eval_sharedc<12, 15,2,1,15, 5,3,7, 1>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);

    tp17_eval_diag_ij_k0<T, 5, 4>(z,u,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL, 2,2,2,2, 13);
   
    tp17_path_eval_sharedc<14, 12,2,2,12, 5,5,5, 4>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
    
    tp17_path_eval_sharedc<15,  7,2,3, 7, 5,7,3, 9>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
    
    tp17_path_eval_sharedc<16, 16,2,3,16, 5,7,7, 9>(z,u,lane,grad_out,x_uv,x_iu,x_jv,smem_c,grad_x_uv,grad_x_iu,smem_j,STRIDE,
                                                    uv_seg_offsets,iu_seg_offsets,jv_seg_offsets,kv_k_offsets,c_offsets,
                                                    U,K_TOTAL,UV_TOTAL,IU_TOTAL,JV_TOTAL);
    
    
  }

  __syncthreads();

  // 4) reduce smem_j over u for each jj
  __shared__ T warp_sum[8][JJ + PAD];

#pragma unroll
  for (int jj = 0; jj < JJ; ++jj) {
    T v = (T)0;
    if (active) v = smem_j[u * STRIDE + jj];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) warp_sum[warp][jj] = v;
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int jj = 0; jj < JJ; ++jj) {
      T v = (lane < 8) ? warp_sum[lane][jj] : (T)0;
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
      if (lane == 0) {
        int jv_idx, j_local;
        if (jj == 0) { jv_idx = 0; j_local = 0; }
        else if (jj < 4) { jv_idx = 1; j_local = jj - 1; }
        else if (jj < 9) { jv_idx = 2; j_local = jj - 4; }
        else { jv_idx = 3; j_local = jj - 9; }

        int jv_base = (int)jv_seg_offsets[jv_idx];
        grad_x_jv[(int64_t)z * JV_TOTAL + (int64_t)(jv_base + j_local)] += v;
      }
    }
  }
}

std::vector<torch::Tensor> tp17_bwd_fused(
    torch::Tensor grad_out,      // [Z,K_TOTAL,U]  (V=1)
    torch::Tensor x_uv,          // [Z,UV_TOTAL]
    torch::Tensor x_iu,          // [Z,IU_TOTAL]
    torch::Tensor x_jv,          // [Z,JV_TOTAL]
    torch::Tensor c_all,         // [C_TOTAL]
    torch::Tensor path_indices,  // [17,4] int32
    torch::Tensor uv_seg_offsets,// int32
    torch::Tensor iu_seg_offsets,// int32
    torch::Tensor jv_seg_offsets,// int32
    torch::Tensor kv_k_offsets,  // int32
    torch::Tensor c_offsets,     // [18] int32
    torch::Tensor i_dims,        // [17] int32
    torch::Tensor j_dims,        // [17] int32
    torch::Tensor k_dims,        // [17] int32
    const int64_t K_TOTAL,
    const int64_t U,
    const int64_t V
) {

    x_uv = x_uv.contiguous();
    x_iu = x_iu.contiguous();
    x_jv = x_jv.contiguous();

    const int64_t Z = x_uv.size(0);
    
    grad_out = grad_out.view({Z, K_TOTAL, U*V}).contiguous();

    path_indices = path_indices.contiguous();
    uv_seg_offsets = uv_seg_offsets.contiguous();
    iu_seg_offsets = iu_seg_offsets.contiguous();
    jv_seg_offsets = jv_seg_offsets.contiguous();
    kv_k_offsets = kv_k_offsets.contiguous();

    i_dims = i_dims.contiguous();
    j_dims = j_dims.contiguous();
    k_dims = k_dims.contiguous();

    c_offsets = c_offsets.contiguous();

  TORCH_CHECK(path_indices.scalar_type() == torch::kInt32, "path_indices must be int32");
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must be [Z,K_TOTAL,U] (V=1)");
  TORCH_CHECK(path_indices.size(0) == 17 && path_indices.size(1) == 4, "path_indices must be [17,4]");

  TORCH_CHECK(U == 224, "This fused kernel assumes U=224");
  TORCH_CHECK(x_uv.size(0) == Z && x_iu.size(0) == Z && x_jv.size(0) == Z, "Z mismatch");

  int32_t UV_TOTAL = (int32_t)x_uv.size(1);
  int32_t IU_TOTAL = (int32_t)x_iu.size(1);
  int32_t JV_TOTAL = (int32_t)x_jv.size(1);

  auto grad_x_uv = torch::zeros_like(x_uv);
  auto grad_x_iu = torch::zeros_like(x_iu);
  auto grad_x_jv = torch::zeros_like(x_jv);

  dim3 grid(Z);
  dim3 block(256); 

  
  int C_TOTAL = c_all.size(0); 
  size_t shmem = (size_t)U * (16+1) * (size_t)x_uv.element_size() + (size_t)C_TOTAL *(size_t)x_uv.element_size();

  AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp17_bwd_fused", [&](){
    tp17_bwd_fused_kernel_sharedc<scalar_t><<<grid, block, shmem>>>(
      (const scalar_t*)grad_out.data_ptr<scalar_t>(),
      (const scalar_t*)x_uv.data_ptr<scalar_t>(),
      (const scalar_t*)x_iu.data_ptr<scalar_t>(),
      (const scalar_t*)x_jv.data_ptr<scalar_t>(),
      (const scalar_t*)c_all.data_ptr<scalar_t>(),
      
      uv_seg_offsets.data_ptr<int32_t>(),
      iu_seg_offsets.data_ptr<int32_t>(),
      jv_seg_offsets.data_ptr<int32_t>(),
      kv_k_offsets.data_ptr<int32_t>(),
      c_offsets.data_ptr<int32_t>(),

      (scalar_t*)grad_x_uv.data_ptr<scalar_t>(),
      (scalar_t*)grad_x_iu.data_ptr<scalar_t>(),
      (scalar_t*)grad_x_jv.data_ptr<scalar_t>(),

      Z, U, K_TOTAL, UV_TOTAL, IU_TOTAL, JV_TOTAL
    );
  });

  return {grad_x_uv, grad_x_iu, grad_x_jv};
}


TORCH_LIBRARY(cwtp_bwd, m)
{
    m.def("backward", &tp_channel_wise_groupk_bwd_launch);
}
