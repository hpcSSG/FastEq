#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

// 瓶颈在原子写操作
template <typename scalar_t, int MAX_K_DIM>
__global__ void tp_channel_wise_sparse_groupk_bwd_kernel(
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

        // 在 MACE-OFF 中通常 V=1，但这里保持通用
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

                // 遍历这个 k 的所有 nnz（本 path 内连续）
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
                    atomicAdd(&grad_x_jv_z[xjv_index], d_xjv);

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

std::vector<torch::Tensor> tp_channel_wise_bwd_launch(
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


template<typename T, int MAX_K, int MAX_I, int MAX_J>
__global__ void tp_cwtp_bwd_one_kernel_groupij(
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const T* __restrict__ grad_out,          // [Z,K_TOTAL,U,1] contiguous

    const int32_t* __restrict__ path_indices,// [P,4]
    const int32_t* __restrict__ k_dims,      // [P]
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,

    // group-i
    const uint8_t* __restrict__ cg_j_i,
    const uint8_t* __restrict__ cg_k_i,
    const T*       __restrict__ cg_val_i,
    const int32_t* __restrict__ nnz_offsets_i, // [P]
    const int32_t* __restrict__ nnz_i_offsets, // [P,MAX_I]
    const int32_t* __restrict__ nnz_i_counts,  // [P,MAX_I]

    // group-j
    const uint8_t* __restrict__ cg_i_j,
    const uint8_t* __restrict__ cg_k_j,
    const T*       __restrict__ cg_val_j,
    const int32_t* __restrict__ nnz_offsets_j, // [P]
    const int32_t* __restrict__ nnz_j_offsets, // [P,MAX_J]
    const int32_t* __restrict__ nnz_j_counts,  // [P,MAX_J]

    // NEW: orders
    const int32_t* __restrict__ order_uv,    // [P]
    const int32_t* __restrict__ order_iu,    // [P]
    const int32_t* __restrict__ order_jv,    // [P]

    // outputs (assume zero-inited in launch)
    T* __restrict__ grad_x_uv,
    T* __restrict__ grad_x_iu,
    T* __restrict__ grad_x_jv,

    int Z, int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int K_TOTAL, int U, int V, int P
){
    int z = blockIdx.x;
    int u = threadIdx.x;
    if (z>=Z || u>=U) return;
    // V==1 only
    if (V != 1) return;

    extern __shared__ unsigned char smem[];
    T* s_iu = (T*)smem;
    T* s_jv = s_iu + IU_TOTAL;

    const T* x_iu_z = x_iu + (size_t)z*IU_TOTAL;
    const T* x_jv_z = x_jv + (size_t)z*JV_TOTAL;
    for(int idx=u; idx<IU_TOTAL; idx+=blockDim.x) s_iu[idx]=x_iu_z[idx];
    for(int idx=u; idx<JV_TOTAL; idx+=blockDim.x) s_jv[idx]=x_jv_z[idx];
    __syncthreads();

    const T* x_uv_z = x_uv + (size_t)z*UV_TOTAL;
    const T* go_z   = grad_out + (size_t)z * (size_t)K_TOTAL * (size_t)U; // [K_TOTAL,U]
    T* guv_z = grad_x_uv + (size_t)z*UV_TOTAL;
    T* giu_z = grad_x_iu + (size_t)z*IU_TOTAL;
    T* gjv_z = grad_x_jv + (size_t)z*JV_TOTAL;

    const int lane = threadIdx.x & 31;

    // =========================
    // Pass 1: grad_uv (order_uv)  —— 每个 uv_idx 只写一次（=）
    // =========================
    {
        int cur_uv_idx = -1;
        int cur_uv_base = 0;
        T acc_uv = T(0);
        T xuv_cur = T(0);

        for(int t=0; t<P; ++t){
            int p = order_uv[t];
            int uv_idx = path_indices[p*4 + 0];
            int iu_idx = path_indices[p*4 + 1];
            int jv_idx = path_indices[p*4 + 2];
            int kv_idx = path_indices[p*4 + 3];

            int k_dim = k_dims[p];
            if(k_dim<=0 || k_dim>MAX_K) continue;

            if (uv_idx != cur_uv_idx){
                if (cur_uv_idx != -1){
                    // outputs 是 zero-inited，且每 uv_idx 只出现一次 -> 直接赋值，避免 RMW
                    guv_z[cur_uv_base + u] = acc_uv;
                }
                cur_uv_idx = uv_idx;
                cur_uv_base = uv_idx * (U * V);
                acc_uv = T(0);
                xuv_cur = x_uv_z[cur_uv_base + u]; // V==1
            }

            int iu_base = iu_seg_offsets[iu_idx];
            int jv_base = jv_seg_offsets[jv_idx];
            int k_base  = kv_k_offsets[kv_idx];

            // preload gk (scope 限制生命周期，降寄存器压力)
            T gk[MAX_K];
            #pragma unroll
            for(int kk=0; kk<MAX_K; ++kk) gk[kk]=T(0);
            #pragma unroll
            for(int kk=0; kk<MAX_K; ++kk){
                if(kk<k_dim){
                    gk[kk] = go_z[(k_base + kk)*U + u];
                }
            }

            // 用 group-i 表来扫 nnz（也行：这里我们只需要 (j,k,c)，不需要 i）
            int base_i = nnz_offsets_i[p];
            #pragma unroll
            for(int ii=0; ii<MAX_I; ++ii){
                int off = nnz_i_offsets[p*MAX_I + ii];
                int cnt = nnz_i_counts [p*MAX_I + ii];
                if (cnt<=0) continue;

                // xiu 对该 ii 固定，提到外面，减少 shared load
                T xiu = s_iu[iu_base + ii*U + u];

                for(int s=0; s<cnt; ++s){
                    int idx = base_i + off + s;
                    int j   = (int)cg_j_i[idx];
                    int kk  = (int)cg_k_i[idx];
                    T   c   = cg_val_i[idx];

                    T xjv = s_jv[jv_base + j]; // V==1
                    // grad_uv += gk* c * xiu * xjv
                    acc_uv += gk[kk] * c * xiu * xjv;
                }
            }
        }

        if (cur_uv_idx != -1){
            guv_z[cur_uv_base + u] = acc_uv;
        }
    }

    // =========================
    // Pass 2: grad_iu (order_iu) —— 每个 iu_idx 只 flush 一次（每 i 一次，=）
    // =========================
    {
        int cur_iu_idx = -1;
        int cur_iu_base = 0;
        T acc_iu[MAX_I];
        #pragma unroll
        for(int ii=0; ii<MAX_I; ++ii) acc_iu[ii]=T(0);

        for(int t=0; t<P; ++t){
            int p = order_iu[t];
            int uv_idx = path_indices[p*4 + 0];
            int iu_idx = path_indices[p*4 + 1];
            int jv_idx = path_indices[p*4 + 2];
            int kv_idx = path_indices[p*4 + 3];

            int k_dim = k_dims[p];
            if(k_dim<=0 || k_dim>MAX_K) continue;

            int uv_base = uv_idx*(U*V);
            T xuv = x_uv_z[uv_base + u];

            int iu_base = iu_seg_offsets[iu_idx];
            int jv_base = jv_seg_offsets[jv_idx];
            int k_base  = kv_k_offsets[kv_idx];

            if (iu_idx != cur_iu_idx){
                if (cur_iu_idx != -1){
                    // 每个 iu_idx 只 flush 一次 -> 直接赋值避免 RMW
                    #pragma unroll
                    for(int ii=0; ii<MAX_I; ++ii){
                        if(acc_iu[ii]!=T(0)){
                            giu_z[cur_iu_base + ii*U + u] = acc_iu[ii];
                        }
                    }
                }
                cur_iu_idx = iu_idx;
                cur_iu_base = iu_base;
                #pragma unroll
                for(int ii=0; ii<MAX_I; ++ii) acc_iu[ii]=T(0);
            }

            T gk[MAX_K];
            #pragma unroll
            for(int kk=0; kk<MAX_K; ++kk) gk[kk]=T(0);
            #pragma unroll
            for(int kk=0; kk<MAX_K; ++kk){
                if(kk<k_dim) gk[kk] = go_z[(k_base + kk)*U + u];
            }

            int base_i = nnz_offsets_i[p];
            #pragma unroll
            for(int ii=0; ii<MAX_I; ++ii){
                int off = nnz_i_offsets[p*MAX_I + ii];
                int cnt = nnz_i_counts [p*MAX_I + ii];
                if (cnt<=0) continue;

                for(int s=0; s<cnt; ++s){
                    int idx = base_i + off + s;
                    int j   = (int)cg_j_i[idx];
                    int kk  = (int)cg_k_i[idx];
                    T   c   = cg_val_i[idx];
                    T xjv = s_jv[jv_base + j];
                    // grad_iu(ii,u) += gk * c * xuv * xjv
                    acc_iu[ii] += gk[kk] * c * xuv * xjv;
                }
            }
        }

        if (cur_iu_idx != -1){
            #pragma unroll
            for(int ii=0; ii<MAX_I; ++ii){
                if(acc_iu[ii]!=T(0)){
                    giu_z[cur_iu_base + ii*U + u] = acc_iu[ii];
                }
            }
        }
    }

    // =========================
    // Pass 3: grad_jv (order_jv) —— 每个 jv_idx 只 flush 一次（每 j 每 warp atomic）
    // =========================
    {
        int cur_jv_idx = -1;
        int cur_jv_base = 0;
        T acc_jv[MAX_J];
        #pragma unroll
        for(int jj=0; jj<MAX_J; ++jj) acc_jv[jj]=T(0);

        for(int t=0; t<P; ++t){
            int p = order_jv[t];
            int uv_idx = path_indices[p*4 + 0];
            int iu_idx = path_indices[p*4 + 1];
            int jv_idx = path_indices[p*4 + 2];
            int kv_idx = path_indices[p*4 + 3];

            int k_dim = k_dims[p];
            if(k_dim<=0 || k_dim>MAX_K) continue;

            int jv_base = jv_seg_offsets[jv_idx];
            if (jv_idx != cur_jv_idx){
                if (cur_jv_idx != -1){
                    // flush: warp reduce then atomic (每 jv_idx 只做一次)
                    #pragma unroll
                    for(int jj=0; jj<MAX_J; ++jj){
                        T v = warp_sum(acc_jv[jj]);
                        if (lane==0 && v!=T(0)) atomicAdd(&gjv_z[cur_jv_base + jj], v);
                    }
                }
                cur_jv_idx = jv_idx;
                cur_jv_base = jv_base;
                #pragma unroll
                for(int jj=0; jj<MAX_J; ++jj) acc_jv[jj]=T(0);
            }

            int uv_base = uv_idx*(U*V);
            T xuv = x_uv_z[uv_base + u];

            int iu_base = iu_seg_offsets[iu_idx];
            int k_base  = kv_k_offsets[kv_idx];

            T gk[MAX_K];
            #pragma unroll
            for(int kk=0; kk<MAX_K; ++kk) gk[kk]=T(0);
            #pragma unroll
            for(int kk=0; kk<MAX_K; ++kk){
                if(kk<k_dim) gk[kk] = go_z[(k_base + kk)*U + u];
            }

            int base_j = nnz_offsets_j[p];
            #pragma unroll
            for(int jj=0; jj<MAX_J; ++jj){
                int off = nnz_j_offsets[p*MAX_J + jj];
                int cnt = nnz_j_counts [p*MAX_J + jj];
                if (cnt<=0) continue;

                for(int s=0; s<cnt; ++s){
                    int idx = base_j + off + s;
                    int i   = (int)cg_i_j[idx];
                    int kk  = (int)cg_k_j[idx];
                    T   c   = cg_val_j[idx];
                    T xiu = s_iu[iu_base + i*U + u];
                    // grad_jv(jj) += gk * c * xuv * xiu
                    acc_jv[jj] += gk[kk] * c * xuv * xiu;
                }
            }
        }

        if (cur_jv_idx != -1){
            #pragma unroll
            for(int jj=0; jj<MAX_J; ++jj){
                T v = warp_sum(acc_jv[jj]);
                if (lane==0 && v!=T(0)) atomicAdd(&gjv_z[cur_jv_base + jj], v);
            }
        }
    }
}

std::vector<torch::Tensor> tp_channel_wise_groupij_bwd_launch(
    torch::Tensor x_uv,            // [Z, UV_TOTAL]
    torch::Tensor x_iu,            // [Z, IU_TOTAL]
    torch::Tensor x_jv,            // [Z, JV_TOTAL]
    torch::Tensor grad_out,        // [Z, K_TOTAL, U, V] or flatten ok, but最好传 4D contiguous

    torch::Tensor path_indices,    // [P,4] int32
    torch::Tensor k_dims,          // [P] int32
    torch::Tensor iu_seg_offsets,  // int32
    torch::Tensor jv_seg_offsets,  // int32
    torch::Tensor kv_k_offsets,    // int32

    // group-i CG
    torch::Tensor cg_i_i,          // [nnz_total_i] uint8
    torch::Tensor cg_j_i,          // [nnz_total_i] uint8
    torch::Tensor cg_k_i,          // [nnz_total_i] uint8
    torch::Tensor cg_val_i,        // [nnz_total_i] float/double
    torch::Tensor nnz_offsets_i,   // [P] int32
    torch::Tensor nnz_i_offsets,   // [P, MAX_I_DIM] int32
    torch::Tensor nnz_i_counts,    // [P, MAX_I_DIM] int32

    // group-j CG
    torch::Tensor cg_i_j,          // [nnz_total_j] uint8
    torch::Tensor cg_j_j,          // [nnz_total_j] uint8
    torch::Tensor cg_k_j,          // [nnz_total_j] uint8
    torch::Tensor cg_val_j,        // [nnz_total_j]
    torch::Tensor nnz_offsets_j,   // [P] int32
    torch::Tensor nnz_j_offsets,   // [P, MAX_J_DIM] int32
    torch::Tensor nnz_j_counts,    // [P, MAX_J_DIM] int32
    torch::Tensor order_uv,
    torch::Tensor order_iu,
    torch::Tensor order_jv,
    const int64_t K_TOTAL,
    const int64_t U,
    const int64_t V
) {
    // -------- checks --------
    TORCH_CHECK(x_uv.is_cuda() && x_iu.is_cuda() && x_jv.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");

    TORCH_CHECK(path_indices.is_cuda() && k_dims.is_cuda(), "path meta must be CUDA");
    TORCH_CHECK(iu_seg_offsets.is_cuda() && jv_seg_offsets.is_cuda() && kv_k_offsets.is_cuda(),
                "seg offsets must be CUDA");

    TORCH_CHECK(cg_i_i.is_cuda() && cg_j_i.is_cuda() && cg_k_i.is_cuda() && cg_val_i.is_cuda(),
                "group-i cg must be CUDA");
    TORCH_CHECK(nnz_offsets_i.is_cuda() && nnz_i_offsets.is_cuda() && nnz_i_counts.is_cuda(),
                "group-i nnz meta must be CUDA");

    TORCH_CHECK(cg_i_j.is_cuda() && cg_j_j.is_cuda() && cg_k_j.is_cuda() && cg_val_j.is_cuda(),
                "group-j cg must be CUDA");
    TORCH_CHECK(nnz_offsets_j.is_cuda() && nnz_j_offsets.is_cuda() && nnz_j_counts.is_cuda(),
                "group-j nnz meta must be CUDA");

    x_uv = x_uv.contiguous();
    x_iu = x_iu.contiguous();
    x_jv = x_jv.contiguous();
    grad_out = grad_out.contiguous();

    path_indices = path_indices.contiguous();
    k_dims = k_dims.contiguous();
    iu_seg_offsets = iu_seg_offsets.contiguous();
    jv_seg_offsets = jv_seg_offsets.contiguous();
    kv_k_offsets = kv_k_offsets.contiguous();

    cg_i_i = cg_i_i.contiguous();
    cg_j_i = cg_j_i.contiguous();
    cg_k_i = cg_k_i.contiguous();
    cg_val_i = cg_val_i.contiguous();
    nnz_offsets_i = nnz_offsets_i.contiguous();
    nnz_i_offsets = nnz_i_offsets.contiguous();
    nnz_i_counts  = nnz_i_counts.contiguous();

    cg_i_j = cg_i_j.contiguous();
    cg_j_j = cg_j_j.contiguous();
    cg_k_j = cg_k_j.contiguous();
    cg_val_j = cg_val_j.contiguous();
    nnz_offsets_j = nnz_offsets_j.contiguous();
    nnz_j_offsets = nnz_j_offsets.contiguous();
    nnz_j_counts  = nnz_j_counts.contiguous();

    // shapes
    const int Z = (int)x_uv.size(0);
    const int UV_TOTAL = (int)x_uv.size(1);
    const int IU_TOTAL = (int)x_iu.size(1);
    const int JV_TOTAL = (int)x_jv.size(1);

    grad_out = grad_out.view({Z, K_TOTAL, U, V});

    TORCH_CHECK(x_iu.size(0) == Z && x_jv.size(0) == Z, "batch mismatch");

    // grad_out expected [Z, K_TOTAL, U, V]
    TORCH_CHECK(grad_out.dim() == 4, "grad_out must be [Z, K_TOTAL, U, V] contiguous");
    TORCH_CHECK(grad_out.size(0) == Z, "grad_out Z mismatch");

    TORCH_CHECK(V == 1, "This v1 kernel/launch assumes V==1 (MACE-OFF).");

    TORCH_CHECK((int)path_indices.size(1) == 4, "path_indices must be [P,4]");
    const int P = (int)path_indices.size(0);

    // outputs
    auto grad_x_uv = torch::zeros_like(x_uv);
    auto grad_x_iu = torch::zeros_like(x_iu);
    auto grad_x_jv = torch::zeros_like(x_jv);

    // launch config
    int threads = U;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;
    dim3 bdim(threads, 1, 1);
    dim3 gdim(Z, 1, 1);

    // shared: s_iu[IU_TOTAL] + s_jv[JV_TOTAL]
    const size_t smem_bytes = (size_t)(IU_TOTAL + JV_TOTAL) * x_uv.element_size();
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int MAX_K_DIM = 8;
    constexpr int MAX_I_DIM = 8;
    constexpr int MAX_J_DIM = 8;

    AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp_channel_wise_groupij_bwd_launch", [&] {
        tp_cwtp_bwd_one_kernel_groupij<scalar_t, MAX_K_DIM, MAX_I_DIM, MAX_J_DIM>
            <<<gdim, bdim, smem_bytes, stream>>>(
                x_uv.data_ptr<scalar_t>(),
                x_iu.data_ptr<scalar_t>(),
                x_jv.data_ptr<scalar_t>(),
                grad_out.data_ptr<scalar_t>(),

                path_indices.data_ptr<int32_t>(),
                k_dims.data_ptr<int32_t>(),
                iu_seg_offsets.data_ptr<int32_t>(),
                jv_seg_offsets.data_ptr<int32_t>(),
                kv_k_offsets.data_ptr<int32_t>(),

                //cg_i_i.data_ptr<uint8_t>(),
                cg_j_i.data_ptr<uint8_t>(),
                cg_k_i.data_ptr<uint8_t>(),
                cg_val_i.data_ptr<scalar_t>(),
                nnz_offsets_i.data_ptr<int32_t>(),
                nnz_i_offsets.data_ptr<int32_t>(),
                nnz_i_counts.data_ptr<int32_t>(),

                cg_i_j.data_ptr<uint8_t>(),
                //cg_j_j.data_ptr<uint8_t>(),
                cg_k_j.data_ptr<uint8_t>(),
                cg_val_j.data_ptr<scalar_t>(),
                nnz_offsets_j.data_ptr<int32_t>(),
                nnz_j_offsets.data_ptr<int32_t>(),
                nnz_j_counts.data_ptr<int32_t>(),

                order_uv.data_ptr<int32_t>(),
                order_iu.data_ptr<int32_t>(),
                order_jv.data_ptr<int32_t>(),

                grad_x_uv.data_ptr<scalar_t>(),
                grad_x_iu.data_ptr<scalar_t>(),
                grad_x_jv.data_ptr<scalar_t>(),

                Z, UV_TOTAL, IU_TOTAL, JV_TOTAL,
                K_TOTAL, U, V, P
            );

        CUDA_CHECK(cudaGetLastError());
    });

    return {grad_x_uv, grad_x_iu, grad_x_jv};
}


TORCH_LIBRARY(cwtp_bwd, m)
{
    m.def("backward", &tp_channel_wise_groupij_bwd_launch);
}
