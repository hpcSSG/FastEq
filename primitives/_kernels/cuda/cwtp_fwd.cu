#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>


#define CUDA_CHECK(expr)                                                   \
    do {                                                                   \
        cudaError_t _err = (expr);                                         \
        TORCH_CHECK(_err == cudaSuccess,                                   \
                    "CUDA error: ", cudaGetErrorString(_err),              \
                    " (error code ", static_cast<int>(_err), ")");         \
    } while (0)

/*
===================== For mace small ============================================
*/


__device__ __forceinline__ int grid_b0() {
    return blockIdx.x + blockIdx.y * gridDim.x;
}
__device__ __forceinline__ int grid_bstride() {
    return gridDim.x * gridDim.y;
}



#define MAX_C_SIZE 1024  // 1M elements
__constant__ float  c_all_const_f[MAX_C_SIZE];
__constant__ double c_all_const_d[MAX_C_SIZE];

template <typename scalar_t>
struct CAllConst;

template <>
struct CAllConst<float> {
    static __device__ __forceinline__ float load(int idx) {
        return c_all_const_f[idx];
    }
};

template <>
struct CAllConst<double> {
    static __device__ __forceinline__ double load(int idx) {
        return c_all_const_d[idx];
    }
};


constexpr int MAX_NNZ_TOTAL = 224;  // mace-medium=86，mace-large=215 根据模型调整

// only for mace small
__global__ void fwd_mace_small_kernel(
    const double* __restrict__ x,     // [B,U]
    const double* __restrict__ y,     // [B,16]
    const double* __restrict__ w,     // [B,4,U]
    double* __restrict__ out,         // [B,16,U]
    double* __restrict__ b_buf,       // [B,4,U]
    int B, int U, int KS, int P)
{
    const int lane  = threadIdx.x & 31;    // 单 warp
    const int slots = U / 4;               // 96/4=24
    if (lane >= slots) return;             // 其余 8 lane 直接退出

    for (int b = grid_b0(); b < B; b += grid_bstride()) {

        // y[b,:] 放 shared（128B）
        __shared__ double y_sh[16]; // sum of K dim = 16; 
        if (lane < KS) y_sh[lane] = y[size_t(b)*KS + lane];
        __syncwarp();

        const int u = lane * 4;
        const size_t off_u   = size_t(u);
        const size_t off_bu  = size_t(b) * U + off_u;
        const size_t off_bk0 = size_t(b) * KS * U;

        // 读取 x4，一次即可
        const double4 x4 = *reinterpret_cast<const double4*>(&x[off_bu]);

        // 预读四条 w_p4，计算 base_p4 = x4 * w_p4，并写入 b_buf
        const size_t off_w0 = (size_t(b)*P + 0) * U + off_u;
        const size_t off_w1 = (size_t(b)*P + 1) * U + off_u;
        const size_t off_w2 = (size_t(b)*P + 2) * U + off_u;
        const size_t off_w3 = (size_t(b)*P + 3) * U + off_u;

        const double4 w0 = *reinterpret_cast<const double4*>(&w[off_w0]);
        const double4 w1 = *reinterpret_cast<const double4*>(&w[off_w1]);
        const double4 w2 = *reinterpret_cast<const double4*>(&w[off_w2]);
        const double4 w3 = *reinterpret_cast<const double4*>(&w[off_w3]);

        double4 b0; b0.x = x4.x*w0.x; b0.y = x4.y*w0.y; b0.z = x4.z*w0.z; b0.w = x4.w*w0.w;
        double4 b1; b1.x = x4.x*w1.x; b1.y = x4.y*w1.y; b1.z = x4.z*w1.z; b1.w = x4.w*w1.w;
        double4 b2; b2.x = x4.x*w2.x; b2.y = x4.y*w2.y; b2.z = x4.z*w2.z; b2.w = x4.w*w2.w;
        double4 b3; b3.x = x4.x*w3.x; b3.y = x4.y*w3.y; b3.z = x4.z*w3.z; b3.w = x4.w*w3.w;

        *reinterpret_cast<double4*>(&b_buf[off_w0]) = b0;
        *reinterpret_cast<double4*>(&b_buf[off_w1]) = b1;
        *reinterpret_cast<double4*>(&b_buf[off_w2]) = b2;
        *reinterpret_cast<double4*>(&b_buf[off_w3]) = b3;

        // 写 out：按 path 的 k 段分组，out[b,k,u:u+3] = base_p4 * y[b,k]
        // path 0: k={0}
        {
            const int k = 0;
            const double yk = y_sh[k];
            double4 o; o.x = b0.x*yk; o.y = b0.y*yk; o.z = b0.z*yk; o.w = b0.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
        // path 1: k={1,2,3}
        #pragma unroll
        for (int k = 1; k <= 3; ++k) {
            const double yk = y_sh[k];
            double4 o; o.x = b1.x*yk; o.y = b1.y*yk; o.z = b1.z*yk; o.w = b1.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
        // path 2: k={4,5,6,7,8}
        #pragma unroll
        for (int k = 4; k <= 8; ++k) {
            const double yk = y_sh[k];
            double4 o; o.x = b2.x*yk; o.y = b2.y*yk; o.z = b2.z*yk; o.w = b2.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
        // path 3: k={9..15}
        #pragma unroll
        for (int k = 9; k <= 15; ++k) {
            const double yk = y_sh[k];
            double4 o; o.x = b3.x*yk; o.y = b3.y*yk; o.z = b3.z*yk; o.w = b3.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> cwtp_mace_small_forward(
    const at::Tensor& x,     // [B,U]
    const at::Tensor& y,     // [B,16]
    const at::Tensor& w      // [B,4,U]
){
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type()==at::kDouble && y.scalar_type()==at::kDouble && w.scalar_type()==at::kDouble, "expect double dtype");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && w.is_contiguous(), "expect contiguous");

    constexpr int P  = 4;    
    constexpr int U_FIXED = 96;
    constexpr int KS = 16;

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);

    //TORCH_CHECK(U == U_FIXED, "U must be 96");
    TORCH_CHECK(y.sizes() == at::IntArrayRef({B, KS}), "y must be [B,16]");
    TORCH_CHECK(w.sizes() == at::IntArrayRef({B, P * U}), "w must be [B,4,96]");

    auto out   = at::empty({B, KS, U}, x.options());
    auto b_buf = at::empty({B, P,  U}, x.options());

    // 2D grid 自动适配 B>65535
    const int max_xy = 65535;
    int gx_dim = (B > max_xy) ? max_xy : static_cast<int>(B);
    int gy_dim = static_cast<int>((B + gx_dim - 1) / gx_dim);
    if (gy_dim > max_xy) gy_dim = max_xy;

    dim3 grid(gx_dim, gy_dim); // 覆盖任意大 B；剩余用 stride
    dim3 block(32);            // 单 warp

    fwd_mace_small_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        b_buf.data_ptr<double>(),
        (int)B, (int)U, int(KS), int(P)
    );
    out = out.reshape({B, KS * U});
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, b_buf}; // out:[B,16*96], b_buf:[B,4,96]
}

/*
========================= common implemenation ===========================================
*/

template <typename scalar_t>
__global__ void tp_channel_wise_kernel(
    const scalar_t* __restrict__ x_uv,          // [Z, UV_TOTAL]
    const scalar_t* __restrict__ x_iu,          // [Z, IU_TOTAL]
    const scalar_t* __restrict__ x_jv,          // [Z, JV_TOTAL]

    const scalar_t* __restrict__ c_all,         // 所有 path 的 c 拼在一起

    const int* __restrict__ path_indices,   // [num_paths, 4] : (uv_idx, iu_idx, jv_idx, kv_idx)
    const int* __restrict__ i_dims,         // [num_paths]
    const int* __restrict__ j_dims,         // [num_paths]
    const int* __restrict__ k_dims,         // [num_paths]
    const int* __restrict__ c_offsets,      // [num_paths], c_all 中的偏移 (以元素为单位)

    const int* __restrict__ iu_seg_offsets, // [iu_seg_count], offset in x_iu row
    const int* __restrict__ jv_seg_offsets, // [jv_seg_count], offset in x_jv row
    const int* __restrict__ kv_k_offsets,   // [kv_seg_count], offset in K 维

    scalar_t* __restrict__ out,                 // [Z, K_TOTAL, U, V]

    int Z,
    int UV_TOTAL,
    int IU_TOTAL,
    int JV_TOTAL,
    int K_TOTAL,
    int U,
    int V,
    int num_paths
) {
    int z = blockIdx.x;  // 一个 block 处理一个 batch 元素
    if (z >= Z) return;

    int tu = threadIdx.x;  // 映射到 u 维

    extern __shared__ unsigned char smem_raw[];
    scalar_t* s_iu = reinterpret_cast<scalar_t*>(smem_raw);          // [IU_TOTAL]
    scalar_t* s_jv = s_iu + IU_TOTAL;                                // [JV_TOTAL]

    // 1. 把 x_iu[z,:], x_jv[z,:] 搬到 shared
    const scalar_t* x_iu_z = x_iu + z * IU_TOTAL;
    const scalar_t* x_jv_z = x_jv + z * JV_TOTAL;

    for (int idx = tu; idx < IU_TOTAL; idx += blockDim.x) {
        s_iu[idx] = x_iu_z[idx];
    }
    for (int idx = tu; idx < JV_TOTAL; idx += blockDim.x) {
        s_jv[idx] = x_jv_z[idx];
    }
    __syncthreads();

    if (tu >= U) return;

    const scalar_t* x_uv_z = x_uv + z * UV_TOTAL;
    scalar_t* out_z = out + z * (K_TOTAL * U * V);  // out[z,:,:,:] 起点

    for (int p = 0; p < num_paths; ++p) {
        int uv_idx = path_indices[p * 4 + 0];
        int iu_idx = path_indices[p * 4 + 1];
        int jv_idx = path_indices[p * 4 + 2];
        int kv_idx = path_indices[p * 4 + 3];

        int i_dim = i_dims[p];
        int j_dim = j_dims[p];
        int k_dim = k_dims[p];

        int c_offset = c_offsets[p];

        int uv_base = uv_idx * (U * V);          // 每个 uv seg 长度 U*V
        int iu_base = iu_seg_offsets[iu_idx];    // x_iu row 内的 offset
        int jv_base = jv_seg_offsets[jv_idx];    // x_jv row 内的 offset
        int k_base  = kv_k_offsets[kv_idx];      // K 维起始 index

        const scalar_t* c_path = c_all + c_offset;   // [i_dim * j_dim * k_dim]

        int u_idx = tu;

        // 对每个 v：
        for (int v_idx = 0; v_idx < V; ++v_idx) {
            // x_uv[z, uv_seg][u_idx, v_idx]
            scalar_t xuv_uv = x_uv_z[uv_base + u_idx * V + v_idx];
            for (int k = 0; k < k_dim; ++k) {
                scalar_t acc = static_cast<scalar_t>(0);

                for (int i = 0; i < i_dim; ++i) {
                    // x_iu[z, iu_seg][i, u]
                    scalar_t xiu_iu = s_iu[iu_base + i * U + u_idx];

                    for (int j = 0; j < j_dim; ++j) {
                        // x_jv[z, jv_seg][j, v]
                        scalar_t xjv_jv = s_jv[jv_base + j * V + v_idx];

                        int c_idx = (i * j_dim + j) * k_dim + k; // row-major [i,j,k]
                        //scalar_t cij_k = c_path[c_idx];
                        scalar_t cij_k = CAllConst<scalar_t>::load(c_idx);

                        acc += cij_k * xuv_uv * xiu_iu * xjv_jv;
                    }
                }

                int global_k = k_base + k;
                // out[z, global_k, u_idx, v_idx] -> flat index
                int out_index = (global_k * U + u_idx) * V + v_idx;
                out_z[out_index] = acc;
            }
        }
    }
}


// opt1: CG sparse, 由于MACE 中 V=1， 所以暂时不放到 shared memory 里
/*
Memory Throughput: 85.31%
DRAM Throughput: 29.35%
L2 Cache Throughput: 92.80%
Compute (SM) Throughput: 30.59%
Achieved Occupancy: 36.81% 
Duration: 3.5 ms

*/
template <typename scalar_t, int MAX_K_DIM>
__global__ void tp_channel_wise_sparse_kernel(
    const scalar_t* __restrict__ x_uv,          // [Z, UV_TOTAL]
    const scalar_t* __restrict__ x_iu,          // [Z, IU_TOTAL]
    const scalar_t* __restrict__ x_jv,          // [Z, JV_TOTAL]

    const int32_t* __restrict__ path_indices,   // [num_paths, 4] : (uv_idx, iu_idx, jv_idx, kv_idx)
    const int32_t* __restrict__ i_dims,         // [num_paths]  (兼容性保留)
    const int32_t* __restrict__ j_dims,         // [num_paths]  (兼容性保留)
    const int32_t* __restrict__ k_dims,         // [num_paths] 
    const int32_t* __restrict__ iu_seg_offsets, // [iu_seg_count]
    const int32_t* __restrict__ jv_seg_offsets, // [jv_seg_count]
    const int32_t* __restrict__ kv_k_offsets,   // [kv_seg_count]

    // 稀疏 CG 信息
    const int32_t* __restrict__ nnz_per_path,   // [num_paths]
    const int32_t* __restrict__ nnz_offsets,    // [num_paths]
    const uint8_t* __restrict__ cg_i_all,       // [nnz_total]
    const uint8_t* __restrict__ cg_j_all,       // [nnz_total]
    const uint8_t* __restrict__ cg_k_all,       // [nnz_total]
    const scalar_t* __restrict__ cg_val_all,    // [nnz_total]

    scalar_t* __restrict__ out,                 // [Z, K_TOTAL, U, V]

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

    int u = threadIdx.x;  // 线程在 U 维并行
    if (u >= U) return;

    extern __shared__ unsigned char smem_raw[];
    scalar_t* s_iu = reinterpret_cast<scalar_t*>(smem_raw);          // [IU_TOTAL]
    scalar_t* s_jv = s_iu + IU_TOTAL;                                // [JV_TOTAL]

    const scalar_t* x_iu_z = x_iu + z * IU_TOTAL;
    const scalar_t* x_jv_z = x_jv + z * JV_TOTAL;

    // 1. 把 x_iu[z,:], x_jv[z,:] 搬到 shared
    int threads_in_block = blockDim.x;
    for (int idx = u; idx < IU_TOTAL; idx += threads_in_block) {
        s_iu[idx] = x_iu_z[idx];
    }
    for (int idx = u; idx < JV_TOTAL; idx += threads_in_block) {
        s_jv[idx] = x_jv_z[idx];
    }
    __syncthreads();

    const scalar_t* x_uv_z = x_uv + z * UV_TOTAL;
    scalar_t* out_z = out + z * (K_TOTAL * U * V);

    // 2. 遍历所有 path
    for (int p = 0; p < num_paths; ++p) {
        int uv_idx = path_indices[p * 4 + 0];
        int iu_idx = path_indices[p * 4 + 1];
        int jv_idx = path_indices[p * 4 + 2];
        int kv_idx = path_indices[p * 4 + 3];

        int k_dim = k_dims[p];
        int nnz   = nnz_per_path[p];
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

        // 对每个 v:
        for (int v_idx = 0; v_idx < V; ++v_idx) {
            // x_uv[z, uv_seg][u, v]
            scalar_t xuv_uv = x_uv_z[uv_base + u * V + v_idx];
            scalar_t acc_local[MAX_K_DIM];
            #pragma unroll
            for (int kk = 0; kk < MAX_K_DIM; ++kk) {
                acc_local[kk] = static_cast<scalar_t>(0);
            }

            for (int t = 0; t < nnz; ++t) {
                int idx = nnz_off + t;

                int i = cg_i_all[idx];
                int j = cg_j_all[idx];
                int k = cg_k_all[idx];  // 0..k_dim-1

                if (k >= k_dim) continue;

                scalar_t c = cg_val_all[idx];

                // x_iu[z, iu_seg][i, u]
                scalar_t xiu_iu = s_iu[iu_base + i * U + u];

                // x_jv[z, jv_seg][j, v]
                scalar_t xjv_jv = s_jv[jv_base + j * V + v_idx];

                acc_local[k] += c * xuv_uv * xiu_iu * xjv_jv;
            }

            // 写回
            for (int k = 0; k < k_dim; ++k) {
                int global_k = k_base + k;
                int out_index = (global_k * U + u) * V + v_idx;
                out_z[out_index] = acc_local[k];
            }
        }
    }
}

// Opt2: 由于回写占了大头，所以通过group k 写局部，减少global回写次数
/*
    DRAM Frequency                  Ghz         2.62
    SM Frequency                    Ghz         1.60
    Elapsed Cycles                cycle    2,874,643
    Memory Throughput                 %        75.83
    DRAM Throughput                   %        67.82
    Duration                         ms         1.79
    L1/TEX Cache Throughput           %        75.99
    L2 Cache Throughput               %        64.20
    SM Active Cycles              cycle 2,863,095.42
    Compute (SM) Throughput           %        77.48
*/
template <typename scalar_t, int MAX_K_DIM>
__global__ void tp_channel_wise_sparse_groupk_kernel(
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

    // off = nnz_k_offsets[p * MAX_K_DIM + k_local]
    // cnt = nnz_k_counts [p * MAX_K_DIM + k_local]
    const int32_t* __restrict__ nnz_k_offsets,  // [num_paths * MAX_K_DIM]
    const int32_t* __restrict__ nnz_k_counts,   // [num_paths * MAX_K_DIM]

    // 稀疏 CG 系数（global memory）
    const uint8_t* __restrict__ cg_i_all,       // [nnz_total]
    const uint8_t* __restrict__ cg_j_all,       // [nnz_total]
    const scalar_t* __restrict__ cg_val_all,    // [nnz_total]

    scalar_t* __restrict__ out,                 // [Z, K_TOTAL, U, V]

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

    const scalar_t* x_uv_z = x_uv + (size_t)z * UV_TOTAL;
    scalar_t* out_z = out + (size_t)z * (K_TOTAL * U * V);

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

        // In Mace-OFF, V=1
        for (int v_idx = 0; v_idx < V; ++v_idx) {
            scalar_t xuv_uv = x_uv_z[uv_base + u * V + v_idx];

            // 按 k 分组
            for (int k_local = 0; k_local < k_dim; ++k_local) {
                int meta_idx    = p * MAX_K_DIM + k_local;
                int local_off   = nnz_k_offsets[meta_idx];
                int local_count = nnz_k_counts[meta_idx];

                if (local_count <= 0)
                    continue;

                scalar_t acc = static_cast<scalar_t>(0);

                // 遍历这个 k 的所有 nnz（本 path 内连续）
                for (int tt = 0; tt < local_count; ++tt) {
                    int t   = local_off + tt;
                    int idx = nnz_off + t; // global nnz index

                    int i = static_cast<int>(cg_i_all[idx]);
                    int j = static_cast<int>(cg_j_all[idx]);
                    scalar_t c = cg_val_all[idx];

                    scalar_t xiu_iu = s_iu[iu_base + i * U + u];         // x_iu[z, iu_seg][i, u]
                    scalar_t xjv_jv = s_jv[jv_base + j * V + v_idx];     // x_jv[z, jv_seg][j, v]

                    acc += c * xuv_uv * xiu_iu * xjv_jv;
                }

                int global_k  = k_base + k_local;
                int out_index = (global_k * U + u) * V + v_idx;
                out_z[out_index] = acc;
            }
        }
    }
}

torch::Tensor tp_channel_wise_fwd_launch(
    torch::Tensor x_uv,            // [Z, UV_TOTAL]
    torch::Tensor x_iu,            // [Z, IU_TOTAL]
    torch::Tensor x_jv,            // [Z, JV_TOTAL]
    torch::Tensor c_all,           // [sum_p i_p*j_p*k_p]
    torch::Tensor path_indices,    // [num_paths, 4], int32
    torch::Tensor i_dims,          // [num_paths], int32
    torch::Tensor j_dims,          // [num_paths], int32
    torch::Tensor k_dims,          // [num_paths], int32
    torch::Tensor c_offsets,       // [num_paths], int32
    torch::Tensor iu_seg_offsets,  // [iu_seg_count], int32
    torch::Tensor jv_seg_offsets,  // [jv_seg_count], int32
    torch::Tensor kv_k_offsets,    // [kv_seg_count], int32

    // 稀疏 CG
    torch::Tensor nnz_per_path,    // [num_paths], int32
    torch::Tensor nnz_offsets,     // [num_paths], int32
    torch::Tensor nnz_k_offsets,
    torch::Tensor nnz_k_counts,
    torch::Tensor cg_i_all,        // [nnz_total], int32
    torch::Tensor cg_j_all,        // [nnz_total], int32
    torch::Tensor cg_k_all,        // [nnz_total], int32
    torch::Tensor cg_val_all,      // [nnz_total], same dtype as x_uv

    const int64_t U,
    const int64_t V,
    const int64_t K_TOTAL
) {
    TORCH_CHECK(x_uv.is_cuda(), "x_uv must be CUDA");
    TORCH_CHECK(x_iu.is_cuda(), "x_iu must be CUDA");
    TORCH_CHECK(x_jv.is_cuda(), "x_jv must be CUDA");
    TORCH_CHECK(cg_val_all.is_cuda(), "cg_val_all must be CUDA");

    x_uv = x_uv.contiguous();
    x_iu = x_iu.contiguous();
    x_jv = x_jv.contiguous();
    c_all = c_all.contiguous();
    path_indices = path_indices.contiguous();
    i_dims = i_dims.contiguous();
    j_dims = j_dims.contiguous();
    k_dims = k_dims.contiguous();
    c_offsets = c_offsets.contiguous();
    iu_seg_offsets = iu_seg_offsets.contiguous();
    jv_seg_offsets = jv_seg_offsets.contiguous();
    kv_k_offsets = kv_k_offsets.contiguous();

    nnz_per_path   = nnz_per_path.contiguous();
    nnz_offsets    = nnz_offsets.contiguous();
    cg_i_all       = cg_i_all.contiguous();
    cg_j_all       = cg_j_all.contiguous();
    cg_k_all       = cg_k_all.contiguous();
    cg_val_all     = cg_val_all.contiguous();

    auto Z = x_uv.size(0);
    auto UV_TOTAL = x_uv.size(1);
    auto IU_TOTAL = x_iu.size(1);
    auto JV_TOTAL = x_jv.size(1);

    TORCH_CHECK(x_iu.size(0) == Z && x_jv.size(0) == Z, "batch dim mismatch");
    TORCH_CHECK(path_indices.dim() == 2 && path_indices.size(1) == 4,
                "path_indices must be [num_paths,4]");

    int num_paths = path_indices.size(0);

    constexpr int MAX_K_DIM = 8; // 当前最大 k_dim = 7

    auto out = torch::empty({Z, K_TOTAL, U, V}, x_uv.options());

    int threads = static_cast<int>(U);
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;

    int blocks = static_cast<int>(Z);
    size_t smem_bytes = (IU_TOTAL + JV_TOTAL) * x_uv.element_size();
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp_channel_wise_sparse_groupk_kernel", [&] {

        tp_channel_wise_sparse_groupk_kernel<scalar_t, MAX_K_DIM>
            <<<blocks, threads, smem_bytes, stream>>>(
                x_uv.data_ptr<scalar_t>(),
                x_iu.data_ptr<scalar_t>(),
                x_jv.data_ptr<scalar_t>(),
                path_indices.data_ptr<int>(),
                //i_dims.data_ptr<int>(),
                //j_dims.data_ptr<int>(),
                k_dims.data_ptr<int>(),
                iu_seg_offsets.data_ptr<int>(),
                jv_seg_offsets.data_ptr<int>(),
                kv_k_offsets.data_ptr<int>(),
                nnz_per_path.data_ptr<int>(),
                nnz_offsets.data_ptr<int>(),
                nnz_k_offsets.data_ptr<int>(),
                nnz_k_counts.data_ptr<int>(),
                cg_i_all.data_ptr<uint8_t>(),
                cg_j_all.data_ptr<uint8_t>(),
                cg_val_all.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                (int)Z,
                (int)UV_TOTAL,
                (int)IU_TOTAL,
                (int)JV_TOTAL,
                (int)K_TOTAL,
                (int)U,
                (int)V,
                num_paths
            );
        
        out = out.view({Z, K_TOTAL * U * V});
        CUDA_CHECK(cudaGetLastError());
    });

    return out;
}

TORCH_LIBRARY(cwtp_fwd, m)
{
    //m.def("forward", &cwtp_mace_small_forward);
    m.def("forward", &tp_channel_wise_fwd_launch);
}
