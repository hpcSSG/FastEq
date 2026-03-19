#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include "cuda_utils.hpp"

#define CWTP_DEFINE_CONSTANTS
#include "cwtp_helper.cuh"

// opt1: CG sparse + path 内groupk, 由于MACE 中 V=1， 所以暂时不放到 shared memory 里
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

// Opt2: 离线将cg系数groupk
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

    for (int idx = u; idx < IU_TOTAL; idx += threads_in_block) {
        s_iu[idx] = x_iu_z[idx];
    }
    for (int idx = u; idx < JV_TOTAL; idx += threads_in_block) {
        s_jv[idx] = x_jv_z[idx];
    }
    __syncthreads();

    const scalar_t* x_uv_z = x_uv + (size_t)z * UV_TOTAL;
    scalar_t* out_z = out + (size_t)z * (K_TOTAL * U * V);

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

// Opt2: 离线将cg系数groupk
template <typename scalar_t, int MAX_K_DIM>
__global__ void tp_channel_wise_sparse_groupk_constant_kernel(
    const scalar_t* __restrict__ x_uv,          // [Z, UV_TOTAL]
    const scalar_t* __restrict__ x_iu,          // [Z, IU_TOTAL]
    const scalar_t* __restrict__ x_jv,          // [Z, JV_TOTAL]

    const int32_t* __restrict__ path_indices,   // [num_paths, 4], 不使用兼容接口
    const int32_t* __restrict__ k_dims,         // [num_paths], 不使用兼容接口 
    const int32_t* __restrict__ iu_seg_offsets, // [iu_seg_count], 不使用兼容接口
    const int32_t* __restrict__ jv_seg_offsets, // [jv_seg_count], 不使用兼容接口
    const int32_t* __restrict__ kv_k_offsets,   // [kv_seg_count], 不使用兼容接口

    const int32_t* __restrict__ nnz_per_path,   // [num_paths], 不使用兼容接口
    const int32_t* __restrict__ nnz_offsets,    // [num_paths], 不使用兼容接口

    const int32_t* __restrict__ nnz_k_offsets,  // [num_paths * MAX_K_DIM], 不使用兼容接口
    const int32_t* __restrict__ nnz_k_counts,   // [num_paths * MAX_K_DIM], 不使用兼容接口

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

                // 遍历这个 k 的所有 nnz
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

// ELL+packed + per-thread scalar cache (NO dynamic indexing arrays) to avoid local-memory stack traffic.
// - Works best for V=1 (MACE-OFF). V>1 is supported via fallback path.
// - MAX_K_DIM=8, MAX_I=5, MAX_J=7

template <typename T, int E>
__device__ __forceinline__ T eval_row_shared_gather_V1(
    const uint16_t* __restrict__ ij_row,
    const T* __restrict__ val_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    #pragma unroll
    for (int e = 0; e < E; ++e) {
        uint16_t ij = ij_row[e];
        T c = val_row[e];
        int i = (int)(ij & 0xFF);
        int j = (int)(ij >> 8);
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_row_shared_gather_V1_runtime(
    int E,
    const uint16_t* __restrict__ ij_row,
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    #pragma unroll 1
    for (int e = 0; e < E; ++e) {
        uint16_t ij = ij_row[e];
        T c = v_row[e];
        int i = (int)(ij & 0xFF);
        int j = (int)(ij >> 8);
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

// -----------------------------
// Kernel: ELL packed bucket, V==1
// 把 CG 的 (i,j) 非零做成了每行固定 E 项的 ELL-packed（ij_row[e] / val_row[e] 连续），并且 s_iu / s_jv 都在 shared 里, 访存更规则一些
// -----------------------------
template <typename T, int MAX_K_DIM = 8>
__global__ void tp_channel_wise_sparse_groupk_ell_kernel(
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const int32_t* __restrict__ k_dims,
    const int4* __restrict__ meta1_4,
    const int4* __restrict__ meta2_4,
    T* __restrict__ out,
    int Z,
    int UV_TOTAL,
    int IU_TOTAL,
    int JV_TOTAL,
    int K_TOTAL,
    int U,
    int num_paths
) {
    
    int z = (int)blockIdx.x;
    if (z >= Z) return;
    int u = (int)threadIdx.x;
    if (u >= U) return;
    extern __shared__ unsigned char smem_raw[];
    T* s_iu = reinterpret_cast<T*>(smem_raw);
    T* s_jv = s_iu + (size_t)IU_TOTAL;
    const T* x_iu_z = x_iu + (size_t)z * IU_TOTAL;
    const T* x_jv_z = x_jv + (size_t)z * JV_TOTAL;
    int threads = (int)blockDim.x;
    for (int idx = u; idx < IU_TOTAL; idx += threads) s_iu[idx] = x_iu_z[idx]; // TODO, pipeline
    for (int idx = u; idx < JV_TOTAL; idx += threads) s_jv[idx] = x_jv_z[idx];
    
    __syncthreads();
    const T* x_uv_z = x_uv + (size_t)z * UV_TOTAL;
    T* out_z = out + (size_t)z * (size_t)(K_TOTAL * U);
    
    const T* cval = ell_val_const_ptr<T>();
    for (int p = 0; p < num_paths; ++p) {

        /*
        int uv_idx = path_indices[p * 4 + 0];
        int iu_idx = path_indices[p * 4 + 1];
        int jv_idx = path_indices[p * 4 + 2];
        int kv_idx = path_indices[p * 4 + 3];
        int uv_base = uv_idx * U; 
        int iu_base = iu_seg_offsets[iu_idx];
        int jv_base = jv_seg_offsets[jv_idx];
        int k_base  = kv_k_offsets[kv_idx];
        */
        
        // packed vectorize load
        int4 m1 = meta1_4[p];
        int4 m2 = meta2_4[p];
        int uv_base, iu_base, jv_base, k_base;
        int k_dim, E, base;
        uv_base = m1.x; iu_base = m1.y; jv_base = m1.z; k_base = m1.w;
        k_dim   = m2.x; E       = m2.y; base    = m2.z;

        T xuv_uv = x_uv_z[uv_base + u];
        #pragma unroll 1
        for (int k_local = 0; k_local < k_dim; ++k_local) {
            int row = base + k_local * E;
            const uint16_t* ij_row = c_ell_ij + row;  // constant
            const T*        v_row  = cval     + row;  // constant
            T acc;
            switch (E) {
                case 1: acc = eval_row_shared_gather_V1<T,1>(ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base); break;
                case 3: acc = eval_row_shared_gather_V1<T,3>(ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base); break;
                case 4: acc = eval_row_shared_gather_V1<T,4>(ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base); break;
                case 5: acc = eval_row_shared_gather_V1<T,5>(ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base); break;
                case 6: acc = eval_row_shared_gather_V1<T,6>(ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base); break;
                case 8: acc = eval_row_shared_gather_V1<T,8>(ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base); break;
                default:
                    acc = eval_row_shared_gather_V1_runtime<T>(E, ij_row, v_row, s_iu, iu_base, U, u, s_jv, jv_base);
                    break;
            }
            out_z[(k_base + k_local) * U + u] = acc * xuv_uv;
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

    //ell data and Packed base offsets 
    torch::Tensor ell_ij,
    torch::Tensor ell_val,
    torch::Tensor meta1,            // packed bases_offsets, [num_paths, 4], int32
    torch::Tensor meta2,            // packed kdims, ell, [num_paths, 4], int32

    const int64_t U,
    const int64_t V,
    const int64_t K_TOTAL
) {
    /* TORCH_CHECK(x_uv.is_cuda(), "x_uv must be CUDA");
    TORCH_CHECK(x_iu.is_cuda(), "x_iu must be CUDA");
    TORCH_CHECK(x_jv.is_cuda(), "x_jv must be CUDA");
    TORCH_CHECK(cg_val_all.is_cuda(), "cg_val_all must be CUDA"); */

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

    int num_paths = path_indices.size(0);
    constexpr int MAX_K_DIM = 8; // 当前最大 k_dim = 7

    TORCH_CHECK(x_iu.size(0) == Z && x_jv.size(0) == Z, "batch dim mismatch");
    TORCH_CHECK(path_indices.dim() == 2 && path_indices.size(1) == 4,
                "path_indices must be [num_paths,4]");
    
    int ell_total = (int)ell_ij.numel();
    TORCH_CHECK(ell_total == (int)ell_val.numel(), "ell_ij and ell_val numel mismatch");
    TORCH_CHECK(num_paths <= MAX_PATHS_CONST, "num_paths exceeds MAX_PATHS_CONST");
    TORCH_CHECK(ell_total <= ELL_MAX_CONST, "ell_total exceeds ELL_MAX_CONST");

    

    auto out = torch::empty({Z, K_TOTAL, U, V}, x_uv.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    int threads = static_cast<int>(U);
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;

    int blocks = static_cast<int>(Z);
    size_t smem_bytes = (IU_TOTAL + JV_TOTAL) * x_uv.element_size();

    // copy meta to constant (D2D)
    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_ell_ij,
        ell_ij.data_ptr<uint16_t>(),
        sizeof(uint16_t) * ell_total,
        0,
        cudaMemcpyDeviceToDevice,
        stream));

    AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp_channel_wise_sparse_groupk_kernel", [&] {
        
        if constexpr (std::is_same<scalar_t, float>::value) {
            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                c_ell_val_f32,
                ell_val.data_ptr<float>(),
                sizeof(float) * ell_total,
                0,
                cudaMemcpyDeviceToDevice,
                stream));
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                c_ell_val_f64,
                ell_val.data_ptr<double>(),
                sizeof(double) * ell_total,
                0,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        /*
        tp_channel_wise_sparse_groupk_constant_kernel<scalar_t, MAX_K_DIM>
            <<<blocks, threads, smem_bytes, stream>>>(
                x_uv.data_ptr<scalar_t>(),
                x_iu.data_ptr<scalar_t>(),
                x_jv.data_ptr<scalar_t>(),
                path_indices.data_ptr<int>(),
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
        */

        tp_channel_wise_sparse_groupk_ell_kernel<scalar_t, MAX_K_DIM>
        <<<blocks, threads, smem_bytes, stream>>>(
            x_uv.data_ptr<scalar_t>(),
            x_iu.data_ptr<scalar_t>(),
            x_jv.data_ptr<scalar_t>(),
            k_dims.data_ptr<int>(),
            reinterpret_cast<const int4*>(meta1.data_ptr<int32_t>()),
            reinterpret_cast<const int4*>(meta2.data_ptr<int32_t>()),
            out.data_ptr<scalar_t>(),
            (int)Z,
            (int)UV_TOTAL,
            (int)IU_TOTAL,
            (int)JV_TOTAL,
            (int)K_TOTAL,
            (int)U,
            num_paths
        );
        
        out = out.view({Z, K_TOTAL * U * V});
        CUDA_CHECK(cudaGetLastError());
    });

    return out;
}



// sharedy, path parallel, k-tile to reduce register
// --------------------- helpers ---------------------

template <typename T>
__device__ __forceinline__ T pick_x5_switch(int i, T x0, T x1, T x2, T x3, T x4) {
  T xi;
  switch (i) {
    default: xi = x0; break;
    case 1:  xi = x1; break;
    case 2:  xi = x2; break;
    case 3:  xi = x3; break;
    case 4:  xi = x4; break;
  }
  return xi;
}

// --------------------- eval: compile-time E, k-tile(4) ---------------------

template <typename T, int E>
__device__ __forceinline__ void eval_ell_innerk_sharedy_xscalar_k4(
    int kdim, int base, int k0,               // compute ks in [k0, k0+3]
    T x0, T x1, T x2, T x3, T x4,
    const T* __restrict__ s_y,                // shared y, length >= 8
    T &b0, T &b1, T &b2, T &b3                // accum for k0..k0+3
) {
  const T* cval = ell_val_const_ptr<T>();

  #pragma unroll
  for (int e = 0; e < E; ++e) {

    #define DO_ONE(KK, ACC) do { \
      int kk = (KK); \
      if (kk < kdim) { \
        int slot = base + kk * E + e; \
        uint16_t ij = c_ell_ij[slot]; \
        T c = cval[slot];            \
        int i = (int)(ij & 0xFF); \
        int j = (int)(ij >> 8); \
        T xi = pick_x5_switch<T>(i, x0, x1, x2, x3, x4); \
        T yj = s_y[j]; \
        ACC = fma(c, xi * yj, ACC); \
      } \
    } while (0)

    DO_ONE(k0 + 0, b0);
    DO_ONE(k0 + 1, b1);
    DO_ONE(k0 + 2, b2);
    DO_ONE(k0 + 3, b3);

    #undef DO_ONE
  }
}

// --------------------- eval: runtime E, k-tile(4) ---------------------

template <typename T>
__device__ __forceinline__ void eval_ell_innerk_sharedy_xscalar_k4_runtimeE(
    int E, int kdim, int base, int k0,
    T x0, T x1, T x2, T x3, T x4,
    const T* __restrict__ s_y,
    T &b0, T &b1, T &b2, T &b3
) {
  const T* cval = ell_val_const_ptr<T>();

  #pragma unroll 1
  for (int e = 0; e < E; ++e) {

    auto step = [&](int kk, T &acc) {
      if (kk < kdim) {
        int slot = base + kk * E + e;
        uint16_t ij = c_ell_ij[slot];
        T c = cval[slot];
        int i = (int)(ij & 0xFF);
        int j = (int)(ij >> 8);
        T xi = pick_x5_switch<T>(i, x0, x1, x2, x3, x4);
        T yj = s_y[j];
        acc = fma(c, xi * yj, acc);
      }
    };

    step(k0 + 0, b0);
    step(k0 + 1, b1);
    step(k0 + 2, b2);
    step(k0 + 3, b3);
  }
}

// ==================== path 分组 for mace-large paths=17 ===================
// 3 groups by iu_idx: {0},{1},{2}
// offsets: [0,4,11,17]
__device__ __constant__ int32_t c_group_offsets[4] = {0, 4, 10, 17};

__device__ __constant__ int32_t c_group_pids[17] = {
  // G0 iu=0
  0, 1, 2, 3,
  // G1 iu=1
  4, 5, 6, 7, 8, 9,
  // G2 iu=2
  10, 11, 12, 13, 14, 15, 16
};

__device__ __constant__ int32_t c_group_iu_idx[3] = {0, 1, 2};


template <typename T, int MAX_K_DIM = 8>
__global__ void cwtp_fwd_group3_by_iu_k4_sharedy_xscalar(
    const T* __restrict__ x_uv,
    const T* __restrict__ x_iu,
    const T* __restrict__ x_jv,
    const int32_t* __restrict__ path_indices, // [P,4] uv,iu,jv,kv
    const int32_t* __restrict__ i_dims,       // [P] (1/3/5)
    const int32_t* __restrict__ j_dims,       // [P] (<=7)
    const int32_t* __restrict__ k_dims,       // [P] (<=MAX_K_DIM)
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    T* __restrict__ out,                      // [Z, K_TOTAL*U]
    int Z,
    int UV_TOTAL,
    int IU_TOTAL,
    int JV_TOTAL,
    int K_TOTAL,
    int U,
    int num_paths   // should be 17 here, but keep for safety
) {
  int z   = (int)blockIdx.x;
  int gid = (int)blockIdx.y; // 0..2
  if (z >= Z) return;
  if ((unsigned)gid >= 3u) return;

  int u = (int)threadIdx.x;
  if (u >= U) return;

  const T* x_uv_z = x_uv + (size_t)z * (size_t)UV_TOTAL;
  const T* x_iu_z = x_iu + (size_t)z * (size_t)IU_TOTAL;
  const T* x_jv_z = x_jv + (size_t)z * (size_t)JV_TOTAL;
  T* out_z        = out  + (size_t)z * (size_t)(K_TOTAL * U);

  // ---------------- shared X for this group (iu_idx fixed) ----------------
  int iu_idx  = c_group_iu_idx[gid];
  int iu_base = iu_seg_offsets[iu_idx];

  // load maximum 5 rows once; safe because IU_TOTAL includes these segments
  T x0 = x_iu_z[iu_base + 0 * U + u];
  T x1 = x_iu_z[iu_base + 1 * U + u];
  T x2 = x_iu_z[iu_base + 2 * U + u];
  T x3 = x_iu_z[iu_base + 3 * U + u];
  T x4 = x_iu_z[iu_base + 4 * U + u];

  __shared__ T s_y[8];

  int beg = c_group_offsets[gid];
  int end = c_group_offsets[gid + 1];

  // ---------------- loop pids in this group ----------------
  #pragma unroll 1
  for (int t = beg; t < end; ++t) {
    int pid = c_group_pids[t];
    if (pid >= num_paths) continue;

    // per pid meta
    int uv_idx = path_indices[pid * 4 + 0];
    int jv_idx = path_indices[pid * 4 + 2];
    int kv_idx = path_indices[pid * 4 + 3];

    int idim = i_dims[pid];
    int jdim = j_dims[pid];
    int kdim = k_dims[pid];

    if (idim <= 0 || idim > 5) continue;
    if (jdim <= 0 || jdim > 7) continue;
    if (kdim <= 0 || kdim > MAX_K_DIM) continue;

    int jv_base = jv_seg_offsets[jv_idx];
    int k_base  = kv_k_offsets[kv_idx];

    // load y to shared (tid<8)
    if (threadIdx.x < 8) {
      if (threadIdx.x < jdim) s_y[threadIdx.x] = x_jv_z[jv_base + threadIdx.x];
      else                    s_y[threadIdx.x] = (T)0;
    }
    __syncthreads();

    // per thread xuv scalar (V==1)
    T xuv_uv = x_uv_z[uv_idx * U + u];

    // ELL meta from constant (you already have these)
    int E    = c_ell_E[pid];
    int base = c_ell_base[pid];
    if (E <= 0) { __syncthreads(); continue; }

    // -------- k tile 0..3 --------
    T b0=0,b1=0,b2=0,b3=0;
    switch (E) {
      case 1: eval_ell_innerk_sharedy_xscalar_k4<T,1>(kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3); break;
      case 3: eval_ell_innerk_sharedy_xscalar_k4<T,3>(kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3); break;
      case 4: eval_ell_innerk_sharedy_xscalar_k4<T,4>(kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3); break;
      case 5: eval_ell_innerk_sharedy_xscalar_k4<T,5>(kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3); break;
      case 6: eval_ell_innerk_sharedy_xscalar_k4<T,6>(kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3); break;
      case 8: eval_ell_innerk_sharedy_xscalar_k4<T,8>(kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3); break;
      default:
        eval_ell_innerk_sharedy_xscalar_k4_runtimeE<T>(E, kdim, base, 0, x0,x1,x2,x3,x4, s_y, b0,b1,b2,b3);
        break;
    }
    if (kdim > 0) out_z[(k_base + 0) * U + u] = b0 * xuv_uv;
    if (kdim > 1) out_z[(k_base + 1) * U + u] = b1 * xuv_uv;
    if (kdim > 2) out_z[(k_base + 2) * U + u] = b2 * xuv_uv;
    if (kdim > 3) out_z[(k_base + 3) * U + u] = b3 * xuv_uv;

    // -------- k tile 4..7 --------
    if (kdim > 4) {
      T c0=0,c1=0,c2=0,c3=0;
      switch (E) {
        case 1: eval_ell_innerk_sharedy_xscalar_k4<T,1>(kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3); break;
        case 3: eval_ell_innerk_sharedy_xscalar_k4<T,3>(kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3); break;
        case 4: eval_ell_innerk_sharedy_xscalar_k4<T,4>(kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3); break;
        case 5: eval_ell_innerk_sharedy_xscalar_k4<T,5>(kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3); break;
        case 6: eval_ell_innerk_sharedy_xscalar_k4<T,6>(kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3); break;
        case 8: eval_ell_innerk_sharedy_xscalar_k4<T,8>(kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3); break;
        default:
          eval_ell_innerk_sharedy_xscalar_k4_runtimeE<T>(E, kdim, base, 4, x0,x1,x2,x3,x4, s_y, c0,c1,c2,c3);
          break;
      }
      if (kdim > 4) out_z[(k_base + 4) * U + u] = c0 * xuv_uv;
      if (kdim > 5) out_z[(k_base + 5) * U + u] = c1 * xuv_uv;
      if (kdim > 6) out_z[(k_base + 6) * U + u] = c2 * xuv_uv;
      if (kdim > 7) out_z[(k_base + 7) * U + u] = c3 * xuv_uv;
    }

    // protect next pid's s_y write
    __syncthreads();
  }
}


// path parallel for cwtp
torch::Tensor tp_channel_wise_pp_fwd_launch(
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

    //ELL CG
    torch::Tensor ell_E,
    torch::Tensor ell_base,
    torch::Tensor ell_ij,
    torch::Tensor ell_val,


    const int64_t U,
    const int64_t V,
    const int64_t K_TOTAL
) {
    /* TORCH_CHECK(x_uv.is_cuda(), "x_uv must be CUDA");
    TORCH_CHECK(x_iu.is_cuda(), "x_iu must be CUDA");
    TORCH_CHECK(x_jv.is_cuda(), "x_jv must be CUDA"); */

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
    auto stream = at::cuda::getCurrentCUDAStream();

    ell_E   = ell_E.contiguous();
    ell_base= ell_base.contiguous();
    ell_ij  = ell_ij.contiguous();
    ell_val = ell_val.contiguous();

    //TORCH_CHECK(ell_E.is_cuda() && ell_base.is_cuda() && ell_ij.is_cuda() && ell_val.is_cuda(), "ell_* must be CUDA");
    TORCH_CHECK(ell_E.scalar_type() == at::kInt, "ell_E must be int32");
    TORCH_CHECK(ell_base.scalar_type() == at::kInt, "ell_base must be int32");
    TORCH_CHECK(ell_ij.scalar_type() == at::kUInt16, "ell_ij must be uint16");
    TORCH_CHECK(ell_val.scalar_type() == x_uv.scalar_type(), "ell_val dtype must match x_uv dtype");

    int ell_total = (int)ell_ij.numel();
    TORCH_CHECK(ell_total == (int)ell_val.numel(), "ell_ij and ell_val numel mismatch");
    TORCH_CHECK(num_paths <= MAX_PATHS_CONST, "num_paths exceeds MAX_PATHS_CONST");
    TORCH_CHECK(ell_total <= ELL_MAX_CONST, "ell_total exceeds ELL_MAX_CONST");

    // copy meta to constant (D2D)
    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_ell_E,
        ell_E.data_ptr<int32_t>(),
        sizeof(int32_t) * num_paths,
        0,
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_ell_base,
        ell_base.data_ptr<int32_t>(),
        sizeof(int32_t) * num_paths,
        0,
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_ell_ij,
        ell_ij.data_ptr<uint16_t>(),
        sizeof(uint16_t) * ell_total,
        0,
        cudaMemcpyDeviceToDevice,
        stream));

    

    AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "cwtp_fwd_group3_by_iu_k4_sharedy_xscalar", [&] {

        // copy ell_val to typed constant
        if constexpr (std::is_same<scalar_t, float>::value) {
            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                c_ell_val_f32,
                ell_val.data_ptr<float>(),
                sizeof(float) * ell_total,
                0,
                cudaMemcpyDeviceToDevice,
                stream));
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                c_ell_val_f64,
                ell_val.data_ptr<double>(),
                sizeof(double) * ell_total,
                0,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        
        dim3 grid(Z, 3, 1);
        dim3 block(U, 1, 1);
       //cwtp_fwd_path_parallel_ell_innerk_template_kernel_k4_sharedy_xscalar<scalar_t> // grid.y = num_paths
       cwtp_fwd_group3_by_iu_k4_sharedy_xscalar<scalar_t, 8> // grid.y = 3
        <<<grid, block, 0, stream>>>(
            x_uv.data_ptr<scalar_t>(),
            x_iu.data_ptr<scalar_t>(),
            x_jv.data_ptr<scalar_t>(),
            (const int32_t*)path_indices.data_ptr<int32_t>(),
            (const int32_t*)i_dims.data_ptr<int32_t>(),
            (const int32_t*)j_dims.data_ptr<int32_t>(),
            (const int32_t*)k_dims.data_ptr<int32_t>(),
            (const int32_t*)iu_seg_offsets.data_ptr<int32_t>(),
            (const int32_t*)jv_seg_offsets.data_ptr<int32_t>(),
            (const int32_t*)kv_k_offsets.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            (int)Z,
            (int)UV_TOTAL,
            (int)IU_TOTAL,
            (int)JV_TOTAL,
            (int)K_TOTAL,
            (int)U,
            num_paths);
        out = out.view({Z, K_TOTAL * U * V});
        CUDA_CHECK(cudaGetLastError());
    });

    return out;
}



TORCH_LIBRARY(cwtp_fwd, m)
{
    m.def("forward", &tp_channel_wise_fwd_launch);
    m.def("forward_pp", &tp_channel_wise_pp_fwd_launch);
}
