#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "cuda_utils.hpp"

// -------------------------------------------------------------------------------------------------
// Backward kernel for fused sender-major TP + scatter_sum
// -------------------------------------------------------------------------------------------------
template <typename scalar_t, int MAX_K_DIM, int JV_MAX>
__global__ void tp_channel_wise_sparse_groupk_fused_scatter_sender_major_bwd_kernel(
    const scalar_t* __restrict__ grad_out_nodes, // [N, C]
    const scalar_t* __restrict__ x_uv_e,         // [E, UV_TOTAL]
    const scalar_t* __restrict__ x_jv_e,         // [E, JV_TOTAL]
    const scalar_t* __restrict__ x_iu_n,         // [N, IU_TOTAL]

    const int32_t* __restrict__ receiver,        // [E]
    const int32_t* __restrict__ row_ptr_s,       // [N+1]

    const int32_t* __restrict__ path_indices,    // [P,4]
    const int32_t* __restrict__ k_dims,          // [P]
    const int32_t* __restrict__ iu_seg_offsets,  // 
    const int32_t* __restrict__ jv_seg_offsets,  // 
    const int32_t* __restrict__ kv_k_offsets,    //

    const int32_t* __restrict__ nnz_per_path,    // [P]
    const int32_t* __restrict__ nnz_offsets,     // [P]
    const int32_t* __restrict__ nnz_k_offsets,   // [P*MAX_K_DIM]
    const int32_t* __restrict__ nnz_k_counts,    // [P*MAX_K_DIM]

    const uint8_t* __restrict__ cg_i_all,        // [nnz_total]
    const uint8_t* __restrict__ cg_j_all,        // [nnz_total]
    const scalar_t* __restrict__ cg_val_all,     // [nnz_total]

    // grads
    scalar_t* __restrict__ grad_x_uv_e,          // [E, UV_TOTAL]
    scalar_t* __restrict__ grad_x_jv_e,          // [E, JV_TOTAL]
    scalar_t* __restrict__ grad_x_iu_n,          // [N, IU_TOTAL]

    // sizes
    int N, int E,
    int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int K_TOTAL, int U, int V,
    int num_paths)
{
  const int s = (int)blockIdx.x;      // sender node
  const int tid = (int)threadIdx.x;
  const int u = tid;

  if (s >= N) return;
  const bool active = (u < U);
  if (!active) return;

  // ------ shared: cache sender feats + cache grad_x_iu for this sender ------
  extern __shared__ unsigned char smem_raw[];
  scalar_t* s_iu  = reinterpret_cast<scalar_t*>(smem_raw);              // [IU_TOTAL]
  scalar_t* s_giu = s_iu + IU_TOTAL;                                     // [IU_TOTAL]

  const scalar_t* xiu_s = x_iu_n + (size_t)s * IU_TOTAL;

  // load xiu_s -> s_iu and init s_giu=0 (one-time per sender)
  for (int idx = u; idx < IU_TOTAL; idx += blockDim.x) {
    s_iu[idx]  = xiu_s[idx];
    s_giu[idx] = (scalar_t)0;
  }
  __syncthreads();

  const int e0 = row_ptr_s[s];
  const int e1 = row_ptr_s[s + 1];
  if (e0 >= e1) {
    // no edges -> grad_x_iu is zero, nothing to write
    return;
  }

  // warp reduce scratch: [max 8 warps][JV_MAX]
  __shared__ scalar_t warp_sum_sh[8][JV_MAX];

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  // -------- process edges of this sender (sender-major) --------
  for (int e = e0; e < e1; ++e) {
    const int r = receiver[e];

    const scalar_t* xuv = x_uv_e + (size_t)e * UV_TOTAL;
    const scalar_t* xjv = x_jv_e + (size_t)e * JV_TOTAL;

    scalar_t* gxuv = grad_x_uv_e + (size_t)e * UV_TOTAL;
    scalar_t* gxjv = grad_x_jv_e + (size_t)e * JV_TOTAL;

    const scalar_t* go_r = grad_out_nodes + (size_t)r * (size_t)(K_TOTAL * U * V);

    // -------- jv partial: per-thread register accumulate over all contributions --------
    // requires JV_TOTAL <= 16
    scalar_t jtmp[JV_MAX];
    #pragma unroll
    for (int jj = 0; jj < JV_MAX; ++jj) jtmp[jj] = (scalar_t)0;

    // -------- main loops: mirror forward structure (p -> v -> k_local -> nnz) --------
    for (int p = 0; p < num_paths; ++p) {
      const int uv_idx = path_indices[p * 4 + 0];
      const int iu_idx = path_indices[p * 4 + 1];
      const int jv_idx = path_indices[p * 4 + 2];
      const int kv_idx = path_indices[p * 4 + 3];

      const int k_dim   = k_dims[p];
      const int nnz     = nnz_per_path[p];
      const int nnz_off = nnz_offsets[p];

      if (k_dim <= 0 || nnz <= 0) continue;
      if (k_dim > MAX_K_DIM) return;

      const int uv_base = uv_idx * (U * V);
      const int iu_base = iu_seg_offsets[iu_idx];
      const int jv_base = jv_seg_offsets[jv_idx];
      const int k_base  = kv_k_offsets[kv_idx];

      for (int v_idx = 0; v_idx < V; ++v_idx) {
        const int uv_off = uv_base + u * V + v_idx;
        const scalar_t xuv_uv = xuv[uv_off];

        #pragma unroll
        for (int k_local = 0; k_local < MAX_K_DIM; ++k_local) {
          if (k_local >= k_dim) break;

          const int meta_idx    = p * MAX_K_DIM + k_local;
          const int local_off   = nnz_k_offsets[meta_idx];
          const int local_count = nnz_k_counts[meta_idx];
          if (local_count <= 0) continue;

          const int global_k  = k_base + k_local;
          const int out_index = (global_k * U + u) * V + v_idx;
          const scalar_t go = go_r[out_index];

          // d/dxuv for this (p,k_local,u,v): sum_{nnz} go*c*xiu*xjv
          scalar_t acc_duv = (scalar_t)0;

          #pragma unroll 1
          for (int tt = 0; tt < local_count; ++tt) {
            const int idx = nnz_off + (local_off + tt);

            const int i = (int)cg_i_all[idx];
            const int j = (int)cg_j_all[idx];
            const scalar_t c = cg_val_all[idx];

            const int iu_off = iu_base + i * U + u;
            const int jv_off = jv_base + j * V + v_idx;   // global index inside [0, JV_TOTAL)

            const scalar_t xiu = s_iu[iu_off];
            const scalar_t xj  = xjv[jv_off];

            // grad x_uv
            acc_duv = fma(go * c, xiu * xj, acc_duv);

            // grad x_iu (accumulate in shared cache, one-time global write at end)
            s_giu[iu_off] = fma(go * c, xuv_uv * xj, s_giu[iu_off]);

            // grad x_jv (register partial, reduced over u later)
            if (jv_off < JV_MAX) {
              jtmp[jv_off] = fma(go * c, xuv_uv * xiu, jtmp[jv_off]);
            }
          }

          // write grad x_uv (per-edge, per-u unique, no atomic)
          gxuv[uv_off] += acc_duv;
        }
      }
    }

    // -------- reduce jtmp over u threads and write grad_x_jv_e[e, :] --------
    // warp-level reduce for each jj in [0, JV_TOTAL)
    for (int jj = 0; jj < JV_TOTAL; ++jj) {
      scalar_t v = jtmp[jj];

      unsigned mask = 0xffffffffu;
      // warp reduce
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(mask, v, off);
      }
      if (lane == 0) {
        warp_sum_sh[warp][jj] = v;
      }
    }
    __syncthreads();

    // warp0 reduces across warps
    if (warp == 0) {
      for (int jj = 0; jj < JV_TOTAL; ++jj) {
        scalar_t v = (lane < num_warps) ? warp_sum_sh[lane][jj] : (scalar_t)0;
        unsigned mask0 = 0xffffffffu;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          v += __shfl_down_sync(mask0, v, off);
        }
        if (lane == 0) {
          gxjv[jj] += v;  // per-edge unique writer block -> no atomic
        }
      }
    }
    __syncthreads(); // reuse warp_sum_sh for next edge
  }

  // -------- flush grad_x_iu_n for this sender once (fix your L1->L2 pain point) --------
  scalar_t* gxi_s = grad_x_iu_n + (size_t)s * IU_TOTAL;
  for (int idx = u; idx < IU_TOTAL; idx += blockDim.x) {
    gxi_s[idx] += s_giu[idx];
  }
}


std::vector<torch::Tensor> tp_groupk_fused_sender_scatter_bwd_launch(
    torch::Tensor grad_out_nodes,   // [N, K_TOTAL*U*V]
    torch::Tensor x_uv,             // [E, UV_TOTAL]
    torch::Tensor x_iu,             // [N, IU_TOTAL]
    torch::Tensor x_jv,             // [E, JV_TOTAL]
    torch::Tensor receiver,         // [E] int32
    torch::Tensor row_ptr_s,        // [N+1] int32

    // meta
    torch::Tensor path_indices,     // [P,4] int32
    torch::Tensor k_dims,           // [P] int32
    torch::Tensor iu_seg_offsets,   // int32
    torch::Tensor jv_seg_offsets,   // int32
    torch::Tensor kv_k_offsets,     // int32
    torch::Tensor nnz_per_path,     // [P] int32
    torch::Tensor nnz_offsets,      // [P] int32
    torch::Tensor nnz_k_offsets,    // [P*MAX_K_DIM] int32
    torch::Tensor nnz_k_counts,     // [P*MAX_K_DIM] int32
    torch::Tensor cg_i_all,         // [nnz_total] uint8
    torch::Tensor cg_j_all,         // [nnz_total] uint8
    torch::Tensor cg_val_all,       // [nnz_total] float/double

    const int64_t U,
    const int64_t V,
    const int64_t K_TOTAL
) {
  TORCH_CHECK(grad_out_nodes.is_cuda(), "grad_out_nodes must be CUDA");
  TORCH_CHECK(x_uv.is_cuda() && x_iu.is_cuda() && x_jv.is_cuda(), "x_uv/x_iu/x_jv must be CUDA");
  TORCH_CHECK(receiver.is_cuda() && row_ptr_s.is_cuda(), "receiver/row_ptr_s must be CUDA");

  grad_out_nodes = grad_out_nodes.contiguous();
  x_uv = x_uv.contiguous();
  x_iu = x_iu.contiguous();
  x_jv = x_jv.contiguous();
  receiver = receiver.contiguous();
  row_ptr_s = row_ptr_s.contiguous();

  path_indices = path_indices.contiguous();
  k_dims = k_dims.contiguous();
  iu_seg_offsets = iu_seg_offsets.contiguous();
  jv_seg_offsets = jv_seg_offsets.contiguous();
  kv_k_offsets = kv_k_offsets.contiguous();
  nnz_per_path = nnz_per_path.contiguous();
  nnz_offsets = nnz_offsets.contiguous();
  nnz_k_offsets = nnz_k_offsets.contiguous();
  nnz_k_counts = nnz_k_counts.contiguous();
  cg_i_all = cg_i_all.contiguous();
  cg_j_all = cg_j_all.contiguous();
  cg_val_all = cg_val_all.contiguous();

  const int64_t E = x_uv.size(0);
  const int64_t N = x_iu.size(0);

  TORCH_CHECK(x_jv.size(0) == E, "x_jv must be [E, JV_TOTAL]");
  TORCH_CHECK(receiver.numel() == E, "receiver must be [E]");
  TORCH_CHECK(row_ptr_s.numel() == N + 1, "row_ptr_s must be [N+1]");

  const int UV_TOTAL = (int)x_uv.size(1);
  const int IU_TOTAL = (int)x_iu.size(1);
  const int JV_TOTAL = (int)x_jv.size(1);

  const int64_t C = K_TOTAL * U * V;
  TORCH_CHECK(grad_out_nodes.size(0) == N && grad_out_nodes.size(1) == C,
              "grad_out_nodes must be [N, K_TOTAL*U*V]");

  TORCH_CHECK(JV_TOTAL <= 32, "This bwd kernel assumes JV_TOTAL<=32. Need tiling otherwise.");

  auto grad_x_uv = torch::zeros_like(x_uv);
  auto grad_x_iu = torch::zeros_like(x_iu);
  auto grad_x_jv = torch::zeros_like(x_jv);

  constexpr int MAX_K_DIM = 8;
  constexpr int JV_MAX = 16;

  int block_u = (int)U;
  if (block_u < 32) block_u = 32;
  if (block_u > 1024) block_u = 1024;
  TORCH_CHECK((int)U <= block_u, "U>1024 needs tiling");

  dim3 grid((unsigned int)N, 1, 1);
  dim3 block((unsigned int)block_u, 1, 1);

  size_t shmem = (size_t)2 * (size_t)IU_TOTAL * (size_t)x_uv.element_size();
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp_groupk_fused_sender_scatter_bwd", [&](){
    tp_channel_wise_sparse_groupk_fused_scatter_sender_major_bwd_kernel<scalar_t, MAX_K_DIM, JV_MAX>
      <<<grid, block, shmem, stream>>>(
        (const scalar_t*)grad_out_nodes.data_ptr<scalar_t>(),
        (const scalar_t*)x_uv.data_ptr<scalar_t>(),
        (const scalar_t*)x_jv.data_ptr<scalar_t>(),
        (const scalar_t*)x_iu.data_ptr<scalar_t>(),
        receiver.data_ptr<int32_t>(),
        row_ptr_s.data_ptr<int32_t>(),
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
        (scalar_t*)grad_x_uv.data_ptr<scalar_t>(),
        (scalar_t*)grad_x_jv.data_ptr<scalar_t>(),
        (scalar_t*)grad_x_iu.data_ptr<scalar_t>(),
        (int)N, (int)E,
        (int)UV_TOTAL, (int)IU_TOTAL, (int)JV_TOTAL,
        (int)K_TOTAL, (int)U, (int)V,
        (int)path_indices.size(0)
      );
  });

  CUDA_CHECK(cudaGetLastError());
  return {grad_x_uv, grad_x_iu, grad_x_jv};
}


template<typename T>
struct XiuAcc {
  const T* __restrict__ s_iu;   // [IU_TOTAL], cached x_iu[s]
  T* __restrict__ s_giu;        // [IU_TOTAL], cached grad_x_iu[s]
  int iu_base;                  // per-path
  int U;                        // runtime U

  __device__ __forceinline__ T load(int i, int u) const {
    return s_iu[iu_base + i * U + u];
  }
  __device__ __forceinline__ void add(int i, int u, T v) const {
    int off = iu_base + i * U + u;
    s_giu[off] = (T)(s_giu[off] + v);
  }
};

#define MAKE_XIUACC(IU_IDX) XiuAcc<T>{ s_iu, s_giu, (int)iu_seg_offsets[IU_IDX], U }

template<typename T, int JB>
__device__ __forceinline__ void tp17_eval_single_edge_sender(
    int u,
    const T* __restrict__ go_edge,   // [K_TOTAL*U] (V=1), = grad_out_nodes[r]
    const T* __restrict__ x_uv_e,    // [UV_TOTAL]
    const T* __restrict__ x_jv_e,    // [JV_TOTAL]
    const T* __restrict__ c_all,     // [C_TOTAL]
    T* __restrict__ grad_x_uv_e,     // [UV_TOTAL]
    T* __restrict__ jtmp,            // [16] register
    XiuAcc<T> xiu,

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,

    int U, int K_TOTAL, int UV_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv_e[uv_base + u];
  const T xjv = x_jv_e[jv_base + 0];
  const T c   = c_all[c_base + 0];
  const T gk  = go_edge[(int64_t)(k_base + 0) * U + u];

  const T gc = gk * c;
  const T xiu0 = xiu.load(0, u);

  grad_x_uv_e[uv_base + u] += gc * xiu0 * xjv;
  xiu.add(0, u, gc * xuv * xjv);
  jtmp[JB + 0] += gc * xuv * xiu0;
}

template<typename T, int D, int JB>
__device__ __forceinline__ void tp17_eval_diag_jk_i0_edge_sender(
    int u,
    const T* __restrict__ go_edge,
    const T* __restrict__ x_uv_e,
    const T* __restrict__ x_jv_e,
    const T* __restrict__ c_all,
    T* __restrict__ grad_x_uv_e,
    T* __restrict__ jtmp,
    XiuAcc<T> xiu,

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,

    int U, int K_TOTAL, int UV_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv_e[uv_base + u];
  const T xiu0 = xiu.load(0, u);

  #pragma unroll
  for (int t = 0; t < D; ++t) {
    const T xjv = x_jv_e[jv_base + t];
    const T c   = c_all[c_base + t * (D + 1)];  // (0,t,t)
    const T gk  = go_edge[(int64_t)(k_base + t) * U + u];

    const T gc = gk * c;
    grad_x_uv_e[uv_base + u] += gc * xiu0 * xjv;
    xiu.add(0, u, gc * xuv * xjv);
    jtmp[JB + t] += gc * xuv * xiu0;
  }
}

template<typename T, int D, int JB>
__device__ __forceinline__ void tp17_eval_diag_ik_j0_edge_sender(
    int u,
    const T* __restrict__ go_edge,
    const T* __restrict__ x_uv_e,
    const T* __restrict__ x_jv_e,
    const T* __restrict__ c_all,
    T* __restrict__ grad_x_uv_e,
    T* __restrict__ jtmp,
    XiuAcc<T> xiu,

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,

    int U, int K_TOTAL, int UV_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv_e[uv_base + u];
  const T xjv0 = x_jv_e[jv_base + 0]; // j=0 only

  #pragma unroll
  for (int t = 0; t < D; ++t) {
    const T xiu_t = xiu.load(t, u);
    const T c     = c_all[c_base + t * (D + 1)]; // (t,0,t)
    const T gk    = go_edge[(int64_t)(k_base + t) * U + u];

    const T gc = gk * c;
    grad_x_uv_e[uv_base + u] += gc * xiu_t * xjv0;
    xiu.add(t, u, gc * xuv * xjv0);
    jtmp[JB + 0] += gc * xuv * xiu_t; // j only 0 -> global jj=JB
  }
}

template<typename T, int D, int JB>
__device__ __forceinline__ void tp17_eval_diag_ij_k0_edge_sender(
    int u,
    const T* __restrict__ go_edge,
    const T* __restrict__ x_uv_e,
    const T* __restrict__ x_jv_e,
    const T* __restrict__ c_all,
    T* __restrict__ grad_x_uv_e,
    T* __restrict__ jtmp,
    XiuAcc<T> xiu,

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,

    int U, int K_TOTAL, int UV_TOTAL, int JV_TOTAL,
    int UV_IDX, int IU_IDX, int JV_IDX, int KV_IDX, int P
){
  const int uv_base = (int)uv_seg_offsets[UV_IDX];
  const int jv_base = (int)jv_seg_offsets[JV_IDX];
  const int k_base  = (int)kv_k_offsets[KV_IDX];
  const int c_base  = (int)c_offsets[P];

  const T xuv = x_uv_e[uv_base + u];
  const T gk  = go_edge[(int64_t)(k_base + 0) * U + u];

  #pragma unroll
  for (int t = 0; t < D; ++t) {
    const T xiu_t = xiu.load(t, u);
    const T xjv_t = x_jv_e[jv_base + t];
    const T c     = c_all[c_base + t * (D + 1)]; // (t,t,0)
    const T gc    = gk * c;

    grad_x_uv_e[uv_base + u] += gc * xiu_t * xjv_t;
    xiu.add(t, u, gc * xuv * xjv_t);
    jtmp[JB + t] += gc * xuv * xiu_t;
  }
}

template<int P, int UV, int IU, int JV, int KV, int I, int J, int K, int JB, typename T>
__device__ __forceinline__ void tp17_path_eval_sharedc_edge_sender(
    int u, int lane,
    const T* __restrict__ go_edge,   // [K_TOTAL*U]
    const T* __restrict__ x_uv_e,    // [UV_TOTAL]
    const T* __restrict__ x_jv_e,    // [JV_TOTAL]
    const T* __restrict__ c_all,     // [C_TOTAL]
    T* __restrict__ grad_x_uv_e,     // [UV_TOTAL]
    T* __restrict__ jtmp,            // [16]
    XiuAcc<T> xiu,

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets,

    int U_runtime, int K_TOTAL,
    int UV_TOTAL, int JV_TOTAL
){
  const int uv_base = (int)uv_seg_offsets[UV];
  const int jv_base = (int)jv_seg_offsets[JV];
  const int k_base  = (int)kv_k_offsets[KV];

  const T* c_ptr = c_all + (int)c_offsets[P];

  // xuv
  const T xuv = x_uv_e[uv_base + u];

  // xj[J] : warp broadcast (same trick as your code)
  T xj[J];
  #pragma unroll
  for (int j = 0; j < J; ++j) {
    T v = (lane == 0) ? x_jv_e[jv_base + j] : (T)0;
    xj[j] = __shfl_sync(0xffffffff, v, 0);
  }

  // xiu[I] from shared
  T xiu_reg[I];
  #pragma unroll
  for (int i = 0; i < I; ++i) xiu_reg[i] = xiu.load(i, u);

  // accum
  T acc_uv = (T)0;
  T acc_iu[I];
  #pragma unroll
  for (int i = 0; i < I; ++i) acc_iu[i] = (T)0;

  T jloc[J];
  #pragma unroll
  for (int j = 0; j < J; ++j) jloc[j] = (T)0;

  #pragma unroll
  for (int kk = 0; kk < K; ++kk) {
    const T go = go_edge[(int64_t)(k_base + kk) * (int64_t)U_runtime + u];

    // uv/iu
    #pragma unroll
    for (int i = 0; i < I; ++i) {
      T s = (T)0;
      #pragma unroll
      for (int j = 0; j < J; ++j) {
        const T c = c_ptr[((i * J + j) * K + kk)];
        s = fma(c, xj[j], s);
      }
      const T go_s = go * s;
      acc_uv    = fma(go_s, xiu_reg[i], acc_uv);
      acc_iu[i] = fma(go_s, xuv,        acc_iu[i]);
    }

    // jv partial
    #pragma unroll
    for (int j = 0; j < J; ++j) {
      T tj = (T)0;
      #pragma unroll
      for (int i = 0; i < I; ++i) {
        const T c = c_ptr[((i * J + j) * K + kk)];
        tj = fma(c, xiu_reg[i], tj);
      }
      jloc[j] = fma(go, xuv * tj, jloc[j]);
    }
  }

  // write back uv
  grad_x_uv_e[uv_base + u] += acc_uv;

  // write back iu -> shared grad cache
  #pragma unroll
  for (int i = 0; i < I; ++i) {
    xiu.add(i, u, acc_iu[i]);
  }

  // jv partial -> register array
  #pragma unroll
  for (int j = 0; j < J; ++j) {
    jtmp[JB + j] += jloc[j];
  }
}

// ---- jv reduce helper (per-edge) ----
template<typename T, int JJ>
__device__ __forceinline__ void reduce_jtmp_write_grad_jv_per_edge(
    int tid, int lane, int warp, int num_warps, bool active,
    const T* __restrict__ jtmp,              // [JJ] register
    T* __restrict__ gx_jv_e,                 // [JV_TOTAL] per-edge grad
    const int32_t* __restrict__ jv_seg_offsets
){
  __shared__ T warp_sum_sh[8][JJ]; // blockDim<=256 -> max 8 warps

  // warp reduce each jj
#pragma unroll
  for (int jj = 0; jj < JJ; ++jj) {
    T v = active ? jtmp[jj] : (T)0;
    unsigned mask = __activemask();
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(mask, v, off);
    }
    if (lane == 0) warp_sum_sh[warp][jj] = v;
  }
  __syncthreads();

  // warp0 sum across warps (assume num_warps <= 8)
  if (warp == 0) {
#pragma unroll
    for (int jj = 0; jj < JJ; ++jj) {
      T v = (lane < num_warps) ? warp_sum_sh[lane][jj] : (T)0;
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) {  // good for <=8 warps
        v += __shfl_down_sync(0xffffffffu, v, off);
      }

      if (lane == 0) {
        // same jj mapping as your cwtp
        int jv_idx, j_local;
        if (jj == 0) { jv_idx = 0; j_local = 0; }
        else if (jj < 4) { jv_idx = 1; j_local = jj - 1; }
        else if (jj < 9) { jv_idx = 2; j_local = jj - 4; }
        else { jv_idx = 3; j_local = jj - 9; }

        int jv_base = (int)jv_seg_offsets[jv_idx];
        gx_jv_e[jv_base + j_local] += v; // per-edge write
      }
    }
  }
  __syncthreads();
}


template <typename T, int NUM_PATHS>
__global__ void tp17_bwd_fused_sender_major_densec_kernel(
    const T* __restrict__ grad_out_nodes,  // [N, K_TOTAL*U] (V=1)
    const T* __restrict__ x_uv,            // [E, UV_TOTAL]
    const T* __restrict__ x_iu,            // [N, IU_TOTAL]
    const T* __restrict__ x_jv,            // [E, JV_TOTAL]
    const T* __restrict__ c_all,           // [C_TOTAL]  dense cg

    const int32_t* __restrict__ row_ptr_s, // [N+1]  sender-major CSR
    const int32_t* __restrict__ receiver,  // [E]    receiver per edge

    const int32_t* __restrict__ uv_seg_offsets,
    const int32_t* __restrict__ iu_seg_offsets,
    const int32_t* __restrict__ jv_seg_offsets,
    const int32_t* __restrict__ kv_k_offsets,
    const int32_t* __restrict__ c_offsets, // [NUM_PATHS+1]

    T* __restrict__ grad_x_uv,             // [E, UV_TOTAL]
    T* __restrict__ grad_x_iu,             // [N, IU_TOTAL]
    T* __restrict__ grad_x_jv,             // [E, JV_TOTAL]

    int N, int E,
    int U, int K_TOTAL,
    int UV_TOTAL, int IU_TOTAL, int JV_TOTAL
){
  constexpr int JJ = 16;

  const int s   = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int u   = tid;

  if (s >= N) return;
  const bool active = (u < U);

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5; // assume <= 8

  // shared: [xiu][g_xiu]
  extern __shared__ unsigned char smem_raw[];
  T* s_iu  = reinterpret_cast<T*>(smem_raw);          // [IU_TOTAL]
  T* s_giu = s_iu + IU_TOTAL;                         // [IU_TOTAL]

  // load xiu(sender) and zero giu
  const T* xiu_s = x_iu + (int64_t)s * IU_TOTAL;
  for (int idx = tid; idx < IU_TOTAL; idx += blockDim.x) {
    s_iu[idx]  = xiu_s[idx];
    s_giu[idx] = (T)0;
  }
  __syncthreads();

  // edges for this sender
  const int e0 = row_ptr_s[s];
  const int e1 = row_ptr_s[s + 1];
  if (e0 >= e1) return;

  // iterate edges (serial in e, parallel in u)
  for (int e = e0; e < e1; ++e) {
    const int r = receiver[e];

    const T* go_edge = grad_out_nodes + (int64_t)r * (int64_t)(K_TOTAL * U);
    const T* x_uv_e  = x_uv + (int64_t)e * UV_TOTAL;
    const T* x_jv_e  = x_jv + (int64_t)e * JV_TOTAL;

    T* gx_uv_e = grad_x_uv + (int64_t)e * UV_TOTAL;
    T* gx_jv_e = grad_x_jv + (int64_t)e * JV_TOTAL;

    // per-edge register j partial
    T jtmp[JJ];
#pragma unroll
    for (int jj = 0; jj < JJ; ++jj) jtmp[jj] = (T)0;

    auto dispatch_tp4 = [&] __device__ () {
    // path0: (0,0,0,0)  cg [1,1,1]
    tp17_eval_single_edge_sender<T, 0>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(/*IU_IDX*/0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/0, /*IU_IDX*/0, /*JV_IDX*/0, /*KV_IDX*/0, /*P*/0);

    // path1: diag jk, i=0  cg [1,3,3]
    tp17_eval_diag_jk_i0_edge_sender<T, 3, 1>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(/*IU_IDX*/0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/1, /*IU_IDX*/0, /*JV_IDX*/1, /*KV_IDX*/1, /*P*/1);

    // path2: cg [1,5,5]
    tp17_eval_diag_jk_i0_edge_sender<T, 5, 4>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(/*IU_IDX*/0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/2, /*IU_IDX*/0, /*JV_IDX*/2, /*KV_IDX*/2, /*P*/2);

    // path3: cg [1,7,7]
    tp17_eval_diag_jk_i0_edge_sender<T, 7, 9>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(/*IU_IDX*/0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/3, /*IU_IDX*/0, /*JV_IDX*/3, /*KV_IDX*/3, /*P*/3);
    };

auto dispatch_tp10 = [&] __device__ () {
    // path0 indices: (0,0,0,0) cg [1,1,1]
    tp17_eval_single_edge_sender<T, 0>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/0, /*IU_IDX*/0, /*JV_IDX*/0, /*KV_IDX*/0, /*P*/0);

    // path1 indices: (2,0,1,2) cg [1,3,3]  (0,t,t)
    tp17_eval_diag_jk_i0_edge_sender<T, 3, 1>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/2, /*IU_IDX*/0, /*JV_IDX*/1, /*KV_IDX*/2, /*P*/1);

    // path2 indices: (5,0,2,5) cg [1,5,5]
    tp17_eval_diag_jk_i0_edge_sender<T, 5, 4>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/5, /*IU_IDX*/0, /*JV_IDX*/2, /*KV_IDX*/5, /*P*/2);

    // path3 indices: (8,0,3,8) cg [1,7,7]
    tp17_eval_diag_jk_i0_edge_sender<T, 7, 9>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/8, /*IU_IDX*/0, /*JV_IDX*/3, /*KV_IDX*/8, /*P*/3);

    // path4 indices: (3,1,0,3) cg [3,1,3]  (t,0,t)
    tp17_eval_diag_ik_j0_edge_sender<T, 3, 0>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/3, /*IU_IDX*/1, /*JV_IDX*/0, /*KV_IDX*/3, /*P*/4);

    // path5 indices: (1,1,1,1) cg [3,3,1]  (t,t,0)
    tp17_eval_diag_ij_k0_edge_sender<T, 3, 1>(
        u,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        /*UV_IDX*/1, /*IU_IDX*/1, /*JV_IDX*/1, /*KV_IDX*/1, /*P*/5);

    // path6 indices: (6,1,1,6) cg [3,3,5]
    tp17_path_eval_sharedc_edge_sender< 6,  6,1,1, 6, 3,3,5, 1, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    // path7 indices: (4,1,2,4) cg [3,5,3]
    tp17_path_eval_sharedc_edge_sender< 7,  4,1,2, 4, 3,5,3, 4, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    // path8 indices: (9,1,2,9) cg [3,5,7]
    tp17_path_eval_sharedc_edge_sender< 8,  9,1,2, 9, 3,5,7, 4, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    // path9 indices: (7,1,3,7) cg [3,7,5]
    tp17_path_eval_sharedc_edge_sender< 9,  7,1,3, 7, 3,7,5, 9, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);
};

auto dispatch_tp17 = [&] __device__ () {
    tp17_eval_single_edge_sender<T, 0>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        0,0,0,0, 0);

    tp17_eval_diag_jk_i0_edge_sender<T, 3, 1>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        3,0,1,3, 1);

    tp17_eval_diag_jk_i0_edge_sender<T, 5, 4>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        8,0,2,8, 2);

    tp17_eval_diag_jk_i0_edge_sender<T, 7, 9>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(0),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        13,0,3,13, 3);

    tp17_eval_diag_ik_j0_edge_sender<T, 3, 0>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        4,1,0,4, 4);

    tp17_eval_diag_ij_k0_edge_sender<T, 3, 1>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        1,1,1,1, 5);

    tp17_path_eval_sharedc_edge_sender< 6,  9,1,1, 9, 3,3,5, 1, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_path_eval_sharedc_edge_sender< 7,  5,1,2, 5, 3,5,3, 4, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_path_eval_sharedc_edge_sender< 8, 14,1,2,14, 3,5,7, 4, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_path_eval_sharedc_edge_sender< 9, 10,1,3,10, 3,7,5, 9, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(1),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_eval_diag_ik_j0_edge_sender<T, 5, 0>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        11,2,0,11, 10);

    tp17_path_eval_sharedc_edge_sender<11,  6,2,1, 6, 5,3,3, 1, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_path_eval_sharedc_edge_sender<12, 15,2,1,15, 5,3,7, 1, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_eval_diag_ij_k0_edge_sender<T, 5, 4>(
        u, go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL,
        2,2,2,2, 13);

    tp17_path_eval_sharedc_edge_sender<14, 12,2,2,12, 5,5,5, 4, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_path_eval_sharedc_edge_sender<15,  7,2,3, 7, 5,7,3, 9, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);

    tp17_path_eval_sharedc_edge_sender<16, 16,2,3,16, 5,7,7, 9, T>(
        u, lane,
        go_edge, x_uv_e, x_jv_e, c_all,
        gx_uv_e, jtmp, MAKE_XIUACC(2),
        uv_seg_offsets, jv_seg_offsets, kv_k_offsets, c_offsets,
        U, K_TOTAL, UV_TOTAL, JV_TOTAL);
    };

    if (active) {
      if constexpr (NUM_PATHS == 4) {
        dispatch_tp4();
      } else if constexpr (NUM_PATHS == 10) {
        dispatch_tp10();
      } else if constexpr (NUM_PATHS == 17) {
        dispatch_tp17();
      }
    }
    __syncthreads(); // make sure all threads finished jtmp

    // reduce jtmp -> grad_x_jv(edge)
    reduce_jtmp_write_grad_jv_per_edge<T, JJ>(
        tid, lane, warp, num_warps, active,
        jtmp, gx_jv_e, jv_seg_offsets);

    // next edge
  }

  // flush sender grad_x_iu once
  T* gxi_s = grad_x_iu + (int64_t)s * IU_TOTAL;
  for (int idx = tid; idx < IU_TOTAL; idx += blockDim.x) {
    gxi_s[idx] += s_giu[idx];
  }
}

// ----------------------------------------
// Launch
// ----------------------------------------
std::vector<torch::Tensor> tp17_bwd_fused_sender_major_densec_launch(
    torch::Tensor grad_out_nodes,  // [N, K_TOTAL*U] or [N, K_TOTAL, U]
    torch::Tensor x_uv,            // [E, UV_TOTAL]
    torch::Tensor x_iu,            // [N, IU_TOTAL]
    torch::Tensor x_jv,            // [E, JV_TOTAL]
    
    torch::Tensor row_ptr_s,       // [N+1] int32
    torch::Tensor receiver,        // [E]   int32

    torch::Tensor c_all,           // [C_TOTAL]
    torch::Tensor uv_seg_offsets,  // int32
    torch::Tensor iu_seg_offsets,  // int32
    torch::Tensor jv_seg_offsets,  // int32
    torch::Tensor kv_k_offsets,    // int32
    torch::Tensor c_offsets,       // [NUM_PATHS+1] int32

    int64_t U,
    int64_t V,
    int64_t K_TOTAL,
    int64_t num_paths             // 4/10/17 (or 16)
) {
  TORCH_CHECK(grad_out_nodes.is_cuda(), "grad_out_nodes must be CUDA");
  TORCH_CHECK(x_uv.is_cuda() && x_iu.is_cuda() && x_jv.is_cuda(), "x_uv/x_iu/x_jv must be CUDA");
  TORCH_CHECK(c_all.is_cuda(), "c_all must be CUDA");
  TORCH_CHECK(row_ptr_s.is_cuda() && receiver.is_cuda(), "row_ptr_s/receiver must be CUDA");
  TORCH_CHECK(uv_seg_offsets.is_cuda() && iu_seg_offsets.is_cuda() && jv_seg_offsets.is_cuda() &&
              kv_k_offsets.is_cuda() && c_offsets.is_cuda(), "offset tensors must be CUDA");

  // dtype checks
  TORCH_CHECK(row_ptr_s.scalar_type() == torch::kInt32, "row_ptr_s must be int32");
  TORCH_CHECK(receiver.scalar_type()  == torch::kInt32, "receiver must be int32");
  TORCH_CHECK(uv_seg_offsets.scalar_type() == torch::kInt32, "uv_seg_offsets must be int32");
  TORCH_CHECK(iu_seg_offsets.scalar_type() == torch::kInt32, "iu_seg_offsets must be int32");
  TORCH_CHECK(jv_seg_offsets.scalar_type() == torch::kInt32, "jv_seg_offsets must be int32");
  TORCH_CHECK(kv_k_offsets.scalar_type()   == torch::kInt32, "kv_k_offsets must be int32");
  TORCH_CHECK(c_offsets.scalar_type()      == torch::kInt32, "c_offsets must be int32");

  // contiguous
  grad_out_nodes = grad_out_nodes.contiguous();
  x_uv  = x_uv.contiguous();
  x_iu  = x_iu.contiguous();
  x_jv  = x_jv.contiguous();
  c_all = c_all.contiguous();

  row_ptr_s      = row_ptr_s.contiguous();
  receiver       = receiver.contiguous();
  uv_seg_offsets = uv_seg_offsets.contiguous();
  iu_seg_offsets = iu_seg_offsets.contiguous();
  jv_seg_offsets = jv_seg_offsets.contiguous();
  kv_k_offsets   = kv_k_offsets.contiguous();
  c_offsets      = c_offsets.contiguous();

  // sizes
  const int64_t N = x_iu.size(0);
  const int64_t E = x_uv.size(0);
  TORCH_CHECK(x_uv.dim() == 2, "x_uv must be [E, UV_TOTAL]");
  TORCH_CHECK(x_iu.dim() == 2, "x_iu must be [N, IU_TOTAL]");
  TORCH_CHECK(x_jv.dim() == 2, "x_jv must be [E, JV_TOTAL]");
  TORCH_CHECK(x_jv.size(0) == E, "x_jv batch(E) mismatch");
  TORCH_CHECK(receiver.numel() == E, "receiver must have length E");
  TORCH_CHECK(row_ptr_s.numel() == N + 1, "row_ptr_s must have length N+1");

  const int64_t UV_TOTAL = x_uv.size(1);
  const int64_t IU_TOTAL = x_iu.size(1);
  const int64_t JV_TOTAL = x_jv.size(1);

  // grad_out_nodes shape normalize to [N, K_TOTAL*U]
  if (grad_out_nodes.dim() == 3) {
    TORCH_CHECK(grad_out_nodes.size(0) == N, "grad_out_nodes N mismatch");
    TORCH_CHECK(grad_out_nodes.size(1) == K_TOTAL, "grad_out_nodes K_TOTAL mismatch");
    TORCH_CHECK(grad_out_nodes.size(2) == U, "grad_out_nodes U mismatch");
    grad_out_nodes = grad_out_nodes.view({N, K_TOTAL * U}).contiguous();
  } else {
    TORCH_CHECK(grad_out_nodes.dim() == 2, "grad_out_nodes must be [N, K_TOTAL*U] or [N,K_TOTAL,U]");
    TORCH_CHECK(grad_out_nodes.size(0) == N, "grad_out_nodes N mismatch");
    TORCH_CHECK(grad_out_nodes.size(1) == K_TOTAL * U, "grad_out_nodes second dim must be K_TOTAL*U");
  }

  // allocate grads
  auto grad_x_uv = torch::zeros_like(x_uv);
  auto grad_x_iu = torch::zeros_like(x_iu);
  auto grad_x_jv = torch::zeros_like(x_jv);

  // threads: keep <=256; recommend 256 (U=224 typical)
  int threads = 256;
  if (U <= 128) threads = 128;
  // still must be multiple of 32
  if (threads % 32) threads = ((threads + 31) / 32) * 32;
  TORCH_CHECK(threads <= 256, "threads must be <=256 for current warp0 reduce impl");

  dim3 grid((unsigned)N);
  dim3 block((unsigned)threads);

  // shared: s_iu + s_giu
  const size_t shmem = (size_t)2 * (size_t)IU_TOTAL * (size_t)x_uv.element_size();

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp17_bwd_fused_sender_major_densec_launch", [&] {
    const auto* go_ptr   = (const scalar_t*)grad_out_nodes.data_ptr<scalar_t>();
    const auto* xuv_ptr  = (const scalar_t*)x_uv.data_ptr<scalar_t>();
    const auto* xiu_ptr  = (const scalar_t*)x_iu.data_ptr<scalar_t>();
    const auto* xjv_ptr  = (const scalar_t*)x_jv.data_ptr<scalar_t>();
    const auto* c_ptr    = (const scalar_t*)c_all.data_ptr<scalar_t>();

    auto* gxuv_ptr = (scalar_t*)grad_x_uv.data_ptr<scalar_t>();
    auto* gxiu_ptr = (scalar_t*)grad_x_iu.data_ptr<scalar_t>();
    auto* gxjv_ptr = (scalar_t*)grad_x_jv.data_ptr<scalar_t>();

    const int32_t* rowptr_ptr = row_ptr_s.data_ptr<int32_t>();
    const int32_t* recv_ptr   = receiver.data_ptr<int32_t>();

    const int32_t* uv_off_ptr = uv_seg_offsets.data_ptr<int32_t>();
    const int32_t* iu_off_ptr = iu_seg_offsets.data_ptr<int32_t>();
    const int32_t* jv_off_ptr = jv_seg_offsets.data_ptr<int32_t>();
    const int32_t* kv_off_ptr = kv_k_offsets.data_ptr<int32_t>();
    const int32_t* c_off_ptr  = c_offsets.data_ptr<int32_t>();

    const int n_paths = (int)num_paths;

    if (n_paths == 4) {
      tp17_bwd_fused_sender_major_densec_kernel<scalar_t, 4>
          <<<grid, block, shmem, stream>>>(
              go_ptr, xuv_ptr, xiu_ptr, xjv_ptr, c_ptr,
              rowptr_ptr, recv_ptr,
              uv_off_ptr, iu_off_ptr, jv_off_ptr, kv_off_ptr, c_off_ptr,
              gxuv_ptr, gxiu_ptr, gxjv_ptr,
              (int)N, (int)E,
              (int)U, (int)K_TOTAL,
              (int)UV_TOTAL, (int)IU_TOTAL, (int)JV_TOTAL);
    } else if (n_paths == 10) {
      tp17_bwd_fused_sender_major_densec_kernel<scalar_t, 10>
          <<<grid, block, shmem, stream>>>(
              go_ptr, xuv_ptr, xiu_ptr, xjv_ptr, c_ptr,
              rowptr_ptr, recv_ptr,
              uv_off_ptr, iu_off_ptr, jv_off_ptr, kv_off_ptr, c_off_ptr,
              gxuv_ptr, gxiu_ptr, gxjv_ptr,
              (int)N, (int)E,
              (int)U, (int)K_TOTAL,
              (int)UV_TOTAL, (int)IU_TOTAL, (int)JV_TOTAL);
    } else if (n_paths == 17) {
      tp17_bwd_fused_sender_major_densec_kernel<scalar_t, 17>
          <<<grid, block, shmem, stream>>>(
              go_ptr, xuv_ptr, xiu_ptr, xjv_ptr, c_ptr,
              rowptr_ptr, recv_ptr,
              uv_off_ptr, iu_off_ptr, jv_off_ptr, kv_off_ptr, c_off_ptr,
              gxuv_ptr, gxiu_ptr, gxjv_ptr,
              (int)N, (int)E,
              (int)U, (int)K_TOTAL,
              (int)UV_TOTAL, (int)IU_TOTAL, (int)JV_TOTAL);
    } else {
      TORCH_CHECK(false, "Unsupported num_paths = ", n_paths, " (expected 4/10/17)");
    }

    CUDA_CHECK(cudaGetLastError());
  });

  return {grad_x_uv, grad_x_iu, grad_x_jv};
}

    

TORCH_LIBRARY(mptp_bwd, m)
{
    m.def("backward", &tp17_bwd_fused_sender_major_densec_launch);
}