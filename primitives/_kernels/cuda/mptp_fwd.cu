#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cub/cub.cuh>
#include "cuda_utils.hpp"

// ----------------------------------------------------------------------------
// 1) row_ptr_s 构建：sender 已排序（非降序），写 run 边界
// row_ptr_s: [N+1]，初始化为 -1，且 row_ptr_s[0]=0,row_ptr_s[N]=E
// ----------------------------------------------------------------------------
__global__ void build_row_ptr_from_sorted_sender(
    const int32_t* __restrict__ sender, // [E], sorted
    int32_t* __restrict__ row_ptr_s,    // [N+1]
    int E)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;

  int32_t s = sender[e];

  // run start
  if (e == 0 || sender[e - 1] != s) {
    row_ptr_s[s] = e;
  }
  // run end -> s+1
  if (e == E - 1 || sender[e + 1] != s) {
    row_ptr_s[s + 1] = e + 1;
  }
}

// CUB inclusive max scan to fill holes in row_ptr_s
static void fill_row_ptr_holes_prefix_max_int32(
    int32_t* d_row_ptr, int n_plus_1, cudaStream_t stream)
{
  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(
      d_temp, temp_bytes,
      d_row_ptr, d_row_ptr,
      cub::Max(),
      n_plus_1,
      stream);
  CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
  cub::DeviceScan::InclusiveScan(
      d_temp, temp_bytes,
      d_row_ptr, d_row_ptr,
      cub::Max(),
      n_plus_1,
      stream);
  CUDA_CHECK(cudaFreeAsync(d_temp, stream));
}

// ----------------------------------------------------------------------------
// 2) 融合 kernel：sender-major CSR
//    一个 block 处理一个 sender s；threadIdx.x 对应 u
//    遍历该 sender 的边段 [row_ptr_s[s], row_ptr_s[s+1])
//    对每条边在线计算 TP(group-k) 的输出并 atomicAdd 到 out_nodes[receiver]
// ----------------------------------------------------------------------------
template <typename scalar_t, int MAX_K_DIM>
__global__ void tp_channel_wise_sparse_groupk_fused_scatter_sender_major_kernel(
    // per-edge inputs
    const scalar_t* __restrict__ x_uv_e,     // [E, UV_TOTAL]
    const scalar_t* __restrict__ x_jv_e,     // [E, JV_TOTAL]

    // per-node inputs (sender feats)
    const scalar_t* __restrict__ x_iu_n,     // [N, IU_TOTAL]

    // graph
    const int32_t* __restrict__ receiver,   // [E]
    const int32_t* __restrict__ row_ptr_s,  // [N+1]

    // TP meta (same as your group-k kernel)
    const int32_t* __restrict__ path_indices,   // [P,4]
    const int32_t* __restrict__ k_dims,         // [P]
    const int32_t* __restrict__ iu_seg_offsets, // [iu_seg_count]
    const int32_t* __restrict__ jv_seg_offsets, // [jv_seg_count]
    const int32_t* __restrict__ kv_k_offsets,   // [kv_seg_count]

    const int32_t* __restrict__ nnz_per_path,   // [P]
    const int32_t* __restrict__ nnz_offsets,    // [P]
    const int32_t* __restrict__ nnz_k_offsets,  // [P*MAX_K_DIM]
    const int32_t* __restrict__ nnz_k_counts,   // [P*MAX_K_DIM]

    const uint8_t* __restrict__ cg_i_all,       // [nnz_total]
    const uint8_t* __restrict__ cg_j_all,       // [nnz_total]
    const scalar_t* __restrict__ cg_val_all,    // [nnz_total]

    // out
    scalar_t* __restrict__ out_nodes,           // [N, K_TOTAL*U*V]  (scatter_sum result, flattened)

    // sizes
    int N, int E,
    int UV_TOTAL, int IU_TOTAL, int JV_TOTAL,
    int K_TOTAL, int U, int V,
    int num_paths)
{
  int s = blockIdx.x;     // sender node
  int u = threadIdx.x;    // u channel
  if (s >= N || u >= U) return;

  // ---- cache sender feats to shared (reuse across edges of this sender) ----
  extern __shared__ unsigned char smem_raw[];
  scalar_t* s_iu = reinterpret_cast<scalar_t*>(smem_raw); // [IU_TOTAL]

  const scalar_t* x_iu_s = x_iu_n + (size_t)s * IU_TOTAL;
  for (int idx = u; idx < IU_TOTAL; idx += blockDim.x) {
    s_iu[idx] = x_iu_s[idx];
  }
  __syncthreads();

  int e0 = row_ptr_s[s];
  int e1 = row_ptr_s[s + 1];
  if (e0 >= e1) return;

  int last_r = -1;

  for (int e = e0; e < e1; ++e) {
    int r = receiver[e];

    const scalar_t* x_uv = x_uv_e + (size_t)e * UV_TOTAL;
    const scalar_t* x_jv = x_jv_e + (size_t)e * JV_TOTAL;

    // ---- same inner structure as your group-k TP ----
    for (int p = 0; p < num_paths; ++p) {
      int uv_idx = path_indices[p * 4 + 0];
      int iu_idx = path_indices[p * 4 + 1];
      int jv_idx = path_indices[p * 4 + 2];
      int kv_idx = path_indices[p * 4 + 3];

      int k_dim   = k_dims[p];
      int nnz     = nnz_per_path[p];
      int nnz_off = nnz_offsets[p];

      if (k_dim <= 0 || nnz <= 0) continue;
      if (k_dim > MAX_K_DIM) return;

      int uv_base = uv_idx * (U * V);
      int iu_base = iu_seg_offsets[iu_idx];
      int jv_base = jv_seg_offsets[jv_idx];
      int k_base  = kv_k_offsets[kv_idx];

      for (int v_idx = 0; v_idx < V; ++v_idx) {
        scalar_t xuv_uv = x_uv[uv_base + u * V + v_idx];

        #pragma unroll
        for (int k_local = 0; k_local < MAX_K_DIM; ++k_local) {
          if (k_local >= k_dim) break;

          int meta_idx    = p * MAX_K_DIM + k_local;
          int local_off   = nnz_k_offsets[meta_idx];
          int local_count = nnz_k_counts[meta_idx];
          if (local_count <= 0) continue;

          scalar_t acc = (scalar_t)0;

          #pragma unroll 1
          for (int tt = 0; tt < local_count; ++tt) {
            int t   = local_off + tt;
            int idx = nnz_off + t;

            int i = (int)cg_i_all[idx];
            int j = (int)cg_j_all[idx];
            scalar_t c = cg_val_all[idx];

            scalar_t xiu = s_iu[iu_base + i * U + u];
            scalar_t xjv = x_jv[jv_base + j * V + v_idx];

            // acc += c * xuv * xiu * xjv
            acc = fma(c * xjv, xuv_uv * xiu, acc);
          }

          int global_k  = k_base + k_local;
          int out_index = (global_k * U + u) * V + v_idx; // [0, K_TOTAL*U*V)

          // fused scatter_sum: out_nodes[r, out_index] += acc
          atomicAdd(&out_nodes[(size_t)r * (K_TOTAL * U * V) + out_index], acc);
        }
      }
    }

    last_r = r;
  }
}

std::vector<torch::Tensor> tp_channel_wise_fused_sender_scatter_launch(
    torch::Tensor x_uv,            // tp_weights [E, UV_TOTAL]  (per-edge)
    torch::Tensor x_iu,            // node_feats [N, IU_TOTAL]  (per-node sender feats)
    torch::Tensor x_jv,            // edge_attrs [E, JV_TOTAL]  (per-edge)
    torch::Tensor c_all,           // [sum_p i_p*j_p*k_p]
    torch::Tensor path_indices,    // [P,4] int32
    torch::Tensor i_dims,          // [P] int32
    torch::Tensor j_dims,          // [P] int32
    torch::Tensor k_dims,          // [P] int32
    torch::Tensor c_offsets,       // [num_paths], int32
    torch::Tensor iu_seg_offsets,  // int32
    torch::Tensor jv_seg_offsets,  // int32
    torch::Tensor kv_k_offsets,    // int32

    torch::Tensor nnz_per_path,    // [P] int32
    torch::Tensor nnz_offsets,     // [P] int32
    torch::Tensor nnz_k_offsets,   // [P*MAX_K_DIM] int32
    torch::Tensor nnz_k_counts,    // [P*MAX_K_DIM] int32

    torch::Tensor cg_i_all,        // [nnz_total] uint8
    torch::Tensor cg_j_all,        // [nnz_total] uint8
    torch::Tensor cg_k_all,        // [nnz_total] uint8
    torch::Tensor cg_val_all,      // [nnz_total] float/double

    torch::Tensor sender,          // [E] int32, sorted
    torch::Tensor receiver,        // [E] int32

    const int64_t U,
    const int64_t V,
    const int64_t K_TOTAL
) {
  TORCH_CHECK(x_uv.is_cuda() && x_iu.is_cuda() && x_jv.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(cg_val_all.is_cuda(), "cg_val_all must be CUDA");
  TORCH_CHECK(sender.is_cuda() && receiver.is_cuda(), "sender/receiver must be CUDA");

  TORCH_CHECK(sender.scalar_type() == torch::kInt32, "sender must be int32 (sorted)");
  TORCH_CHECK(receiver.scalar_type() == torch::kInt32, "receiver must be int32");
  TORCH_CHECK(path_indices.scalar_type() == torch::kInt32, "path_indices must be int32");
  TORCH_CHECK(k_dims.scalar_type() == torch::kInt32, "k_dims must be int32");

  x_uv = x_uv.contiguous();
  x_iu = x_iu.contiguous();
  x_jv = x_jv.contiguous();

  path_indices   = path_indices.contiguous();
  k_dims         = k_dims.contiguous();
  iu_seg_offsets = iu_seg_offsets.contiguous();
  jv_seg_offsets = jv_seg_offsets.contiguous();
  kv_k_offsets   = kv_k_offsets.contiguous();

  nnz_per_path  = nnz_per_path.contiguous();
  nnz_offsets   = nnz_offsets.contiguous();
  nnz_k_offsets = nnz_k_offsets.contiguous();
  nnz_k_counts  = nnz_k_counts.contiguous();

  cg_i_all    = cg_i_all.contiguous();
  cg_j_all    = cg_j_all.contiguous();
  cg_val_all  = cg_val_all.contiguous();

  sender   = sender.contiguous();
  receiver = receiver.contiguous();

  // sizes
  int64_t E = x_uv.size(0);
  int64_t N = x_iu.size(0);

  TORCH_CHECK(x_jv.size(0) == E, "x_jv must have same first dim as x_uv (E)");
  TORCH_CHECK(sender.numel() == E && receiver.numel() == E, "sender/receiver size must be E");

  int UV_TOTAL = (int)x_uv.size(1);
  int IU_TOTAL = (int)x_iu.size(1);
  int JV_TOTAL = (int)x_jv.size(1);

  TORCH_CHECK(path_indices.dim() == 2 && path_indices.size(1) == 4, "path_indices must be [P,4]");
  int num_paths = (int)path_indices.size(0);

  constexpr int MAX_K_DIM = 8;

  auto row_ptr_s = torch::empty({N + 1}, torch::TensorOptions().dtype(torch::kInt32).device(sender.device()));
  row_ptr_s.fill_(-1);
  row_ptr_s.index_put_({0}, 0);
  row_ptr_s.index_put_({N}, (int)E);

  auto stream = at::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = (E + threads - 1) / threads;
  build_row_ptr_from_sorted_sender<<<blocks, threads, 0, stream>>>(
      sender.data_ptr<int32_t>(),
      row_ptr_s.data_ptr<int32_t>(),
      (int)E);
  CUDA_CHECK(cudaGetLastError());

  // fill holes with inclusive max scan
  fill_row_ptr_holes_prefix_max_int32(row_ptr_s.data_ptr<int32_t>(), (int)(N + 1), stream);
  CUDA_CHECK(cudaGetLastError());

  int64_t C = K_TOTAL * U * V;
  auto out_nodes = torch::zeros({N, C}, x_uv.options());

  int block_u = (int)U;
  if (block_u < 32) block_u = 32;
  if (block_u > 1024) block_u = 1024;
  TORCH_CHECK((int)U <= block_u, "U > 1024 not supported in this mapping (needs 2D tiling)");

  dim3 grid((unsigned int)N, 1, 1);
  dim3 block((unsigned int)block_u, 1, 1);

  size_t smem_bytes = (size_t)IU_TOTAL * x_uv.element_size(); // only cache sender feats

  AT_DISPATCH_FLOATING_TYPES(x_uv.scalar_type(), "tp_groupk_fused_scatter_sender_major", [&](){
    tp_channel_wise_sparse_groupk_fused_scatter_sender_major_kernel<scalar_t, MAX_K_DIM>
      <<<grid, block, smem_bytes, stream>>>(
        x_uv.data_ptr<scalar_t>(),
        x_jv.data_ptr<scalar_t>(),
        x_iu.data_ptr<scalar_t>(),
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
        out_nodes.data_ptr<scalar_t>(),
        (int)N, (int)E,
        (int)UV_TOTAL, (int)IU_TOTAL, (int)JV_TOTAL,
        (int)K_TOTAL, (int)U, (int)V,
        (int)num_paths
      );
  });

  CUDA_CHECK(cudaGetLastError());
  return {out_nodes, row_ptr_s} ; // out_nodes: [N, K_TOTAL*U*V]
}

TORCH_LIBRARY(mptp_fwd, m)
{
    m.def("forward", &tp_channel_wise_fused_sender_scatter_launch);
}