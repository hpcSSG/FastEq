#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void stc_fwd_kernel(
    const scalar_t* __restrict__ x1,        // [B, num_a, u]
    const scalar_t* __restrict__ x0_g,      // [B, num_i, u]
    const scalar_t* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,          // [num_paths, 5]
    const int* __restrict__ path_lens,      // [num_paths]
    scalar_t* __restrict__ out,             // [B, num_out_segments, u] 
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared  = reinterpret_cast<scalar_t*>(smem);                 // size: num_a * u
    scalar_t* x0g_shared = x1_shared + static_cast<size_t>(num_a) * u;        // size: num_i * u

    int b = blockIdx.x;   // batch index
    int j = threadIdx.x;  // channel index

    if (b >= B || j >= u) return;

    // 每个 block 对应一个样本 b，每个线程处理一个 channel j
    // 预加载 x1[b, :, :] 到 shared memory
    for (int a = threadIdx.x; a < num_a * u; a += blockDim.x) {
        int a_idx = a / u;
        int u_idx = a % u;
        x1_shared[a] = x1[b * num_a * u + a_idx * u + u_idx];
    }

    // 预加载 x0_g[b, :, :] 到 shared memory
    for (int i = threadIdx.x; i < num_i * u; i += blockDim.x) {
        int i_idx = i / u;
        int u_idx = i % u;
        x0g_shared[i] = x0_g[b * num_i * u + i_idx * u + u_idx];
    }

    __syncthreads();

    // 对每个 path 独立计算并写入对应的 out 段
    for (int p = 0; p < num_paths; ++p) {
        const scalar_t coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;   // paths[p, :] 起始位置

        // a_idx: 第一个 x1 段索引
        const int a_idx = path[0];

        // d_idx: x0_g 的段索引（倒数第二个）
        const int d_idx = (len == 3) ? path[1] : path[len - 2];

        // out_seg: 输出段索引（最后一个）
        const int out_seg = (len == 3) ? path[2] : path[len - 1];

        if (out_seg < 0 || out_seg >= num_out_segments) {
            continue;
        }

        scalar_t val = x1_shared[a_idx * u + j];

        if (len >= 4) {
            int b_idx = path[1];
            val *= x1_shared[b_idx * u + j];
        }
        if (len == 5) {
            int c_idx = path[2];
            val *= x1_shared[c_idx * u + j];
        }

        val *= x0g_shared[d_idx * u + j];
        val *= coeff;

        const int out_idx = ((b * num_out_segments) + out_seg) * u + j;
        atomicAdd(&out[out_idx], val);
    }
}

template <typename scalar_t>
__global__ void stc_fwd_kernel_opt(
    const scalar_t* __restrict__ x1,        // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,      // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,          // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,      // [num_paths]
    scalar_t* __restrict__ out,             // [B, num_out_segments, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared  = reinterpret_cast<scalar_t*>(smem);                         // size: num_a * u
    scalar_t* x0g_shared = x1_shared + static_cast<size_t>(num_a) * u;                // size: num_i * u

    const int b = blockIdx.x;   // batch index
    const int j = threadIdx.x;  // channel index

    if (b >= B || j >= u) return;

    // ---------------- 预加载 x1[b,:,:], x0_g[b,:,:] 到 shared ----------------
    for (int a = threadIdx.x; a < num_a * u; a += blockDim.x) {
        int a_idx = a / u;
        int u_idx = a % u;
        x1_shared[a] = x1[b * num_a * u + a_idx * u + u_idx];
    }

    for (int i = threadIdx.x; i < num_i * u; i += blockDim.x) {
        int i_idx = i / u;
        int u_idx = i % u;
        x0g_shared[i] = x0_g[b * num_i * u + i_idx * u + u_idx];
    }

    __syncthreads();

    // ---------------- 为每个 out_segment 准备 accumulator ----------------
    // 可以模板化 MAX_OUT_SEG
    scalar_t acc_local_max[10];  // small/meidum/large num_out_segments = 1/4/9;
    const int max_seg = 10;
    const int seg_lim = (num_out_segments < max_seg) ? num_out_segments : max_seg;

    for (int s = 0; s < seg_lim; ++s) {
        acc_local_max[s] = scalar_t(0);
    }

    // ---------------- 遍历所有 paths，累加到对应的 out_segment ----------------
    
    for (int p = 0; p < num_paths; ++p) {
        const scalar_t coeff = coeffs[p];
        const int len        = path_lens[p];
        const int* path      = paths + p * 5;

        // x1 indices
        const int a_idx = path[0];

        // x0_g index d_idx: 倒数第二个
        const int d_idx = (len == 3) ? path[1] : path[len - 2];

        // 输出段索引 out_seg: 最后一个
        const int out_seg = (len == 3) ? path[2] : path[len - 1];

        if (out_seg < 0 || out_seg >= seg_lim) {
            continue; // 越界就直接跳过（理论上不该发生）
        }

        scalar_t val = x1_shared[a_idx * u + j];

        if (len >= 4) {
            int b_idx = path[1];
            val *= x1_shared[b_idx * u + j];
        }
        if (len == 5) {
            int c_idx = path[2];
            val *= x1_shared[c_idx * u + j];
        }

        val *= x0g_shared[d_idx * u + j];
        val *= coeff;

        // 累加到该 path 对应的 out_segment
        acc_local_max[out_seg] += val;
    }

    // ---------------- 循环结束后，统一写回 global out ----------------
    // out 逻辑形状: [B, num_out_segments, u]
    for (int s = 0; s < seg_lim; ++s) {
        const int out_idx = ((b * num_out_segments) + s) * u + j;
        // 每个 (b, s, j) 只由该 thread 写一次，**不需要 atomicAdd**
        out[out_idx] = acc_local_max[s];
    }
}

template <typename scalar_t>
__global__ void stc_fwd_kernel_tiled(
    const scalar_t* __restrict__ x1,        // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,      // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,          // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,      // [num_paths]
    scalar_t* __restrict__ out,             // [B, num_out_segments, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments, int TILE_U)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared  = reinterpret_cast<scalar_t*>(smem);                             // [num_a, TILE_U]
    scalar_t* x0g_shared = x1_shared + static_cast<size_t>(num_a) * TILE_U;               // [num_i, TILE_U]

    const int b  = blockIdx.x;   // batch index
    const int tx = threadIdx.x;  // 0..TILE_U-1, 局部 channel index
    const int MAX_OUT_SEG = 10;  // small/meidum/large num_out_segments = 1/4/9
    if (b >= B) return;
    if (num_out_segments > MAX_OUT_SEG) return;

    // 沿 u 方向分 tile 处理
    for (int base_u = 0; base_u < u; base_u += TILE_U) {
        const int j = base_u + tx;  // 全局 channel index

        // ---------------- 预加载 x1[b, :, j..j+TILE_U) 到 shared ----------------
        // x1_shared shape: [num_a, TILE_U] -> 下标 a*TILE_U + local_u
        for (int idx = tx; idx < num_a * TILE_U; idx += blockDim.x) {
            int a_idx    = idx / TILE_U;
            int local_u  = idx % TILE_U;
            int j_global = base_u + local_u;
            scalar_t v   = scalar_t(0);
            if (j_global < u) {
                v = x1[( (b * num_a + a_idx) * u ) + j_global];
            }
            x1_shared[idx] = v;
        }

        // ---------------- 预加载 x0_g[b, :, j..j+TILE_U) 到 shared ----------------
        // x0g_shared shape: [num_i, TILE_U]
        for (int idx = tx; idx < num_i * TILE_U; idx += blockDim.x) {
            int i_idx    = idx / TILE_U;
            int local_u  = idx % TILE_U;
            int j_global = base_u + local_u;
            scalar_t v   = scalar_t(0);
            if (j_global < u) {
                v = x0_g[( (b * num_i + i_idx) * u ) + j_global];
            }
            x0g_shared[idx] = v;
        }

        __syncthreads();

        if (j >= u) {
            __syncthreads();
            continue;   // 这个 thread 在当前 tile 无效
        }

        // 每个线程负责 (b, j)，对所有 path 计算贡献，分段累加
        scalar_t acc[MAX_OUT_SEG];
        const int seg_lim = num_out_segments;
        for (int s = 0; s < seg_lim; ++s) {
            acc[s] = scalar_t(0);
        }

        for (int p = 0; p < num_paths; ++p) {
            const scalar_t coeff = coeffs[p];
            const int len        = path_lens[p];
            const int* path      = paths + p * 5;

            const int a_idx   = path[0];
            const int d_idx   = (len == 3) ? path[1] : path[len - 2];
            const int out_seg = (len == 3) ? path[2] : path[len - 1];

            if (out_seg < 0 || out_seg >= seg_lim)
                continue;

            const int local_a = a_idx * TILE_U + (j - base_u);
            const int local_d = d_idx * TILE_U + (j - base_u);

            scalar_t val = x1_shared[local_a];

            if (len >= 4) {
                int b_idx = path[1];
                const int local_b = b_idx * TILE_U + (j - base_u);
                val *= x1_shared[local_b];
            }
            if (len == 5) {
                int c_idx = path[2];
                const int local_c = c_idx * TILE_U + (j - base_u);
                val *= x1_shared[local_c];
            }

            val *= x0g_shared[local_d];
            val *= coeff;

            acc[out_seg] += val;
        }

        // 写回 out[b, out_seg, j]
        for (int s = 0; s < seg_lim; ++s) {
            const int out_idx = ((b * num_out_segments) + s) * u + j;
            out[out_idx] = acc[s];
        }

        __syncthreads();
    }
}


template <typename scalar_t>
__global__ void stc_fwd_kernel_tiled_opt(
    const scalar_t* __restrict__ x1,        // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,      // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,          // [num_paths, 5]
    const int* __restrict__ path_lens,      // [num_paths]
    scalar_t* __restrict__ out,             // [B, num_out_segments, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments, int TILE_U)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    // 只缓存 x1: [num_a, TILE_U]
    scalar_t* x1_shared = reinterpret_cast<scalar_t*>(smem);

    const int b  = blockIdx.x;   // batch index
    const int tx = threadIdx.x;  // 0 .. TILE_U-1

    if (b >= B) return;
    const int MAX_OUT_SEG = 9;  // small/meidum/large num_out_segments = 1/4/9
    if (num_out_segments > MAX_OUT_SEG) return;

    // 沿 u 方向分 tile
    for (int base_u = 0; base_u < u; base_u += TILE_U) {
        const int j = base_u + tx;  // 当前线程负责的 global u index

        // --- 预加载 x1[b, :, base_u : base_u + TILE_U) 到 shared ---
        for (int idx = tx; idx < num_a * TILE_U; idx += blockDim.x) {
            int a_idx    = idx / TILE_U;        // 0 .. num_a-1
            int local_u  = idx % TILE_U;        // 0 .. TILE_U-1
            int j_global = base_u + local_u;    // 对应全局 u 维 index

            scalar_t v = scalar_t(0);
            if (j_global < u) {
                v = x1[((b * num_a + a_idx) * u) + j_global];
            }
            x1_shared[idx] = v;
        }

        __syncthreads();

        if (j >= u) {
            __syncthreads();
            continue;
        }

        const int local_u = j - base_u;

        // 对这个 (b, j) 的所有 out_segments 做局部累加
        scalar_t acc[MAX_OUT_SEG];
        const int seg_lim = num_out_segments;
        for (int s = 0; s < seg_lim; ++s) acc[s] = scalar_t(0);

        // ---- 遍历所有 paths ----
        for (int p = 0; p < num_paths; ++p) {
            const scalar_t coeff = coeffs[p];
            const int len        = path_lens[p];
            const int* path      = paths + p * 5;

            const int a_idx   = path[0];
            const int d_idx   = (len == 3) ? path[1] : path[len - 2];
            const int out_seg = (len == 3) ? path[2] : path[len - 1];

            if (out_seg < 0 || out_seg >= seg_lim)
                continue;

            const int local_a = a_idx * TILE_U + local_u;

            scalar_t val = x1_shared[local_a];

            if (len >= 4) {
                const int b_idx   = path[1];
                const int local_b = b_idx * TILE_U + local_u;
                val *= x1_shared[local_b];
            }
            if (len == 5) {
                const int c_idx   = path[2];
                const int local_c = c_idx * TILE_U + local_u;
                val *= x1_shared[local_c];
            }

            // x0_g 直接 global load
            const scalar_t x0_val =
                x0_g[((b * num_i + d_idx) * u) + j];

            val *= x0_val;
            val *= coeff;

            acc[out_seg] += val;
        }

        // ---- 将局部 acc 写回 global out ----
        for (int s = 0; s < seg_lim; ++s) {
            const int out_idx = ((b * num_out_segments) + s) * u + j;
            out[out_idx] = acc[s];
        }

        __syncthreads();
    }
}


template <typename scalar_t>
__global__  void stc_fwd_kernel_notiled(
    const scalar_t* __restrict__ x1,        // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,      // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,          // [num_paths, 5]
    const int* __restrict__ path_lens,      // [num_paths]
    scalar_t* __restrict__ out,             // [B, num_out_segments, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared   = reinterpret_cast<scalar_t*>(smem);                        // [num_a * u]
    scalar_t* out_shared  = x1_shared + static_cast<size_t>(num_a) * u;               // [num_out_segments * u]

    const int b  = blockIdx.x;   // batch index
    const int j  = threadIdx.x;  // channel index

    if (b >= B) return;
    if (j >= u) return;

    const int MAX_OUT_SEG = 9;  // small/meidum/large num_out_segments = 1/4/9
    if (num_out_segments > MAX_OUT_SEG) return;

    const int seg_lim = num_out_segments;

    // -----------------------------
    // 1. 预加载 x1[b, :, :] 到 shared
    // -----------------------------
    for (int idx = j; idx < num_a * u; idx += blockDim.x) {
        int a_idx = idx / u;   // 0 .. num_a-1
        int jj    = idx % u;   // 0 .. u-1
        x1_shared[idx] = x1[((b * num_a + a_idx) * u) + jj];
    }

    // -----------------------------
    // 2. 初始化 out_shared[seg, j] = 0
    // 每个线程负责所有 seg 对应的这一列 j
    // -----------------------------
    for (int s = 0; s < seg_lim; ++s) {
        out_shared[s * u + j] = scalar_t(0);
    }

    __syncthreads();

    // -----------------------------
    // 3. 遍历所有 paths，累加到 out_shared[seg, j]
    // -----------------------------
    for (int p = 0; p < num_paths; ++p) {
        const scalar_t coeff = coeffs[p];
        const int len        = path_lens[p];
        const int* path      = paths + p * 5;

        const int a_idx = path[0];
        int d_idx, out_seg;
        int b_idx = -1, c_idx = -1;

        if (len == 3) {
            d_idx   = path[1];
            out_seg = path[2];
        } else if (len == 4) {
            b_idx   = path[1];
            d_idx   = path[2];
            out_seg = path[3];
        } else {  // len == 5
            b_idx   = path[1];
            c_idx   = path[2];
            d_idx   = path[3];
            out_seg = path[4];
        }

        if (out_seg < 0 || out_seg >= seg_lim)
            continue;

        // x1 部分：从 shared 读
        scalar_t val = x1_shared[a_idx * u + j];

        if (len >= 4) {
            val *= x1_shared[b_idx * u + j];
        }
        if (len == 5) {
            val *= x1_shared[c_idx * u + j];
        }

        // x0_g 直接 global load
        const scalar_t x0_val =
            x0_g[((b * num_i + d_idx) * u) + j];

        val *= x0_val * coeff;

        // 这里 (out_seg, j) 对于 block 内是 thread 独占的，不需要 atomic
        out_shared[out_seg * u + j] += val;
    }

    __syncthreads();

    // -----------------------------
    // 4. 把 out_shared 写回 global out[b, seg, j]
    // -----------------------------
    for (int s = 0; s < seg_lim; ++s) {
        out[((b * num_out_segments) + s) * u + j] = out_shared[s * u + j];
    }
}


// --------------------------------------
// Launcher：支持 float32 / float64
// --------------------------------------
at::Tensor stc_fwd_launcher(
    at::Tensor x1,            // [B, num_a, u]
    at::Tensor x0_g,          // [B, num_i, u]
    at::Tensor coeffs,        // [num_paths]
    at::Tensor paths_tensor,  // [num_paths, 5]
    at::Tensor path_lens,     // [num_paths]
    const int64_t num_out_segments)     
{
    TORCH_CHECK(x1.is_cuda(), "x1 must be CUDA");
    TORCH_CHECK(x0_g.is_cuda(), "x0_g must be CUDA");
    TORCH_CHECK(coeffs.is_cuda(), "coeffs must be CUDA");
    TORCH_CHECK(paths_tensor.is_cuda(), "paths_tensor must be CUDA");
    TORCH_CHECK(path_lens.is_cuda(), "path_lens must be CUDA");

    auto dtype = x1.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "x1 must be float32 or float64");
    TORCH_CHECK(x0_g.scalar_type() == dtype,
                "x0_g must have same dtype as x1");
    TORCH_CHECK(coeffs.scalar_type() == dtype,
                "coeffs must have same dtype as x1");

    x1           = x1.contiguous();
    x0_g         = x0_g.contiguous();
    coeffs       = coeffs.contiguous();
    paths_tensor = paths_tensor.contiguous();
    path_lens    = path_lens.contiguous();

    const int B         = x1.size(0);
    const int num_a     = x1.size(1);
    const int u         = x1.size(2);
    const int num_i     = x0_g.size(1);
    const int num_paths = coeffs.size(0);
    const int TILE_U = u; // 可调, shared memory 约束， small/meidum/large u = 96, 128, 224/ TILE_U = 96, 96, 64

    at::Tensor out = at::zeros({B, num_out_segments * u}, x1.options()).contiguous();

    dim3 blockDim(TILE_U); 
    // dim3 gridDim(B);    // 每个 block 处理一个样本 b
    dim3 gridDim(B);

    //const size_t shared_elems = static_cast<size_t>(num_a) * TILE_U
    //                          + static_cast<size_t>(num_i) * TILE_U;

    //const size_t shared_elems     = static_cast<size_t>(num_a) * TILE_U;
    const size_t shared_elems =
        static_cast<size_t>(num_a) * u +
        static_cast<size_t>(num_out_segments) * u;  // x1_shared + out_shared
    const size_t shared_mem_bytes = shared_elems * x1.element_size();
    

    cudaStream_t cur_stream =
        c10::cuda::getCurrentCUDAStream(x1.device().index()).stream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "stc_fwd_kernel_tiled", [&] {
        using scalar_t = scalar_t;
        
        stc_fwd_kernel_notiled<scalar_t><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
            x1.data_ptr<scalar_t>(),
            x0_g.data_ptr<scalar_t>(),
            coeffs.data_ptr<scalar_t>(),
            paths_tensor.data_ptr<int>(),
            path_lens.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            B, num_paths, u, num_a, num_i, num_out_segments);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// --------------------------------------
// Torch 注册
// --------------------------------------
TORCH_LIBRARY(stc_fwd, m)
{
    m.def("forward", &stc_fwd_launcher);
}
