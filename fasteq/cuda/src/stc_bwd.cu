#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr);  \
  if (_err != cudaSuccess)    \
    AT_ERROR("CUDA error: ", cudaGetErrorString(_err), " at ", __FILE__, ":", __LINE__); \
} while(0)


template<typename scalar_t>
__global__ void stc_bwd_kernel(
    const scalar_t* __restrict__ grad_out, // [B, u] -> flattened
    const scalar_t* __restrict__ x1,       // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,     // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,   // [num_paths]
    const int* __restrict__ paths,         // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,     // [num_paths]
    scalar_t* __restrict__ grad_x1,        // [B, num_a, u] -> flattened (output)
    int B, int num_paths, int u, int num_a, int num_i)
{
    // 动态 shared memory，按标量类型对齐
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared  = reinterpret_cast<scalar_t*>(smem);           // size: num_a * u
    scalar_t* x0g_shared = x1_shared + num_a * u;                       // size: num_i * u

    const int b = blockIdx.x;
    const int j = threadIdx.x;

    if (b >= B || j >= u) return;

    // 预加载 x1[b, :, :] 与 x0_g[b, :, :]（与前向一致）
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

    const scalar_t go = grad_out[b * u + j];  // dL/d out[b,j]

    // 遍历所有 path，累加对参与的 x1 索引的梯度
    for (int p = 0; p < num_paths; ++p) {
        const scalar_t coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;

        // 基本索引
        const int a_idx = path[0];
        const int i_idx = (len == 3) ? path[1] : path[len - 2];

        // 参与乘积的值
        const scalar_t x1_a = x1_shared[a_idx * u + j];
        const scalar_t x0g  = x0g_shared[i_idx * u + j];

        if (len == 3) {
            // path: [a, i, *]
            const scalar_t contrib_a = go * coeff * x0g;
            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);

        } else if (len == 4) {
            // path: [a, b, i, *]
            const int b_idx = path[1];
            const scalar_t x1_b = x1_shared[b_idx * u + j];

            const scalar_t base = go * coeff * x0g;
            const scalar_t contrib_a = base * x1_b;
            const scalar_t contrib_b = base * x1_a;

            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
            atomicAdd(&grad_x1[b * num_a * u + b_idx * u + j], contrib_b);

        } else if (len == 5) {
            // path: [a, b, c, i, *]
            const int b_idx = path[1];
            const int c_idx = path[2];
            const scalar_t x1_b = x1_shared[b_idx * u + j];
            const scalar_t x1_c = x1_shared[c_idx * u + j];

            const scalar_t base = go * coeff * x0g;

            const scalar_t contrib_a = base * (x1_b * x1_c);
            const scalar_t contrib_b = base * (x1_a * x1_c);
            const scalar_t contrib_c = base * (x1_a * x1_b);

            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
            atomicAdd(&grad_x1[b * num_a * u + b_idx * u + j], contrib_b);
            atomicAdd(&grad_x1[b * num_a * u + c_idx * u + j], contrib_c);
        }
        // 其他 len 情形不在当前定义范围内，忽略
    }
}

// -----------------------------
// Launcher：支持 float32 / float64
// -----------------------------
at::Tensor stc_bwd_launcher(
    at::Tensor grad_out,     // [B, u], float32/float64, contiguous
    at::Tensor x1,           // [B, num_a, u], same dtype
    at::Tensor x0_g,         // [B, num_i, u], same dtype
    at::Tensor coeffs,       // [num_paths],  same dtype
    at::Tensor paths_tensor, // [num_paths, 5], int
    at::Tensor path_lens     // [num_paths],  int
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(x1.is_cuda(),       "x1 must be CUDA");
    TORCH_CHECK(x0_g.is_cuda(),     "x0_g must be CUDA");
    TORCH_CHECK(coeffs.is_cuda(),   "coeffs must be CUDA");
    TORCH_CHECK(paths_tensor.is_cuda(), "paths_tensor must be CUDA");
    TORCH_CHECK(path_lens.is_cuda(),    "path_lens must be CUDA");

    auto dtype = x1.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "x1 must be float32 or float64");
    TORCH_CHECK(x0_g.scalar_type() == dtype,
                "x0_g must have same dtype as x1");
    TORCH_CHECK(coeffs.scalar_type() == dtype,
                "coeffs must have same dtype as x1");
    TORCH_CHECK(grad_out.scalar_type() == dtype,
                "grad_out must have same dtype as x1");

    TORCH_CHECK(paths_tensor.scalar_type() == at::kInt,
                "paths_tensor must be int32");
    TORCH_CHECK(path_lens.scalar_type() == at::kInt,
                "path_lens must be int32");

    grad_out     = grad_out.contiguous();
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

    auto grad_x1 = at::zeros_like(x1); // [B, num_a, u]

    dim3 blockDim(u);
    dim3 gridDim(B);

    // shared memory: (num_a*u + num_i*u) * sizeof(scalar_t)
    const size_t shared_elems = static_cast<size_t>(num_a) * u
                              + static_cast<size_t>(num_i) * u;
    const size_t shared_mem_bytes = shared_elems * x1.element_size();

    cudaStream_t cur_stream =
        c10::cuda::getCurrentCUDAStream(grad_out.device().index()).stream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "stc_bwd_kernel", [&] {
        using scalar_t = scalar_t;
        stc_bwd_kernel<scalar_t><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x1.data_ptr<scalar_t>(),
            x0_g.data_ptr<scalar_t>(),
            coeffs.data_ptr<scalar_t>(),
            paths_tensor.data_ptr<int>(),
            path_lens.data_ptr<int>(),
            grad_x1.data_ptr<scalar_t>(),
            B, num_paths, u, num_a, num_i
        );
    });

    CUDA_CHECK(cudaGetLastError());
    return grad_x1;
}

template <typename scalar_t>
__global__  void stc_bwd_kernel_v1(
    const scalar_t* __restrict__ x1,         // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,       // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,     // [num_paths]
    const int* __restrict__ paths,           // [num_paths, 5]
    const int* __restrict__ path_lens,       // [num_paths]
    const scalar_t* __restrict__ grad_out,   // [B, num_out_segments, u] -> flattened
    scalar_t* __restrict__ grad_x1,          // [B, num_a, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared       = reinterpret_cast<scalar_t*>(smem);                        // [num_a * u]
    scalar_t* grad_x1_shared  = x1_shared + static_cast<size_t>(num_a) * u;               // [num_a * u]

    const int b = blockIdx.x;
    const int j = threadIdx.x;

    if (b >= B || j >= u) return;
    const int MAX_OUT_SEG = 9;  // small/meidum/large num_out_segments = 1/4/9
    if (num_out_segments > MAX_OUT_SEG) return;

    const int seg_lim = num_out_segments;

    // -----------------------------
    // 1. preload x1[b, :, :] 到 shared
    // -----------------------------
    for (int idx = j; idx < num_a * u; idx += blockDim.x) {
        int a_idx = idx / u;
        int jj    = idx % u;
        x1_shared[idx] = x1[((b * num_a + a_idx) * u) + jj];
    }

    // 2. 初始化 grad_x1_shared 为 0
    for (int idx = j; idx < num_a * u; idx += blockDim.x) {
        grad_x1_shared[idx] = scalar_t(0);
    }

    __syncthreads();

    // -----------------------------
    // 3. 遍历所有 paths，累加到 grad_x1_shared[a*u + j]
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

        // grad_out[b, out_seg, j]
        const scalar_t g_out =
            grad_out[((b * num_out_segments) + out_seg) * u + j];

        if (g_out == scalar_t(0)) continue;

        // x0_g[b, d_idx, j]
        const scalar_t x0_val =
            x0_g[((b * num_i + d_idx) * u) + j];

        // base = dL/df * df/dx0-less-part = grad_out * coeff * x0
        const scalar_t base = g_out * coeff * x0_val;

        // 取出 x1[a,b,c] (在这个 b 下的 j 列)
        const scalar_t x1_a = x1_shared[a_idx * u + j];

        if (len == 3) {
            // f = coeff * x1[a] * x0
            // dL/dx1[a] = base
            grad_x1_shared[a_idx * u + j] += base;

        } else if (len == 4) {
            // f = coeff * x1[a] * x1[b] * x0
            // d/da = base * x1[b]
            // d/db = base * x1[a]
            const scalar_t x1_b = x1_shared[b_idx * u + j];

            grad_x1_shared[a_idx * u + j] += base * x1_b;
            grad_x1_shared[b_idx * u + j] += base * x1_a;

        } else { // len == 5
            // f = coeff * x1[a] * x1[b] * x1[c] * x0
            const scalar_t x1_b = x1_shared[b_idx * u + j];
            const scalar_t x1_c = x1_shared[c_idx * u + j];

            // d/da = base * x1[b] * x1[c]
            // d/db = base * x1[a] * x1[c]
            // d/dc = base * x1[a] * x1[b]
            const scalar_t da = base * x1_b * x1_c;
            const scalar_t db = base * x1_a * x1_c;
            const scalar_t dc = base * x1_a * x1_b;

            grad_x1_shared[a_idx * u + j] += da;
            grad_x1_shared[b_idx * u + j] += db;
            grad_x1_shared[c_idx * u + j] += dc;
        }
    }

    __syncthreads();

    // -----------------------------
    // 4. grad_x1_shared -> grad_x1[b, :, j]
    // -----------------------------
    for (int idx = j; idx < num_a * u; idx += blockDim.x) {
        int a_idx = idx / u;
        int jj    = idx % u;
        grad_x1[((b * num_a + a_idx) * u) + jj] = grad_x1_shared[idx];
    }
}


template <typename scalar_t>
__global__  // grad_x1 不放进shared memory
void stc_bwd_kernel_v2(
    const scalar_t* __restrict__ x1,         // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,       // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,     // [num_paths]
    const int* __restrict__ paths,           // [num_paths, 5]
    const int* __restrict__ path_lens,       // [num_paths]
    const scalar_t* __restrict__ grad_out,   // [B, num_out_segments, u] -> flattened
    scalar_t* __restrict__ grad_x1,          // [B, num_a, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared = reinterpret_cast<scalar_t*>(smem);  // [num_a * u]

    const int b = blockIdx.x;
    const int j = threadIdx.x;

    if (b >= B || j >= u) return;
    const int MAX_OUT_SEG = 9;  // small/meidum/large num_out_segments = 1/4/9
    if (num_out_segments > MAX_OUT_SEG) return;

    const int seg_lim = num_out_segments;

    // -----------------------------
    // 1. preload x1[b, :, :] 到 shared
    // -----------------------------
    for (int idx = j; idx < num_a * u; idx += blockDim.x) {
        int a_idx = idx / u;
        int jj    = idx % u;
        x1_shared[idx] = x1[((b * num_a + a_idx) * u) + jj];
    }

    __syncthreads();

    // -----------------------------
    // 2. 遍历所有 paths，对 grad_x1[b, :, j] 直接累加
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

        // grad_out[b, out_seg, j]
        const scalar_t g_out =
            grad_out[((b * num_out_segments) + out_seg) * u + j];
        if (g_out == scalar_t(0)) continue;

        // x0_g[b, d_idx, j]
        const scalar_t x0_val =
            x0_g[((b * num_i + d_idx) * u) + j];

        // base = dL/df * df/(x1 部分以外) = grad_out * coeff * x0
        const scalar_t base = g_out * coeff * x0_val;

        // 取出 x1[a,b,c] (在这个 b 下的 j 列)
        const scalar_t x1_a = x1_shared[a_idx * u + j];

        // 计算各自增量并直接写回 global grad_x1
        // grad_x1[b, a_idx, j]
        scalar_t* grad_a_ptr =
            &grad_x1[((b * num_a + a_idx) * u) + j];

        if (len == 3) {
            // f = coeff * x1[a] * x0
            // dL/dx1[a] = base
            *grad_a_ptr += base;

        } else if (len == 4) {
            // f = coeff * x1[a] * x1[b] * x0
            const scalar_t x1_b = x1_shared[b_idx * u + j];

            *grad_a_ptr += base * x1_b;

            scalar_t* grad_b_ptr =
                &grad_x1[((b * num_a + b_idx) * u) + j];
            *grad_b_ptr += base * x1_a;

        } else { // len == 5
            // f = coeff * x1[a] * x1[b] * x1[c] * x0
            const scalar_t x1_b = x1_shared[b_idx * u + j];
            const scalar_t x1_c = x1_shared[c_idx * u + j];

            const scalar_t da = base * x1_b * x1_c;
            const scalar_t db = base * x1_a * x1_c;
            const scalar_t dc = base * x1_a * x1_b;

            *grad_a_ptr += da;

            scalar_t* grad_b_ptr =
                &grad_x1[((b * num_a + b_idx) * u) + j];
            scalar_t* grad_c_ptr =
                &grad_x1[((b * num_a + c_idx) * u) + j];

            *grad_b_ptr += db;
            *grad_c_ptr += dc;
        }
    }
}

// TILE_U 可调：比如 32 / 64 / 128
template <typename scalar_t, int TILE_U>
__global__ void stc_bwd_kernel_tiled(
    const scalar_t* __restrict__ x1,         // [B, num_a, u]
    const scalar_t* __restrict__ x0_g,       // [B, num_i, u]
    const scalar_t* __restrict__ coeffs,     // [num_paths]
    const int* __restrict__ paths,           // [num_paths, 5]
    const int* __restrict__ path_lens,       // [num_paths]
    const scalar_t* __restrict__ grad_out,   // [B, num_out_segments, u]
    scalar_t* __restrict__ grad_x1,          // [B, num_a, u]
    int B, int num_paths, int u, int num_a, int num_i, int num_out_segments)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared      = reinterpret_cast<scalar_t*>(smem);                        // [num_a * TILE_U]
    scalar_t* grad_x1_shared = x1_shared + static_cast<size_t>(num_a) * TILE_U;          // [num_a * TILE_U]

    const int b       = blockIdx.x;      // batch index
    const int tile_id = blockIdx.y;      // tile index along U
    const int lj      = threadIdx.x;     // local j in [0, TILE_U)
    const int j       = tile_id * TILE_U + lj;  // global j

    if (b >= B || j >= u) return;
    constexpr int MAX_OUT_SEG = 9;
    if (num_out_segments > MAX_OUT_SEG) return;

    const int seg_lim = num_out_segments;

    // -----------------------------
    // 1. preload x1[b, :, j] 到 shared，初始化 grad_x1_shared=0
    //    布局：[a, lj] -> x1_shared[a * TILE_U + lj]
    // -----------------------------
    for (int a = 0; a < num_a; ++a) {
        const int idx_global = ((b * num_a + a) * u) + j;
        const int idx_shared = a * TILE_U + lj;
        x1_shared[idx_shared]      = x1[idx_global];
        grad_x1_shared[idx_shared] = scalar_t(0);
    }

    __syncthreads();

    // 每个线程负责这个 (b, j) 上所有 a 的梯度
    // 先把 grad_out[b, seg, j] 读到寄存器
    scalar_t g_out_seg[MAX_OUT_SEG];
    for (int s = 0; s < seg_lim; ++s) {
        g_out_seg[s] = grad_out[((b * num_out_segments) + s) * u + j];
    }

    // -----------------------------
    // 2. 遍历所有 paths，累加到 grad_x1_shared[a * TILE_U + lj]
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

        const scalar_t g_out_val = g_out_seg[out_seg];
        if (g_out_val == scalar_t(0)) continue;

        // x0_g[b, d_idx, j]
        const scalar_t x0_val =
            x0_g[((b * num_i + d_idx) * u) + j];

        const scalar_t base = g_out_val * coeff * x0_val;

        // x1[a], x1[b], x1[c] at this (b, j)
        const scalar_t x1_a =
            x1_shared[a_idx * TILE_U + lj];

        scalar_t* grad_a_ptr =
            &grad_x1_shared[a_idx * TILE_U + lj];

        if (len == 3) {
            // f = coeff * x1[a] * x0
            *grad_a_ptr += base;

        } else if (len == 4) {
            // f = coeff * x1[a] * x1[b] * x0
            const scalar_t x1_b =
                x1_shared[b_idx * TILE_U + lj];

            *grad_a_ptr += base * x1_b;

            scalar_t* grad_b_ptr =
                &grad_x1_shared[b_idx * TILE_U + lj];
            *grad_b_ptr += base * x1_a;

        } else { // len == 5
            // f = coeff * x1[a] * x1[b] * x1[c] * x0
            const scalar_t x1_b =
                x1_shared[b_idx * TILE_U + lj];
            const scalar_t x1_c =
                x1_shared[c_idx * TILE_U + lj];

            const scalar_t da = base * x1_b * x1_c;
            const scalar_t db = base * x1_a * x1_c;
            const scalar_t dc = base * x1_a * x1_b;

            *grad_a_ptr += da;

            scalar_t* grad_b_ptr =
                &grad_x1_shared[b_idx * TILE_U + lj];
            scalar_t* grad_c_ptr =
                &grad_x1_shared[c_idx * TILE_U + lj];

            *grad_b_ptr += db;
            *grad_c_ptr += dc;
        }
    }

    __syncthreads();

    // -----------------------------
    // 3. 把 grad_x1_shared 写回 global grad_x1[b, :, j]
    // -----------------------------
    for (int a = 0; a < num_a; ++a) {
        const int idx_shared = a * TILE_U + lj;
        const int idx_global = ((b * num_a + a) * u) + j;
        grad_x1[idx_global] = grad_x1_shared[idx_shared];
    }
}

at::Tensor stc_bwd_x1_launcher(
    at::Tensor grad_out,       // [B, num_out_segments * u] (from forward)
    at::Tensor x1,             // [B, num_a, u]
    at::Tensor x0_g,           // [B, num_i, u]
    at::Tensor coeffs,         // [num_paths]
    at::Tensor paths_tensor,   // [num_paths, 5]
    at::Tensor path_lens,      // [num_paths]
    const int64_t num_out_segments)
{
    TORCH_CHECK(x1.is_cuda(), "x1 must be CUDA");
    TORCH_CHECK(x0_g.is_cuda(), "x0_g must be CUDA");
    TORCH_CHECK(coeffs.is_cuda(), "coeffs must be CUDA");
    TORCH_CHECK(paths_tensor.is_cuda(), "paths_tensor must be CUDA");
    TORCH_CHECK(path_lens.is_cuda(), "path_lens must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");

    auto dtype = x1.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "x1 must be float32 or float64");
    TORCH_CHECK(x0_g.scalar_type() == dtype,
                "x0_g must have same dtype as x1");
    TORCH_CHECK(coeffs.scalar_type() == dtype,
                "coeffs must have same dtype as x1");
    TORCH_CHECK(grad_out.scalar_type() == dtype,
                "grad_out must have same dtype as x1");

    x1           = x1.contiguous();
    x0_g         = x0_g.contiguous();
    coeffs       = coeffs.contiguous();
    paths_tensor = paths_tensor.contiguous();
    path_lens    = path_lens.contiguous();
    grad_out     = grad_out.contiguous();

    const int B         = x1.size(0);
    const int num_a     = x1.size(1);
    const int u         = x1.size(2);
    const int num_i     = x0_g.size(1);
    const int num_paths = coeffs.size(0);

    TORCH_CHECK(
        grad_out.size(0) == B,
        "grad_out batch dim must match x1");
    TORCH_CHECK(
        grad_out.size(1) == num_out_segments * u,
        "grad_out second dim must be num_out_segments * u");

    // grad_x1: [B, num_a, u]
    at::Tensor grad_x1 = at::zeros_like(x1);

    if (u < 224) {
        
        dim3 blockDim(u);
        dim3 gridDim(B);

        // shared grad_out and x1
        const size_t shared_elems =
            static_cast<size_t>(num_a) * u * 2; // x1_shared + grad_x1_shared
        const size_t shared_mem_bytes =
            shared_elems * x1.element_size();

        cudaStream_t cur_stream =
            c10::cuda::getCurrentCUDAStream(x1.device().index()).stream();

        AT_DISPATCH_FLOATING_TYPES(dtype, "stc_bwd_kernel_v1", [&] {
            using scalar_t = scalar_t;
            stc_bwd_kernel_v1<scalar_t><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
                x1.data_ptr<scalar_t>(),
                x0_g.data_ptr<scalar_t>(),
                coeffs.data_ptr<scalar_t>(),
                paths_tensor.data_ptr<int>(),
                path_lens.data_ptr<int>(),
                grad_out.data_ptr<scalar_t>(),
                grad_x1.data_ptr<scalar_t>(),
                B, num_paths, u, num_a, num_i, static_cast<int>(num_out_segments));
        });
    } else {
        
        // only x1 to shared memory
        /*
        dim3 blockDim(u);
        dim3 gridDim(B);
        const size_t shared_elems =
        static_cast<size_t>(num_a) * u;
        */

        // for tile
        
        constexpr int TILE_U = 96;
        const int n_tiles = (u + TILE_U - 1) / TILE_U;
        dim3 blockDim(TILE_U);
        dim3 gridDim(B, n_tiles);
        const size_t shared_elems =
            static_cast<size_t>(num_a) * TILE_U * 2; // x1_shared + grad_x1_shared
        
        const size_t shared_mem_bytes =
            shared_elems * x1.element_size();

        cudaStream_t cur_stream =
            c10::cuda::getCurrentCUDAStream(x1.device().index()).stream();

        AT_DISPATCH_FLOATING_TYPES(dtype, "stc_bwd_kernel_tiled", [&] {
            using scalar_t = scalar_t;
            //stc_bwd_kernel_v2<scalar_t><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
            stc_bwd_kernel_tiled<scalar_t, TILE_U><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
                x1.data_ptr<scalar_t>(),
                x0_g.data_ptr<scalar_t>(),
                coeffs.data_ptr<scalar_t>(),
                paths_tensor.data_ptr<int>(),
                path_lens.data_ptr<int>(),
                grad_out.data_ptr<scalar_t>(),
                grad_x1.data_ptr<scalar_t>(),
                B, num_paths, u, num_a, num_i, static_cast<int>(num_out_segments));
        });
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x1;
}


TORCH_LIBRARY(stc_bwd, m)
{
    m.def("backward", &stc_bwd_x1_launcher);
}
