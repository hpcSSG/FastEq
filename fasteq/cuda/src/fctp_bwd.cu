#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// b_all:        [B, 1, V]
// w_all:        [P, U, V, W]
// cg_i_all:     [P, nnz_max]
// cg_j_all:     [P, nnz_max]   // 未用
// cg_k_all:     [P, nnz_max]   // global k ∈ [0, K_total)
// cg_val_all:   [P, nnz_max]
// nnz_per_path: [P]
// K_per_path:   [P]
// path_offset:  [P]            // 未用
//
// grad_out:     [B, K_total, W]
// grad_a:       [B, I_total, U]
//
// fast path 约束：U==96/128/224, W%4==0, nnz_max<=7, B<=65535
template<typename scalar_t>
__global__ void fused_fctp_kernel_bwd_grad_a_multipath(
    const scalar_t* __restrict__ b_all,        // [B,1,V]
    const scalar_t* __restrict__ w_all,        // [P,U,V,W]
    const int*     __restrict__ cg_i_all,      // [P,nnz_max]
    const int*     __restrict__ cg_j_all,      // [P,nnz_max]  // 未使用
    const int*     __restrict__ cg_k_all,      // [P,nnz_max]
    const scalar_t* __restrict__ cg_val_all,   // [P,nnz_max]
    const int*     __restrict__ nnz_per_path,  // [P]
    const int*     __restrict__ K_per_path,    // [P]
    const int*     __restrict__ path_offset,   // [P]
    int nnz_max,
    int P, int B, int I_total, int U, int V, int W,
    int K_max, int K_total,
    const scalar_t* __restrict__ grad_out,     // [B,K_total,W]
    scalar_t*       __restrict__ grad_a        // [B,I_total,U]
)
{
    const int p = blockIdx.x;   // path id
    const int b = blockIdx.y;   // batch id
    if (p >= P || b >= B) return;

    const int nnz_p = nnz_per_path[p];
    const int K_p   = K_per_path[p];
    if (nnz_p <= 0 || K_p <= 0) return;

    // global index
    const int*      cg_i = cg_i_all   + (size_t)p * nnz_max;
    const int*      cg_j = cg_j_all   + (size_t)p * nnz_max;   // 未使用
    const int*      cg_k = cg_k_all   + (size_t)p * nnz_max;
    const scalar_t* cg_v = cg_val_all + (size_t)p * nnz_max;

    // b_all[b]: [1,V] -> [V]
    const scalar_t* __restrict__ brow = b_all + (size_t)b * V;

    // w_all[p]: [U,V,W]
    const scalar_t* __restrict__ W_p  = w_all + (size_t)p * U * V * W;

    // grad_out[b]: [K_total,W]
    const scalar_t* __restrict__ dO_b = grad_out + (size_t)b * K_total * W;

    // grad_a[b]: [I_total,U]
    scalar_t* __restrict__ dA_b = grad_a + (size_t)b * I_total * U;

    // -----------------------------
    // 1) 对该 (p,b) 计算 v* = argmax_v b[b,0,v]
    // -----------------------------
    int vstar = 0;
    if (threadIdx.x == 0) {
        scalar_t best = std::numeric_limits<scalar_t>::lowest();
        int idx = 0;
        for (int v = 0; v < V; ++v) {
            scalar_t x = brow[v];
            if (x > best) { best = x; idx = v; }
        }
        vstar = idx;
    }
    __shared__ int sh_v;
    if (threadIdx.x == 0)
        sh_v = vstar;
    __syncthreads();
    const int v = sh_v;

    // -----------------------------
    // 2) shared memory 布局
    //
    //    sh_Wt: [W, U+1]     → 存 W_p[:, v*, :]
    //    sh_dO: [W]          → 存当前 triple 的 grad_out[b,k_global,:]
    //
    //    与 forward 一致地用 [w, u_pad] 来避免 bank 冲突:
    //    index = w * U_pad + u
    // -----------------------------
    const int U_pad = U + 1;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shmem_raw[];
    scalar_t* sh_Wt = reinterpret_cast<scalar_t*>(shmem_raw);             // [W * U_pad]
    scalar_t* sh_dO = sh_Wt + (size_t)W * U_pad;                          // [W]

    // Vec4 类型，用于 vectorized 读
    using Vec4 = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double4
    >::type;

    const int tx = blockDim.x;    // = U
    const int u  = threadIdx.x;   // 0..U-1

    // -----------------------------
    // 3) 一次性把 W_p[:, v*, :] -> sh_Wt[w,u]
    //    使用 Vec4 沿 w 维做 vector load:
    //       Wv = W/4
    //       t ∈ [0, U*Wv)
    // -----------------------------
    const int Wv  = W / 4;
    const int UWv = U * Wv;

    for (int t = u; t < UWv; t += tx) {
        int u_idx = t / Wv;
        int wv    = t - u_idx * Wv;  // 0..Wv-1
        int w0    = wv << 2;         // 4 元素起始位置

        const size_t off = ((size_t)u_idx * V + (size_t)v) * W + (size_t)w0;
        const Vec4 r = *reinterpret_cast<const Vec4*>(W_p + off);

        sh_Wt[((size_t)w0 + 0) * U_pad + u_idx] = (scalar_t)r.x;
        sh_Wt[((size_t)w0 + 1) * U_pad + u_idx] = (scalar_t)r.y;
        sh_Wt[((size_t)w0 + 2) * U_pad + u_idx] = (scalar_t)r.z;
        sh_Wt[((size_t)w0 + 3) * U_pad + u_idx] = (scalar_t)r.w;
    }
    __syncthreads();

    if (u >= U) return;

    // -----------------------------
    // 4) 对每个 triple (i_global, k_global, val)：
    //
    //    先把 grad_out[b,k_global,:] 读入 sh_dO[W]
    //    再对该 triple 的所有 u（每个线程一个 u）：
    //      acc_u = Σ_w sh_dO[w] * sh_Wt[w,u]
    //      grad_a[b,i_global,u] += val * acc_u
    // -----------------------------
    for (int t = 0; t < nnz_p; ++t) {
        const int i_global = cg_i[t];
        const int k_global = cg_k[t];
        const scalar_t val = cg_v[t];

        // 4a) 用 Vec4 把 dO_b[k_global, :] -> sh_dO[w]
        for (int tv = u; tv < Wv; tv += tx) {
            int w0 = tv << 2;
            const size_t off = (size_t)k_global * W + (size_t)w0;
            const Vec4 r = *reinterpret_cast<const Vec4*>(dO_b + off);
            sh_dO[w0 + 0] = (scalar_t)r.x;
            sh_dO[w0 + 1] = (scalar_t)r.y;
            sh_dO[w0 + 2] = (scalar_t)r.z;
            sh_dO[w0 + 3] = (scalar_t)r.w;
        }
        __syncthreads();

        // 4b) 当前线程负责固定的 u，遍历 w 做 inner product
        scalar_t acc_u = scalar_t(0);
        for (int w = 0; w < W; ++w) {
            const scalar_t gout = sh_dO[w];
            const scalar_t Wuv  = sh_Wt[(size_t)w * U_pad + u];
            acc_u += gout * Wuv;
        }

        // 4c) grad_a 累加:
        //     dL/dA[b,i_global,u] += val * acc_u
        atomicAdd(&dA_b[(size_t)i_global * U + u], val * acc_u);

        __syncthreads();  // 确保所有线程用完当前 sh_dO 后再覆盖
    }
}

at::Tensor launch_fused_multipath_fctp_backward(
    at::Tensor grad_out,     // [B, K_total, W]
    at::Tensor w_all,        // [P, U, V, W]
    at::Tensor a_all,        // [B, I_total, U]
    at::Tensor b_all,        // [B, 1, V]
    at::Tensor cg_i_all,     // [P, nnz_max]
    at::Tensor cg_j_all,     // [P, nnz_max]
    at::Tensor cg_k_all,     // [P, nnz_max]
    at::Tensor cg_val_all,   // [P, nnz_max]
    at::Tensor nnz_per_path, // [P]
    at::Tensor K_per_path,   // [P]
    at::Tensor path_offset,  // [P]
    const int64_t U,
    const int64_t V,
    const int64_t W,
    const int64_t K_total
)
{
    TORCH_CHECK(b_all.is_cuda() && w_all.is_cuda()
             && cg_i_all.is_cuda() && cg_j_all.is_cuda() && cg_k_all.is_cuda()
             && cg_val_all.is_cuda() && nnz_per_path.is_cuda()
             && K_per_path.is_cuda() && path_offset.is_cuda()
             && grad_out.is_cuda(),
             "all tensors must be CUDA");

    auto dtype = b_all.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                "b_all must be float32 or float64");
    TORCH_CHECK(w_all.scalar_type() == dtype &&
                cg_val_all.scalar_type() == dtype &&
                grad_out.scalar_type()  == dtype,
                "dtypes must match");

    c10::cuda::CUDAGuard device_guard(b_all.get_device());

    const int B       = (int)b_all.size(0);
    const int P       = (int)nnz_per_path.size(0);
    TORCH_CHECK(a_all.size(1) % U == 0, "a_all.size(1) must be divisible by U");
    const int I_total = static_cast<int>(a_all.size(1) / U);

    a_all = a_all.view({B, I_total, U});
    b_all = b_all.view({B, 1, V});
    w_all = w_all.view({P, U, V, W});
    grad_out = grad_out.view({B, K_total, W});

    TORCH_CHECK(b_all.size(1) == 1 && b_all.size(2) == V,
                "b_all must be [B,1,V]");
    TORCH_CHECK(w_all.size(0) == P && w_all.size(1) == U &&
                w_all.size(2) == V && w_all.size(3) == W,
                "w_all must be [P,U,V,W]");
    TORCH_CHECK(grad_out.size(0) == B &&
                grad_out.size(1) == K_total &&
                grad_out.size(2) == W,
                "grad_out must be [B,K_total,W]");
    

    b_all        = b_all.contiguous();
    w_all        = w_all.contiguous();
    cg_i_all     = cg_i_all.contiguous();
    cg_j_all     = cg_j_all.contiguous();
    cg_k_all     = cg_k_all.contiguous();
    cg_val_all   = cg_val_all.contiguous();
    nnz_per_path = nnz_per_path.contiguous();
    K_per_path   = K_per_path.contiguous();
    path_offset  = path_offset.contiguous();
    grad_out     = grad_out.contiguous();

    const int nnz_max = (int)cg_i_all.size(1);
    TORCH_CHECK(cg_i_all.size(0)==P && cg_j_all.size(0)==P &&
                cg_k_all.size(0)==P && cg_val_all.size(0)==P,
                "cg_* shape mismatch");
    TORCH_CHECK(nnz_max <= 7, "nnz_max<=7 required");
    TORCH_CHECK(W % 4 == 0, "W%4==0 required for Vec4 loads");

    auto grad_a = at::zeros({B, I_total, U}, b_all.options());

    // K_max 目前只是为了接口对齐（kernel 里没实际用它做线程分配）
    auto K_per_path_cpu = K_per_path.to(at::kCPU);
    int K_max = 0;
    for (int p = 0; p < P; ++p) {
        int Kp = K_per_path_cpu[p].item<int>();
        if (Kp > K_max) K_max = Kp;
    }

    const int tx = (int)U;   // 每个线程负责一个 u
    const int ty = 1;
    dim3 block(tx, ty, 1);
    dim3 grid(P, B, 1);

    const int U_pad = (int)U + 1;
    // shared: Wt[W*U_pad] + dO[W]
    size_t shmem_elems = (size_t)W * U_pad + (size_t)W;
    size_t shmem_bytes = shmem_elems * grad_a.element_size();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_fctp_backward_grad_a_multipath", [&] {
        using scalar_t_ = scalar_t;
        cudaFuncSetAttribute(
            fused_fctp_kernel_bwd_grad_a_multipath<scalar_t_>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem_bytes);

        fused_fctp_kernel_bwd_grad_a_multipath<scalar_t_>
            <<<grid, block, (int)shmem_bytes, stream>>>(
                b_all.data_ptr<scalar_t_>(),
                w_all.data_ptr<scalar_t_>(),
                cg_i_all.data_ptr<int>(),
                cg_j_all.data_ptr<int>(),
                cg_k_all.data_ptr<int>(),
                cg_val_all.data_ptr<scalar_t_>(),
                nnz_per_path.data_ptr<int>(),
                K_per_path.data_ptr<int>(),
                path_offset.data_ptr<int>(),
                nnz_max,
                P, B, I_total, (int)U, (int)V, (int)W,
                K_max, (int)K_total,
                grad_out.data_ptr<scalar_t_>(),
                grad_a.data_ptr<scalar_t_>());
    });

    grad_a = grad_a.view({B, I_total * U});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_a;
}

template<typename scalar_t, int U_TILE>
__global__ void fused_fctp_kernel_bwd_grad_a_multipath_tiledU(
    const scalar_t* __restrict__ b_all,        // [B,1,V]
    const scalar_t* __restrict__ w_all,        // [P,U,V,W]
    const int*     __restrict__ cg_i_all,      // [P,nnz_max]
    const int*     __restrict__ cg_j_all,      // [P,nnz_max]  // 未使用
    const int*     __restrict__ cg_k_all,      // [P,nnz_max]
    const scalar_t* __restrict__ cg_val_all,   // [P,nnz_max]
    const int*     __restrict__ nnz_per_path,  // [P]
    const int*     __restrict__ K_per_path,    // [P]
    const int*     __restrict__ path_offset,   // [P]
    int nnz_max,
    int P, int B, int I_total, int U, int V, int W,
    int K_max, int K_total,
    const scalar_t* __restrict__ grad_out,     // [B,K_total,W]
    scalar_t*       __restrict__ grad_a        // [B,I_total,U]
)
{
    const int p = blockIdx.x;   // path id
    const int b = blockIdx.y;   // batch id
    if (p >= P || b >= B) return;

    const int nnz_p = nnz_per_path[p];
    const int K_p   = K_per_path[p];
    if (nnz_p <= 0 || K_p <= 0) return;

    // global index
    const int*      cg_i = cg_i_all   + (size_t)p * nnz_max;
    const int*      cg_j = cg_j_all   + (size_t)p * nnz_max;   // 未使用
    const int*      cg_k = cg_k_all   + (size_t)p * nnz_max;
    const scalar_t* cg_v = cg_val_all + (size_t)p * nnz_max;

    // b_all[b]: [1,V] -> [V]
    const scalar_t* __restrict__ brow = b_all + (size_t)b * V;

    // w_all[p]: [U,V,W]
    const scalar_t* __restrict__ W_p  = w_all + (size_t)p * U * V * W;

    // grad_out[b]: [K_total,W]
    const scalar_t* __restrict__ dO_b = grad_out + (size_t)b * K_total * W;

    // grad_a[b]: [I_total,U]
    scalar_t* __restrict__ dA_b = grad_a + (size_t)b * I_total * U;

    // -----------------------------
    // 1) 对该 (p,b) 计算 v* = argmax_v b[b,0,v]
    // -----------------------------
    int vstar = 0;
    if (threadIdx.x == 0) {
        scalar_t best = std::numeric_limits<scalar_t>::lowest();
        int idx = 0;
        for (int v = 0; v < V; ++v) {
            scalar_t x = brow[v];
            if (x > best) { best = x; idx = v; }
        }
        vstar = idx;
    }
    __shared__ int sh_v;
    if (threadIdx.x == 0)
        sh_v = vstar;
    __syncthreads();
    const int v = sh_v;

    // -----------------------------
    // 2) shared memory 布局 (tileU)
    //
    //    sh_Wt: [W, U_TILE+1] → 存 W_p[u_base : u_base+U_TILE, v*, :]
    //    sh_dO: [W]           → 存当前 triple 的 grad_out[b,k_global,:]
    // -----------------------------
    const int U_pad = U_TILE + 1;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shmem_raw[];
    scalar_t* sh_Wt = reinterpret_cast<scalar_t*>(shmem_raw);             // [W * U_pad]
    scalar_t* sh_dO = sh_Wt + (size_t)W * U_pad;                          // [W]

    using Vec4 = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double4
    >::type;

    const int tx      = blockDim.x;    // 通常设为 U_TILE
    const int u_local = threadIdx.x;   // tile 内局部 u 索引: 0..U_TILE-1

    // Vec4 沿 w 维读
    const int Wv  = W / 4;
    const int tv_stride = tx;

    // -----------------------------
    // 3) 沿 U 方向分块：每个 tile 先把 W_p 的这一块 preload 到 shared
    // -----------------------------
    for (int u_base = 0; u_base < U; u_base += U_TILE) {
        const int U_this = min(U_TILE, U - u_base);

        // 3a) 加载当前 U tile 的 W_p[u_base : u_base+U_this, v*, :] -> sh_Wt[w,u_off]
        //
        // t ∈ [0, U_this * Wv)
        // u_off = t / Wv ∈ [0, U_this)
        // wv    = t - u_off * Wv ∈ [0, Wv)
        // w0    = wv * 4
        for (int t = u_local; t < U_this * Wv; t += tv_stride) {
            int u_off = t / Wv;           // tile 内 u 索引
            int wv    = t - u_off * Wv;
            int w0    = wv << 2;          // 4 元素起始 w

            int u_idx = u_base + u_off;   // 全局 u

            const size_t off = ((size_t)u_idx * V + (size_t)v) * W + (size_t)w0;
            const Vec4 r = *reinterpret_cast<const Vec4*>(W_p + off);

            sh_Wt[((size_t)w0 + 0) * U_pad + u_off] = (scalar_t)r.x;
            sh_Wt[((size_t)w0 + 1) * U_pad + u_off] = (scalar_t)r.y;
            sh_Wt[((size_t)w0 + 2) * U_pad + u_off] = (scalar_t)r.z;
            sh_Wt[((size_t)w0 + 3) * U_pad + u_off] = (scalar_t)r.w;
        }
        __syncthreads();

        // -----------------------------
        // 4) 对每个 triple (i_global, k_global, val)：
        //
        //    先把 grad_out[b,k_global,:] 读入 sh_dO[W]
        //    再对该 tile 内的所有 u_off（每个线程1~多个 u_off）：
        //      acc_u = Σ_w sh_dO[w] * sh_Wt[w,u_off]
        //      grad_a[b,i_global,u_glb] += val * acc_u
        // -----------------------------
        for (int t = 0; t < nnz_p; ++t) {
            const int i_global = cg_i[t];
            const int k_global = cg_k[t];
            const scalar_t val = cg_v[t];

            // 4a) 用 Vec4 把 dO_b[k_global, :] -> sh_dO[w]
            for (int tv = u_local; tv < Wv; tv += tv_stride) {
                int w0 = tv << 2;
                const size_t off = (size_t)k_global * W + (size_t)w0;
                const Vec4 r = *reinterpret_cast<const Vec4*>(dO_b + off);
                sh_dO[w0 + 0] = (scalar_t)r.x;
                sh_dO[w0 + 1] = (scalar_t)r.y;
                sh_dO[w0 + 2] = (scalar_t)r.z;
                sh_dO[w0 + 3] = (scalar_t)r.w;
            }
            __syncthreads();

            // 4b) 当前线程负责 tile 内若干个 u_off（stride=blockDim.x）
            for (int u_off = u_local; u_off < U_this; u_off += tx) {
                int u_glb = u_base + u_off;   // 全局 u index

                scalar_t acc_u = scalar_t(0);
                for (int w = 0; w < W; ++w) {
                    const scalar_t gout = sh_dO[w];
                    const scalar_t Wuv  = sh_Wt[(size_t)w * U_pad + u_off];
                    acc_u += gout * Wuv;
                }

                // grad_a[b,i_global,u_glb] += val * acc_u
                atomicAdd(&dA_b[(size_t)i_global * U + u_glb], val * acc_u);
            }

            __syncthreads();  // 确保所有线程用完当前 sh_dO 后再覆盖
        }
        // 下一轮 u_base 会重新填充 sh_Wt；上一轮 triple 已全部用完当前 tile
    }
}

at::Tensor launch_fused_multipath_fctp_tiled_backward(
    at::Tensor grad_out,     // [B, K_total, W]
    at::Tensor w_all,        // [P, U, V, W]
    at::Tensor a_all,        // [B, I_total, U]
    at::Tensor b_all,        // [B, 1, V]
    at::Tensor cg_i_all,     // [P, nnz_max]
    at::Tensor cg_j_all,     // [P, nnz_max]
    at::Tensor cg_k_all,     // [P, nnz_max]
    at::Tensor cg_val_all,   // [P, nnz_max]
    at::Tensor nnz_per_path, // [P]
    at::Tensor K_per_path,   // [P]
    at::Tensor path_offset,  // [P]
    const int64_t U,
    const int64_t V,
    const int64_t W,
    const int64_t K_total
)
{
    TORCH_CHECK(b_all.is_cuda() && w_all.is_cuda()
             && cg_i_all.is_cuda() && cg_j_all.is_cuda() && cg_k_all.is_cuda()
             && cg_val_all.is_cuda() && nnz_per_path.is_cuda()
             && K_per_path.is_cuda() && path_offset.is_cuda()
             && grad_out.is_cuda(),
             "all tensors must be CUDA");

    auto dtype = b_all.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                "b_all must be float32 or float64");
    TORCH_CHECK(w_all.scalar_type() == dtype &&
                cg_val_all.scalar_type() == dtype &&
                grad_out.scalar_type()  == dtype,
                "dtypes must match");

    c10::cuda::CUDAGuard device_guard(b_all.get_device());

    const int B       = (int)b_all.size(0);
    const int P       = (int)nnz_per_path.size(0);
    TORCH_CHECK(a_all.size(1) % U == 0, "a_all.size(1) must be divisible by U");
    const int I_total = static_cast<int>(a_all.size(1) / U);

    a_all = a_all.view({B, I_total, U});
    b_all = b_all.view({B, 1, V});
    w_all = w_all.view({P, U, V, W});
    grad_out = grad_out.view({B, K_total, W});

    TORCH_CHECK(b_all.size(1) == 1 && b_all.size(2) == V,
                "b_all must be [B,1,V]");
    TORCH_CHECK(w_all.size(0) == P && w_all.size(1) == U &&
                w_all.size(2) == V && w_all.size(3) == W,
                "w_all must be [P,U,V,W]");
    TORCH_CHECK(grad_out.size(0) == B &&
                grad_out.size(1) == K_total &&
                grad_out.size(2) == W,
                "grad_out must be [B,K_total,W]");
    

    b_all        = b_all.contiguous();
    w_all        = w_all.contiguous();
    cg_i_all     = cg_i_all.contiguous();
    cg_j_all     = cg_j_all.contiguous();
    cg_k_all     = cg_k_all.contiguous();
    cg_val_all   = cg_val_all.contiguous();
    nnz_per_path = nnz_per_path.contiguous();
    K_per_path   = K_per_path.contiguous();
    path_offset  = path_offset.contiguous();
    grad_out     = grad_out.contiguous();

    const int nnz_max = (int)cg_i_all.size(1);
    TORCH_CHECK(cg_i_all.size(0)==P && cg_j_all.size(0)==P &&
                cg_k_all.size(0)==P && cg_val_all.size(0)==P,
                "cg_* shape mismatch");
    TORCH_CHECK(nnz_max <= 7, "nnz_max<=7 required");
    TORCH_CHECK(W % 4 == 0, "W%4==0 required for Vec4 loads");

    auto grad_a = at::zeros({B, I_total, U}, b_all.options());

    // K_max 目前只是为了接口对齐（kernel 里没实际用它做线程分配）
    auto K_per_path_cpu = K_per_path.to(at::kCPU);
    int K_max = 0;
    for (int p = 0; p < P; ++p) {
        int Kp = K_per_path_cpu[p].item<int>();
        if (Kp > K_max) K_max = Kp;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_fctp_backward_grad_a_multipath_tiledU", [&] {
        using scalar_t_ = scalar_t;

        if (std::is_same<scalar_t_, double>::value) {
            constexpr int U_TILE = 16;      
            const int tx = U_TILE;             // 每个 block 处理一个 tile 的 U
            dim3 block(tx, 1, 1);
            dim3 grid(P, B, 1);

            const int U_pad = U_TILE + 1;
            size_t shmem_elems = (size_t)W * U_pad + (size_t)W; // sh_Wt + sh_dO
            size_t shmem_bytes = shmem_elems * sizeof(scalar_t_);

            fused_fctp_kernel_bwd_grad_a_multipath_tiledU<scalar_t_, U_TILE>
                <<<grid, block, (int)shmem_bytes, stream>>>(
                    b_all.data_ptr<scalar_t_>(),
                    w_all.data_ptr<scalar_t_>(),
                    cg_i_all.data_ptr<int>(),
                    cg_j_all.data_ptr<int>(),
                    cg_k_all.data_ptr<int>(),
                    cg_val_all.data_ptr<scalar_t_>(),
                    nnz_per_path.data_ptr<int>(),
                    K_per_path.data_ptr<int>(),
                    path_offset.data_ptr<int>(),
                    nnz_max,
                    P, B, I_total, (int)U, (int)V, (int)W,
                    K_max, (int)K_total,
                    grad_out.data_ptr<scalar_t_>(),
                    grad_a.data_ptr<scalar_t_>());
        } else {
            constexpr int U_TILE = 32;
            const int tx = U_TILE;
            dim3 block(tx, 1, 1);
            dim3 grid(P, B, 1);

            const int U_pad = U_TILE + 1;
            size_t shmem_elems = (size_t)W * U_pad + (size_t)W;
            size_t shmem_bytes = shmem_elems * sizeof(scalar_t_);

            fused_fctp_kernel_bwd_grad_a_multipath_tiledU<scalar_t_, U_TILE>
                <<<grid, block, (int)shmem_bytes, stream>>>(
                    b_all.data_ptr<scalar_t_>(),
                    w_all.data_ptr<scalar_t_>(),
                    cg_i_all.data_ptr<int>(),
                    cg_j_all.data_ptr<int>(),
                    cg_k_all.data_ptr<int>(),
                    cg_val_all.data_ptr<scalar_t_>(),
                    nnz_per_path.data_ptr<int>(),
                    K_per_path.data_ptr<int>(),
                    path_offset.data_ptr<int>(),
                    nnz_max,
                    P, B, I_total, (int)U, (int)V, (int)W,
                    K_max, (int)K_total,
                    grad_out.data_ptr<scalar_t_>(),
                    grad_a.data_ptr<scalar_t_>());
            }
        });

    grad_a = grad_a.view({B, I_total * U});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_a;
}



TORCH_LIBRARY(fctp_fused_multipath_bwd, m) {
    m.def("backward", &launch_fused_multipath_fctp_tiled_backward);
}