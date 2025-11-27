#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// 多 path 版 fast kernel（concat 输出）：
//
// a_seg:        [B, I_total, U]
// b_seg:        [B, 1, V]          // fast path: J==1
// w_all:        [P, U, V, W]
//
// cg_i_all:     [P, nnz_max]   // 全局行索引 i_global ∈ [0, I_total)
// cg_j_all:     [P, nnz_max]   // 全局 j 索引（目前没用到，保留通用性）
// cg_k_all:     [P, nnz_max]   // 全局列索引 k_global ∈ [0, K_total)
// cg_val_all:   [P, nnz_max]
//
// nnz_per_path: [P]
// K_per_path:   [P]
// path_offset:  [P]   // 每个 path 的 K 段起始偏移，sum(K_per_path) = K_total
//
// out_final:    [B, K_total, W]
//
// 约束：U==96, W%4==0, K_max<=7, nnz_max<=7, J==1, B<=65535
template<typename scalar_t>
__global__ void fused_fctp_kernel_fwd_multipath(
    const scalar_t* __restrict__ a_seg,        // [B, I_total, U]
    const scalar_t* __restrict__ b_seg,        // [B, 1, V]
    const scalar_t* __restrict__ w_all,        // [P, U, V, W]
    const int*     __restrict__ cg_i_all,      // [P, nnz_max]
    const int*     __restrict__ cg_j_all,      // [P, nnz_max]  // 当前未使用
    const int*     __restrict__ cg_k_all,      // [P, nnz_max]
    const scalar_t* __restrict__ cg_val_all,   // [P, nnz_max]
    const int*     __restrict__ nnz_per_path,  // [P]
    const int*     __restrict__ K_per_path,    // [P]
    const int*     __restrict__ path_offset,   // [P]
    int nnz_max,
    int P, int B, int I_total, int U, int V, int W,
    int K_max, int K_total,
    scalar_t*      __restrict__ out_final      // [B, K_total, W]
)
{
    // 一个 block 对应一个 (p, b)
    const int p = blockIdx.x;   // path id
    const int b = blockIdx.y;   // batch id
    if (p >= P || b >= B) return;

    const int nnz_p = nnz_per_path[p];
    const int K_p   = K_per_path[p];
    if (nnz_p <= 0 || K_p <= 0) return;

    const int k_base = path_offset[p];   // 该 path 在全局 K 维的起始位置

    // cg 指针（全部是 global 索引）
    const int*      cg_i = cg_i_all   + (size_t)p * nnz_max;
    const int*      cg_j = cg_j_all   + (size_t)p * nnz_max;   // 目前未使用
    const int*      cg_k = cg_k_all   + (size_t)p * nnz_max;
    const scalar_t* cg_v = cg_val_all + (size_t)p * nnz_max;

    // a_seg[b]: [I_total, U]
    const scalar_t* __restrict__ A_b  = a_seg + (size_t)b * I_total * U;
    // b_seg[b]: [1, V] → 当作 [V]
    const scalar_t* __restrict__ brow = b_seg + (size_t)b * V;
    // w_all[p]: [U, V, W]
    const scalar_t* __restrict__ W_p  = w_all + (size_t)p * U * V * W;
    // out_final[b]: [K_total, W]
    scalar_t* __restrict__ Ob_base    = out_final + (size_t)b * K_total * W;

    // 1) 对该 (p,b) 做 argmax_v b[b,0,v]
    int vstar = 0;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        scalar_t best = std::numeric_limits<scalar_t>::lowest();
        int idx = 0;
        for (int v = 0; v < V; ++v) {
            scalar_t x = brow[v];
            if (x > best) { best = x; idx = v; }
        }
        vstar = idx;
    }
    __shared__ int sh_v;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        sh_v = vstar;
    __syncthreads();
    const int v = sh_v;

    // 2) shared: Wt[W, U+1] + Asel[nnz_max, U+1]
    const int U_pad = U + 1;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shmem_raw[];
    scalar_t* sh_Wt   = reinterpret_cast<scalar_t*>(shmem_raw);               // [W, U_pad]
    scalar_t* sh_Asel = sh_Wt + (size_t)W * U_pad;                            // [nnz_max, U_pad]

    using Vec4 = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double4
    >::type;

    const int tx = blockDim.x;
    const int ty = blockDim.y; // = K_max

    const int Wv  = W / 4;
    const int UWv = U * Wv;

    // 2a) 把 W_p[:, v, :] → sh_Wt[w,u]（Vec4 向量读，scalar 写）
    for (int t = threadIdx.y * tx + threadIdx.x;
         t < UWv;
         t += tx * ty) {
        int u  = t / Wv;
        int wv = t - u * Wv;
        int w0 = wv << 2;

        const size_t off = ((size_t)u * V + (size_t)v) * W + (size_t)w0;
        const Vec4 r = *reinterpret_cast<const Vec4*>(W_p + off);

        sh_Wt[((size_t)w0 + 0) * U_pad + u] = (scalar_t)r.x;
        sh_Wt[((size_t)w0 + 1) * U_pad + u] = (scalar_t)r.y;
        sh_Wt[((size_t)w0 + 2) * U_pad + u] = (scalar_t)r.z;
        sh_Wt[((size_t)w0 + 3) * U_pad + u] = (scalar_t)r.w;
    }

    // 2b) 只加载 nnz_p 行 A[b, i_global, :] → sh_Asel[pidx, :]
    for (int t = threadIdx.y * tx + threadIdx.x;
         t < nnz_p * U;
         t += tx * ty) {
        int idx = t / U;        // 0..nnz_p-1
        int uidx = t - idx * U;
        int i_global = cg_i[idx];   // 全局 i ∈ [0, I_total)
        sh_Asel[(size_t)idx * U_pad + uidx] = A_b[(size_t)i_global * U + uidx];
    }
    __syncthreads();

    // 3) 每个线程负责一个 (k_local, w)，k_global = k_base + k_local
    const int w = threadIdx.x;
    const int k_local = threadIdx.y;       // 0..K_max-1

    if (w >= W || k_local >= K_p) {
        return;
    }

    const int k_global = k_base + k_local; // 对应 out 的全局 K index
    const scalar_t* __restrict__ Wcol = sh_Wt + (size_t)w * U_pad;
    scalar_t acc = scalar_t(0);

    #pragma unroll
    for (int pidx = 0; pidx < 7; ++pidx) {    // nnz_max <= 7
        if (pidx >= nnz_p) break;
        if (cg_k[pidx] != k_global) continue;  // 现在 cg_k 是 global k

        const scalar_t* __restrict__ Arow = sh_Asel + (size_t)pidx * U_pad;

        scalar_t s = scalar_t(0);
        int uu = 0;
        #pragma unroll
        for (; uu + 3 < U; uu += 4) {
            s += Arow[uu+0] * Wcol[uu+0]
               + Arow[uu+1] * Wcol[uu+1]
               + Arow[uu+2] * Wcol[uu+2]
               + Arow[uu+3] * Wcol[uu+3];
        }
        for (; uu < U; ++uu)
            s += Arow[uu] * Wcol[uu];

        acc += s * cg_v[pidx];
    }

    // 写 out_final[b, k_global, w]
    Ob_base[(size_t)k_global * W + w] = acc;
}


// Launcher：Python 侧调用的接口
at::Tensor launch_fused_fctp_forward_multipath_fwd(
    at::Tensor a_seg,        // [B, I_total, U]
    at::Tensor b_seg,        // [B, 1, V]
    at::Tensor w_all,        // [P, U, V, W]
    at::Tensor cg_i_all,     // [P, nnz_max]
    at::Tensor cg_j_all,     // [P, nnz_max]
    at::Tensor cg_k_all,     // [P, nnz_max]
    at::Tensor cg_val_all,   // [P, nnz_max]
    at::Tensor nnz_per_path, // [P]
    at::Tensor K_per_path,   // [P]
    at::Tensor path_offset   // [P]
)
{
    TORCH_CHECK(a_seg.is_cuda() && b_seg.is_cuda() && w_all.is_cuda()
             && cg_i_all.is_cuda() && cg_j_all.is_cuda() && cg_k_all.is_cuda()
             && cg_val_all.is_cuda() && nnz_per_path.is_cuda()
             && K_per_path.is_cuda() && path_offset.is_cuda(),
             "all tensors must be CUDA");

    auto dtype = a_seg.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                "a_seg must be float32 or float64");
    TORCH_CHECK(b_seg.scalar_type()  == dtype &&
                w_all.scalar_type()  == dtype &&
                cg_val_all.scalar_type() == dtype,
                "dtypes must match");

    c10::cuda::CUDAGuard device_guard(a_seg.get_device());

    a_seg        = a_seg.contiguous();
    b_seg        = b_seg.contiguous();
    w_all        = w_all.contiguous();
    cg_i_all     = cg_i_all.contiguous();
    cg_j_all     = cg_j_all.contiguous();
    cg_k_all     = cg_k_all.contiguous();
    cg_val_all   = cg_val_all.contiguous();
    nnz_per_path = nnz_per_path.contiguous();
    K_per_path   = K_per_path.contiguous();
    path_offset  = path_offset.contiguous();

    const int B       = (int)a_seg.size(0);
    const int I_total = (int)a_seg.size(1);
    const int U       = (int)a_seg.size(2);
    TORCH_CHECK(b_seg.size(0)==B && b_seg.size(1)==1,
                "b_seg must be [B,1,V] for this fast path (J==1)");
    const int V       = (int)b_seg.size(2);

    const int P       = (int)w_all.size(0);
    TORCH_CHECK(w_all.size(1)==U && w_all.size(2)==V,
                "w_all must be [P,U,V,W]");
    const int W       = (int)w_all.size(3);

    const int nnz_max = (int)cg_i_all.size(1);
    TORCH_CHECK(cg_i_all.size(0)==P && cg_j_all.size(0)==P &&
                cg_k_all.size(0)==P && cg_val_all.size(0)==P,
                "cg_* shape mismatch");
    TORCH_CHECK(nnz_max <= 7, "nnz_max<=7 required");

    // 从 CPU 侧计算 K_max, K_total，并简单校验 path_offset
    auto K_per_path_cpu  = K_per_path.to(at::kCPU);
    auto path_offset_cpu = path_offset.to(at::kCPU);

    int K_max    = 0;
    int K_total  = 0;
    for (int p = 0; p < P; ++p) {
        int Kp = K_per_path_cpu[p].item<int>();
        TORCH_CHECK(Kp > 0 && Kp <= 7,
                    "each K_p must be in (0,7]");
        if (Kp > K_max) K_max = Kp;
        K_total += Kp;
    }
    {
        int last_off = path_offset_cpu[P-1].item<int>();
        int last_K   = K_per_path_cpu[P-1].item<int>();
        TORCH_CHECK(last_off + last_K == K_total,
                    "path_offset is inconsistent with K_per_path / K_total");
    }

    TORCH_CHECK(U==96, "fast path requires U==96");
    TORCH_CHECK(W % 4 == 0, "fast path requires W%4==0");
    TORCH_CHECK((uint64_t)B <= 65535, "B<=65535 required");

    // 输出: [B, K_total, W]
    auto out = at::empty({B, K_total, W}, a_seg.options());

    const int tx = 128;
    const int ty = K_max;      // 统一对齐到全局 K_max
    dim3 block(tx, ty, 1);
    dim3 grid(P, B, 1);        // 一个 block = (p,b)

    const int U_pad = U + 1;
    size_t shmem_elems = (size_t)W * U_pad + (size_t)nnz_max * U_pad;
    size_t shmem_bytes = shmem_elems * a_seg.element_size();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_fctp_forward_multipath_concat", [&] {
        using scalar_t_ = scalar_t;
        cudaFuncSetAttribute(
            fused_fctp_kernel_fwd_multipath<scalar_t_>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem_bytes);

        fused_fctp_kernel_fwd_multipath<scalar_t_>
            <<<grid, block, (int)shmem_bytes, stream>>>(
                a_seg.data_ptr<scalar_t_>(),
                b_seg.data_ptr<scalar_t_>(),
                w_all.data_ptr<scalar_t_>(),
                cg_i_all.data_ptr<int>(),
                cg_j_all.data_ptr<int>(),
                cg_k_all.data_ptr<int>(),
                cg_val_all.data_ptr<scalar_t_>(),
                nnz_per_path.data_ptr<int>(),
                K_per_path.data_ptr<int>(),
                path_offset.data_ptr<int>(),
                nnz_max,
                P, B, I_total, U, V, W,
                K_max, K_total,
                out.data_ptr<scalar_t_>());
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
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
// fast path 约束：U==96, W%4==0, nnz_max<=7, B<=65535
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
    //
    //    原始布局: W_p[u, v, w]
    //    我们需要: Wt[w, u]（带 pad）
    //
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
    const int I_total = K_total;

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &launch_fused_fctp_forward_multipath_fwd, "Forward: multi-paths fused fully connected tensor product");
    m.def("bwd", &launch_fused_multipath_fctp_backward, "Backward: multi-paths fused fully connected tensor product");
}