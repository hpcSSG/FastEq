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
// a_all:        [B, I_total, U]
// b_all:        [B, 1, V]          // fast path: J==1
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

/*
DRAM Frequency                  Ghz         2.62
SM Frequency                    Ghz         1.59
Elapsed Cycles                cycle    1,681,558
Memory Throughput                 %        69.08
DRAM Throughput                   %         2.41
Duration                         ms         1.06
L1/TEX Cache Throughput           %        70.53
L2 Cache Throughput               %        17.00
SM Active Cycles              cycle 1,646,820.16
Compute (SM) Throughput           %        32.64
*/
// Mace-OFF large 的shared memory 超过最大可分配
template<typename scalar_t>
__global__ void fused_fctp_kernel_fwd_multipath(
    const scalar_t* __restrict__ a_all,        // [B, I_total, U]
    const scalar_t* __restrict__ b_all,        // [B, 1, V]
    const scalar_t* __restrict__ w_all,        // [P, U, V, W]
    const int*     __restrict__ cg_i_all,      // [P, nnz_max]
    const int*     __restrict__ cg_j_all,      // [P, nnz_max]
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

    // a_all[b]: [I_total, U]
    const scalar_t* __restrict__ A_b  = a_all + (size_t)b * I_total * U;
    // b_all[b]: [1, V] → 当作 [V]
    const scalar_t* __restrict__ brow = b_all + (size_t)b * V;
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


constexpr int W_TILE = 16;

template<typename scalar_t>
__global__ void fused_fctp_kernel_fwd_multipath_tiledW(
    const scalar_t* __restrict__ a_all,        // [B, I_total, U]
    const scalar_t* __restrict__ b_all,        // [B, 1, V]
    const scalar_t* __restrict__ w_all,        // [P, U, V, W]
    const int*     __restrict__ cg_i_all,      // [P, nnz_max]
    const int*     __restrict__ cg_j_all,      // [P, nnz_max]
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

    const int*      cg_i = cg_i_all   + (size_t)p * nnz_max;
    //const int*      cg_j = cg_j_all   + (size_t)p * nnz_max; 
    const int*      cg_k = cg_k_all   + (size_t)p * nnz_max;
    const scalar_t* cg_v = cg_val_all + (size_t)p * nnz_max;

    // a_all[b]: [I_total, U]
    const scalar_t* __restrict__ A_b  = a_all + (size_t)b * I_total * U;
    // b_all[b]: [1, V]
    const scalar_t* __restrict__ brow = b_all + (size_t)b * V;
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

    // 2) shared: Wt[W_TILE, U+1] + Asel[nnz_max, U+1]
    const int U_pad = U + 1;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shmem_raw[];
    scalar_t* sh_Wt   = reinterpret_cast<scalar_t*>(shmem_raw);               // [W_TILE, U_pad]
    scalar_t* sh_Asel = sh_Wt + (size_t)W_TILE * U_pad;                        // [nnz_max, U_pad]

    using Vec4 = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double4
    >::type;

    const int tx = blockDim.x;
    const int ty = blockDim.y;

    const int k_local = threadIdx.y;   // 0..K_max-1
    const int w_thread = threadIdx.x;  // 0..tx-1

    // 先加载 A 的 nnz_p 行到 shared, 整个 W 共用
    for (int t = threadIdx.y * tx + threadIdx.x;
         t < nnz_p * U;
         t += tx * ty) {
        int idx  = t / U;        // 0..nnz_p-1
        int uidx = t - idx * U;
        int i_global = cg_i[idx];   // 全局 i ∈ [0, I_total)
        sh_Asel[(size_t)idx * U_pad + uidx] = A_b[(size_t)i_global * U + uidx];
    }
    __syncthreads();

    // k_local 超出 K_p 的线程不参与计算，但必须参与所有 __syncthreads
    const bool alive = (k_local < K_p);

    // 3) 沿 W 方向分块
    for (int w_base = 0; w_base < W; w_base += W_TILE) {
        const int W_this   = min(W_TILE, W - w_base);
        const int Wv_this  = W_this / 4;       // Vec4 数量
        const int Wv_off   = w_base / 4;       // 本 tile 在 Vec4 维度的起始偏移

        // 3a) 把 W_p[:, v, :] 对当前 tile 的部分加载到 sh_Wt[w_local, u]
        const int num_vec = U * Wv_this;
        for (int t = threadIdx.y * tx + threadIdx.x;
             t < num_vec;
             t += tx * ty) {
            int u   = t / Wv_this;          // 0..U-1
            int wvt = t - u * Wv_this;      // 0..Wv_this-1
            int wv  = Wv_off + wvt;         // 全局 vec index
            int w0  = wv << 2;              // 全局 w 起始（保证 4 对齐）

            const size_t off = ((size_t)u * V + (size_t)v) * W + (size_t)w0;
            const Vec4 r = *reinterpret_cast<const Vec4*>(W_p + off);

            int w_local0 = w0 - w_base;     // 0..W_this-1，步长为 4

            sh_Wt[((size_t)w_local0 + 0) * U_pad + u] = (scalar_t)r.x;
            sh_Wt[((size_t)w_local0 + 1) * U_pad + u] = (scalar_t)r.y;
            sh_Wt[((size_t)w_local0 + 2) * U_pad + u] = (scalar_t)r.z;
            sh_Wt[((size_t)w_local0 + 3) * U_pad + u] = (scalar_t)r.w;
        }

        __syncthreads();  // 确保该 tile 的 W 已加载完

        const int w_local  = w_thread;           // 0..tx-1
        const bool active  = alive && (w_local < W_this);

        scalar_t acc = scalar_t(0);

        if (active) {
            const int w_global = w_base + w_local;
            const int k_global = k_base + k_local;

            const scalar_t* __restrict__ Wcol = sh_Wt + (size_t)w_local * U_pad;

            // 每个 (k_global, w_global) 累加所有 nnz
            #pragma unroll
            for (int pidx = 0; pidx < 7; ++pidx) {    // nnz_max <= 7
                if (pidx >= nnz_p) break;
                if (cg_k[pidx] != k_global) continue;  // cg_k 是 global k

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

            // 写 tile (k_global, w_global)
            Ob_base[(size_t)k_global * W + w_global] = acc;
        }
        __syncthreads();
    }
}



// Launcher：Python 侧调用的接口
at::Tensor launch_fused_multipath_fctp(
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
    TORCH_CHECK(a_all.is_cuda() && b_all.is_cuda() && w_all.is_cuda()
             && cg_i_all.is_cuda() && cg_j_all.is_cuda() && cg_k_all.is_cuda()
             && cg_val_all.is_cuda() && nnz_per_path.is_cuda()
             && K_per_path.is_cuda() && path_offset.is_cuda(),
             "all tensors must be CUDA");

    auto dtype = a_all.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                "a_all must be float32 or float64");
    TORCH_CHECK(b_all.scalar_type()  == dtype &&
                w_all.scalar_type()  == dtype &&
                cg_val_all.scalar_type() == dtype,
                "dtypes must match");

    // c10::cuda::CUDAGuard device_guard(a_all.get_device());

    const int B       = (int)a_all.size(0);
    TORCH_CHECK(a_all.size(1) % U == 0, "a_all.size(1) must be divisible by U");
    const int I_total = static_cast<int>(a_all.size(1) / U);
    const int P       = (int)nnz_per_path.size(0);
    
    a_all = a_all.view({B, I_total, U});
    b_all = b_all.view({B, 1, V});
    w_all = w_all.view({P, U, V, W});

    a_all        = a_all.contiguous();
    b_all        = b_all.contiguous();
    w_all        = w_all.contiguous();
    cg_i_all     = cg_i_all.contiguous();
    cg_j_all     = cg_j_all.contiguous();
    cg_k_all     = cg_k_all.contiguous();
    cg_val_all   = cg_val_all.contiguous();
    nnz_per_path = nnz_per_path.contiguous();
    K_per_path   = K_per_path.contiguous();
    path_offset  = path_offset.contiguous();
    
    const int nnz_max = (int)cg_i_all.size(1);
    int K_max = nnz_max;
    TORCH_CHECK(cg_i_all.size(0)==P && cg_j_all.size(0)==P &&
                cg_k_all.size(0)==P && cg_val_all.size(0)==P,
                "cg_* shape mismatch");
    TORCH_CHECK(nnz_max <= 7, "nnz_max<=7 required");

    TORCH_CHECK(U==96 || U==128 || U==224, "fast path requires U==96/128/224");
    TORCH_CHECK(W==96 || W==128 || W==224, "fast path requires W==96/128/224");
    TORCH_CHECK((uint64_t)B <= 65535, "B<=65535 required");

    // 输出: [B, K_total, W]
    auto out = at::empty({B, K_total, W}, a_all.options());

    const int tx = 128;
    const int ty = K_max;      // 统一对齐到全局 K_max
    dim3 block(tx, ty, 1);
    dim3 grid(P, B, 1);        // 一个 block = (p,b)

    const int U_pad = U + 1;
    size_t shmem_elems = (size_t)W * U_pad + (size_t)nnz_max * U_pad;
    size_t shmem_bytes = shmem_elems * a_all.element_size();
    //std::cout<<"fast fctp launch param, P:"<<P<<" B:"<<B<<" shmem_byte:"<<shmem_bytes<<std::endl;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_fctp_forward_multipath_concat", [&] {
        using scalar_t_ = scalar_t;
        cudaFuncSetAttribute(
            fused_fctp_kernel_fwd_multipath<scalar_t_>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem_bytes);

        fused_fctp_kernel_fwd_multipath<scalar_t_>
            <<<grid, block, (int)shmem_bytes, stream>>>(
                a_all.data_ptr<scalar_t_>(),
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
                P, B, I_total, U, V, W,
                K_max, K_total,
                out.data_ptr<scalar_t_>());
    });

    out = out.view({B, K_total * W});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Launcher：Python 侧调用的接口
at::Tensor launch_fused_multipath_fctp_tile(
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
    TORCH_CHECK(a_all.is_cuda() && b_all.is_cuda() && w_all.is_cuda()
             && cg_i_all.is_cuda() && cg_j_all.is_cuda() && cg_k_all.is_cuda()
             && cg_val_all.is_cuda() && nnz_per_path.is_cuda()
             && K_per_path.is_cuda() && path_offset.is_cuda(),
             "all tensors must be CUDA");

    auto dtype = a_all.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                "a_all must be float32 or float64");
    TORCH_CHECK(b_all.scalar_type()  == dtype &&
                w_all.scalar_type()  == dtype &&
                cg_val_all.scalar_type() == dtype,
                "dtypes must match");

    // c10::cuda::CUDAGuard device_guard(a_all.get_device());

    const int B       = (int)a_all.size(0);
    TORCH_CHECK(a_all.size(1) % U == 0, "a_all.size(1) must be divisible by U");
    const int I_total = static_cast<int>(a_all.size(1) / U);
    const int P       = (int)nnz_per_path.size(0);
    
    a_all = a_all.view({B, I_total, U});
    b_all = b_all.view({B, 1, V});
    w_all = w_all.view({P, U, V, W});

    a_all        = a_all.contiguous();
    b_all        = b_all.contiguous();
    w_all        = w_all.contiguous();
    cg_i_all     = cg_i_all.contiguous();
    cg_j_all     = cg_j_all.contiguous();
    cg_k_all     = cg_k_all.contiguous();
    cg_val_all   = cg_val_all.contiguous();
    nnz_per_path = nnz_per_path.contiguous();
    K_per_path   = K_per_path.contiguous();
    path_offset  = path_offset.contiguous();
    
    const int nnz_max = (int)cg_i_all.size(1);
    int K_max = nnz_max;
    TORCH_CHECK(cg_i_all.size(0)==P && cg_j_all.size(0)==P &&
                cg_k_all.size(0)==P && cg_val_all.size(0)==P,
                "cg_* shape mismatch");
    TORCH_CHECK(nnz_max <= 7, "nnz_max<=7 required");

    TORCH_CHECK(U==96 || U==128 || U==224, "fast path requires U==96/128/224");
    TORCH_CHECK(W==96 || W==128 || W==224, "fast path requires W==96/128/224");
    TORCH_CHECK((uint64_t)B <= 65535, "B<=65535 required");

    // 输出: [B, K_total, W]
    auto out = at::empty({B, K_total, W}, a_all.options());

    const int tx = 32;
    const int ty = K_max;
    dim3 block(tx, ty, 1);
    dim3 grid(P, B, 1);
    const int U_pad = U + 1;

    size_t shmem_elems = (size_t)W_TILE * U_pad + (size_t)nnz_max * U_pad;
    size_t shmem_bytes = shmem_elems * a_all.element_size();
    //std::cout << "fast fctp launch param, P:" << P
    //        << " B:" << B
    //        << " shmem_byte:" << shmem_bytes << std::endl;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_fctp_kernel_fwd_multipath_tiledW", [&] {
        using scalar_t_ = scalar_t;

        fused_fctp_kernel_fwd_multipath_tiledW<scalar_t_>
            <<<grid, block, (int)shmem_bytes, stream>>>(
                a_all.data_ptr<scalar_t_>(),
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
                P, B, I_total, U, V, W,
                K_max, K_total,
                out.data_ptr<scalar_t_>());
    });


    out = out.view({B, K_total * W});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


TORCH_LIBRARY(fctp_fused_multipath_fwd, m) {
    m.def("forward", &launch_fused_multipath_fctp_tile);
}