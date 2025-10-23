#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cfloat>
#include <cstdint>

template <typename T> __host__ __device__ inline T ceil_div(T a, T b){ return (a + b - 1) / b; }

// 后向（仅对 a_seg 求梯度），无原子版：
// 约束：J==1, U==96, W%4==0, K<=7, nnz<=7, B<=65535
// 输入：
//   b_seg   : [B,1,V]         （求 v*=argmax_v b[b,0,v]）
//   w_seg   : [U,V,W]         （取 v* 切片参与）
//   grad_out: [B,K,W]
//   cg_i    : [nnz]   （每个非零的 i_p ∈ [0,I)）
//   cg_k    : [nnz]   （每个非零的 k_p ∈ [0,K)）
//   cg_val  : [nnz]   （每个非零的权重 c_p）
// 输出：
//   grad_a  : [B,I,U] （仅 a_seg 的梯度）
__global__ void fused_fctp_backward_kernel(
    const double* __restrict__ b_seg,     // [B,1,V]
    const double* __restrict__ w_seg,     // [U,V,W]
    const double* __restrict__ grad_out,  // [B,K,W]
    const int*    __restrict__ cg_i,      // [nnz]  (≤7)
    const int*    __restrict__ cg_k,      // [nnz]
    const double* __restrict__ cg_val,    // [nnz]
    int nnz, int K,
    double*       __restrict__ grad_a,    // [B,I,U]
    int B, int I, int U, int V, int W)
{
    const unsigned b = blockIdx.y;                 // J=1 → 一块一个样本 b
    if (b >= (unsigned)B) return;

    const double* __restrict__ brow = b_seg + (size_t)b * V;         // [V]
    const double* __restrict__ dOb  = grad_out + (size_t)b * K * W;  // [K,W]
    double*       __restrict__ dAb  = grad_a  + (size_t)b * I * U;   // [I,U]

    // ---- 1) argmax_v ----
    int vstar = 0;
    if (threadIdx.x==0 && threadIdx.y==0){
        double best=-DBL_MAX; int idx=0;
        #pragma unroll
        for (int v=0; v<V; ++v) { double x=brow[v]; if (x>best){best=x; idx=v;} }
        vstar = idx;
    }
    __shared__ int sh_v; if (threadIdx.x==0 && threadIdx.y==0) sh_v=vstar; __syncthreads();
    const int v = sh_v;

    // ---- 2) 共享内存布局 ----
    // Wt  : [W, U_pad]  存 w_seg^T（消 bank 冲突）
    // G   : [K, U]      G[k,u] = sum_w grad_out[b,k,w] * W[u,w]
    // tmp : [nnz, U]    tmp[p,u] = cg_val[p] * G[cg_k[p], u]
    const int U_pad = U + 1;     // 96 -> 97
    extern __shared__ double shmem[];
    double* sh_Wt  = shmem;                                  // size W*U_pad
    double* sh_G   = sh_Wt  + (size_t)W * U_pad;             // size K*U
    double* sh_tmp = sh_G   + (size_t)K * U;                 // size nnz*U

    // （小数组）i 分组：把相同 i 的若干非零合组
    __shared__ int group_i[7];       // unique i 值，最多 7 组
    __shared__ int group_map[7];     // group_map[p] = 该 p 属于哪一组
    __shared__ int Gcnt;             // 组数 (≤ nnz)

    // ---- 3) coop load: w_seg[:,v,:] → sh_Wt[w,u] (double4 读，标量写，带 padding) ----
    const int laneN = blockDim.x * blockDim.y;
    const int Wv = W / 4;                 // 假设 W%4==0
    const int UWv = U * Wv;               // 96 * (W/4)
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < UWv; t += laneN) {
        int u  = t / Wv;
        int wv = t - u * Wv;
        int w0 = wv << 2;
        const size_t off = ((size_t)u * V + (size_t)v) * W + (size_t)w0;
        const double4 r = *reinterpret_cast<const double4*>(w_seg + off);
        sh_Wt[((size_t)w0 + 0) * U_pad + u] = r.x;
        sh_Wt[((size_t)w0 + 1) * U_pad + u] = r.y;
        sh_Wt[((size_t)w0 + 2) * U_pad + u] = r.z;
        sh_Wt[((size_t)w0 + 3) * U_pad + u] = r.w;
    }
    __syncthreads();

    // ---- 4) 预计算 G[k,u] = sum_w grad_out[b,k,w] * W[u,w] ----
    // 线程映射： (x=u in [0,U), y=k in [0,K) )，每线程独立计算一个 (k,u)
    const int u = threadIdx.x;
    const int y = threadIdx.y;
    if (u < U && y < K) {
        const int k = y;
        const double* __restrict__ Wcol_u = sh_Wt + (size_t)0 * U_pad + u; // 使用步长 U_pad 访问 [w,u]
        double acc = 0.0;
        #pragma unroll
        for (int w = 0; w < W; ++w) {
            acc += dOb[(size_t)k * W + w] * Wcol_u[(size_t)w * U_pad];
        }
        sh_G[(size_t)k * U + u] = acc;
    }
    __syncthreads();

    // ---- 5) 组建 unique i 的分组 & 为 tmp 准备映射 ----
    if (threadIdx.x==0 && threadIdx.y==0) {
        int g = 0;
        #pragma unroll
        for (int p=0; p<7; ++p) {
            if (p >= nnz) break;
            int ip = cg_i[p];
            int gid = -1;
            // 查找是否已有该 i 的组
            #pragma unroll
            for (int t=0; t<7; ++t) {
                if (t >= g) break;
                if (group_i[t] == ip) { gid = t; break; }
            }
            if (gid == -1) { group_i[g] = ip; gid = g; ++g; }
            group_map[p] = gid;
        }
        Gcnt = g;
    }
    __syncthreads();

    // ---- 6) 计算 tmp[p,u] = cg_val[p] * G[cg_k[p], u] （并行： (x=u, y=p) ）----
    const int p = threadIdx.y;
    if (u < U && p < nnz) {
        const int kp = cg_k[p];
        const double c = cg_val[p];
        sh_tmp[(size_t)p * U + u] = c * sh_G[(size_t)kp * U + u];
    }
    __syncthreads();

    // ---- 7) 对每个 unique 组 g 并行归并 tmp → dA 行，无原子写回 ----
    //   线程映射： (x=u, y=g)；每个组独有一个 y 线负责整行的写
    const int g = threadIdx.y;
    if (u < U && g < Gcnt) {
        double acc = 0.0;
        // 累加所有属于该组 g 的 p
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            if (t >= nnz) break;
            if (group_map[t] == g) acc += sh_tmp[(size_t)t * U + u];
        }
        const int irow = group_i[g];
        dAb[(size_t)irow * U + u] += acc;   // 唯一写者：无原子
    }
}

// ----------------- C++ 封装（仅返回 dA） -----------------
at::Tensor launch_fused_fctp_backward(
    at::Tensor b_seg,       // [B,1,V], f64, cuda
    at::Tensor w_seg,       // [U,V,W], f64, cuda
    at::Tensor grad_out,    // [B,K,W], f64, cuda
    at::Tensor cg_indices,  // [nnz,2] or [nnz,3]： (i,k) 或 (i,j,k)；J=1 用 (i,k)
    at::Tensor cg_values    // [nnz], f64
)
{
    TORCH_CHECK(b_seg.is_cuda() && w_seg.is_cuda() && grad_out.is_cuda()
             && cg_indices.is_cuda() && cg_values.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(b_seg.scalar_type()==at::kDouble && w_seg.scalar_type()==at::kDouble
             && grad_out.scalar_type()==at::kDouble && cg_values.scalar_type()==at::kDouble,
             "float64 required");

    b_seg    = b_seg.contiguous();
    w_seg    = w_seg.contiguous();
    grad_out = grad_out.contiguous();

    const int B = (int)b_seg.size(0);
    const int U = (int)w_seg.size(0);
    const int V = (int)w_seg.size(1);
    const int W = (int)w_seg.size(2);
    const int K = (int)grad_out.size(1);
    const int I = K;

    TORCH_CHECK((int)U==96 && (W % 4)==0 && (uint64_t)B <= 65535, "fast path: U=96, W%4==0, B<=65535");
    TORCH_CHECK((int)K <= 7 && (int)K > 0, "K<=7 required");
    TORCH_CHECK(cg_indices.dim()==2 && (cg_indices.size(1)==2 || cg_indices.size(1)==3),
                "cg_indices must be [nnz,2] or [nnz,3]");

    int nnz = (int)cg_indices.size(0);
    TORCH_CHECK(nnz > 0 && nnz <= 7, "nnz in (0,7] required");

    // 取 (i,k,val)，转 int32
    at::Tensor cg_i = cg_indices.select(1,0).to(at::kInt).contiguous();
    at::Tensor cg_k = cg_indices.select(1, cg_indices.size(1)==2 ? 1 : 2).to(at::kInt).contiguous();
    at::Tensor cg_v = cg_values.contiguous();

    // 输出梯度（置零）
    auto dA = at::zeros({(long)B, (long)I, (long)U}, w_seg.options()); // f64, cuda

    // 线程块：x 覆盖 U=96（→128），y 覆盖 max(K, nnz)（≤7）
    const int tx = 128;
    const int ty = K; // K == nnz
    dim3 block(tx, ty, 1);
    dim3 grid(1, (unsigned)B, 1);

    // 动态共享内存：Wt[W,U+1] + G[K,U] + tmp[nnz,U]
    const int U_pad = (int)U + 1;
    size_t shmem_bytes =
        ((size_t)W * U_pad + (size_t)K * U + (size_t)nnz * U) * sizeof(double);

    cudaFuncSetAttribute(
        fused_fctp_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_fctp_backward_kernel<<<grid, block, (int)shmem_bytes, stream>>>(
        b_seg.data_ptr<double>(),
        w_seg.data_ptr<double>(),
        grad_out.data_ptr<double>(),
        cg_i.data_ptr<int>(),
        cg_k.data_ptr<int>(),
        cg_v.data_ptr<double>(),
        nnz, (int)K,
        dA.data_ptr<double>(),
        (int)B, (int)I, (int)U, (int)V, (int)W);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dA;
}

TORCH_LIBRARY(fctp_fused3_bwd, m) {
    m.def("backward", &launch_fused_fctp_backward);
}
