#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cfloat>
#include <cstdint>

template <typename T> __host__ __device__ inline T ceil_div(T a, T b){ return (a + b - 1) / b; }

// 反向：仅对 a_seg 求梯度
// 输入：
//   b_seg:   [B, 1, V]        (仅用来得到 v*)
//   w_seg:   [U, V, W]        (取 v* 切片并参与累加)
//   grad_out:[B, K, W]        (上游梯度；与前向 out 对齐)
//   cg_i:    [nnz]            (行 i_p，0..I-1)
//   cg_k:    [nnz]            (列 k_p，0..K-1)
//   cg_val:  [nnz]            (对应权重 c_p)
// 维度： U=96，W%4==0，nnz<=7，K<=7，J=1；B*J==B<=65535
// 输出：
//   grad_a:  [B, I, U]
__global__ void fused_fctp_kernel_bwd(
    const double* __restrict__ b_seg,     // [B,1,V]
    const double* __restrict__ w_seg,     // [U,V,W]
    const double* __restrict__ grad_out,  // [B,K,W]
    const int*    __restrict__ cg_i,      // [nnz]
    const int*    __restrict__ cg_k,      // [nnz]
    const double* __restrict__ cg_val,    // [nnz]
    int nnz, int K,
    double*       __restrict__ grad_a,    // [B,I,U] (已清零)
    int B, int I, int U, int V, int W)
{
    const unsigned b = blockIdx.y;                 // J=1 → grid.y = B
    if (b >= (unsigned)B) return;

    const double* __restrict__ brow = b_seg + (size_t)b * V;         // [V]
    const double* __restrict__ dOb  = grad_out + (size_t)b * K * W;  // [K,W]
    double*       __restrict__ dAb  = grad_a  + (size_t)b * I * U;   // [I,U]

    // 1) argmax_v b_seg[b,0,v]
    int vstar = 0;
    if (threadIdx.x==0 && threadIdx.y==0){
        double best=-DBL_MAX; int idx=0;
        #pragma unroll
        for (int v=0; v<V; ++v) { double x=brow[v]; if (x>best){best=x; idx=v;} }
        vstar = idx;
    }
    __shared__ int sh_v; if (threadIdx.x==0 && threadIdx.y==0) sh_v=vstar; __syncthreads();
    const int v = sh_v;

    // 2) shared: 只需要 Wt[W, U+1] 用于累加  (padding 避免 bank 冲突)
    const int U_pad = U + 1;  // 96 -> 97
    extern __shared__ double shmem[];
    double* sh_Wt = shmem;    // [W * U_pad], 存 w_seg^T

    // coop load: w_seg[:,v,:] → sh_Wt[w,u]  (double4 read, scalar write)
    const int Wv  = W / 4;                // fast path: W%4==0
    const int UWv = U * Wv;               // 96 * (W/4)
    const int laneN = blockDim.x * blockDim.y;

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

    // 3) 线程映射：(x=u, y=p)  → 每线程负责一个 (u,p) 的累加
    const int u = threadIdx.x;   // 0..U-1 (blockDim.x >= U=96 → 取 128)
    const int p = threadIdx.y;   // 0..nnz-1 (blockDim.y = nnz ≤ 7)
    if (u < U && p < nnz) {
        const int i = cg_i[p];
        const int k = cg_k[p];
        const double c = cg_val[p];

        const double* __restrict__ Wrow = sh_Wt + (size_t)0 * U_pad + u; // 访问模式：Wrow[w*U_pad + u]

        // 累加 sum_w grad_out[b,k,w] * W[u,w]
        double acc = 0.0;
        // 展开 w 方向（W=96 通常不大；如果很大可再 vectorize 读取 grad_out）
        for (int w = 0; w < W; ++w) {
            const double go = dOb[(size_t)k * W + w];
            const double wu = Wrow[(size_t)w * U_pad]; // 等价 sh_Wt[w*U_pad + u]
            acc += go * wu;
        }
        acc *= c;

        // 原子累加到 grad_a[b, i, u]
        atomicAdd(&dAb[(size_t)i * U + u], acc);
    }
}

// ----------------- C++ 封装（后向，仅返回 dA） -----------------
at::Tensor launch_fused_fctp_backward(
    at::Tensor b_seg,       // [B,1,V], f64, cuda
    at::Tensor w_seg,       // [U,V,W], f64, cuda
    at::Tensor grad_out,    // [B,K,W], f64, cuda
    at::Tensor cg_indices,  // [nnz,2] (i,k) 或 [nnz,3] (i,j,k)，J=1 可用 [i,k]
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
    
    TORCH_CHECK((int)U == 96 && (W % 4)==0 && (uint64_t)B <= 65535, "fast path requires U=96, W%4==0, B<=65535");

    TORCH_CHECK(cg_indices.dim()==2 && (cg_indices.size(1)==2 || cg_indices.size(1)==3),
                "cg_indices must be [nnz,2] or [nnz,3]");
    int nnz = (int)cg_indices.size(0);
    TORCH_CHECK(nnz > 0 && nnz <= 7, "nnz must be in (0,7]");
    

    // 取 (i,k,val)，转 int32
    at::Tensor cg_i = cg_indices.select(1,0).to(at::kInt).contiguous();
    at::Tensor cg_k = cg_indices.select(1, cg_indices.size(1)==2 ? 1 : 2).to(at::kInt).contiguous();
    at::Tensor cg_v = cg_values.contiguous();

    // 输出梯度
    auto dA = at::zeros({(long)B, (long)I, (long)U}, w_seg.options()); // f64, cuda

    // 启动配置：x 覆盖 U=96（→128），y 覆盖 nnz（≤7）
    const int tx = 128;
    const int ty = nnz;
    dim3 block(tx, ty, 1);
    dim3 grid(1, (unsigned)B, 1);

    // 共享内存：Wt[W, U+1]（不需要 A）
    const int U_pad = (int)U + 1;
    size_t shmem_bytes = ((size_t)W * U_pad) * sizeof(double);

    cudaFuncSetAttribute(
        fused_fctp_kernel_bwd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_fctp_kernel_bwd<<<grid, block, (int)shmem_bytes, stream>>>(
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
