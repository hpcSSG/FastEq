// fused_argmax_spmm_tiny_f64.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cfloat>
#include <cstdint>

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

// a_seg:[B,I,U], b_seg:[B,J,V], w_seg:[U,V,W],  out:[B,K,W]
// 稀疏三元组（最多 7 个）：cg_i/cg_j/cg_k/cg_val，长度 nnz<=7
__global__ void fused_argmax_spmm_tiny_kernel_f64(
    const double* __restrict__ a_seg,   // [B,I,U]
    const double* __restrict__ b_seg,   // [B,J,V]
    const double* __restrict__ w_seg,   // [U,V,W]
    const int*    __restrict__ cg_i,    // [nnz]
    const int*    __restrict__ cg_j,    // [nnz]
    const int*    __restrict__ cg_k,    // [nnz]
    const double* __restrict__ cg_val,  // [nnz]
    int nnz,
    double*       __restrict__ out,     // [B,K,W]
    int B, int I, int U, int V, int W, int J, int K)
{
    const unsigned bJ = blockIdx.y;
    if (bJ >= (unsigned)(B*J)) return;

    const int b = (J==1) ? (int)bJ : int(bJ / J);
    const int j = (J==1) ? 0       : int(bJ % J);

    const double* __restrict__ A_b   = a_seg + (size_t)b * I * U;                 // [I,U]
    const double* __restrict__ b_row = b_seg + (size_t)b * J * V + (size_t)j * V; // [V]
    double*       __restrict__ O_b   = out   + (size_t)b * K * W;                 // [K,W]

    // ---- argmax v ----
    int vstar = 0;
    if (threadIdx.x==0 && threadIdx.y==0) {
        double best = -DBL_MAX; int idx = 0;
        for (int v=0; v<V; ++v) { double x = b_row[v]; if (x>best){best=x; idx=v;} }
        vstar = idx;
    }
    __shared__ int sh_v;
    if (threadIdx.x==0 && threadIdx.y==0) sh_v = vstar;
    __syncthreads();
    const int v = sh_v;

    // ---- shared with padding to avoid bank conflicts ----
    const int U_pad = U + 1;  // 96 -> 97
    extern __shared__ double shmem[];
    double* sh_Wt = shmem;                     // [W * U_pad]
    double* sh_A  = shmem + (size_t)W * U_pad; // [I * U_pad]

    // coop load: w_seg[:,v,:] -> sh_Wt[w,u] with padding (double4 read)
    const int Wv = W / 4;                  // 假设 W%4==0；若不满足可加标量分支
    const int UWv = U * Wv;
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < UWv; t += blockDim.x * blockDim.y) {
        int u  = t / Wv;
        int wv = t - u * Wv;
        int w0 = wv << 2;
        const size_t off_g = ((size_t)u * V + (size_t)v) * W + (size_t)w0;
        const double4 r = *reinterpret_cast<const double4*>(w_seg + off_g);
        sh_Wt[((size_t)w0 + 0) * U_pad + u] = r.x;
        sh_Wt[((size_t)w0 + 1) * U_pad + u] = r.y;
        sh_Wt[((size_t)w0 + 2) * U_pad + u] = r.z;
        sh_Wt[((size_t)w0 + 3) * U_pad + u] = r.w;
    }

    // coop load: A_b -> sh_A[i,u] with padding
    const int IU = I * U;
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < IU; t += blockDim.x * blockDim.y) {
        int i = t / U;
        int u = t - i * U;
        sh_A[(size_t)i * U_pad + u] = A_b[(size_t)i * U + u];
    }
    __syncthreads();

    // ---- compute s(i,w), then scatter-add using tiny nnz list ----
    const int w = threadIdx.x;   // 0..W-1
    const int i = threadIdx.y;   // 0..I-1
    if (w < W && i < I) {
        const double* __restrict__ Wcol = sh_Wt + (size_t)w * U_pad;
        const double* __restrict__ Arow = sh_A  + (size_t)i * U_pad;

        double s = 0.0;
        int uu = 0;
#pragma unroll
        for (; uu + 3 < U; uu += 4) {
            s += Arow[uu+0]*Wcol[uu+0]
               + Arow[uu+1]*Wcol[uu+1]
               + Arow[uu+2]*Wcol[uu+2]
               + Arow[uu+3]*Wcol[uu+3];
        }
        for (; uu < U; ++uu) s += Arow[uu]*Wcol[uu];

        // 遍历极小 nnz 列表（≤7）：匹配 (j,i) 后写入
#pragma unroll
        for (int p = 0; p < 7; ++p) {    // 上限展开；nnz<7时条件失效分支很轻
            if (p >= nnz) break;
            int ji = cg_j[p];
            if (ji != j) continue;       // 若 J==1，可去掉此判断
            if (cg_i[p] == i) {
                const int k = cg_k[p];
                const double c = cg_val[p];
                atomicAdd(&O_b[(size_t)k * W + w], s * c);
            }
        }
    }
}

// ---------------- C++ 封装 ----------------
at::Tensor launch_fused_fctp_forward(
    at::Tensor a_seg,      // [B,I,U], f64, cuda
    at::Tensor b_seg,      // [B,J,V], f64, cuda
    at::Tensor w_seg,      // [U,V,W], f64, cuda
    at::Tensor cg_indices, // [nnz,3] (i,j,k), int32 or int64, cuda
    at::Tensor cg_values,  // [nnz], f64, cuda
    int64_t K)             // out K
{
    TORCH_CHECK(a_seg.is_cuda() && b_seg.is_cuda() && w_seg.is_cuda()
             && cg_indices.is_cuda() && cg_values.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a_seg.scalar_type()==at::kDouble && b_seg.scalar_type()==at::kDouble
             && w_seg.scalar_type()==at::kDouble && cg_values.scalar_type()==at::kDouble,
             "float64 required");
    a_seg = a_seg.contiguous();
    b_seg = b_seg.contiguous();
    w_seg = w_seg.contiguous();

    const int B = (int)a_seg.size(0);
    const int I = (int)a_seg.size(1);
    const int U = (int)a_seg.size(2);
    const int J = (int)b_seg.size(1);
    const int V = (int)b_seg.size(2);
    const int W = (int)w_seg.size(2);
    TORCH_CHECK(w_seg.size(0)==U && w_seg.size(1)==V, "U/V mismatch");
    TORCH_CHECK((int64_t)K > 0, "K > 0");
    TORCH_CHECK(cg_indices.dim()==2 && cg_indices.size(1)==3, "cg_indices must be [nnz,3]");

    // 读取极少的 nnz（≤7），转成 int32 三元组
    int nnz = (int)cg_indices.size(0);
    TORCH_CHECK(nnz <= 7, "tiny path requires nnz <= 7");

    at::Tensor cg_i = cg_indices.select(1,0).to(at::kInt).contiguous();
    at::Tensor cg_j = cg_indices.select(1,1).to(at::kInt).contiguous();
    at::Tensor cg_k = cg_indices.select(1,2).to(at::kInt).contiguous();
    at::Tensor cg_v = cg_values.contiguous();

    auto out = at::zeros({B, (int)K, W}, a_seg.options());

    // 快路径约束（与你的常用形状一致）
    const uint64_t BJ = (uint64_t)B * (uint64_t)J;
    TORCH_CHECK(U==96 && (W%4==0) && BJ<=65535, "fast path requires U=96, W%4==0, BJ<=65535");

    // 线程块：x 覆盖 W(96→128)，y 覆盖 I(≤32)
    const int tx = 128;
    const int ty = (I <= 32 ? I : 32);
    dim3 block(tx, ty, 1);
    dim3 grid(1, (unsigned)BJ, 1);

    // 动态共享内存：Wt[W,U+1] + A[I,U+1]
    const int U_pad = U + 1;
    size_t shmem_bytes = ((size_t)W * U_pad + (size_t)I * U_pad) * sizeof(double);

    cudaFuncSetAttribute(
        fused_argmax_spmm_tiny_kernel_f64,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_argmax_spmm_tiny_kernel_f64<<<grid, block, (int)shmem_bytes, stream>>>(
        a_seg.data_ptr<double>(),
        b_seg.data_ptr<double>(),
        w_seg.data_ptr<double>(),
        cg_i.data_ptr<int>(),
        cg_j.data_ptr<int>(),
        cg_k.data_ptr<int>(),
        cg_v.data_ptr<double>(),
        nnz,
        out.data_ptr<double>(),
        B, I, U, V, W, J, (int)K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

TORCH_LIBRARY(fctp_fused3, m) {
    m.def("forward", &launch_fused_fctp_forward);
}
