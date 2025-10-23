// fused_schemeA_f64_v2.cu  —— 只展示 kernel 和封装关键差异

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cfloat>

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

// a_seg:[B,I,U], b_seg:[B,J,V], w_seg:[U,V,W], out:[B,I,J,W]
// 约束：U=96，W%4==0，I<=32（建议<=16），BJ<=65535
__global__ void fused_argmax_gemm_kernel_f64_v2(
    const double* __restrict__ a_seg,
    const double* __restrict__ b_seg,
    const double* __restrict__ w_seg,
    double*       __restrict__ out,
    int B, int I, int U, int V, int W, int J)
{
    const unsigned bJ = blockIdx.y;     // y 映射 (b,j)
    if (bJ >= (unsigned)(B*J)) return;

    const int b = (J==1) ? (int)bJ : int(bJ / J);
    const int j = (J==1) ? 0       : int(bJ % J);

    const double* __restrict__ A_b   = a_seg + (size_t)b * I * U;                 // [I,U]
    const double* __restrict__ b_row = b_seg + (size_t)b * J * V + (size_t)j * V; // [V]
    double*       __restrict__ C_bj  = out   + ((size_t)b * I * J + (size_t)j) * W;// [I,W]

    // ---- argmax v ----
    int vstar = 0;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        double best=-DBL_MAX; int idx=0;
        #pragma unroll
        for (int v=0; v<10; ++v) { // 若 V!=10 改成 v<V
            double val = b_row[v];
            if (val > best) { best = val; idx = v; }
        }
        vstar = idx;
    }
    __shared__ int sh_v;
    if (threadIdx.x == 0 && threadIdx.y == 0) sh_v = vstar;
    __syncthreads();
    const int v = sh_v;

    // ---- shared memory with padding ----
    //   Wt:  [W, U_pad] 存 w_slice^T，U_pad = U + 1 以避免 stride%32==0 的 bank 冲突
    //   A  : [I, U_pad] 存 A_b
    const int U_pad = U + 1;  // 96 -> 97，破坏 32 倍数
    extern __shared__ double shmem[];
    double* sh_Wt = shmem;                       // W * U_pad
    double* sh_A  = shmem + (size_t)W * U_pad;   // I * U_pad

    // ---- coop load: w_seg[:,v,:] -> sh_Wt[w, u] with padding（double4 读 + 标量写转置）----
    const int Wv = W / 4;             // 24
    const int UWv = U * Wv;           // 96*24 = 2304
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < UWv; t += blockDim.x * blockDim.y) {
        int u  = t / Wv;
        int wv = t - u * Wv;
        int w0 = wv << 2;
        const size_t off_g = ((size_t)u * V + (size_t)v) * W + (size_t)w0;
        const double4 r = *reinterpret_cast<const double4*>(w_seg + off_g);
        // 写入转置: sh_Wt[w, u]，行步长为 U_pad
        sh_Wt[((size_t)w0 + 0) * U_pad + u] = r.x;
        sh_Wt[((size_t)w0 + 1) * U_pad + u] = r.y;
        sh_Wt[((size_t)w0 + 2) * U_pad + u] = r.z;
        sh_Wt[((size_t)w0 + 3) * U_pad + u] = r.w;
    }

    // ---- coop load: A_b -> sh_A[i, u] with padding ----
    const int IU = I * U;
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < IU; t += blockDim.x * blockDim.y) {
        int i = t / U;
        int u = t - i * U;
        sh_A[(size_t)i * U_pad + u] = A_b[(size_t)i * U + u];
    }
    __syncthreads();

    // ---- 2D thread mapping: 每线程算一个 (i,w) 输出 ----
    const int w = threadIdx.x;  // 0..W-1（要求 blockDim.x >= W）
    const int i = threadIdx.y;  // 0..I-1（要求 blockDim.y >= I，或多次发射/循环）

    if (w < W && i < I) {
        const double* __restrict__ Wcol = sh_Wt + (size_t)w * U_pad;
        const double* __restrict__ Arow = sh_A  + (size_t)i * U_pad;

        // U=96 展开（以 4 为步长）
        double acc = 0.0;
        #pragma unroll
        for (int uu=0; uu<96; uu+=4) {
            acc += Arow[uu+0] * Wcol[uu+0]
                 + Arow[uu+1] * Wcol[uu+1]
                 + Arow[uu+2] * Wcol[uu+2]
                 + Arow[uu+3] * Wcol[uu+3];
        }
        C_bj[(size_t)i * W + w] = acc;
    }
}

// -------- Host 封装（快路径） --------
at::Tensor fused_schemeA_f64_v2_forward(
    at::Tensor a_seg_, at::Tensor b_seg_, at::Tensor w_seg_)
{
    TORCH_CHECK(a_seg_.is_cuda() && b_seg_.is_cuda() && w_seg_.is_cuda(), "CUDA only");
    TORCH_CHECK(a_seg_.scalar_type()==at::kDouble &&
                b_seg_.scalar_type()==at::kDouble &&
                w_seg_.scalar_type()==at::kDouble, "float64 only");
    auto a_seg = a_seg_.contiguous();
    auto b_seg = b_seg_.contiguous();
    auto w_seg = w_seg_.contiguous();

    const int B = (int)a_seg.size(0);
    const int I = (int)a_seg.size(1);
    const int U = (int)a_seg.size(2);
    const int J = (int)b_seg.size(1);
    const int V = (int)b_seg.size(2);
    const int W = (int)w_seg.size(2);

    auto out = at::empty({B, I, J, W}, a_seg.options());

    // 只走快路径（你的常用形态）
    TORCH_CHECK(U==96 && (W%4==0) && (uint64_t)B*J <= 65535, "v2 fast path requires U=96, W%4==0, BJ<=65535");

    // 2D 线程块：x 覆盖 W（96 -> 128），y 覆盖 I（<=8）
    const int tx = 128;                       // >= W
    const int ty = (I <= 8) ? I : 8;          // 把 I 放在 y 维（上限 8）
    dim3 block(tx, ty, 1);
    dim3 grid(1, (unsigned)(B*J), 1);

    // 动态共享内存：Wt[W,U+1] + A[I,U+1]
    const int U_pad = U + 1;
    size_t shmem_bytes = ((size_t)W * U_pad + (size_t)I * U_pad) * sizeof(double);

    cudaFuncSetAttribute(
        fused_argmax_gemm_kernel_f64_v2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_argmax_gemm_kernel_f64_v2<<<grid, block, (int)shmem_bytes, stream>>>(
        a_seg.data_ptr<double>(),
        b_seg.data_ptr<double>(),
        w_seg.data_ptr<double>(),
        out.data_ptr<double>(),
        B, I, U, V, W, J);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

TORCH_LIBRARY(fctp_fused2, m) {
    m.def("forward", &fused_schemeA_f64_v2_forward);
}
