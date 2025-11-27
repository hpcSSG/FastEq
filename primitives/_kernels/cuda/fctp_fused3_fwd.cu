#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <type_traits>

#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr);  \
  if (_err != cudaSuccess)    \
    AT_ERROR("CUDA error: ", cudaGetErrorString(_err), " at ", __FILE__, ":", __LINE__); \
} while(0)

template <typename T> __host__ __device__ inline T ceil_div(T a, T b){
    return (a + b - 1) / b;
}

// a_seg:[B,I,U], b_seg:[B,1,V], w_seg:[U,V,W], out:[B,K,W]
// 稀疏三元组（nnz<=7）：cg_i:[nnz], cg_k:[nnz], cg_val:[nnz] （J=1 固定，不传 cg_j）
template<typename scalar_t>
__global__ void fused_fctp_kernel_fwd(
    const scalar_t* __restrict__ a_seg,    // [B,I,U]
    const scalar_t* __restrict__ b_seg,    // [B,1,V]
    const scalar_t* __restrict__ w_seg,    // [U,V,W]
    const int*     __restrict__ cg_i,      // [nnz] ∈ [0,I)
    const int*     __restrict__ cg_k,      // [nnz] ∈ [0,K)
    const scalar_t* __restrict__ cg_val,   // [nnz]
    int nnz, int K,
    scalar_t*      __restrict__ out,       // [B,K,W]
    int B, int I, int U, int V, int W)
{
    const unsigned b = blockIdx.y;                     // J=1 → grid.y=B
    if (b >= (unsigned)B) return;

    const scalar_t* __restrict__ A_b   = a_seg + (size_t)b * I * U;     // [I,U]
    const scalar_t* __restrict__ brow  = b_seg + (size_t)b * V;         // [V]
    scalar_t*       __restrict__ Ob    = out   + (size_t)b * K * W;     // [K,W]

    // 1) argmax_v b_seg[b,0,v]
    int vstar = 0;
    if (threadIdx.x==0 && threadIdx.y==0){
        scalar_t best = std::numeric_limits<scalar_t>::lowest();
        int idx = 0;
        #pragma unroll
        for (int v=0; v<V; ++v) {
            scalar_t x = brow[v];
            if (x > best){ best = x; idx = v; }
        }
        vstar = idx;
    }
    __shared__ int sh_v;
    if (threadIdx.x==0 && threadIdx.y==0) sh_v = vstar;
    __syncthreads();
    const int v = sh_v;

    // 2) shared: Wt[W, U+1] + A_sel[nnz, U+1]（只放 nnz 涉及到的 i 行）
    const int U_pad = U + 1;                    // 消 bank 冲突

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shmem_raw[];
    scalar_t* sh_Wt   = reinterpret_cast<scalar_t*>(shmem_raw);                    // W*U_pad
    scalar_t* sh_Asel = sh_Wt + (size_t)W * U_pad;                                 // nnz*U_pad

    // 2a) load w_seg[:,v,:] → sh_Wt[w,u]  (vector read, scalar write)

    // 根据 scalar_t 选择 float4 / double4
    using Vec4 = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double4
    >::type;

    const int Wv = W / 4;
    const int UWv = U * Wv;
    for (int t = threadIdx.y * blockDim.x + threadIdx.x;
         t < UWv;
         t += blockDim.x * blockDim.y) {
        int u  = t / Wv;
        int wv = t - u * Wv;
        int w0 = wv << 2;
        const size_t off = ((size_t)u * V + (size_t)v) * W + (size_t)w0;
        const Vec4 r = *reinterpret_cast<const Vec4*>(w_seg + off);
        sh_Wt[((size_t)w0 + 0) * U_pad + u] = (scalar_t)r.x;
        sh_Wt[((size_t)w0 + 1) * U_pad + u] = (scalar_t)r.y;
        sh_Wt[((size_t)w0 + 2) * U_pad + u] = (scalar_t)r.z;
        sh_Wt[((size_t)w0 + 3) * U_pad + u] = (scalar_t)r.w;
    }

    // 2b) 只加载 nnz 条 i 行到 sh_Asel[p,:]
    for (int t = threadIdx.y * blockDim.x + threadIdx.x;
         t < nnz * U;
         t += blockDim.x * blockDim.y) {
        int p = t / U;
        int uidx = t - p * U;
        int irow = cg_i[p];
        sh_Asel[(size_t)p * U_pad + uidx] = A_b[(size_t)irow * U + uidx];
    }
    __syncthreads();

    // 3) 计算 out[b,k,w]：线程 (x=w, y=k)；遍历 nnz 列表累加（无原子）
    const int w = threadIdx.x;   // 0..W-1
    const int k = threadIdx.y;   // 0..K-1
    if (w < W && k < K) {
        const scalar_t* __restrict__ Wcol = sh_Wt + (size_t)w * U_pad;

        scalar_t acc = scalar_t(0);
    #pragma unroll
        for (int p = 0; p < 7; ++p) {          // nnz≤7：上限展开
            if (p >= nnz) break;
            if (cg_k[p] != k) continue;

            const scalar_t* __restrict__ Arow = sh_Asel + (size_t)p * U_pad;
            scalar_t s = scalar_t(0);
            int uu = 0;
        #pragma unroll
            for (; uu + 3 < U; uu += 4) { // ILP
                s += Arow[uu+0]*Wcol[uu+0]
                   + Arow[uu+1]*Wcol[uu+1]
                   + Arow[uu+2]*Wcol[uu+2]
                   + Arow[uu+3]*Wcol[uu+3];
            }
            for (; uu < U; ++uu) s += Arow[uu]*Wcol[uu];

            acc += s * cg_val[p];
        }
        Ob[(size_t)k * W + w] = acc;
    }
}


at::Tensor launch_fused_fctp_forward(
    at::Tensor a_seg,      // [B,I,U], f32/f64, cuda
    at::Tensor b_seg,      // [B,1,V], f32/f64, cuda
    at::Tensor w_seg,      // [U,V,W], f32/f64, cuda
    at::Tensor cg_indices, // [nnz,2] or [nnz,3]，这里用 [i,k] 或 [i,j,k]；J=1 时只需 (i,k)
    at::Tensor cg_values   // [nnz], f32/f64, cuda
    )
{
    TORCH_CHECK(a_seg.is_cuda() && b_seg.is_cuda() && w_seg.is_cuda()
             && cg_indices.is_cuda() && cg_values.is_cuda(), "CUDA tensors required");

    auto dtype = a_seg.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "a_seg must be float32 or float64");
    TORCH_CHECK(b_seg.scalar_type()  == dtype, "b_seg must match a_seg dtype");
    TORCH_CHECK(w_seg.scalar_type()  == dtype, "w_seg must match a_seg dtype");
    TORCH_CHECK(cg_values.scalar_type() == dtype, "cg_values must match a_seg dtype");

    a_seg = a_seg.contiguous();
    b_seg = b_seg.contiguous();
    w_seg = w_seg.contiguous();

    const int B = (int)a_seg.size(0);
    const int I = (int)a_seg.size(1);
    const int U = (int)a_seg.size(2);

    const int K = I;
 
    TORCH_CHECK(b_seg.size(1)==1, "this fast path requires J==1");
    const int V = (int)b_seg.size(2);
    TORCH_CHECK(w_seg.size(1)==U && w_seg.size(2)==V, "U/V mismatch");
    const int W = (int)w_seg.size(3);
    TORCH_CHECK((int)K <= 7 && K>0, "K<=7 required");
    w_seg = w_seg.view({U, V, W});                 


    // 读取 nnz（≤7），提取 (i,k,val)
    TORCH_CHECK(cg_indices.dim()==2 && (cg_indices.size(1)==2 || cg_indices.size(1)==3),
                "cg_indices must be [nnz,2] (i,k) or [nnz,3] (i,j,k)");
    int nnz = (int)cg_indices.size(0);
    TORCH_CHECK(nnz <= 7, "nnz<=7 required");

    at::Tensor cg_i = cg_indices.select(1,0).to(at::kInt).contiguous();
    at::Tensor cg_k = cg_indices.select(1, cg_indices.size(1)==2 ? 1 : 2).to(at::kInt).contiguous();
    at::Tensor cg_v = cg_values.contiguous();

    auto out = at::empty({B, (int)K, W}, a_seg.options());

    // 快路径约束：U=96, W%4==0, B<=65535
    TORCH_CHECK(U==96 && (W%4==0) && (uint64_t)B <= 65535,
                "fast path requires U=96, W%4==0, B<=65535, J=1");

    // 线程块：x 覆盖 W(96→128)，y 覆盖 K(≤7)
    const int tx = 128;
    const int ty = (int)K;     // 1..7
    dim3 block(tx, ty, 1); // 一个 block 负责一个 batch 样本的整个 [K,W]
    dim3 grid(1, (unsigned)B, 1);

    // 动态 shared：Wt[W,U+1] + Asel[nnz,U+1]，按元素大小计算
    const int U_pad = U + 1;
    size_t shmem_elems = (size_t)W * U_pad + (size_t)nnz * U_pad;
    size_t shmem_bytes = shmem_elems * a_seg.element_size();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "fused_fctp_forward", [&] {
        using scalar_t = scalar_t;
        cudaFuncSetAttribute(
            fused_fctp_kernel_fwd<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem_bytes);

        fused_fctp_kernel_fwd<scalar_t>
            <<<grid, block, (int)shmem_bytes, stream>>>(
                a_seg.data_ptr<scalar_t>(),
                b_seg.data_ptr<scalar_t>(),
                w_seg.data_ptr<scalar_t>(),
                cg_i.data_ptr<int>(),
                cg_k.data_ptr<int>(),
                cg_v.data_ptr<scalar_t>(),
                nnz, (int)K,
                out.data_ptr<scalar_t>(),
                B, I, U, V, W);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

TORCH_LIBRARY(fctp_fused3_fwd, m) {
    m.def("forward", &launch_fused_fctp_forward);
}
