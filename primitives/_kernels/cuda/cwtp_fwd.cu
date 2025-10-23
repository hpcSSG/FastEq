#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

__constant__ int kOff[4]  = {0, 1, 4, 9};   // 前缀和：0,1,4,9；sum=16

// -----------------------------------------------------------------------------
// BU_LOOPK 版本：线程按 (b,u) 并行，只读一次 x[b,u]；每个 k 选择对应 path 的 w[b,p,u]
// 写出 out[b,k,u]，提升对 x/w 的复用、减少访存
// -----------------------------------------------------------------------------
__global__ void cwtp_kernel_bu_loopk(
    const double* __restrict__ x,   // [B, U]
    const double* __restrict__ y,   // [B, dim_sum]
    const double* __restrict__ w,   // [B, 4*U]  (P==4, V==1)
    double* __restrict__ out,       // [B, dim_sum, U]
    int B, int U, int dim_sum)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= B || u >= U) return;

    // 读一次 x[b,u]
    const double xv = x[(size_t)b * U + u];

    // 便捷指针
    const double* __restrict__ y_b   = y   + (size_t)b * dim_sum;         // y[b,0]
    double*       __restrict__ out_b = out + (size_t)b * dim_sum * U + u;  // out[b,0,u]

    // 阈值（根据 kOff）
    const int o1 = kOff[1];  // 1
    const int o2 = kOff[2];  // 4
    const int o3 = kOff[3];  // 9

    // 按 k 循环写出；dim_sum=16 时可让编译器展开
#pragma unroll
    for (int k = 0; k < 16; ++k) {   // 若 dim_sum 非 16，可改为 for (int k=0;k<dim_sum;++k)
        if (k >= dim_sum) break;

        // 选择 path p（P==4 的三段阈值判断）
        int p;
        if      (k < o1) p = 0;
        else if (k < o2) p = 1;
        else if (k < o3) p = 2;
        else             p = 3;

        // 读 w[b,p,u]；按 u 连续访问，合并读
        const double wv = w[(size_t)b * (4 * U) + p * U + u];

        // 读 y[b,k]（按 k 前进，缓存命中率较高）
        const double yv = y_b[k];

        // out[b,k,u] = x[b,u] * w[b,p,u] * y[b,k]
        out_b[(size_t)k * U] = xv * wv * yv;
    }
}

at::Tensor cwtp_forward(
    at::Tensor x,   // [B, U], double, CUDA, contiguous
    at::Tensor y,   // [B, dim_sum], double, CUDA, contiguous
    at::Tensor w)   // [B, 4*U], double, CUDA, contiguous
{
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && w.is_cuda(), "x/y/w must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kDouble &&
                y.scalar_type() == at::kDouble &&
                w.scalar_type() == at::kDouble, "x/y/w must be double");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && w.is_contiguous(),
                "x/y/w must be contiguous");

    const int B       = x.size(0);
    const int U       = x.size(1);
    const int dim_sum = y.size(1);
    const int P       = 4;

    TORCH_CHECK(w.size(0) == B, "w.shape[0] must be B");
    TORCH_CHECK(w.size(1) == P * U, "w.shape[1] must be 4*U (P==4)");

    // 输出 [B, dim_sum, U]
    at::Tensor out = at::empty({B, dim_sum, U}, x.options()).contiguous();

    // 启动配置：grid.y=B，grid.x 覆盖 U
    const int threads = 256;
    dim3 blockDim(threads);
    dim3 gridDim((U + threads - 1) / threads, B);
    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    cwtp_kernel_bu_loopk<<<gridDim, blockDim, 0, cur_stream>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        B, U, dim_sum);
    out = out.reshape({B, dim_sum * U});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;  // [B, dim_sum * U]
}

TORCH_LIBRARY(cwtp_fwd, m)
{
    m.def("forward", &cwtp_forward);
}
