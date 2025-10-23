#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

__constant__ int kOff[4]  = {0, 1, 4, 9};  // 前缀和；与前向一致

// -----------------------------------------------------------------------------
// 反向 kernel：与前向同样的 (b,u) 并行、loop over k 的访存与复用策略
// 输入：
//   x:        [B, U]
//   y:        [B, dim_sum]
//   w:        [B, 4*U]
//   grad_out: [B, dim_sum, U]  (对 out 的上游梯度)
// 输出：
//   dx: [B, U]
//   dy: [B, dim_sum]     (atomicAdd 聚合所有 u)
//   dw: [B, 4*U]
// -----------------------------------------------------------------------------
__global__ void cwtp_kernel_bwd_bu_loopk(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ w,
    const double* __restrict__ grad_out,
    double* __restrict__ dx,
    double* __restrict__ dy,
    double* __restrict__ dw,
    int B, int U, int dim_sum)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= B || u >= U) return;

    // 便捷基址
    const double  xv     = x[(size_t)b * U + u];                         // x[b,u]
    const double* y_b    = y  + (size_t)b * dim_sum;                     // y[b,0]
    const double* go_bu  = grad_out + (size_t)b * dim_sum * U + u;       // grad_out[b,0,u]
    const double* w_b    = w  + (size_t)b * (4 * U);                     // w[b,0,u] 起点
    double*       dx_b   = dx + (size_t)b * U;                           // dx[b,0]
    double*       dy_b   = dy + (size_t)b * dim_sum;                     // dy[b,0]
    double*       dw_b   = dw + (size_t)b * (4 * U);                     // dw[b,0,u] 起点

    // 阈值（与前向一致）
    const int o1 = kOff[1];  // 1
    const int o2 = kOff[2];  // 4
    const int o3 = kOff[3];  // 9

    // 线程私有累加器
    double dx_acc  = 0.0;
    double dw_acc0 = 0.0;
    double dw_acc1 = 0.0;
    double dw_acc2 = 0.0;
    double dw_acc3 = 0.0;

#pragma unroll
    for (int k = 0; k < 16; ++k) { // 若 dim_sum != 16，可自动被 break 限制
        if (k >= dim_sum) break;

        // path 选择（与前向一致）
        int p;
        if      (k < o1) p = 0;
        else if (k < o2) p = 1;
        else if (k < o3) p = 2;
        else             p = 3;

        // 取常用值
        const double yv = y_b[k];                            // y[b,k]
        const double wv = w_b[p * U + u];                    // w[b,p,u]
        const double g  = go_bu[(size_t)k * U];              // grad_out[b,k,u]

        // dx 累加： sum_k g * w * y
        dx_acc += g * wv * yv;

        // dy 原子累加： sum_u g * x * w
        atomicAdd(&dy_b[k], g * xv * wv);

        // dw 每 p 独立累加： sum_{k∈path(p)} g * x * y(k)
        const double gx = g * xv;
        if      (p == 0) dw_acc0 += gx * yv;
        else if (p == 1) dw_acc1 += gx * yv;
        else if (p == 2) dw_acc2 += gx * yv;
        else             dw_acc3 += gx * yv;
    }

    // 写回（无竞争）
    dx_b[u]                 = dx_acc;
    dw_b[0 * U + u]         = dw_acc0;
    dw_b[1 * U + u]         = dw_acc1;
    dw_b[2 * U + u]         = dw_acc2;
    dw_b[3 * U + u]         = dw_acc3;
}

// ---------------------------------- C++ 封装 ----------------------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor> cwtp_backward(
    at::Tensor x,          // [B, U], double, CUDA, contiguous
    at::Tensor y,          // [B, dim_sum], double, CUDA, contiguous
    at::Tensor w,          // [B, 4*U], double, CUDA, contiguous
    at::Tensor grad_out)   // [B, dim_sum, U], double, CUDA, contiguous
{
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && w.is_cuda() && grad_out.is_cuda(),
                "x/y/w/grad_out must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kDouble &&
                y.scalar_type() == at::kDouble &&
                w.scalar_type() == at::kDouble &&
                grad_out.scalar_type() == at::kDouble, "all tensors must be double");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && w.is_contiguous() && grad_out.is_contiguous(),
                "all tensors must be contiguous");

    const int B       = x.size(0);
    const int U       = x.size(1);
    const int dim_sum = y.size(1);
    const int P       = 4;

    grad_out = grad_out.reshape({B, dim_sum, U});
    TORCH_CHECK(grad_out.size(0) == B && grad_out.size(1) == dim_sum && grad_out.size(2) == U,
                "grad_out must be [B, dim_sum, U]");
    TORCH_CHECK(w.size(0) == B && w.size(1) == P * U, "w must be [B, 4*U]");

    // 分配梯度张量
    auto opts = x.options();
    at::Tensor dx = at::empty({B, U}, opts).contiguous();               // 逐线程唯一写
    at::Tensor dy = at::zeros({B, dim_sum}, opts).contiguous();         // 需 atomicAdd -> 置零
    at::Tensor dw = at::empty({B, P * U}, opts).contiguous();           // 逐线程唯一写

    // 启动配置与前向一致
    const int threads = 256;
    dim3 blockDim(threads);
    dim3 gridDim((U + threads - 1) / threads, B);

    cwtp_kernel_bwd_bu_loopk<<<gridDim, blockDim>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        grad_out.data_ptr<double>(),
        dx.data_ptr<double>(),
        dy.data_ptr<double>(),
        dw.data_ptr<double>(),
        B, U, dim_sum);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dy, dw};
}

TORCH_LIBRARY(cwtp_bwd, m)
{
    m.def("backward", &cwtp_backward);
}