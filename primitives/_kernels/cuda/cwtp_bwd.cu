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
// 输入： x:[B,U], y:[B,dim_sum], w:[B,4*U], grad_out:[B,dim_sum,U]
// 输出： dx:[B,U], dy:[B,dim_sum], dw:[B,4*U]
// 兼容 B > 65535：b = blockIdx.y + blockIdx.z * gridDim.y
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
    const unsigned int b = blockIdx.y + blockIdx.z * gridDim.y;  // 线性化 b
    if (b >= (unsigned)B || u >= U) return;

    // 基址
    const size_t off_x   = (size_t)b * U + u;                // x[b,u]
    const size_t off_y   = (size_t)b * dim_sum;              // y[b,0]
    const size_t off_go  = (size_t)b * dim_sum * U + u;      // grad_out[b,0,u]
    const size_t off_w   = (size_t)b * (4 * U);              // w[b,0,0] 起点
    const size_t off_dx  = (size_t)b * U;                    // dx[b,0]
    const size_t off_dy  = (size_t)b * dim_sum;              // dy[b,0]
    const size_t off_dw  = (size_t)b * (4 * U);              // dw[b,0,0]

    const double  xv     = x[off_x];
    const double* y_b    = y  + off_y;
    const double* go_bu  = grad_out + off_go;
    const double* w_b    = w  + off_w;
    double*       dx_b   = dx + off_dx;
    double*       dy_b   = dy + off_dy;
    double*       dw_b   = dw + off_dw;

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

    // 注意：按实际 dim_sum 迭代
    for (int k = 0; k < dim_sum; ++k) {
        int p;
        if      (k < o1) p = 0;
        else if (k < o2) p = 1;
        else if (k < o3) p = 2;
        else             p = 3;

        const double yv = y_b[k];                 // y[b,k]
        const double wv = w_b[(size_t)p * U + u]; // w[b,p,u]
        const double g  = go_bu[(size_t)k * U];   // grad_out[b,k,u]

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
    dx_b[u]          = dx_acc;
    dw_b[0 * U + u]  = dw_acc0;
    dw_b[1 * U + u]  = dw_acc1;
    dw_b[2 * U + u]  = dw_acc2;
    dw_b[3 * U + u]  = dw_acc3;
}

template <typename T>
static inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

// 单次 launch：B 拆到 (grid.y, grid.z)，兼容 B ≤ 65535*65535
static inline void launch_cwtp_bwd_once(
    const double* x, const double* y, const double* w, const double* grad_out,
    double* dx, double* dy, double* dw,
    int B, int U, int dim_sum, cudaStream_t stream)
{
    const int threads = 256;
    dim3 blockDim(threads, 1, 1);

    const unsigned int maxY = 65535u;
    const unsigned int gy   = (B <= (int)maxY) ? (unsigned)B : maxY;
    unsigned int gz         = ceil_div((unsigned)B, maxY);
    if (gz > 65535u) gz = 65535u;  // 极端保护，>时改走分批

    dim3 gridDim(ceil_div(U, threads), gy, gz);

    cwtp_kernel_bwd_bu_loopk<<<gridDim, blockDim, 0, stream>>>(
        x, y, w, grad_out, dx, dy, dw, B, U, dim_sum);
}

// 分批 launch：当 B > 65535*65535 或根据策略强制分批时使用
static inline void launch_cwtp_bwd_chunked(
    const double* x, const double* y, const double* w, const double* grad_out,
    double* dx, double* dy, double* dw,
    int B, int U, int dim_sum, cudaStream_t stream)
{
    const int threads = 256;
    dim3 blockDim(threads, 1, 1);
    const int maxY = 65535;

    for (int start = 0; start < B; ) {
        const int chunk = std::min(maxY, B - start);

        // 指针推进，kernel 内仍按 b 从 0..chunk-1 线性化（此处 z=1，y=chunk）
        const double* x_ptr   = x   + (size_t)start * U;
        const double* y_ptr   = y   + (size_t)start * dim_sum;
        const double* w_ptr   = w   + (size_t)start * (4 * U);
        const double* go_ptr  = grad_out + (size_t)start * (size_t)dim_sum * U;

        double* dx_ptr        = dx + (size_t)start * U;
        double* dy_ptr        = dy + (size_t)start * dim_sum;
        double* dw_ptr        = dw + (size_t)start * (4 * U);

        dim3 gridDim(ceil_div(U, threads), (unsigned)chunk, 1);

        cwtp_kernel_bwd_bu_loopk<<<gridDim, blockDim, 0, stream>>>(
            x_ptr, y_ptr, w_ptr, go_ptr, dx_ptr, dy_ptr, dw_ptr,
            /*B=*/chunk, U, dim_sum);

        start += chunk;
    }
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
    at::Tensor dx = at::empty({B, U}, opts).contiguous();        // 逐线程唯一写
    at::Tensor dy = at::zeros({B, dim_sum}, opts).contiguous();  // atomicAdd -> 置零
    at::Tensor dw = at::empty({B, P * U}, opts).contiguous();    // 逐线程唯一写

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    // B 不超过 65535*65535：一次 launch；否则分批
    const uint64_t B64 = static_cast<uint64_t>(B);
    const uint64_t LIM = 65535ull * 65535ull;
    if (B64 <= LIM) {
        launch_cwtp_bwd_once(
            x.data_ptr<double>(), y.data_ptr<double>(), w.data_ptr<double>(),
            grad_out.data_ptr<double>(),
            dx.data_ptr<double>(), dy.data_ptr<double>(), dw.data_ptr<double>(),
            B, U, dim_sum, stream);
    } else {
        launch_cwtp_bwd_chunked(
            x.data_ptr<double>(), y.data_ptr<double>(), w.data_ptr<double>(),
            grad_out.data_ptr<double>(),
            dx.data_ptr<double>(), dy.data_ptr<double>(), dw.data_ptr<double>(),
            B, U, dim_sum, stream);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dy, dw};
}

TORCH_LIBRARY(cwtp_bwd, m)
{
    m.def("backward", &cwtp_backward);
}
