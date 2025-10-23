#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

__constant__ int kOff[4]  = {0, 1, 4, 9};   // 前缀和：0,1,4,9；sum=16

// 线性化 b：b = blockIdx.y + blockIdx.z * gridDim.y
__global__ void cwtp_kernel_bu_loopk(
    const double* __restrict__ x,   // [B, U]
    const double* __restrict__ y,   // [B, dim_sum]
    const double* __restrict__ w,   // [B, 4*U]  (P==4, V==1)
    double* __restrict__ out,       // [B, dim_sum, U]
    int B, int U, int dim_sum)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y + blockIdx.z * gridDim.y;  // 线性化 batch
    if (b >= (unsigned)B || u >= U) return;

    // 读一次 x[b,u]
    const size_t off_x  = (size_t)b * U + u;
    const double xv     = x[off_x];

    // 便捷指针
    const double* __restrict__ y_b   = y   + (size_t)b * dim_sum;           // y[b,0]
    double*       __restrict__ out_b = out + (size_t)b * dim_sum * U + u;   // out[b,0,u]

    // 阈值（根据 kOff）
    const int o1 = kOff[1];  // 1
    const int o2 = kOff[2];  // 4
    const int o3 = kOff[3];  // 9

    // 按 k 循环写出；与 dim_sum 解耦
    // 如果你的 dim_sum 常为 16，编译器仍会很好地展开这个小循环
    for (int k = 0; k < dim_sum; ++k) {
        int p;
        if      (k < o1) p = 0;
        else if (k < o2) p = 1;
        else if (k < o3) p = 2;
        else             p = 3;

        // 读 w[b,p,u]；按 u 连续访问，合并读
        const double wv = w[(size_t)b * (4 * U) + (size_t)p * U + u];
        const double yv = y_b[k];

        // out[b,k,u] = x[b,u] * w[b,p,u] * y[b,k]
        out_b[(size_t)k * U] = xv * wv * yv;
    }
}

template <typename T>
static inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

static inline void launch_cwtp_kernel_once(
    const double* x, const double* y, const double* w, double* out,
    int B, int U, int dim_sum, cudaStream_t stream)
{
    const int threads = 256;
    dim3 blockDim(threads, 1, 1);

    // grid.x 覆盖 U；grid.y / grid.z 线性化覆盖 B（兼容 B > 65535）
    const unsigned int maxY = 65535u;
    const unsigned int gy   = (B <= (int)maxY) ? (unsigned)B : maxY;
    unsigned int gz         = ceil_div((unsigned)B, maxY);

    // 保护：极端超大 B 时（> 65535*65535）避免非法配置（外层会分批处理）
    if (gz > 65535u) gz = 65535u;

    dim3 gridDim(ceil_div(U, threads), gy, gz);

    cwtp_kernel_bu_loopk<<<gridDim, blockDim, 0, stream>>>(
        x, y, w, out, B, U, dim_sum);
}

// 分批 launch：当 B 极端巨大时使用（> 65535*65535）
static inline void launch_cwtp_kernel_chunked(
    const double* x, const double* y, const double* w, double* out,
    int B, int U, int dim_sum, cudaStream_t stream)
{
    const int threads = 256;
    dim3 blockDim(threads, 1, 1);
    const int maxY = 65535;

    for (int start = 0; start < B; ) {
        // 本批大小（用 grid.y；grid.z=1）
        const int chunk = std::min(maxY, B - start);

        // 我们在 kernel 内仍用线性化 b，但这里选择 z=1，y=chunk 更简单
        dim3 gridDim(ceil_div(U, threads), (unsigned)chunk, 1);

        // 偏移量通过把指针推进来实现（避免在 kernel 增加参数）
        const double* x_ptr   = x   + (size_t)start * U;
        const double* y_ptr   = y   + (size_t)start * dim_sum;
        const double* w_ptr   = w   + (size_t)start * (4 * U);
        double*       out_ptr = out + (size_t)start * (size_t)dim_sum * U;

        cwtp_kernel_bu_loopk<<<gridDim, blockDim, 0, stream>>>(
            x_ptr, y_ptr, w_ptr, out_ptr, /*B=*/chunk, U, dim_sum);

        start += chunk;
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

    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    // 若 B 不超过 65535*65535：一次 launch 覆盖
    const uint64_t B64 = static_cast<uint64_t>(B);
    const uint64_t LIM = 65535ull * 65535ull;
    if (B64 <= LIM) {
        launch_cwtp_kernel_once(
            x.data_ptr<double>(), y.data_ptr<double>(), w.data_ptr<double>(),
            out.data_ptr<double>(), B, U, dim_sum, cur_stream);
    } else {
        // 极端大 B：分批
        launch_cwtp_kernel_chunked(
            x.data_ptr<double>(), y.data_ptr<double>(), w.data_ptr<double>(),
            out.data_ptr<double>(), B, U, dim_sum, cur_stream);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 如需原先的扁平输出
    return out.reshape({B, dim_sum * U});
}

TORCH_LIBRARY(cwtp_fwd, m)
{
    m.def("forward", &cwtp_forward);
}
