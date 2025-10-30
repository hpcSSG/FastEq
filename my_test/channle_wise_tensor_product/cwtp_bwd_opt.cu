#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

__constant__ int kDims[4] = {1, 3, 5, 7};
__constant__ int kOff[4]  = {0, 1, 4, 9}; // 前缀和

// 每个 block 处理：固定 b，处理 [u0, u0+Utile) 这段 u；聚合 grad_y 到共享后一次性原子写回
template<int Utile>
__global__ void cwtp_bu_tile_kernel(
    const double* __restrict__ x,       // [B, U]
    const double* __restrict__ y,       // [B, dim_sum]
    const double* __restrict__ w,       // [B, num_paths * U]
    const double* __restrict__ grad_out,// [B, U*dim_sum]，按 u*dim_sum + d 布局
    double* __restrict__ grad_x,        // [B, U]
    double* __restrict__ grad_y,        // [B, dim_sum]
    double* __restrict__ grad_w,        // [B, num_paths * U]
    int B, int U, int dim_sum, int num_paths)
{
    // 共享：grad_y 聚合（16），可选缓存 y[b,:]（16）
    __shared__ double s_grad_y[16];
    __shared__ double s_y[16];

    const int b  = blockIdx.y;
    const int u0 = blockIdx.x * Utile;

    // 保护
    if (b >= B) return;

    // 1) 初始化 s_grad_y
    for (int d = threadIdx.x; d < dim_sum; d += blockDim.x) {
        s_grad_y[d] = 0.0;
    }
    __syncthreads();

    // 2) 可选：把 y[b, :] 缓存到共享（提升复用）
    for (int d = threadIdx.x; d < dim_sum; d += blockDim.x) {
        s_y[d] = y[(size_t)b * dim_sum + d];
    }
    __syncthreads();

    // 3) 遍历本 tile 中的 u（每个线程步进处理多个 u）
    for (int tu = threadIdx.x; tu < Utile; tu += blockDim.x) {
        int u = u0 + tu;
        if (u >= U) continue;

        const size_t base_out = ((size_t)b * U + u) * dim_sum; // grad_out[b, u, d]
        const double xv = x[(size_t)b * U + u];

        // 预取 w[b, :, u]
        double wv0 = w[(size_t)b * (num_paths * U) + 0 * U + u];
        double wv1 = w[(size_t)b * (num_paths * U) + 1 * U + u];
        double wv2 = w[(size_t)b * (num_paths * U) + 2 * U + u];
        double wv3 = w[(size_t)b * (num_paths * U) + 3 * U + u];

        double gx_acc = 0.0;        // grad_x for (b,u)
        double gw_acc0 = 0.0;       // grad_w for (b,0,u)
        double gw_acc1 = 0.0;       // grad_w for (b,1,u)
        double gw_acc2 = 0.0;       // grad_w for (b,2,u)
        double gw_acc3 = 0.0;       // grad_w for (b,3,u)

        // 4) 遍历 d（dim_sum=16），用 kOff 映射到 path/i
        #pragma unroll
        for (int d = 0; d < 16; ++d) {
            if (d >= dim_sum) break;

            // 路径快速判定：kOff={0,1,4,9}
            int path = (d < 1) ? 0 : (d < 4) ? 1 : (d < 9) ? 2 : 3;
            int i    = d - kOff[path];
            (void)i; // 如果需要 i，可用

            const double g  = grad_out[base_out + d];
            const double yd = s_y[d];

            // grad_x 累加：sum_d g * y * w(path,u)
            // grad_w[path,u] 累加：sum_d g * x * y (仅该 path 的 d 参与)
            if (path == 0) {
                gx_acc  += g * yd * wv0;
                gw_acc0 += g * xv * yd;
                // grad_y 聚合：sum_u g * x * w(path,u)
                atomicAdd(&s_grad_y[d], g * xv * wv0);
            } else if (path == 1) {
                gx_acc  += g * yd * wv1;
                gw_acc1 += g * xv * yd;
                atomicAdd(&s_grad_y[d], g * xv * wv1);
            } else if (path == 2) {
                gx_acc  += g * yd * wv2;
                gw_acc2 += g * xv * yd;
                atomicAdd(&s_grad_y[d], g * xv * wv2);
            } else { // path == 3
                gx_acc  += g * yd * wv3;
                gw_acc3 += g * xv * yd;
                atomicAdd(&s_grad_y[d], g * xv * wv3);
            }
        }

        // 5) 无需原子：每个 (b,u) 唯一写回
        grad_x[(size_t)b * U + u] = gx_acc;
        grad_w[(size_t)b * (num_paths * U) + 0 * U + u] = gw_acc0;
        grad_w[(size_t)b * (num_paths * U) + 1 * U + u] = gw_acc1;
        grad_w[(size_t)b * (num_paths * U) + 2 * U + u] = gw_acc2;
        grad_w[(size_t)b * (num_paths * U) + 3 * U + u] = gw_acc3;
    }

    __syncthreads();

    // 6) 仅 16 次原子：把 s_grad_y 聚合写回
    for (int d = threadIdx.x; d < dim_sum; d += blockDim.x) {
        double v = s_grad_y[d];
        if (v != 0.0) {
            atomicAdd(&grad_y[(size_t)b * dim_sum + d], v);
        }
    }
}

// ========== launcher ==========
std::tuple<at::Tensor, at::Tensor, at::Tensor> cwtp_backward_opt(
    at::Tensor x,           // [B,U], double, cuda
    at::Tensor y,           // [B,dim_sum], double, cuda
    at::Tensor w,           // [B,num_paths*U], double, cuda
    at::Tensor grad_out)    // [B,U*dim_sum], double, cuda
{
    const int B         = x.size(0);
    const int U         = x.size(1);
    const int dim_sum   = y.size(1);
    const int num_paths = 4;   // 固定 4 条路径
    TORCH_CHECK(dim_sum == 16, "This optimized kernel assumes dim_sum == 16");

    auto opts = x.options();
    at::Tensor dx = at::empty({B, U}, opts).contiguous();
    at::Tensor dy = at::zeros({B, dim_sum}, opts).contiguous(); // 需要原子累加
    at::Tensor dw = at::empty({B, num_paths * U}, opts).contiguous();

    // 网格/线程配置
    constexpr int Utile = 64;          // 每个 block 处理 64 个 u（可调：32/64/96）
    const int grid_x    = (U + Utile - 1) / Utile;
    dim3 grid(grid_x, B, 1);
    dim3 block(128, 1, 1);             // 128 线程（吞吐/延迟折中）
    // 共享内存只用到了 16*2 doubles（s_grad_y + s_y） -> 256 bytes，非常小

    auto stream = at::cuda::getCurrentCUDAStream();
    cwtp_bu_tile_kernel<Utile><<<grid, block, 0, stream>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        grad_out.data_ptr<double>(),
        dx.data_ptr<double>(),
        dy.data_ptr<double>(),
        dw.data_ptr<double>(),
        B, U, dim_sum, num_paths);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dy, dw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &cwtp_backward_opt, "ChannelWise TP Backward (Optimized)");
}
