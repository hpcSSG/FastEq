#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>


constexpr int P  = 4;    // paths
constexpr int KS = 16;   // dim_sum = 16
constexpr int U_FIXED = 96; // U=96, 可被4整除

// 2D grid helpers
__device__ __forceinline__ int grid_b0() {
    return blockIdx.x + blockIdx.y * gridDim.x;
}
__device__ __forceinline__ int grid_bstride() {
    return gridDim.x * gridDim.y;
}

// 前向：warp32（blockDim.x=32），沿 U 维 double4 向量化，按 path 分段写出
// x:[B,U], y:[B,16], w:[B,4,U] -> out:[B,16,U], b_buf:[B,4,U]
__global__ void fwd_kernel_vec4_warp32_grouped_bstride(
    const double* __restrict__ x,     // [B,U]
    const double* __restrict__ y,     // [B,16]
    const double* __restrict__ w,     // [B,4,U]
    double* __restrict__ out,         // [B,16,U]
    double* __restrict__ b_buf,       // [B,4,U]
    int B, int U)
{
    const int lane  = threadIdx.x & 31;    // 单 warp
    const int slots = U / 4;               // 96/4=24
    if (lane >= slots) return;             // 其余 8 lane 直接退出

    for (int b = grid_b0(); b < B; b += grid_bstride()) {

        // y[b,:] 放 shared（128B）
        __shared__ double y_sh[KS];
        if (lane < KS) y_sh[lane] = y[size_t(b)*KS + lane];
        __syncwarp();

        const int u = lane * 4;
        const size_t off_u   = size_t(u);
        const size_t off_bu  = size_t(b) * U + off_u;
        const size_t off_bk0 = size_t(b) * KS * U;

        // 读取 x4，一次即可
        const double4 x4 = *reinterpret_cast<const double4*>(&x[off_bu]);

        // 预读四条 w_p4，计算 base_p4 = x4 * w_p4，并写入 b_buf
        const size_t off_w0 = (size_t(b)*P + 0) * U + off_u;
        const size_t off_w1 = (size_t(b)*P + 1) * U + off_u;
        const size_t off_w2 = (size_t(b)*P + 2) * U + off_u;
        const size_t off_w3 = (size_t(b)*P + 3) * U + off_u;

        const double4 w0 = *reinterpret_cast<const double4*>(&w[off_w0]);
        const double4 w1 = *reinterpret_cast<const double4*>(&w[off_w1]);
        const double4 w2 = *reinterpret_cast<const double4*>(&w[off_w2]);
        const double4 w3 = *reinterpret_cast<const double4*>(&w[off_w3]);

        double4 b0; b0.x = x4.x*w0.x; b0.y = x4.y*w0.y; b0.z = x4.z*w0.z; b0.w = x4.w*w0.w;
        double4 b1; b1.x = x4.x*w1.x; b1.y = x4.y*w1.y; b1.z = x4.z*w1.z; b1.w = x4.w*w1.w;
        double4 b2; b2.x = x4.x*w2.x; b2.y = x4.y*w2.y; b2.z = x4.z*w2.z; b2.w = x4.w*w2.w;
        double4 b3; b3.x = x4.x*w3.x; b3.y = x4.y*w3.y; b3.z = x4.z*w3.z; b3.w = x4.w*w3.w;

        *reinterpret_cast<double4*>(&b_buf[off_w0]) = b0;
        *reinterpret_cast<double4*>(&b_buf[off_w1]) = b1;
        *reinterpret_cast<double4*>(&b_buf[off_w2]) = b2;
        *reinterpret_cast<double4*>(&b_buf[off_w3]) = b3;

        // 写 out：按 path 的 k 段分组，out[b,k,u:u+3] = base_p4 * y[b,k]
        // path 0: k={0}
        {
            const int k = 0;
            const double yk = y_sh[k];
            double4 o; o.x = b0.x*yk; o.y = b0.y*yk; o.z = b0.z*yk; o.w = b0.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
        // path 1: k={1,2,3}
        #pragma unroll
        for (int k = 1; k <= 3; ++k) {
            const double yk = y_sh[k];
            double4 o; o.x = b1.x*yk; o.y = b1.y*yk; o.z = b1.z*yk; o.w = b1.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
        // path 2: k={4,5,6,7,8}
        #pragma unroll
        for (int k = 4; k <= 8; ++k) {
            const double yk = y_sh[k];
            double4 o; o.x = b2.x*yk; o.y = b2.y*yk; o.z = b2.z*yk; o.w = b2.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
        // path 3: k={9..15}
        #pragma unroll
        for (int k = 9; k <= 15; ++k) {
            const double yk = y_sh[k];
            double4 o; o.x = b3.x*yk; o.y = b3.y*yk; o.z = b3.z*yk; o.w = b3.w*yk;
            *reinterpret_cast<double4*>(&out[off_bk0 + size_t(k)*U + off_u]) = o;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> cwtp_forward(
    const at::Tensor& x,     // [B,U]
    const at::Tensor& y,     // [B,16]
    const at::Tensor& w      // [B,4,U]
){
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type()==at::kDouble && y.scalar_type()==at::kDouble && w.scalar_type()==at::kDouble, "expect double dtype");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && w.is_contiguous(), "expect contiguous");

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);

    TORCH_CHECK(U == U_FIXED, "U must be 96");
    TORCH_CHECK(y.sizes() == at::IntArrayRef({B, KS}), "y must be [B,16]");
    TORCH_CHECK(w.sizes() == at::IntArrayRef({B, P * U}), "w must be [B,4,96]");

    auto out   = at::empty({B, KS, U}, x.options());
    auto b_buf = at::empty({B, P,  U}, x.options());

    // 2D grid 自动适配 B>65535
    const int max_xy = 65535;
    int gx_dim = (B > max_xy) ? max_xy : static_cast<int>(B);
    int gy_dim = static_cast<int>((B + gx_dim - 1) / gx_dim);
    if (gy_dim > max_xy) gy_dim = max_xy;

    dim3 grid(gx_dim, gy_dim); // 覆盖任意大 B；剩余用 stride
    dim3 block(32);            // 单 warp

    fwd_kernel_vec4_warp32_grouped_bstride<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        b_buf.data_ptr<double>(),
        (int)B, (int)U
    );
    out = out.reshape({B, KS * U});
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, b_buf}; // out:[B,16*96], b_buf:[B,4,96]
}

TORCH_LIBRARY(cwtp_fwd, m)
{
    m.def("forward", &cwtp_forward);
}
