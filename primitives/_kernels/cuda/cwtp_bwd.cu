#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int P  = 4;    // paths
constexpr int KS = 16;   // dim_sum = 16
constexpr int U_FIXED = 96; // U=96, 可被4整除

// 计算 2D 网格下的起始 b 与步长
__device__ __forceinline__ int grid_b0() {
    return blockIdx.x + blockIdx.y * gridDim.x;
}
__device__ __forceinline__ int grid_bstride() {
    return gridDim.x * gridDim.y;
}

// 支持 B>65535 的 2D-grid + grid-stride 版本
__global__ void bwd_kernel_vec4_warp32_grouped_db_bstride(
    const double* __restrict__ grad_out, // [B, KS, U]
    const double* __restrict__ x,        // [B, U]
    const double* __restrict__ y,        // [B, KS]
    const double* __restrict__ w,        // [B, P, U]
    const double* __restrict__ b_buf,    // [B, P, U]
    double* __restrict__ gx,             // [B, U]
    double* __restrict__ gy,             // [B, KS]
    double* __restrict__ gw,             // [B, P, U]
    int B, int U)
{
    const int lane  = threadIdx.x & 31;     // 单 warp
    const int slots = U / 4;                // 96/4=24
    if (lane >= slots) return;              // 多余 lane 直接退出

    // 2D grid + stride 遍历 b
    for (int b = grid_b0(); b < B; b += grid_bstride()) {

        // y[b,:] 放 shared（128B）
        __shared__ double y_sh[KS];
        if (lane < KS) y_sh[lane] = y[size_t(b)*KS + lane];
        __syncwarp();

        const int u = lane * 4;
        const size_t off_u     = size_t(u);
        const size_t off_bu    = size_t(b) * U + off_u;
        const size_t off_bk0   = size_t(b) * KS * U;

        // 只读一次 x4
        const double4 x4 = *reinterpret_cast<const double4*>(&x[off_bu]);

        // 必要累加器
        double4 gx4 = make_double4(0,0,0,0);
        double4 gw0 = make_double4(0,0,0,0);
        double4 gw1 = make_double4(0,0,0,0);
        double4 gw2 = make_double4(0,0,0,0);
        double4 gw3 = make_double4(0,0,0,0);

        // ---- path 0: k={0}（完全展开）----
        {
            const size_t off_pw = (size_t(b)*P + 0) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            const int k = 0;
            const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k)*U + off_u]);
            const double   yk = y_sh[k];

            // gy 现场 warp 归约并写回
            double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[size_t(b)*KS + k] = v;

            const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
            gw0.x += t0*x4.x; gw0.y += t1*x4.y; gw0.z += t2*x4.z; gw0.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
        }

        // ---- path 1: k={1,2,3}（短段：展开）----
        {
            const size_t off_pw = (size_t(b)*P + 1) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            #pragma unroll
            for (int k = 1; k <= 3; ++k) {
                const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k)*U + off_u]);
                const double   yk = y_sh[k];

                double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
                unsigned mask = 0xffffffffu;
                v += __shfl_down_sync(mask, v, 16);
                v += __shfl_down_sync(mask, v, 8);
                v += __shfl_down_sync(mask, v, 4);
                v += __shfl_down_sync(mask, v, 2);
                v += __shfl_down_sync(mask, v, 1);
                if (lane == 0) gy[size_t(b)*KS + k] = v;

                const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
                gw1.x += t0*x4.x; gw1.y += t1*x4.y; gw1.z += t2*x4.z; gw1.w += t3*x4.w;
                gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
            }
        }

        // ---- path 2: k={4..8}（长段：双缓冲 + 禁止展开）----
        {
            const size_t off_pw = (size_t(b)*P + 2) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            double4 g4_cur = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(4)*U + off_u]);
            double  yk_cur = y_sh[4];

            #pragma unroll 1
            for (int k = 4; k <= 8; ++k) {
                double4 g4_next; double yk_next = 0.0;
                if (k < 8) {
                    g4_next = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k+1)*U + off_u]);
                    yk_next = y_sh[k+1];
                }

                double v = g4_cur.x*bp.x + g4_cur.y*bp.y + g4_cur.z*bp.z + g4_cur.w*bp.w;
                unsigned mask = 0xffffffffu;
                v += __shfl_down_sync(mask, v, 16);
                v += __shfl_down_sync(mask, v, 8);
                v += __shfl_down_sync(mask, v, 4);
                v += __shfl_down_sync(mask, v, 2);
                v += __shfl_down_sync(mask, v, 1);
                if (lane == 0) gy[size_t(b)*KS + k] = v;

                const double t0 = g4_cur.x * yk_cur, t1 = g4_cur.y * yk_cur;
                const double t2 = g4_cur.z * yk_cur, t3 = g4_cur.w * yk_cur;
                gw2.x += t0*x4.x; gw2.y += t1*x4.y; gw2.z += t2*x4.z; gw2.w += t3*x4.w;
                gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;

                g4_cur = g4_next; yk_cur = yk_next;
            }
        }

        // ---- path 3: k={9..15}（长段：双缓冲 + 禁止展开）----
        {
            const size_t off_pw = (size_t(b)*P + 3) * U + off_u;
            const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
            const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

            double4 g4_cur = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(9)*U + off_u]);
            double  yk_cur = y_sh[9];

            #pragma unroll 1
            for (int k = 9; k <= 15; ++k) {
                double4 g4_next; double yk_next = 0.0;
                if (k < 15) {
                    g4_next = *reinterpret_cast<const double4*>(&grad_out[off_bk0 + size_t(k+1)*U + off_u]);
                    yk_next = y_sh[k+1];
                }

                double v = g4_cur.x*bp.x + g4_cur.y*bp.y + g4_cur.z*bp.z + g4_cur.w*bp.w;
                unsigned mask = 0xffffffffu;
                v += __shfl_down_sync(mask, v, 16);
                v += __shfl_down_sync(mask, v, 8);
                v += __shfl_down_sync(mask, v, 4);
                v += __shfl_down_sync(mask, v, 2);
                v += __shfl_down_sync(mask, v, 1);
                if (lane == 0) gy[size_t(b)*KS + k] = v;

                const double t0 = g4_cur.x * yk_cur, t1 = g4_cur.y * yk_cur;
                const double t2 = g4_cur.z * yk_cur, t3 = g4_cur.w * yk_cur;
                gw3.x += t0*x4.x; gw3.y += t1*x4.y; gw3.z += t2*x4.z; gw3.w += t3*x4.w;
                gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;

                g4_cur = g4_next; yk_cur = yk_next;
            }
        }

        // 向量写回
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 0)*U + off_u]) = gw0;
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 1)*U + off_u]) = gw1;
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 2)*U + off_u]) = gw2;
        *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 3)*U + off_u]) = gw3;
        *reinterpret_cast<double4*>(&gx[off_bu])                      = gx4;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cwtp_backward(
    const at::Tensor& grad_out, // [B, KS, U]
    const at::Tensor& x,        // [B, U]
    const at::Tensor& y,        // [B, KS]
    const at::Tensor& w,        // [B, P, U]
    const at::Tensor& b_buf     // [B, P, U]
){
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && y.is_cuda() && w.is_cuda() && b_buf.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(grad_out.scalar_type()==at::kDouble && x.scalar_type()==at::kDouble &&
                y.scalar_type()==at::kDouble && w.scalar_type()==at::kDouble && b_buf.scalar_type()==at::kDouble,
                "expect double dtype");
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() && y.is_contiguous()
                && w.is_contiguous() && b_buf.is_contiguous(), "expect contiguous");

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);

    TORCH_CHECK(U == U_FIXED, "U must be 96");
    TORCH_CHECK(grad_out.sizes() == at::IntArrayRef({B, KS * U}), "grad_out must be [B,16,96]");
    TORCH_CHECK(w.sizes()        == at::IntArrayRef({B, P * U}),   "w must be [B,4,96]");
    TORCH_CHECK(b_buf.sizes()    == at::IntArrayRef({B, P, U}),   "b_buf must be [B,4,96]");

    auto gx = at::empty_like(x);
    auto gy = at::empty_like(y);
    auto gw = at::empty_like(w);

    // 计算 2D 网格：grid.x <= 65535, grid.y <= 65535；多余用 stride 覆盖
    const int max_xy = 65535;
    int gx_dim = (B > max_xy) ? max_xy : static_cast<int>(B);
    int gy_dim = static_cast<int>((B + gx_dim - 1) / gx_dim);
    if (gy_dim > max_xy) gy_dim = max_xy; // 极大 B 时由 stride 继续覆盖

    dim3 grid(gx_dim, gy_dim); // 最多 ≈ 4.29e9 blocks 覆盖能力
    dim3 block(32);            // 单 warp

    bwd_kernel_vec4_warp32_grouped_db_bstride<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<double>(),
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        b_buf.data_ptr<double>(),
        gx.data_ptr<double>(),
        gy.data_ptr<double>(),
        gw.data_ptr<double>(),
        (int)B, (int)U
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, gy, gw};
}

TORCH_LIBRARY(cwtp_bwd, m)
{
    m.def("backward", &cwtp_backward);
}
