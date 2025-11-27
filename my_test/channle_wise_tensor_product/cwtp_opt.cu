// einsum_simplified_v1.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int P       = 4;   // paths = 4  (dim_list=[1,3,5,7])
constexpr int DIM_SUM = 16;  // 1+3+5+7
constexpr int KS = DIM_SUM;  // KS = DIM_SUM
constexpr int U_FIXED = 96;  // U=96 (按你的场景固化)
                               // V=1 已在实现中固化

// 段长度与前缀和常量
__constant__ int kDims_c[P] = {1, 3, 5, 7};
__constant__ int kOffs_c[P] = {0, 1, 4, 9};

inline __device__ int idx_x(int b, int u, int B, int U) {
    return b * U + u;
}
inline __device__ int idx_y(int b, int k, int B, int KS) {
    return b * KS + k; // KS = DIM_SUM
}
inline __device__ int idx_w(int b, int p, int u, int B, int P_, int U) {
    return (b * P_ + p) * U + u;
}
inline __device__ int idx_out(int b, int k, int u, int B, int KS, int U) {
    // layout [B, KS, U]
    return (b * KS + k) * U + u;
}

inline __device__ int idx_bku(int b, int k, int u, int B, int KS, int U) {
    // [B, KS, U]
    return (b * KS + k) * U + u;
}

// ---------------------------------------------
// 前向：out[b,k,u] = (x[b,u]*w[b,p,u]) * y[b,k]
// 额外写出 b_buf[b,p,u] = x[b,u]*w[b,p,u]
// gridDim.x = B, blockDim.x >= 96
// ---------------------------------------------
__global__ void fwd_kernel_v1(
    const double* __restrict__ x,      // [B, U]
    const double* __restrict__ y,      // [B, 16]
    const double* __restrict__ w,      // [B, 4, U]
    double* __restrict__ out,          // [B, 16, U]
    double* __restrict__ b_buf,        // [B, 4, U]  新增缓存
    int B, int U
){
    const int b  = blockIdx.x;
    if (b >= B) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int u = tid; u < U; u += nthreads) {
        const double xbu = x[idx_x(b, u, B, U)];

        #pragma unroll
        for (int p = 0; p < P; ++p) {
            const int d   = kDims_c[p];
            const int off = kOffs_c[p];
            const double wpu  = w[idx_w(b, p, u, B, P, U)];
            const double base = xbu * wpu;

            // 写 b_buf 以便后向复用
            b_buf[idx_w(b, p, u, B, P, U)] = base;

            // 段内展开写 out
            #pragma unroll
            for (int kk = 0; kk < 7; ++kk) {
                if (kk < d) {
                    const int k = off + kk;
                    const double ybk = y[idx_y(b, k, B, DIM_SUM)];
                    out[idx_out(b, k, u, B, DIM_SUM, U)] = base * ybk;
                }
            }
        }
    }
}

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

// ---------------------------------------------
// 后向：给定 grad_out:[B,16,U]，求
// gx:[B,U], gy:[B,16], gw:[B,4,U]
// 使用 b_buf 来计算 gy，减少后向访存
// ---------------------------------------------
__global__ void bwd_kernel_v1(
    const double* __restrict__ grad_out, // [B, 16, U]
    const double* __restrict__ x,        // [B, U]
    const double* __restrict__ y,        // [B, 16]
    const double* __restrict__ w,        // [B, 4, U]
    const double* __restrict__ b_buf,    // [B, 4, U]  前向缓存
    double* __restrict__ gx,             // [B, U]
    double* __restrict__ gy,             // [B, 16]
    double* __restrict__ gw,             // [B, 4, U]
    int B, int U
){
    const int b  = blockIdx.x;
    if (b >= B) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // 每线程寄存器局部累加 gy 的 16 路通道
    double gy_local[DIM_SUM];
    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) gy_local[k] = 0.0;

    for (int u = tid; u < U; u += nthreads) {
        const double xbu = x[idx_x(b, u, B, U)];

        // 每个 path 的 gb 累加器（gb_p = sum_k G* y）
        double gb_p[P];
        #pragma unroll
        for (int p = 0; p < P; ++p) gb_p[p] = 0.0;

        // 使用 b_buf 直接计算对 gy 的贡献；同时计算 gb_p
        #pragma unroll
        for (int p = 0; p < P; ++p) {
            const int d   = kDims_c[p];
            const int off = kOffs_c[p];

            const double bpu = b_buf[idx_w(b, p, u, B, P, U)]; // 读取缓存

            #pragma unroll
            for (int kk = 0; kk < 7; ++kk) {
                if (kk < d) {
                    const int k = off + kk;
                    const double G   = grad_out[idx_out(b, k, u, B, DIM_SUM, U)];
                    const double ybk = y[idx_y(b, k, B, DIM_SUM)];
                    gy_local[k] += G * bpu;   // gy 累加使用 b_buf
                    gb_p[p]     += G * ybk;   // gb_p 用于 gx/gw
                }
            }
        }

        // 写 gw、gx（逐 u 写，无需原子）
        double gx_val = 0.0;
        #pragma unroll
        for (int p = 0; p < P; ++p) {
            const double wpu = w[idx_w(b, p, u, B, P, U)];
            gx_val += gb_p[p] * wpu;                 // gx += gb_p * w_p
            gw[idx_w(b, p, u, B, P, U)] = gb_p[p] * xbu;  // gw = gb_p * x
        }
        gx[idx_x(b, u, B, U)] = gx_val;
    }

    // ---------- 归约 gy_local 到 gy[b,:] ----------
    extern __shared__ double smem[];
    double* sm_row = smem + tid * DIM_SUM;

    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) sm_row[k] = gy_local[k];
    __syncthreads();

    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double* sm_next = sm_row + stride * DIM_SUM;
            #pragma unroll
            for (int k = 0; k < DIM_SUM; ++k) {
                sm_row[k] += sm_next[k];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int k = 0; k < DIM_SUM; ++k) {
            gy[idx_y(b, k, B, DIM_SUM)] = sm_row[k];
        }
    }
}

// === helper: warp reduce for double ===
__inline__ __device__ double warp_reduce_sum(double v) {
    unsigned mask = 0xffffffffu;
    // 32->1 归约
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// ===========================================================
// 后向（转置版）：读取 grad_out_T:[B, U, 16]（连续的 16 个 double）
// - 输入：grad_out_T, x:[B,U], y:[B,16], w:[B,4,U], b_buf:[B,4,U]
// - 输出：gx:[B,U], gy:[B,16], gw:[B,4,U]
// 设计：每 block 处理一个 b；每线程处理 2 个 u（u0=tid, u1=tid+blockDim.x）
// 关键：一次把 y[b,:] 拉到寄存器；grad_out_T 用 double4 向量化加载；
// gy 采用 warp reduce + 轻量共享内存跨 warp 汇总。
// ===========================================================
__global__ void bwd_kernel_v2(
    const double* __restrict__ grad_out_T, // [B, U, 16]  <- 重要：已在 Python 端转置并 contiguous
    const double* __restrict__ x,          // [B, U]
    const double* __restrict__ y,          // [B, 16]
    const double* __restrict__ w,          // [B, 4, U]
    const double* __restrict__ b_buf,      // [B, 4, U]
    double* __restrict__ gx,               // [B, U]
    double* __restrict__ gy,               // [B, 16]
    double* __restrict__ gw,               // [B, 4, U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warp_id  = tid >> 5;   // 0..(nthreads/32-1)
    const int lane     = tid & 31;

    // 1) y[b,:] → yreg
    double yreg[DIM_SUM];
    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) {
        yreg[k] = y[b*DIM_SUM + k];
    }

    // 每线程的 16 路 gy 局部累加
    double gy_local[DIM_SUM];
    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) gy_local[k] = 0.0;

    // 每线程处理两个 u
    int u0 = tid;
    int u1 = tid + nthreads;

    auto process_u = [&](int u) {
        if (u >= U) return;

        const double xbu = x[b*U + u];

        // b_buf 四条（p=0..3），用于 gy
        const double b0 = b_buf[((b*P + 0)*U) + u];
        const double b1 = b_buf[((b*P + 1)*U) + u];
        const double b2 = b_buf[((b*P + 2)*U) + u];
        const double b3 = b_buf[((b*P + 3)*U) + u];

        // 连续加载 grad_out_T[b,u,0..15] 共 16 doubles = 128B
        // 依赖 PyTorch cuda allocator 的对齐（>=256B），以及偏移 (b*U+u)*16*8B 是 128B 的倍数 → 16B 对齐满足 double4
        const double4* gptr4 = reinterpret_cast<const double4*>(
            &grad_out_T[( (b*U + u) * DIM_SUM ) + 0]
        );
        double4 g4_0 = gptr4[0]; // k=0..3
        double4 g4_1 = gptr4[1]; // k=4..7
        double4 g4_2 = gptr4[2]; // k=8..11
        double4 g4_3 = gptr4[3]; // k=12..15

        // 分段 gb_p
        double gb_p0 = 0.0, gb_p1 = 0.0, gb_p2 = 0.0, gb_p3 = 0.0;

        // --- k=0..3： 0→p0；1..3→p1 ---
        {
            const double g0 = g4_0.x, g1 = g4_0.y, g2 = g4_0.z, g3 = g4_0.w;
            gy_local[0] += g0 * b0;
            gy_local[1] += g1 * b1; gy_local[2] += g2 * b1; gy_local[3] += g3 * b1;
            gb_p0 += g0 * yreg[0];
            gb_p1 += g1 * yreg[1] + g2 * yreg[2] + g3 * yreg[3];
        }
        // --- k=4..7：全属 p2 ---
        {
            const double g4 = g4_1.x, g5 = g4_1.y, g6 = g4_1.z, g7 = g4_1.w;
            gy_local[4] += g4 * b2; gy_local[5] += g5 * b2; gy_local[6] += g6 * b2; gy_local[7] += g7 * b2;
            gb_p2 += g4 * yreg[4] + g5 * yreg[5] + g6 * yreg[6] + g7 * yreg[7];
        }
        // --- k=8..11：8→p2 的第5个；9..11→p3 ---
        {
            const double g8  = g4_2.x, g9  = g4_2.y, g10 = g4_2.z, g11 = g4_2.w;
            gy_local[8]  += g8  * b2;
            gy_local[9]  += g9  * b3; gy_local[10] += g10 * b3; gy_local[11] += g11 * b3;
            gb_p2 += g8  * yreg[8];
            gb_p3 += g9  * yreg[9] + g10 * yreg[10] + g11 * yreg[11];
        }
        // --- k=12..15：全属 p3 ---
        {
            const double g12 = g4_3.x, g13 = g4_3.y, g14 = g4_3.z, g15 = g4_3.w;
            gy_local[12] += g12 * b3; gy_local[13] += g13 * b3;
            gy_local[14] += g14 * b3; gy_local[15] += g15 * b3;
            gb_p3 += g12 * yreg[12] + g13 * yreg[13] + g14 * yreg[14] + g15 * yreg[15];
        }

        // 写 gw、gx（逐 u，无需原子）
        const double w0 = w[((b*P + 0)*U) + u];
        const double w1 = w[((b*P + 1)*U) + u];
        const double w2 = w[((b*P + 2)*U) + u];
        const double w3 = w[((b*P + 3)*U) + u];

        gw[((b*P + 0)*U) + u] = gb_p0 * xbu;
        gw[((b*P + 1)*U) + u] = gb_p1 * xbu;
        gw[((b*P + 2)*U) + u] = gb_p2 * xbu;
        gw[((b*P + 3)*U) + u] = gb_p3 * xbu;

        gx[b*U + u] = gb_p0 * w0 + gb_p1 * w1 + gb_p2 * w2 + gb_p3 * w3;
    };

    process_u(u0);
    process_u(u1);

    // 2) warp 级归约 + 轻量共享内存跨 warp 汇总
    extern __shared__ double smem[]; // 大小 = (nthreads/32) * 16 * sizeof(double)
    double* sm_warp = smem + warp_id * DIM_SUM;

    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) {
        double v = warp_reduce_sum(gy_local[k]);
        if (lane == 0) sm_warp[k] = v;
    }
    __syncthreads();

    // 仅 warp 0 汇总所有 warp 的 16 通道并写回 gy[b,:]
    if (warp_id == 0) {
        #pragma unroll
        for (int k = 0; k < DIM_SUM; ++k) {
            double s = 0.0;
            // 多 warp 汇总：把每个 warp 的 lane0 写入的值，分布式累加到本 warp 的 32 个 lane，再做一次 warp reduce
            for (int w = lane; w < (nthreads >> 5); w += 32) {
                s += smem[w * DIM_SUM + k];
            }
            s = warp_reduce_sum(s);
            if (lane == 0) {
                gy[b*DIM_SUM + k] = s;
            }
        }
    }
}

// ======================================================
// 后向 (double4 向量化版) — grad_out:[B,16,U], U=96. cost: 1.9ms
// ======================================================
__global__ void bwd_kernel_vec4_slow(
    const double* __restrict__ grad_out, // [B,16,U]
    const double* __restrict__ x,        // [B,U]
    const double* __restrict__ y,        // [B,16]
    const double* __restrict__ w,        // [B,4,U]
    const double* __restrict__ b_buf,    // [B,4,U]
    double* __restrict__ gx,             // [B,U]
    double* __restrict__ gy,             // [B,16]
    double* __restrict__ gw,             // [B,4,U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;

    // -------- y[b,:] 缓存到寄存器 --------
    double yreg[DIM_SUM];
    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k)
        yreg[k] = y[b*DIM_SUM + k];

    // 线程局部 gy 累加
    double gy_local[DIM_SUM];
    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) gy_local[k] = 0.0;

    // 每线程处理 4 个连续 u： u = tid*4 + step*(blockDim.x*4)
    const int u_stride = nthreads * 4;
    for (int u = tid * 4; u < U; u += u_stride) {
        // ---- 向量加载 x ----
        const double4* x4_ptr = reinterpret_cast<const double4*>(&x[b*U + u]);
        double4 x4 = *x4_ptr;

        // 向量化 load w/b_buf
        double4 w0 = *reinterpret_cast<const double4*>(&w[((b*P + 0)*U) + u]);
        double4 w1 = *reinterpret_cast<const double4*>(&w[((b*P + 1)*U) + u]);
        double4 w2 = *reinterpret_cast<const double4*>(&w[((b*P + 2)*U) + u]);
        double4 w3 = *reinterpret_cast<const double4*>(&w[((b*P + 3)*U) + u]);

        double4 b0 = *reinterpret_cast<const double4*>(&b_buf[((b*P + 0)*U) + u]);
        double4 b1 = *reinterpret_cast<const double4*>(&b_buf[((b*P + 1)*U) + u]);
        double4 b2 = *reinterpret_cast<const double4*>(&b_buf[((b*P + 2)*U) + u]);
        double4 b3 = *reinterpret_cast<const double4*>(&b_buf[((b*P + 3)*U) + u]);

        // 向量 gb_p 四组 (4 lanes × 4 paths)
        double4 gb0 = make_double4(0.0,0.0,0.0,0.0);
        double4 gb1 = gb0, gb2 = gb0, gb3 = gb0;

        // ---- 循环 k ----
        #pragma unroll
        for (int k = 0; k < DIM_SUM; ++k) {
            double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*DIM_SUM + k)*U + u]);
            double yk = yreg[k];

            int p;
            if      (k == 0)         p = 0;
            else if (k <= 3)         p = 1;
            else if (k <= 8)         p = 2;
            else                     p = 3;

            // 选择对应的 b4
            double4 bp;
            if      (p==0) bp = b0;
            else if (p==1) bp = b1;
            else if (p==2) bp = b2;
            else           bp = b3;

            // gy_local[k] += sum(g4 * bp)
            gy_local[k] += g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;

            // gb_p += g4 * yk
            double4 tmp = make_double4(g4.x*yk, g4.y*yk, g4.z*yk, g4.w*yk);
            if      (p==0) { gb0.x+=tmp.x; gb0.y+=tmp.y; gb0.z+=tmp.z; gb0.w+=tmp.w; }
            else if (p==1) { gb1.x+=tmp.x; gb1.y+=tmp.y; gb1.z+=tmp.z; gb1.w+=tmp.w; }
            else if (p==2) { gb2.x+=tmp.x; gb2.y+=tmp.y; gb2.z+=tmp.z; gb2.w+=tmp.w; }
            else           { gb3.x+=tmp.x; gb3.y+=tmp.y; gb3.z+=tmp.z; gb3.w+=tmp.w; }
        }

        // ---- 写 gw ----
        double4 gw0 = make_double4(gb0.x*x4.x, gb0.y*x4.y, gb0.z*x4.z, gb0.w*x4.w);
        double4 gw1 = make_double4(gb1.x*x4.x, gb1.y*x4.y, gb1.z*x4.z, gb1.w*x4.w);
        double4 gw2 = make_double4(gb2.x*x4.x, gb2.y*x4.y, gb2.z*x4.z, gb2.w*x4.w);
        double4 gw3 = make_double4(gb3.x*x4.x, gb3.y*x4.y, gb3.z*x4.z, gb3.w*x4.w);
        *reinterpret_cast<double4*>(&gw[((b*P + 0)*U)+u]) = gw0;
        *reinterpret_cast<double4*>(&gw[((b*P + 1)*U)+u]) = gw1;
        *reinterpret_cast<double4*>(&gw[((b*P + 2)*U)+u]) = gw2;
        *reinterpret_cast<double4*>(&gw[((b*P + 3)*U)+u]) = gw3;

        // ---- 写 gx ----
        double4 gx4;
        gx4.x = gb0.x*w0.x + gb1.x*w1.x + gb2.x*w2.x + gb3.x*w3.x;
        gx4.y = gb0.y*w0.y + gb1.y*w1.y + gb2.y*w2.y + gb3.y*w3.y;
        gx4.z = gb0.z*w0.z + gb1.z*w1.z + gb2.z*w2.z + gb3.z*w3.z;
        gx4.w = gb0.w*w0.w + gb1.w*w1.w + gb2.w*w2.w + gb3.w*w3.w;
        *reinterpret_cast<double4*>(&gx[b*U + u]) = gx4;
    }

    // -------- warp reduce -> SMEM -> gy[b,:] 汇总 --------
    extern __shared__ double smem[];
    double* sm_warp = smem + warp_id * DIM_SUM;

    #pragma unroll
    for (int k = 0; k < DIM_SUM; ++k) {
        double v = warp_reduce_sum(gy_local[k]);
        if (lane == 0) sm_warp[k] = v;
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int k = 0; k < DIM_SUM; ++k) {
            double s = 0.0;
            for (int w_ = lane; w_ < (nthreads >> 5); w_ += 32)
                s += smem[w_ * DIM_SUM + k];
            s = warp_reduce_sum(s);
            if (lane == 0) gy[b*DIM_SUM + k] = s;
        }
    }
}

// 单warp（blockDim.x=32）版：U=96, vec4，低寄存器 & 无跨warp归约
__launch_bounds__(32, 8)
__global__ void bwd_kernel_vec4_warp32(
    const double* __restrict__ grad_out, // [B,16,U]
    const double* __restrict__ x,        // [B,U]
    const double* __restrict__ y,        // [B,16]
    const double* __restrict__ w,        // [B,4,U]
    const double* __restrict__ b_buf,    // [B,4,U]
    double* __restrict__ gx,             // [B,U]
    double* __restrict__ gy,             // [B,16]
    double* __restrict__ gw,             // [B,4,U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int lane = threadIdx.x & 31;            // 0..31
    const int slots = U / 4;                      // 96/4=24
    if (lane >= slots) return;                    // 多余的 8 lane 直接退出

    // ---- y[b,:] -> shared（16 个 double）----
    __shared__ double y_sh[16];
    if (lane < 16) y_sh[lane] = y[b*16 + lane];
    __syncwarp();

    // ---- 本线程负责的 u 起点 ----
    const int u = lane * 4;

    // 预取 x, w, b_buf（每个只读一次）
    const double4 x4 = *reinterpret_cast<const double4*>(&x[b*U + u]);
    const double4 w0 = *reinterpret_cast<const double4*>(&w[((b*4 + 0)*U) + u]);
    const double4 w1 = *reinterpret_cast<const double4*>(&w[((b*4 + 1)*U) + u]);
    const double4 w2 = *reinterpret_cast<const double4*>(&w[((b*4 + 2)*U) + u]);
    const double4 w3 = *reinterpret_cast<const double4*>(&w[((b*4 + 3)*U) + u]);
    const double4 b0 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 0)*U) + u]);
    const double4 b1 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 1)*U) + u]);
    const double4 b2 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 2)*U) + u]);
    const double4 b3 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 3)*U) + u]);

    // 直接累计到 gx/gw；gy 分别累加到寄存器后做 warp 内归约
    double4 gx4 = make_double4(0.0,0.0,0.0,0.0);
    double4 gw0 = make_double4(0.0,0.0,0.0,0.0);
    double4 gw1 = make_double4(0.0,0.0,0.0,0.0);
    double4 gw2 = make_double4(0.0,0.0,0.0,0.0);
    double4 gw3 = make_double4(0.0,0.0,0.0,0.0);

    double gy_acc[16];
    #pragma unroll 1
    for (int k = 0; k < 16; ++k) gy_acc[k] = 0.0;

    // ---- 遍历 k=0..15，按段选择 path，直接即时累计 ----
    #pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
        const double yk = y_sh[k];

        int p = (k==0) ? 0 : (k<=3) ? 1 : (k<=8) ? 2 : 3;
        const double4 bp = (p==0)? b0 : (p==1)? b1 : (p==2)? b2 : b3;
        const double4 wp = (p==0)? w0 : (p==1)? w1 : (p==2)? w2 : w3;

        // gy 累加：dot(g4, bp)
        gy_acc[k] += g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;

        // tmp = g4 * yk
        const double4 tmp = make_double4(g4.x*yk, g4.y*yk, g4.z*yk, g4.w*yk);

        // gw_p += tmp * x4
        if      (p==0) { gw0.x+=tmp.x*x4.x; gw0.y+=tmp.y*x4.y; gw0.z+=tmp.z*x4.z; gw0.w+=tmp.w*x4.w; }
        else if (p==1) { gw1.x+=tmp.x*x4.x; gw1.y+=tmp.y*x4.y; gw1.z+=tmp.z*x4.z; gw1.w+=tmp.w*x4.w; }
        else if (p==2) { gw2.x+=tmp.x*x4.x; gw2.y+=tmp.y*x4.y; gw2.z+=tmp.z*x4.z; gw2.w+=tmp.w*x4.w; }
        else           { gw3.x+=tmp.x*x4.x; gw3.y+=tmp.y*x4.y; gw3.z+=tmp.z*x4.z; gw3.w+=tmp.w*x4.w; }

        // gx += tmp * w_p
        gx4.x += tmp.x*wp.x; gx4.y += tmp.y*wp.y; gx4.z += tmp.z*wp.z; gx4.w += tmp.w*wp.w;
    }

    // ---- 向量写回 gw/gx ----
    *reinterpret_cast<double4*>(&gw[((b*4 + 0)*U) + u]) = gw0;
    *reinterpret_cast<double4*>(&gw[((b*4 + 1)*U) + u]) = gw1;
    *reinterpret_cast<double4*>(&gw[((b*4 + 2)*U) + u]) = gw2;
    *reinterpret_cast<double4*>(&gw[((b*4 + 3)*U) + u]) = gw3;
    *reinterpret_cast<double4*>(&gx[b*U + u])          = gx4;

    // ---- warp 内归约 16 个通道并写 gy[b,:]（仅 lane0 落地）----
    #pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        double v = gy_acc[k];
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (lane == 0) gy[b*16 + k] = v;
    }
}


__global__ void bwd_kernel_vec4_warp32_stream(
    const double* __restrict__ grad_out, // [B,16,U]
    const double* __restrict__ x,        // [B,U]
    const double* __restrict__ y,        // [B,16]
    const double* __restrict__ w,        // [B,4,U]
    const double* __restrict__ b_buf,    // [B,4,U]
    double* __restrict__ gx,             // [B,U]
    double* __restrict__ gy,             // [B,16]
    double* __restrict__ gw,             // [B,4,U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int lane  = threadIdx.x & 31;   // 0..31
    const int slots = U / 4;              // 96/4=24
    if (lane >= slots) return;

    __shared__ double y_sh[16];
    if (lane < 16) y_sh[lane] = y[b*16 + lane];
    __syncwarp();

    const int u   = lane * 4;
    const double4 x4 = *reinterpret_cast<const double4*>(&x[b*U + u]);

    const double4 w0 = *reinterpret_cast<const double4*>(&w[((b*4 + 0)*U) + u]);
    const double4 w1 = *reinterpret_cast<const double4*>(&w[((b*4 + 1)*U) + u]);
    const double4 w2 = *reinterpret_cast<const double4*>(&w[((b*4 + 2)*U) + u]);
    const double4 w3 = *reinterpret_cast<const double4*>(&w[((b*4 + 3)*U) + u]);

    const double4 b0 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 0)*U) + u]);
    const double4 b1 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 1)*U) + u]);
    const double4 b2 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 2)*U) + u]);
    const double4 b3 = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 3)*U) + u]);

    double4 gx4 = make_double4(0,0,0,0);
    double4 gw0 = make_double4(0,0,0,0);
    double4 gw1 = make_double4(0,0,0,0);
    double4 gw2 = make_double4(0,0,0,0);
    double4 gw3 = make_double4(0,0,0,0);

    #pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
        const double   yk = y_sh[k];

        // p 映射
        const int p = (k==0) ? 0 : (k<=3) ? 1 : (k<=8) ? 2 : 3;
        const double4 bp = (p==0)? b0 : (p==1)? b1 : (p==2)? b2 : b3;
        const double4 wp = (p==0)? w0 : (p==1)? w1 : (p==2)? w2 : w3;

        // —— 现场归约 gy：v = dot(g4, bp)；warp reduce 后 lane0 直接写回
        double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (lane == 0) gy[b*16 + k] = v;

        // —— 即时累计 gx/gw（不保留 tmp）
        const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;

        if (p==0) { gw0.x += t0*x4.x; gw0.y += t1*x4.y; gw0.z += t2*x4.z; gw0.w += t3*x4.w; }
        if (p==1) { gw1.x += t0*x4.x; gw1.y += t1*x4.y; gw1.z += t2*x4.z; gw1.w += t3*x4.w; }
        if (p==2) { gw2.x += t0*x4.x; gw2.y += t1*x4.y; gw2.z += t2*x4.z; gw2.w += t3*x4.w; }
        if (p==3) { gw3.x += t0*x4.x; gw3.y += t1*x4.y; gw3.z += t2*x4.z; gw3.w += t3*x4.w; }

        gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
    }

    *reinterpret_cast<double4*>(&gw[((b*4 + 0)*U) + u]) = gw0;
    *reinterpret_cast<double4*>(&gw[((b*4 + 1)*U) + u]) = gw1;
    *reinterpret_cast<double4*>(&gw[((b*4 + 2)*U) + u]) = gw2;
    *reinterpret_cast<double4*>(&gw[((b*4 + 3)*U) + u]) = gw3;
    *reinterpret_cast<double4*>(&gx[b*U + u])          = gx4;
}

// 单warp、U=96、vec4、流式gy归约 + 按需加载 w/b，进一步降寄存器
__global__ void bwd_kernel_vec4_warp32_stream_noprefetch(
    const double* __restrict__ grad_out, // [B,16,U]
    const double* __restrict__ x,        // [B,U]
    const double* __restrict__ y,        // [B,16]
    const double* __restrict__ w,        // [B,4,U]
    const double* __restrict__ b_buf,    // [B,4,U]
    double* __restrict__ gx,             // [B,U]
    double* __restrict__ gy,             // [B,16]
    double* __restrict__ gw,             // [B,4,U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int lane  = threadIdx.x & 31;   // 0..31
    const int slots = U / 4;              // 96/4=24
    if (lane >= slots) return;            // 剩余 8 个lane 直接退出

    // y -> shared（128B）
    __shared__ double y_sh[16];
    if (lane < 16) y_sh[lane] = y[b*16 + lane];
    __syncwarp();

    const int u = lane * 4;

    // 固定只读一次 x4
    const double4 x4 = *reinterpret_cast<const double4*>(&x[b*U + u]);

    // 必要的累加器（最小化保留）
    double4 gx4 = make_double4(0,0,0,0);
    double4 gw0 = make_double4(0,0,0,0);
    double4 gw1 = make_double4(0,0,0,0);
    double4 gw2 = make_double4(0,0,0,0);
    double4 gw3 = make_double4(0,0,0,0);

    #pragma unroll 1
    for (int k = 0; k < 16; ++k) {
        const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
        const double   yk = y_sh[k];

        // path 映射
        const int p = (k==0) ? 0 : (k<=3) ? 1 : (k<=8) ? 2 : 3;

        // —— 即时加载需要的 b_p4 / w_p4（不长驻寄存器）
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[((b*4 + p)*U) + u]);
        const double4 wp = *reinterpret_cast<const double4*>(&w    [((b*4 + p)*U) + u]);

        // —— gy：现场warp归约并落地
        double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (lane == 0) gy[b*16 + k] = v;

        // —— tmp = g4 * yk；即时贡献到 gw_p / gx4
        const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;

        if      (p==0) { gw0.x += t0*x4.x; gw0.y += t1*x4.y; gw0.z += t2*x4.z; gw0.w += t3*x4.w; }
        else if (p==1) { gw1.x += t0*x4.x; gw1.y += t1*x4.y; gw1.z += t2*x4.z; gw1.w += t3*x4.w; }
        else if (p==2) { gw2.x += t0*x4.x; gw2.y += t1*x4.y; gw2.z += t2*x4.z; gw2.w += t3*x4.w; }
        else           { gw3.x += t0*x4.x; gw3.y += t1*x4.y; gw3.z += t2*x4.z; gw3.w += t3*x4.w; }

        gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
    }

    // 写回（一次 vec4）
    *reinterpret_cast<double4*>(&gw[((b*4 + 0)*U) + u]) = gw0;
    *reinterpret_cast<double4*>(&gw[((b*4 + 1)*U) + u]) = gw1;
    *reinterpret_cast<double4*>(&gw[((b*4 + 2)*U) + u]) = gw2;
    *reinterpret_cast<double4*>(&gw[((b*4 + 3)*U) + u]) = gw3;
    *reinterpret_cast<double4*>(&gx[b*U + u])          = gx4;
}

// 单warp，U=96，vec4，按 path 分段分组（减少 w/b 反复加载）+ 流式 gy 归约
__global__ void bwd_kernel_vec4_warp32_stream_grouped(
    const double* __restrict__ grad_out, // [B,16,U]
    const double* __restrict__ x,        // [B,U]
    const double* __restrict__ y,        // [B,16]
    const double* __restrict__ w,        // [B,4,U]
    const double* __restrict__ b_buf,    // [B,4,U]
    double* __restrict__ gx,             // [B,U]
    double* __restrict__ gy,             // [B,16]
    double* __restrict__ gw,             // [B,4,U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int lane  = threadIdx.x & 31;   // 0..31
    const int slots = U / 4;              // 96/4=24
    if (lane >= slots) return;

    // y -> shared
    __shared__ double y_sh[16];
    if (lane < 16) y_sh[lane] = y[b*16 + lane];
    __syncwarp();

    const int u  = lane * 4;
    const double4 x4 = *reinterpret_cast<const double4*>(&x[b*U + u]);

    // 累加器（必要最少）
    double4 gx4 = make_double4(0,0,0,0);
    double4 gw0 = make_double4(0,0,0,0);
    double4 gw1 = make_double4(0,0,0,0);
    double4 gw2 = make_double4(0,0,0,0);
    double4 gw3 = make_double4(0,0,0,0);

    // ---- path 0：k = {0} ----
    {
        const double4 wp = *reinterpret_cast<const double4*>(&w    [((b*4 + 0)*U) + u]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 0)*U) + u]);

        const int k = 0;
        const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
        const double   yk = y_sh[k];

        // gy 流式归约写回
        double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (lane == 0) gy[b*16 + k] = v;

        // tmp
        const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
        // gw0 / gx
        gw0.x += t0*x4.x; gw0.y += t1*x4.y; gw0.z += t2*x4.z; gw0.w += t3*x4.w;
        gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
    }

    // ---- path 1：k = {1,2,3} ----
    {
        const double4 wp = *reinterpret_cast<const double4*>(&w    [((b*4 + 1)*U) + u]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 1)*U) + u]);

        #pragma unroll
        for (int k = 1; k <= 3; ++k) {
            const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
            const double   yk = y_sh[k];

            double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[b*16 + k] = v;

            const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
            gw1.x += t0*x4.x; gw1.y += t1*x4.y; gw1.z += t2*x4.z; gw1.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
        }
    }

    // ---- path 2：k = {4,5,6,7,8} ----
    {
        const double4 wp = *reinterpret_cast<const double4*>(&w    [((b*4 + 2)*U) + u]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 2)*U) + u]);

        #pragma unroll
        for (int k = 4; k <= 8; ++k) {
            const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
            const double   yk = y_sh[k];

            double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[b*16 + k] = v;

            const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
            gw2.x += t0*x4.x; gw2.y += t1*x4.y; gw2.z += t2*x4.z; gw2.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
        }
    }

    // ---- path 3：k = {9..15} ----
    {
        const double4 wp = *reinterpret_cast<const double4*>(&w    [((b*4 + 3)*U) + u]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[((b*4 + 3)*U) + u]);

        #pragma unroll
        for (int k = 9; k <= 15; ++k) {
            const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[(b*16 + k)*U + u]);
            const double   yk = y_sh[k];

            double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[b*16 + k] = v;

            const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
            gw3.x += t0*x4.x; gw3.y += t1*x4.y; gw3.z += t2*x4.z; gw3.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
        }
    }

    // 向量写回
    *reinterpret_cast<double4*>(&gw[((b*4 + 0)*U) + u]) = gw0;
    *reinterpret_cast<double4*>(&gw[((b*4 + 1)*U) + u]) = gw1;
    *reinterpret_cast<double4*>(&gw[((b*4 + 2)*U) + u]) = gw2;
    *reinterpret_cast<double4*>(&gw[((b*4 + 3)*U) + u]) = gw3;
    *reinterpret_cast<double4*>(&gx[b*U + u])          = gx4;
}

// ---------------- warp32 + vec4 + grouped + double-buffer ----------------
__global__ void bwd_kernel_vec4_warp32_grouped_db(
    const double* __restrict__ grad_out, // [B, KS, U]
    const double* __restrict__ x,        // [B, U]
    const double* __restrict__ y,        // [B, KS]
    const double* __restrict__ w,        // [B, P, U]
    const double* __restrict__ b_buf,    // [B, P, U] (前向缓存的 base = x*w)
    double* __restrict__ gx,             // [B, U]
    double* __restrict__ gy,             // [B, KS]
    double* __restrict__ gw,             // [B, P, U]
    int B, int U)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int lane  = threadIdx.x & 31;   // 0..31
    const int slots = U / 4;              // 96/4 = 24
    if (lane >= slots) return;            // 剩余 8 lane 直接退出（单warp）

    // y[b,:] 放 shared（128B）
    __shared__ double y_sh[KS];
    if (lane < KS) y_sh[lane] = y[b*KS + lane];
    __syncwarp();

    const int u = lane * 4;
    const size_t off_u = size_t(u);
    const size_t off_bu = size_t(b) * U + off_u;
    const size_t off_bk_base = size_t(b) * KS * U;

    // 固定只读一次 x4
    const double4 x4 = *reinterpret_cast<const double4*>(&x[off_bu]);

    // 必要的累加器
    double4 gx4 = make_double4(0,0,0,0);
    double4 gw0 = make_double4(0,0,0,0);
    double4 gw1 = make_double4(0,0,0,0);
    double4 gw2 = make_double4(0,0,0,0);
    double4 gw3 = make_double4(0,0,0,0);

    // ---- path 0: k = {0}  （单次，完全展开）----
    {
        const size_t off_pw = (size_t(b)*P + 0) * U + off_u;
        const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

        const int k = 0;
        const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[off_bk_base + size_t(k)*U + off_u]);
        const double   yk = y_sh[k];

        // gy：现场 warp 归约并写回
        double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (lane == 0) gy[b*KS + k] = v;

        // tmp = g4 * yk，累计 gw0/gx
        const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
        gw0.x += t0*x4.x; gw0.y += t1*x4.y; gw0.z += t2*x4.z; gw0.w += t3*x4.w;
        gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
    }

    // ---- path 1: k = {1,2,3}  （短段：允许完全展开）----
    {
        const size_t off_pw = (size_t(b)*P + 1) * U + off_u;
        const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

        #pragma unroll
        for (int k = 1; k <= 3; ++k) {
            const double4 g4 = *reinterpret_cast<const double4*>(&grad_out[off_bk_base + size_t(k)*U + off_u]);
            const double   yk = y_sh[k];

            double v = g4.x*bp.x + g4.y*bp.y + g4.z*bp.z + g4.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[b*KS + k] = v;

            const double t0 = g4.x * yk, t1 = g4.y * yk, t2 = g4.z * yk, t3 = g4.w * yk;
            gw1.x += t0*x4.x; gw1.y += t1*x4.y; gw1.z += t2*x4.z; gw1.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;
        }
    }

    // ---- path 2: k = {4,5,6,7,8}  （长段：双缓冲 + 禁止展开）----
    {
        const size_t off_pw = (size_t(b)*P + 2) * U + off_u;
        const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

        // 预取第一条
        double4 g4_cur = *reinterpret_cast<const double4*>(&grad_out[off_bk_base + size_t(4)*U + off_u]);
        double  yk_cur = y_sh[4];

        #pragma unroll 1
        for (int k = 4; k <= 8; ++k) {
            // 预取下一条
            double4 g4_next; double yk_next = 0.0;
            if (k < 8) {
                g4_next = *reinterpret_cast<const double4*>(&grad_out[off_bk_base + size_t(k+1)*U + off_u]);
                yk_next = y_sh[k+1];
            }

            // 当前条计算
            double v = g4_cur.x*bp.x + g4_cur.y*bp.y + g4_cur.z*bp.z + g4_cur.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[b*KS + k] = v;

            const double t0 = g4_cur.x * yk_cur, t1 = g4_cur.y * yk_cur;
            const double t2 = g4_cur.z * yk_cur, t3 = g4_cur.w * yk_cur;
            gw2.x += t0*x4.x; gw2.y += t1*x4.y; gw2.z += t2*x4.z; gw2.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;

            // 交换到下一条
            g4_cur = g4_next; yk_cur = yk_next;
        }
    }

    // ---- path 3: k = {9..15}  （长段：双缓冲 + 禁止展开）----
    {
        const size_t off_pw = (size_t(b)*P + 3) * U + off_u;
        const double4 wp = *reinterpret_cast<const double4*>(&w[off_pw]);
        const double4 bp = *reinterpret_cast<const double4*>(&b_buf[off_pw]);

        double4 g4_cur = *reinterpret_cast<const double4*>(&grad_out[off_bk_base + size_t(9)*U + off_u]);
        double  yk_cur = y_sh[9];

        #pragma unroll 1
        for (int k = 9; k <= 15; ++k) {
            double4 g4_next; double yk_next = 0.0;
            if (k < 15) {
                g4_next = *reinterpret_cast<const double4*>(&grad_out[off_bk_base + size_t(k+1)*U + off_u]);
                yk_next = y_sh[k+1];
            }

            double v = g4_cur.x*bp.x + g4_cur.y*bp.y + g4_cur.z*bp.z + g4_cur.w*bp.w;
            unsigned mask = 0xffffffffu;
            v += __shfl_down_sync(mask, v, 16);
            v += __shfl_down_sync(mask, v, 8);
            v += __shfl_down_sync(mask, v, 4);
            v += __shfl_down_sync(mask, v, 2);
            v += __shfl_down_sync(mask, v, 1);
            if (lane == 0) gy[b*KS + k] = v;

            const double t0 = g4_cur.x * yk_cur, t1 = g4_cur.y * yk_cur;
            const double t2 = g4_cur.z * yk_cur, t3 = g4_cur.w * yk_cur;
            gw3.x += t0*x4.x; gw3.y += t1*x4.y; gw3.z += t2*x4.z; gw3.w += t3*x4.w;
            gx4.x += t0*wp.x; gx4.y += t1*wp.y; gx4.z += t2*wp.z; gx4.w += t3*wp.w;

            g4_cur = g4_next; yk_cur = yk_next;
        }
    }

    // 写回（vec4）
    *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 0)*U + off_u]) = gw0;
    *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 1)*U + off_u]) = gw1;
    *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 2)*U + off_u]) = gw2;
    *reinterpret_cast<double4*>(&gw[(size_t(b)*P + 3)*U + off_u]) = gw3;
    *reinterpret_cast<double4*>(&gx[off_bu])                      = gx4;
}



// ---------------------------------------------
// PyTorch 封装
// ---------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> fwd_launcher(
    const at::Tensor& x,     // [B, U]   double, cuda, contiguous
    const at::Tensor& y,     // [B, 16]
    const at::Tensor& w      // [B, 4, U]
) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && w.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kDouble, "expect double");
    TORCH_CHECK(y.scalar_type() == at::kDouble, "expect double");
    TORCH_CHECK(w.scalar_type() == at::kDouble, "expect double");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && w.is_contiguous(), "expect contiguous");

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);
    TORCH_CHECK(U == U_FIXED, "This kernel expects U=96");

    auto out   = at::empty({B, DIM_SUM, U}, x.options()); // [B,16,96]
    auto b_buf = at::empty({B, P, U}, x.options());       // [B,4,96]

    dim3 grid(B);
    dim3 block(128); // >= 96
    fwd_kernel_v1<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        b_buf.data_ptr<double>(),
        (int)B, (int)U
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out_flat = out.view({B, DIM_SUM * U});   // 方便兼容旧接口
    return {out, out_flat, b_buf};
}


std::tuple<at::Tensor, at::Tensor> fwd_warp32_grouped_bstride_launcher(
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
    TORCH_CHECK(w.sizes() == at::IntArrayRef({B, P, U}), "w must be [B,4,96]");

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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, b_buf}; // out:[B,16,96], b_buf:[B,4,96]
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> bwd_launcher(
    const at::Tensor& grad_out, // [B,16,96]
    const at::Tensor& x,        // [B,96]
    const at::Tensor& y,        // [B,16]
    const at::Tensor& w,        // [B,4,96]
    const at::Tensor& b_buf     // [B,4,96]  前向缓存
) {
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && y.is_cuda() && w.is_cuda() && b_buf.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(grad_out.scalar_type() == at::kDouble && x.scalar_type() == at::kDouble && y.scalar_type() == at::kDouble && w.scalar_type() == at::kDouble && b_buf.scalar_type() == at::kDouble, "expect double");
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() && y.is_contiguous() && w.is_contiguous() && b_buf.is_contiguous(), "expect contiguous");

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);
    TORCH_CHECK(U == U_FIXED, "This kernel expects U=96");
    TORCH_CHECK(grad_out.sizes() == at::IntArrayRef({B, DIM_SUM, U}), "grad_out must be [B,16,96]");
    TORCH_CHECK(b_buf.sizes()    == at::IntArrayRef({B, P, U}),        "b_buf must be [B,4,96]");

    auto gx = at::empty_like(x);          // [B,96]
    auto gy = at::empty_like(y);          // [B,16]
    auto gw = at::empty_like(w);          // [B,4,96]

    dim3 grid(B);
    dim3 block(128);
    size_t smem_bytes = static_cast<size_t>(block.x) * DIM_SUM * sizeof(double); // nthreads * 16
    bwd_kernel_v1<<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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

// grad_out 转置与不转置复用一个launcher
std::tuple<at::Tensor, at::Tensor, at::Tensor> bwd_launcher_v2(
    const at::Tensor& grad_out_T, // [B, U, 16]  (已转置 + contiguous)
    const at::Tensor& x,          // [B, 96]
    const at::Tensor& y,          // [B, 16]
    const at::Tensor& w,          // [B, 4, 96]
    const at::Tensor& b_buf       // [B, 4, 96]
) {
    TORCH_CHECK(grad_out_T.is_cuda() && x.is_cuda() && y.is_cuda() && w.is_cuda() && b_buf.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(grad_out_T.scalar_type() == at::kDouble && x.scalar_type() == at::kDouble && y.scalar_type() == at::kDouble && w.scalar_type() == at::kDouble && b_buf.scalar_type() == at::kDouble, "expect double");
    TORCH_CHECK(grad_out_T.is_contiguous() && x.is_contiguous() && y.is_contiguous() && w.is_contiguous() && b_buf.is_contiguous(), "expect contiguous");

    const int64_t B = x.size(0);
    const int64_t U = x.size(1);
    TORCH_CHECK(U == U_FIXED, "This kernel expects U=96");
    //TORCH_CHECK(grad_out_T.sizes() == at::IntArrayRef({B, U, DIM_SUM}), "grad_out_T must be [B,96,16]");
    
    TORCH_CHECK(b_buf.sizes()    == at::IntArrayRef({B, P, U}),         "b_buf must be [B,4,96]");

    auto gx = at::empty_like(x);          // [B,96]
    auto gy = at::empty_like(y);          // [B,16]
    auto gw = at::empty_like(w);          // [B,4,96]

    dim3 grid(B);
    dim3 block(32);
    size_t smem_bytes = (block.x / 32) * DIM_SUM * sizeof(double); // 每 warp 16 doubles
    bwd_kernel_vec4_warp32_grouped_db<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out_T.data_ptr<double>(),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd_warp32_grouped_bstride_launcher, "einsum_simplified_v1 forward (V=1, dim=[1,3,5,7]) with b_buf");
    m.def("bwd", &bwd_launcher_v2, "einsum_simplified_v1 backward (V=1, dim=[1,3,5,7]) using b_buf");
}
