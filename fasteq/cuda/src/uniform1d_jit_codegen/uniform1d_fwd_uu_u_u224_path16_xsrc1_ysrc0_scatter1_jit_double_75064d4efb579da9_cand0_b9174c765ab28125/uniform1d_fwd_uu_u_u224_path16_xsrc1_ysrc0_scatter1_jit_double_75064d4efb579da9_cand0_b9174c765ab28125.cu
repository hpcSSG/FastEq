#include <stdint.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPGuard.h>
  using gpuStream_t = hipStream_t;
  #define getCurrentGPUStream at::hip::getCurrentHIPStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpuStream_t = cudaStream_t;
  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

using GPU_Guard = c10::DeviceGuard;

template <typename scalar_t, typename index_t>
__global__ void uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)
{
    const int e_local = (int)blockIdx.x;
    if (e_local >= B) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    constexpr int U_CONST = 224;
    (void)U_CONST;

    const int e_orig = e_local;
    const int w_row  = (WB == 1 ? 0 : e_orig);

    const int x_row = src_idx[e_orig];
    const int y_row = e_orig;
    const int out_row = dst_idx[e_orig];

    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;
    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;
    const index_t y_base = (index_t)y_row * (index_t)Ky;
    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;

    for (int u_base = 0; u_base < U; u_base += 32) {
        const int u = u_base + lane;
        if (u < U) {
            scalar_t r0;
            scalar_t r1;
            scalar_t r2;

            // direct single-use output writeback enabled for 16 out accumulator(s)

            // inst 0: load | global key=(0, 0, (0, 0, 0, 0, -2, 1), "('y', 9)")
            r0 = y[y_base + (index_t)9];
            // inst 1: load | global key=(0, 0, (0, 0, 1, 0, -7, 7), "('w', 3)")
            r1 = w[w_base + (index_t)672 + (index_t)u];
            // inst 2: load | global key=(1, 1, (2, 1, 8, 0, -18, 16), "('x', 0)")
            r2 = x[x_base + (index_t)0 + (index_t)u];
            // inst 3: fma_u1d_direct | path#9: direct out[9] += x[0] * y[9] * w[3] * 1.0000000000000007
            atomicAdd(&out[out_base + (index_t)2016 + (index_t)u], scalar_t(1.0000000000000007) * (r1 * r2) * r0);
            // inst 4: release | last use after path#9
            // inst 5: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 15)")
            r0 = y[y_base + (index_t)15];
            // inst 6: fma_u1d_direct | path#15: direct out[15] += x[0] * y[15] * w[3] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)3360 + (index_t)u], scalar_t(1.0000000000000004) * (r1 * r2) * r0);
            // inst 7: release | last use after path#15
            // inst 8: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 14)")
            r0 = y[y_base + (index_t)14];
            // inst 9: fma_u1d_direct | path#14: direct out[14] += x[0] * y[14] * w[3] * 0.9999999999999998
            atomicAdd(&out[out_base + (index_t)3136 + (index_t)u], scalar_t(0.9999999999999998) * (r1 * r2) * r0);
            // inst 10: release | last use after path#14
            // inst 11: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 13)")
            r0 = y[y_base + (index_t)13];
            // inst 12: fma_u1d_direct | path#13: direct out[13] += x[0] * y[13] * w[3] * 0.9999999999999996
            atomicAdd(&out[out_base + (index_t)2912 + (index_t)u], scalar_t(0.9999999999999996) * (r1 * r2) * r0);
            // inst 13: release | last use after path#13
            // inst 14: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 12)")
            r0 = y[y_base + (index_t)12];
            // inst 15: fma_u1d_direct | path#12: direct out[12] += x[0] * y[12] * w[3] * 0.9999999999999996
            atomicAdd(&out[out_base + (index_t)2688 + (index_t)u], scalar_t(0.9999999999999996) * (r1 * r2) * r0);
            // inst 16: release | last use after path#12
            // inst 17: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 11)")
            r0 = y[y_base + (index_t)11];
            // inst 18: fma_u1d_direct | path#11: direct out[11] += x[0] * y[11] * w[3] * 0.9999999999999991
            atomicAdd(&out[out_base + (index_t)2464 + (index_t)u], scalar_t(0.9999999999999991) * (r1 * r2) * r0);
            // inst 19: release | last use after path#11
            // inst 20: load | global key=(1, 2, (3, 1, 2, 0, 0, 1), "('y', 10)")
            r0 = y[y_base + (index_t)10];
            // inst 21: fma_u1d_direct | path#10: direct out[10] += x[0] * y[10] * w[3] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)2240 + (index_t)u], scalar_t(1.0000000000000004) * (r1 * r2) * r0);
            // inst 22: release | last use after path#10
            // inst 23: release | last use after path#10
            // inst 24: load | global key=(0, 0, (0, 0, 5, 0, -5, 5), "('w', 2)")
            r1 = w[w_base + (index_t)448 + (index_t)u];
            // inst 25: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 8)")
            r0 = y[y_base + (index_t)8];
            // inst 26: fma_u1d_direct | path#8: direct out[8] += x[0] * y[8] * w[2] * 1.0000000000000004
            atomicAdd(&out[out_base + (index_t)1792 + (index_t)u], scalar_t(1.0000000000000004) * (r1 * r2) * r0);
            // inst 27: release | last use after path#8
            // inst 28: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 7)")
            r0 = y[y_base + (index_t)7];
            // inst 29: fma_u1d_direct | path#7: direct out[7] += x[0] * y[7] * w[2] * 1.0
            atomicAdd(&out[out_base + (index_t)1568 + (index_t)u], scalar_t(1.0) * (r1 * r2) * r0);
            // inst 30: release | last use after path#7
            // inst 31: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 6)")
            r0 = y[y_base + (index_t)6];
            // inst 32: fma_u1d_direct | path#6: direct out[6] += x[0] * y[6] * w[2] * 0.9999999999999997
            atomicAdd(&out[out_base + (index_t)1344 + (index_t)u], scalar_t(0.9999999999999997) * (r1 * r2) * r0);
            // inst 33: release | last use after path#6
            // inst 34: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 5)")
            r0 = y[y_base + (index_t)5];
            // inst 35: fma_u1d_direct | path#5: direct out[5] += x[0] * y[5] * w[2] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)1120 + (index_t)u], scalar_t(1.0000000000000002) * (r1 * r2) * r0);
            // inst 36: release | last use after path#5
            // inst 37: load | global key=(1, 2, (3, 1, 2, 0, 0, 1), "('y', 4)")
            r0 = y[y_base + (index_t)4];
            // inst 38: fma_u1d_direct | path#4: direct out[4] += x[0] * y[4] * w[2] * 0.9999999999999993
            atomicAdd(&out[out_base + (index_t)896 + (index_t)u], scalar_t(0.9999999999999993) * (r1 * r2) * r0);
            // inst 39: release | last use after path#4
            // inst 40: release | last use after path#4
            // inst 41: load | global key=(0, 0, (0, 0, 3, 0, -3, 3), "('w', 1)")
            r1 = w[w_base + (index_t)224 + (index_t)u];
            // inst 42: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 3)")
            r0 = y[y_base + (index_t)3];
            // inst 43: fma_u1d_direct | path#3: direct out[3] += x[0] * y[3] * w[1] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)672 + (index_t)u], scalar_t(1.0000000000000002) * (r1 * r2) * r0);
            // inst 44: release | last use after path#3
            // inst 45: load | global key=(1, 1, (1, 1, 2, 0, 0, 1), "('y', 2)")
            r0 = y[y_base + (index_t)2];
            // inst 46: fma_u1d_direct | path#2: direct out[2] += x[0] * y[2] * w[1] * 0.9999999999999998
            atomicAdd(&out[out_base + (index_t)448 + (index_t)u], scalar_t(0.9999999999999998) * (r1 * r2) * r0);
            // inst 47: release | last use after path#2
            // inst 48: load | global key=(1, 2, (3, 1, 2, 0, 0, 1), "('y', 1)")
            r0 = y[y_base + (index_t)1];
            // inst 49: fma_u1d_direct | path#1: direct out[1] += x[0] * y[1] * w[1] * 1.0000000000000002
            atomicAdd(&out[out_base + (index_t)224 + (index_t)u], scalar_t(1.0000000000000002) * (r1 * r2) * r0);
            // inst 50: release | last use after path#1
            // inst 51: release | last use after path#1
            // inst 52: load | global key=(0, 0, (0, 0, 1, 0, -1, 1), "('y', 0)")
            r1 = y[y_base + (index_t)0];
            // inst 53: load | global key=(1, 3, (5, 1, 2, 0, 0, 1), "('w', 0)")
            r0 = w[w_base + (index_t)0 + (index_t)u];
            // inst 54: fma_u1d_direct | path#0: direct out[0] += x[0] * y[0] * w[0] * 1.0
            atomicAdd(&out[out_base + (index_t)0 + (index_t)u], scalar_t(1.0) * (r0 * r2) * r1);
            // inst 55: release | last use after path#0
            // inst 56: release | last use after path#0
            // inst 57: release | last use after path#0
        }
    }
}

static inline bool mul_fits_int32(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}

static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32(a, b)) return false;
    return mul_fits_int32(a * b, c);
}

static inline bool should_use_int32_index_fwd(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && out_ok;
}

template <typename scalar_t, typename index_t>
void launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_typed(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd<scalar_t, index_t><<<grid, block, 0, stream>>>(
        w, x, y, out,
        src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    constexpr bool kUseXSrc = true;
    constexpr bool kUseYSrc = false;
    constexpr bool kUseScatter = true;
    constexpr bool kModeScalarY = true;
    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,
                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_typed<scalar_t, int32_t>(
            w, x, y, out, src_idx, dst_idx,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_typed<scalar_t, int64_t>(
            w, x, y, out, src_idx, dst_idx,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_auto<scalar_t>(
        w, x, y, out, src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S, stream);
}



torch::Tensor launcher_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd(
    torch::Tensor w,
    torch::Tensor x_all,
    torch::Tensor y,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    int64_t V64)
{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : [B,Ky,1] or [S,Ky,1]
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "w/x_all/y must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");

    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");

    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");


    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int)dst_idx.size(0);
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");

    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(1) > 0, "Iw must be > 0");

    TORCH_CHECK((int)x_all.size(1) > 0, "Ix must be > 0");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");

    TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");

    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");
    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");


    auto out = torch::zeros({S, V, U}, w.options());

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd", [&] {

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd, "uniform1d_lars_all_inputs_u224_path16_uu_u_xsrc1_ysrc0_scatter1_fwd forward jit impl");
}
