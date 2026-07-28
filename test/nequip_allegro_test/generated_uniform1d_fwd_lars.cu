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
__global__ void uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd(
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
    const int warp_id = tid >> 5;
    const int warp_count = blockDim.x >> 5;

    constexpr int U_CONST = 64;
    (void)U_CONST;

    const int e_orig = e_local;
    const int w_row  = (WB == 1 ? 0 : e_orig);

    const int x_row = e_local;
    const int y_row = src_idx[e_orig];
    const int out_row = e_orig;

    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;
    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;
    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;
    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;

    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {
        const int u = u_base + lane;
        if (u < U) {
            scalar_t r0;
            scalar_t r1;

            // inst 0: load | placement=register accesses=7 paths=7 lifetime=7
            r0 = w[w_base + (index_t)192 + (index_t)u];
            // inst 1: init_acc | placement=register accesses=16 paths=16 lifetime=16
            r1 = scalar_t(0);
            // inst 2: fma_u1d_placed | path#9: out[0] += x[9] * y[9] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)576 + (index_t)u]) * y[y_base + (index_t)576 + (index_t)u];
            // inst 3: fma_u1d_placed | path#15: out[0] += x[15] * y[15] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)960 + (index_t)u]) * y[y_base + (index_t)960 + (index_t)u];
            // inst 4: fma_u1d_placed | path#14: out[0] += x[14] * y[14] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)896 + (index_t)u]) * y[y_base + (index_t)896 + (index_t)u];
            // inst 5: fma_u1d_placed | path#13: out[0] += x[13] * y[13] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)832 + (index_t)u]) * y[y_base + (index_t)832 + (index_t)u];
            // inst 6: fma_u1d_placed | path#12: out[0] += x[12] * y[12] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)768 + (index_t)u]) * y[y_base + (index_t)768 + (index_t)u];
            // inst 7: fma_u1d_placed | path#11: out[0] += x[11] * y[11] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)704 + (index_t)u]) * y[y_base + (index_t)704 + (index_t)u];
            // inst 8: fma_u1d_placed | path#10: out[0] += x[10] * y[10] * w[3] * 0.37796446681022644
            r1 += scalar_t(0.37796446681022644) * (r0 * x[x_base + (index_t)640 + (index_t)u]) * y[y_base + (index_t)640 + (index_t)u];
            // inst 9: release | last use at scheduled path position 6
            // inst 10: load | placement=register accesses=5 paths=5 lifetime=5
            r0 = w[w_base + (index_t)128 + (index_t)u];
            // inst 11: fma_u1d_placed | path#8: out[0] += x[8] * y[8] * w[2] * 0.4472135901451111
            r1 += scalar_t(0.4472135901451111) * (r0 * x[x_base + (index_t)512 + (index_t)u]) * y[y_base + (index_t)512 + (index_t)u];
            // inst 12: fma_u1d_placed | path#7: out[0] += x[7] * y[7] * w[2] * 0.4472135901451111
            r1 += scalar_t(0.4472135901451111) * (r0 * x[x_base + (index_t)448 + (index_t)u]) * y[y_base + (index_t)448 + (index_t)u];
            // inst 13: fma_u1d_placed | path#6: out[0] += x[6] * y[6] * w[2] * 0.4472135901451111
            r1 += scalar_t(0.4472135901451111) * (r0 * x[x_base + (index_t)384 + (index_t)u]) * y[y_base + (index_t)384 + (index_t)u];
            // inst 14: fma_u1d_placed | path#5: out[0] += x[5] * y[5] * w[2] * 0.4472135901451111
            r1 += scalar_t(0.4472135901451111) * (r0 * x[x_base + (index_t)320 + (index_t)u]) * y[y_base + (index_t)320 + (index_t)u];
            // inst 15: fma_u1d_placed | path#4: out[0] += x[4] * y[4] * w[2] * 0.4472135901451111
            r1 += scalar_t(0.4472135901451111) * (r0 * x[x_base + (index_t)256 + (index_t)u]) * y[y_base + (index_t)256 + (index_t)u];
            // inst 16: release | last use at scheduled path position 11
            // inst 17: load | placement=register accesses=3 paths=3 lifetime=3
            r0 = w[w_base + (index_t)64 + (index_t)u];
            // inst 18: fma_u1d_placed | path#3: out[0] += x[3] * y[3] * w[1] * 0.5773502588272095
            r1 += scalar_t(0.5773502588272095) * (r0 * x[x_base + (index_t)192 + (index_t)u]) * y[y_base + (index_t)192 + (index_t)u];
            // inst 19: fma_u1d_placed | path#2: out[0] += x[2] * y[2] * w[1] * 0.5773502588272095
            r1 += scalar_t(0.5773502588272095) * (r0 * x[x_base + (index_t)128 + (index_t)u]) * y[y_base + (index_t)128 + (index_t)u];
            // inst 20: fma_u1d_placed | path#1: out[0] += x[1] * y[1] * w[1] * 0.5773502588272095
            r1 += scalar_t(0.5773502588272095) * (r0 * x[x_base + (index_t)64 + (index_t)u]) * y[y_base + (index_t)64 + (index_t)u];
            // inst 21: release | last use at scheduled path position 14
            // inst 22: fma_u1d_placed | path#0: out[0] += x[0] * y[0] * w[0] * 1.0
            r1 += scalar_t(1.0) * (w[w_base + (index_t)0 + (index_t)u] * x[x_base + (index_t)0 + (index_t)u]) * y[y_base + (index_t)0 + (index_t)u];
            // inst 23: store_acc_placed | last use at scheduled path position 15
            out[out_base + (index_t)0 + (index_t)u] += r1;
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
void launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd_typed(
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
    size_t lars_shared_bytes = (size_t)0 * (size_t)block.x * sizeof(scalar_t);
    uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(
        w, x, y, out,
        src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    constexpr bool kUseXSrc = false;
    constexpr bool kUseYSrc = true;
    constexpr bool kUseScatter = false;
    constexpr bool kModeScalarY = false;
    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,
                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd_typed<scalar_t, int32_t>(
            w, x, y, out, src_idx, dst_idx,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd_typed<scalar_t, int64_t>(
            w, x, y, out, src_idx, dst_idx,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd_auto<scalar_t>(
        w, x, y, out, src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S, stream);
}



torch::Tensor launcher_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd(
    torch::Tensor w,
    torch::Tensor x_all,
    torch::Tensor y,
    torch::Tensor src_idx,
    int64_t V64)
{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : [B,Ky,U] or [S,Ky,U]
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "w/x_all/y must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");

    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");


    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int)src_idx.size(0);
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

    TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");

    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");



    auto out = torch::zeros({B, V, U}, w.options());

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd", [&] {

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                nullptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd, "uniform1d_lars_all_inputs_u64_path16_uuuu_xsrc0_ysrc1_scatter0_fwd forward jit impl");
}
