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
  using GPU_Guard = c10::cuda::CUDAGuard;
  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

  using GPU_Guard = c10::DeviceGuard;
#include "cuda_utils.hpp"

template <typename scalar_t, typename index_t>
__global__ void uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)
{
    const int e_local = (int)blockIdx.x;
    if (e_local >= B) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    constexpr int U_CONST = 128;
    (void)U_CONST;

    const int e_orig = b_list ? b_list[e_local] : e_local;
    const int w_row  = (WB == 1 ? 0 : e_orig);

    const int x_row = src_idx[e_orig];
    const int y_row = e_orig;
    const int out_row = dst_idx[e_orig];

    const index_t w_base = (index_t)w_row  * (index_t)Iw * (index_t)U;
    const index_t x_base = (index_t)x_row  * (index_t)Ix * (index_t)U;
    const index_t y_base = (index_t)y_row  * (index_t)Ky;
    const index_t out_base = (index_t)out_row * (index_t)V  * (index_t)U;

    // full-resident output accumulators across all phases
    scalar_t out_acc_v_0;
    scalar_t out_acc_v_1;
    scalar_t out_acc_v_2;
    scalar_t out_acc_v_3;
    scalar_t out_acc_v_4;
    scalar_t out_acc_v_5;
    scalar_t out_acc_v_6;
    scalar_t out_acc_v_7;
    scalar_t out_acc_v_8;
    scalar_t out_acc_v_9;
    scalar_t out_acc_v_10;
    scalar_t out_acc_v_11;
    scalar_t out_acc_v_12;
    scalar_t out_acc_v_13;
    scalar_t out_acc_v_14;
    scalar_t out_acc_v_15;

    // phase-local operand / pair slots
    scalar_t wi_slot_0;
    scalar_t wi_slot_1;
    scalar_t wi_slot_2;
    scalar_t wi_slot_3;
    scalar_t x_slot_0;
    scalar_t y_slot_0;
    scalar_t y_slot_1;
    scalar_t y_slot_2;
    scalar_t y_slot_3;
    scalar_t y_slot_4;
    scalar_t y_slot_5;
    scalar_t y_slot_6;
    scalar_t y_slot_7;
    scalar_t y_slot_8;
    scalar_t y_slot_9;
    scalar_t y_slot_10;
    scalar_t y_slot_11;
    scalar_t y_slot_12;
    scalar_t y_slot_13;
    scalar_t y_slot_14;
    scalar_t y_slot_15;
    scalar_t pair_slot_0;
    scalar_t pair_slot_1;
    scalar_t pair_slot_2;
    scalar_t pair_slot_3;

    for (int u_base = 0; u_base < U; u_base += 32) {
        int u = u_base + lane;
        if (u < U) {

            // reset output accumulators
            out_acc_v_0 = scalar_t(0);
            out_acc_v_1 = scalar_t(0);
            out_acc_v_2 = scalar_t(0);
            out_acc_v_3 = scalar_t(0);
            out_acc_v_4 = scalar_t(0);
            out_acc_v_5 = scalar_t(0);
            out_acc_v_6 = scalar_t(0);
            out_acc_v_7 = scalar_t(0);
            out_acc_v_8 = scalar_t(0);
            out_acc_v_9 = scalar_t(0);
            out_acc_v_10 = scalar_t(0);
            out_acc_v_11 = scalar_t(0);
            out_acc_v_12 = scalar_t(0);
            out_acc_v_13 = scalar_t(0);
            out_acc_v_14 = scalar_t(0);
            out_acc_v_15 = scalar_t(0);

            // ===== phase 0: main_pair_kind=wx =====
            {
                // phase-local wi preload
                wi_slot_0 = w[w_base + (index_t)0 + (index_t)u];
                wi_slot_1 = w[w_base + (index_t)128 + (index_t)u];
                wi_slot_2 = w[w_base + (index_t)256 + (index_t)u];
                wi_slot_3 = w[w_base + (index_t)384 + (index_t)u];

                // phase-local x preload
                x_slot_0 = x[x_base + (index_t)0 + (index_t)u];

                // phase-local y preload
                y_slot_0 = y[y_base + (index_t)0];
                y_slot_1 = y[y_base + (index_t)1];
                y_slot_2 = y[y_base + (index_t)2];
                y_slot_3 = y[y_base + (index_t)3];
                y_slot_4 = y[y_base + (index_t)4];
                y_slot_5 = y[y_base + (index_t)5];
                y_slot_6 = y[y_base + (index_t)6];
                y_slot_7 = y[y_base + (index_t)7];
                y_slot_8 = y[y_base + (index_t)8];
                y_slot_9 = y[y_base + (index_t)9];
                y_slot_10 = y[y_base + (index_t)10];
                y_slot_11 = y[y_base + (index_t)11];
                y_slot_12 = y[y_base + (index_t)12];
                y_slot_13 = y[y_base + (index_t)13];
                y_slot_14 = y[y_base + (index_t)14];
                y_slot_15 = y[y_base + (index_t)15];

                // phase-local pair cache kind=wx
                pair_slot_0 = wi_slot_0 * x_slot_0;
                pair_slot_1 = wi_slot_1 * x_slot_0;
                pair_slot_2 = wi_slot_2 * x_slot_0;
                pair_slot_3 = wi_slot_3 * x_slot_0;

                // ---- subphase 0 ----
                out_acc_v_0 += scalar_t(1.0) * pair_slot_0 * y_slot_0;
                out_acc_v_1 += scalar_t(1.0000000000000002) * pair_slot_1 * y_slot_1;
                out_acc_v_2 += scalar_t(0.9999999999999998) * pair_slot_1 * y_slot_2;
                out_acc_v_3 += scalar_t(1.0000000000000002) * pair_slot_1 * y_slot_3;
                out_acc_v_4 += scalar_t(0.9999999999999993) * pair_slot_2 * y_slot_4;
                out_acc_v_5 += scalar_t(1.0000000000000002) * pair_slot_2 * y_slot_5;
                out_acc_v_6 += scalar_t(0.9999999999999997) * pair_slot_2 * y_slot_6;
                out_acc_v_7 += scalar_t(1.0) * pair_slot_2 * y_slot_7;
                out_acc_v_8 += scalar_t(1.0000000000000004) * pair_slot_2 * y_slot_8;
                out_acc_v_9 += scalar_t(1.0000000000000007) * pair_slot_3 * y_slot_9;
                out_acc_v_10 += scalar_t(1.0000000000000004) * pair_slot_3 * y_slot_10;
                out_acc_v_11 += scalar_t(0.9999999999999991) * pair_slot_3 * y_slot_11;
                out_acc_v_12 += scalar_t(0.9999999999999996) * pair_slot_3 * y_slot_12;
                out_acc_v_13 += scalar_t(0.9999999999999996) * pair_slot_3 * y_slot_13;
                out_acc_v_14 += scalar_t(0.9999999999999998) * pair_slot_3 * y_slot_14;
                out_acc_v_15 += scalar_t(1.0000000000000004) * pair_slot_3 * y_slot_15;

            }

            // write out
            atomicAdd(&out[out_base + (index_t)0 + (index_t)u], out_acc_v_0);
            atomicAdd(&out[out_base + (index_t)128 + (index_t)u], out_acc_v_1);
            atomicAdd(&out[out_base + (index_t)256 + (index_t)u], out_acc_v_2);
            atomicAdd(&out[out_base + (index_t)384 + (index_t)u], out_acc_v_3);
            atomicAdd(&out[out_base + (index_t)512 + (index_t)u], out_acc_v_4);
            atomicAdd(&out[out_base + (index_t)640 + (index_t)u], out_acc_v_5);
            atomicAdd(&out[out_base + (index_t)768 + (index_t)u], out_acc_v_6);
            atomicAdd(&out[out_base + (index_t)896 + (index_t)u], out_acc_v_7);
            atomicAdd(&out[out_base + (index_t)1024 + (index_t)u], out_acc_v_8);
            atomicAdd(&out[out_base + (index_t)1152 + (index_t)u], out_acc_v_9);
            atomicAdd(&out[out_base + (index_t)1280 + (index_t)u], out_acc_v_10);
            atomicAdd(&out[out_base + (index_t)1408 + (index_t)u], out_acc_v_11);
            atomicAdd(&out[out_base + (index_t)1536 + (index_t)u], out_acc_v_12);
            atomicAdd(&out[out_base + (index_t)1664 + (index_t)u], out_acc_v_13);
            atomicAdd(&out[out_base + (index_t)1792 + (index_t)u], out_acc_v_14);
            atomicAdd(&out[out_base + (index_t)1920 + (index_t)u], out_acc_v_15);
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
    bool y_ok = false;
    if (mode_scalar_y) {
        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
        y_ok = mul_fits_int32(y_dim0, (int64_t)Ky);
    } else {
        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
        y_ok = mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);
    }
    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && out_ok;
}

template <typename scalar_t, typename index_t>
void launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_typed(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd<scalar_t, index_t><<<grid, block, 0, stream>>>(
        w, x, y, out,
        src_idx, dst_idx, b_list,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    constexpr bool kUseXSrc = true;
    constexpr bool kUseYSrc = false;
    constexpr bool kUseScatter = true;
    constexpr bool kModeScalarY = true;
    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,
                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_typed<scalar_t, int32_t>(
            w, x, y, out, src_idx, dst_idx, b_list,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_typed<scalar_t, int64_t>(
            w, x, y, out, src_idx, dst_idx, b_list,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd_auto<scalar_t>(
        w, x, y, out, src_idx, dst_idx, b_list,
        B, WB, Iw, Ix, Ky, V, U, S, stream);
}



torch::Tensor launcher_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd(
    torch::Tensor w,
    torch::Tensor x_all,
    torch::Tensor y,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    torch::Tensor b_list,
    int64_t V64)
{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : [B,Ky,1] or [S,Ky,1]
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used
    //   b_list : [B] int32 optional, enabled when scatter is used

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


    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA/HIP");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }

    auto out = torch::zeros({S, V, U}, w.options());

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd", [&] {

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd, "uniform1d_u128_path16_uu_u_xsrc1_ysrc0_scatter1_fwd forward jit impl");
}
