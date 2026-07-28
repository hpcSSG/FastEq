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

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum_xor_lars_bwd(scalar_t v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        v += __shfl_xor(v, offset);
#else
        v += __shfl_xor_sync(0xffffffff, v, offset);
#endif
    }
    return v;
}

template <typename scalar_t, typename index_t>
__global__ void uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_y,
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
    extern __shared__ unsigned char lars_smem_raw[];
    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);
    constexpr int U_CONST = 64;
    (void)U_CONST;
    const int e_orig = e_local;
    const int w_row  = (WB == 1 ? 0 : e_orig);

    const int src = src_idx[e_orig];
    const int dst = e_orig;

    const int x_row = e_local;
    const int y_row = src;
    const int go_row = e_orig;

    const index_t w_base  = (index_t)w_row * (index_t)256;
    const index_t x_base  = (index_t)x_row * (index_t)1024;
    const index_t gx_base = (index_t)x_row * (index_t)1024;
    const index_t y_base  = (index_t)y_row * (index_t)1024;
    const index_t gy_base = (index_t)y_row * (index_t)1024;
    const index_t go_base = (index_t)go_row * (index_t)64;

    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {
        const int u = u_base + lane;
        if (u < U) {
            scalar_t r0;

            // inst 0: load_shared | placement=shared accesses=32 paths=16 lifetime=16
            lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid] = grad_out[go_base + (index_t)0 + (index_t)u];
            // inst 1: load_shared | placement=shared accesses=14 paths=7 lifetime=14
            lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] = w[w_base + (index_t)192 + (index_t)u];
            // inst 2: bwd_fma_placed | path#9: gw[3] += go[0]*x[9]*y[9], gx[9] += w[3]*go[0]*y[9], gy[9] += w[3]*go[0]*x[9]
            atomicAdd(&grad_x[gx_base + (index_t)576 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)576 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)576 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)576 + (index_t)u]);
            // inst 3: load | placement=register accesses=10 paths=5 lifetime=5
            r0 = w[w_base + (index_t)128 + (index_t)u];
            // inst 4: bwd_fma_placed | path#8: gw[2] += go[0]*x[8]*y[8], gx[8] += w[2]*go[0]*y[8], gy[8] += w[2]*go[0]*x[8]
            atomicAdd(&grad_x[gx_base + (index_t)512 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)512 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)512 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)512 + (index_t)u]);
            // inst 5: bwd_fma_placed | path#7: gw[2] += go[0]*x[7]*y[7], gx[7] += w[2]*go[0]*y[7], gy[7] += w[2]*go[0]*x[7]
            atomicAdd(&grad_x[gx_base + (index_t)448 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)448 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)448 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)448 + (index_t)u]);
            // inst 6: bwd_fma_placed | path#6: gw[2] += go[0]*x[6]*y[6], gx[6] += w[2]*go[0]*y[6], gy[6] += w[2]*go[0]*x[6]
            atomicAdd(&grad_x[gx_base + (index_t)384 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)384 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)384 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)384 + (index_t)u]);
            // inst 7: bwd_fma_placed | path#5: gw[2] += go[0]*x[5]*y[5], gx[5] += w[2]*go[0]*y[5], gy[5] += w[2]*go[0]*x[5]
            atomicAdd(&grad_x[gx_base + (index_t)320 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)320 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)320 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)320 + (index_t)u]);
            // inst 8: bwd_fma_placed | path#4: gw[2] += go[0]*x[4]*y[4], gx[4] += w[2]*go[0]*y[4], gy[4] += w[2]*go[0]*x[4]
            atomicAdd(&grad_x[gx_base + (index_t)256 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)256 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)256 + (index_t)u], scalar_t(0.4472135901451111) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)256 + (index_t)u]);
            // inst 9: release | last use at scheduled path position 5
            // inst 10: load | placement=register accesses=6 paths=3 lifetime=9
            r0 = w[w_base + (index_t)64 + (index_t)u];
            // inst 11: bwd_fma_placed | path#3: gw[1] += go[0]*x[3]*y[3], gx[3] += w[1]*go[0]*y[3], gy[3] += w[1]*go[0]*x[3]
            atomicAdd(&grad_x[gx_base + (index_t)192 + (index_t)u], scalar_t(0.5773502588272095) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)192 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)192 + (index_t)u], scalar_t(0.5773502588272095) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)192 + (index_t)u]);
            // inst 12: bwd_fma_placed | path#2: gw[1] += go[0]*x[2]*y[2], gx[2] += w[1]*go[0]*y[2], gy[2] += w[1]*go[0]*x[2]
            atomicAdd(&grad_x[gx_base + (index_t)128 + (index_t)u], scalar_t(0.5773502588272095) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)128 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)128 + (index_t)u], scalar_t(0.5773502588272095) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)128 + (index_t)u]);
            // inst 13: bwd_fma_placed | path#15: gw[3] += go[0]*x[15]*y[15], gx[15] += w[3]*go[0]*y[15], gy[15] += w[3]*go[0]*x[15]
            atomicAdd(&grad_x[gx_base + (index_t)960 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)960 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)960 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)960 + (index_t)u]);
            // inst 14: bwd_fma_placed | path#14: gw[3] += go[0]*x[14]*y[14], gx[14] += w[3]*go[0]*y[14], gy[14] += w[3]*go[0]*x[14]
            atomicAdd(&grad_x[gx_base + (index_t)896 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)896 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)896 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)896 + (index_t)u]);
            // inst 15: bwd_fma_placed | path#13: gw[3] += go[0]*x[13]*y[13], gx[13] += w[3]*go[0]*y[13], gy[13] += w[3]*go[0]*x[13]
            atomicAdd(&grad_x[gx_base + (index_t)832 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)832 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)832 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)832 + (index_t)u]);
            // inst 16: bwd_fma_placed | path#12: gw[3] += go[0]*x[12]*y[12], gx[12] += w[3]*go[0]*y[12], gy[12] += w[3]*go[0]*x[12]
            atomicAdd(&grad_x[gx_base + (index_t)768 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)768 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)768 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)768 + (index_t)u]);
            // inst 17: bwd_fma_placed | path#11: gw[3] += go[0]*x[11]*y[11], gx[11] += w[3]*go[0]*y[11], gy[11] += w[3]*go[0]*x[11]
            atomicAdd(&grad_x[gx_base + (index_t)704 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)704 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)704 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)704 + (index_t)u]);
            // inst 18: bwd_fma_placed | path#10: gw[3] += go[0]*x[10]*y[10], gx[10] += w[3]*go[0]*y[10], gy[10] += w[3]*go[0]*x[10]
            atomicAdd(&grad_x[gx_base + (index_t)640 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)640 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)640 + (index_t)u], scalar_t(0.37796446681022644) * (lars_smem[(size_t)0 * (size_t)blockDim.x + (size_t)tid] * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)640 + (index_t)u]);
            // inst 19: release_shared | last use at scheduled path position 13
            // inst 20: bwd_fma_placed | path#1: gw[1] += go[0]*x[1]*y[1], gx[1] += w[1]*go[0]*y[1], gy[1] += w[1]*go[0]*x[1]
            atomicAdd(&grad_x[gx_base + (index_t)64 + (index_t)u], scalar_t(0.5773502588272095) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)64 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)64 + (index_t)u], scalar_t(0.5773502588272095) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)64 + (index_t)u]);
            // inst 21: release | last use at scheduled path position 14
            // inst 22: load | placement=register accesses=2 paths=1 lifetime=1
            r0 = w[w_base + (index_t)0 + (index_t)u];
            // inst 23: bwd_fma_placed | path#0: gw[0] += go[0]*x[0]*y[0], gx[0] += w[0]*go[0]*y[0], gy[0] += w[0]*go[0]*x[0]
            atomicAdd(&grad_x[gx_base + (index_t)0 + (index_t)u], scalar_t(1.0) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * y[y_base + (index_t)0 + (index_t)u]);
            atomicAdd(&grad_y[gy_base + (index_t)0 + (index_t)u], scalar_t(1.0) * (r0 * lars_smem[(size_t)1 * (size_t)blockDim.x + (size_t)tid]) * x[x_base + (index_t)0 + (index_t)u]);
            // inst 24: release_shared | last use at scheduled path position 15
            // inst 25: release | last use at scheduled path position 15
        }
    }
}

static inline bool mul_fits_int32_bwd(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}

static inline bool mul3_fits_int32_bwd(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32_bwd(a, b)) return false;
    return mul_fits_int32_bwd(a * b, c);
}

static inline bool should_use_int32_index_bwd(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32_bwd(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32_bwd(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool go_ok = mul3_fits_int32_bwd(go_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && go_ok;
}

template <typename scalar_t, typename index_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd_typed(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    dim3 block(32);
    dim3 grid(B);
    size_t lars_shared_bytes = (size_t)2 * (size_t)block.x * sizeof(scalar_t);
    uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(
        w, x, y, grad_out, grad_x, grad_y,
        src_idx, dst_idx,
        B, WB, Iw, Ix, Ky, V, U, S);
}

template <typename scalar_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd_auto(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    constexpr bool kUseXSrc = false;
    constexpr bool kUseYSrc = true;
    constexpr bool kUseScatter = false;
    constexpr bool kModeScalarY = false;
    if (should_use_int32_index_bwd(B, WB, Iw, Ix, Ky, V, U, S,
                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {
        launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd_typed<scalar_t, int32_t>(
            w, x, y, grad_out, grad_x, grad_y,
            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
    } else {
        launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd_typed<scalar_t, int64_t>(
            w, x, y, grad_out, grad_x, grad_y,
            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
    }
}

template <typename scalar_t>
void launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd(
    const scalar_t* w,
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* grad_out,
    scalar_t* grad_x,
    scalar_t* grad_y,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    gpuStream_t stream)
{
    launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd_auto<scalar_t>(
        w, x, y, grad_out, grad_x, grad_y,
        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);
}



std::vector<torch::Tensor> launcher_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd(
    torch::Tensor w,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor grad_out,
    torch::Tensor src_idx,
    int64_t V64)
{
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : [B,Ky,U] or [S,Ky,U]
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");

    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");


    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int)src_idx.size(0);
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");



    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros_like(y);

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd", [&] {

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                nullptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    });

    GPU_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd, "uniform1d_lars_bwd_all_inputs_accall_u64_path16_uuuu_xsrc0_ysrc1_scatter0_nogradw_bwd backward LARS fused jit impl");
}
