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

template <typename scalar_t, typename index_t, int TILE_U>
__global__ void stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x0,
    scalar_t* __restrict__ grad_x1,
    int B, int X1, int X0, int V, int U)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared = reinterpret_cast<scalar_t*>(smem);
    scalar_t* grad_x1_shared = x1_shared + (index_t)X1 * (index_t)TILE_U;

    const int b = (int)blockIdx.x;
    const int tile_id = (int)blockIdx.y;
    const int lj = (int)threadIdx.x;
    const int u = tile_id * TILE_U + lj;
    if (b >= B) return;

    for (int a = 0; a < X1; ++a) {
        const index_t idx_global = ((index_t)b * (index_t)X1 + (index_t)a) * (index_t)U + (index_t)u;
        const index_t idx_shared = (index_t)a * (index_t)TILE_U + (index_t)lj;
        x1_shared[idx_shared] = x1[idx_global];
        grad_x1_shared[idx_shared] = scalar_t(0);
    }

    __syncthreads();

    // bwd inst 0: path#17: grad_x1 for out[0] += coeff * grad_out * x0[5]
    const scalar_t base_0 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.3535533905932738) * x0[((index_t)b * (index_t)X0 + (index_t)5) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_0 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_0 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_0 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 1: path#1: grad_x1 for out[0] += coeff * grad_out * x0[1]
    const scalar_t base_1 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.5000000000000001) * x0[((index_t)b * (index_t)X0 + (index_t)1) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += (base_1 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += (base_1 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 2: path#0: grad_x1 for out[0] += coeff * grad_out * x0[0]
    const scalar_t base_2 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(1.0) * x0[((index_t)b * (index_t)X0 + (index_t)0) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += base_2;

    // bwd inst 3: path#31: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_3 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_3 * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_3 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_3 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 4: path#26: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_4 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_4 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_4 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_4 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 5: path#15: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_5 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += (base_5 * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += (base_5 * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 6: path#10: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_6 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += (base_6 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += (base_6 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 7: path#30: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_7 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_7 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_7 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_7 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 8: path#14: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_8 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += (base_8 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += (base_8 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 9: path#28: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_9 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_9 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_9 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_9 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 10: path#12: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_10 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += (base_10 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += (base_10 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 11: path#27: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_11 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_11 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_11 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_11 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 12: path#11: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_12 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += (base_12 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += (base_12 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 13: path#32: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_13 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_13 * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_13 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_13 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 14: path#16: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_14 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += (base_14 * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += (base_14 * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 15: path#29: grad_x1 for out[0] += coeff * grad_out * x0[8]
    const scalar_t base_15 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313785) * x0[((index_t)b * (index_t)X0 + (index_t)8) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_15 * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_15 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_15 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 16: path#13: grad_x1 for out[0] += coeff * grad_out * x0[4]
    const scalar_t base_16 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.18898223650461363) * x0[((index_t)b * (index_t)X0 + (index_t)4) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += (base_16 * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += (base_16 * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 17: path#85: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_17 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1636634176769943) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_17 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_17 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_17 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 18: path#86: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_18 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.08451542547285167) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_18 * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_18 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_18 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 19: path#87: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_19 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1636634176769943) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_19 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_19 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_19 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 20: path#84: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_20 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.21128856368212917) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_20 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_20 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_20 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 21: path#88: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_21 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.21128856368212917) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_21 * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_21 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_21 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 22: path#73: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_22 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.08451542547285167) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_22 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_22 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_22 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 23: path#71: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_23 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1636634176769943) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_23 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_23 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_23 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 24: path#74: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_24 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.1636634176769943) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_24 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_24 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_24 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 25: path#70: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_25 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.21128856368212917) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_25 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_25 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_25 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 26: path#72: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_26 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.21128856368212917) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_26 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_26 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_26 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 27: path#89: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_27 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.13363062095621223) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_27 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_27 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_27 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 28: path#93: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_28 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.13363062095621223) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_28 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_28 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_28 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 29: path#91: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_29 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.26726124191242445) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_29 * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_29 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_29 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 30: path#90: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_30 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.10350983390135315) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_30 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_30 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_30 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 31: path#92: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_31 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.10350983390135315) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_31 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_31 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_31 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 32: path#79: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_32 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.08964214570007953) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_32 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_32 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_32 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 33: path#81: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_33 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.08964214570007953) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_33 * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_33 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_33 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 34: path#80: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_34 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.11952286093343939) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_34 * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_34 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_34 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 35: path#78: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_35 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.1494035761667992) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_35 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_35 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_35 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 36: path#82: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_36 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.1494035761667992) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_36 * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_36 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_36 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 37: path#77: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_37 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.253546276418555) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_37 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_37 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_37 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 38: path#68: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_38 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1267731382092775) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_38 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_38 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_38 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 39: path#76: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_39 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1267731382092775) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_39 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_39 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_39 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 40: path#69: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_40 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.21957751641342) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_40 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_40 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_40 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 41: path#83: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_41 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.21957751641342) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_41 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_41 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_41 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 42: path#75: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_42 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.08451542547285167) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_42 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_42 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_42 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 43: path#63: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_43 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.43915503282683993) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_43 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_43 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_43 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 44: path#62: grad_x1 for out[0] += coeff * grad_out * x0[11]
    const scalar_t base_44 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.253546276418555) * x0[((index_t)b * (index_t)X0 + (index_t)11) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_44 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_44 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_44 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 45: path#66: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_45 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.2070196678027063) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_45 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_45 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_45 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 46: path#64: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_46 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.13363062095621223) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_46 * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_46 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_46 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 47: path#65: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_47 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.26726124191242445) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_47 * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_47 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_47 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 48: path#67: grad_x1 for out[0] += coeff * grad_out * x0[12]
    const scalar_t base_48 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.13363062095621223) * x0[((index_t)b * (index_t)X0 + (index_t)12) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_48 * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_48 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_48 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 49: path#25: grad_x1 for out[0] += coeff * grad_out * x0[7]
    const scalar_t base_49 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.27386127875258304) * x0[((index_t)b * (index_t)X0 + (index_t)7) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_49 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_49 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_49 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 50: path#21: grad_x1 for out[0] += coeff * grad_out * x0[7]
    const scalar_t base_50 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.27386127875258304) * x0[((index_t)b * (index_t)X0 + (index_t)7) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_50 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_50 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_50 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 51: path#23: grad_x1 for out[0] += coeff * grad_out * x0[7]
    const scalar_t base_51 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.27386127875258304) * x0[((index_t)b * (index_t)X0 + (index_t)7) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_51 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_51 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_51 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 52: path#22: grad_x1 for out[0] += coeff * grad_out * x0[7]
    const scalar_t base_52 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.27386127875258304) * x0[((index_t)b * (index_t)X0 + (index_t)7) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_52 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_52 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_52 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 53: path#24: grad_x1 for out[0] += coeff * grad_out * x0[7]
    const scalar_t base_53 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.27386127875258304) * x0[((index_t)b * (index_t)X0 + (index_t)7) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_53 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_53 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_53 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 54: path#9: grad_x1 for out[0] += coeff * grad_out * x0[3]
    const scalar_t base_54 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.223606797749979) * x0[((index_t)b * (index_t)X0 + (index_t)3) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += (base_54 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += (base_54 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 55: path#5: grad_x1 for out[0] += coeff * grad_out * x0[3]
    const scalar_t base_55 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.223606797749979) * x0[((index_t)b * (index_t)X0 + (index_t)3) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += (base_55 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += (base_55 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 56: path#7: grad_x1 for out[0] += coeff * grad_out * x0[3]
    const scalar_t base_56 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.223606797749979) * x0[((index_t)b * (index_t)X0 + (index_t)3) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += (base_56 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += (base_56 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 57: path#6: grad_x1 for out[0] += coeff * grad_out * x0[3]
    const scalar_t base_57 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.223606797749979) * x0[((index_t)b * (index_t)X0 + (index_t)3) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += (base_57 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += (base_57 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 58: path#8: grad_x1 for out[0] += coeff * grad_out * x0[3]
    const scalar_t base_58 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.223606797749979) * x0[((index_t)b * (index_t)X0 + (index_t)3) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += (base_58 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += (base_58 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 59: path#60: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_59 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.05976143046671969) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_59 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_59 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_59 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 60: path#55: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_60 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.05976143046671969) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_60 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_60 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_60 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 61: path#57: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_61 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.2070196678027063) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_61 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_61 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_61 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 62: path#56: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_62 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1889822365046136) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_62 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_62 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_62 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 63: path#58: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_63 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.14638501094227999) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_63 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_63 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_63 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 64: path#61: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_64 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313788) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_64 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_64 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_64 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 65: path#54: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_65 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313788) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_65 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_65 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_65 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 66: path#59: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_66 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1889822365046136) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_66 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_66 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_66 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 67: path#43: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_67 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23145502494313788) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_67 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_67 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)9 * (index_t)TILE_U + (index_t)lj] += ((base_67 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 68: path#38: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_68 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.23145502494313788) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_68 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_68 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)15 * (index_t)TILE_U + (index_t)lj] += ((base_68 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 69: path#41: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_69 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.2070196678027063) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_69 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_69 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_69 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 70: path#39: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_70 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.14638501094227999) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_70 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_70 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_70 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 71: path#44: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_71 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.05976143046671969) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_71 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_71 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_71 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 72: path#37: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_72 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.05976143046671969) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_72 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_72 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_72 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 73: path#40: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_73 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.1889822365046136) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_73 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_73 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_73 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 74: path#42: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_74 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1889822365046136) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_74 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_74 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_74 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 75: path#49: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_75 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.253546276418555) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_75 * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_75 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)12 * (index_t)TILE_U + (index_t)lj] += ((base_75 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 76: path#51: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_76 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1889822365046136) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_76 * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_76 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)14 * (index_t)TILE_U + (index_t)lj] += ((base_76 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 77: path#47: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_77 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.1889822365046136) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_77 * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_77 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)10 * (index_t)TILE_U + (index_t)lj] += ((base_77 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 78: path#48: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_78 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23904572186687875) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_78 * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_78 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)11 * (index_t)TILE_U + (index_t)lj] += ((base_78 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 79: path#50: grad_x1 for out[0] += coeff * grad_out * x0[10]
    const scalar_t base_79 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.23904572186687875) * x0[((index_t)b * (index_t)X0 + (index_t)10) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_79 * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_79 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)13 * (index_t)TILE_U + (index_t)lj] += ((base_79 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 80: path#36: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_80 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.38729833462074176) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_80 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_80 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)4 * (index_t)TILE_U + (index_t)lj] += ((base_80 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 81: path#35: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_81 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.38729833462074176) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_81 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_81 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)5 * (index_t)TILE_U + (index_t)lj] += ((base_81 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 82: path#46: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_82 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.38729833462074176) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_82 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_82 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)7 * (index_t)TILE_U + (index_t)lj] += ((base_82 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 83: path#33: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_83 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.1118033988749895) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_83 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_83 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_83 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 84: path#52: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_84 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.1118033988749895) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_84 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_84 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_84 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 85: path#45: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_85 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.223606797749979) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_85 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_85 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)6 * (index_t)TILE_U + (index_t)lj] += ((base_85 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 86: path#34: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_86 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(-0.19364916731037085) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_86 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_86 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_86 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 87: path#53: grad_x1 for out[0] += coeff * grad_out * x0[9]
    const scalar_t base_87 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.19364916731037085) * x0[((index_t)b * (index_t)X0 + (index_t)9) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_87 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_87 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)8 * (index_t)TILE_U + (index_t)lj] += ((base_87 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 88: path#18: grad_x1 for out[0] += coeff * grad_out * x0[6]
    const scalar_t base_88 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.35355339059327373) * x0[((index_t)b * (index_t)X0 + (index_t)6) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_88 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_88 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += ((base_88 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 89: path#19: grad_x1 for out[0] += coeff * grad_out * x0[6]
    const scalar_t base_89 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.35355339059327373) * x0[((index_t)b * (index_t)X0 + (index_t)6) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_89 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_89 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += ((base_89 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 90: path#20: grad_x1 for out[0] += coeff * grad_out * x0[6]
    const scalar_t base_90 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.35355339059327373) * x0[((index_t)b * (index_t)X0 + (index_t)6) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj] += ((base_90 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_90 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += ((base_90 * x1_shared[(index_t)0 * (index_t)TILE_U + (index_t)lj]) * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 91: path#2: grad_x1 for out[0] += coeff * grad_out * x0[2]
    const scalar_t base_91 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.2886751345948129) * x0[((index_t)b * (index_t)X0 + (index_t)2) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += (base_91 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj] += (base_91 * x1_shared[(index_t)1 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 92: path#3: grad_x1 for out[0] += coeff * grad_out * x0[2]
    const scalar_t base_92 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.2886751345948129) * x0[((index_t)b * (index_t)X0 + (index_t)2) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += (base_92 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj] += (base_92 * x1_shared[(index_t)2 * (index_t)TILE_U + (index_t)lj]);

    // bwd inst 93: path#4: grad_x1 for out[0] += coeff * grad_out * x0[2]
    const scalar_t base_93 = grad_out[((index_t)b * (index_t)V + (index_t)0) * (index_t)U + (index_t)u] * scalar_t(0.2886751345948129) * x0[((index_t)b * (index_t)X0 + (index_t)2) * (index_t)U + (index_t)u];
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += (base_93 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);
    grad_x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj] += (base_93 * x1_shared[(index_t)3 * (index_t)TILE_U + (index_t)lj]);

    __syncthreads();

    for (int a = 0; a < X1; ++a) {
        const index_t idx_global = ((index_t)b * (index_t)X1 + (index_t)a) * (index_t)U + (index_t)u;
        const index_t idx_shared = (index_t)a * (index_t)TILE_U + (index_t)lj;
        grad_x1[idx_global] = grad_x1_shared[idx_shared];
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

template <typename scalar_t, typename index_t, int TILE_U>
void launch_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5_typed(
    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1,
    int B, int X1, int X0, int V, int U, gpuStream_t stream)
{
    dim3 block(TILE_U);
    dim3 grid(B, U / TILE_U);
    size_t smem_bytes = (size_t)2 * (size_t)X1 * (size_t)TILE_U * sizeof(scalar_t);
    stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5<scalar_t, index_t, TILE_U><<<grid, block, smem_bytes, stream>>>(
        grad_out, x1, x0, grad_x1, B, X1, X0, V, U);
}

template <typename scalar_t, int TILE_U>
void launch_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5(
    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1,
    int B, int X1, int X0, int V, int U, gpuStream_t stream)
{
    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&
                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&
                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);
    if (use_i32) {
        launch_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5_typed<scalar_t, int32_t, TILE_U>(grad_out, x1, x0, grad_x1, B, X1, X0, V, U, stream);
    } else {
        launch_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5_typed<scalar_t, int64_t, TILE_U>(grad_out, x1, x0, grad_x1, B, X1, X0, V, U, stream);
    }
}

torch::Tensor launcher_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5(torch::Tensor grad_out, torch::Tensor x1, torch::Tensor x0, int64_t V64) {
    TORCH_CHECK(grad_out.is_cuda() && x1.is_cuda() && x0.is_cuda(), "grad_out/x1/x0 must be CUDA/HIP");
    TORCH_CHECK(grad_out.is_contiguous() && x1.is_contiguous() && x0.is_contiguous(), "grad_out/x1/x0 must be contiguous");
    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3, "x1/x0 must be [B,S,U]");
    TORCH_CHECK(grad_out.scalar_type() == x1.scalar_type() && x1.scalar_type() == x0.scalar_type(), "grad_out/x1/x0 dtype mismatch");
    int B = (int)x1.size(0);
    int X1 = (int)x1.size(1);
    int U = (int)x1.size(2);
    int X0 = (int)x0.size(1);
    int V = (int)V64;
    TORCH_CHECK((int)x0.size(0) == B, "x0 batch mismatch");
    TORCH_CHECK((int)x0.size(2) == U, "x0 U mismatch");
    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of TILE_U");
    TORCH_CHECK(grad_out.numel() == (int64_t)B * (int64_t)V * (int64_t)U, "grad_out numel mismatch; expected B*V*U");
    TORCH_CHECK(U == 224, "U mismatch for generated STC backward kernel");
    TORCH_CHECK(V == 1, "V mismatch for generated STC backward kernel");
    auto grad_x1 = torch::zeros_like(x1);
    GPU_Guard device_guard(x1.device());
    gpuStream_t stream = getCurrentGPUStream(x1.device().index());
    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5", [&] {
        launch_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5<scalar_t, 32>((const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (const scalar_t*)x1.data_ptr<scalar_t>(),
            (const scalar_t*)x0.data_ptr<scalar_t>(),
            (scalar_t*)grad_x1.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);
    });
    GPU_KERNEL_LAUNCH_CHECK();
    return grad_x1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &launcher_stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5, "stc_lars_u1d_bwd_x1_filejit_cd8a354ca154145d_u224_path94_maxlen5 STC backward x1-only jit impl");
}