#include <stdint.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void uniform1d_codegen_two_warp_vgroup_path16_u224_fwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x_all,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V, int U)
{
    int b_global = (int)blockIdx.x;
    int ublk     = (int)blockIdx.y;
    if (b_global >= B) return;
    int b = b_list ? b_list[b_global] : b_global;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= 2) return;

    int u = (ublk << 5) + lane;
    if (u >= U) return;

    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t y_base = (int64_t)b * Ky;

    if (warp == 0) {
        // preload w(i, u)
        scalar_t wi_0 = w[((int64_t)b * Iw + 0) * (int64_t)U + u];
        scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
        scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
        scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];

        // preload x(j, u)
        scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];

        // preload y(k)
        scalar_t yk_0 = y[y_base + 0];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];

        // per-v accumulation
        scalar_t sum_v_0 = scalar_t(0);
        sum_v_0 += wi_0 * xj_0 * yk_0;

        scalar_t sum_v_2 = scalar_t(0);
        sum_v_2 += wi_1 * xj_0 * yk_2;

        scalar_t sum_v_4 = scalar_t(0);
        sum_v_4 += wi_2 * xj_0 * yk_4;

        scalar_t sum_v_6 = scalar_t(0);
        sum_v_6 += wi_2 * xj_0 * yk_6;

        scalar_t sum_v_8 = scalar_t(0);
        sum_v_8 += wi_2 * xj_0 * yk_8;

        scalar_t sum_v_10 = scalar_t(0);
        sum_v_10 += wi_3 * xj_0 * yk_10;

        scalar_t sum_v_12 = scalar_t(0);
        sum_v_12 += wi_3 * xj_0 * yk_12;

        scalar_t sum_v_14 = scalar_t(0);
        sum_v_14 += wi_3 * xj_0 * yk_14;

        // writeback
        atomicAdd(&out[((int64_t)dst * V + 0) * (int64_t)U + u], sum_v_0);
        atomicAdd(&out[((int64_t)dst * V + 2) * (int64_t)U + u], sum_v_2);
        atomicAdd(&out[((int64_t)dst * V + 4) * (int64_t)U + u], sum_v_4);
        atomicAdd(&out[((int64_t)dst * V + 6) * (int64_t)U + u], sum_v_6);
        atomicAdd(&out[((int64_t)dst * V + 8) * (int64_t)U + u], sum_v_8);
        atomicAdd(&out[((int64_t)dst * V + 10) * (int64_t)U + u], sum_v_10);
        atomicAdd(&out[((int64_t)dst * V + 12) * (int64_t)U + u], sum_v_12);
        atomicAdd(&out[((int64_t)dst * V + 14) * (int64_t)U + u], sum_v_14);
    }

    if (warp == 1) {
        // preload w(i, u)
        scalar_t wi_1 = w[((int64_t)b * Iw + 1) * (int64_t)U + u];
        scalar_t wi_2 = w[((int64_t)b * Iw + 2) * (int64_t)U + u];
        scalar_t wi_3 = w[((int64_t)b * Iw + 3) * (int64_t)U + u];

        // preload x(j, u)
        scalar_t xj_0 = x_all[((int64_t)src * Ix + 0) * (int64_t)U + u];

        // preload y(k)
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_15 = y[y_base + 15];

        // per-v accumulation
        scalar_t sum_v_1 = scalar_t(0);
        sum_v_1 += wi_1 * xj_0 * yk_1;

        scalar_t sum_v_3 = scalar_t(0);
        sum_v_3 += wi_1 * xj_0 * yk_3;

        scalar_t sum_v_5 = scalar_t(0);
        sum_v_5 += wi_2 * xj_0 * yk_5;

        scalar_t sum_v_7 = scalar_t(0);
        sum_v_7 += wi_2 * xj_0 * yk_7;

        scalar_t sum_v_9 = scalar_t(0);
        sum_v_9 += wi_3 * xj_0 * yk_9;

        scalar_t sum_v_11 = scalar_t(0);
        sum_v_11 += wi_3 * xj_0 * yk_11;

        scalar_t sum_v_13 = scalar_t(0);
        sum_v_13 += wi_3 * xj_0 * yk_13;

        scalar_t sum_v_15 = scalar_t(0);
        sum_v_15 += wi_3 * xj_0 * yk_15;

        // writeback
        atomicAdd(&out[((int64_t)dst * V + 1) * (int64_t)U + u], sum_v_1);
        atomicAdd(&out[((int64_t)dst * V + 3) * (int64_t)U + u], sum_v_3);
        atomicAdd(&out[((int64_t)dst * V + 5) * (int64_t)U + u], sum_v_5);
        atomicAdd(&out[((int64_t)dst * V + 7) * (int64_t)U + u], sum_v_7);
        atomicAdd(&out[((int64_t)dst * V + 9) * (int64_t)U + u], sum_v_9);
        atomicAdd(&out[((int64_t)dst * V + 11) * (int64_t)U + u], sum_v_11);
        atomicAdd(&out[((int64_t)dst * V + 13) * (int64_t)U + u], sum_v_13);
        atomicAdd(&out[((int64_t)dst * V + 15) * (int64_t)U + u], sum_v_15);
    }

}

// launcher helper
template <typename scalar_t>
void launch_uniform1d_codegen_two_warp_vgroup_path16_u224_fwd(
    const scalar_t* w,
    const scalar_t* x_all,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int Iw, int Ix, int Ky, int V, int U,
    cudaStream_t stream)
{
    dim3 block(64);  // 2 warps
    dim3 grid(B, (U + 31) / 32);
    uniform1d_codegen_two_warp_vgroup_path16_u224_fwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, out, src_idx, dst_idx, b_list,
        B, Iw, Ix, Ky, V, U);
}