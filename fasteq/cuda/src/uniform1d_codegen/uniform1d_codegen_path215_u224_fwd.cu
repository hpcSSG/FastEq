#include <stdint.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void uniform1d_codegen_two_warp_vgroup_path215_u224_fwd(
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ x_all,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ src_idx,
    const int32_t* __restrict__ dst_idx,
    const int32_t* __restrict__ b_list,
    int B, int Iw, int Ix, int Ky, int V)
{
    int b_global = (int)blockIdx.x;
    if (b_global >= B) return;
    int b = b_list ? b_list[b_global] : b_global;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= 2) return;

    int src = src_idx[b];
    int dst = dst_idx[b];

    int64_t w_base = (int64_t)b   * Iw * 32;
    int64_t x_base = (int64_t)src * Ix * 32;
    int64_t y_base = (int64_t)b   * Ky;
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    if (warp == 0) {
        // preload w(i)
        scalar_t wi_7 = w[w_base + 7LL * 32 + lane];
        scalar_t wi_16 = w[w_base + 16LL * 32 + lane];
        scalar_t wi_12 = w[w_base + 12LL * 32 + lane];
        scalar_t wi_15 = w[w_base + 15LL * 32 + lane];
        scalar_t wi_6 = w[w_base + 6LL * 32 + lane];
        scalar_t wi_11 = w[w_base + 11LL * 32 + lane];
        scalar_t wi_10 = w[w_base + 10LL * 32 + lane];
        scalar_t wi_1 = w[w_base + 1LL * 32 + lane];
        scalar_t wi_5 = w[w_base + 5LL * 32 + lane];
        scalar_t wi_14 = w[w_base + 14LL * 32 + lane];
        scalar_t wi_9 = w[w_base + 9LL * 32 + lane];
        scalar_t wi_4 = w[w_base + 4LL * 32 + lane];
        scalar_t wi_3 = w[w_base + 3LL * 32 + lane];
        scalar_t wi_8 = w[w_base + 8LL * 32 + lane];
        scalar_t wi_13 = w[w_base + 13LL * 32 + lane];

        // preload x(j)
        scalar_t xj_4 = x_all[x_base + 4LL * 32 + lane];
        scalar_t xj_5 = x_all[x_base + 5LL * 32 + lane];
        scalar_t xj_6 = x_all[x_base + 6LL * 32 + lane];
        scalar_t xj_7 = x_all[x_base + 7LL * 32 + lane];
        scalar_t xj_8 = x_all[x_base + 8LL * 32 + lane];
        scalar_t xj_1 = x_all[x_base + 1LL * 32 + lane];
        scalar_t xj_2 = x_all[x_base + 2LL * 32 + lane];
        scalar_t xj_3 = x_all[x_base + 3LL * 32 + lane];
        scalar_t xj_0 = x_all[x_base + 0LL * 32 + lane];

        // preload y(k)
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_15 = scalar_t(0);
        sum_v_15 += scalar_t(-0.119522861f) * wi_7 * xj_4 * yk_13;
        sum_v_15 += scalar_t(-0.46291005f) * wi_7 * xj_4 * yk_15;
        sum_v_15 += scalar_t(-0.292770022f) * wi_7 * xj_5 * yk_12;
        sum_v_15 += scalar_t(-0.377964473f) * wi_7 * xj_5 * yk_14;
        sum_v_15 += scalar_t(0.414039336f) * wi_7 * xj_6 * yk_11;
        sum_v_15 += scalar_t(0.377964473f) * wi_7 * xj_7 * yk_10;
        sum_v_15 += scalar_t(0.46291005f) * wi_7 * xj_8 * yk_9;
        sum_v_15 += scalar_t(0.119522861f) * wi_7 * xj_8 * yk_11;

        scalar_t sum_v_16 = scalar_t(0);
        sum_v_16 += scalar_t(0.377964473f) * wi_7 * xj_4 * yk_10;
        sum_v_16 += scalar_t(0.478091444f) * wi_7 * xj_5 * yk_11;
        sum_v_16 += scalar_t(0.507092553f) * wi_7 * xj_6 * yk_12;
        sum_v_16 += scalar_t(0.478091444f) * wi_7 * xj_7 * yk_13;
        sum_v_16 += scalar_t(0.377964473f) * wi_7 * xj_8 * yk_14;

        scalar_t sum_v_66 = scalar_t(0);
        sum_v_66 += scalar_t(0.447213595f) * wi_16 * xj_4 * yk_13;
        sum_v_66 += scalar_t(0.288675135f) * wi_16 * xj_4 * yk_15;
        sum_v_66 += scalar_t(0.182574186f) * wi_16 * xj_5 * yk_12;
        sum_v_66 += scalar_t(-0.353553391f) * wi_16 * xj_5 * yk_14;
        sum_v_66 += scalar_t(0.387298335f) * wi_16 * xj_6 * yk_11;
        sum_v_66 += scalar_t(0.353553391f) * wi_16 * xj_7 * yk_10;
        sum_v_66 += scalar_t(-0.288675135f) * wi_16 * xj_8 * yk_9;
        sum_v_66 += scalar_t(-0.447213595f) * wi_16 * xj_8 * yk_11;

        scalar_t sum_v_64 = scalar_t(0);
        sum_v_64 += scalar_t(-0.288675135f) * wi_16 * xj_4 * yk_13;
        sum_v_64 += scalar_t(0.456435465f) * wi_16 * xj_5 * yk_14;
        sum_v_64 += scalar_t(-0.645497224f) * wi_16 * xj_6 * yk_9;
        sum_v_64 += scalar_t(0.456435465f) * wi_16 * xj_7 * yk_10;
        sum_v_64 += scalar_t(-0.288675135f) * wi_16 * xj_8 * yk_11;

        scalar_t sum_v_41 = scalar_t(0);
        sum_v_41 += scalar_t(0.46291005f) * wi_12 * xj_4 * yk_5;
        sum_v_41 += scalar_t(0.46291005f) * wi_12 * xj_5 * yk_4;
        sum_v_41 += scalar_t(0.267261242f) * wi_12 * xj_6 * yk_7;
        sum_v_41 += scalar_t(0.267261242f) * wi_12 * xj_7 * yk_6;
        sum_v_41 += scalar_t(0.46291005f) * wi_12 * xj_7 * yk_8;
        sum_v_41 += scalar_t(0.46291005f) * wi_12 * xj_8 * yk_7;

        scalar_t sum_v_40 = scalar_t(0);
        sum_v_40 += scalar_t(-0.534522484f) * wi_12 * xj_4 * yk_4;
        sum_v_40 += scalar_t(0.267261242f) * wi_12 * xj_5 * yk_5;
        sum_v_40 += scalar_t(0.534522484f) * wi_12 * xj_6 * yk_6;
        sum_v_40 += scalar_t(0.267261242f) * wi_12 * xj_7 * yk_7;
        sum_v_40 += scalar_t(-0.534522484f) * wi_12 * xj_8 * yk_8;

        scalar_t sum_v_38 = scalar_t(0);
        sum_v_38 += scalar_t(-0.534522484f) * wi_12 * xj_4 * yk_6;
        sum_v_38 += scalar_t(0.46291005f) * wi_12 * xj_5 * yk_7;
        sum_v_38 += scalar_t(-0.534522484f) * wi_12 * xj_6 * yk_4;
        sum_v_38 += scalar_t(0.46291005f) * wi_12 * xj_7 * yk_5;

        scalar_t sum_v_59 = scalar_t(0);
        sum_v_59 += scalar_t(-0.182574186f) * wi_15 * xj_4 * yk_3;
        sum_v_59 += scalar_t(0.730296743f) * wi_15 * xj_5 * yk_2;
        sum_v_59 += scalar_t(0.632455532f) * wi_15 * xj_6 * yk_1;
        sum_v_59 += scalar_t(0.182574186f) * wi_15 * xj_8 * yk_1;

        scalar_t sum_v_14 = scalar_t(0);
        sum_v_14 += scalar_t(0.547722558f) * wi_6 * xj_4 * yk_1;
        sum_v_14 += scalar_t(-0.316227766f) * wi_6 * xj_6 * yk_3;
        sum_v_14 += scalar_t(0.547722558f) * wi_6 * xj_7 * yk_2;
        sum_v_14 += scalar_t(0.547722558f) * wi_6 * xj_8 * yk_3;

        scalar_t sum_v_60 = scalar_t(0);
        sum_v_60 += scalar_t(-0.447213595f) * wi_15 * xj_5 * yk_1;
        sum_v_60 += scalar_t(0.774596669f) * wi_15 * xj_6 * yk_2;
        sum_v_60 += scalar_t(-0.447213595f) * wi_15 * xj_7 * yk_3;

        scalar_t sum_v_58 = scalar_t(0);
        sum_v_58 += scalar_t(0.577350269f) * wi_15 * xj_4 * yk_2;
        sum_v_58 += scalar_t(0.577350269f) * wi_15 * xj_5 * yk_3;
        sum_v_58 += scalar_t(0.577350269f) * wi_15 * xj_7 * yk_1;

        scalar_t sum_v_65 = scalar_t(0);
        sum_v_65 += scalar_t(-0.577350269f) * wi_16 * xj_4 * yk_12;
        sum_v_65 += scalar_t(0.353553391f) * wi_16 * xj_5 * yk_13;
        sum_v_65 += scalar_t(-0.456435465f) * wi_16 * xj_5 * yk_15;
        sum_v_65 += scalar_t(0.456435465f) * wi_16 * xj_7 * yk_9;
        sum_v_65 += scalar_t(0.353553391f) * wi_16 * xj_7 * yk_11;

        scalar_t sum_v_37 = scalar_t(0);
        sum_v_37 += wi_11 * xj_8 * yk_0;

        scalar_t sum_v_36 = scalar_t(0);
        sum_v_36 += wi_11 * xj_7 * yk_0;

        scalar_t sum_v_34 = scalar_t(0);
        sum_v_34 += wi_11 * xj_5 * yk_0;

        scalar_t sum_v_32 = scalar_t(0);
        sum_v_32 += scalar_t(0.597614305f) * wi_10 * xj_1 * yk_9;
        sum_v_32 += scalar_t(0.15430335f) * wi_10 * xj_1 * yk_11;
        sum_v_32 += scalar_t(0.487950036f) * wi_10 * xj_2 * yk_14;
        sum_v_32 += scalar_t(-0.15430335f) * wi_10 * xj_3 * yk_13;
        sum_v_32 += scalar_t(0.597614305f) * wi_10 * xj_3 * yk_15;

        scalar_t sum_v_29 = scalar_t(0);
        sum_v_29 += scalar_t(-0.377964473f) * wi_10 * xj_1 * yk_12;
        sum_v_29 += scalar_t(-0.487950036f) * wi_10 * xj_1 * yk_14;
        sum_v_29 += scalar_t(0.6172134f) * wi_10 * xj_2 * yk_11;
        sum_v_29 += scalar_t(0.487950036f) * wi_10 * xj_3 * yk_10;

        scalar_t sum_v_30 = scalar_t(0);
        sum_v_30 += scalar_t(0.534522484f) * wi_10 * xj_1 * yk_11;
        sum_v_30 += scalar_t(0.654653671f) * wi_10 * xj_2 * yk_12;
        sum_v_30 += scalar_t(0.534522484f) * wi_10 * xj_3 * yk_13;

        scalar_t sum_v_1 = scalar_t(0);
        sum_v_1 += scalar_t(0.577350269f) * wi_1 * xj_1 * yk_1;
        sum_v_1 += scalar_t(0.577350269f) * wi_1 * xj_2 * yk_2;
        sum_v_1 += scalar_t(0.577350269f) * wi_1 * xj_3 * yk_3;

        scalar_t sum_v_9 = scalar_t(0);
        sum_v_9 += scalar_t(-0.316227766f) * wi_5 * xj_1 * yk_6;
        sum_v_9 += scalar_t(-0.547722558f) * wi_5 * xj_1 * yk_8;
        sum_v_9 += scalar_t(0.547722558f) * wi_5 * xj_2 * yk_5;
        sum_v_9 += scalar_t(0.547722558f) * wi_5 * xj_3 * yk_4;

        scalar_t sum_v_53 = scalar_t(0);
        sum_v_53 += scalar_t(-0.447213595f) * wi_14 * xj_1 * yk_5;
        sum_v_53 += scalar_t(0.774596669f) * wi_14 * xj_2 * yk_6;
        sum_v_53 += scalar_t(-0.447213595f) * wi_14 * xj_3 * yk_7;

        scalar_t sum_v_52 = scalar_t(0);
        sum_v_52 += scalar_t(0.632455532f) * wi_14 * xj_1 * yk_6;
        sum_v_52 += scalar_t(0.182574186f) * wi_14 * xj_1 * yk_8;
        sum_v_52 += scalar_t(0.730296743f) * wi_14 * xj_2 * yk_5;
        sum_v_52 += scalar_t(-0.182574186f) * wi_14 * xj_3 * yk_4;

        scalar_t sum_v_51 = scalar_t(0);
        sum_v_51 += scalar_t(0.577350269f) * wi_14 * xj_1 * yk_7;
        sum_v_51 += scalar_t(0.577350269f) * wi_14 * xj_2 * yk_4;
        sum_v_51 += scalar_t(0.577350269f) * wi_14 * xj_3 * yk_5;

        scalar_t sum_v_50 = scalar_t(0);
        sum_v_50 += scalar_t(0.707106781f) * wi_14 * xj_1 * yk_8;
        sum_v_50 += scalar_t(0.707106781f) * wi_14 * xj_3 * yk_4;

        scalar_t sum_v_27 = scalar_t(0);
        sum_v_27 += scalar_t(-0.707106781f) * wi_9 * xj_1 * yk_1;
        sum_v_27 += scalar_t(0.707106781f) * wi_9 * xj_3 * yk_3;

        scalar_t sum_v_24 = scalar_t(0);
        sum_v_24 += scalar_t(0.707106781f) * wi_9 * xj_1 * yk_2;
        sum_v_24 += scalar_t(0.707106781f) * wi_9 * xj_2 * yk_1;

        scalar_t sum_v_8 = scalar_t(0);
        sum_v_8 += wi_4 * xj_3 * yk_0;

        scalar_t sum_v_6 = scalar_t(0);
        sum_v_6 += wi_4 * xj_1 * yk_0;

        scalar_t sum_v_3 = scalar_t(0);
        sum_v_3 += wi_3 * xj_0 * yk_1;

        scalar_t sum_v_5 = scalar_t(0);
        sum_v_5 += wi_3 * xj_0 * yk_3;

        scalar_t sum_v_19 = scalar_t(0);
        sum_v_19 += wi_8 * xj_0 * yk_5;

        scalar_t sum_v_21 = scalar_t(0);
        sum_v_21 += wi_8 * xj_0 * yk_7;

        scalar_t sum_v_43 = scalar_t(0);
        sum_v_43 += wi_13 * xj_0 * yk_9;

        scalar_t sum_v_45 = scalar_t(0);
        sum_v_45 += wi_13 * xj_0 * yk_11;

        scalar_t sum_v_47 = scalar_t(0);
        sum_v_47 += wi_13 * xj_0 * yk_13;

        scalar_t sum_v_49 = scalar_t(0);
        sum_v_49 += wi_13 * xj_0 * yk_15;

        // writeback
        atomicAdd(&out[o_base + (15LL << 5)], sum_v_15);
        atomicAdd(&out[o_base + (16LL << 5)], sum_v_16);
        atomicAdd(&out[o_base + (66LL << 5)], sum_v_66);
        atomicAdd(&out[o_base + (64LL << 5)], sum_v_64);
        atomicAdd(&out[o_base + (41LL << 5)], sum_v_41);
        atomicAdd(&out[o_base + (40LL << 5)], sum_v_40);
        atomicAdd(&out[o_base + (38LL << 5)], sum_v_38);
        atomicAdd(&out[o_base + (59LL << 5)], sum_v_59);
        atomicAdd(&out[o_base + (14LL << 5)], sum_v_14);
        atomicAdd(&out[o_base + (60LL << 5)], sum_v_60);
        atomicAdd(&out[o_base + (58LL << 5)], sum_v_58);
        atomicAdd(&out[o_base + (65LL << 5)], sum_v_65);
        atomicAdd(&out[o_base + (37LL << 5)], sum_v_37);
        atomicAdd(&out[o_base + (36LL << 5)], sum_v_36);
        atomicAdd(&out[o_base + (34LL << 5)], sum_v_34);
        atomicAdd(&out[o_base + (32LL << 5)], sum_v_32);
        atomicAdd(&out[o_base + (29LL << 5)], sum_v_29);
        atomicAdd(&out[o_base + (30LL << 5)], sum_v_30);
        atomicAdd(&out[o_base + (1LL << 5)], sum_v_1);
        atomicAdd(&out[o_base + (9LL << 5)], sum_v_9);
        atomicAdd(&out[o_base + (53LL << 5)], sum_v_53);
        atomicAdd(&out[o_base + (52LL << 5)], sum_v_52);
        atomicAdd(&out[o_base + (51LL << 5)], sum_v_51);
        atomicAdd(&out[o_base + (50LL << 5)], sum_v_50);
        atomicAdd(&out[o_base + (27LL << 5)], sum_v_27);
        atomicAdd(&out[o_base + (24LL << 5)], sum_v_24);
        atomicAdd(&out[o_base + (8LL << 5)], sum_v_8);
        atomicAdd(&out[o_base + (6LL << 5)], sum_v_6);
        atomicAdd(&out[o_base + (3LL << 5)], sum_v_3);
        atomicAdd(&out[o_base + (5LL << 5)], sum_v_5);
        atomicAdd(&out[o_base + (19LL << 5)], sum_v_19);
        atomicAdd(&out[o_base + (21LL << 5)], sum_v_21);
        atomicAdd(&out[o_base + (43LL << 5)], sum_v_43);
        atomicAdd(&out[o_base + (45LL << 5)], sum_v_45);
        atomicAdd(&out[o_base + (47LL << 5)], sum_v_47);
        atomicAdd(&out[o_base + (49LL << 5)], sum_v_49);
    }

    if (warp == 1) {
        // preload w(i)
        scalar_t wi_7 = w[w_base + 7LL * 32 + lane];
        scalar_t wi_16 = w[w_base + 16LL * 32 + lane];
        scalar_t wi_12 = w[w_base + 12LL * 32 + lane];
        scalar_t wi_2 = w[w_base + 2LL * 32 + lane];
        scalar_t wi_6 = w[w_base + 6LL * 32 + lane];
        scalar_t wi_15 = w[w_base + 15LL * 32 + lane];
        scalar_t wi_11 = w[w_base + 11LL * 32 + lane];
        scalar_t wi_10 = w[w_base + 10LL * 32 + lane];
        scalar_t wi_9 = w[w_base + 9LL * 32 + lane];
        scalar_t wi_5 = w[w_base + 5LL * 32 + lane];
        scalar_t wi_14 = w[w_base + 14LL * 32 + lane];
        scalar_t wi_4 = w[w_base + 4LL * 32 + lane];
        scalar_t wi_0 = w[w_base + 0LL * 32 + lane];
        scalar_t wi_3 = w[w_base + 3LL * 32 + lane];
        scalar_t wi_8 = w[w_base + 8LL * 32 + lane];
        scalar_t wi_13 = w[w_base + 13LL * 32 + lane];

        // preload x(j)
        scalar_t xj_4 = x_all[x_base + 4LL * 32 + lane];
        scalar_t xj_5 = x_all[x_base + 5LL * 32 + lane];
        scalar_t xj_6 = x_all[x_base + 6LL * 32 + lane];
        scalar_t xj_7 = x_all[x_base + 7LL * 32 + lane];
        scalar_t xj_8 = x_all[x_base + 8LL * 32 + lane];
        scalar_t xj_1 = x_all[x_base + 1LL * 32 + lane];
        scalar_t xj_2 = x_all[x_base + 2LL * 32 + lane];
        scalar_t xj_3 = x_all[x_base + 3LL * 32 + lane];
        scalar_t xj_0 = x_all[x_base + 0LL * 32 + lane];

        // preload y(k)
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_17 = scalar_t(0);
        sum_v_17 += scalar_t(0.46291005f) * wi_7 * xj_4 * yk_9;
        sum_v_17 += scalar_t(-0.119522861f) * wi_7 * xj_4 * yk_11;
        sum_v_17 += scalar_t(0.377964473f) * wi_7 * xj_5 * yk_10;
        sum_v_17 += scalar_t(0.414039336f) * wi_7 * xj_6 * yk_13;
        sum_v_17 += scalar_t(-0.292770022f) * wi_7 * xj_7 * yk_12;
        sum_v_17 += scalar_t(0.377964473f) * wi_7 * xj_7 * yk_14;
        sum_v_17 += scalar_t(-0.119522861f) * wi_7 * xj_8 * yk_13;
        sum_v_17 += scalar_t(0.46291005f) * wi_7 * xj_8 * yk_15;

        scalar_t sum_v_68 = scalar_t(0);
        sum_v_68 += scalar_t(-0.288675135f) * wi_16 * xj_4 * yk_9;
        sum_v_68 += scalar_t(0.447213595f) * wi_16 * xj_4 * yk_11;
        sum_v_68 += scalar_t(0.353553391f) * wi_16 * xj_5 * yk_10;
        sum_v_68 += scalar_t(0.387298335f) * wi_16 * xj_6 * yk_13;
        sum_v_68 += scalar_t(0.182574186f) * wi_16 * xj_7 * yk_12;
        sum_v_68 += scalar_t(0.353553391f) * wi_16 * xj_7 * yk_14;
        sum_v_68 += scalar_t(0.447213595f) * wi_16 * xj_8 * yk_13;
        sum_v_68 += scalar_t(-0.288675135f) * wi_16 * xj_8 * yk_15;

        scalar_t sum_v_67 = scalar_t(0);
        sum_v_67 += scalar_t(-0.577350269f) * wi_16 * xj_4 * yk_10;
        sum_v_67 += scalar_t(0.182574186f) * wi_16 * xj_5 * yk_11;
        sum_v_67 += scalar_t(0.516397779f) * wi_16 * xj_6 * yk_12;
        sum_v_67 += scalar_t(0.182574186f) * wi_16 * xj_7 * yk_13;
        sum_v_67 += scalar_t(-0.577350269f) * wi_16 * xj_8 * yk_14;

        scalar_t sum_v_70 = scalar_t(0);
        sum_v_70 += scalar_t(0.288675135f) * wi_16 * xj_4 * yk_11;
        sum_v_70 += scalar_t(-0.456435465f) * wi_16 * xj_5 * yk_10;
        sum_v_70 += scalar_t(-0.645497224f) * wi_16 * xj_6 * yk_15;
        sum_v_70 += scalar_t(0.456435465f) * wi_16 * xj_7 * yk_14;
        sum_v_70 += scalar_t(-0.288675135f) * wi_16 * xj_8 * yk_13;

        scalar_t sum_v_39 = scalar_t(0);
        sum_v_39 += scalar_t(0.46291005f) * wi_12 * xj_4 * yk_7;
        sum_v_39 += scalar_t(0.267261242f) * wi_12 * xj_5 * yk_6;
        sum_v_39 += scalar_t(-0.46291005f) * wi_12 * xj_5 * yk_8;
        sum_v_39 += scalar_t(0.267261242f) * wi_12 * xj_6 * yk_5;
        sum_v_39 += scalar_t(0.46291005f) * wi_12 * xj_7 * yk_4;
        sum_v_39 += scalar_t(-0.46291005f) * wi_12 * xj_8 * yk_5;

        scalar_t sum_v_2 = scalar_t(0);
        sum_v_2 += scalar_t(0.447213595f) * wi_2 * xj_4 * yk_4;
        sum_v_2 += scalar_t(0.447213595f) * wi_2 * xj_5 * yk_5;
        sum_v_2 += scalar_t(0.447213595f) * wi_2 * xj_6 * yk_6;
        sum_v_2 += scalar_t(0.447213595f) * wi_2 * xj_7 * yk_7;
        sum_v_2 += scalar_t(0.447213595f) * wi_2 * xj_8 * yk_8;

        scalar_t sum_v_42 = scalar_t(0);
        sum_v_42 += scalar_t(-0.46291005f) * wi_12 * xj_5 * yk_5;
        sum_v_42 += scalar_t(-0.534522484f) * wi_12 * xj_6 * yk_8;
        sum_v_42 += scalar_t(0.46291005f) * wi_12 * xj_7 * yk_7;
        sum_v_42 += scalar_t(-0.534522484f) * wi_12 * xj_8 * yk_6;

        scalar_t sum_v_12 = scalar_t(0);
        sum_v_12 += scalar_t(0.547722558f) * wi_6 * xj_4 * yk_3;
        sum_v_12 += scalar_t(0.547722558f) * wi_6 * xj_5 * yk_2;
        sum_v_12 += scalar_t(-0.316227766f) * wi_6 * xj_6 * yk_1;
        sum_v_12 += scalar_t(-0.547722558f) * wi_6 * xj_8 * yk_1;

        scalar_t sum_v_61 = scalar_t(0);
        sum_v_61 += scalar_t(-0.182574186f) * wi_15 * xj_4 * yk_1;
        sum_v_61 += scalar_t(0.632455532f) * wi_15 * xj_6 * yk_3;
        sum_v_61 += scalar_t(0.730296743f) * wi_15 * xj_7 * yk_2;
        sum_v_61 += scalar_t(-0.182574186f) * wi_15 * xj_8 * yk_3;

        scalar_t sum_v_13 = scalar_t(0);
        sum_v_13 += scalar_t(0.547722558f) * wi_6 * xj_5 * yk_1;
        sum_v_13 += scalar_t(0.632455532f) * wi_6 * xj_6 * yk_2;
        sum_v_13 += scalar_t(0.547722558f) * wi_6 * xj_7 * yk_3;

        scalar_t sum_v_69 = scalar_t(0);
        sum_v_69 += scalar_t(0.456435465f) * wi_16 * xj_5 * yk_9;
        sum_v_69 += scalar_t(-0.353553391f) * wi_16 * xj_5 * yk_11;
        sum_v_69 += scalar_t(0.353553391f) * wi_16 * xj_7 * yk_13;
        sum_v_69 += scalar_t(0.456435465f) * wi_16 * xj_7 * yk_15;
        sum_v_69 += scalar_t(-0.577350269f) * wi_16 * xj_8 * yk_12;

        scalar_t sum_v_62 = scalar_t(0);
        sum_v_62 += scalar_t(-0.577350269f) * wi_15 * xj_5 * yk_1;
        sum_v_62 += scalar_t(0.577350269f) * wi_15 * xj_7 * yk_3;
        sum_v_62 += scalar_t(0.577350269f) * wi_15 * xj_8 * yk_2;

        scalar_t sum_v_63 = scalar_t(0);
        sum_v_63 += scalar_t(-0.707106781f) * wi_15 * xj_4 * yk_1;
        sum_v_63 += scalar_t(0.707106781f) * wi_15 * xj_8 * yk_3;

        scalar_t sum_v_57 = scalar_t(0);
        sum_v_57 += scalar_t(0.707106781f) * wi_15 * xj_4 * yk_3;
        sum_v_57 += scalar_t(0.707106781f) * wi_15 * xj_8 * yk_1;

        scalar_t sum_v_35 = scalar_t(0);
        sum_v_35 += wi_11 * xj_6 * yk_0;

        scalar_t sum_v_33 = scalar_t(0);
        sum_v_33 += wi_11 * xj_4 * yk_0;

        scalar_t sum_v_28 = scalar_t(0);
        sum_v_28 += scalar_t(-0.15430335f) * wi_10 * xj_1 * yk_13;
        sum_v_28 += scalar_t(-0.597614305f) * wi_10 * xj_1 * yk_15;
        sum_v_28 += scalar_t(0.487950036f) * wi_10 * xj_2 * yk_10;
        sum_v_28 += scalar_t(0.597614305f) * wi_10 * xj_3 * yk_9;
        sum_v_28 += scalar_t(-0.15430335f) * wi_10 * xj_3 * yk_11;

        scalar_t sum_v_31 = scalar_t(0);
        sum_v_31 += scalar_t(0.487950036f) * wi_10 * xj_1 * yk_10;
        sum_v_31 += scalar_t(0.6172134f) * wi_10 * xj_2 * yk_13;
        sum_v_31 += scalar_t(-0.377964473f) * wi_10 * xj_3 * yk_12;
        sum_v_31 += scalar_t(0.487950036f) * wi_10 * xj_3 * yk_14;

        scalar_t sum_v_25 = scalar_t(0);
        sum_v_25 += scalar_t(-0.40824829f) * wi_9 * xj_1 * yk_1;
        sum_v_25 += scalar_t(0.816496581f) * wi_9 * xj_2 * yk_2;
        sum_v_25 += scalar_t(-0.40824829f) * wi_9 * xj_3 * yk_3;

        scalar_t sum_v_11 = scalar_t(0);
        sum_v_11 += scalar_t(0.547722558f) * wi_5 * xj_1 * yk_4;
        sum_v_11 += scalar_t(0.547722558f) * wi_5 * xj_2 * yk_7;
        sum_v_11 += scalar_t(-0.316227766f) * wi_5 * xj_3 * yk_6;
        sum_v_11 += scalar_t(0.547722558f) * wi_5 * xj_3 * yk_8;

        scalar_t sum_v_10 = scalar_t(0);
        sum_v_10 += scalar_t(0.547722558f) * wi_5 * xj_1 * yk_5;
        sum_v_10 += scalar_t(0.632455532f) * wi_5 * xj_2 * yk_6;
        sum_v_10 += scalar_t(0.547722558f) * wi_5 * xj_3 * yk_7;

        scalar_t sum_v_54 = scalar_t(0);
        sum_v_54 += scalar_t(-0.182574186f) * wi_14 * xj_1 * yk_4;
        sum_v_54 += scalar_t(0.730296743f) * wi_14 * xj_2 * yk_7;
        sum_v_54 += scalar_t(0.632455532f) * wi_14 * xj_3 * yk_6;
        sum_v_54 += scalar_t(-0.182574186f) * wi_14 * xj_3 * yk_8;

        scalar_t sum_v_55 = scalar_t(0);
        sum_v_55 += scalar_t(-0.577350269f) * wi_14 * xj_1 * yk_5;
        sum_v_55 += scalar_t(0.577350269f) * wi_14 * xj_2 * yk_8;
        sum_v_55 += scalar_t(0.577350269f) * wi_14 * xj_3 * yk_7;

        scalar_t sum_v_56 = scalar_t(0);
        sum_v_56 += scalar_t(-0.707106781f) * wi_14 * xj_1 * yk_4;
        sum_v_56 += scalar_t(0.707106781f) * wi_14 * xj_3 * yk_8;

        scalar_t sum_v_23 = scalar_t(0);
        sum_v_23 += scalar_t(0.707106781f) * wi_9 * xj_1 * yk_3;
        sum_v_23 += scalar_t(0.707106781f) * wi_9 * xj_3 * yk_1;

        scalar_t sum_v_26 = scalar_t(0);
        sum_v_26 += scalar_t(0.707106781f) * wi_9 * xj_2 * yk_3;
        sum_v_26 += scalar_t(0.707106781f) * wi_9 * xj_3 * yk_2;

        scalar_t sum_v_7 = scalar_t(0);
        sum_v_7 += wi_4 * xj_2 * yk_0;

        scalar_t sum_v_0 = scalar_t(0);
        sum_v_0 += wi_0 * xj_0 * yk_0;

        scalar_t sum_v_4 = scalar_t(0);
        sum_v_4 += wi_3 * xj_0 * yk_2;

        scalar_t sum_v_18 = scalar_t(0);
        sum_v_18 += wi_8 * xj_0 * yk_4;

        scalar_t sum_v_20 = scalar_t(0);
        sum_v_20 += wi_8 * xj_0 * yk_6;

        scalar_t sum_v_22 = scalar_t(0);
        sum_v_22 += wi_8 * xj_0 * yk_8;

        scalar_t sum_v_44 = scalar_t(0);
        sum_v_44 += wi_13 * xj_0 * yk_10;

        scalar_t sum_v_46 = scalar_t(0);
        sum_v_46 += wi_13 * xj_0 * yk_12;

        scalar_t sum_v_48 = scalar_t(0);
        sum_v_48 += wi_13 * xj_0 * yk_14;

        // writeback
        atomicAdd(&out[o_base + (17LL << 5)], sum_v_17);
        atomicAdd(&out[o_base + (68LL << 5)], sum_v_68);
        atomicAdd(&out[o_base + (67LL << 5)], sum_v_67);
        atomicAdd(&out[o_base + (70LL << 5)], sum_v_70);
        atomicAdd(&out[o_base + (39LL << 5)], sum_v_39);
        atomicAdd(&out[o_base + (2LL << 5)], sum_v_2);
        atomicAdd(&out[o_base + (42LL << 5)], sum_v_42);
        atomicAdd(&out[o_base + (12LL << 5)], sum_v_12);
        atomicAdd(&out[o_base + (61LL << 5)], sum_v_61);
        atomicAdd(&out[o_base + (13LL << 5)], sum_v_13);
        atomicAdd(&out[o_base + (69LL << 5)], sum_v_69);
        atomicAdd(&out[o_base + (62LL << 5)], sum_v_62);
        atomicAdd(&out[o_base + (63LL << 5)], sum_v_63);
        atomicAdd(&out[o_base + (57LL << 5)], sum_v_57);
        atomicAdd(&out[o_base + (35LL << 5)], sum_v_35);
        atomicAdd(&out[o_base + (33LL << 5)], sum_v_33);
        atomicAdd(&out[o_base + (28LL << 5)], sum_v_28);
        atomicAdd(&out[o_base + (31LL << 5)], sum_v_31);
        atomicAdd(&out[o_base + (25LL << 5)], sum_v_25);
        atomicAdd(&out[o_base + (11LL << 5)], sum_v_11);
        atomicAdd(&out[o_base + (10LL << 5)], sum_v_10);
        atomicAdd(&out[o_base + (54LL << 5)], sum_v_54);
        atomicAdd(&out[o_base + (55LL << 5)], sum_v_55);
        atomicAdd(&out[o_base + (56LL << 5)], sum_v_56);
        atomicAdd(&out[o_base + (23LL << 5)], sum_v_23);
        atomicAdd(&out[o_base + (26LL << 5)], sum_v_26);
        atomicAdd(&out[o_base + (7LL << 5)], sum_v_7);
        atomicAdd(&out[o_base + (0LL << 5)], sum_v_0);
        atomicAdd(&out[o_base + (4LL << 5)], sum_v_4);
        atomicAdd(&out[o_base + (18LL << 5)], sum_v_18);
        atomicAdd(&out[o_base + (20LL << 5)], sum_v_20);
        atomicAdd(&out[o_base + (22LL << 5)], sum_v_22);
        atomicAdd(&out[o_base + (44LL << 5)], sum_v_44);
        atomicAdd(&out[o_base + (46LL << 5)], sum_v_46);
        atomicAdd(&out[o_base + (48LL << 5)], sum_v_48);
    }

}

// launcher helper
template <typename scalar_t>
void launch_uniform1d_codegen_two_warp_vgroup_path215_u224_fwd(
    const scalar_t* w,
    const scalar_t* x_all,
    const scalar_t* y,
    scalar_t* out,
    const int32_t* src_idx,
    const int32_t* dst_idx,
    const int32_t* b_list,
    int B, int Iw, int Ix, int Ky, int V,
    cudaStream_t stream)
{
    dim3 block(64);  // 2 warps
    dim3 grid(B);
    uniform1d_codegen_two_warp_vgroup_path215_u224_fwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, out, src_idx, dst_idx, b_list,
        B, Iw, Ix, Ky, V);
}