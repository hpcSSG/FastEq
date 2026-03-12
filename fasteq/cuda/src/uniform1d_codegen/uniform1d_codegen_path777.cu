#include <stdint.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void uniform1d_codegen_two_warp_vgroup_path777(
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
        scalar_t wi_22 = w[w_base + 22LL * 32 + lane];
        scalar_t wi_7 = w[w_base + 7LL * 32 + lane];
        scalar_t wi_39 = w[w_base + 39LL * 32 + lane];
        scalar_t wi_54 = w[w_base + 54LL * 32 + lane];
        scalar_t wi_49 = w[w_base + 49LL * 32 + lane];
        scalar_t wi_18 = w[w_base + 18LL * 32 + lane];
        scalar_t wi_27 = w[w_base + 27LL * 32 + lane];
        scalar_t wi_53 = w[w_base + 53LL * 32 + lane];
        scalar_t wi_38 = w[w_base + 38LL * 32 + lane];
        scalar_t wi_48 = w[w_base + 48LL * 32 + lane];
        scalar_t wi_35 = w[w_base + 35LL * 32 + lane];
        scalar_t wi_24 = w[w_base + 24LL * 32 + lane];
        scalar_t wi_15 = w[w_base + 15LL * 32 + lane];
        scalar_t wi_20 = w[w_base + 20LL * 32 + lane];
        scalar_t wi_5 = w[w_base + 5LL * 32 + lane];
        scalar_t wi_33 = w[w_base + 33LL * 32 + lane];
        scalar_t wi_13 = w[w_base + 13LL * 32 + lane];
        scalar_t wi_14 = w[w_base + 14LL * 32 + lane];
        scalar_t wi_23 = w[w_base + 23LL * 32 + lane];
        scalar_t wi_4 = w[w_base + 4LL * 32 + lane];
        scalar_t wi_34 = w[w_base + 34LL * 32 + lane];
        scalar_t wi_44 = w[w_base + 44LL * 32 + lane];
        scalar_t wi_50 = w[w_base + 50LL * 32 + lane];
        scalar_t wi_32 = w[w_base + 32LL * 32 + lane];
        scalar_t wi_19 = w[w_base + 19LL * 32 + lane];
        scalar_t wi_12 = w[w_base + 12LL * 32 + lane];
        scalar_t wi_3 = w[w_base + 3LL * 32 + lane];
        scalar_t wi_11 = w[w_base + 11LL * 32 + lane];
        scalar_t wi_10 = w[w_base + 10LL * 32 + lane];
        scalar_t wi_1 = w[w_base + 1LL * 32 + lane];
        scalar_t wi_9 = w[w_base + 9LL * 32 + lane];
        scalar_t wi_8 = w[w_base + 8LL * 32 + lane];
        scalar_t wi_28 = w[w_base + 28LL * 32 + lane];
        scalar_t wi_40 = w[w_base + 40LL * 32 + lane];
        scalar_t wi_41 = w[w_base + 41LL * 32 + lane];
        scalar_t wi_29 = w[w_base + 29LL * 32 + lane];
        scalar_t wi_30 = w[w_base + 30LL * 32 + lane];
        scalar_t wi_42 = w[w_base + 42LL * 32 + lane];
        scalar_t wi_43 = w[w_base + 43LL * 32 + lane];
        scalar_t wi_31 = w[w_base + 31LL * 32 + lane];
        scalar_t wi_26 = w[w_base + 26LL * 32 + lane];
        scalar_t wi_17 = w[w_base + 17LL * 32 + lane];
        scalar_t wi_21 = w[w_base + 21LL * 32 + lane];
        scalar_t wi_37 = w[w_base + 37LL * 32 + lane];
        scalar_t wi_52 = w[w_base + 52LL * 32 + lane];
        scalar_t wi_47 = w[w_base + 47LL * 32 + lane];
        scalar_t wi_46 = w[w_base + 46LL * 32 + lane];
        scalar_t wi_25 = w[w_base + 25LL * 32 + lane];
        scalar_t wi_16 = w[w_base + 16LL * 32 + lane];
        scalar_t wi_36 = w[w_base + 36LL * 32 + lane];
        scalar_t wi_51 = w[w_base + 51LL * 32 + lane];
        scalar_t wi_45 = w[w_base + 45LL * 32 + lane];

        // preload x(j)
        scalar_t xj_15 = x_all[x_base + 15LL * 32 + lane];
        scalar_t xj_16 = x_all[x_base + 16LL * 32 + lane];
        scalar_t xj_17 = x_all[x_base + 17LL * 32 + lane];
        scalar_t xj_18 = x_all[x_base + 18LL * 32 + lane];
        scalar_t xj_19 = x_all[x_base + 19LL * 32 + lane];
        scalar_t xj_20 = x_all[x_base + 20LL * 32 + lane];
        scalar_t xj_21 = x_all[x_base + 21LL * 32 + lane];
        scalar_t xj_5 = x_all[x_base + 5LL * 32 + lane];
        scalar_t xj_7 = x_all[x_base + 7LL * 32 + lane];
        scalar_t xj_9 = x_all[x_base + 9LL * 32 + lane];
        scalar_t xj_4 = x_all[x_base + 4LL * 32 + lane];
        scalar_t xj_6 = x_all[x_base + 6LL * 32 + lane];
        scalar_t xj_8 = x_all[x_base + 8LL * 32 + lane];
        scalar_t xj_3 = x_all[x_base + 3LL * 32 + lane];
        scalar_t xj_2 = x_all[x_base + 2LL * 32 + lane];
        scalar_t xj_1 = x_all[x_base + 1LL * 32 + lane];
        scalar_t xj_0 = x_all[x_base + 0LL * 32 + lane];
        scalar_t xj_10 = x_all[x_base + 10LL * 32 + lane];
        scalar_t xj_11 = x_all[x_base + 11LL * 32 + lane];
        scalar_t xj_12 = x_all[x_base + 12LL * 32 + lane];
        scalar_t xj_13 = x_all[x_base + 13LL * 32 + lane];
        scalar_t xj_14 = x_all[x_base + 14LL * 32 + lane];

        // preload y(k)
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_50 = scalar_t(0);
        sum_v_50 += scalar_t(-0.231455028f) * wi_22 * xj_15 * yk_10;
        sum_v_50 += scalar_t(0.231455028f) * wi_22 * xj_16 * yk_9;
        sum_v_50 += scalar_t(-0.298807144f) * wi_22 * xj_16 * yk_11;
        sum_v_50 += scalar_t(0.298807144f) * wi_22 * xj_17 * yk_10;
        sum_v_50 += scalar_t(0.462910056f) * wi_22 * xj_18 * yk_13;
        sum_v_50 += scalar_t(-0.462910056f) * wi_22 * xj_19 * yk_12;
        sum_v_50 += scalar_t(0.298807144f) * wi_22 * xj_19 * yk_14;
        sum_v_50 += scalar_t(-0.298807144f) * wi_22 * xj_20 * yk_13;
        sum_v_50 += scalar_t(0.231455028f) * wi_22 * xj_20 * yk_15;
        sum_v_50 += scalar_t(-0.231455028f) * wi_22 * xj_21 * yk_14;

        scalar_t sum_v_7 = scalar_t(0);
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_15 * yk_9;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_16 * yk_10;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_17 * yk_11;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_18 * yk_12;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_19 * yk_13;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_20 * yk_14;
        sum_v_7 += scalar_t(0.377964467f) * wi_7 * xj_21 * yk_15;

        scalar_t sum_v_134 = scalar_t(0);
        sum_v_134 += scalar_t(0.38575837f) * wi_39 * xj_15 * yk_14;
        sum_v_134 += scalar_t(0.298807144f) * wi_39 * xj_16 * yk_13;
        sum_v_134 += scalar_t(-0.38575837f) * wi_39 * xj_16 * yk_15;
        sum_v_134 += scalar_t(0.154303342f) * wi_39 * xj_17 * yk_12;
        sum_v_134 += scalar_t(-0.298807144f) * wi_39 * xj_17 * yk_14;
        sum_v_134 += scalar_t(0.154303342f) * wi_39 * xj_18 * yk_11;
        sum_v_134 += scalar_t(0.298807144f) * wi_39 * xj_19 * yk_10;
        sum_v_134 += scalar_t(0.38575837f) * wi_39 * xj_20 * yk_9;
        sum_v_134 += scalar_t(-0.298807144f) * wi_39 * xj_20 * yk_11;
        sum_v_134 += scalar_t(-0.38575837f) * wi_39 * xj_21 * yk_10;

        scalar_t sum_v_137 = scalar_t(0);
        sum_v_137 += scalar_t(-0.243975028f) * wi_39 * xj_15 * yk_11;
        sum_v_137 += scalar_t(-0.243975028f) * wi_39 * xj_17 * yk_9;
        sum_v_137 += scalar_t(-0.377964467f) * wi_39 * xj_17 * yk_11;
        sum_v_137 += scalar_t(-0.487950057f) * wi_39 * xj_18 * yk_14;
        sum_v_137 += scalar_t(0.377964467f) * wi_39 * xj_19 * yk_13;
        sum_v_137 += scalar_t(-0.243975028f) * wi_39 * xj_19 * yk_15;
        sum_v_137 += scalar_t(-0.487950057f) * wi_39 * xj_20 * yk_12;
        sum_v_137 += scalar_t(-0.243975028f) * wi_39 * xj_21 * yk_13;

        scalar_t sum_v_238 = scalar_t(0);
        sum_v_238 += scalar_t(0.408248305f) * wi_54 * xj_15 * yk_10;
        sum_v_238 += scalar_t(-0.408248305f) * wi_54 * xj_16 * yk_9;
        sum_v_238 += scalar_t(0.408248305f) * wi_54 * xj_18 * yk_13;
        sum_v_238 += scalar_t(-0.408248305f) * wi_54 * xj_19 * yk_12;
        sum_v_238 += scalar_t(-0.408248305f) * wi_54 * xj_20 * yk_15;
        sum_v_238 += scalar_t(0.408248305f) * wi_54 * xj_21 * yk_14;

        scalar_t sum_v_240 = scalar_t(0);
        sum_v_240 += scalar_t(-0.408248305f) * wi_54 * xj_15 * yk_14;
        sum_v_240 += scalar_t(-0.408248305f) * wi_54 * xj_16 * yk_15;
        sum_v_240 += scalar_t(0.408248305f) * wi_54 * xj_17 * yk_12;
        sum_v_240 += scalar_t(-0.408248305f) * wi_54 * xj_18 * yk_11;
        sum_v_240 += scalar_t(0.408248305f) * wi_54 * xj_20 * yk_9;
        sum_v_240 += scalar_t(0.408248305f) * wi_54 * xj_21 * yk_10;

        scalar_t sum_v_242 = scalar_t(0);
        sum_v_242 += scalar_t(-0.408248305f) * wi_54 * xj_15 * yk_12;
        sum_v_242 += scalar_t(0.408248305f) * wi_54 * xj_16 * yk_13;
        sum_v_242 += scalar_t(-0.408248305f) * wi_54 * xj_17 * yk_14;
        sum_v_242 += scalar_t(0.408248305f) * wi_54 * xj_18 * yk_9;
        sum_v_242 += scalar_t(-0.408248305f) * wi_54 * xj_19 * yk_10;
        sum_v_242 += scalar_t(0.408248305f) * wi_54 * xj_20 * yk_11;

        scalar_t sum_v_203 = scalar_t(0);
        sum_v_203 += scalar_t(-0.288675129f) * wi_49 * xj_15 * yk_8;
        sum_v_203 += scalar_t(0.353553385f) * wi_49 * xj_16 * yk_7;
        sum_v_203 += scalar_t(0.387298346f) * wi_49 * xj_17 * yk_6;
        sum_v_203 += scalar_t(-0.44721359f) * wi_49 * xj_17 * yk_8;
        sum_v_203 += scalar_t(0.182574183f) * wi_49 * xj_18 * yk_5;
        sum_v_203 += scalar_t(0.44721359f) * wi_49 * xj_19 * yk_4;
        sum_v_203 += scalar_t(-0.353553385f) * wi_49 * xj_20 * yk_5;
        sum_v_203 += scalar_t(0.288675129f) * wi_49 * xj_21 * yk_4;

        scalar_t sum_v_204 = scalar_t(0);
        sum_v_204 += scalar_t(-0.577350259f) * wi_49 * xj_16 * yk_4;
        sum_v_204 += scalar_t(0.182574183f) * wi_49 * xj_17 * yk_5;
        sum_v_204 += scalar_t(0.516397774f) * wi_49 * xj_18 * yk_6;
        sum_v_204 += scalar_t(0.182574183f) * wi_49 * xj_19 * yk_7;
        sum_v_204 += scalar_t(-0.577350259f) * wi_49 * xj_20 * yk_8;

        scalar_t sum_v_201 = scalar_t(0);
        sum_v_201 += scalar_t(-0.645497203f) * wi_49 * xj_15 * yk_6;
        sum_v_201 += scalar_t(0.456435472f) * wi_49 * xj_16 * yk_7;
        sum_v_201 += scalar_t(-0.288675129f) * wi_49 * xj_17 * yk_8;
        sum_v_201 += scalar_t(-0.288675129f) * wi_49 * xj_19 * yk_4;
        sum_v_201 += scalar_t(0.456435472f) * wi_49 * xj_20 * yk_5;

        scalar_t sum_v_38 = scalar_t(0);
        sum_v_38 += scalar_t(0.462910056f) * wi_18 * xj_15 * yk_8;
        sum_v_38 += scalar_t(0.377964467f) * wi_18 * xj_16 * yk_7;
        sum_v_38 += scalar_t(0.414039344f) * wi_18 * xj_17 * yk_6;
        sum_v_38 += scalar_t(0.119522862f) * wi_18 * xj_17 * yk_8;
        sum_v_38 += scalar_t(-0.292769998f) * wi_18 * xj_18 * yk_5;
        sum_v_38 += scalar_t(-0.119522862f) * wi_18 * xj_19 * yk_4;
        sum_v_38 += scalar_t(-0.377964467f) * wi_18 * xj_20 * yk_5;
        sum_v_38 += scalar_t(-0.462910056f) * wi_18 * xj_21 * yk_4;

        scalar_t sum_v_74 = scalar_t(0);
        sum_v_74 += scalar_t(0.422577113f) * wi_27 * xj_15 * yk_4;
        sum_v_74 += scalar_t(0.327326834f) * wi_27 * xj_17 * yk_4;
        sum_v_74 += scalar_t(0.534522474f) * wi_27 * xj_18 * yk_7;
        sum_v_74 += scalar_t(-0.377964467f) * wi_27 * xj_19 * yk_6;
        sum_v_74 += scalar_t(0.327326834f) * wi_27 * xj_19 * yk_8;
        sum_v_74 += scalar_t(0.422577113f) * wi_27 * xj_21 * yk_8;

        scalar_t sum_v_77 = scalar_t(0);
        sum_v_77 += scalar_t(0.422577113f) * wi_27 * xj_15 * yk_7;
        sum_v_77 += scalar_t(0.597614288f) * wi_27 * xj_16 * yk_6;
        sum_v_77 += scalar_t(-0.327326834f) * wi_27 * xj_17 * yk_7;
        sum_v_77 += scalar_t(0.267261237f) * wi_27 * xj_18 * yk_4;
        sum_v_77 += scalar_t(-0.327326834f) * wi_27 * xj_19 * yk_5;
        sum_v_77 += scalar_t(-0.422577113f) * wi_27 * xj_21 * yk_5;

        scalar_t sum_v_135 = scalar_t(0);
        sum_v_135 += scalar_t(-0.545544744f) * wi_39 * xj_15 * yk_9;
        sum_v_135 += scalar_t(0.327326834f) * wi_39 * xj_17 * yk_11;
        sum_v_135 += scalar_t(0.436435789f) * wi_39 * xj_18 * yk_12;
        sum_v_135 += scalar_t(0.327326834f) * wi_39 * xj_19 * yk_13;
        sum_v_135 += scalar_t(-0.545544744f) * wi_39 * xj_21 * yk_15;

        scalar_t sum_v_206 = scalar_t(0);
        sum_v_206 += scalar_t(0.456435472f) * wi_49 * xj_15 * yk_5;
        sum_v_206 += scalar_t(-0.353553385f) * wi_49 * xj_17 * yk_5;
        sum_v_206 += scalar_t(-0.577350259f) * wi_49 * xj_18 * yk_8;
        sum_v_206 += scalar_t(0.353553385f) * wi_49 * xj_19 * yk_7;
        sum_v_206 += scalar_t(0.456435472f) * wi_49 * xj_21 * yk_7;

        scalar_t sum_v_234 = scalar_t(0);
        sum_v_234 += scalar_t(-0.353553385f) * wi_53 * xj_15 * yk_3;
        sum_v_234 += scalar_t(0.577350259f) * wi_53 * xj_16 * yk_2;
        sum_v_234 += scalar_t(-0.456435472f) * wi_53 * xj_17 * yk_3;
        sum_v_234 += scalar_t(-0.456435472f) * wi_53 * xj_19 * yk_1;
        sum_v_234 += scalar_t(0.353553385f) * wi_53 * xj_21 * yk_1;

        scalar_t sum_v_132 = scalar_t(0);
        sum_v_132 += scalar_t(0.597614288f) * wi_38 * xj_15 * yk_1;
        sum_v_132 += scalar_t(0.154303342f) * wi_38 * xj_17 * yk_1;
        sum_v_132 += scalar_t(-0.154303342f) * wi_38 * xj_19 * yk_3;
        sum_v_132 += scalar_t(0.487950057f) * wi_38 * xj_20 * yk_2;
        sum_v_132 += scalar_t(0.597614288f) * wi_38 * xj_21 * yk_3;

        scalar_t sum_v_75 = scalar_t(0);
        sum_v_75 += scalar_t(-0.597614288f) * wi_27 * xj_16 * yk_8;
        sum_v_75 += scalar_t(-0.377964467f) * wi_27 * xj_17 * yk_7;
        sum_v_75 += scalar_t(0.377964467f) * wi_27 * xj_19 * yk_5;
        sum_v_75 += scalar_t(0.597614288f) * wi_27 * xj_20 * yk_4;

        scalar_t sum_v_131 = scalar_t(0);
        sum_v_131 += scalar_t(0.487950057f) * wi_38 * xj_16 * yk_1;
        sum_v_131 += scalar_t(-0.377964467f) * wi_38 * xj_18 * yk_3;
        sum_v_131 += scalar_t(0.617213368f) * wi_38 * xj_19 * yk_2;
        sum_v_131 += scalar_t(0.487950057f) * wi_38 * xj_20 * yk_3;

        scalar_t sum_v_233 = scalar_t(0);
        sum_v_233 += scalar_t(-0.456435472f) * wi_53 * xj_16 * yk_3;
        sum_v_233 += scalar_t(0.288675129f) * wi_53 * xj_17 * yk_2;
        sum_v_233 += scalar_t(-0.707106769f) * wi_53 * xj_18 * yk_1;
        sum_v_233 += scalar_t(0.456435472f) * wi_53 * xj_20 * yk_1;

        scalar_t sum_v_235 = scalar_t(0);
        sum_v_235 += scalar_t(0.866025388f) * wi_53 * xj_15 * yk_2;
        sum_v_235 += scalar_t(-0.353553385f) * wi_53 * xj_16 * yk_3;
        sum_v_235 += scalar_t(-0.353553385f) * wi_53 * xj_20 * yk_1;

        scalar_t sum_v_200 = scalar_t(0);
        sum_v_200 += wi_48 * xj_21 * yk_0;

        scalar_t sum_v_199 = scalar_t(0);
        sum_v_199 += wi_48 * xj_20 * yk_0;

        scalar_t sum_v_130 = scalar_t(0);
        sum_v_130 += scalar_t(0.534522474f) * wi_38 * xj_17 * yk_1;
        sum_v_130 += scalar_t(0.654653668f) * wi_38 * xj_18 * yk_2;
        sum_v_130 += scalar_t(0.534522474f) * wi_38 * xj_19 * yk_3;

        scalar_t sum_v_109 = scalar_t(0);
        sum_v_109 += scalar_t(-0.154303342f) * wi_35 * xj_5 * yk_13;
        sum_v_109 += scalar_t(-0.597614288f) * wi_35 * xj_5 * yk_15;
        sum_v_109 += scalar_t(0.487950057f) * wi_35 * xj_7 * yk_10;
        sum_v_109 += scalar_t(0.597614288f) * wi_35 * xj_9 * yk_9;
        sum_v_109 += scalar_t(-0.154303342f) * wi_35 * xj_9 * yk_11;

        scalar_t sum_v_111 = scalar_t(0);
        sum_v_111 += scalar_t(-0.377964467f) * wi_35 * xj_5 * yk_12;
        sum_v_111 += scalar_t(-0.487950057f) * wi_35 * xj_5 * yk_14;
        sum_v_111 += scalar_t(0.617213368f) * wi_35 * xj_7 * yk_11;
        sum_v_111 += scalar_t(0.487950057f) * wi_35 * xj_9 * yk_10;

        scalar_t sum_v_113 = scalar_t(0);
        sum_v_113 += scalar_t(0.534522474f) * wi_35 * xj_5 * yk_11;
        sum_v_113 += scalar_t(0.654653668f) * wi_35 * xj_7 * yk_12;
        sum_v_113 += scalar_t(0.534522474f) * wi_35 * xj_9 * yk_13;

        scalar_t sum_v_56 = scalar_t(0);
        sum_v_56 += scalar_t(-0.408248276f) * wi_24 * xj_5 * yk_4;
        sum_v_56 += scalar_t(0.408248276f) * wi_24 * xj_7 * yk_7;
        sum_v_56 += scalar_t(-0.707106769f) * wi_24 * xj_9 * yk_6;
        sum_v_56 += scalar_t(-0.408248276f) * wi_24 * xj_9 * yk_8;

        scalar_t sum_v_27 = scalar_t(0);
        sum_v_27 += scalar_t(-0.316227764f) * wi_15 * xj_5 * yk_6;
        sum_v_27 += scalar_t(-0.547722578f) * wi_15 * xj_5 * yk_8;
        sum_v_27 += scalar_t(0.547722578f) * wi_15 * xj_7 * yk_5;
        sum_v_27 += scalar_t(0.547722578f) * wi_15 * xj_9 * yk_4;

        scalar_t sum_v_29 = scalar_t(0);
        sum_v_29 += scalar_t(0.547722578f) * wi_15 * xj_5 * yk_5;
        sum_v_29 += scalar_t(0.632455528f) * wi_15 * xj_7 * yk_6;
        sum_v_29 += scalar_t(0.547722578f) * wi_15 * xj_9 * yk_7;

        scalar_t sum_v_44 = scalar_t(0);
        sum_v_44 += scalar_t(-0.707106769f) * wi_20 * xj_5 * yk_3;
        sum_v_44 += scalar_t(0.707106769f) * wi_20 * xj_9 * yk_1;

        scalar_t sum_v_5 = scalar_t(0);
        sum_v_5 += scalar_t(0.577350259f) * wi_5 * xj_5 * yk_1;
        sum_v_5 += scalar_t(0.577350259f) * wi_5 * xj_7 * yk_2;
        sum_v_5 += scalar_t(0.577350259f) * wi_5 * xj_9 * yk_3;

        scalar_t sum_v_105 = scalar_t(0);
        sum_v_105 += scalar_t(0.707106769f) * wi_33 * xj_7 * yk_3;
        sum_v_105 += scalar_t(0.707106769f) * wi_33 * xj_9 * yk_2;

        scalar_t sum_v_99 = scalar_t(0);
        sum_v_99 += scalar_t(0.707106769f) * wi_33 * xj_5 * yk_3;
        sum_v_99 += scalar_t(0.707106769f) * wi_33 * xj_9 * yk_1;

        scalar_t sum_v_46 = scalar_t(0);
        sum_v_46 += scalar_t(0.707106769f) * wi_20 * xj_5 * yk_2;
        sum_v_46 += scalar_t(-0.707106769f) * wi_20 * xj_7 * yk_1;

        scalar_t sum_v_23 = scalar_t(0);
        sum_v_23 += wi_13 * xj_7 * yk_0;

        scalar_t sum_v_30 = scalar_t(0);
        sum_v_30 += scalar_t(0.547722578f) * wi_14 * xj_4 * yk_4;
        sum_v_30 += scalar_t(0.547722578f) * wi_14 * xj_6 * yk_7;
        sum_v_30 += scalar_t(-0.316227764f) * wi_14 * xj_8 * yk_6;
        sum_v_30 += scalar_t(0.547722578f) * wi_14 * xj_8 * yk_8;

        scalar_t sum_v_59 = scalar_t(0);
        sum_v_59 += scalar_t(0.707106769f) * wi_23 * xj_4 * yk_6;
        sum_v_59 += scalar_t(-0.408248276f) * wi_23 * xj_4 * yk_8;
        sum_v_59 += scalar_t(-0.408248276f) * wi_23 * xj_6 * yk_5;
        sum_v_59 += scalar_t(0.408248276f) * wi_23 * xj_8 * yk_4;

        scalar_t sum_v_53 = scalar_t(0);
        sum_v_53 += scalar_t(0.408248276f) * wi_23 * xj_4 * yk_5;
        sum_v_53 += scalar_t(0.816496551f) * wi_23 * xj_6 * yk_8;
        sum_v_53 += scalar_t(-0.408248276f) * wi_23 * xj_8 * yk_7;

        scalar_t sum_v_4 = scalar_t(0);
        sum_v_4 += scalar_t(0.577350259f) * wi_4 * xj_4 * yk_1;
        sum_v_4 += scalar_t(0.577350259f) * wi_4 * xj_6 * yk_2;
        sum_v_4 += scalar_t(0.577350259f) * wi_4 * xj_8 * yk_3;

        scalar_t sum_v_108 = scalar_t(0);
        sum_v_108 += scalar_t(-0.154303342f) * wi_34 * xj_4 * yk_13;
        sum_v_108 += scalar_t(-0.597614288f) * wi_34 * xj_4 * yk_15;
        sum_v_108 += scalar_t(0.487950057f) * wi_34 * xj_6 * yk_10;
        sum_v_108 += scalar_t(0.597614288f) * wi_34 * xj_8 * yk_9;
        sum_v_108 += scalar_t(-0.154303342f) * wi_34 * xj_8 * yk_11;

        scalar_t sum_v_110 = scalar_t(0);
        sum_v_110 += scalar_t(-0.377964467f) * wi_34 * xj_4 * yk_12;
        sum_v_110 += scalar_t(-0.487950057f) * wi_34 * xj_4 * yk_14;
        sum_v_110 += scalar_t(0.617213368f) * wi_34 * xj_6 * yk_11;
        sum_v_110 += scalar_t(0.487950057f) * wi_34 * xj_8 * yk_10;

        scalar_t sum_v_112 = scalar_t(0);
        sum_v_112 += scalar_t(0.534522474f) * wi_34 * xj_4 * yk_11;
        sum_v_112 += scalar_t(0.654653668f) * wi_34 * xj_6 * yk_12;
        sum_v_112 += scalar_t(0.534522474f) * wi_34 * xj_8 * yk_13;

        scalar_t sum_v_174 = scalar_t(0);
        sum_v_174 += scalar_t(-0.182574183f) * wi_44 * xj_4 * yk_4;
        sum_v_174 += scalar_t(0.730296731f) * wi_44 * xj_6 * yk_7;
        sum_v_174 += scalar_t(0.632455528f) * wi_44 * xj_8 * yk_6;
        sum_v_174 += scalar_t(-0.182574183f) * wi_44 * xj_8 * yk_8;

        scalar_t sum_v_172 = scalar_t(0);
        sum_v_172 += scalar_t(-0.44721359f) * wi_44 * xj_4 * yk_5;
        sum_v_172 += scalar_t(0.774596691f) * wi_44 * xj_6 * yk_6;
        sum_v_172 += scalar_t(-0.44721359f) * wi_44 * xj_8 * yk_7;

        scalar_t sum_v_208 = scalar_t(0);
        sum_v_208 += scalar_t(0.353553385f) * wi_50 * xj_4 * yk_10;
        sum_v_208 += scalar_t(0.866025388f) * wi_50 * xj_6 * yk_15;
        sum_v_208 += scalar_t(-0.353553385f) * wi_50 * xj_8 * yk_14;

        scalar_t sum_v_210 = scalar_t(0);
        sum_v_210 += scalar_t(-0.353553385f) * wi_50 * xj_4 * yk_9;
        sum_v_210 += scalar_t(0.456435472f) * wi_50 * xj_4 * yk_11;
        sum_v_210 += scalar_t(0.577350259f) * wi_50 * xj_6 * yk_14;
        sum_v_210 += scalar_t(-0.456435472f) * wi_50 * xj_8 * yk_13;
        sum_v_210 += scalar_t(-0.353553385f) * wi_50 * xj_8 * yk_15;

        scalar_t sum_v_212 = scalar_t(0);
        sum_v_212 += scalar_t(-0.456435472f) * wi_50 * xj_4 * yk_10;
        sum_v_212 += scalar_t(0.288675129f) * wi_50 * xj_6 * yk_13;
        sum_v_212 += scalar_t(-0.707106769f) * wi_50 * xj_8 * yk_12;
        sum_v_212 += scalar_t(-0.456435472f) * wi_50 * xj_8 * yk_14;

        scalar_t sum_v_178 = scalar_t(0);
        sum_v_178 += scalar_t(-0.707106769f) * wi_44 * xj_4 * yk_4;
        sum_v_178 += scalar_t(0.707106769f) * wi_44 * xj_8 * yk_8;

        scalar_t sum_v_106 = scalar_t(0);
        sum_v_106 += scalar_t(-0.707106769f) * wi_32 * xj_4 * yk_1;
        sum_v_106 += scalar_t(0.707106769f) * wi_32 * xj_8 * yk_3;

        scalar_t sum_v_43 = scalar_t(0);
        sum_v_43 += scalar_t(-0.707106769f) * wi_19 * xj_4 * yk_3;
        sum_v_43 += scalar_t(0.707106769f) * wi_19 * xj_8 * yk_1;

        scalar_t sum_v_45 = scalar_t(0);
        sum_v_45 += scalar_t(0.707106769f) * wi_19 * xj_4 * yk_2;
        sum_v_45 += scalar_t(-0.707106769f) * wi_19 * xj_6 * yk_1;

        scalar_t sum_v_104 = scalar_t(0);
        sum_v_104 += scalar_t(0.707106769f) * wi_32 * xj_6 * yk_3;
        sum_v_104 += scalar_t(0.707106769f) * wi_32 * xj_8 * yk_2;

        scalar_t sum_v_24 = scalar_t(0);
        sum_v_24 += wi_12 * xj_8 * yk_0;

        scalar_t sum_v_21 = scalar_t(0);
        sum_v_21 += wi_13 * xj_5 * yk_0;

        scalar_t sum_v_3 = scalar_t(0);
        sum_v_3 += wi_3 * xj_3 * yk_0;

        scalar_t sum_v_15 = scalar_t(0);
        sum_v_15 += wi_11 * xj_3 * yk_2;

        scalar_t sum_v_18 = scalar_t(0);
        sum_v_18 += wi_10 * xj_2 * yk_3;

        scalar_t sum_v_10 = scalar_t(0);
        sum_v_10 += wi_10 * xj_2 * yk_1;

        scalar_t sum_v_1 = scalar_t(0);
        sum_v_1 += wi_1 * xj_1 * yk_0;

        scalar_t sum_v_13 = scalar_t(0);
        sum_v_13 += wi_9 * xj_1 * yk_2;

        scalar_t sum_v_16 = scalar_t(0);
        sum_v_16 += wi_8 * xj_0 * yk_3;

        scalar_t sum_v_8 = scalar_t(0);
        sum_v_8 += wi_8 * xj_0 * yk_1;

        scalar_t sum_v_78 = scalar_t(0);
        sum_v_78 += wi_28 * xj_0 * yk_4;

        scalar_t sum_v_86 = scalar_t(0);
        sum_v_86 += wi_28 * xj_0 * yk_6;

        scalar_t sum_v_94 = scalar_t(0);
        sum_v_94 += wi_28 * xj_0 * yk_8;

        scalar_t sum_v_142 = scalar_t(0);
        sum_v_142 += wi_40 * xj_0 * yk_10;

        scalar_t sum_v_150 = scalar_t(0);
        sum_v_150 += wi_40 * xj_0 * yk_12;

        scalar_t sum_v_158 = scalar_t(0);
        sum_v_158 += wi_40 * xj_0 * yk_14;

        scalar_t sum_v_163 = scalar_t(0);
        sum_v_163 += wi_41 * xj_1 * yk_15;

        scalar_t sum_v_155 = scalar_t(0);
        sum_v_155 += wi_41 * xj_1 * yk_13;

        scalar_t sum_v_147 = scalar_t(0);
        sum_v_147 += wi_41 * xj_1 * yk_11;

        scalar_t sum_v_139 = scalar_t(0);
        sum_v_139 += wi_41 * xj_1 * yk_9;

        scalar_t sum_v_91 = scalar_t(0);
        sum_v_91 += wi_29 * xj_1 * yk_7;

        scalar_t sum_v_83 = scalar_t(0);
        sum_v_83 += wi_29 * xj_1 * yk_5;

        scalar_t sum_v_80 = scalar_t(0);
        sum_v_80 += wi_30 * xj_2 * yk_4;

        scalar_t sum_v_88 = scalar_t(0);
        sum_v_88 += wi_30 * xj_2 * yk_6;

        scalar_t sum_v_96 = scalar_t(0);
        sum_v_96 += wi_30 * xj_2 * yk_8;

        scalar_t sum_v_144 = scalar_t(0);
        sum_v_144 += wi_42 * xj_2 * yk_10;

        scalar_t sum_v_152 = scalar_t(0);
        sum_v_152 += wi_42 * xj_2 * yk_12;

        scalar_t sum_v_160 = scalar_t(0);
        sum_v_160 += wi_42 * xj_2 * yk_14;

        scalar_t sum_v_165 = scalar_t(0);
        sum_v_165 += wi_43 * xj_3 * yk_15;

        scalar_t sum_v_157 = scalar_t(0);
        sum_v_157 += wi_43 * xj_3 * yk_13;

        scalar_t sum_v_149 = scalar_t(0);
        sum_v_149 += wi_43 * xj_3 * yk_11;

        scalar_t sum_v_141 = scalar_t(0);
        sum_v_141 += wi_43 * xj_3 * yk_9;

        scalar_t sum_v_93 = scalar_t(0);
        sum_v_93 += wi_31 * xj_3 * yk_7;

        scalar_t sum_v_85 = scalar_t(0);
        sum_v_85 += wi_31 * xj_3 * yk_5;

        scalar_t sum_v_72 = scalar_t(0);
        sum_v_72 += scalar_t(-0.267261237f) * wi_26 * xj_10 * yk_12;
        sum_v_72 += scalar_t(0.327326834f) * wi_26 * xj_11 * yk_13;
        sum_v_72 += scalar_t(0.422577113f) * wi_26 * xj_11 * yk_15;
        sum_v_72 += scalar_t(-0.597614288f) * wi_26 * xj_12 * yk_10;
        sum_v_72 += scalar_t(-0.422577113f) * wi_26 * xj_13 * yk_9;
        sum_v_72 += scalar_t(0.327326834f) * wi_26 * xj_13 * yk_11;

        scalar_t sum_v_37 = scalar_t(0);
        sum_v_37 += scalar_t(0.462910056f) * wi_17 * xj_10 * yk_9;
        sum_v_37 += scalar_t(-0.119522862f) * wi_17 * xj_10 * yk_11;
        sum_v_37 += scalar_t(0.377964467f) * wi_17 * xj_11 * yk_10;
        sum_v_37 += scalar_t(0.414039344f) * wi_17 * xj_12 * yk_13;
        sum_v_37 += scalar_t(-0.292769998f) * wi_17 * xj_13 * yk_12;
        sum_v_37 += scalar_t(0.377964467f) * wi_17 * xj_13 * yk_14;
        sum_v_37 += scalar_t(-0.119522862f) * wi_17 * xj_14 * yk_13;
        sum_v_37 += scalar_t(0.462910056f) * wi_17 * xj_14 * yk_15;

        scalar_t sum_v_36 = scalar_t(0);
        sum_v_36 += scalar_t(0.377964467f) * wi_17 * xj_10 * yk_10;
        sum_v_36 += scalar_t(0.478091449f) * wi_17 * xj_11 * yk_11;
        sum_v_36 += scalar_t(0.507092535f) * wi_17 * xj_12 * yk_12;
        sum_v_36 += scalar_t(0.478091449f) * wi_17 * xj_13 * yk_13;
        sum_v_36 += scalar_t(0.377964467f) * wi_17 * xj_14 * yk_14;

        scalar_t sum_v_49 = scalar_t(0);
        sum_v_49 += scalar_t(0.316227764f) * wi_21 * xj_10 * yk_7;
        sum_v_49 += scalar_t(0.547722578f) * wi_21 * xj_11 * yk_6;
        sum_v_49 += scalar_t(0.316227764f) * wi_21 * xj_11 * yk_8;
        sum_v_49 += scalar_t(-0.547722578f) * wi_21 * xj_12 * yk_5;
        sum_v_49 += scalar_t(-0.316227764f) * wi_21 * xj_13 * yk_4;
        sum_v_49 += scalar_t(-0.316227764f) * wi_21 * xj_14 * yk_5;

        scalar_t sum_v_125 = scalar_t(0);
        sum_v_125 += scalar_t(-0.534522474f) * wi_37 * xj_10 * yk_4;
        sum_v_125 += scalar_t(0.267261237f) * wi_37 * xj_11 * yk_5;
        sum_v_125 += scalar_t(0.534522474f) * wi_37 * xj_12 * yk_6;
        sum_v_125 += scalar_t(0.267261237f) * wi_37 * xj_13 * yk_7;
        sum_v_125 += scalar_t(-0.534522474f) * wi_37 * xj_14 * yk_8;

        scalar_t sum_v_126 = scalar_t(0);
        sum_v_126 += scalar_t(0.462910056f) * wi_37 * xj_10 * yk_5;
        sum_v_126 += scalar_t(0.462910056f) * wi_37 * xj_11 * yk_4;
        sum_v_126 += scalar_t(0.267261237f) * wi_37 * xj_12 * yk_7;
        sum_v_126 += scalar_t(0.267261237f) * wi_37 * xj_13 * yk_6;
        sum_v_126 += scalar_t(0.462910056f) * wi_37 * xj_13 * yk_8;
        sum_v_126 += scalar_t(0.462910056f) * wi_37 * xj_14 * yk_7;

        scalar_t sum_v_226 = scalar_t(0);
        sum_v_226 += scalar_t(-0.387298346f) * wi_52 * xj_10 * yk_7;
        sum_v_226 += scalar_t(0.44721359f) * wi_52 * xj_11 * yk_6;
        sum_v_226 += scalar_t(-0.387298346f) * wi_52 * xj_11 * yk_8;
        sum_v_226 += scalar_t(-0.44721359f) * wi_52 * xj_12 * yk_5;
        sum_v_226 += scalar_t(0.387298346f) * wi_52 * xj_13 * yk_4;
        sum_v_226 += scalar_t(0.387298346f) * wi_52 * xj_14 * yk_5;

        scalar_t sum_v_222 = scalar_t(0);
        sum_v_222 += scalar_t(0.5f) * wi_52 * xj_10 * yk_5;
        sum_v_222 += scalar_t(-0.5f) * wi_52 * xj_11 * yk_4;
        sum_v_222 += scalar_t(0.5f) * wi_52 * xj_13 * yk_8;
        sum_v_222 += scalar_t(-0.5f) * wi_52 * xj_14 * yk_7;

        scalar_t sum_v_191 = scalar_t(0);
        sum_v_191 += scalar_t(-0.288675129f) * wi_47 * xj_10 * yk_9;
        sum_v_191 += scalar_t(0.44721359f) * wi_47 * xj_10 * yk_11;
        sum_v_191 += scalar_t(0.353553385f) * wi_47 * xj_11 * yk_10;
        sum_v_191 += scalar_t(0.387298346f) * wi_47 * xj_12 * yk_13;
        sum_v_191 += scalar_t(0.182574183f) * wi_47 * xj_13 * yk_12;
        sum_v_191 += scalar_t(0.353553385f) * wi_47 * xj_13 * yk_14;
        sum_v_191 += scalar_t(0.44721359f) * wi_47 * xj_14 * yk_13;
        sum_v_191 += scalar_t(-0.288675129f) * wi_47 * xj_14 * yk_15;

        scalar_t sum_v_190 = scalar_t(0);
        sum_v_190 += scalar_t(-0.577350259f) * wi_47 * xj_10 * yk_10;
        sum_v_190 += scalar_t(0.182574183f) * wi_47 * xj_11 * yk_11;
        sum_v_190 += scalar_t(0.516397774f) * wi_47 * xj_12 * yk_12;
        sum_v_190 += scalar_t(0.182574183f) * wi_47 * xj_13 * yk_13;
        sum_v_190 += scalar_t(-0.577350259f) * wi_47 * xj_14 * yk_14;

        scalar_t sum_v_184 = scalar_t(0);
        sum_v_184 += scalar_t(-0.182574183f) * wi_46 * xj_10 * yk_1;
        sum_v_184 += scalar_t(0.632455528f) * wi_46 * xj_12 * yk_3;
        sum_v_184 += scalar_t(0.730296731f) * wi_46 * xj_13 * yk_2;
        sum_v_184 += scalar_t(-0.182574183f) * wi_46 * xj_14 * yk_3;

        scalar_t sum_v_66 = scalar_t(0);
        sum_v_66 += scalar_t(-0.408248276f) * wi_25 * xj_10 * yk_3;
        sum_v_66 += scalar_t(0.408248276f) * wi_25 * xj_11 * yk_2;
        sum_v_66 += scalar_t(-0.707106769f) * wi_25 * xj_12 * yk_1;
        sum_v_66 += scalar_t(0.408248276f) * wi_25 * xj_14 * yk_1;

        scalar_t sum_v_34 = scalar_t(0);
        sum_v_34 += scalar_t(0.547722578f) * wi_16 * xj_10 * yk_1;
        sum_v_34 += scalar_t(-0.316227764f) * wi_16 * xj_12 * yk_3;
        sum_v_34 += scalar_t(0.547722578f) * wi_16 * xj_13 * yk_2;
        sum_v_34 += scalar_t(0.547722578f) * wi_16 * xj_14 * yk_3;

        scalar_t sum_v_68 = scalar_t(0);
        sum_v_68 += scalar_t(0.422577113f) * wi_26 * xj_11 * yk_9;
        sum_v_68 += scalar_t(0.327326834f) * wi_26 * xj_11 * yk_11;
        sum_v_68 += scalar_t(0.597614288f) * wi_26 * xj_12 * yk_14;
        sum_v_68 += scalar_t(-0.327326834f) * wi_26 * xj_13 * yk_13;
        sum_v_68 += scalar_t(0.422577113f) * wi_26 * xj_13 * yk_15;
        sum_v_68 += scalar_t(0.267261237f) * wi_26 * xj_14 * yk_12;

        scalar_t sum_v_70 = scalar_t(0);
        sum_v_70 += scalar_t(-0.597614288f) * wi_26 * xj_10 * yk_14;
        sum_v_70 += scalar_t(-0.377964467f) * wi_26 * xj_11 * yk_13;
        sum_v_70 += scalar_t(0.377964467f) * wi_26 * xj_13 * yk_11;
        sum_v_70 += scalar_t(0.597614288f) * wi_26 * xj_14 * yk_10;

        scalar_t sum_v_123 = scalar_t(0);
        sum_v_123 += scalar_t(-0.534522474f) * wi_37 * xj_10 * yk_6;
        sum_v_123 += scalar_t(0.462910056f) * wi_37 * xj_11 * yk_7;
        sum_v_123 += scalar_t(-0.534522474f) * wi_37 * xj_12 * yk_4;
        sum_v_123 += scalar_t(0.462910056f) * wi_37 * xj_13 * yk_5;

        scalar_t sum_v_183 = scalar_t(0);
        sum_v_183 += scalar_t(-0.44721359f) * wi_46 * xj_11 * yk_1;
        sum_v_183 += scalar_t(0.774596691f) * wi_46 * xj_12 * yk_2;
        sum_v_183 += scalar_t(-0.44721359f) * wi_46 * xj_13 * yk_3;

        scalar_t sum_v_188 = scalar_t(0);
        sum_v_188 += scalar_t(-0.577350259f) * wi_47 * xj_10 * yk_12;
        sum_v_188 += scalar_t(0.353553385f) * wi_47 * xj_11 * yk_13;
        sum_v_188 += scalar_t(-0.456435472f) * wi_47 * xj_11 * yk_15;
        sum_v_188 += scalar_t(0.456435472f) * wi_47 * xj_13 * yk_9;
        sum_v_188 += scalar_t(0.353553385f) * wi_47 * xj_13 * yk_11;

        scalar_t sum_v_185 = scalar_t(0);
        sum_v_185 += scalar_t(-0.577350259f) * wi_46 * xj_11 * yk_1;
        sum_v_185 += scalar_t(0.577350259f) * wi_46 * xj_13 * yk_3;
        sum_v_185 += scalar_t(0.577350259f) * wi_46 * xj_14 * yk_2;

        scalar_t sum_v_67 = scalar_t(0);
        sum_v_67 += scalar_t(0.816496551f) * wi_25 * xj_10 * yk_2;
        sum_v_67 += scalar_t(-0.408248276f) * wi_25 * xj_11 * yk_3;
        sum_v_67 += scalar_t(-0.408248276f) * wi_25 * xj_13 * yk_1;

        scalar_t sum_v_65 = scalar_t(0);
        sum_v_65 += scalar_t(-0.707106769f) * wi_25 * xj_11 * yk_3;
        sum_v_65 += scalar_t(0.707106769f) * wi_25 * xj_13 * yk_1;

        scalar_t sum_v_119 = scalar_t(0);
        sum_v_119 += wi_36 * xj_11 * yk_0;

        scalar_t sum_v_120 = scalar_t(0);
        sum_v_120 += wi_36 * xj_12 * yk_0;

        scalar_t sum_v_122 = scalar_t(0);
        sum_v_122 += wi_36 * xj_14 * yk_0;

        scalar_t sum_v_186 = scalar_t(0);
        sum_v_186 += scalar_t(-0.707106769f) * wi_46 * xj_10 * yk_1;
        sum_v_186 += scalar_t(0.707106769f) * wi_46 * xj_14 * yk_3;

        scalar_t sum_v_227 = scalar_t(0);
        sum_v_227 += scalar_t(0.707106769f) * wi_52 * xj_10 * yk_6;
        sum_v_227 += scalar_t(-0.707106769f) * wi_52 * xj_12 * yk_4;

        scalar_t sum_v_221 = scalar_t(0);
        sum_v_221 += scalar_t(0.353553385f) * wi_51 * xj_5 * yk_14;
        sum_v_221 += scalar_t(-0.866025388f) * wi_51 * xj_7 * yk_9;
        sum_v_221 += scalar_t(0.353553385f) * wi_51 * xj_9 * yk_10;

        scalar_t sum_v_219 = scalar_t(0);
        sum_v_219 += scalar_t(0.456435472f) * wi_51 * xj_5 * yk_13;
        sum_v_219 += scalar_t(-0.353553385f) * wi_51 * xj_5 * yk_15;
        sum_v_219 += scalar_t(-0.577350259f) * wi_51 * xj_7 * yk_10;
        sum_v_219 += scalar_t(0.353553385f) * wi_51 * xj_9 * yk_9;
        sum_v_219 += scalar_t(0.456435472f) * wi_51 * xj_9 * yk_11;

        scalar_t sum_v_213 = scalar_t(0);
        sum_v_213 += scalar_t(-0.456435472f) * wi_51 * xj_5 * yk_10;
        sum_v_213 += scalar_t(0.288675129f) * wi_51 * xj_7 * yk_13;
        sum_v_213 += scalar_t(-0.707106769f) * wi_51 * xj_9 * yk_12;
        sum_v_213 += scalar_t(-0.456435472f) * wi_51 * xj_9 * yk_14;

        scalar_t sum_v_177 = scalar_t(0);
        sum_v_177 += scalar_t(-0.577350259f) * wi_45 * xj_5 * yk_5;
        sum_v_177 += scalar_t(0.577350259f) * wi_45 * xj_7 * yk_8;
        sum_v_177 += scalar_t(0.577350259f) * wi_45 * xj_9 * yk_7;

        scalar_t sum_v_175 = scalar_t(0);
        sum_v_175 += scalar_t(-0.182574183f) * wi_45 * xj_5 * yk_4;
        sum_v_175 += scalar_t(0.730296731f) * wi_45 * xj_7 * yk_7;
        sum_v_175 += scalar_t(0.632455528f) * wi_45 * xj_9 * yk_6;
        sum_v_175 += scalar_t(-0.182574183f) * wi_45 * xj_9 * yk_8;

        scalar_t sum_v_173 = scalar_t(0);
        sum_v_173 += scalar_t(-0.44721359f) * wi_45 * xj_5 * yk_5;
        sum_v_173 += scalar_t(0.774596691f) * wi_45 * xj_7 * yk_6;
        sum_v_173 += scalar_t(-0.44721359f) * wi_45 * xj_9 * yk_7;

        scalar_t sum_v_179 = scalar_t(0);
        sum_v_179 += scalar_t(-0.707106769f) * wi_45 * xj_5 * yk_4;
        sum_v_179 += scalar_t(0.707106769f) * wi_45 * xj_9 * yk_8;

        scalar_t sum_v_196 = scalar_t(0);
        sum_v_196 += wi_48 * xj_17 * yk_0;

        scalar_t sum_v_194 = scalar_t(0);
        sum_v_194 += wi_48 * xj_15 * yk_0;

        // writeback
        atomicAdd(&out[o_base + (50LL << 5)], sum_v_50);
        atomicAdd(&out[o_base + (7LL << 5)], sum_v_7);
        atomicAdd(&out[o_base + (134LL << 5)], sum_v_134);
        atomicAdd(&out[o_base + (137LL << 5)], sum_v_137);
        atomicAdd(&out[o_base + (238LL << 5)], sum_v_238);
        atomicAdd(&out[o_base + (240LL << 5)], sum_v_240);
        atomicAdd(&out[o_base + (242LL << 5)], sum_v_242);
        atomicAdd(&out[o_base + (203LL << 5)], sum_v_203);
        atomicAdd(&out[o_base + (204LL << 5)], sum_v_204);
        atomicAdd(&out[o_base + (201LL << 5)], sum_v_201);
        atomicAdd(&out[o_base + (38LL << 5)], sum_v_38);
        atomicAdd(&out[o_base + (74LL << 5)], sum_v_74);
        atomicAdd(&out[o_base + (77LL << 5)], sum_v_77);
        atomicAdd(&out[o_base + (135LL << 5)], sum_v_135);
        atomicAdd(&out[o_base + (206LL << 5)], sum_v_206);
        atomicAdd(&out[o_base + (234LL << 5)], sum_v_234);
        atomicAdd(&out[o_base + (132LL << 5)], sum_v_132);
        atomicAdd(&out[o_base + (75LL << 5)], sum_v_75);
        atomicAdd(&out[o_base + (131LL << 5)], sum_v_131);
        atomicAdd(&out[o_base + (233LL << 5)], sum_v_233);
        atomicAdd(&out[o_base + (235LL << 5)], sum_v_235);
        atomicAdd(&out[o_base + (200LL << 5)], sum_v_200);
        atomicAdd(&out[o_base + (199LL << 5)], sum_v_199);
        atomicAdd(&out[o_base + (130LL << 5)], sum_v_130);
        atomicAdd(&out[o_base + (109LL << 5)], sum_v_109);
        atomicAdd(&out[o_base + (111LL << 5)], sum_v_111);
        atomicAdd(&out[o_base + (113LL << 5)], sum_v_113);
        atomicAdd(&out[o_base + (56LL << 5)], sum_v_56);
        atomicAdd(&out[o_base + (27LL << 5)], sum_v_27);
        atomicAdd(&out[o_base + (29LL << 5)], sum_v_29);
        atomicAdd(&out[o_base + (44LL << 5)], sum_v_44);
        atomicAdd(&out[o_base + (5LL << 5)], sum_v_5);
        atomicAdd(&out[o_base + (105LL << 5)], sum_v_105);
        atomicAdd(&out[o_base + (99LL << 5)], sum_v_99);
        atomicAdd(&out[o_base + (46LL << 5)], sum_v_46);
        atomicAdd(&out[o_base + (23LL << 5)], sum_v_23);
        atomicAdd(&out[o_base + (30LL << 5)], sum_v_30);
        atomicAdd(&out[o_base + (59LL << 5)], sum_v_59);
        atomicAdd(&out[o_base + (53LL << 5)], sum_v_53);
        atomicAdd(&out[o_base + (4LL << 5)], sum_v_4);
        atomicAdd(&out[o_base + (108LL << 5)], sum_v_108);
        atomicAdd(&out[o_base + (110LL << 5)], sum_v_110);
        atomicAdd(&out[o_base + (112LL << 5)], sum_v_112);
        atomicAdd(&out[o_base + (174LL << 5)], sum_v_174);
        atomicAdd(&out[o_base + (172LL << 5)], sum_v_172);
        atomicAdd(&out[o_base + (208LL << 5)], sum_v_208);
        atomicAdd(&out[o_base + (210LL << 5)], sum_v_210);
        atomicAdd(&out[o_base + (212LL << 5)], sum_v_212);
        atomicAdd(&out[o_base + (178LL << 5)], sum_v_178);
        atomicAdd(&out[o_base + (106LL << 5)], sum_v_106);
        atomicAdd(&out[o_base + (43LL << 5)], sum_v_43);
        atomicAdd(&out[o_base + (45LL << 5)], sum_v_45);
        atomicAdd(&out[o_base + (104LL << 5)], sum_v_104);
        atomicAdd(&out[o_base + (24LL << 5)], sum_v_24);
        atomicAdd(&out[o_base + (21LL << 5)], sum_v_21);
        atomicAdd(&out[o_base + (3LL << 5)], sum_v_3);
        atomicAdd(&out[o_base + (15LL << 5)], sum_v_15);
        atomicAdd(&out[o_base + (18LL << 5)], sum_v_18);
        atomicAdd(&out[o_base + (10LL << 5)], sum_v_10);
        atomicAdd(&out[o_base + (1LL << 5)], sum_v_1);
        atomicAdd(&out[o_base + (13LL << 5)], sum_v_13);
        atomicAdd(&out[o_base + (16LL << 5)], sum_v_16);
        atomicAdd(&out[o_base + (8LL << 5)], sum_v_8);
        atomicAdd(&out[o_base + (78LL << 5)], sum_v_78);
        atomicAdd(&out[o_base + (86LL << 5)], sum_v_86);
        atomicAdd(&out[o_base + (94LL << 5)], sum_v_94);
        atomicAdd(&out[o_base + (142LL << 5)], sum_v_142);
        atomicAdd(&out[o_base + (150LL << 5)], sum_v_150);
        atomicAdd(&out[o_base + (158LL << 5)], sum_v_158);
        atomicAdd(&out[o_base + (163LL << 5)], sum_v_163);
        atomicAdd(&out[o_base + (155LL << 5)], sum_v_155);
        atomicAdd(&out[o_base + (147LL << 5)], sum_v_147);
        atomicAdd(&out[o_base + (139LL << 5)], sum_v_139);
        atomicAdd(&out[o_base + (91LL << 5)], sum_v_91);
        atomicAdd(&out[o_base + (83LL << 5)], sum_v_83);
        atomicAdd(&out[o_base + (80LL << 5)], sum_v_80);
        atomicAdd(&out[o_base + (88LL << 5)], sum_v_88);
        atomicAdd(&out[o_base + (96LL << 5)], sum_v_96);
        atomicAdd(&out[o_base + (144LL << 5)], sum_v_144);
        atomicAdd(&out[o_base + (152LL << 5)], sum_v_152);
        atomicAdd(&out[o_base + (160LL << 5)], sum_v_160);
        atomicAdd(&out[o_base + (165LL << 5)], sum_v_165);
        atomicAdd(&out[o_base + (157LL << 5)], sum_v_157);
        atomicAdd(&out[o_base + (149LL << 5)], sum_v_149);
        atomicAdd(&out[o_base + (141LL << 5)], sum_v_141);
        atomicAdd(&out[o_base + (93LL << 5)], sum_v_93);
        atomicAdd(&out[o_base + (85LL << 5)], sum_v_85);
        atomicAdd(&out[o_base + (72LL << 5)], sum_v_72);
        atomicAdd(&out[o_base + (37LL << 5)], sum_v_37);
        atomicAdd(&out[o_base + (36LL << 5)], sum_v_36);
        atomicAdd(&out[o_base + (49LL << 5)], sum_v_49);
        atomicAdd(&out[o_base + (125LL << 5)], sum_v_125);
        atomicAdd(&out[o_base + (126LL << 5)], sum_v_126);
        atomicAdd(&out[o_base + (226LL << 5)], sum_v_226);
        atomicAdd(&out[o_base + (222LL << 5)], sum_v_222);
        atomicAdd(&out[o_base + (191LL << 5)], sum_v_191);
        atomicAdd(&out[o_base + (190LL << 5)], sum_v_190);
        atomicAdd(&out[o_base + (184LL << 5)], sum_v_184);
        atomicAdd(&out[o_base + (66LL << 5)], sum_v_66);
        atomicAdd(&out[o_base + (34LL << 5)], sum_v_34);
        atomicAdd(&out[o_base + (68LL << 5)], sum_v_68);
        atomicAdd(&out[o_base + (70LL << 5)], sum_v_70);
        atomicAdd(&out[o_base + (123LL << 5)], sum_v_123);
        atomicAdd(&out[o_base + (183LL << 5)], sum_v_183);
        atomicAdd(&out[o_base + (188LL << 5)], sum_v_188);
        atomicAdd(&out[o_base + (185LL << 5)], sum_v_185);
        atomicAdd(&out[o_base + (67LL << 5)], sum_v_67);
        atomicAdd(&out[o_base + (65LL << 5)], sum_v_65);
        atomicAdd(&out[o_base + (119LL << 5)], sum_v_119);
        atomicAdd(&out[o_base + (120LL << 5)], sum_v_120);
        atomicAdd(&out[o_base + (122LL << 5)], sum_v_122);
        atomicAdd(&out[o_base + (186LL << 5)], sum_v_186);
        atomicAdd(&out[o_base + (227LL << 5)], sum_v_227);
        atomicAdd(&out[o_base + (221LL << 5)], sum_v_221);
        atomicAdd(&out[o_base + (219LL << 5)], sum_v_219);
        atomicAdd(&out[o_base + (213LL << 5)], sum_v_213);
        atomicAdd(&out[o_base + (177LL << 5)], sum_v_177);
        atomicAdd(&out[o_base + (175LL << 5)], sum_v_175);
        atomicAdd(&out[o_base + (173LL << 5)], sum_v_173);
        atomicAdd(&out[o_base + (179LL << 5)], sum_v_179);
        atomicAdd(&out[o_base + (196LL << 5)], sum_v_196);
        atomicAdd(&out[o_base + (194LL << 5)], sum_v_194);
    }

    if (warp == 1) {
        // preload w(i)
        scalar_t wi_22 = w[w_base + 22LL * 32 + lane];
        scalar_t wi_39 = w[w_base + 39LL * 32 + lane];
        scalar_t wi_54 = w[w_base + 54LL * 32 + lane];
        scalar_t wi_49 = w[w_base + 49LL * 32 + lane];
        scalar_t wi_18 = w[w_base + 18LL * 32 + lane];
        scalar_t wi_27 = w[w_base + 27LL * 32 + lane];
        scalar_t wi_53 = w[w_base + 53LL * 32 + lane];
        scalar_t wi_38 = w[w_base + 38LL * 32 + lane];
        scalar_t wi_48 = w[w_base + 48LL * 32 + lane];
        scalar_t wi_33 = w[w_base + 33LL * 32 + lane];
        scalar_t wi_35 = w[w_base + 35LL * 32 + lane];
        scalar_t wi_24 = w[w_base + 24LL * 32 + lane];
        scalar_t wi_15 = w[w_base + 15LL * 32 + lane];
        scalar_t wi_20 = w[w_base + 20LL * 32 + lane];
        scalar_t wi_12 = w[w_base + 12LL * 32 + lane];
        scalar_t wi_14 = w[w_base + 14LL * 32 + lane];
        scalar_t wi_23 = w[w_base + 23LL * 32 + lane];
        scalar_t wi_32 = w[w_base + 32LL * 32 + lane];
        scalar_t wi_34 = w[w_base + 34LL * 32 + lane];
        scalar_t wi_44 = w[w_base + 44LL * 32 + lane];
        scalar_t wi_50 = w[w_base + 50LL * 32 + lane];
        scalar_t wi_19 = w[w_base + 19LL * 32 + lane];
        scalar_t wi_13 = w[w_base + 13LL * 32 + lane];
        scalar_t wi_11 = w[w_base + 11LL * 32 + lane];
        scalar_t wi_10 = w[w_base + 10LL * 32 + lane];
        scalar_t wi_2 = w[w_base + 2LL * 32 + lane];
        scalar_t wi_9 = w[w_base + 9LL * 32 + lane];
        scalar_t wi_8 = w[w_base + 8LL * 32 + lane];
        scalar_t wi_0 = w[w_base + 0LL * 32 + lane];
        scalar_t wi_28 = w[w_base + 28LL * 32 + lane];
        scalar_t wi_40 = w[w_base + 40LL * 32 + lane];
        scalar_t wi_41 = w[w_base + 41LL * 32 + lane];
        scalar_t wi_29 = w[w_base + 29LL * 32 + lane];
        scalar_t wi_30 = w[w_base + 30LL * 32 + lane];
        scalar_t wi_42 = w[w_base + 42LL * 32 + lane];
        scalar_t wi_43 = w[w_base + 43LL * 32 + lane];
        scalar_t wi_31 = w[w_base + 31LL * 32 + lane];
        scalar_t wi_17 = w[w_base + 17LL * 32 + lane];
        scalar_t wi_21 = w[w_base + 21LL * 32 + lane];
        scalar_t wi_6 = w[w_base + 6LL * 32 + lane];
        scalar_t wi_37 = w[w_base + 37LL * 32 + lane];
        scalar_t wi_52 = w[w_base + 52LL * 32 + lane];
        scalar_t wi_47 = w[w_base + 47LL * 32 + lane];
        scalar_t wi_46 = w[w_base + 46LL * 32 + lane];
        scalar_t wi_16 = w[w_base + 16LL * 32 + lane];
        scalar_t wi_25 = w[w_base + 25LL * 32 + lane];
        scalar_t wi_26 = w[w_base + 26LL * 32 + lane];
        scalar_t wi_36 = w[w_base + 36LL * 32 + lane];
        scalar_t wi_51 = w[w_base + 51LL * 32 + lane];
        scalar_t wi_45 = w[w_base + 45LL * 32 + lane];

        // preload x(j)
        scalar_t xj_15 = x_all[x_base + 15LL * 32 + lane];
        scalar_t xj_16 = x_all[x_base + 16LL * 32 + lane];
        scalar_t xj_17 = x_all[x_base + 17LL * 32 + lane];
        scalar_t xj_18 = x_all[x_base + 18LL * 32 + lane];
        scalar_t xj_19 = x_all[x_base + 19LL * 32 + lane];
        scalar_t xj_20 = x_all[x_base + 20LL * 32 + lane];
        scalar_t xj_21 = x_all[x_base + 21LL * 32 + lane];
        scalar_t xj_5 = x_all[x_base + 5LL * 32 + lane];
        scalar_t xj_7 = x_all[x_base + 7LL * 32 + lane];
        scalar_t xj_9 = x_all[x_base + 9LL * 32 + lane];
        scalar_t xj_6 = x_all[x_base + 6LL * 32 + lane];
        scalar_t xj_4 = x_all[x_base + 4LL * 32 + lane];
        scalar_t xj_8 = x_all[x_base + 8LL * 32 + lane];
        scalar_t xj_3 = x_all[x_base + 3LL * 32 + lane];
        scalar_t xj_2 = x_all[x_base + 2LL * 32 + lane];
        scalar_t xj_1 = x_all[x_base + 1LL * 32 + lane];
        scalar_t xj_0 = x_all[x_base + 0LL * 32 + lane];
        scalar_t xj_10 = x_all[x_base + 10LL * 32 + lane];
        scalar_t xj_11 = x_all[x_base + 11LL * 32 + lane];
        scalar_t xj_12 = x_all[x_base + 12LL * 32 + lane];
        scalar_t xj_13 = x_all[x_base + 13LL * 32 + lane];
        scalar_t xj_14 = x_all[x_base + 14LL * 32 + lane];

        // preload y(k)
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_52 = scalar_t(0);
        sum_v_52 += scalar_t(0.231455028f) * wi_22 * xj_15 * yk_14;
        sum_v_52 += scalar_t(0.298807144f) * wi_22 * xj_16 * yk_13;
        sum_v_52 += scalar_t(0.231455028f) * wi_22 * xj_16 * yk_15;
        sum_v_52 += scalar_t(0.462910056f) * wi_22 * xj_17 * yk_12;
        sum_v_52 += scalar_t(0.298807144f) * wi_22 * xj_17 * yk_14;
        sum_v_52 += scalar_t(-0.462910056f) * wi_22 * xj_18 * yk_11;
        sum_v_52 += scalar_t(-0.298807144f) * wi_22 * xj_19 * yk_10;
        sum_v_52 += scalar_t(-0.231455028f) * wi_22 * xj_20 * yk_9;
        sum_v_52 += scalar_t(-0.298807144f) * wi_22 * xj_20 * yk_11;
        sum_v_52 += scalar_t(-0.231455028f) * wi_22 * xj_21 * yk_10;

        scalar_t sum_v_136 = scalar_t(0);
        sum_v_136 += scalar_t(0.38575837f) * wi_39 * xj_15 * yk_10;
        sum_v_136 += scalar_t(0.38575837f) * wi_39 * xj_16 * yk_9;
        sum_v_136 += scalar_t(0.298807144f) * wi_39 * xj_16 * yk_11;
        sum_v_136 += scalar_t(0.298807144f) * wi_39 * xj_17 * yk_10;
        sum_v_136 += scalar_t(0.154303342f) * wi_39 * xj_18 * yk_13;
        sum_v_136 += scalar_t(0.154303342f) * wi_39 * xj_19 * yk_12;
        sum_v_136 += scalar_t(0.298807144f) * wi_39 * xj_19 * yk_14;
        sum_v_136 += scalar_t(0.298807144f) * wi_39 * xj_20 * yk_13;
        sum_v_136 += scalar_t(0.38575837f) * wi_39 * xj_20 * yk_15;
        sum_v_136 += scalar_t(0.38575837f) * wi_39 * xj_21 * yk_14;

        scalar_t sum_v_237 = scalar_t(0);
        sum_v_237 += scalar_t(-0.408248305f) * wi_54 * xj_15 * yk_11;
        sum_v_237 += scalar_t(0.408248305f) * wi_54 * xj_17 * yk_9;
        sum_v_237 += scalar_t(0.408248305f) * wi_54 * xj_18 * yk_14;
        sum_v_237 += scalar_t(0.408248305f) * wi_54 * xj_19 * yk_15;
        sum_v_237 += scalar_t(-0.408248305f) * wi_54 * xj_20 * yk_12;
        sum_v_237 += scalar_t(-0.408248305f) * wi_54 * xj_21 * yk_13;

        scalar_t sum_v_236 = scalar_t(0);
        sum_v_236 += scalar_t(0.408248305f) * wi_54 * xj_16 * yk_11;
        sum_v_236 += scalar_t(-0.408248305f) * wi_54 * xj_17 * yk_10;
        sum_v_236 += scalar_t(-0.408248305f) * wi_54 * xj_18 * yk_15;
        sum_v_236 += scalar_t(0.408248305f) * wi_54 * xj_19 * yk_14;
        sum_v_236 += scalar_t(-0.408248305f) * wi_54 * xj_20 * yk_13;
        sum_v_236 += scalar_t(0.408248305f) * wi_54 * xj_21 * yk_12;

        scalar_t sum_v_239 = scalar_t(0);
        sum_v_239 += scalar_t(0.408248305f) * wi_54 * xj_15 * yk_15;
        sum_v_239 += scalar_t(-0.408248305f) * wi_54 * xj_16 * yk_14;
        sum_v_239 += scalar_t(-0.408248305f) * wi_54 * xj_17 * yk_13;
        sum_v_239 += scalar_t(0.408248305f) * wi_54 * xj_19 * yk_11;
        sum_v_239 += scalar_t(0.408248305f) * wi_54 * xj_20 * yk_10;
        sum_v_239 += scalar_t(-0.408248305f) * wi_54 * xj_21 * yk_9;

        scalar_t sum_v_241 = scalar_t(0);
        sum_v_241 += scalar_t(0.408248305f) * wi_54 * xj_15 * yk_13;
        sum_v_241 += scalar_t(0.408248305f) * wi_54 * xj_16 * yk_12;
        sum_v_241 += scalar_t(0.408248305f) * wi_54 * xj_17 * yk_15;
        sum_v_241 += scalar_t(-0.408248305f) * wi_54 * xj_18 * yk_10;
        sum_v_241 += scalar_t(-0.408248305f) * wi_54 * xj_19 * yk_9;
        sum_v_241 += scalar_t(-0.408248305f) * wi_54 * xj_21 * yk_11;

        scalar_t sum_v_205 = scalar_t(0);
        sum_v_205 += scalar_t(-0.288675129f) * wi_49 * xj_15 * yk_4;
        sum_v_205 += scalar_t(0.353553385f) * wi_49 * xj_16 * yk_5;
        sum_v_205 += scalar_t(0.44721359f) * wi_49 * xj_17 * yk_4;
        sum_v_205 += scalar_t(0.182574183f) * wi_49 * xj_18 * yk_7;
        sum_v_205 += scalar_t(0.387298346f) * wi_49 * xj_19 * yk_6;
        sum_v_205 += scalar_t(0.44721359f) * wi_49 * xj_19 * yk_8;
        sum_v_205 += scalar_t(0.353553385f) * wi_49 * xj_20 * yk_7;
        sum_v_205 += scalar_t(-0.288675129f) * wi_49 * xj_21 * yk_8;

        scalar_t sum_v_207 = scalar_t(0);
        sum_v_207 += scalar_t(-0.456435472f) * wi_49 * xj_16 * yk_5;
        sum_v_207 += scalar_t(0.288675129f) * wi_49 * xj_17 * yk_4;
        sum_v_207 += scalar_t(-0.288675129f) * wi_49 * xj_19 * yk_8;
        sum_v_207 += scalar_t(0.456435472f) * wi_49 * xj_20 * yk_7;
        sum_v_207 += scalar_t(-0.645497203f) * wi_49 * xj_21 * yk_6;

        scalar_t sum_v_40 = scalar_t(0);
        sum_v_40 += scalar_t(0.462910056f) * wi_18 * xj_15 * yk_4;
        sum_v_40 += scalar_t(0.377964467f) * wi_18 * xj_16 * yk_5;
        sum_v_40 += scalar_t(-0.119522862f) * wi_18 * xj_17 * yk_4;
        sum_v_40 += scalar_t(-0.292769998f) * wi_18 * xj_18 * yk_7;
        sum_v_40 += scalar_t(0.414039344f) * wi_18 * xj_19 * yk_6;
        sum_v_40 += scalar_t(-0.119522862f) * wi_18 * xj_19 * yk_8;
        sum_v_40 += scalar_t(0.377964467f) * wi_18 * xj_20 * yk_7;
        sum_v_40 += scalar_t(0.462910056f) * wi_18 * xj_21 * yk_8;

        scalar_t sum_v_73 = scalar_t(0);
        sum_v_73 += scalar_t(-0.422577113f) * wi_27 * xj_15 * yk_5;
        sum_v_73 += scalar_t(-0.327326834f) * wi_27 * xj_17 * yk_5;
        sum_v_73 += scalar_t(-0.267261237f) * wi_27 * xj_18 * yk_8;
        sum_v_73 += scalar_t(0.327326834f) * wi_27 * xj_19 * yk_7;
        sum_v_73 += scalar_t(-0.597614288f) * wi_27 * xj_20 * yk_6;
        sum_v_73 += scalar_t(-0.422577113f) * wi_27 * xj_21 * yk_7;

        scalar_t sum_v_76 = scalar_t(0);
        sum_v_76 += scalar_t(-0.422577113f) * wi_27 * xj_15 * yk_8;
        sum_v_76 += scalar_t(0.377964467f) * wi_27 * xj_17 * yk_6;
        sum_v_76 += scalar_t(0.327326834f) * wi_27 * xj_17 * yk_8;
        sum_v_76 += scalar_t(-0.534522474f) * wi_27 * xj_18 * yk_5;
        sum_v_76 += scalar_t(-0.327326834f) * wi_27 * xj_19 * yk_4;
        sum_v_76 += scalar_t(0.422577113f) * wi_27 * xj_21 * yk_4;

        scalar_t sum_v_133 = scalar_t(0);
        sum_v_133 += scalar_t(-0.243975028f) * wi_39 * xj_15 * yk_13;
        sum_v_133 += scalar_t(-0.487950057f) * wi_39 * xj_16 * yk_12;
        sum_v_133 += scalar_t(0.377964467f) * wi_39 * xj_17 * yk_13;
        sum_v_133 += scalar_t(0.243975028f) * wi_39 * xj_17 * yk_15;
        sum_v_133 += scalar_t(-0.487950057f) * wi_39 * xj_18 * yk_10;
        sum_v_133 += scalar_t(-0.243975028f) * wi_39 * xj_19 * yk_9;
        sum_v_133 += scalar_t(0.377964467f) * wi_39 * xj_19 * yk_11;
        sum_v_133 += scalar_t(0.243975028f) * wi_39 * xj_21 * yk_11;

        scalar_t sum_v_202 = scalar_t(0);
        sum_v_202 += scalar_t(0.456435472f) * wi_49 * xj_15 * yk_7;
        sum_v_202 += scalar_t(0.353553385f) * wi_49 * xj_17 * yk_7;
        sum_v_202 += scalar_t(-0.577350259f) * wi_49 * xj_18 * yk_4;
        sum_v_202 += scalar_t(0.353553385f) * wi_49 * xj_19 * yk_5;
        sum_v_202 += scalar_t(-0.456435472f) * wi_49 * xj_21 * yk_5;

        scalar_t sum_v_230 = scalar_t(0);
        sum_v_230 += scalar_t(0.353553385f) * wi_53 * xj_15 * yk_1;
        sum_v_230 += scalar_t(-0.456435472f) * wi_53 * xj_17 * yk_1;
        sum_v_230 += scalar_t(0.456435472f) * wi_53 * xj_19 * yk_3;
        sum_v_230 += scalar_t(-0.577350259f) * wi_53 * xj_20 * yk_2;
        sum_v_230 += scalar_t(0.353553385f) * wi_53 * xj_21 * yk_3;

        scalar_t sum_v_128 = scalar_t(0);
        sum_v_128 += scalar_t(0.597614288f) * wi_38 * xj_15 * yk_3;
        sum_v_128 += scalar_t(0.487950057f) * wi_38 * xj_16 * yk_2;
        sum_v_128 += scalar_t(-0.154303342f) * wi_38 * xj_17 * yk_3;
        sum_v_128 += scalar_t(-0.154303342f) * wi_38 * xj_19 * yk_1;
        sum_v_128 += scalar_t(-0.597614288f) * wi_38 * xj_21 * yk_1;

        scalar_t sum_v_51 = scalar_t(0);
        sum_v_51 += scalar_t(-0.566946685f) * wi_22 * xj_15 * yk_15;
        sum_v_51 += scalar_t(-0.377964497f) * wi_22 * xj_16 * yk_14;
        sum_v_51 += scalar_t(-0.188982248f) * wi_22 * xj_17 * yk_13;
        sum_v_51 += scalar_t(0.188982248f) * wi_22 * xj_19 * yk_11;
        sum_v_51 += scalar_t(0.377964497f) * wi_22 * xj_20 * yk_10;
        sum_v_51 += scalar_t(0.566946685f) * wi_22 * xj_21 * yk_9;

        scalar_t sum_v_39 = scalar_t(0);
        sum_v_39 += scalar_t(0.377964467f) * wi_18 * xj_16 * yk_4;
        sum_v_39 += scalar_t(0.478091449f) * wi_18 * xj_17 * yk_5;
        sum_v_39 += scalar_t(0.507092535f) * wi_18 * xj_18 * yk_6;
        sum_v_39 += scalar_t(0.478091449f) * wi_18 * xj_19 * yk_7;
        sum_v_39 += scalar_t(0.377964467f) * wi_18 * xj_20 * yk_8;

        scalar_t sum_v_129 = scalar_t(0);
        sum_v_129 += scalar_t(0.487950057f) * wi_38 * xj_16 * yk_3;
        sum_v_129 += scalar_t(0.617213368f) * wi_38 * xj_17 * yk_2;
        sum_v_129 += scalar_t(-0.377964467f) * wi_38 * xj_18 * yk_1;
        sum_v_129 += scalar_t(-0.487950057f) * wi_38 * xj_20 * yk_1;

        scalar_t sum_v_231 = scalar_t(0);
        sum_v_231 += scalar_t(0.456435472f) * wi_53 * xj_16 * yk_1;
        sum_v_231 += scalar_t(0.707106769f) * wi_53 * xj_18 * yk_3;
        sum_v_231 += scalar_t(-0.288675129f) * wi_53 * xj_19 * yk_2;
        sum_v_231 += scalar_t(0.456435472f) * wi_53 * xj_20 * yk_3;

        scalar_t sum_v_229 = scalar_t(0);
        sum_v_229 += scalar_t(-0.353553385f) * wi_53 * xj_16 * yk_1;
        sum_v_229 += scalar_t(0.353553385f) * wi_53 * xj_20 * yk_3;
        sum_v_229 += scalar_t(-0.866025388f) * wi_53 * xj_21 * yk_2;

        scalar_t sum_v_198 = scalar_t(0);
        sum_v_198 += wi_48 * xj_19 * yk_0;

        scalar_t sum_v_232 = scalar_t(0);
        sum_v_232 += scalar_t(-0.707106769f) * wi_53 * xj_17 * yk_3;
        sum_v_232 += scalar_t(0.707106769f) * wi_53 * xj_19 * yk_1;

        scalar_t sum_v_103 = scalar_t(0);
        sum_v_103 += scalar_t(-0.408248276f) * wi_33 * xj_5 * yk_1;
        sum_v_103 += scalar_t(0.816496551f) * wi_33 * xj_7 * yk_2;
        sum_v_103 += scalar_t(-0.408248276f) * wi_33 * xj_9 * yk_3;

        scalar_t sum_v_117 = scalar_t(0);
        sum_v_117 += scalar_t(0.597614288f) * wi_35 * xj_5 * yk_9;
        sum_v_117 += scalar_t(0.154303342f) * wi_35 * xj_5 * yk_11;
        sum_v_117 += scalar_t(0.487950057f) * wi_35 * xj_7 * yk_14;
        sum_v_117 += scalar_t(-0.154303342f) * wi_35 * xj_9 * yk_13;
        sum_v_117 += scalar_t(0.597614288f) * wi_35 * xj_9 * yk_15;

        scalar_t sum_v_115 = scalar_t(0);
        sum_v_115 += scalar_t(0.487950057f) * wi_35 * xj_5 * yk_10;
        sum_v_115 += scalar_t(0.617213368f) * wi_35 * xj_7 * yk_13;
        sum_v_115 += scalar_t(-0.377964467f) * wi_35 * xj_9 * yk_12;
        sum_v_115 += scalar_t(0.487950057f) * wi_35 * xj_9 * yk_14;

        scalar_t sum_v_62 = scalar_t(0);
        sum_v_62 += scalar_t(0.408248276f) * wi_24 * xj_5 * yk_7;
        sum_v_62 += scalar_t(-0.816496551f) * wi_24 * xj_7 * yk_4;
        sum_v_62 += scalar_t(0.408248276f) * wi_24 * xj_9 * yk_5;

        scalar_t sum_v_60 = scalar_t(0);
        sum_v_60 += scalar_t(0.707106769f) * wi_24 * xj_5 * yk_6;
        sum_v_60 += scalar_t(-0.408248276f) * wi_24 * xj_5 * yk_8;
        sum_v_60 += scalar_t(-0.408248276f) * wi_24 * xj_7 * yk_5;
        sum_v_60 += scalar_t(0.408248276f) * wi_24 * xj_9 * yk_4;

        scalar_t sum_v_31 = scalar_t(0);
        sum_v_31 += scalar_t(0.547722578f) * wi_15 * xj_5 * yk_4;
        sum_v_31 += scalar_t(0.547722578f) * wi_15 * xj_7 * yk_7;
        sum_v_31 += scalar_t(-0.316227764f) * wi_15 * xj_9 * yk_6;
        sum_v_31 += scalar_t(0.547722578f) * wi_15 * xj_9 * yk_8;

        scalar_t sum_v_54 = scalar_t(0);
        sum_v_54 += scalar_t(0.408248276f) * wi_24 * xj_5 * yk_5;
        sum_v_54 += scalar_t(0.816496551f) * wi_24 * xj_7 * yk_8;
        sum_v_54 += scalar_t(-0.408248276f) * wi_24 * xj_9 * yk_7;

        scalar_t sum_v_58 = scalar_t(0);
        sum_v_58 += scalar_t(-0.707106769f) * wi_24 * xj_5 * yk_7;
        sum_v_58 += scalar_t(0.707106769f) * wi_24 * xj_9 * yk_5;

        scalar_t sum_v_42 = scalar_t(0);
        sum_v_42 += scalar_t(0.707106769f) * wi_20 * xj_7 * yk_3;
        sum_v_42 += scalar_t(-0.707106769f) * wi_20 * xj_9 * yk_2;

        scalar_t sum_v_107 = scalar_t(0);
        sum_v_107 += scalar_t(-0.707106769f) * wi_33 * xj_5 * yk_1;
        sum_v_107 += scalar_t(0.707106769f) * wi_33 * xj_9 * yk_3;

        scalar_t sum_v_101 = scalar_t(0);
        sum_v_101 += scalar_t(0.707106769f) * wi_33 * xj_5 * yk_2;
        sum_v_101 += scalar_t(0.707106769f) * wi_33 * xj_7 * yk_1;

        scalar_t sum_v_22 = scalar_t(0);
        sum_v_22 += wi_12 * xj_6 * yk_0;

        scalar_t sum_v_26 = scalar_t(0);
        sum_v_26 += scalar_t(-0.316227764f) * wi_14 * xj_4 * yk_6;
        sum_v_26 += scalar_t(-0.547722578f) * wi_14 * xj_4 * yk_8;
        sum_v_26 += scalar_t(0.547722578f) * wi_14 * xj_6 * yk_5;
        sum_v_26 += scalar_t(0.547722578f) * wi_14 * xj_8 * yk_4;

        scalar_t sum_v_55 = scalar_t(0);
        sum_v_55 += scalar_t(-0.408248276f) * wi_23 * xj_4 * yk_4;
        sum_v_55 += scalar_t(0.408248276f) * wi_23 * xj_6 * yk_7;
        sum_v_55 += scalar_t(-0.707106769f) * wi_23 * xj_8 * yk_6;
        sum_v_55 += scalar_t(-0.408248276f) * wi_23 * xj_8 * yk_8;

        scalar_t sum_v_61 = scalar_t(0);
        sum_v_61 += scalar_t(0.408248276f) * wi_23 * xj_4 * yk_7;
        sum_v_61 += scalar_t(-0.816496551f) * wi_23 * xj_6 * yk_4;
        sum_v_61 += scalar_t(0.408248276f) * wi_23 * xj_8 * yk_5;

        scalar_t sum_v_28 = scalar_t(0);
        sum_v_28 += scalar_t(0.547722578f) * wi_14 * xj_4 * yk_5;
        sum_v_28 += scalar_t(0.632455528f) * wi_14 * xj_6 * yk_6;
        sum_v_28 += scalar_t(0.547722578f) * wi_14 * xj_8 * yk_7;

        scalar_t sum_v_102 = scalar_t(0);
        sum_v_102 += scalar_t(-0.408248276f) * wi_32 * xj_4 * yk_1;
        sum_v_102 += scalar_t(0.816496551f) * wi_32 * xj_6 * yk_2;
        sum_v_102 += scalar_t(-0.408248276f) * wi_32 * xj_8 * yk_3;

        scalar_t sum_v_116 = scalar_t(0);
        sum_v_116 += scalar_t(0.597614288f) * wi_34 * xj_4 * yk_9;
        sum_v_116 += scalar_t(0.154303342f) * wi_34 * xj_4 * yk_11;
        sum_v_116 += scalar_t(0.487950057f) * wi_34 * xj_6 * yk_14;
        sum_v_116 += scalar_t(-0.154303342f) * wi_34 * xj_8 * yk_13;
        sum_v_116 += scalar_t(0.597614288f) * wi_34 * xj_8 * yk_15;

        scalar_t sum_v_114 = scalar_t(0);
        sum_v_114 += scalar_t(0.487950057f) * wi_34 * xj_4 * yk_10;
        sum_v_114 += scalar_t(0.617213368f) * wi_34 * xj_6 * yk_13;
        sum_v_114 += scalar_t(-0.377964467f) * wi_34 * xj_8 * yk_12;
        sum_v_114 += scalar_t(0.487950057f) * wi_34 * xj_8 * yk_14;

        scalar_t sum_v_168 = scalar_t(0);
        sum_v_168 += scalar_t(0.577350259f) * wi_44 * xj_4 * yk_7;
        sum_v_168 += scalar_t(0.577350259f) * wi_44 * xj_6 * yk_4;
        sum_v_168 += scalar_t(0.577350259f) * wi_44 * xj_8 * yk_5;

        scalar_t sum_v_170 = scalar_t(0);
        sum_v_170 += scalar_t(0.632455528f) * wi_44 * xj_4 * yk_6;
        sum_v_170 += scalar_t(0.182574183f) * wi_44 * xj_4 * yk_8;
        sum_v_170 += scalar_t(0.730296731f) * wi_44 * xj_6 * yk_5;
        sum_v_170 += scalar_t(-0.182574183f) * wi_44 * xj_8 * yk_4;

        scalar_t sum_v_176 = scalar_t(0);
        sum_v_176 += scalar_t(-0.577350259f) * wi_44 * xj_4 * yk_5;
        sum_v_176 += scalar_t(0.577350259f) * wi_44 * xj_6 * yk_8;
        sum_v_176 += scalar_t(0.577350259f) * wi_44 * xj_8 * yk_7;

        scalar_t sum_v_218 = scalar_t(0);
        sum_v_218 += scalar_t(0.456435472f) * wi_50 * xj_4 * yk_13;
        sum_v_218 += scalar_t(-0.353553385f) * wi_50 * xj_4 * yk_15;
        sum_v_218 += scalar_t(-0.577350259f) * wi_50 * xj_6 * yk_10;
        sum_v_218 += scalar_t(0.353553385f) * wi_50 * xj_8 * yk_9;
        sum_v_218 += scalar_t(0.456435472f) * wi_50 * xj_8 * yk_11;

        scalar_t sum_v_216 = scalar_t(0);
        sum_v_216 += scalar_t(0.707106769f) * wi_50 * xj_4 * yk_12;
        sum_v_216 += scalar_t(-0.456435472f) * wi_50 * xj_4 * yk_14;
        sum_v_216 += scalar_t(-0.288675129f) * wi_50 * xj_6 * yk_11;
        sum_v_216 += scalar_t(0.456435472f) * wi_50 * xj_8 * yk_10;

        scalar_t sum_v_220 = scalar_t(0);
        sum_v_220 += scalar_t(0.353553385f) * wi_50 * xj_4 * yk_14;
        sum_v_220 += scalar_t(-0.866025388f) * wi_50 * xj_6 * yk_9;
        sum_v_220 += scalar_t(0.353553385f) * wi_50 * xj_8 * yk_10;

        scalar_t sum_v_214 = scalar_t(0);
        sum_v_214 += scalar_t(-0.707106769f) * wi_50 * xj_4 * yk_13;
        sum_v_214 += scalar_t(0.707106769f) * wi_50 * xj_8 * yk_11;

        scalar_t sum_v_166 = scalar_t(0);
        sum_v_166 += scalar_t(0.707106769f) * wi_44 * xj_4 * yk_8;
        sum_v_166 += scalar_t(0.707106769f) * wi_44 * xj_8 * yk_4;

        scalar_t sum_v_98 = scalar_t(0);
        sum_v_98 += scalar_t(0.707106769f) * wi_32 * xj_4 * yk_3;
        sum_v_98 += scalar_t(0.707106769f) * wi_32 * xj_8 * yk_1;

        scalar_t sum_v_57 = scalar_t(0);
        sum_v_57 += scalar_t(-0.707106769f) * wi_23 * xj_4 * yk_7;
        sum_v_57 += scalar_t(0.707106769f) * wi_23 * xj_8 * yk_5;

        scalar_t sum_v_100 = scalar_t(0);
        sum_v_100 += scalar_t(0.707106769f) * wi_32 * xj_4 * yk_2;
        sum_v_100 += scalar_t(0.707106769f) * wi_32 * xj_6 * yk_1;

        scalar_t sum_v_41 = scalar_t(0);
        sum_v_41 += scalar_t(0.707106769f) * wi_19 * xj_6 * yk_3;
        sum_v_41 += scalar_t(-0.707106769f) * wi_19 * xj_8 * yk_2;

        scalar_t sum_v_25 = scalar_t(0);
        sum_v_25 += wi_13 * xj_9 * yk_0;

        scalar_t sum_v_20 = scalar_t(0);
        sum_v_20 += wi_12 * xj_4 * yk_0;

        scalar_t sum_v_11 = scalar_t(0);
        sum_v_11 += wi_11 * xj_3 * yk_1;

        scalar_t sum_v_19 = scalar_t(0);
        sum_v_19 += wi_11 * xj_3 * yk_3;

        scalar_t sum_v_14 = scalar_t(0);
        sum_v_14 += wi_10 * xj_2 * yk_2;

        scalar_t sum_v_2 = scalar_t(0);
        sum_v_2 += wi_2 * xj_2 * yk_0;

        scalar_t sum_v_9 = scalar_t(0);
        sum_v_9 += wi_9 * xj_1 * yk_1;

        scalar_t sum_v_17 = scalar_t(0);
        sum_v_17 += wi_9 * xj_1 * yk_3;

        scalar_t sum_v_12 = scalar_t(0);
        sum_v_12 += wi_8 * xj_0 * yk_2;

        scalar_t sum_v_0 = scalar_t(0);
        sum_v_0 += wi_0 * xj_0 * yk_0;

        scalar_t sum_v_82 = scalar_t(0);
        sum_v_82 += wi_28 * xj_0 * yk_5;

        scalar_t sum_v_90 = scalar_t(0);
        sum_v_90 += wi_28 * xj_0 * yk_7;

        scalar_t sum_v_138 = scalar_t(0);
        sum_v_138 += wi_40 * xj_0 * yk_9;

        scalar_t sum_v_146 = scalar_t(0);
        sum_v_146 += wi_40 * xj_0 * yk_11;

        scalar_t sum_v_154 = scalar_t(0);
        sum_v_154 += wi_40 * xj_0 * yk_13;

        scalar_t sum_v_162 = scalar_t(0);
        sum_v_162 += wi_40 * xj_0 * yk_15;

        scalar_t sum_v_159 = scalar_t(0);
        sum_v_159 += wi_41 * xj_1 * yk_14;

        scalar_t sum_v_151 = scalar_t(0);
        sum_v_151 += wi_41 * xj_1 * yk_12;

        scalar_t sum_v_143 = scalar_t(0);
        sum_v_143 += wi_41 * xj_1 * yk_10;

        scalar_t sum_v_95 = scalar_t(0);
        sum_v_95 += wi_29 * xj_1 * yk_8;

        scalar_t sum_v_87 = scalar_t(0);
        sum_v_87 += wi_29 * xj_1 * yk_6;

        scalar_t sum_v_79 = scalar_t(0);
        sum_v_79 += wi_29 * xj_1 * yk_4;

        scalar_t sum_v_84 = scalar_t(0);
        sum_v_84 += wi_30 * xj_2 * yk_5;

        scalar_t sum_v_92 = scalar_t(0);
        sum_v_92 += wi_30 * xj_2 * yk_7;

        scalar_t sum_v_140 = scalar_t(0);
        sum_v_140 += wi_42 * xj_2 * yk_9;

        scalar_t sum_v_148 = scalar_t(0);
        sum_v_148 += wi_42 * xj_2 * yk_11;

        scalar_t sum_v_156 = scalar_t(0);
        sum_v_156 += wi_42 * xj_2 * yk_13;

        scalar_t sum_v_164 = scalar_t(0);
        sum_v_164 += wi_42 * xj_2 * yk_15;

        scalar_t sum_v_161 = scalar_t(0);
        sum_v_161 += wi_43 * xj_3 * yk_14;

        scalar_t sum_v_153 = scalar_t(0);
        sum_v_153 += wi_43 * xj_3 * yk_12;

        scalar_t sum_v_145 = scalar_t(0);
        sum_v_145 += wi_43 * xj_3 * yk_10;

        scalar_t sum_v_97 = scalar_t(0);
        sum_v_97 += wi_31 * xj_3 * yk_8;

        scalar_t sum_v_89 = scalar_t(0);
        sum_v_89 += wi_31 * xj_3 * yk_6;

        scalar_t sum_v_81 = scalar_t(0);
        sum_v_81 += wi_31 * xj_3 * yk_4;

        scalar_t sum_v_35 = scalar_t(0);
        sum_v_35 += scalar_t(-0.119522862f) * wi_17 * xj_10 * yk_13;
        sum_v_35 += scalar_t(-0.462910056f) * wi_17 * xj_10 * yk_15;
        sum_v_35 += scalar_t(-0.292769998f) * wi_17 * xj_11 * yk_12;
        sum_v_35 += scalar_t(-0.377964467f) * wi_17 * xj_11 * yk_14;
        sum_v_35 += scalar_t(0.414039344f) * wi_17 * xj_12 * yk_11;
        sum_v_35 += scalar_t(0.377964467f) * wi_17 * xj_13 * yk_10;
        sum_v_35 += scalar_t(0.462910056f) * wi_17 * xj_14 * yk_9;
        sum_v_35 += scalar_t(0.119522862f) * wi_17 * xj_14 * yk_11;

        scalar_t sum_v_47 = scalar_t(0);
        sum_v_47 += scalar_t(-0.316227764f) * wi_21 * xj_10 * yk_5;
        sum_v_47 += scalar_t(0.316227764f) * wi_21 * xj_11 * yk_4;
        sum_v_47 += scalar_t(0.547722578f) * wi_21 * xj_12 * yk_7;
        sum_v_47 += scalar_t(-0.547722578f) * wi_21 * xj_13 * yk_6;
        sum_v_47 += scalar_t(0.316227764f) * wi_21 * xj_13 * yk_8;
        sum_v_47 += scalar_t(-0.316227764f) * wi_21 * xj_14 * yk_7;

        scalar_t sum_v_6 = scalar_t(0);
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_10 * yk_4;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_11 * yk_5;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_12 * yk_6;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_13 * yk_7;
        sum_v_6 += scalar_t(0.44721359f) * wi_6 * xj_14 * yk_8;

        scalar_t sum_v_124 = scalar_t(0);
        sum_v_124 += scalar_t(0.462910056f) * wi_37 * xj_10 * yk_7;
        sum_v_124 += scalar_t(0.267261237f) * wi_37 * xj_11 * yk_6;
        sum_v_124 += scalar_t(-0.462910056f) * wi_37 * xj_11 * yk_8;
        sum_v_124 += scalar_t(0.267261237f) * wi_37 * xj_12 * yk_5;
        sum_v_124 += scalar_t(0.462910056f) * wi_37 * xj_13 * yk_4;
        sum_v_124 += scalar_t(-0.462910056f) * wi_37 * xj_14 * yk_5;

        scalar_t sum_v_224 = scalar_t(0);
        sum_v_224 += scalar_t(0.387298346f) * wi_52 * xj_10 * yk_5;
        sum_v_224 += scalar_t(-0.387298346f) * wi_52 * xj_11 * yk_4;
        sum_v_224 += scalar_t(0.44721359f) * wi_52 * xj_12 * yk_7;
        sum_v_224 += scalar_t(-0.44721359f) * wi_52 * xj_13 * yk_6;
        sum_v_224 += scalar_t(-0.387298346f) * wi_52 * xj_13 * yk_8;
        sum_v_224 += scalar_t(0.387298346f) * wi_52 * xj_14 * yk_7;

        scalar_t sum_v_225 = scalar_t(0);
        sum_v_225 += scalar_t(0.316227764f) * wi_52 * xj_10 * yk_8;
        sum_v_225 += scalar_t(-0.632455528f) * wi_52 * xj_11 * yk_7;
        sum_v_225 += scalar_t(0.632455528f) * wi_52 * xj_13 * yk_5;
        sum_v_225 += scalar_t(-0.316227764f) * wi_52 * xj_14 * yk_4;

        scalar_t sum_v_228 = scalar_t(0);
        sum_v_228 += scalar_t(0.5f) * wi_52 * xj_10 * yk_7;
        sum_v_228 += scalar_t(-0.5f) * wi_52 * xj_11 * yk_8;
        sum_v_228 += scalar_t(-0.5f) * wi_52 * xj_13 * yk_4;
        sum_v_228 += scalar_t(0.5f) * wi_52 * xj_14 * yk_5;

        scalar_t sum_v_193 = scalar_t(0);
        sum_v_193 += scalar_t(0.288675129f) * wi_47 * xj_10 * yk_11;
        sum_v_193 += scalar_t(-0.456435472f) * wi_47 * xj_11 * yk_10;
        sum_v_193 += scalar_t(-0.645497203f) * wi_47 * xj_12 * yk_15;
        sum_v_193 += scalar_t(0.456435472f) * wi_47 * xj_13 * yk_14;
        sum_v_193 += scalar_t(-0.288675129f) * wi_47 * xj_14 * yk_13;

        scalar_t sum_v_189 = scalar_t(0);
        sum_v_189 += scalar_t(0.44721359f) * wi_47 * xj_10 * yk_13;
        sum_v_189 += scalar_t(0.288675129f) * wi_47 * xj_10 * yk_15;
        sum_v_189 += scalar_t(0.182574183f) * wi_47 * xj_11 * yk_12;
        sum_v_189 += scalar_t(-0.353553385f) * wi_47 * xj_11 * yk_14;
        sum_v_189 += scalar_t(0.387298346f) * wi_47 * xj_12 * yk_11;
        sum_v_189 += scalar_t(0.353553385f) * wi_47 * xj_13 * yk_10;
        sum_v_189 += scalar_t(-0.288675129f) * wi_47 * xj_14 * yk_9;
        sum_v_189 += scalar_t(-0.44721359f) * wi_47 * xj_14 * yk_11;

        scalar_t sum_v_187 = scalar_t(0);
        sum_v_187 += scalar_t(-0.288675129f) * wi_47 * xj_10 * yk_13;
        sum_v_187 += scalar_t(0.456435472f) * wi_47 * xj_11 * yk_14;
        sum_v_187 += scalar_t(-0.645497203f) * wi_47 * xj_12 * yk_9;
        sum_v_187 += scalar_t(0.456435472f) * wi_47 * xj_13 * yk_10;
        sum_v_187 += scalar_t(-0.288675129f) * wi_47 * xj_14 * yk_11;

        scalar_t sum_v_182 = scalar_t(0);
        sum_v_182 += scalar_t(-0.182574183f) * wi_46 * xj_10 * yk_3;
        sum_v_182 += scalar_t(0.730296731f) * wi_46 * xj_11 * yk_2;
        sum_v_182 += scalar_t(0.632455528f) * wi_46 * xj_12 * yk_1;
        sum_v_182 += scalar_t(0.182574183f) * wi_46 * xj_14 * yk_1;

        scalar_t sum_v_32 = scalar_t(0);
        sum_v_32 += scalar_t(0.547722578f) * wi_16 * xj_10 * yk_3;
        sum_v_32 += scalar_t(0.547722578f) * wi_16 * xj_11 * yk_2;
        sum_v_32 += scalar_t(-0.316227764f) * wi_16 * xj_12 * yk_1;
        sum_v_32 += scalar_t(-0.547722578f) * wi_16 * xj_14 * yk_1;

        scalar_t sum_v_64 = scalar_t(0);
        sum_v_64 += scalar_t(0.408248276f) * wi_25 * xj_10 * yk_1;
        sum_v_64 += scalar_t(0.707106769f) * wi_25 * xj_12 * yk_3;
        sum_v_64 += scalar_t(-0.408248276f) * wi_25 * xj_13 * yk_2;
        sum_v_64 += scalar_t(0.408248276f) * wi_25 * xj_14 * yk_3;

        scalar_t sum_v_69 = scalar_t(0);
        sum_v_69 += scalar_t(-0.422577113f) * wi_26 * xj_10 * yk_9;
        sum_v_69 += scalar_t(-0.327326834f) * wi_26 * xj_10 * yk_11;
        sum_v_69 += scalar_t(0.377964467f) * wi_26 * xj_12 * yk_13;
        sum_v_69 += scalar_t(-0.534522474f) * wi_26 * xj_13 * yk_12;
        sum_v_69 += scalar_t(-0.327326834f) * wi_26 * xj_14 * yk_13;
        sum_v_69 += scalar_t(-0.422577113f) * wi_26 * xj_14 * yk_15;

        scalar_t sum_v_71 = scalar_t(0);
        sum_v_71 += scalar_t(0.327326834f) * wi_26 * xj_10 * yk_13;
        sum_v_71 += scalar_t(-0.422577113f) * wi_26 * xj_10 * yk_15;
        sum_v_71 += scalar_t(0.534522474f) * wi_26 * xj_11 * yk_12;
        sum_v_71 += scalar_t(-0.377964467f) * wi_26 * xj_12 * yk_11;
        sum_v_71 += scalar_t(0.422577113f) * wi_26 * xj_14 * yk_9;
        sum_v_71 += scalar_t(-0.327326834f) * wi_26 * xj_14 * yk_11;

        scalar_t sum_v_48 = scalar_t(0);
        sum_v_48 += scalar_t(-0.632455528f) * wi_21 * xj_10 * yk_8;
        sum_v_48 += scalar_t(-0.316227764f) * wi_21 * xj_11 * yk_7;
        sum_v_48 += scalar_t(0.316227764f) * wi_21 * xj_13 * yk_5;
        sum_v_48 += scalar_t(0.632455528f) * wi_21 * xj_14 * yk_4;

        scalar_t sum_v_127 = scalar_t(0);
        sum_v_127 += scalar_t(-0.462910056f) * wi_37 * xj_11 * yk_5;
        sum_v_127 += scalar_t(-0.534522474f) * wi_37 * xj_12 * yk_8;
        sum_v_127 += scalar_t(0.462910056f) * wi_37 * xj_13 * yk_7;
        sum_v_127 += scalar_t(-0.534522474f) * wi_37 * xj_14 * yk_6;

        scalar_t sum_v_181 = scalar_t(0);
        sum_v_181 += scalar_t(0.577350259f) * wi_46 * xj_10 * yk_2;
        sum_v_181 += scalar_t(0.577350259f) * wi_46 * xj_11 * yk_3;
        sum_v_181 += scalar_t(0.577350259f) * wi_46 * xj_13 * yk_1;

        scalar_t sum_v_192 = scalar_t(0);
        sum_v_192 += scalar_t(0.456435472f) * wi_47 * xj_11 * yk_9;
        sum_v_192 += scalar_t(-0.353553385f) * wi_47 * xj_11 * yk_11;
        sum_v_192 += scalar_t(0.353553385f) * wi_47 * xj_13 * yk_13;
        sum_v_192 += scalar_t(0.456435472f) * wi_47 * xj_13 * yk_15;
        sum_v_192 += scalar_t(-0.577350259f) * wi_47 * xj_14 * yk_12;

        scalar_t sum_v_63 = scalar_t(0);
        sum_v_63 += scalar_t(-0.408248276f) * wi_25 * xj_11 * yk_1;
        sum_v_63 += scalar_t(0.408248276f) * wi_25 * xj_13 * yk_3;
        sum_v_63 += scalar_t(-0.816496551f) * wi_25 * xj_14 * yk_2;

        scalar_t sum_v_33 = scalar_t(0);
        sum_v_33 += scalar_t(0.547722578f) * wi_16 * xj_11 * yk_1;
        sum_v_33 += scalar_t(0.632455528f) * wi_16 * xj_12 * yk_2;
        sum_v_33 += scalar_t(0.547722578f) * wi_16 * xj_13 * yk_3;

        scalar_t sum_v_118 = scalar_t(0);
        sum_v_118 += wi_36 * xj_10 * yk_0;

        scalar_t sum_v_121 = scalar_t(0);
        sum_v_121 += wi_36 * xj_13 * yk_0;

        scalar_t sum_v_180 = scalar_t(0);
        sum_v_180 += scalar_t(0.707106769f) * wi_46 * xj_10 * yk_3;
        sum_v_180 += scalar_t(0.707106769f) * wi_46 * xj_14 * yk_1;

        scalar_t sum_v_223 = scalar_t(0);
        sum_v_223 += scalar_t(0.707106769f) * wi_52 * xj_12 * yk_8;
        sum_v_223 += scalar_t(-0.707106769f) * wi_52 * xj_14 * yk_6;

        scalar_t sum_v_211 = scalar_t(0);
        sum_v_211 += scalar_t(-0.353553385f) * wi_51 * xj_5 * yk_9;
        sum_v_211 += scalar_t(0.456435472f) * wi_51 * xj_5 * yk_11;
        sum_v_211 += scalar_t(0.577350259f) * wi_51 * xj_7 * yk_14;
        sum_v_211 += scalar_t(-0.456435472f) * wi_51 * xj_9 * yk_13;
        sum_v_211 += scalar_t(-0.353553385f) * wi_51 * xj_9 * yk_15;

        scalar_t sum_v_217 = scalar_t(0);
        sum_v_217 += scalar_t(0.707106769f) * wi_51 * xj_5 * yk_12;
        sum_v_217 += scalar_t(-0.456435472f) * wi_51 * xj_5 * yk_14;
        sum_v_217 += scalar_t(-0.288675129f) * wi_51 * xj_7 * yk_11;
        sum_v_217 += scalar_t(0.456435472f) * wi_51 * xj_9 * yk_10;

        scalar_t sum_v_209 = scalar_t(0);
        sum_v_209 += scalar_t(0.353553385f) * wi_51 * xj_5 * yk_10;
        sum_v_209 += scalar_t(0.866025388f) * wi_51 * xj_7 * yk_15;
        sum_v_209 += scalar_t(-0.353553385f) * wi_51 * xj_9 * yk_14;

        scalar_t sum_v_171 = scalar_t(0);
        sum_v_171 += scalar_t(0.632455528f) * wi_45 * xj_5 * yk_6;
        sum_v_171 += scalar_t(0.182574183f) * wi_45 * xj_5 * yk_8;
        sum_v_171 += scalar_t(0.730296731f) * wi_45 * xj_7 * yk_5;
        sum_v_171 += scalar_t(-0.182574183f) * wi_45 * xj_9 * yk_4;

        scalar_t sum_v_169 = scalar_t(0);
        sum_v_169 += scalar_t(0.577350259f) * wi_45 * xj_5 * yk_7;
        sum_v_169 += scalar_t(0.577350259f) * wi_45 * xj_7 * yk_4;
        sum_v_169 += scalar_t(0.577350259f) * wi_45 * xj_9 * yk_5;

        scalar_t sum_v_167 = scalar_t(0);
        sum_v_167 += scalar_t(0.707106769f) * wi_45 * xj_5 * yk_8;
        sum_v_167 += scalar_t(0.707106769f) * wi_45 * xj_9 * yk_4;

        scalar_t sum_v_215 = scalar_t(0);
        sum_v_215 += scalar_t(-0.707106769f) * wi_51 * xj_5 * yk_13;
        sum_v_215 += scalar_t(0.707106769f) * wi_51 * xj_9 * yk_11;

        scalar_t sum_v_197 = scalar_t(0);
        sum_v_197 += wi_48 * xj_18 * yk_0;

        scalar_t sum_v_195 = scalar_t(0);
        sum_v_195 += wi_48 * xj_16 * yk_0;

        // writeback
        atomicAdd(&out[o_base + (52LL << 5)], sum_v_52);
        atomicAdd(&out[o_base + (136LL << 5)], sum_v_136);
        atomicAdd(&out[o_base + (237LL << 5)], sum_v_237);
        atomicAdd(&out[o_base + (236LL << 5)], sum_v_236);
        atomicAdd(&out[o_base + (239LL << 5)], sum_v_239);
        atomicAdd(&out[o_base + (241LL << 5)], sum_v_241);
        atomicAdd(&out[o_base + (205LL << 5)], sum_v_205);
        atomicAdd(&out[o_base + (207LL << 5)], sum_v_207);
        atomicAdd(&out[o_base + (40LL << 5)], sum_v_40);
        atomicAdd(&out[o_base + (73LL << 5)], sum_v_73);
        atomicAdd(&out[o_base + (76LL << 5)], sum_v_76);
        atomicAdd(&out[o_base + (133LL << 5)], sum_v_133);
        atomicAdd(&out[o_base + (202LL << 5)], sum_v_202);
        atomicAdd(&out[o_base + (230LL << 5)], sum_v_230);
        atomicAdd(&out[o_base + (128LL << 5)], sum_v_128);
        atomicAdd(&out[o_base + (51LL << 5)], sum_v_51);
        atomicAdd(&out[o_base + (39LL << 5)], sum_v_39);
        atomicAdd(&out[o_base + (129LL << 5)], sum_v_129);
        atomicAdd(&out[o_base + (231LL << 5)], sum_v_231);
        atomicAdd(&out[o_base + (229LL << 5)], sum_v_229);
        atomicAdd(&out[o_base + (198LL << 5)], sum_v_198);
        atomicAdd(&out[o_base + (232LL << 5)], sum_v_232);
        atomicAdd(&out[o_base + (103LL << 5)], sum_v_103);
        atomicAdd(&out[o_base + (117LL << 5)], sum_v_117);
        atomicAdd(&out[o_base + (115LL << 5)], sum_v_115);
        atomicAdd(&out[o_base + (62LL << 5)], sum_v_62);
        atomicAdd(&out[o_base + (60LL << 5)], sum_v_60);
        atomicAdd(&out[o_base + (31LL << 5)], sum_v_31);
        atomicAdd(&out[o_base + (54LL << 5)], sum_v_54);
        atomicAdd(&out[o_base + (58LL << 5)], sum_v_58);
        atomicAdd(&out[o_base + (42LL << 5)], sum_v_42);
        atomicAdd(&out[o_base + (107LL << 5)], sum_v_107);
        atomicAdd(&out[o_base + (101LL << 5)], sum_v_101);
        atomicAdd(&out[o_base + (22LL << 5)], sum_v_22);
        atomicAdd(&out[o_base + (26LL << 5)], sum_v_26);
        atomicAdd(&out[o_base + (55LL << 5)], sum_v_55);
        atomicAdd(&out[o_base + (61LL << 5)], sum_v_61);
        atomicAdd(&out[o_base + (28LL << 5)], sum_v_28);
        atomicAdd(&out[o_base + (102LL << 5)], sum_v_102);
        atomicAdd(&out[o_base + (116LL << 5)], sum_v_116);
        atomicAdd(&out[o_base + (114LL << 5)], sum_v_114);
        atomicAdd(&out[o_base + (168LL << 5)], sum_v_168);
        atomicAdd(&out[o_base + (170LL << 5)], sum_v_170);
        atomicAdd(&out[o_base + (176LL << 5)], sum_v_176);
        atomicAdd(&out[o_base + (218LL << 5)], sum_v_218);
        atomicAdd(&out[o_base + (216LL << 5)], sum_v_216);
        atomicAdd(&out[o_base + (220LL << 5)], sum_v_220);
        atomicAdd(&out[o_base + (214LL << 5)], sum_v_214);
        atomicAdd(&out[o_base + (166LL << 5)], sum_v_166);
        atomicAdd(&out[o_base + (98LL << 5)], sum_v_98);
        atomicAdd(&out[o_base + (57LL << 5)], sum_v_57);
        atomicAdd(&out[o_base + (100LL << 5)], sum_v_100);
        atomicAdd(&out[o_base + (41LL << 5)], sum_v_41);
        atomicAdd(&out[o_base + (25LL << 5)], sum_v_25);
        atomicAdd(&out[o_base + (20LL << 5)], sum_v_20);
        atomicAdd(&out[o_base + (11LL << 5)], sum_v_11);
        atomicAdd(&out[o_base + (19LL << 5)], sum_v_19);
        atomicAdd(&out[o_base + (14LL << 5)], sum_v_14);
        atomicAdd(&out[o_base + (2LL << 5)], sum_v_2);
        atomicAdd(&out[o_base + (9LL << 5)], sum_v_9);
        atomicAdd(&out[o_base + (17LL << 5)], sum_v_17);
        atomicAdd(&out[o_base + (12LL << 5)], sum_v_12);
        atomicAdd(&out[o_base + (0LL << 5)], sum_v_0);
        atomicAdd(&out[o_base + (82LL << 5)], sum_v_82);
        atomicAdd(&out[o_base + (90LL << 5)], sum_v_90);
        atomicAdd(&out[o_base + (138LL << 5)], sum_v_138);
        atomicAdd(&out[o_base + (146LL << 5)], sum_v_146);
        atomicAdd(&out[o_base + (154LL << 5)], sum_v_154);
        atomicAdd(&out[o_base + (162LL << 5)], sum_v_162);
        atomicAdd(&out[o_base + (159LL << 5)], sum_v_159);
        atomicAdd(&out[o_base + (151LL << 5)], sum_v_151);
        atomicAdd(&out[o_base + (143LL << 5)], sum_v_143);
        atomicAdd(&out[o_base + (95LL << 5)], sum_v_95);
        atomicAdd(&out[o_base + (87LL << 5)], sum_v_87);
        atomicAdd(&out[o_base + (79LL << 5)], sum_v_79);
        atomicAdd(&out[o_base + (84LL << 5)], sum_v_84);
        atomicAdd(&out[o_base + (92LL << 5)], sum_v_92);
        atomicAdd(&out[o_base + (140LL << 5)], sum_v_140);
        atomicAdd(&out[o_base + (148LL << 5)], sum_v_148);
        atomicAdd(&out[o_base + (156LL << 5)], sum_v_156);
        atomicAdd(&out[o_base + (164LL << 5)], sum_v_164);
        atomicAdd(&out[o_base + (161LL << 5)], sum_v_161);
        atomicAdd(&out[o_base + (153LL << 5)], sum_v_153);
        atomicAdd(&out[o_base + (145LL << 5)], sum_v_145);
        atomicAdd(&out[o_base + (97LL << 5)], sum_v_97);
        atomicAdd(&out[o_base + (89LL << 5)], sum_v_89);
        atomicAdd(&out[o_base + (81LL << 5)], sum_v_81);
        atomicAdd(&out[o_base + (35LL << 5)], sum_v_35);
        atomicAdd(&out[o_base + (47LL << 5)], sum_v_47);
        atomicAdd(&out[o_base + (6LL << 5)], sum_v_6);
        atomicAdd(&out[o_base + (124LL << 5)], sum_v_124);
        atomicAdd(&out[o_base + (224LL << 5)], sum_v_224);
        atomicAdd(&out[o_base + (225LL << 5)], sum_v_225);
        atomicAdd(&out[o_base + (228LL << 5)], sum_v_228);
        atomicAdd(&out[o_base + (193LL << 5)], sum_v_193);
        atomicAdd(&out[o_base + (189LL << 5)], sum_v_189);
        atomicAdd(&out[o_base + (187LL << 5)], sum_v_187);
        atomicAdd(&out[o_base + (182LL << 5)], sum_v_182);
        atomicAdd(&out[o_base + (32LL << 5)], sum_v_32);
        atomicAdd(&out[o_base + (64LL << 5)], sum_v_64);
        atomicAdd(&out[o_base + (69LL << 5)], sum_v_69);
        atomicAdd(&out[o_base + (71LL << 5)], sum_v_71);
        atomicAdd(&out[o_base + (48LL << 5)], sum_v_48);
        atomicAdd(&out[o_base + (127LL << 5)], sum_v_127);
        atomicAdd(&out[o_base + (181LL << 5)], sum_v_181);
        atomicAdd(&out[o_base + (192LL << 5)], sum_v_192);
        atomicAdd(&out[o_base + (63LL << 5)], sum_v_63);
        atomicAdd(&out[o_base + (33LL << 5)], sum_v_33);
        atomicAdd(&out[o_base + (118LL << 5)], sum_v_118);
        atomicAdd(&out[o_base + (121LL << 5)], sum_v_121);
        atomicAdd(&out[o_base + (180LL << 5)], sum_v_180);
        atomicAdd(&out[o_base + (223LL << 5)], sum_v_223);
        atomicAdd(&out[o_base + (211LL << 5)], sum_v_211);
        atomicAdd(&out[o_base + (217LL << 5)], sum_v_217);
        atomicAdd(&out[o_base + (209LL << 5)], sum_v_209);
        atomicAdd(&out[o_base + (171LL << 5)], sum_v_171);
        atomicAdd(&out[o_base + (169LL << 5)], sum_v_169);
        atomicAdd(&out[o_base + (167LL << 5)], sum_v_167);
        atomicAdd(&out[o_base + (215LL << 5)], sum_v_215);
        atomicAdd(&out[o_base + (197LL << 5)], sum_v_197);
        atomicAdd(&out[o_base + (195LL << 5)], sum_v_195);
    }

}

// launcher helper
template <typename scalar_t>
void launch_uniform1d_codegen_two_warp_vgroup_path777(
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
    uniform1d_codegen_two_warp_vgroup_path777<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, out, src_idx, dst_idx, b_list,
        B, Iw, Ix, Ky, V);
}