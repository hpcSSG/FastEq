#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cstdint>
#include "../cuda_utils.hpp"

template <typename scalar_t>
__global__ void uniform1d_combine_u32_path1490_fwd(
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

    int64_t w_base = (int64_t)b   * Iw * 32;
    int64_t x_base = (int64_t)src * Ix * 32;
    int64_t y_base = (int64_t)b   * Ky;
    int64_t o_base = ((int64_t)dst * V) * 32 + lane;

    if (warp == 0) {
        // preload w(i)
        scalar_t wi_26 = w[w_base + 26LL * 32 + lane];
        scalar_t wi_3 = w[w_base + 3LL * 32 + lane];
        scalar_t wi_50 = w[w_base + 50LL * 32 + lane];
        scalar_t wi_37 = w[w_base + 37LL * 32 + lane];
        scalar_t wi_82 = w[w_base + 82LL * 32 + lane];
        scalar_t wi_93 = w[w_base + 93LL * 32 + lane];
        scalar_t wi_67 = w[w_base + 67LL * 32 + lane];
        scalar_t wi_49 = w[w_base + 49LL * 32 + lane];
        scalar_t wi_81 = w[w_base + 81LL * 32 + lane];
        scalar_t wi_24 = w[w_base + 24LL * 32 + lane];
        scalar_t wi_35 = w[w_base + 35LL * 32 + lane];
        scalar_t wi_10 = w[w_base + 10LL * 32 + lane];
        scalar_t wi_64 = w[w_base + 64LL * 32 + lane];
        scalar_t wi_47 = w[w_base + 47LL * 32 + lane];
        scalar_t wi_78 = w[w_base + 78LL * 32 + lane];
        scalar_t wi_89 = w[w_base + 89LL * 32 + lane];
        scalar_t wi_77 = w[w_base + 77LL * 32 + lane];
        scalar_t wi_46 = w[w_base + 46LL * 32 + lane];
        scalar_t wi_23 = w[w_base + 23LL * 32 + lane];
        scalar_t wi_43 = w[w_base + 43LL * 32 + lane];
        scalar_t wi_32 = w[w_base + 32LL * 32 + lane];
        scalar_t wi_1 = w[w_base + 1LL * 32 + lane];
        scalar_t wi_21 = w[w_base + 21LL * 32 + lane];
        scalar_t wi_41 = w[w_base + 41LL * 32 + lane];
        scalar_t wi_60 = w[w_base + 60LL * 32 + lane];
        scalar_t wi_75 = w[w_base + 75LL * 32 + lane];
        scalar_t wi_86 = w[w_base + 86LL * 32 + lane];
        scalar_t wi_74 = w[w_base + 74LL * 32 + lane];
        scalar_t wi_85 = w[w_base + 85LL * 32 + lane];
        scalar_t wi_59 = w[w_base + 59LL * 32 + lane];
        scalar_t wi_42 = w[w_base + 42LL * 32 + lane];
        scalar_t wi_40 = w[w_base + 40LL * 32 + lane];
        scalar_t wi_31 = w[w_base + 31LL * 32 + lane];
        scalar_t wi_0 = w[w_base + 0LL * 32 + lane];
        scalar_t wi_20 = w[w_base + 20LL * 32 + lane];
        scalar_t wi_29 = w[w_base + 29LL * 32 + lane];
        scalar_t wi_19 = w[w_base + 19LL * 32 + lane];
        scalar_t wi_39 = w[w_base + 39LL * 32 + lane];
        scalar_t wi_28 = w[w_base + 28LL * 32 + lane];
        scalar_t wi_58 = w[w_base + 58LL * 32 + lane];
        scalar_t wi_73 = w[w_base + 73LL * 32 + lane];
        scalar_t wi_84 = w[w_base + 84LL * 32 + lane];
        scalar_t wi_83 = w[w_base + 83LL * 32 + lane];
        scalar_t wi_72 = w[w_base + 72LL * 32 + lane];
        scalar_t wi_57 = w[w_base + 57LL * 32 + lane];
        scalar_t wi_55 = w[w_base + 55LL * 32 + lane];
        scalar_t wi_38 = w[w_base + 38LL * 32 + lane];
        scalar_t wi_18 = w[w_base + 18LL * 32 + lane];
        scalar_t wi_8 = w[w_base + 8LL * 32 + lane];
        scalar_t wi_27 = w[w_base + 27LL * 32 + lane];
        scalar_t wi_16 = w[w_base + 16LL * 32 + lane];
        scalar_t wi_7 = w[w_base + 7LL * 32 + lane];
        scalar_t wi_15 = w[w_base + 15LL * 32 + lane];
        scalar_t wi_14 = w[w_base + 14LL * 32 + lane];
        scalar_t wi_5 = w[w_base + 5LL * 32 + lane];
        scalar_t wi_13 = w[w_base + 13LL * 32 + lane];
        scalar_t wi_12 = w[w_base + 12LL * 32 + lane];
        scalar_t wi_22 = w[w_base + 22LL * 32 + lane];
        scalar_t wi_45 = w[w_base + 45LL * 32 + lane];
        scalar_t wi_34 = w[w_base + 34LL * 32 + lane];
        scalar_t wi_33 = w[w_base + 33LL * 32 + lane];
        scalar_t wi_61 = w[w_base + 61LL * 32 + lane];
        scalar_t wi_62 = w[w_base + 62LL * 32 + lane];
        scalar_t wi_76 = w[w_base + 76LL * 32 + lane];
        scalar_t wi_88 = w[w_base + 88LL * 32 + lane];
        scalar_t wi_87 = w[w_base + 87LL * 32 + lane];
        scalar_t wi_25 = w[w_base + 25LL * 32 + lane];
        scalar_t wi_36 = w[w_base + 36LL * 32 + lane];
        scalar_t wi_66 = w[w_base + 66LL * 32 + lane];
        scalar_t wi_91 = w[w_base + 91LL * 32 + lane];
        scalar_t wi_80 = w[w_base + 80LL * 32 + lane];
        scalar_t wi_48 = w[w_base + 48LL * 32 + lane];
        scalar_t wi_65 = w[w_base + 65LL * 32 + lane];
        scalar_t wi_90 = w[w_base + 90LL * 32 + lane];
        scalar_t wi_71 = w[w_base + 71LL * 32 + lane];
        scalar_t wi_70 = w[w_base + 70LL * 32 + lane];
        scalar_t wi_69 = w[w_base + 69LL * 32 + lane];
        scalar_t wi_68 = w[w_base + 68LL * 32 + lane];
        scalar_t wi_51 = w[w_base + 51LL * 32 + lane];
        scalar_t wi_52 = w[w_base + 52LL * 32 + lane];
        scalar_t wi_53 = w[w_base + 53LL * 32 + lane];
        scalar_t wi_54 = w[w_base + 54LL * 32 + lane];
        scalar_t wi_56 = w[w_base + 56LL * 32 + lane];
        scalar_t wi_63 = w[w_base + 63LL * 32 + lane];
        scalar_t wi_92 = w[w_base + 92LL * 32 + lane];
        scalar_t wi_79 = w[w_base + 79LL * 32 + lane];
        scalar_t wi_44 = w[w_base + 44LL * 32 + lane];

        // preload x(j)
        scalar_t xj_33 = x_all[x_base + 33LL * 32 + lane];
        scalar_t xj_34 = x_all[x_base + 34LL * 32 + lane];
        scalar_t xj_35 = x_all[x_base + 35LL * 32 + lane];
        scalar_t xj_36 = x_all[x_base + 36LL * 32 + lane];
        scalar_t xj_37 = x_all[x_base + 37LL * 32 + lane];
        scalar_t xj_38 = x_all[x_base + 38LL * 32 + lane];
        scalar_t xj_39 = x_all[x_base + 39LL * 32 + lane];
        scalar_t xj_21 = x_all[x_base + 21LL * 32 + lane];
        scalar_t xj_22 = x_all[x_base + 22LL * 32 + lane];
        scalar_t xj_23 = x_all[x_base + 23LL * 32 + lane];
        scalar_t xj_24 = x_all[x_base + 24LL * 32 + lane];
        scalar_t xj_25 = x_all[x_base + 25LL * 32 + lane];
        scalar_t xj_11 = x_all[x_base + 11LL * 32 + lane];
        scalar_t xj_13 = x_all[x_base + 13LL * 32 + lane];
        scalar_t xj_15 = x_all[x_base + 15LL * 32 + lane];
        scalar_t xj_10 = x_all[x_base + 10LL * 32 + lane];
        scalar_t xj_14 = x_all[x_base + 14LL * 32 + lane];
        scalar_t xj_12 = x_all[x_base + 12LL * 32 + lane];
        scalar_t xj_5 = x_all[x_base + 5LL * 32 + lane];
        scalar_t xj_7 = x_all[x_base + 7LL * 32 + lane];
        scalar_t xj_9 = x_all[x_base + 9LL * 32 + lane];
        scalar_t xj_4 = x_all[x_base + 4LL * 32 + lane];
        scalar_t xj_8 = x_all[x_base + 8LL * 32 + lane];
        scalar_t xj_6 = x_all[x_base + 6LL * 32 + lane];
        scalar_t xj_3 = x_all[x_base + 3LL * 32 + lane];
        scalar_t xj_2 = x_all[x_base + 2LL * 32 + lane];
        scalar_t xj_1 = x_all[x_base + 1LL * 32 + lane];
        scalar_t xj_0 = x_all[x_base + 0LL * 32 + lane];
        scalar_t xj_16 = x_all[x_base + 16LL * 32 + lane];
        scalar_t xj_17 = x_all[x_base + 17LL * 32 + lane];
        scalar_t xj_18 = x_all[x_base + 18LL * 32 + lane];
        scalar_t xj_19 = x_all[x_base + 19LL * 32 + lane];
        scalar_t xj_20 = x_all[x_base + 20LL * 32 + lane];
        scalar_t xj_26 = x_all[x_base + 26LL * 32 + lane];
        scalar_t xj_27 = x_all[x_base + 27LL * 32 + lane];
        scalar_t xj_28 = x_all[x_base + 28LL * 32 + lane];
        scalar_t xj_29 = x_all[x_base + 29LL * 32 + lane];
        scalar_t xj_30 = x_all[x_base + 30LL * 32 + lane];
        scalar_t xj_31 = x_all[x_base + 31LL * 32 + lane];
        scalar_t xj_32 = x_all[x_base + 32LL * 32 + lane];

        // preload y(k)
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_54 = scalar_t(0);
        sum_v_54 += scalar_t(-0.231455028f) * wi_26 * xj_33 * yk_10;
        sum_v_54 += scalar_t(0.231455028f) * wi_26 * xj_34 * yk_9;
        sum_v_54 += scalar_t(-0.298807144f) * wi_26 * xj_34 * yk_11;
        sum_v_54 += scalar_t(0.298807144f) * wi_26 * xj_35 * yk_10;
        sum_v_54 += scalar_t(0.462910056f) * wi_26 * xj_36 * yk_13;
        sum_v_54 += scalar_t(-0.462910056f) * wi_26 * xj_37 * yk_12;
        sum_v_54 += scalar_t(0.298807144f) * wi_26 * xj_37 * yk_14;
        sum_v_54 += scalar_t(-0.298807144f) * wi_26 * xj_38 * yk_13;
        sum_v_54 += scalar_t(0.231455028f) * wi_26 * xj_38 * yk_15;
        sum_v_54 += scalar_t(-0.231455028f) * wi_26 * xj_39 * yk_14;

        scalar_t sum_v_3 = scalar_t(0);
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_33 * yk_9;
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_34 * yk_10;
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_35 * yk_11;
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_36 * yk_12;
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_37 * yk_13;
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_38 * yk_14;
        sum_v_3 += scalar_t(0.377964467f) * wi_3 * xj_39 * yk_15;

        scalar_t sum_v_151 = scalar_t(0);
        sum_v_151 += scalar_t(0.38575837f) * wi_50 * xj_33 * yk_14;
        sum_v_151 += scalar_t(0.298807144f) * wi_50 * xj_34 * yk_13;
        sum_v_151 += scalar_t(-0.38575837f) * wi_50 * xj_34 * yk_15;
        sum_v_151 += scalar_t(0.154303357f) * wi_50 * xj_35 * yk_12;
        sum_v_151 += scalar_t(-0.298807144f) * wi_50 * xj_35 * yk_14;
        sum_v_151 += scalar_t(0.154303357f) * wi_50 * xj_36 * yk_11;
        sum_v_151 += scalar_t(0.298807144f) * wi_50 * xj_37 * yk_10;
        sum_v_151 += scalar_t(0.38575837f) * wi_50 * xj_38 * yk_9;
        sum_v_151 += scalar_t(-0.298807144f) * wi_50 * xj_38 * yk_11;
        sum_v_151 += scalar_t(-0.38575837f) * wi_50 * xj_39 * yk_10;

        scalar_t sum_v_154 = scalar_t(0);
        sum_v_154 += scalar_t(-0.243975028f) * wi_50 * xj_33 * yk_11;
        sum_v_154 += scalar_t(-0.243975028f) * wi_50 * xj_35 * yk_9;
        sum_v_154 += scalar_t(-0.377964467f) * wi_50 * xj_35 * yk_11;
        sum_v_154 += scalar_t(-0.487950057f) * wi_50 * xj_36 * yk_14;
        sum_v_154 += scalar_t(0.377964467f) * wi_50 * xj_37 * yk_13;
        sum_v_154 += scalar_t(-0.243975028f) * wi_50 * xj_37 * yk_15;
        sum_v_154 += scalar_t(-0.487950057f) * wi_50 * xj_38 * yk_12;
        sum_v_154 += scalar_t(-0.243975028f) * wi_50 * xj_39 * yk_13;

        scalar_t sum_v_152 = scalar_t(0);
        sum_v_152 += scalar_t(-0.545544744f) * wi_50 * xj_33 * yk_9;
        sum_v_152 += scalar_t(0.327326834f) * wi_50 * xj_35 * yk_11;
        sum_v_152 += scalar_t(0.436435789f) * wi_50 * xj_36 * yk_12;
        sum_v_152 += scalar_t(0.327326834f) * wi_50 * xj_37 * yk_13;
        sum_v_152 += scalar_t(-0.545544744f) * wi_50 * xj_39 * yk_15;

        scalar_t sum_v_89 = scalar_t(0);
        sum_v_89 += scalar_t(0.462910056f) * wi_37 * xj_33 * yk_4;
        sum_v_89 += scalar_t(0.377964467f) * wi_37 * xj_34 * yk_5;
        sum_v_89 += scalar_t(-0.119522862f) * wi_37 * xj_35 * yk_4;
        sum_v_89 += scalar_t(-0.292769998f) * wi_37 * xj_36 * yk_7;
        sum_v_89 += scalar_t(0.414039344f) * wi_37 * xj_37 * yk_6;
        sum_v_89 += scalar_t(-0.119522862f) * wi_37 * xj_37 * yk_8;
        sum_v_89 += scalar_t(0.377964467f) * wi_37 * xj_38 * yk_7;
        sum_v_89 += scalar_t(0.462910056f) * wi_37 * xj_39 * yk_8;

        scalar_t sum_v_55 = scalar_t(0);
        sum_v_55 += scalar_t(-0.566946685f) * wi_26 * xj_33 * yk_15;
        sum_v_55 += scalar_t(-0.377964497f) * wi_26 * xj_34 * yk_14;
        sum_v_55 += scalar_t(-0.188982248f) * wi_26 * xj_35 * yk_13;
        sum_v_55 += scalar_t(0.188982248f) * wi_26 * xj_37 * yk_11;
        sum_v_55 += scalar_t(0.377964497f) * wi_26 * xj_38 * yk_10;
        sum_v_55 += scalar_t(0.566946685f) * wi_26 * xj_39 * yk_9;

        scalar_t sum_v_340 = scalar_t(0);
        sum_v_340 += scalar_t(0.408248305f) * wi_82 * xj_33 * yk_10;
        sum_v_340 += scalar_t(-0.408248305f) * wi_82 * xj_34 * yk_9;
        sum_v_340 += scalar_t(0.408248305f) * wi_82 * xj_36 * yk_13;
        sum_v_340 += scalar_t(-0.408248305f) * wi_82 * xj_37 * yk_12;
        sum_v_340 += scalar_t(-0.408248305f) * wi_82 * xj_38 * yk_15;
        sum_v_340 += scalar_t(0.408248305f) * wi_82 * xj_39 * yk_14;

        scalar_t sum_v_338 = scalar_t(0);
        sum_v_338 += scalar_t(0.408248305f) * wi_82 * xj_34 * yk_11;
        sum_v_338 += scalar_t(-0.408248305f) * wi_82 * xj_35 * yk_10;
        sum_v_338 += scalar_t(-0.408248305f) * wi_82 * xj_36 * yk_15;
        sum_v_338 += scalar_t(0.408248305f) * wi_82 * xj_37 * yk_14;
        sum_v_338 += scalar_t(-0.408248305f) * wi_82 * xj_38 * yk_13;
        sum_v_338 += scalar_t(0.408248305f) * wi_82 * xj_39 * yk_12;

        scalar_t sum_v_343 = scalar_t(0);
        sum_v_343 += scalar_t(0.408248305f) * wi_82 * xj_33 * yk_13;
        sum_v_343 += scalar_t(0.408248305f) * wi_82 * xj_34 * yk_12;
        sum_v_343 += scalar_t(0.408248305f) * wi_82 * xj_35 * yk_15;
        sum_v_343 += scalar_t(-0.408248305f) * wi_82 * xj_36 * yk_10;
        sum_v_343 += scalar_t(-0.408248305f) * wi_82 * xj_37 * yk_9;
        sum_v_343 += scalar_t(-0.408248305f) * wi_82 * xj_39 * yk_11;

        scalar_t sum_v_417 = scalar_t(0);
        sum_v_417 += scalar_t(-0.288675129f) * wi_93 * xj_33 * yk_8;
        sum_v_417 += scalar_t(0.353553385f) * wi_93 * xj_34 * yk_7;
        sum_v_417 += scalar_t(0.387298346f) * wi_93 * xj_35 * yk_6;
        sum_v_417 += scalar_t(-0.44721356f) * wi_93 * xj_35 * yk_8;
        sum_v_417 += scalar_t(0.182574183f) * wi_93 * xj_36 * yk_5;
        sum_v_417 += scalar_t(0.44721356f) * wi_93 * xj_37 * yk_4;
        sum_v_417 += scalar_t(-0.353553385f) * wi_93 * xj_38 * yk_5;
        sum_v_417 += scalar_t(0.288675129f) * wi_93 * xj_39 * yk_4;

        scalar_t sum_v_418 = scalar_t(0);
        sum_v_418 += scalar_t(-0.577350259f) * wi_93 * xj_34 * yk_4;
        sum_v_418 += scalar_t(0.182574183f) * wi_93 * xj_35 * yk_5;
        sum_v_418 += scalar_t(0.516397774f) * wi_93 * xj_36 * yk_6;
        sum_v_418 += scalar_t(0.182574183f) * wi_93 * xj_37 * yk_7;
        sum_v_418 += scalar_t(-0.577350259f) * wi_93 * xj_38 * yk_8;

        scalar_t sum_v_421 = scalar_t(0);
        sum_v_421 += scalar_t(-0.456435442f) * wi_93 * xj_34 * yk_5;
        sum_v_421 += scalar_t(0.288675129f) * wi_93 * xj_35 * yk_4;
        sum_v_421 += scalar_t(-0.288675129f) * wi_93 * xj_37 * yk_8;
        sum_v_421 += scalar_t(0.456435442f) * wi_93 * xj_38 * yk_7;
        sum_v_421 += scalar_t(-0.645497203f) * wi_93 * xj_39 * yk_6;

        scalar_t sum_v_416 = scalar_t(0);
        sum_v_416 += scalar_t(0.456435442f) * wi_93 * xj_33 * yk_7;
        sum_v_416 += scalar_t(0.353553385f) * wi_93 * xj_35 * yk_7;
        sum_v_416 += scalar_t(-0.577350259f) * wi_93 * xj_36 * yk_4;
        sum_v_416 += scalar_t(0.353553385f) * wi_93 * xj_37 * yk_5;
        sum_v_416 += scalar_t(-0.456435442f) * wi_93 * xj_39 * yk_5;

        scalar_t sum_v_239 = scalar_t(0);
        sum_v_239 += scalar_t(0.422577113f) * wi_67 * xj_33 * yk_7;
        sum_v_239 += scalar_t(0.597614288f) * wi_67 * xj_34 * yk_6;
        sum_v_239 += scalar_t(-0.327326834f) * wi_67 * xj_35 * yk_7;
        sum_v_239 += scalar_t(0.267261237f) * wi_67 * xj_36 * yk_4;
        sum_v_239 += scalar_t(-0.327326834f) * wi_67 * xj_37 * yk_5;
        sum_v_239 += scalar_t(-0.422577113f) * wi_67 * xj_39 * yk_5;

        scalar_t sum_v_236 = scalar_t(0);
        sum_v_236 += scalar_t(0.422577113f) * wi_67 * xj_33 * yk_4;
        sum_v_236 += scalar_t(0.327326834f) * wi_67 * xj_35 * yk_4;
        sum_v_236 += scalar_t(0.534522474f) * wi_67 * xj_36 * yk_7;
        sum_v_236 += scalar_t(-0.377964467f) * wi_67 * xj_37 * yk_6;
        sum_v_236 += scalar_t(0.327326834f) * wi_67 * xj_37 * yk_8;
        sum_v_236 += scalar_t(0.422577113f) * wi_67 * xj_39 * yk_8;

        scalar_t sum_v_145 = scalar_t(0);
        sum_v_145 += scalar_t(0.597614288f) * wi_49 * xj_33 * yk_3;
        sum_v_145 += scalar_t(0.487950057f) * wi_49 * xj_34 * yk_2;
        sum_v_145 += scalar_t(-0.154303357f) * wi_49 * xj_35 * yk_3;
        sum_v_145 += scalar_t(-0.154303357f) * wi_49 * xj_37 * yk_1;
        sum_v_145 += scalar_t(-0.597614288f) * wi_49 * xj_39 * yk_1;

        scalar_t sum_v_332 = scalar_t(0);
        sum_v_332 += scalar_t(0.353553385f) * wi_81 * xj_33 * yk_1;
        sum_v_332 += scalar_t(-0.456435442f) * wi_81 * xj_35 * yk_1;
        sum_v_332 += scalar_t(0.456435442f) * wi_81 * xj_37 * yk_3;
        sum_v_332 += scalar_t(-0.577350259f) * wi_81 * xj_38 * yk_2;
        sum_v_332 += scalar_t(0.353553385f) * wi_81 * xj_39 * yk_3;

        scalar_t sum_v_333 = scalar_t(0);
        sum_v_333 += scalar_t(0.456435442f) * wi_81 * xj_34 * yk_1;
        sum_v_333 += scalar_t(0.707106769f) * wi_81 * xj_36 * yk_3;
        sum_v_333 += scalar_t(-0.288675129f) * wi_81 * xj_37 * yk_2;
        sum_v_333 += scalar_t(0.456435442f) * wi_81 * xj_38 * yk_3;

        scalar_t sum_v_337 = scalar_t(0);
        sum_v_337 += scalar_t(0.866025388f) * wi_81 * xj_33 * yk_2;
        sum_v_337 += scalar_t(-0.353553385f) * wi_81 * xj_34 * yk_3;
        sum_v_337 += scalar_t(-0.353553385f) * wi_81 * xj_38 * yk_1;

        scalar_t sum_v_237 = scalar_t(0);
        sum_v_237 += scalar_t(-0.597614288f) * wi_67 * xj_34 * yk_8;
        sum_v_237 += scalar_t(-0.377964467f) * wi_67 * xj_35 * yk_7;
        sum_v_237 += scalar_t(0.377964467f) * wi_67 * xj_37 * yk_5;
        sum_v_237 += scalar_t(0.597614288f) * wi_67 * xj_38 * yk_4;

        scalar_t sum_v_146 = scalar_t(0);
        sum_v_146 += scalar_t(0.487950057f) * wi_49 * xj_34 * yk_3;
        sum_v_146 += scalar_t(0.617213428f) * wi_49 * xj_35 * yk_2;
        sum_v_146 += scalar_t(-0.377964467f) * wi_49 * xj_36 * yk_1;
        sum_v_146 += scalar_t(-0.487950057f) * wi_49 * xj_38 * yk_1;

        scalar_t sum_v_147 = scalar_t(0);
        sum_v_147 += scalar_t(0.534522474f) * wi_49 * xj_35 * yk_1;
        sum_v_147 += scalar_t(0.654653668f) * wi_49 * xj_36 * yk_2;
        sum_v_147 += scalar_t(0.534522474f) * wi_49 * xj_37 * yk_3;

        scalar_t sum_v_50 = scalar_t(0);
        sum_v_50 += scalar_t(0.462910056f) * wi_24 * xj_21 * yk_9;
        sum_v_50 += scalar_t(-0.119522862f) * wi_24 * xj_21 * yk_11;
        sum_v_50 += scalar_t(0.377964467f) * wi_24 * xj_22 * yk_10;
        sum_v_50 += scalar_t(0.414039344f) * wi_24 * xj_23 * yk_13;
        sum_v_50 += scalar_t(-0.292769998f) * wi_24 * xj_24 * yk_12;
        sum_v_50 += scalar_t(0.377964467f) * wi_24 * xj_24 * yk_14;
        sum_v_50 += scalar_t(-0.119522862f) * wi_24 * xj_25 * yk_13;
        sum_v_50 += scalar_t(0.462910056f) * wi_24 * xj_25 * yk_15;

        scalar_t sum_v_81 = scalar_t(0);
        sum_v_81 += scalar_t(-0.316227764f) * wi_35 * xj_21 * yk_5;
        sum_v_81 += scalar_t(0.316227764f) * wi_35 * xj_22 * yk_4;
        sum_v_81 += scalar_t(0.547722518f) * wi_35 * xj_23 * yk_7;
        sum_v_81 += scalar_t(-0.547722518f) * wi_35 * xj_24 * yk_6;
        sum_v_81 += scalar_t(0.316227764f) * wi_35 * xj_24 * yk_8;
        sum_v_81 += scalar_t(-0.316227764f) * wi_35 * xj_25 * yk_7;

        scalar_t sum_v_10 = scalar_t(0);
        sum_v_10 += scalar_t(0.44721359f) * wi_10 * xj_21 * yk_4;
        sum_v_10 += scalar_t(0.44721359f) * wi_10 * xj_22 * yk_5;
        sum_v_10 += scalar_t(0.44721359f) * wi_10 * xj_23 * yk_6;
        sum_v_10 += scalar_t(0.44721359f) * wi_10 * xj_24 * yk_7;
        sum_v_10 += scalar_t(0.44721359f) * wi_10 * xj_25 * yk_8;

        scalar_t sum_v_82 = scalar_t(0);
        sum_v_82 += scalar_t(-0.632455528f) * wi_35 * xj_21 * yk_8;
        sum_v_82 += scalar_t(-0.316227764f) * wi_35 * xj_22 * yk_7;
        sum_v_82 += scalar_t(0.316227764f) * wi_35 * xj_24 * yk_5;
        sum_v_82 += scalar_t(0.632455528f) * wi_35 * xj_25 * yk_4;

        scalar_t sum_v_221 = scalar_t(0);
        sum_v_221 += scalar_t(0.462910056f) * wi_64 * xj_21 * yk_7;
        sum_v_221 += scalar_t(0.267261237f) * wi_64 * xj_22 * yk_6;
        sum_v_221 += scalar_t(-0.462910056f) * wi_64 * xj_22 * yk_8;
        sum_v_221 += scalar_t(0.267261237f) * wi_64 * xj_23 * yk_5;
        sum_v_221 += scalar_t(0.462910056f) * wi_64 * xj_24 * yk_4;
        sum_v_221 += scalar_t(-0.462910056f) * wi_64 * xj_25 * yk_5;

        scalar_t sum_v_220 = scalar_t(0);
        sum_v_220 += scalar_t(-0.534522474f) * wi_64 * xj_21 * yk_6;
        sum_v_220 += scalar_t(0.462910056f) * wi_64 * xj_22 * yk_7;
        sum_v_220 += scalar_t(-0.534522474f) * wi_64 * xj_23 * yk_4;
        sum_v_220 += scalar_t(0.462910056f) * wi_64 * xj_24 * yk_5;

        scalar_t sum_v_139 = scalar_t(0);
        sum_v_139 += scalar_t(-0.267261237f) * wi_47 * xj_21 * yk_12;
        sum_v_139 += scalar_t(0.327326834f) * wi_47 * xj_22 * yk_13;
        sum_v_139 += scalar_t(0.422577113f) * wi_47 * xj_22 * yk_15;
        sum_v_139 += scalar_t(-0.597614288f) * wi_47 * xj_23 * yk_10;
        sum_v_139 += scalar_t(-0.422577113f) * wi_47 * xj_24 * yk_9;
        sum_v_139 += scalar_t(0.327326834f) * wi_47 * xj_24 * yk_11;

        scalar_t sum_v_136 = scalar_t(0);
        sum_v_136 += scalar_t(-0.422577113f) * wi_47 * xj_21 * yk_9;
        sum_v_136 += scalar_t(-0.327326834f) * wi_47 * xj_21 * yk_11;
        sum_v_136 += scalar_t(0.377964467f) * wi_47 * xj_23 * yk_13;
        sum_v_136 += scalar_t(-0.534522474f) * wi_47 * xj_24 * yk_12;
        sum_v_136 += scalar_t(-0.327326834f) * wi_47 * xj_25 * yk_13;
        sum_v_136 += scalar_t(-0.422577113f) * wi_47 * xj_25 * yk_15;

        scalar_t sum_v_312 = scalar_t(0);
        sum_v_312 += scalar_t(0.44721356f) * wi_78 * xj_21 * yk_13;
        sum_v_312 += scalar_t(0.288675129f) * wi_78 * xj_21 * yk_15;
        sum_v_312 += scalar_t(0.182574183f) * wi_78 * xj_22 * yk_12;
        sum_v_312 += scalar_t(-0.353553385f) * wi_78 * xj_22 * yk_14;
        sum_v_312 += scalar_t(0.387298346f) * wi_78 * xj_23 * yk_11;
        sum_v_312 += scalar_t(0.353553385f) * wi_78 * xj_24 * yk_10;
        sum_v_312 += scalar_t(-0.288675129f) * wi_78 * xj_25 * yk_9;
        sum_v_312 += scalar_t(-0.44721356f) * wi_78 * xj_25 * yk_11;

        scalar_t sum_v_313 = scalar_t(0);
        sum_v_313 += scalar_t(-0.577350259f) * wi_78 * xj_21 * yk_10;
        sum_v_313 += scalar_t(0.182574183f) * wi_78 * xj_22 * yk_11;
        sum_v_313 += scalar_t(0.516397774f) * wi_78 * xj_23 * yk_12;
        sum_v_313 += scalar_t(0.182574183f) * wi_78 * xj_24 * yk_13;
        sum_v_313 += scalar_t(-0.577350259f) * wi_78 * xj_25 * yk_14;

        scalar_t sum_v_316 = scalar_t(0);
        sum_v_316 += scalar_t(0.288675129f) * wi_78 * xj_21 * yk_11;
        sum_v_316 += scalar_t(-0.456435442f) * wi_78 * xj_22 * yk_10;
        sum_v_316 += scalar_t(-0.645497203f) * wi_78 * xj_23 * yk_15;
        sum_v_316 += scalar_t(0.456435442f) * wi_78 * xj_24 * yk_14;
        sum_v_316 += scalar_t(-0.288675129f) * wi_78 * xj_25 * yk_13;

        scalar_t sum_v_389 = scalar_t(0);
        sum_v_389 += scalar_t(0.387298346f) * wi_89 * xj_21 * yk_5;
        sum_v_389 += scalar_t(-0.387298346f) * wi_89 * xj_22 * yk_4;
        sum_v_389 += scalar_t(0.44721356f) * wi_89 * xj_23 * yk_7;
        sum_v_389 += scalar_t(-0.44721356f) * wi_89 * xj_24 * yk_6;
        sum_v_389 += scalar_t(-0.387298346f) * wi_89 * xj_24 * yk_8;
        sum_v_389 += scalar_t(0.387298346f) * wi_89 * xj_25 * yk_7;

        scalar_t sum_v_387 = scalar_t(0);
        sum_v_387 += scalar_t(0.49999997f) * wi_89 * xj_21 * yk_5;
        sum_v_387 += scalar_t(-0.49999997f) * wi_89 * xj_22 * yk_4;
        sum_v_387 += scalar_t(0.49999997f) * wi_89 * xj_24 * yk_8;
        sum_v_387 += scalar_t(-0.49999997f) * wi_89 * xj_25 * yk_7;

        scalar_t sum_v_311 = scalar_t(0);
        sum_v_311 += scalar_t(-0.577350259f) * wi_78 * xj_21 * yk_12;
        sum_v_311 += scalar_t(0.353553385f) * wi_78 * xj_22 * yk_13;
        sum_v_311 += scalar_t(-0.456435442f) * wi_78 * xj_22 * yk_15;
        sum_v_311 += scalar_t(0.456435442f) * wi_78 * xj_24 * yk_9;
        sum_v_311 += scalar_t(0.353553385f) * wi_78 * xj_24 * yk_11;

        scalar_t sum_v_304 = scalar_t(0);
        sum_v_304 += scalar_t(0.577350259f) * wi_77 * xj_21 * yk_2;
        sum_v_304 += scalar_t(0.577350259f) * wi_77 * xj_22 * yk_3;
        sum_v_304 += scalar_t(0.577350259f) * wi_77 * xj_24 * yk_1;

        scalar_t sum_v_305 = scalar_t(0);
        sum_v_305 += scalar_t(-0.182574183f) * wi_77 * xj_21 * yk_3;
        sum_v_305 += scalar_t(0.730296731f) * wi_77 * xj_22 * yk_2;
        sum_v_305 += scalar_t(0.632455528f) * wi_77 * xj_23 * yk_1;
        sum_v_305 += scalar_t(0.182574183f) * wi_77 * xj_25 * yk_1;

        scalar_t sum_v_308 = scalar_t(0);
        sum_v_308 += scalar_t(-0.577350259f) * wi_77 * xj_22 * yk_1;
        sum_v_308 += scalar_t(0.577350259f) * wi_77 * xj_24 * yk_3;
        sum_v_308 += scalar_t(0.577350259f) * wi_77 * xj_25 * yk_2;

        scalar_t sum_v_131 = scalar_t(0);
        sum_v_131 += scalar_t(0.408248276f) * wi_46 * xj_21 * yk_1;
        sum_v_131 += scalar_t(0.707106769f) * wi_46 * xj_23 * yk_3;
        sum_v_131 += scalar_t(-0.408248276f) * wi_46 * xj_24 * yk_2;
        sum_v_131 += scalar_t(0.408248276f) * wi_46 * xj_25 * yk_3;

        scalar_t sum_v_45 = scalar_t(0);
        sum_v_45 += scalar_t(0.547722578f) * wi_23 * xj_21 * yk_3;
        sum_v_45 += scalar_t(0.547722578f) * wi_23 * xj_22 * yk_2;
        sum_v_45 += scalar_t(-0.316227794f) * wi_23 * xj_23 * yk_1;
        sum_v_45 += scalar_t(-0.547722578f) * wi_23 * xj_25 * yk_1;

        scalar_t sum_v_46 = scalar_t(0);
        sum_v_46 += scalar_t(0.547722578f) * wi_23 * xj_22 * yk_1;
        sum_v_46 += scalar_t(0.632455587f) * wi_23 * xj_23 * yk_2;
        sum_v_46 += scalar_t(0.547722578f) * wi_23 * xj_24 * yk_3;

        scalar_t sum_v_132 = scalar_t(0);
        sum_v_132 += scalar_t(-0.707106769f) * wi_46 * xj_22 * yk_3;
        sum_v_132 += scalar_t(0.707106769f) * wi_46 * xj_24 * yk_1;

        scalar_t sum_v_119 = scalar_t(0);
        sum_v_119 += scalar_t(0.597614288f) * wi_43 * xj_11 * yk_9;
        sum_v_119 += scalar_t(0.154303357f) * wi_43 * xj_11 * yk_11;
        sum_v_119 += scalar_t(0.487950057f) * wi_43 * xj_13 * yk_14;
        sum_v_119 += scalar_t(-0.154303357f) * wi_43 * xj_15 * yk_13;
        sum_v_119 += scalar_t(0.597614288f) * wi_43 * xj_15 * yk_15;

        scalar_t sum_v_117 = scalar_t(0);
        sum_v_117 += scalar_t(0.487950057f) * wi_43 * xj_11 * yk_10;
        sum_v_117 += scalar_t(0.617213428f) * wi_43 * xj_13 * yk_13;
        sum_v_117 += scalar_t(-0.377964467f) * wi_43 * xj_15 * yk_12;
        sum_v_117 += scalar_t(0.487950057f) * wi_43 * xj_15 * yk_14;

        scalar_t sum_v_115 = scalar_t(0);
        sum_v_115 += scalar_t(0.534522474f) * wi_43 * xj_11 * yk_11;
        sum_v_115 += scalar_t(0.654653668f) * wi_43 * xj_13 * yk_12;
        sum_v_115 += scalar_t(0.534522474f) * wi_43 * xj_15 * yk_13;

        scalar_t sum_v_74 = scalar_t(0);
        sum_v_74 += scalar_t(0.547722578f) * wi_32 * xj_11 * yk_4;
        sum_v_74 += scalar_t(0.547722578f) * wi_32 * xj_13 * yk_7;
        sum_v_74 += scalar_t(-0.316227794f) * wi_32 * xj_15 * yk_6;
        sum_v_74 += scalar_t(0.547722578f) * wi_32 * xj_15 * yk_8;

        scalar_t sum_v_1 = scalar_t(0);
        sum_v_1 += scalar_t(0.577350259f) * wi_1 * xj_11 * yk_1;
        sum_v_1 += scalar_t(0.577350259f) * wi_1 * xj_13 * yk_2;
        sum_v_1 += scalar_t(0.577350259f) * wi_1 * xj_15 * yk_3;

        scalar_t sum_v_37 = scalar_t(0);
        sum_v_37 += scalar_t(0.707106769f) * wi_21 * xj_13 * yk_3;
        sum_v_37 += scalar_t(-0.707106769f) * wi_21 * xj_15 * yk_2;

        scalar_t sum_v_103 = scalar_t(0);
        sum_v_103 += scalar_t(0.707106769f) * wi_41 * xj_11 * yk_2;
        sum_v_103 += scalar_t(0.707106769f) * wi_41 * xj_13 * yk_1;

        scalar_t sum_v_39 = scalar_t(0);
        sum_v_39 += scalar_t(-0.707106769f) * wi_21 * xj_11 * yk_3;
        sum_v_39 += scalar_t(0.707106769f) * wi_21 * xj_15 * yk_1;

        scalar_t sum_v_198 = scalar_t(0);
        sum_v_198 += scalar_t(-0.408248276f) * wi_60 * xj_11 * yk_4;
        sum_v_198 += scalar_t(0.408248276f) * wi_60 * xj_13 * yk_7;
        sum_v_198 += scalar_t(-0.707106769f) * wi_60 * xj_15 * yk_6;
        sum_v_198 += scalar_t(-0.408248276f) * wi_60 * xj_15 * yk_8;

        scalar_t sum_v_204 = scalar_t(0);
        sum_v_204 += scalar_t(0.408248276f) * wi_60 * xj_11 * yk_7;
        sum_v_204 += scalar_t(-0.816496551f) * wi_60 * xj_13 * yk_4;
        sum_v_204 += scalar_t(0.408248276f) * wi_60 * xj_15 * yk_5;

        scalar_t sum_v_200 = scalar_t(0);
        sum_v_200 += scalar_t(-0.707106769f) * wi_60 * xj_11 * yk_7;
        sum_v_200 += scalar_t(0.707106769f) * wi_60 * xj_15 * yk_5;

        scalar_t sum_v_293 = scalar_t(0);
        sum_v_293 += scalar_t(0.456435442f) * wi_75 * xj_11 * yk_13;
        sum_v_293 += scalar_t(-0.353553385f) * wi_75 * xj_11 * yk_15;
        sum_v_293 += scalar_t(-0.577350259f) * wi_75 * xj_13 * yk_10;
        sum_v_293 += scalar_t(0.353553385f) * wi_75 * xj_15 * yk_9;
        sum_v_293 += scalar_t(0.456435442f) * wi_75 * xj_15 * yk_11;

        scalar_t sum_v_291 = scalar_t(0);
        sum_v_291 += scalar_t(0.707106769f) * wi_75 * xj_11 * yk_12;
        sum_v_291 += scalar_t(-0.456435442f) * wi_75 * xj_11 * yk_14;
        sum_v_291 += scalar_t(-0.288675129f) * wi_75 * xj_13 * yk_11;
        sum_v_291 += scalar_t(0.456435442f) * wi_75 * xj_15 * yk_10;

        scalar_t sum_v_295 = scalar_t(0);
        sum_v_295 += scalar_t(0.353553385f) * wi_75 * xj_11 * yk_14;
        sum_v_295 += scalar_t(-0.866025388f) * wi_75 * xj_13 * yk_9;
        sum_v_295 += scalar_t(0.353553385f) * wi_75 * xj_15 * yk_10;

        scalar_t sum_v_364 = scalar_t(0);
        sum_v_364 += scalar_t(0.632455528f) * wi_86 * xj_11 * yk_6;
        sum_v_364 += scalar_t(0.182574183f) * wi_86 * xj_11 * yk_8;
        sum_v_364 += scalar_t(0.730296731f) * wi_86 * xj_13 * yk_5;
        sum_v_364 += scalar_t(-0.182574183f) * wi_86 * xj_15 * yk_4;

        scalar_t sum_v_366 = scalar_t(0);
        sum_v_366 += scalar_t(-0.44721356f) * wi_86 * xj_11 * yk_5;
        sum_v_366 += scalar_t(0.774596691f) * wi_86 * xj_13 * yk_6;
        sum_v_366 += scalar_t(-0.44721356f) * wi_86 * xj_15 * yk_7;

        scalar_t sum_v_360 = scalar_t(0);
        sum_v_360 += scalar_t(0.707106769f) * wi_86 * xj_11 * yk_8;
        sum_v_360 += scalar_t(0.707106769f) * wi_86 * xj_15 * yk_4;

        scalar_t sum_v_288 = scalar_t(0);
        sum_v_288 += scalar_t(-0.707106769f) * wi_74 * xj_10 * yk_13;
        sum_v_288 += scalar_t(0.707106769f) * wi_74 * xj_14 * yk_11;

        scalar_t sum_v_292 = scalar_t(0);
        sum_v_292 += scalar_t(0.456435442f) * wi_74 * xj_10 * yk_13;
        sum_v_292 += scalar_t(-0.353553385f) * wi_74 * xj_10 * yk_15;
        sum_v_292 += scalar_t(-0.577350259f) * wi_74 * xj_12 * yk_10;
        sum_v_292 += scalar_t(0.353553385f) * wi_74 * xj_14 * yk_9;
        sum_v_292 += scalar_t(0.456435442f) * wi_74 * xj_14 * yk_11;

        scalar_t sum_v_290 = scalar_t(0);
        sum_v_290 += scalar_t(0.707106769f) * wi_74 * xj_10 * yk_12;
        sum_v_290 += scalar_t(-0.456435442f) * wi_74 * xj_10 * yk_14;
        sum_v_290 += scalar_t(-0.288675129f) * wi_74 * xj_12 * yk_11;
        sum_v_290 += scalar_t(0.456435442f) * wi_74 * xj_14 * yk_10;

        scalar_t sum_v_282 = scalar_t(0);
        sum_v_282 += scalar_t(0.353553385f) * wi_74 * xj_10 * yk_10;
        sum_v_282 += scalar_t(0.866025388f) * wi_74 * xj_12 * yk_15;
        sum_v_282 += scalar_t(-0.353553385f) * wi_74 * xj_14 * yk_14;

        scalar_t sum_v_361 = scalar_t(0);
        sum_v_361 += scalar_t(0.577350259f) * wi_85 * xj_10 * yk_7;
        sum_v_361 += scalar_t(0.577350259f) * wi_85 * xj_12 * yk_4;
        sum_v_361 += scalar_t(0.577350259f) * wi_85 * xj_14 * yk_5;

        scalar_t sum_v_363 = scalar_t(0);
        sum_v_363 += scalar_t(0.632455528f) * wi_85 * xj_10 * yk_6;
        sum_v_363 += scalar_t(0.182574183f) * wi_85 * xj_10 * yk_8;
        sum_v_363 += scalar_t(0.730296731f) * wi_85 * xj_12 * yk_5;
        sum_v_363 += scalar_t(-0.182574183f) * wi_85 * xj_14 * yk_4;

        scalar_t sum_v_369 = scalar_t(0);
        sum_v_369 += scalar_t(-0.577350259f) * wi_85 * xj_10 * yk_5;
        sum_v_369 += scalar_t(0.577350259f) * wi_85 * xj_12 * yk_8;
        sum_v_369 += scalar_t(0.577350259f) * wi_85 * xj_14 * yk_7;

        scalar_t sum_v_359 = scalar_t(0);
        sum_v_359 += scalar_t(0.707106769f) * wi_85 * xj_10 * yk_8;
        sum_v_359 += scalar_t(0.707106769f) * wi_85 * xj_14 * yk_4;

        scalar_t sum_v_201 = scalar_t(0);
        sum_v_201 += scalar_t(0.707106769f) * wi_59 * xj_10 * yk_6;
        sum_v_201 += scalar_t(-0.408248276f) * wi_59 * xj_10 * yk_8;
        sum_v_201 += scalar_t(-0.408248276f) * wi_59 * xj_12 * yk_5;
        sum_v_201 += scalar_t(0.408248276f) * wi_59 * xj_14 * yk_4;

        scalar_t sum_v_203 = scalar_t(0);
        sum_v_203 += scalar_t(0.408248276f) * wi_59 * xj_10 * yk_7;
        sum_v_203 += scalar_t(-0.816496551f) * wi_59 * xj_12 * yk_4;
        sum_v_203 += scalar_t(0.408248276f) * wi_59 * xj_14 * yk_5;

        scalar_t sum_v_110 = scalar_t(0);
        sum_v_110 += scalar_t(-0.154303357f) * wi_42 * xj_10 * yk_13;
        sum_v_110 += scalar_t(-0.597614288f) * wi_42 * xj_10 * yk_15;
        sum_v_110 += scalar_t(0.487950057f) * wi_42 * xj_12 * yk_10;
        sum_v_110 += scalar_t(0.597614288f) * wi_42 * xj_14 * yk_9;
        sum_v_110 += scalar_t(-0.154303357f) * wi_42 * xj_14 * yk_11;

        scalar_t sum_v_112 = scalar_t(0);
        sum_v_112 += scalar_t(-0.377964467f) * wi_42 * xj_10 * yk_12;
        sum_v_112 += scalar_t(-0.487950057f) * wi_42 * xj_10 * yk_14;
        sum_v_112 += scalar_t(0.617213428f) * wi_42 * xj_12 * yk_11;
        sum_v_112 += scalar_t(0.487950057f) * wi_42 * xj_14 * yk_10;

        scalar_t sum_v_104 = scalar_t(0);
        sum_v_104 += scalar_t(-0.408248276f) * wi_40 * xj_10 * yk_1;
        sum_v_104 += scalar_t(0.816496551f) * wi_40 * xj_12 * yk_2;
        sum_v_104 += scalar_t(-0.408248276f) * wi_40 * xj_14 * yk_3;

        scalar_t sum_v_73 = scalar_t(0);
        sum_v_73 += scalar_t(0.547722578f) * wi_31 * xj_10 * yk_4;
        sum_v_73 += scalar_t(0.547722578f) * wi_31 * xj_12 * yk_7;
        sum_v_73 += scalar_t(-0.316227794f) * wi_31 * xj_14 * yk_6;
        sum_v_73 += scalar_t(0.547722578f) * wi_31 * xj_14 * yk_8;

        scalar_t sum_v_0 = scalar_t(0);
        sum_v_0 += scalar_t(0.577350259f) * wi_0 * xj_10 * yk_1;
        sum_v_0 += scalar_t(0.577350259f) * wi_0 * xj_12 * yk_2;
        sum_v_0 += scalar_t(0.577350259f) * wi_0 * xj_14 * yk_3;

        scalar_t sum_v_36 = scalar_t(0);
        sum_v_36 += scalar_t(0.707106769f) * wi_20 * xj_12 * yk_3;
        sum_v_36 += scalar_t(-0.707106769f) * wi_20 * xj_14 * yk_2;

        scalar_t sum_v_108 = scalar_t(0);
        sum_v_108 += scalar_t(-0.707106769f) * wi_40 * xj_10 * yk_1;
        sum_v_108 += scalar_t(0.707106769f) * wi_40 * xj_14 * yk_3;

        scalar_t sum_v_38 = scalar_t(0);
        sum_v_38 += scalar_t(-0.707106769f) * wi_20 * xj_10 * yk_3;
        sum_v_38 += scalar_t(0.707106769f) * wi_20 * xj_14 * yk_1;

        scalar_t sum_v_102 = scalar_t(0);
        sum_v_102 += scalar_t(0.707106769f) * wi_40 * xj_10 * yk_2;
        sum_v_102 += scalar_t(0.707106769f) * wi_40 * xj_12 * yk_1;

        scalar_t sum_v_65 = scalar_t(0);
        sum_v_65 += wi_29 * xj_12 * yk_0;

        scalar_t sum_v_63 = scalar_t(0);
        sum_v_63 += wi_29 * xj_10 * yk_0;

        scalar_t sum_v_67 = scalar_t(0);
        sum_v_67 += wi_29 * xj_14 * yk_0;

        scalar_t sum_v_31 = scalar_t(0);
        sum_v_31 += scalar_t(-0.316227794f) * wi_19 * xj_5 * yk_6;
        sum_v_31 += scalar_t(-0.547722578f) * wi_19 * xj_5 * yk_8;
        sum_v_31 += scalar_t(0.547722578f) * wi_19 * xj_7 * yk_5;
        sum_v_31 += scalar_t(0.547722578f) * wi_19 * xj_9 * yk_4;

        scalar_t sum_v_33 = scalar_t(0);
        sum_v_33 += scalar_t(0.547722578f) * wi_19 * xj_5 * yk_5;
        sum_v_33 += scalar_t(0.632455587f) * wi_19 * xj_7 * yk_6;
        sum_v_33 += scalar_t(0.547722578f) * wi_19 * xj_9 * yk_7;

        scalar_t sum_v_93 = scalar_t(0);
        sum_v_93 += scalar_t(-0.408248276f) * wi_39 * xj_5 * yk_4;
        sum_v_93 += scalar_t(0.408248276f) * wi_39 * xj_7 * yk_7;
        sum_v_93 += scalar_t(-0.707106769f) * wi_39 * xj_9 * yk_6;
        sum_v_93 += scalar_t(-0.408248276f) * wi_39 * xj_9 * yk_8;

        scalar_t sum_v_99 = scalar_t(0);
        sum_v_99 += scalar_t(0.408248276f) * wi_39 * xj_5 * yk_7;
        sum_v_99 += scalar_t(-0.816496551f) * wi_39 * xj_7 * yk_4;
        sum_v_99 += scalar_t(0.408248276f) * wi_39 * xj_9 * yk_5;

        scalar_t sum_v_95 = scalar_t(0);
        sum_v_95 += scalar_t(-0.707106769f) * wi_39 * xj_5 * yk_7;
        sum_v_95 += scalar_t(0.707106769f) * wi_39 * xj_9 * yk_5;

        scalar_t sum_v_58 = scalar_t(0);
        sum_v_58 += scalar_t(0.707106769f) * wi_28 * xj_7 * yk_3;
        sum_v_58 += scalar_t(-0.707106769f) * wi_28 * xj_9 * yk_2;

        scalar_t sum_v_194 = scalar_t(0);
        sum_v_194 += scalar_t(0.597614288f) * wi_58 * xj_5 * yk_9;
        sum_v_194 += scalar_t(0.154303357f) * wi_58 * xj_5 * yk_11;
        sum_v_194 += scalar_t(0.487950057f) * wi_58 * xj_7 * yk_14;
        sum_v_194 += scalar_t(-0.154303357f) * wi_58 * xj_9 * yk_13;
        sum_v_194 += scalar_t(0.597614288f) * wi_58 * xj_9 * yk_15;

        scalar_t sum_v_192 = scalar_t(0);
        sum_v_192 += scalar_t(0.487950057f) * wi_58 * xj_5 * yk_10;
        sum_v_192 += scalar_t(0.617213428f) * wi_58 * xj_7 * yk_13;
        sum_v_192 += scalar_t(-0.377964467f) * wi_58 * xj_9 * yk_12;
        sum_v_192 += scalar_t(0.487950057f) * wi_58 * xj_9 * yk_14;

        scalar_t sum_v_190 = scalar_t(0);
        sum_v_190 += scalar_t(0.534522474f) * wi_58 * xj_5 * yk_11;
        sum_v_190 += scalar_t(0.654653668f) * wi_58 * xj_7 * yk_12;
        sum_v_190 += scalar_t(0.534522474f) * wi_58 * xj_9 * yk_13;

        scalar_t sum_v_273 = scalar_t(0);
        sum_v_273 += scalar_t(0.632455528f) * wi_73 * xj_5 * yk_6;
        sum_v_273 += scalar_t(0.182574183f) * wi_73 * xj_5 * yk_8;
        sum_v_273 += scalar_t(0.730296731f) * wi_73 * xj_7 * yk_5;
        sum_v_273 += scalar_t(-0.182574183f) * wi_73 * xj_9 * yk_4;

        scalar_t sum_v_275 = scalar_t(0);
        sum_v_275 += scalar_t(-0.44721356f) * wi_73 * xj_5 * yk_5;
        sum_v_275 += scalar_t(0.774596691f) * wi_73 * xj_7 * yk_6;
        sum_v_275 += scalar_t(-0.44721356f) * wi_73 * xj_9 * yk_7;

        scalar_t sum_v_269 = scalar_t(0);
        sum_v_269 += scalar_t(0.707106769f) * wi_73 * xj_5 * yk_8;
        sum_v_269 += scalar_t(0.707106769f) * wi_73 * xj_9 * yk_4;

        scalar_t sum_v_346 = scalar_t(0);
        sum_v_346 += scalar_t(0.353553385f) * wi_84 * xj_5 * yk_10;
        sum_v_346 += scalar_t(0.866025388f) * wi_84 * xj_7 * yk_15;
        sum_v_346 += scalar_t(-0.353553385f) * wi_84 * xj_9 * yk_14;

        scalar_t sum_v_356 = scalar_t(0);
        sum_v_356 += scalar_t(0.456435442f) * wi_84 * xj_5 * yk_13;
        sum_v_356 += scalar_t(-0.353553385f) * wi_84 * xj_5 * yk_15;
        sum_v_356 += scalar_t(-0.577350259f) * wi_84 * xj_7 * yk_10;
        sum_v_356 += scalar_t(0.353553385f) * wi_84 * xj_9 * yk_9;
        sum_v_356 += scalar_t(0.456435442f) * wi_84 * xj_9 * yk_11;

        scalar_t sum_v_354 = scalar_t(0);
        sum_v_354 += scalar_t(0.707106769f) * wi_84 * xj_5 * yk_12;
        sum_v_354 += scalar_t(-0.456435442f) * wi_84 * xj_5 * yk_14;
        sum_v_354 += scalar_t(-0.288675129f) * wi_84 * xj_7 * yk_11;
        sum_v_354 += scalar_t(0.456435442f) * wi_84 * xj_9 * yk_10;

        scalar_t sum_v_351 = scalar_t(0);
        sum_v_351 += scalar_t(-0.707106769f) * wi_83 * xj_4 * yk_13;
        sum_v_351 += scalar_t(0.707106769f) * wi_83 * xj_8 * yk_11;

        scalar_t sum_v_355 = scalar_t(0);
        sum_v_355 += scalar_t(0.456435442f) * wi_83 * xj_4 * yk_13;
        sum_v_355 += scalar_t(-0.353553385f) * wi_83 * xj_4 * yk_15;
        sum_v_355 += scalar_t(-0.577350259f) * wi_83 * xj_6 * yk_10;
        sum_v_355 += scalar_t(0.353553385f) * wi_83 * xj_8 * yk_9;
        sum_v_355 += scalar_t(0.456435442f) * wi_83 * xj_8 * yk_11;

        scalar_t sum_v_353 = scalar_t(0);
        sum_v_353 += scalar_t(0.707106769f) * wi_83 * xj_4 * yk_12;
        sum_v_353 += scalar_t(-0.456435442f) * wi_83 * xj_4 * yk_14;
        sum_v_353 += scalar_t(-0.288675129f) * wi_83 * xj_6 * yk_11;
        sum_v_353 += scalar_t(0.456435442f) * wi_83 * xj_8 * yk_10;

        scalar_t sum_v_345 = scalar_t(0);
        sum_v_345 += scalar_t(0.353553385f) * wi_83 * xj_4 * yk_10;
        sum_v_345 += scalar_t(0.866025388f) * wi_83 * xj_6 * yk_15;
        sum_v_345 += scalar_t(-0.353553385f) * wi_83 * xj_8 * yk_14;

        scalar_t sum_v_278 = scalar_t(0);
        sum_v_278 += scalar_t(-0.577350259f) * wi_72 * xj_4 * yk_5;
        sum_v_278 += scalar_t(0.577350259f) * wi_72 * xj_6 * yk_8;
        sum_v_278 += scalar_t(0.577350259f) * wi_72 * xj_8 * yk_7;

        scalar_t sum_v_276 = scalar_t(0);
        sum_v_276 += scalar_t(-0.182574183f) * wi_72 * xj_4 * yk_4;
        sum_v_276 += scalar_t(0.730296731f) * wi_72 * xj_6 * yk_7;
        sum_v_276 += scalar_t(0.632455528f) * wi_72 * xj_8 * yk_6;
        sum_v_276 += scalar_t(-0.182574183f) * wi_72 * xj_8 * yk_8;

        scalar_t sum_v_274 = scalar_t(0);
        sum_v_274 += scalar_t(-0.44721356f) * wi_72 * xj_4 * yk_5;
        sum_v_274 += scalar_t(0.774596691f) * wi_72 * xj_6 * yk_6;
        sum_v_274 += scalar_t(-0.44721356f) * wi_72 * xj_8 * yk_7;

        scalar_t sum_v_185 = scalar_t(0);
        sum_v_185 += scalar_t(-0.154303357f) * wi_57 * xj_4 * yk_13;
        sum_v_185 += scalar_t(-0.597614288f) * wi_57 * xj_4 * yk_15;
        sum_v_185 += scalar_t(0.487950057f) * wi_57 * xj_6 * yk_10;
        sum_v_185 += scalar_t(0.597614288f) * wi_57 * xj_8 * yk_9;
        sum_v_185 += scalar_t(-0.154303357f) * wi_57 * xj_8 * yk_11;

        scalar_t sum_v_187 = scalar_t(0);
        sum_v_187 += scalar_t(-0.377964467f) * wi_57 * xj_4 * yk_12;
        sum_v_187 += scalar_t(-0.487950057f) * wi_57 * xj_4 * yk_14;
        sum_v_187 += scalar_t(0.617213428f) * wi_57 * xj_6 * yk_11;
        sum_v_187 += scalar_t(0.487950057f) * wi_57 * xj_8 * yk_10;

        scalar_t sum_v_179 = scalar_t(0);
        sum_v_179 += scalar_t(-0.408248276f) * wi_55 * xj_4 * yk_1;
        sum_v_179 += scalar_t(0.816496551f) * wi_55 * xj_6 * yk_2;
        sum_v_179 += scalar_t(-0.408248276f) * wi_55 * xj_8 * yk_3;

        scalar_t sum_v_175 = scalar_t(0);
        sum_v_175 += scalar_t(0.707106769f) * wi_55 * xj_4 * yk_3;
        sum_v_175 += scalar_t(0.707106769f) * wi_55 * xj_8 * yk_1;

        scalar_t sum_v_181 = scalar_t(0);
        sum_v_181 += scalar_t(0.707106769f) * wi_55 * xj_6 * yk_3;
        sum_v_181 += scalar_t(0.707106769f) * wi_55 * xj_8 * yk_2;

        scalar_t sum_v_96 = scalar_t(0);
        sum_v_96 += scalar_t(0.707106769f) * wi_38 * xj_4 * yk_6;
        sum_v_96 += scalar_t(-0.408248276f) * wi_38 * xj_4 * yk_8;
        sum_v_96 += scalar_t(-0.408248276f) * wi_38 * xj_6 * yk_5;
        sum_v_96 += scalar_t(0.408248276f) * wi_38 * xj_8 * yk_4;

        scalar_t sum_v_90 = scalar_t(0);
        sum_v_90 += scalar_t(0.408248276f) * wi_38 * xj_4 * yk_5;
        sum_v_90 += scalar_t(0.816496551f) * wi_38 * xj_6 * yk_8;
        sum_v_90 += scalar_t(-0.408248276f) * wi_38 * xj_8 * yk_7;

        scalar_t sum_v_34 = scalar_t(0);
        sum_v_34 += scalar_t(0.547722578f) * wi_18 * xj_4 * yk_4;
        sum_v_34 += scalar_t(0.547722578f) * wi_18 * xj_6 * yk_7;
        sum_v_34 += scalar_t(-0.316227794f) * wi_18 * xj_8 * yk_6;
        sum_v_34 += scalar_t(0.547722578f) * wi_18 * xj_8 * yk_8;

        scalar_t sum_v_8 = scalar_t(0);
        sum_v_8 += scalar_t(0.577350259f) * wi_8 * xj_4 * yk_1;
        sum_v_8 += scalar_t(0.577350259f) * wi_8 * xj_6 * yk_2;
        sum_v_8 += scalar_t(0.577350259f) * wi_8 * xj_8 * yk_3;

        scalar_t sum_v_59 = scalar_t(0);
        sum_v_59 += scalar_t(-0.707106769f) * wi_27 * xj_4 * yk_3;
        sum_v_59 += scalar_t(0.707106769f) * wi_27 * xj_8 * yk_1;

        scalar_t sum_v_61 = scalar_t(0);
        sum_v_61 += scalar_t(0.707106769f) * wi_27 * xj_4 * yk_2;
        sum_v_61 += scalar_t(-0.707106769f) * wi_27 * xj_6 * yk_1;

        scalar_t sum_v_26 = scalar_t(0);
        sum_v_26 += wi_16 * xj_6 * yk_0;

        scalar_t sum_v_24 = scalar_t(0);
        sum_v_24 += wi_16 * xj_4 * yk_0;

        scalar_t sum_v_28 = scalar_t(0);
        sum_v_28 += wi_16 * xj_8 * yk_0;

        scalar_t sum_v_7 = scalar_t(0);
        sum_v_7 += wi_7 * xj_3 * yk_0;

        scalar_t sum_v_19 = scalar_t(0);
        sum_v_19 += wi_15 * xj_3 * yk_2;

        scalar_t sum_v_22 = scalar_t(0);
        sum_v_22 += wi_14 * xj_2 * yk_3;

        scalar_t sum_v_14 = scalar_t(0);
        sum_v_14 += wi_14 * xj_2 * yk_1;

        scalar_t sum_v_5 = scalar_t(0);
        sum_v_5 += wi_5 * xj_1 * yk_0;

        scalar_t sum_v_17 = scalar_t(0);
        sum_v_17 += wi_13 * xj_1 * yk_2;

        scalar_t sum_v_20 = scalar_t(0);
        sum_v_20 += wi_12 * xj_0 * yk_3;

        scalar_t sum_v_12 = scalar_t(0);
        sum_v_12 += wi_12 * xj_0 * yk_1;

        scalar_t sum_v_42 = scalar_t(0);
        sum_v_42 += scalar_t(-0.316227764f) * wi_22 * xj_16 * yk_5;
        sum_v_42 += scalar_t(0.316227764f) * wi_22 * xj_17 * yk_4;
        sum_v_42 += scalar_t(0.547722518f) * wi_22 * xj_18 * yk_7;
        sum_v_42 += scalar_t(-0.547722518f) * wi_22 * xj_19 * yk_6;
        sum_v_42 += scalar_t(0.316227764f) * wi_22 * xj_19 * yk_8;
        sum_v_42 += scalar_t(-0.316227764f) * wi_22 * xj_20 * yk_7;

        scalar_t sum_v_126 = scalar_t(0);
        sum_v_126 += scalar_t(0.462910056f) * wi_45 * xj_16 * yk_7;
        sum_v_126 += scalar_t(0.267261237f) * wi_45 * xj_17 * yk_6;
        sum_v_126 += scalar_t(-0.462910056f) * wi_45 * xj_17 * yk_8;
        sum_v_126 += scalar_t(0.267261237f) * wi_45 * xj_18 * yk_5;
        sum_v_126 += scalar_t(0.462910056f) * wi_45 * xj_19 * yk_4;
        sum_v_126 += scalar_t(-0.462910056f) * wi_45 * xj_20 * yk_5;

        scalar_t sum_v_127 = scalar_t(0);
        sum_v_127 += scalar_t(-0.534522474f) * wi_45 * xj_16 * yk_4;
        sum_v_127 += scalar_t(0.267261237f) * wi_45 * xj_17 * yk_5;
        sum_v_127 += scalar_t(0.534522474f) * wi_45 * xj_18 * yk_6;
        sum_v_127 += scalar_t(0.267261237f) * wi_45 * xj_19 * yk_7;
        sum_v_127 += scalar_t(-0.534522474f) * wi_45 * xj_20 * yk_8;

        scalar_t sum_v_129 = scalar_t(0);
        sum_v_129 += scalar_t(-0.462910056f) * wi_45 * xj_17 * yk_5;
        sum_v_129 += scalar_t(-0.534522474f) * wi_45 * xj_18 * yk_8;
        sum_v_129 += scalar_t(0.462910056f) * wi_45 * xj_19 * yk_7;
        sum_v_129 += scalar_t(-0.534522474f) * wi_45 * xj_20 * yk_6;

        scalar_t sum_v_80 = scalar_t(0);
        sum_v_80 += scalar_t(0.462910056f) * wi_34 * xj_16 * yk_9;
        sum_v_80 += scalar_t(-0.119522862f) * wi_34 * xj_16 * yk_11;
        sum_v_80 += scalar_t(0.377964467f) * wi_34 * xj_17 * yk_10;
        sum_v_80 += scalar_t(0.414039344f) * wi_34 * xj_18 * yk_13;
        sum_v_80 += scalar_t(-0.292769998f) * wi_34 * xj_19 * yk_12;
        sum_v_80 += scalar_t(0.377964467f) * wi_34 * xj_19 * yk_14;
        sum_v_80 += scalar_t(-0.119522862f) * wi_34 * xj_20 * yk_13;
        sum_v_80 += scalar_t(0.462910056f) * wi_34 * xj_20 * yk_15;

        scalar_t sum_v_77 = scalar_t(0);
        sum_v_77 += scalar_t(0.547722578f) * wi_33 * xj_16 * yk_1;
        sum_v_77 += scalar_t(-0.316227794f) * wi_33 * xj_18 * yk_3;
        sum_v_77 += scalar_t(0.547722578f) * wi_33 * xj_19 * yk_2;
        sum_v_77 += scalar_t(0.547722578f) * wi_33 * xj_20 * yk_3;

        scalar_t sum_v_208 = scalar_t(0);
        sum_v_208 += scalar_t(-0.408248276f) * wi_61 * xj_16 * yk_3;
        sum_v_208 += scalar_t(0.408248276f) * wi_61 * xj_17 * yk_2;
        sum_v_208 += scalar_t(-0.707106769f) * wi_61 * xj_18 * yk_1;
        sum_v_208 += scalar_t(0.408248276f) * wi_61 * xj_20 * yk_1;

        scalar_t sum_v_213 = scalar_t(0);
        sum_v_213 += scalar_t(0.327326834f) * wi_62 * xj_16 * yk_13;
        sum_v_213 += scalar_t(-0.422577113f) * wi_62 * xj_16 * yk_15;
        sum_v_213 += scalar_t(0.534522474f) * wi_62 * xj_17 * yk_12;
        sum_v_213 += scalar_t(-0.377964467f) * wi_62 * xj_18 * yk_11;
        sum_v_213 += scalar_t(0.422577113f) * wi_62 * xj_20 * yk_9;
        sum_v_213 += scalar_t(-0.327326834f) * wi_62 * xj_20 * yk_11;

        scalar_t sum_v_211 = scalar_t(0);
        sum_v_211 += scalar_t(-0.422577113f) * wi_62 * xj_16 * yk_9;
        sum_v_211 += scalar_t(-0.327326834f) * wi_62 * xj_16 * yk_11;
        sum_v_211 += scalar_t(0.377964467f) * wi_62 * xj_18 * yk_13;
        sum_v_211 += scalar_t(-0.534522474f) * wi_62 * xj_19 * yk_12;
        sum_v_211 += scalar_t(-0.327326834f) * wi_62 * xj_20 * yk_13;
        sum_v_211 += scalar_t(-0.422577113f) * wi_62 * xj_20 * yk_15;

        scalar_t sum_v_206 = scalar_t(0);
        sum_v_206 += scalar_t(0.408248276f) * wi_61 * xj_16 * yk_1;
        sum_v_206 += scalar_t(0.707106769f) * wi_61 * xj_18 * yk_3;
        sum_v_206 += scalar_t(-0.408248276f) * wi_61 * xj_19 * yk_2;
        sum_v_206 += scalar_t(0.408248276f) * wi_61 * xj_20 * yk_3;

        scalar_t sum_v_298 = scalar_t(0);
        sum_v_298 += scalar_t(0.387298346f) * wi_76 * xj_16 * yk_5;
        sum_v_298 += scalar_t(-0.387298346f) * wi_76 * xj_17 * yk_4;
        sum_v_298 += scalar_t(0.44721356f) * wi_76 * xj_18 * yk_7;
        sum_v_298 += scalar_t(-0.44721356f) * wi_76 * xj_19 * yk_6;
        sum_v_298 += scalar_t(-0.387298346f) * wi_76 * xj_19 * yk_8;
        sum_v_298 += scalar_t(0.387298346f) * wi_76 * xj_20 * yk_7;

        scalar_t sum_v_296 = scalar_t(0);
        sum_v_296 += scalar_t(0.49999997f) * wi_76 * xj_16 * yk_5;
        sum_v_296 += scalar_t(-0.49999997f) * wi_76 * xj_17 * yk_4;
        sum_v_296 += scalar_t(0.49999997f) * wi_76 * xj_19 * yk_8;
        sum_v_296 += scalar_t(-0.49999997f) * wi_76 * xj_20 * yk_7;

        scalar_t sum_v_380 = scalar_t(0);
        sum_v_380 += scalar_t(-0.288675129f) * wi_88 * xj_16 * yk_13;
        sum_v_380 += scalar_t(0.456435442f) * wi_88 * xj_17 * yk_14;
        sum_v_380 += scalar_t(-0.645497203f) * wi_88 * xj_18 * yk_9;
        sum_v_380 += scalar_t(0.456435442f) * wi_88 * xj_19 * yk_10;
        sum_v_380 += scalar_t(-0.288675129f) * wi_88 * xj_20 * yk_11;

        scalar_t sum_v_382 = scalar_t(0);
        sum_v_382 += scalar_t(0.44721356f) * wi_88 * xj_16 * yk_13;
        sum_v_382 += scalar_t(0.288675129f) * wi_88 * xj_16 * yk_15;
        sum_v_382 += scalar_t(0.182574183f) * wi_88 * xj_17 * yk_12;
        sum_v_382 += scalar_t(-0.353553385f) * wi_88 * xj_17 * yk_14;
        sum_v_382 += scalar_t(0.387298346f) * wi_88 * xj_18 * yk_11;
        sum_v_382 += scalar_t(0.353553385f) * wi_88 * xj_19 * yk_10;
        sum_v_382 += scalar_t(-0.288675129f) * wi_88 * xj_20 * yk_9;
        sum_v_382 += scalar_t(-0.44721356f) * wi_88 * xj_20 * yk_11;

        scalar_t sum_v_386 = scalar_t(0);
        sum_v_386 += scalar_t(0.288675129f) * wi_88 * xj_16 * yk_11;
        sum_v_386 += scalar_t(-0.456435442f) * wi_88 * xj_17 * yk_10;
        sum_v_386 += scalar_t(-0.645497203f) * wi_88 * xj_18 * yk_15;
        sum_v_386 += scalar_t(0.456435442f) * wi_88 * xj_19 * yk_14;
        sum_v_386 += scalar_t(-0.288675129f) * wi_88 * xj_20 * yk_13;

        scalar_t sum_v_377 = scalar_t(0);
        sum_v_377 += scalar_t(-0.182574183f) * wi_87 * xj_16 * yk_1;
        sum_v_377 += scalar_t(0.632455528f) * wi_87 * xj_18 * yk_3;
        sum_v_377 += scalar_t(0.730296731f) * wi_87 * xj_19 * yk_2;
        sum_v_377 += scalar_t(-0.182574183f) * wi_87 * xj_20 * yk_3;

        scalar_t sum_v_374 = scalar_t(0);
        sum_v_374 += scalar_t(0.577350259f) * wi_87 * xj_16 * yk_2;
        sum_v_374 += scalar_t(0.577350259f) * wi_87 * xj_17 * yk_3;
        sum_v_374 += scalar_t(0.577350259f) * wi_87 * xj_19 * yk_1;

        scalar_t sum_v_385 = scalar_t(0);
        sum_v_385 += scalar_t(0.456435442f) * wi_88 * xj_17 * yk_9;
        sum_v_385 += scalar_t(-0.353553385f) * wi_88 * xj_17 * yk_11;
        sum_v_385 += scalar_t(0.353553385f) * wi_88 * xj_19 * yk_13;
        sum_v_385 += scalar_t(0.456435442f) * wi_88 * xj_19 * yk_15;
        sum_v_385 += scalar_t(-0.577350259f) * wi_88 * xj_20 * yk_12;

        scalar_t sum_v_376 = scalar_t(0);
        sum_v_376 += scalar_t(-0.44721356f) * wi_87 * xj_17 * yk_1;
        sum_v_376 += scalar_t(0.774596691f) * wi_87 * xj_18 * yk_2;
        sum_v_376 += scalar_t(-0.44721356f) * wi_87 * xj_19 * yk_3;

        scalar_t sum_v_76 = scalar_t(0);
        sum_v_76 += scalar_t(0.547722578f) * wi_33 * xj_17 * yk_1;
        sum_v_76 += scalar_t(0.632455587f) * wi_33 * xj_18 * yk_2;
        sum_v_76 += scalar_t(0.547722578f) * wi_33 * xj_19 * yk_3;

        scalar_t sum_v_43 = scalar_t(0);
        sum_v_43 += scalar_t(-0.632455528f) * wi_22 * xj_16 * yk_8;
        sum_v_43 += scalar_t(-0.316227764f) * wi_22 * xj_17 * yk_7;
        sum_v_43 += scalar_t(0.316227764f) * wi_22 * xj_19 * yk_5;
        sum_v_43 += scalar_t(0.632455528f) * wi_22 * xj_20 * yk_4;

        scalar_t sum_v_51 = scalar_t(0);
        sum_v_51 += scalar_t(0.462910056f) * wi_25 * xj_26 * yk_8;
        sum_v_51 += scalar_t(0.377964467f) * wi_25 * xj_27 * yk_7;
        sum_v_51 += scalar_t(0.414039344f) * wi_25 * xj_28 * yk_6;
        sum_v_51 += scalar_t(0.119522862f) * wi_25 * xj_28 * yk_8;
        sum_v_51 += scalar_t(-0.292769998f) * wi_25 * xj_29 * yk_5;
        sum_v_51 += scalar_t(-0.119522862f) * wi_25 * xj_30 * yk_4;
        sum_v_51 += scalar_t(-0.377964467f) * wi_25 * xj_31 * yk_5;
        sum_v_51 += scalar_t(-0.462910056f) * wi_25 * xj_32 * yk_4;

        scalar_t sum_v_84 = scalar_t(0);
        sum_v_84 += scalar_t(-0.231455028f) * wi_36 * xj_26 * yk_10;
        sum_v_84 += scalar_t(0.231455028f) * wi_36 * xj_27 * yk_9;
        sum_v_84 += scalar_t(-0.298807144f) * wi_36 * xj_27 * yk_11;
        sum_v_84 += scalar_t(0.298807144f) * wi_36 * xj_28 * yk_10;
        sum_v_84 += scalar_t(0.462910056f) * wi_36 * xj_29 * yk_13;
        sum_v_84 += scalar_t(-0.462910056f) * wi_36 * xj_30 * yk_12;
        sum_v_84 += scalar_t(0.298807144f) * wi_36 * xj_30 * yk_14;
        sum_v_84 += scalar_t(-0.298807144f) * wi_36 * xj_31 * yk_13;
        sum_v_84 += scalar_t(0.231455028f) * wi_36 * xj_31 * yk_15;
        sum_v_84 += scalar_t(-0.231455028f) * wi_36 * xj_32 * yk_14;

        scalar_t sum_v_85 = scalar_t(0);
        sum_v_85 += scalar_t(-0.566946685f) * wi_36 * xj_26 * yk_15;
        sum_v_85 += scalar_t(-0.377964497f) * wi_36 * xj_27 * yk_14;
        sum_v_85 += scalar_t(-0.188982248f) * wi_36 * xj_28 * yk_13;
        sum_v_85 += scalar_t(0.188982248f) * wi_36 * xj_30 * yk_11;
        sum_v_85 += scalar_t(0.377964497f) * wi_36 * xj_31 * yk_10;
        sum_v_85 += scalar_t(0.566946685f) * wi_36 * xj_32 * yk_9;

        scalar_t sum_v_231 = scalar_t(0);
        sum_v_231 += scalar_t(0.38575837f) * wi_66 * xj_26 * yk_14;
        sum_v_231 += scalar_t(0.298807144f) * wi_66 * xj_27 * yk_13;
        sum_v_231 += scalar_t(-0.38575837f) * wi_66 * xj_27 * yk_15;
        sum_v_231 += scalar_t(0.154303357f) * wi_66 * xj_28 * yk_12;
        sum_v_231 += scalar_t(-0.298807144f) * wi_66 * xj_28 * yk_14;
        sum_v_231 += scalar_t(0.154303357f) * wi_66 * xj_29 * yk_11;
        sum_v_231 += scalar_t(0.298807144f) * wi_66 * xj_30 * yk_10;
        sum_v_231 += scalar_t(0.38575837f) * wi_66 * xj_31 * yk_9;
        sum_v_231 += scalar_t(-0.298807144f) * wi_66 * xj_31 * yk_11;
        sum_v_231 += scalar_t(-0.38575837f) * wi_66 * xj_32 * yk_10;

        scalar_t sum_v_234 = scalar_t(0);
        sum_v_234 += scalar_t(-0.243975028f) * wi_66 * xj_26 * yk_11;
        sum_v_234 += scalar_t(-0.243975028f) * wi_66 * xj_28 * yk_9;
        sum_v_234 += scalar_t(-0.377964467f) * wi_66 * xj_28 * yk_11;
        sum_v_234 += scalar_t(-0.487950057f) * wi_66 * xj_29 * yk_14;
        sum_v_234 += scalar_t(0.377964467f) * wi_66 * xj_30 * yk_13;
        sum_v_234 += scalar_t(-0.243975028f) * wi_66 * xj_30 * yk_15;
        sum_v_234 += scalar_t(-0.487950057f) * wi_66 * xj_31 * yk_12;
        sum_v_234 += scalar_t(-0.243975028f) * wi_66 * xj_32 * yk_13;

        scalar_t sum_v_405 = scalar_t(0);
        sum_v_405 += scalar_t(-0.408248305f) * wi_91 * xj_26 * yk_14;
        sum_v_405 += scalar_t(-0.408248305f) * wi_91 * xj_27 * yk_15;
        sum_v_405 += scalar_t(0.408248305f) * wi_91 * xj_28 * yk_12;
        sum_v_405 += scalar_t(-0.408248305f) * wi_91 * xj_29 * yk_11;
        sum_v_405 += scalar_t(0.408248305f) * wi_91 * xj_31 * yk_9;
        sum_v_405 += scalar_t(0.408248305f) * wi_91 * xj_32 * yk_10;

        scalar_t sum_v_403 = scalar_t(0);
        sum_v_403 += scalar_t(0.408248305f) * wi_91 * xj_26 * yk_10;
        sum_v_403 += scalar_t(-0.408248305f) * wi_91 * xj_27 * yk_9;
        sum_v_403 += scalar_t(0.408248305f) * wi_91 * xj_29 * yk_13;
        sum_v_403 += scalar_t(-0.408248305f) * wi_91 * xj_30 * yk_12;
        sum_v_403 += scalar_t(-0.408248305f) * wi_91 * xj_31 * yk_15;
        sum_v_403 += scalar_t(0.408248305f) * wi_91 * xj_32 * yk_14;

        scalar_t sum_v_401 = scalar_t(0);
        sum_v_401 += scalar_t(0.408248305f) * wi_91 * xj_27 * yk_11;
        sum_v_401 += scalar_t(-0.408248305f) * wi_91 * xj_28 * yk_10;
        sum_v_401 += scalar_t(-0.408248305f) * wi_91 * xj_29 * yk_15;
        sum_v_401 += scalar_t(0.408248305f) * wi_91 * xj_30 * yk_14;
        sum_v_401 += scalar_t(-0.408248305f) * wi_91 * xj_31 * yk_13;
        sum_v_401 += scalar_t(0.408248305f) * wi_91 * xj_32 * yk_12;

        scalar_t sum_v_328 = scalar_t(0);
        sum_v_328 += scalar_t(-0.288675129f) * wi_80 * xj_26 * yk_4;
        sum_v_328 += scalar_t(0.353553385f) * wi_80 * xj_27 * yk_5;
        sum_v_328 += scalar_t(0.44721356f) * wi_80 * xj_28 * yk_4;
        sum_v_328 += scalar_t(0.182574183f) * wi_80 * xj_29 * yk_7;
        sum_v_328 += scalar_t(0.387298346f) * wi_80 * xj_30 * yk_6;
        sum_v_328 += scalar_t(0.44721356f) * wi_80 * xj_30 * yk_8;
        sum_v_328 += scalar_t(0.353553385f) * wi_80 * xj_31 * yk_7;
        sum_v_328 += scalar_t(-0.288675129f) * wi_80 * xj_32 * yk_8;

        scalar_t sum_v_327 = scalar_t(0);
        sum_v_327 += scalar_t(-0.577350259f) * wi_80 * xj_27 * yk_4;
        sum_v_327 += scalar_t(0.182574183f) * wi_80 * xj_28 * yk_5;
        sum_v_327 += scalar_t(0.516397774f) * wi_80 * xj_29 * yk_6;
        sum_v_327 += scalar_t(0.182574183f) * wi_80 * xj_30 * yk_7;
        sum_v_327 += scalar_t(-0.577350259f) * wi_80 * xj_31 * yk_8;

        scalar_t sum_v_330 = scalar_t(0);
        sum_v_330 += scalar_t(-0.456435442f) * wi_80 * xj_27 * yk_5;
        sum_v_330 += scalar_t(0.288675129f) * wi_80 * xj_28 * yk_4;
        sum_v_330 += scalar_t(-0.288675129f) * wi_80 * xj_30 * yk_8;
        sum_v_330 += scalar_t(0.456435442f) * wi_80 * xj_31 * yk_7;
        sum_v_330 += scalar_t(-0.645497203f) * wi_80 * xj_32 * yk_6;

        scalar_t sum_v_325 = scalar_t(0);
        sum_v_325 += scalar_t(0.456435442f) * wi_80 * xj_26 * yk_7;
        sum_v_325 += scalar_t(0.353553385f) * wi_80 * xj_28 * yk_7;
        sum_v_325 += scalar_t(-0.577350259f) * wi_80 * xj_29 * yk_4;
        sum_v_325 += scalar_t(0.353553385f) * wi_80 * xj_30 * yk_5;
        sum_v_325 += scalar_t(-0.456435442f) * wi_80 * xj_32 * yk_5;

        scalar_t sum_v_144 = scalar_t(0);
        sum_v_144 += scalar_t(0.422577113f) * wi_48 * xj_26 * yk_7;
        sum_v_144 += scalar_t(0.597614288f) * wi_48 * xj_27 * yk_6;
        sum_v_144 += scalar_t(-0.327326834f) * wi_48 * xj_28 * yk_7;
        sum_v_144 += scalar_t(0.267261237f) * wi_48 * xj_29 * yk_4;
        sum_v_144 += scalar_t(-0.327326834f) * wi_48 * xj_30 * yk_5;
        sum_v_144 += scalar_t(-0.422577113f) * wi_48 * xj_32 * yk_5;

        scalar_t sum_v_141 = scalar_t(0);
        sum_v_141 += scalar_t(0.422577113f) * wi_48 * xj_26 * yk_4;
        sum_v_141 += scalar_t(0.327326834f) * wi_48 * xj_28 * yk_4;
        sum_v_141 += scalar_t(0.534522474f) * wi_48 * xj_29 * yk_7;
        sum_v_141 += scalar_t(-0.377964467f) * wi_48 * xj_30 * yk_6;
        sum_v_141 += scalar_t(0.327326834f) * wi_48 * xj_30 * yk_8;
        sum_v_141 += scalar_t(0.422577113f) * wi_48 * xj_32 * yk_8;

        scalar_t sum_v_225 = scalar_t(0);
        sum_v_225 += scalar_t(0.597614288f) * wi_65 * xj_26 * yk_3;
        sum_v_225 += scalar_t(0.487950057f) * wi_65 * xj_27 * yk_2;
        sum_v_225 += scalar_t(-0.154303357f) * wi_65 * xj_28 * yk_3;
        sum_v_225 += scalar_t(-0.154303357f) * wi_65 * xj_30 * yk_1;
        sum_v_225 += scalar_t(-0.597614288f) * wi_65 * xj_32 * yk_1;

        scalar_t sum_v_395 = scalar_t(0);
        sum_v_395 += scalar_t(0.353553385f) * wi_90 * xj_26 * yk_1;
        sum_v_395 += scalar_t(-0.456435442f) * wi_90 * xj_28 * yk_1;
        sum_v_395 += scalar_t(0.456435442f) * wi_90 * xj_30 * yk_3;
        sum_v_395 += scalar_t(-0.577350259f) * wi_90 * xj_31 * yk_2;
        sum_v_395 += scalar_t(0.353553385f) * wi_90 * xj_32 * yk_3;

        scalar_t sum_v_398 = scalar_t(0);
        sum_v_398 += scalar_t(-0.456435442f) * wi_90 * xj_27 * yk_3;
        sum_v_398 += scalar_t(0.288675129f) * wi_90 * xj_28 * yk_2;
        sum_v_398 += scalar_t(-0.707106769f) * wi_90 * xj_29 * yk_1;
        sum_v_398 += scalar_t(0.456435442f) * wi_90 * xj_31 * yk_1;

        scalar_t sum_v_400 = scalar_t(0);
        sum_v_400 += scalar_t(0.866025388f) * wi_90 * xj_26 * yk_2;
        sum_v_400 += scalar_t(-0.353553385f) * wi_90 * xj_27 * yk_3;
        sum_v_400 += scalar_t(-0.353553385f) * wi_90 * xj_31 * yk_1;

        scalar_t sum_v_228 = scalar_t(0);
        sum_v_228 += scalar_t(0.487950057f) * wi_65 * xj_27 * yk_1;
        sum_v_228 += scalar_t(-0.377964467f) * wi_65 * xj_29 * yk_3;
        sum_v_228 += scalar_t(0.617213428f) * wi_65 * xj_30 * yk_2;
        sum_v_228 += scalar_t(0.487950057f) * wi_65 * xj_31 * yk_3;

        scalar_t sum_v_397 = scalar_t(0);
        sum_v_397 += scalar_t(-0.707106769f) * wi_90 * xj_28 * yk_3;
        sum_v_397 += scalar_t(0.707106769f) * wi_90 * xj_30 * yk_1;

        scalar_t sum_v_373 = scalar_t(0);
        sum_v_373 += scalar_t(0.707106769f) * wi_87 * xj_16 * yk_3;
        sum_v_373 += scalar_t(0.707106769f) * wi_87 * xj_20 * yk_1;

        scalar_t sum_v_297 = scalar_t(0);
        sum_v_297 += scalar_t(0.707106769f) * wi_76 * xj_18 * yk_8;
        sum_v_297 += scalar_t(-0.707106769f) * wi_76 * xj_20 * yk_6;

        scalar_t sum_v_268 = scalar_t(0);
        sum_v_268 += scalar_t(0.707106769f) * wi_72 * xj_4 * yk_8;
        sum_v_268 += scalar_t(0.707106769f) * wi_72 * xj_8 * yk_4;

        scalar_t sum_v_267 = scalar_t(0);
        sum_v_267 += wi_71 * xj_3 * yk_15;

        scalar_t sum_v_259 = scalar_t(0);
        sum_v_259 += wi_71 * xj_3 * yk_13;

        scalar_t sum_v_251 = scalar_t(0);
        sum_v_251 += wi_71 * xj_3 * yk_11;

        scalar_t sum_v_243 = scalar_t(0);
        sum_v_243 += wi_71 * xj_3 * yk_9;

        scalar_t sum_v_246 = scalar_t(0);
        sum_v_246 += wi_70 * xj_2 * yk_10;

        scalar_t sum_v_254 = scalar_t(0);
        sum_v_254 += wi_70 * xj_2 * yk_12;

        scalar_t sum_v_262 = scalar_t(0);
        sum_v_262 += wi_70 * xj_2 * yk_14;

        scalar_t sum_v_265 = scalar_t(0);
        sum_v_265 += wi_69 * xj_1 * yk_15;

        scalar_t sum_v_257 = scalar_t(0);
        sum_v_257 += wi_69 * xj_1 * yk_13;

        scalar_t sum_v_249 = scalar_t(0);
        sum_v_249 += wi_69 * xj_1 * yk_11;

        scalar_t sum_v_241 = scalar_t(0);
        sum_v_241 += wi_69 * xj_1 * yk_9;

        scalar_t sum_v_244 = scalar_t(0);
        sum_v_244 += wi_68 * xj_0 * yk_10;

        scalar_t sum_v_252 = scalar_t(0);
        sum_v_252 += wi_68 * xj_0 * yk_12;

        scalar_t sum_v_260 = scalar_t(0);
        sum_v_260 += wi_68 * xj_0 * yk_14;

        scalar_t sum_v_171 = scalar_t(0);
        sum_v_171 += wi_51 * xj_0 * yk_8;

        scalar_t sum_v_163 = scalar_t(0);
        sum_v_163 += wi_51 * xj_0 * yk_6;

        scalar_t sum_v_155 = scalar_t(0);
        sum_v_155 += wi_51 * xj_0 * yk_4;

        scalar_t sum_v_160 = scalar_t(0);
        sum_v_160 += wi_52 * xj_1 * yk_5;

        scalar_t sum_v_168 = scalar_t(0);
        sum_v_168 += wi_52 * xj_1 * yk_7;

        scalar_t sum_v_173 = scalar_t(0);
        sum_v_173 += wi_53 * xj_2 * yk_8;

        scalar_t sum_v_165 = scalar_t(0);
        sum_v_165 += wi_53 * xj_2 * yk_6;

        scalar_t sum_v_157 = scalar_t(0);
        sum_v_157 += wi_53 * xj_2 * yk_4;

        scalar_t sum_v_162 = scalar_t(0);
        sum_v_162 += wi_54 * xj_3 * yk_5;

        scalar_t sum_v_170 = scalar_t(0);
        sum_v_170 += wi_54 * xj_3 * yk_7;

        scalar_t sum_v_184 = scalar_t(0);
        sum_v_184 += scalar_t(-0.707106769f) * wi_56 * xj_5 * yk_1;
        sum_v_184 += scalar_t(0.707106769f) * wi_56 * xj_9 * yk_3;

        scalar_t sum_v_178 = scalar_t(0);
        sum_v_178 += scalar_t(0.707106769f) * wi_56 * xj_5 * yk_2;
        sum_v_178 += scalar_t(0.707106769f) * wi_56 * xj_7 * yk_1;

        scalar_t sum_v_215 = scalar_t(0);
        sum_v_215 += wi_63 * xj_21 * yk_0;

        scalar_t sum_v_217 = scalar_t(0);
        sum_v_217 += wi_63 * xj_23 * yk_0;

        scalar_t sum_v_219 = scalar_t(0);
        sum_v_219 += wi_63 * xj_25 * yk_0;

        scalar_t sum_v_309 = scalar_t(0);
        sum_v_309 += scalar_t(-0.707106769f) * wi_77 * xj_21 * yk_1;
        sum_v_309 += scalar_t(0.707106769f) * wi_77 * xj_25 * yk_3;

        scalar_t sum_v_409 = scalar_t(0);
        sum_v_409 += wi_92 * xj_34 * yk_0;

        scalar_t sum_v_411 = scalar_t(0);
        sum_v_411 += wi_92 * xj_36 * yk_0;

        scalar_t sum_v_413 = scalar_t(0);
        sum_v_413 += wi_92 * xj_38 * yk_0;

        scalar_t sum_v_392 = scalar_t(0);
        sum_v_392 += scalar_t(0.707106769f) * wi_89 * xj_21 * yk_6;
        sum_v_392 += scalar_t(-0.707106769f) * wi_89 * xj_23 * yk_4;

        scalar_t sum_v_323 = scalar_t(0);
        sum_v_323 += wi_79 * xj_32 * yk_0;

        scalar_t sum_v_321 = scalar_t(0);
        sum_v_321 += wi_79 * xj_30 * yk_0;

        scalar_t sum_v_319 = scalar_t(0);
        sum_v_319 += wi_79 * xj_28 * yk_0;

        scalar_t sum_v_317 = scalar_t(0);
        sum_v_317 += wi_79 * xj_26 * yk_0;

        scalar_t sum_v_123 = scalar_t(0);
        sum_v_123 += wi_44 * xj_19 * yk_0;

        scalar_t sum_v_121 = scalar_t(0);
        sum_v_121 += wi_44 * xj_17 * yk_0;

        // writeback
        atomicAdd(&out[o_base + (54LL << 5)], sum_v_54);
        atomicAdd(&out[o_base + (3LL << 5)], sum_v_3);
        atomicAdd(&out[o_base + (151LL << 5)], sum_v_151);
        atomicAdd(&out[o_base + (154LL << 5)], sum_v_154);
        atomicAdd(&out[o_base + (152LL << 5)], sum_v_152);
        atomicAdd(&out[o_base + (89LL << 5)], sum_v_89);
        atomicAdd(&out[o_base + (55LL << 5)], sum_v_55);
        atomicAdd(&out[o_base + (340LL << 5)], sum_v_340);
        atomicAdd(&out[o_base + (338LL << 5)], sum_v_338);
        atomicAdd(&out[o_base + (343LL << 5)], sum_v_343);
        atomicAdd(&out[o_base + (417LL << 5)], sum_v_417);
        atomicAdd(&out[o_base + (418LL << 5)], sum_v_418);
        atomicAdd(&out[o_base + (421LL << 5)], sum_v_421);
        atomicAdd(&out[o_base + (416LL << 5)], sum_v_416);
        atomicAdd(&out[o_base + (239LL << 5)], sum_v_239);
        atomicAdd(&out[o_base + (236LL << 5)], sum_v_236);
        atomicAdd(&out[o_base + (145LL << 5)], sum_v_145);
        atomicAdd(&out[o_base + (332LL << 5)], sum_v_332);
        atomicAdd(&out[o_base + (333LL << 5)], sum_v_333);
        atomicAdd(&out[o_base + (337LL << 5)], sum_v_337);
        atomicAdd(&out[o_base + (237LL << 5)], sum_v_237);
        atomicAdd(&out[o_base + (146LL << 5)], sum_v_146);
        atomicAdd(&out[o_base + (147LL << 5)], sum_v_147);
        atomicAdd(&out[o_base + (50LL << 5)], sum_v_50);
        atomicAdd(&out[o_base + (81LL << 5)], sum_v_81);
        atomicAdd(&out[o_base + (10LL << 5)], sum_v_10);
        atomicAdd(&out[o_base + (82LL << 5)], sum_v_82);
        atomicAdd(&out[o_base + (221LL << 5)], sum_v_221);
        atomicAdd(&out[o_base + (220LL << 5)], sum_v_220);
        atomicAdd(&out[o_base + (139LL << 5)], sum_v_139);
        atomicAdd(&out[o_base + (136LL << 5)], sum_v_136);
        atomicAdd(&out[o_base + (312LL << 5)], sum_v_312);
        atomicAdd(&out[o_base + (313LL << 5)], sum_v_313);
        atomicAdd(&out[o_base + (316LL << 5)], sum_v_316);
        atomicAdd(&out[o_base + (389LL << 5)], sum_v_389);
        atomicAdd(&out[o_base + (387LL << 5)], sum_v_387);
        atomicAdd(&out[o_base + (311LL << 5)], sum_v_311);
        atomicAdd(&out[o_base + (304LL << 5)], sum_v_304);
        atomicAdd(&out[o_base + (305LL << 5)], sum_v_305);
        atomicAdd(&out[o_base + (308LL << 5)], sum_v_308);
        atomicAdd(&out[o_base + (131LL << 5)], sum_v_131);
        atomicAdd(&out[o_base + (45LL << 5)], sum_v_45);
        atomicAdd(&out[o_base + (46LL << 5)], sum_v_46);
        atomicAdd(&out[o_base + (132LL << 5)], sum_v_132);
        atomicAdd(&out[o_base + (119LL << 5)], sum_v_119);
        atomicAdd(&out[o_base + (117LL << 5)], sum_v_117);
        atomicAdd(&out[o_base + (115LL << 5)], sum_v_115);
        atomicAdd(&out[o_base + (74LL << 5)], sum_v_74);
        atomicAdd(&out[o_base + (1LL << 5)], sum_v_1);
        atomicAdd(&out[o_base + (37LL << 5)], sum_v_37);
        atomicAdd(&out[o_base + (103LL << 5)], sum_v_103);
        atomicAdd(&out[o_base + (39LL << 5)], sum_v_39);
        atomicAdd(&out[o_base + (198LL << 5)], sum_v_198);
        atomicAdd(&out[o_base + (204LL << 5)], sum_v_204);
        atomicAdd(&out[o_base + (200LL << 5)], sum_v_200);
        atomicAdd(&out[o_base + (293LL << 5)], sum_v_293);
        atomicAdd(&out[o_base + (291LL << 5)], sum_v_291);
        atomicAdd(&out[o_base + (295LL << 5)], sum_v_295);
        atomicAdd(&out[o_base + (364LL << 5)], sum_v_364);
        atomicAdd(&out[o_base + (366LL << 5)], sum_v_366);
        atomicAdd(&out[o_base + (360LL << 5)], sum_v_360);
        atomicAdd(&out[o_base + (288LL << 5)], sum_v_288);
        atomicAdd(&out[o_base + (292LL << 5)], sum_v_292);
        atomicAdd(&out[o_base + (290LL << 5)], sum_v_290);
        atomicAdd(&out[o_base + (282LL << 5)], sum_v_282);
        atomicAdd(&out[o_base + (361LL << 5)], sum_v_361);
        atomicAdd(&out[o_base + (363LL << 5)], sum_v_363);
        atomicAdd(&out[o_base + (369LL << 5)], sum_v_369);
        atomicAdd(&out[o_base + (359LL << 5)], sum_v_359);
        atomicAdd(&out[o_base + (201LL << 5)], sum_v_201);
        atomicAdd(&out[o_base + (203LL << 5)], sum_v_203);
        atomicAdd(&out[o_base + (110LL << 5)], sum_v_110);
        atomicAdd(&out[o_base + (112LL << 5)], sum_v_112);
        atomicAdd(&out[o_base + (104LL << 5)], sum_v_104);
        atomicAdd(&out[o_base + (73LL << 5)], sum_v_73);
        atomicAdd(&out[o_base + (0LL << 5)], sum_v_0);
        atomicAdd(&out[o_base + (36LL << 5)], sum_v_36);
        atomicAdd(&out[o_base + (108LL << 5)], sum_v_108);
        atomicAdd(&out[o_base + (38LL << 5)], sum_v_38);
        atomicAdd(&out[o_base + (102LL << 5)], sum_v_102);
        atomicAdd(&out[o_base + (65LL << 5)], sum_v_65);
        atomicAdd(&out[o_base + (63LL << 5)], sum_v_63);
        atomicAdd(&out[o_base + (67LL << 5)], sum_v_67);
        atomicAdd(&out[o_base + (31LL << 5)], sum_v_31);
        atomicAdd(&out[o_base + (33LL << 5)], sum_v_33);
        atomicAdd(&out[o_base + (93LL << 5)], sum_v_93);
        atomicAdd(&out[o_base + (99LL << 5)], sum_v_99);
        atomicAdd(&out[o_base + (95LL << 5)], sum_v_95);
        atomicAdd(&out[o_base + (58LL << 5)], sum_v_58);
        atomicAdd(&out[o_base + (194LL << 5)], sum_v_194);
        atomicAdd(&out[o_base + (192LL << 5)], sum_v_192);
        atomicAdd(&out[o_base + (190LL << 5)], sum_v_190);
        atomicAdd(&out[o_base + (273LL << 5)], sum_v_273);
        atomicAdd(&out[o_base + (275LL << 5)], sum_v_275);
        atomicAdd(&out[o_base + (269LL << 5)], sum_v_269);
        atomicAdd(&out[o_base + (346LL << 5)], sum_v_346);
        atomicAdd(&out[o_base + (356LL << 5)], sum_v_356);
        atomicAdd(&out[o_base + (354LL << 5)], sum_v_354);
        atomicAdd(&out[o_base + (351LL << 5)], sum_v_351);
        atomicAdd(&out[o_base + (355LL << 5)], sum_v_355);
        atomicAdd(&out[o_base + (353LL << 5)], sum_v_353);
        atomicAdd(&out[o_base + (345LL << 5)], sum_v_345);
        atomicAdd(&out[o_base + (278LL << 5)], sum_v_278);
        atomicAdd(&out[o_base + (276LL << 5)], sum_v_276);
        atomicAdd(&out[o_base + (274LL << 5)], sum_v_274);
        atomicAdd(&out[o_base + (185LL << 5)], sum_v_185);
        atomicAdd(&out[o_base + (187LL << 5)], sum_v_187);
        atomicAdd(&out[o_base + (179LL << 5)], sum_v_179);
        atomicAdd(&out[o_base + (175LL << 5)], sum_v_175);
        atomicAdd(&out[o_base + (181LL << 5)], sum_v_181);
        atomicAdd(&out[o_base + (96LL << 5)], sum_v_96);
        atomicAdd(&out[o_base + (90LL << 5)], sum_v_90);
        atomicAdd(&out[o_base + (34LL << 5)], sum_v_34);
        atomicAdd(&out[o_base + (8LL << 5)], sum_v_8);
        atomicAdd(&out[o_base + (59LL << 5)], sum_v_59);
        atomicAdd(&out[o_base + (61LL << 5)], sum_v_61);
        atomicAdd(&out[o_base + (26LL << 5)], sum_v_26);
        atomicAdd(&out[o_base + (24LL << 5)], sum_v_24);
        atomicAdd(&out[o_base + (28LL << 5)], sum_v_28);
        atomicAdd(&out[o_base + (7LL << 5)], sum_v_7);
        atomicAdd(&out[o_base + (19LL << 5)], sum_v_19);
        atomicAdd(&out[o_base + (22LL << 5)], sum_v_22);
        atomicAdd(&out[o_base + (14LL << 5)], sum_v_14);
        atomicAdd(&out[o_base + (5LL << 5)], sum_v_5);
        atomicAdd(&out[o_base + (17LL << 5)], sum_v_17);
        atomicAdd(&out[o_base + (20LL << 5)], sum_v_20);
        atomicAdd(&out[o_base + (12LL << 5)], sum_v_12);
        atomicAdd(&out[o_base + (42LL << 5)], sum_v_42);
        atomicAdd(&out[o_base + (126LL << 5)], sum_v_126);
        atomicAdd(&out[o_base + (127LL << 5)], sum_v_127);
        atomicAdd(&out[o_base + (129LL << 5)], sum_v_129);
        atomicAdd(&out[o_base + (80LL << 5)], sum_v_80);
        atomicAdd(&out[o_base + (77LL << 5)], sum_v_77);
        atomicAdd(&out[o_base + (208LL << 5)], sum_v_208);
        atomicAdd(&out[o_base + (213LL << 5)], sum_v_213);
        atomicAdd(&out[o_base + (211LL << 5)], sum_v_211);
        atomicAdd(&out[o_base + (206LL << 5)], sum_v_206);
        atomicAdd(&out[o_base + (298LL << 5)], sum_v_298);
        atomicAdd(&out[o_base + (296LL << 5)], sum_v_296);
        atomicAdd(&out[o_base + (380LL << 5)], sum_v_380);
        atomicAdd(&out[o_base + (382LL << 5)], sum_v_382);
        atomicAdd(&out[o_base + (386LL << 5)], sum_v_386);
        atomicAdd(&out[o_base + (377LL << 5)], sum_v_377);
        atomicAdd(&out[o_base + (374LL << 5)], sum_v_374);
        atomicAdd(&out[o_base + (385LL << 5)], sum_v_385);
        atomicAdd(&out[o_base + (376LL << 5)], sum_v_376);
        atomicAdd(&out[o_base + (76LL << 5)], sum_v_76);
        atomicAdd(&out[o_base + (43LL << 5)], sum_v_43);
        atomicAdd(&out[o_base + (51LL << 5)], sum_v_51);
        atomicAdd(&out[o_base + (84LL << 5)], sum_v_84);
        atomicAdd(&out[o_base + (85LL << 5)], sum_v_85);
        atomicAdd(&out[o_base + (231LL << 5)], sum_v_231);
        atomicAdd(&out[o_base + (234LL << 5)], sum_v_234);
        atomicAdd(&out[o_base + (405LL << 5)], sum_v_405);
        atomicAdd(&out[o_base + (403LL << 5)], sum_v_403);
        atomicAdd(&out[o_base + (401LL << 5)], sum_v_401);
        atomicAdd(&out[o_base + (328LL << 5)], sum_v_328);
        atomicAdd(&out[o_base + (327LL << 5)], sum_v_327);
        atomicAdd(&out[o_base + (330LL << 5)], sum_v_330);
        atomicAdd(&out[o_base + (325LL << 5)], sum_v_325);
        atomicAdd(&out[o_base + (144LL << 5)], sum_v_144);
        atomicAdd(&out[o_base + (141LL << 5)], sum_v_141);
        atomicAdd(&out[o_base + (225LL << 5)], sum_v_225);
        atomicAdd(&out[o_base + (395LL << 5)], sum_v_395);
        atomicAdd(&out[o_base + (398LL << 5)], sum_v_398);
        atomicAdd(&out[o_base + (400LL << 5)], sum_v_400);
        atomicAdd(&out[o_base + (228LL << 5)], sum_v_228);
        atomicAdd(&out[o_base + (397LL << 5)], sum_v_397);
        atomicAdd(&out[o_base + (373LL << 5)], sum_v_373);
        atomicAdd(&out[o_base + (297LL << 5)], sum_v_297);
        atomicAdd(&out[o_base + (268LL << 5)], sum_v_268);
        atomicAdd(&out[o_base + (267LL << 5)], sum_v_267);
        atomicAdd(&out[o_base + (259LL << 5)], sum_v_259);
        atomicAdd(&out[o_base + (251LL << 5)], sum_v_251);
        atomicAdd(&out[o_base + (243LL << 5)], sum_v_243);
        atomicAdd(&out[o_base + (246LL << 5)], sum_v_246);
        atomicAdd(&out[o_base + (254LL << 5)], sum_v_254);
        atomicAdd(&out[o_base + (262LL << 5)], sum_v_262);
        atomicAdd(&out[o_base + (265LL << 5)], sum_v_265);
        atomicAdd(&out[o_base + (257LL << 5)], sum_v_257);
        atomicAdd(&out[o_base + (249LL << 5)], sum_v_249);
        atomicAdd(&out[o_base + (241LL << 5)], sum_v_241);
        atomicAdd(&out[o_base + (244LL << 5)], sum_v_244);
        atomicAdd(&out[o_base + (252LL << 5)], sum_v_252);
        atomicAdd(&out[o_base + (260LL << 5)], sum_v_260);
        atomicAdd(&out[o_base + (171LL << 5)], sum_v_171);
        atomicAdd(&out[o_base + (163LL << 5)], sum_v_163);
        atomicAdd(&out[o_base + (155LL << 5)], sum_v_155);
        atomicAdd(&out[o_base + (160LL << 5)], sum_v_160);
        atomicAdd(&out[o_base + (168LL << 5)], sum_v_168);
        atomicAdd(&out[o_base + (173LL << 5)], sum_v_173);
        atomicAdd(&out[o_base + (165LL << 5)], sum_v_165);
        atomicAdd(&out[o_base + (157LL << 5)], sum_v_157);
        atomicAdd(&out[o_base + (162LL << 5)], sum_v_162);
        atomicAdd(&out[o_base + (170LL << 5)], sum_v_170);
        atomicAdd(&out[o_base + (184LL << 5)], sum_v_184);
        atomicAdd(&out[o_base + (178LL << 5)], sum_v_178);
        atomicAdd(&out[o_base + (215LL << 5)], sum_v_215);
        atomicAdd(&out[o_base + (217LL << 5)], sum_v_217);
        atomicAdd(&out[o_base + (219LL << 5)], sum_v_219);
        atomicAdd(&out[o_base + (309LL << 5)], sum_v_309);
        atomicAdd(&out[o_base + (409LL << 5)], sum_v_409);
        atomicAdd(&out[o_base + (411LL << 5)], sum_v_411);
        atomicAdd(&out[o_base + (413LL << 5)], sum_v_413);
        atomicAdd(&out[o_base + (392LL << 5)], sum_v_392);
        atomicAdd(&out[o_base + (323LL << 5)], sum_v_323);
        atomicAdd(&out[o_base + (321LL << 5)], sum_v_321);
        atomicAdd(&out[o_base + (319LL << 5)], sum_v_319);
        atomicAdd(&out[o_base + (317LL << 5)], sum_v_317);
        atomicAdd(&out[o_base + (123LL << 5)], sum_v_123);
        atomicAdd(&out[o_base + (121LL << 5)], sum_v_121);
    }

    if (warp == 1) {
        // preload w(i)
        scalar_t wi_26 = w[w_base + 26LL * 32 + lane];
        scalar_t wi_50 = w[w_base + 50LL * 32 + lane];
        scalar_t wi_37 = w[w_base + 37LL * 32 + lane];
        scalar_t wi_82 = w[w_base + 82LL * 32 + lane];
        scalar_t wi_93 = w[w_base + 93LL * 32 + lane];
        scalar_t wi_67 = w[w_base + 67LL * 32 + lane];
        scalar_t wi_49 = w[w_base + 49LL * 32 + lane];
        scalar_t wi_81 = w[w_base + 81LL * 32 + lane];
        scalar_t wi_46 = w[w_base + 46LL * 32 + lane];
        scalar_t wi_47 = w[w_base + 47LL * 32 + lane];
        scalar_t wi_24 = w[w_base + 24LL * 32 + lane];
        scalar_t wi_35 = w[w_base + 35LL * 32 + lane];
        scalar_t wi_64 = w[w_base + 64LL * 32 + lane];
        scalar_t wi_78 = w[w_base + 78LL * 32 + lane];
        scalar_t wi_89 = w[w_base + 89LL * 32 + lane];
        scalar_t wi_77 = w[w_base + 77LL * 32 + lane];
        scalar_t wi_23 = w[w_base + 23LL * 32 + lane];
        scalar_t wi_41 = w[w_base + 41LL * 32 + lane];
        scalar_t wi_43 = w[w_base + 43LL * 32 + lane];
        scalar_t wi_32 = w[w_base + 32LL * 32 + lane];
        scalar_t wi_21 = w[w_base + 21LL * 32 + lane];
        scalar_t wi_60 = w[w_base + 60LL * 32 + lane];
        scalar_t wi_75 = w[w_base + 75LL * 32 + lane];
        scalar_t wi_86 = w[w_base + 86LL * 32 + lane];
        scalar_t wi_74 = w[w_base + 74LL * 32 + lane];
        scalar_t wi_85 = w[w_base + 85LL * 32 + lane];
        scalar_t wi_59 = w[w_base + 59LL * 32 + lane];
        scalar_t wi_42 = w[w_base + 42LL * 32 + lane];
        scalar_t wi_31 = w[w_base + 31LL * 32 + lane];
        scalar_t wi_40 = w[w_base + 40LL * 32 + lane];
        scalar_t wi_20 = w[w_base + 20LL * 32 + lane];
        scalar_t wi_30 = w[w_base + 30LL * 32 + lane];
        scalar_t wi_28 = w[w_base + 28LL * 32 + lane];
        scalar_t wi_9 = w[w_base + 9LL * 32 + lane];
        scalar_t wi_19 = w[w_base + 19LL * 32 + lane];
        scalar_t wi_39 = w[w_base + 39LL * 32 + lane];
        scalar_t wi_56 = w[w_base + 56LL * 32 + lane];
        scalar_t wi_58 = w[w_base + 58LL * 32 + lane];
        scalar_t wi_73 = w[w_base + 73LL * 32 + lane];
        scalar_t wi_84 = w[w_base + 84LL * 32 + lane];
        scalar_t wi_83 = w[w_base + 83LL * 32 + lane];
        scalar_t wi_72 = w[w_base + 72LL * 32 + lane];
        scalar_t wi_57 = w[w_base + 57LL * 32 + lane];
        scalar_t wi_55 = w[w_base + 55LL * 32 + lane];
        scalar_t wi_38 = w[w_base + 38LL * 32 + lane];
        scalar_t wi_18 = w[w_base + 18LL * 32 + lane];
        scalar_t wi_27 = w[w_base + 27LL * 32 + lane];
        scalar_t wi_17 = w[w_base + 17LL * 32 + lane];
        scalar_t wi_15 = w[w_base + 15LL * 32 + lane];
        scalar_t wi_14 = w[w_base + 14LL * 32 + lane];
        scalar_t wi_6 = w[w_base + 6LL * 32 + lane];
        scalar_t wi_13 = w[w_base + 13LL * 32 + lane];
        scalar_t wi_12 = w[w_base + 12LL * 32 + lane];
        scalar_t wi_4 = w[w_base + 4LL * 32 + lane];
        scalar_t wi_2 = w[w_base + 2LL * 32 + lane];
        scalar_t wi_22 = w[w_base + 22LL * 32 + lane];
        scalar_t wi_45 = w[w_base + 45LL * 32 + lane];
        scalar_t wi_34 = w[w_base + 34LL * 32 + lane];
        scalar_t wi_33 = w[w_base + 33LL * 32 + lane];
        scalar_t wi_62 = w[w_base + 62LL * 32 + lane];
        scalar_t wi_61 = w[w_base + 61LL * 32 + lane];
        scalar_t wi_76 = w[w_base + 76LL * 32 + lane];
        scalar_t wi_88 = w[w_base + 88LL * 32 + lane];
        scalar_t wi_87 = w[w_base + 87LL * 32 + lane];
        scalar_t wi_25 = w[w_base + 25LL * 32 + lane];
        scalar_t wi_36 = w[w_base + 36LL * 32 + lane];
        scalar_t wi_11 = w[w_base + 11LL * 32 + lane];
        scalar_t wi_66 = w[w_base + 66LL * 32 + lane];
        scalar_t wi_91 = w[w_base + 91LL * 32 + lane];
        scalar_t wi_80 = w[w_base + 80LL * 32 + lane];
        scalar_t wi_48 = w[w_base + 48LL * 32 + lane];
        scalar_t wi_65 = w[w_base + 65LL * 32 + lane];
        scalar_t wi_90 = w[w_base + 90LL * 32 + lane];
        scalar_t wi_71 = w[w_base + 71LL * 32 + lane];
        scalar_t wi_70 = w[w_base + 70LL * 32 + lane];
        scalar_t wi_69 = w[w_base + 69LL * 32 + lane];
        scalar_t wi_68 = w[w_base + 68LL * 32 + lane];
        scalar_t wi_51 = w[w_base + 51LL * 32 + lane];
        scalar_t wi_52 = w[w_base + 52LL * 32 + lane];
        scalar_t wi_53 = w[w_base + 53LL * 32 + lane];
        scalar_t wi_54 = w[w_base + 54LL * 32 + lane];
        scalar_t wi_63 = w[w_base + 63LL * 32 + lane];
        scalar_t wi_92 = w[w_base + 92LL * 32 + lane];
        scalar_t wi_79 = w[w_base + 79LL * 32 + lane];
        scalar_t wi_44 = w[w_base + 44LL * 32 + lane];

        // preload x(j)
        scalar_t xj_33 = x_all[x_base + 33LL * 32 + lane];
        scalar_t xj_34 = x_all[x_base + 34LL * 32 + lane];
        scalar_t xj_35 = x_all[x_base + 35LL * 32 + lane];
        scalar_t xj_36 = x_all[x_base + 36LL * 32 + lane];
        scalar_t xj_37 = x_all[x_base + 37LL * 32 + lane];
        scalar_t xj_38 = x_all[x_base + 38LL * 32 + lane];
        scalar_t xj_39 = x_all[x_base + 39LL * 32 + lane];
        scalar_t xj_21 = x_all[x_base + 21LL * 32 + lane];
        scalar_t xj_22 = x_all[x_base + 22LL * 32 + lane];
        scalar_t xj_24 = x_all[x_base + 24LL * 32 + lane];
        scalar_t xj_25 = x_all[x_base + 25LL * 32 + lane];
        scalar_t xj_23 = x_all[x_base + 23LL * 32 + lane];
        scalar_t xj_11 = x_all[x_base + 11LL * 32 + lane];
        scalar_t xj_15 = x_all[x_base + 15LL * 32 + lane];
        scalar_t xj_13 = x_all[x_base + 13LL * 32 + lane];
        scalar_t xj_10 = x_all[x_base + 10LL * 32 + lane];
        scalar_t xj_12 = x_all[x_base + 12LL * 32 + lane];
        scalar_t xj_14 = x_all[x_base + 14LL * 32 + lane];
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
        scalar_t xj_16 = x_all[x_base + 16LL * 32 + lane];
        scalar_t xj_17 = x_all[x_base + 17LL * 32 + lane];
        scalar_t xj_18 = x_all[x_base + 18LL * 32 + lane];
        scalar_t xj_19 = x_all[x_base + 19LL * 32 + lane];
        scalar_t xj_20 = x_all[x_base + 20LL * 32 + lane];
        scalar_t xj_26 = x_all[x_base + 26LL * 32 + lane];
        scalar_t xj_27 = x_all[x_base + 27LL * 32 + lane];
        scalar_t xj_28 = x_all[x_base + 28LL * 32 + lane];
        scalar_t xj_29 = x_all[x_base + 29LL * 32 + lane];
        scalar_t xj_30 = x_all[x_base + 30LL * 32 + lane];
        scalar_t xj_31 = x_all[x_base + 31LL * 32 + lane];
        scalar_t xj_32 = x_all[x_base + 32LL * 32 + lane];

        // preload y(k)
        scalar_t yk_14 = y[y_base + 14];
        scalar_t yk_13 = y[y_base + 13];
        scalar_t yk_15 = y[y_base + 15];
        scalar_t yk_12 = y[y_base + 12];
        scalar_t yk_11 = y[y_base + 11];
        scalar_t yk_10 = y[y_base + 10];
        scalar_t yk_9 = y[y_base + 9];
        scalar_t yk_8 = y[y_base + 8];
        scalar_t yk_7 = y[y_base + 7];
        scalar_t yk_6 = y[y_base + 6];
        scalar_t yk_5 = y[y_base + 5];
        scalar_t yk_4 = y[y_base + 4];
        scalar_t yk_1 = y[y_base + 1];
        scalar_t yk_3 = y[y_base + 3];
        scalar_t yk_2 = y[y_base + 2];
        scalar_t yk_0 = y[y_base + 0];

        // per-v accumulation
        scalar_t sum_v_56 = scalar_t(0);
        sum_v_56 += scalar_t(0.231455028f) * wi_26 * xj_33 * yk_14;
        sum_v_56 += scalar_t(0.298807144f) * wi_26 * xj_34 * yk_13;
        sum_v_56 += scalar_t(0.231455028f) * wi_26 * xj_34 * yk_15;
        sum_v_56 += scalar_t(0.462910056f) * wi_26 * xj_35 * yk_12;
        sum_v_56 += scalar_t(0.298807144f) * wi_26 * xj_35 * yk_14;
        sum_v_56 += scalar_t(-0.462910056f) * wi_26 * xj_36 * yk_11;
        sum_v_56 += scalar_t(-0.298807144f) * wi_26 * xj_37 * yk_10;
        sum_v_56 += scalar_t(-0.231455028f) * wi_26 * xj_38 * yk_9;
        sum_v_56 += scalar_t(-0.298807144f) * wi_26 * xj_38 * yk_11;
        sum_v_56 += scalar_t(-0.231455028f) * wi_26 * xj_39 * yk_10;

        scalar_t sum_v_153 = scalar_t(0);
        sum_v_153 += scalar_t(0.38575837f) * wi_50 * xj_33 * yk_10;
        sum_v_153 += scalar_t(0.38575837f) * wi_50 * xj_34 * yk_9;
        sum_v_153 += scalar_t(0.298807144f) * wi_50 * xj_34 * yk_11;
        sum_v_153 += scalar_t(0.298807144f) * wi_50 * xj_35 * yk_10;
        sum_v_153 += scalar_t(0.154303357f) * wi_50 * xj_36 * yk_13;
        sum_v_153 += scalar_t(0.154303357f) * wi_50 * xj_37 * yk_12;
        sum_v_153 += scalar_t(0.298807144f) * wi_50 * xj_37 * yk_14;
        sum_v_153 += scalar_t(0.298807144f) * wi_50 * xj_38 * yk_13;
        sum_v_153 += scalar_t(0.38575837f) * wi_50 * xj_38 * yk_15;
        sum_v_153 += scalar_t(0.38575837f) * wi_50 * xj_39 * yk_14;

        scalar_t sum_v_150 = scalar_t(0);
        sum_v_150 += scalar_t(-0.243975028f) * wi_50 * xj_33 * yk_13;
        sum_v_150 += scalar_t(-0.487950057f) * wi_50 * xj_34 * yk_12;
        sum_v_150 += scalar_t(0.377964467f) * wi_50 * xj_35 * yk_13;
        sum_v_150 += scalar_t(0.243975028f) * wi_50 * xj_35 * yk_15;
        sum_v_150 += scalar_t(-0.487950057f) * wi_50 * xj_36 * yk_10;
        sum_v_150 += scalar_t(-0.243975028f) * wi_50 * xj_37 * yk_9;
        sum_v_150 += scalar_t(0.377964467f) * wi_50 * xj_37 * yk_11;
        sum_v_150 += scalar_t(0.243975028f) * wi_50 * xj_39 * yk_11;

        scalar_t sum_v_87 = scalar_t(0);
        sum_v_87 += scalar_t(0.462910056f) * wi_37 * xj_33 * yk_8;
        sum_v_87 += scalar_t(0.377964467f) * wi_37 * xj_34 * yk_7;
        sum_v_87 += scalar_t(0.414039344f) * wi_37 * xj_35 * yk_6;
        sum_v_87 += scalar_t(0.119522862f) * wi_37 * xj_35 * yk_8;
        sum_v_87 += scalar_t(-0.292769998f) * wi_37 * xj_36 * yk_5;
        sum_v_87 += scalar_t(-0.119522862f) * wi_37 * xj_37 * yk_4;
        sum_v_87 += scalar_t(-0.377964467f) * wi_37 * xj_38 * yk_5;
        sum_v_87 += scalar_t(-0.462910056f) * wi_37 * xj_39 * yk_4;

        scalar_t sum_v_88 = scalar_t(0);
        sum_v_88 += scalar_t(0.377964467f) * wi_37 * xj_34 * yk_4;
        sum_v_88 += scalar_t(0.478091449f) * wi_37 * xj_35 * yk_5;
        sum_v_88 += scalar_t(0.507092535f) * wi_37 * xj_36 * yk_6;
        sum_v_88 += scalar_t(0.478091449f) * wi_37 * xj_37 * yk_7;
        sum_v_88 += scalar_t(0.377964467f) * wi_37 * xj_38 * yk_8;

        scalar_t sum_v_341 = scalar_t(0);
        sum_v_341 += scalar_t(0.408248305f) * wi_82 * xj_33 * yk_15;
        sum_v_341 += scalar_t(-0.408248305f) * wi_82 * xj_34 * yk_14;
        sum_v_341 += scalar_t(-0.408248305f) * wi_82 * xj_35 * yk_13;
        sum_v_341 += scalar_t(0.408248305f) * wi_82 * xj_37 * yk_11;
        sum_v_341 += scalar_t(0.408248305f) * wi_82 * xj_38 * yk_10;
        sum_v_341 += scalar_t(-0.408248305f) * wi_82 * xj_39 * yk_9;

        scalar_t sum_v_339 = scalar_t(0);
        sum_v_339 += scalar_t(-0.408248305f) * wi_82 * xj_33 * yk_11;
        sum_v_339 += scalar_t(0.408248305f) * wi_82 * xj_35 * yk_9;
        sum_v_339 += scalar_t(0.408248305f) * wi_82 * xj_36 * yk_14;
        sum_v_339 += scalar_t(0.408248305f) * wi_82 * xj_37 * yk_15;
        sum_v_339 += scalar_t(-0.408248305f) * wi_82 * xj_38 * yk_12;
        sum_v_339 += scalar_t(-0.408248305f) * wi_82 * xj_39 * yk_13;

        scalar_t sum_v_342 = scalar_t(0);
        sum_v_342 += scalar_t(-0.408248305f) * wi_82 * xj_33 * yk_14;
        sum_v_342 += scalar_t(-0.408248305f) * wi_82 * xj_34 * yk_15;
        sum_v_342 += scalar_t(0.408248305f) * wi_82 * xj_35 * yk_12;
        sum_v_342 += scalar_t(-0.408248305f) * wi_82 * xj_36 * yk_11;
        sum_v_342 += scalar_t(0.408248305f) * wi_82 * xj_38 * yk_9;
        sum_v_342 += scalar_t(0.408248305f) * wi_82 * xj_39 * yk_10;

        scalar_t sum_v_344 = scalar_t(0);
        sum_v_344 += scalar_t(-0.408248305f) * wi_82 * xj_33 * yk_12;
        sum_v_344 += scalar_t(0.408248305f) * wi_82 * xj_34 * yk_13;
        sum_v_344 += scalar_t(-0.408248305f) * wi_82 * xj_35 * yk_14;
        sum_v_344 += scalar_t(0.408248305f) * wi_82 * xj_36 * yk_9;
        sum_v_344 += scalar_t(-0.408248305f) * wi_82 * xj_37 * yk_10;
        sum_v_344 += scalar_t(0.408248305f) * wi_82 * xj_38 * yk_11;

        scalar_t sum_v_419 = scalar_t(0);
        sum_v_419 += scalar_t(-0.288675129f) * wi_93 * xj_33 * yk_4;
        sum_v_419 += scalar_t(0.353553385f) * wi_93 * xj_34 * yk_5;
        sum_v_419 += scalar_t(0.44721356f) * wi_93 * xj_35 * yk_4;
        sum_v_419 += scalar_t(0.182574183f) * wi_93 * xj_36 * yk_7;
        sum_v_419 += scalar_t(0.387298346f) * wi_93 * xj_37 * yk_6;
        sum_v_419 += scalar_t(0.44721356f) * wi_93 * xj_37 * yk_8;
        sum_v_419 += scalar_t(0.353553385f) * wi_93 * xj_38 * yk_7;
        sum_v_419 += scalar_t(-0.288675129f) * wi_93 * xj_39 * yk_8;

        scalar_t sum_v_415 = scalar_t(0);
        sum_v_415 += scalar_t(-0.645497203f) * wi_93 * xj_33 * yk_6;
        sum_v_415 += scalar_t(0.456435442f) * wi_93 * xj_34 * yk_7;
        sum_v_415 += scalar_t(-0.288675129f) * wi_93 * xj_35 * yk_8;
        sum_v_415 += scalar_t(-0.288675129f) * wi_93 * xj_37 * yk_4;
        sum_v_415 += scalar_t(0.456435442f) * wi_93 * xj_38 * yk_5;

        scalar_t sum_v_420 = scalar_t(0);
        sum_v_420 += scalar_t(0.456435442f) * wi_93 * xj_33 * yk_5;
        sum_v_420 += scalar_t(-0.353553385f) * wi_93 * xj_35 * yk_5;
        sum_v_420 += scalar_t(-0.577350259f) * wi_93 * xj_36 * yk_8;
        sum_v_420 += scalar_t(0.353553385f) * wi_93 * xj_37 * yk_7;
        sum_v_420 += scalar_t(0.456435442f) * wi_93 * xj_39 * yk_7;

        scalar_t sum_v_238 = scalar_t(0);
        sum_v_238 += scalar_t(-0.422577113f) * wi_67 * xj_33 * yk_8;
        sum_v_238 += scalar_t(0.377964467f) * wi_67 * xj_35 * yk_6;
        sum_v_238 += scalar_t(0.327326834f) * wi_67 * xj_35 * yk_8;
        sum_v_238 += scalar_t(-0.534522474f) * wi_67 * xj_36 * yk_5;
        sum_v_238 += scalar_t(-0.327326834f) * wi_67 * xj_37 * yk_4;
        sum_v_238 += scalar_t(0.422577113f) * wi_67 * xj_39 * yk_4;

        scalar_t sum_v_235 = scalar_t(0);
        sum_v_235 += scalar_t(-0.422577113f) * wi_67 * xj_33 * yk_5;
        sum_v_235 += scalar_t(-0.327326834f) * wi_67 * xj_35 * yk_5;
        sum_v_235 += scalar_t(-0.267261237f) * wi_67 * xj_36 * yk_8;
        sum_v_235 += scalar_t(0.327326834f) * wi_67 * xj_37 * yk_7;
        sum_v_235 += scalar_t(-0.597614288f) * wi_67 * xj_38 * yk_6;
        sum_v_235 += scalar_t(-0.422577113f) * wi_67 * xj_39 * yk_7;

        scalar_t sum_v_149 = scalar_t(0);
        sum_v_149 += scalar_t(0.597614288f) * wi_49 * xj_33 * yk_1;
        sum_v_149 += scalar_t(0.154303357f) * wi_49 * xj_35 * yk_1;
        sum_v_149 += scalar_t(-0.154303357f) * wi_49 * xj_37 * yk_3;
        sum_v_149 += scalar_t(0.487950057f) * wi_49 * xj_38 * yk_2;
        sum_v_149 += scalar_t(0.597614288f) * wi_49 * xj_39 * yk_3;

        scalar_t sum_v_336 = scalar_t(0);
        sum_v_336 += scalar_t(-0.353553385f) * wi_81 * xj_33 * yk_3;
        sum_v_336 += scalar_t(0.577350259f) * wi_81 * xj_34 * yk_2;
        sum_v_336 += scalar_t(-0.456435442f) * wi_81 * xj_35 * yk_3;
        sum_v_336 += scalar_t(-0.456435442f) * wi_81 * xj_37 * yk_1;
        sum_v_336 += scalar_t(0.353553385f) * wi_81 * xj_39 * yk_1;

        scalar_t sum_v_335 = scalar_t(0);
        sum_v_335 += scalar_t(-0.456435442f) * wi_81 * xj_34 * yk_3;
        sum_v_335 += scalar_t(0.288675129f) * wi_81 * xj_35 * yk_2;
        sum_v_335 += scalar_t(-0.707106769f) * wi_81 * xj_36 * yk_1;
        sum_v_335 += scalar_t(0.456435442f) * wi_81 * xj_38 * yk_1;

        scalar_t sum_v_331 = scalar_t(0);
        sum_v_331 += scalar_t(-0.353553385f) * wi_81 * xj_34 * yk_1;
        sum_v_331 += scalar_t(0.353553385f) * wi_81 * xj_38 * yk_3;
        sum_v_331 += scalar_t(-0.866025388f) * wi_81 * xj_39 * yk_2;

        scalar_t sum_v_148 = scalar_t(0);
        sum_v_148 += scalar_t(0.487950057f) * wi_49 * xj_34 * yk_1;
        sum_v_148 += scalar_t(-0.377964467f) * wi_49 * xj_36 * yk_3;
        sum_v_148 += scalar_t(0.617213428f) * wi_49 * xj_37 * yk_2;
        sum_v_148 += scalar_t(0.487950057f) * wi_49 * xj_38 * yk_3;

        scalar_t sum_v_134 = scalar_t(0);
        sum_v_134 += scalar_t(0.816496551f) * wi_46 * xj_21 * yk_2;
        sum_v_134 += scalar_t(-0.408248276f) * wi_46 * xj_22 * yk_3;
        sum_v_134 += scalar_t(-0.408248276f) * wi_46 * xj_24 * yk_1;

        scalar_t sum_v_137 = scalar_t(0);
        sum_v_137 += scalar_t(-0.597614288f) * wi_47 * xj_21 * yk_14;
        sum_v_137 += scalar_t(-0.377964467f) * wi_47 * xj_22 * yk_13;
        sum_v_137 += scalar_t(0.377964467f) * wi_47 * xj_24 * yk_11;
        sum_v_137 += scalar_t(0.597614288f) * wi_47 * xj_25 * yk_10;

        scalar_t sum_v_48 = scalar_t(0);
        sum_v_48 += scalar_t(-0.119522862f) * wi_24 * xj_21 * yk_13;
        sum_v_48 += scalar_t(-0.462910056f) * wi_24 * xj_21 * yk_15;
        sum_v_48 += scalar_t(-0.292769998f) * wi_24 * xj_22 * yk_12;
        sum_v_48 += scalar_t(-0.377964467f) * wi_24 * xj_22 * yk_14;
        sum_v_48 += scalar_t(0.414039344f) * wi_24 * xj_23 * yk_11;
        sum_v_48 += scalar_t(0.377964467f) * wi_24 * xj_24 * yk_10;
        sum_v_48 += scalar_t(0.462910056f) * wi_24 * xj_25 * yk_9;
        sum_v_48 += scalar_t(0.119522862f) * wi_24 * xj_25 * yk_11;

        scalar_t sum_v_49 = scalar_t(0);
        sum_v_49 += scalar_t(0.377964467f) * wi_24 * xj_21 * yk_10;
        sum_v_49 += scalar_t(0.478091449f) * wi_24 * xj_22 * yk_11;
        sum_v_49 += scalar_t(0.507092535f) * wi_24 * xj_23 * yk_12;
        sum_v_49 += scalar_t(0.478091449f) * wi_24 * xj_24 * yk_13;
        sum_v_49 += scalar_t(0.377964467f) * wi_24 * xj_25 * yk_14;

        scalar_t sum_v_83 = scalar_t(0);
        sum_v_83 += scalar_t(0.316227764f) * wi_35 * xj_21 * yk_7;
        sum_v_83 += scalar_t(0.547722518f) * wi_35 * xj_22 * yk_6;
        sum_v_83 += scalar_t(0.316227764f) * wi_35 * xj_22 * yk_8;
        sum_v_83 += scalar_t(-0.547722518f) * wi_35 * xj_23 * yk_5;
        sum_v_83 += scalar_t(-0.316227764f) * wi_35 * xj_24 * yk_4;
        sum_v_83 += scalar_t(-0.316227764f) * wi_35 * xj_25 * yk_5;

        scalar_t sum_v_222 = scalar_t(0);
        sum_v_222 += scalar_t(-0.534522474f) * wi_64 * xj_21 * yk_4;
        sum_v_222 += scalar_t(0.267261237f) * wi_64 * xj_22 * yk_5;
        sum_v_222 += scalar_t(0.534522474f) * wi_64 * xj_23 * yk_6;
        sum_v_222 += scalar_t(0.267261237f) * wi_64 * xj_24 * yk_7;
        sum_v_222 += scalar_t(-0.534522474f) * wi_64 * xj_25 * yk_8;

        scalar_t sum_v_223 = scalar_t(0);
        sum_v_223 += scalar_t(0.462910056f) * wi_64 * xj_21 * yk_5;
        sum_v_223 += scalar_t(0.462910056f) * wi_64 * xj_22 * yk_4;
        sum_v_223 += scalar_t(0.267261237f) * wi_64 * xj_23 * yk_7;
        sum_v_223 += scalar_t(0.267261237f) * wi_64 * xj_24 * yk_6;
        sum_v_223 += scalar_t(0.462910056f) * wi_64 * xj_24 * yk_8;
        sum_v_223 += scalar_t(0.462910056f) * wi_64 * xj_25 * yk_7;

        scalar_t sum_v_224 = scalar_t(0);
        sum_v_224 += scalar_t(-0.462910056f) * wi_64 * xj_22 * yk_5;
        sum_v_224 += scalar_t(-0.534522474f) * wi_64 * xj_23 * yk_8;
        sum_v_224 += scalar_t(0.462910056f) * wi_64 * xj_24 * yk_7;
        sum_v_224 += scalar_t(-0.534522474f) * wi_64 * xj_25 * yk_6;

        scalar_t sum_v_138 = scalar_t(0);
        sum_v_138 += scalar_t(0.327326834f) * wi_47 * xj_21 * yk_13;
        sum_v_138 += scalar_t(-0.422577113f) * wi_47 * xj_21 * yk_15;
        sum_v_138 += scalar_t(0.534522474f) * wi_47 * xj_22 * yk_12;
        sum_v_138 += scalar_t(-0.377964467f) * wi_47 * xj_23 * yk_11;
        sum_v_138 += scalar_t(0.422577113f) * wi_47 * xj_25 * yk_9;
        sum_v_138 += scalar_t(-0.327326834f) * wi_47 * xj_25 * yk_11;

        scalar_t sum_v_135 = scalar_t(0);
        sum_v_135 += scalar_t(0.422577113f) * wi_47 * xj_22 * yk_9;
        sum_v_135 += scalar_t(0.327326834f) * wi_47 * xj_22 * yk_11;
        sum_v_135 += scalar_t(0.597614288f) * wi_47 * xj_23 * yk_14;
        sum_v_135 += scalar_t(-0.327326834f) * wi_47 * xj_24 * yk_13;
        sum_v_135 += scalar_t(0.422577113f) * wi_47 * xj_24 * yk_15;
        sum_v_135 += scalar_t(0.267261237f) * wi_47 * xj_25 * yk_12;

        scalar_t sum_v_314 = scalar_t(0);
        sum_v_314 += scalar_t(-0.288675129f) * wi_78 * xj_21 * yk_9;
        sum_v_314 += scalar_t(0.44721356f) * wi_78 * xj_21 * yk_11;
        sum_v_314 += scalar_t(0.353553385f) * wi_78 * xj_22 * yk_10;
        sum_v_314 += scalar_t(0.387298346f) * wi_78 * xj_23 * yk_13;
        sum_v_314 += scalar_t(0.182574183f) * wi_78 * xj_24 * yk_12;
        sum_v_314 += scalar_t(0.353553385f) * wi_78 * xj_24 * yk_14;
        sum_v_314 += scalar_t(0.44721356f) * wi_78 * xj_25 * yk_13;
        sum_v_314 += scalar_t(-0.288675129f) * wi_78 * xj_25 * yk_15;

        scalar_t sum_v_310 = scalar_t(0);
        sum_v_310 += scalar_t(-0.288675129f) * wi_78 * xj_21 * yk_13;
        sum_v_310 += scalar_t(0.456435442f) * wi_78 * xj_22 * yk_14;
        sum_v_310 += scalar_t(-0.645497203f) * wi_78 * xj_23 * yk_9;
        sum_v_310 += scalar_t(0.456435442f) * wi_78 * xj_24 * yk_10;
        sum_v_310 += scalar_t(-0.288675129f) * wi_78 * xj_25 * yk_11;

        scalar_t sum_v_391 = scalar_t(0);
        sum_v_391 += scalar_t(-0.387298346f) * wi_89 * xj_21 * yk_7;
        sum_v_391 += scalar_t(0.44721356f) * wi_89 * xj_22 * yk_6;
        sum_v_391 += scalar_t(-0.387298346f) * wi_89 * xj_22 * yk_8;
        sum_v_391 += scalar_t(-0.44721356f) * wi_89 * xj_23 * yk_5;
        sum_v_391 += scalar_t(0.387298346f) * wi_89 * xj_24 * yk_4;
        sum_v_391 += scalar_t(0.387298346f) * wi_89 * xj_25 * yk_5;

        scalar_t sum_v_390 = scalar_t(0);
        sum_v_390 += scalar_t(0.316227764f) * wi_89 * xj_21 * yk_8;
        sum_v_390 += scalar_t(-0.632455528f) * wi_89 * xj_22 * yk_7;
        sum_v_390 += scalar_t(0.632455528f) * wi_89 * xj_24 * yk_5;
        sum_v_390 += scalar_t(-0.316227764f) * wi_89 * xj_25 * yk_4;

        scalar_t sum_v_393 = scalar_t(0);
        sum_v_393 += scalar_t(0.49999997f) * wi_89 * xj_21 * yk_7;
        sum_v_393 += scalar_t(-0.49999997f) * wi_89 * xj_22 * yk_8;
        sum_v_393 += scalar_t(-0.49999997f) * wi_89 * xj_24 * yk_4;
        sum_v_393 += scalar_t(0.49999997f) * wi_89 * xj_25 * yk_5;

        scalar_t sum_v_315 = scalar_t(0);
        sum_v_315 += scalar_t(0.456435442f) * wi_78 * xj_22 * yk_9;
        sum_v_315 += scalar_t(-0.353553385f) * wi_78 * xj_22 * yk_11;
        sum_v_315 += scalar_t(0.353553385f) * wi_78 * xj_24 * yk_13;
        sum_v_315 += scalar_t(0.456435442f) * wi_78 * xj_24 * yk_15;
        sum_v_315 += scalar_t(-0.577350259f) * wi_78 * xj_25 * yk_12;

        scalar_t sum_v_307 = scalar_t(0);
        sum_v_307 += scalar_t(-0.182574183f) * wi_77 * xj_21 * yk_1;
        sum_v_307 += scalar_t(0.632455528f) * wi_77 * xj_23 * yk_3;
        sum_v_307 += scalar_t(0.730296731f) * wi_77 * xj_24 * yk_2;
        sum_v_307 += scalar_t(-0.182574183f) * wi_77 * xj_25 * yk_3;

        scalar_t sum_v_306 = scalar_t(0);
        sum_v_306 += scalar_t(-0.44721356f) * wi_77 * xj_22 * yk_1;
        sum_v_306 += scalar_t(0.774596691f) * wi_77 * xj_23 * yk_2;
        sum_v_306 += scalar_t(-0.44721356f) * wi_77 * xj_24 * yk_3;

        scalar_t sum_v_130 = scalar_t(0);
        sum_v_130 += scalar_t(-0.408248276f) * wi_46 * xj_22 * yk_1;
        sum_v_130 += scalar_t(0.408248276f) * wi_46 * xj_24 * yk_3;
        sum_v_130 += scalar_t(-0.816496551f) * wi_46 * xj_25 * yk_2;

        scalar_t sum_v_133 = scalar_t(0);
        sum_v_133 += scalar_t(-0.408248276f) * wi_46 * xj_21 * yk_3;
        sum_v_133 += scalar_t(0.408248276f) * wi_46 * xj_22 * yk_2;
        sum_v_133 += scalar_t(-0.707106769f) * wi_46 * xj_23 * yk_1;
        sum_v_133 += scalar_t(0.408248276f) * wi_46 * xj_25 * yk_1;

        scalar_t sum_v_47 = scalar_t(0);
        sum_v_47 += scalar_t(0.547722578f) * wi_23 * xj_21 * yk_1;
        sum_v_47 += scalar_t(-0.316227794f) * wi_23 * xj_23 * yk_3;
        sum_v_47 += scalar_t(0.547722578f) * wi_23 * xj_24 * yk_2;
        sum_v_47 += scalar_t(0.547722578f) * wi_23 * xj_25 * yk_3;

        scalar_t sum_v_109 = scalar_t(0);
        sum_v_109 += scalar_t(-0.707106769f) * wi_41 * xj_11 * yk_1;
        sum_v_109 += scalar_t(0.707106769f) * wi_41 * xj_15 * yk_3;

        scalar_t sum_v_105 = scalar_t(0);
        sum_v_105 += scalar_t(-0.408248276f) * wi_41 * xj_11 * yk_1;
        sum_v_105 += scalar_t(0.816496551f) * wi_41 * xj_13 * yk_2;
        sum_v_105 += scalar_t(-0.408248276f) * wi_41 * xj_15 * yk_3;

        scalar_t sum_v_111 = scalar_t(0);
        sum_v_111 += scalar_t(-0.154303357f) * wi_43 * xj_11 * yk_13;
        sum_v_111 += scalar_t(-0.597614288f) * wi_43 * xj_11 * yk_15;
        sum_v_111 += scalar_t(0.487950057f) * wi_43 * xj_13 * yk_10;
        sum_v_111 += scalar_t(0.597614288f) * wi_43 * xj_15 * yk_9;
        sum_v_111 += scalar_t(-0.154303357f) * wi_43 * xj_15 * yk_11;

        scalar_t sum_v_113 = scalar_t(0);
        sum_v_113 += scalar_t(-0.377964467f) * wi_43 * xj_11 * yk_12;
        sum_v_113 += scalar_t(-0.487950057f) * wi_43 * xj_11 * yk_14;
        sum_v_113 += scalar_t(0.617213428f) * wi_43 * xj_13 * yk_11;
        sum_v_113 += scalar_t(0.487950057f) * wi_43 * xj_15 * yk_10;

        scalar_t sum_v_70 = scalar_t(0);
        sum_v_70 += scalar_t(-0.316227794f) * wi_32 * xj_11 * yk_6;
        sum_v_70 += scalar_t(-0.547722578f) * wi_32 * xj_11 * yk_8;
        sum_v_70 += scalar_t(0.547722578f) * wi_32 * xj_13 * yk_5;
        sum_v_70 += scalar_t(0.547722578f) * wi_32 * xj_15 * yk_4;

        scalar_t sum_v_72 = scalar_t(0);
        sum_v_72 += scalar_t(0.547722578f) * wi_32 * xj_11 * yk_5;
        sum_v_72 += scalar_t(0.632455587f) * wi_32 * xj_13 * yk_6;
        sum_v_72 += scalar_t(0.547722578f) * wi_32 * xj_15 * yk_7;

        scalar_t sum_v_107 = scalar_t(0);
        sum_v_107 += scalar_t(0.707106769f) * wi_41 * xj_13 * yk_3;
        sum_v_107 += scalar_t(0.707106769f) * wi_41 * xj_15 * yk_2;

        scalar_t sum_v_41 = scalar_t(0);
        sum_v_41 += scalar_t(0.707106769f) * wi_21 * xj_11 * yk_2;
        sum_v_41 += scalar_t(-0.707106769f) * wi_21 * xj_13 * yk_1;

        scalar_t sum_v_101 = scalar_t(0);
        sum_v_101 += scalar_t(0.707106769f) * wi_41 * xj_11 * yk_3;
        sum_v_101 += scalar_t(0.707106769f) * wi_41 * xj_15 * yk_1;

        scalar_t sum_v_196 = scalar_t(0);
        sum_v_196 += scalar_t(0.408248276f) * wi_60 * xj_11 * yk_5;
        sum_v_196 += scalar_t(0.816496551f) * wi_60 * xj_13 * yk_8;
        sum_v_196 += scalar_t(-0.408248276f) * wi_60 * xj_15 * yk_7;

        scalar_t sum_v_202 = scalar_t(0);
        sum_v_202 += scalar_t(0.707106769f) * wi_60 * xj_11 * yk_6;
        sum_v_202 += scalar_t(-0.408248276f) * wi_60 * xj_11 * yk_8;
        sum_v_202 += scalar_t(-0.408248276f) * wi_60 * xj_13 * yk_5;
        sum_v_202 += scalar_t(0.408248276f) * wi_60 * xj_15 * yk_4;

        scalar_t sum_v_283 = scalar_t(0);
        sum_v_283 += scalar_t(0.353553385f) * wi_75 * xj_11 * yk_10;
        sum_v_283 += scalar_t(0.866025388f) * wi_75 * xj_13 * yk_15;
        sum_v_283 += scalar_t(-0.353553385f) * wi_75 * xj_15 * yk_14;

        scalar_t sum_v_285 = scalar_t(0);
        sum_v_285 += scalar_t(-0.353553385f) * wi_75 * xj_11 * yk_9;
        sum_v_285 += scalar_t(0.456435442f) * wi_75 * xj_11 * yk_11;
        sum_v_285 += scalar_t(0.577350259f) * wi_75 * xj_13 * yk_14;
        sum_v_285 += scalar_t(-0.456435442f) * wi_75 * xj_15 * yk_13;
        sum_v_285 += scalar_t(-0.353553385f) * wi_75 * xj_15 * yk_15;

        scalar_t sum_v_287 = scalar_t(0);
        sum_v_287 += scalar_t(-0.456435442f) * wi_75 * xj_11 * yk_10;
        sum_v_287 += scalar_t(0.288675129f) * wi_75 * xj_13 * yk_13;
        sum_v_287 += scalar_t(-0.707106769f) * wi_75 * xj_15 * yk_12;
        sum_v_287 += scalar_t(-0.456435442f) * wi_75 * xj_15 * yk_14;

        scalar_t sum_v_362 = scalar_t(0);
        sum_v_362 += scalar_t(0.577350259f) * wi_86 * xj_11 * yk_7;
        sum_v_362 += scalar_t(0.577350259f) * wi_86 * xj_13 * yk_4;
        sum_v_362 += scalar_t(0.577350259f) * wi_86 * xj_15 * yk_5;

        scalar_t sum_v_368 = scalar_t(0);
        sum_v_368 += scalar_t(-0.182574183f) * wi_86 * xj_11 * yk_4;
        sum_v_368 += scalar_t(0.730296731f) * wi_86 * xj_13 * yk_7;
        sum_v_368 += scalar_t(0.632455528f) * wi_86 * xj_15 * yk_6;
        sum_v_368 += scalar_t(-0.182574183f) * wi_86 * xj_15 * yk_8;

        scalar_t sum_v_370 = scalar_t(0);
        sum_v_370 += scalar_t(-0.577350259f) * wi_86 * xj_11 * yk_5;
        sum_v_370 += scalar_t(0.577350259f) * wi_86 * xj_13 * yk_8;
        sum_v_370 += scalar_t(0.577350259f) * wi_86 * xj_15 * yk_7;

        scalar_t sum_v_372 = scalar_t(0);
        sum_v_372 += scalar_t(-0.707106769f) * wi_86 * xj_11 * yk_4;
        sum_v_372 += scalar_t(0.707106769f) * wi_86 * xj_15 * yk_8;

        scalar_t sum_v_289 = scalar_t(0);
        sum_v_289 += scalar_t(-0.707106769f) * wi_75 * xj_11 * yk_13;
        sum_v_289 += scalar_t(0.707106769f) * wi_75 * xj_15 * yk_11;

        scalar_t sum_v_284 = scalar_t(0);
        sum_v_284 += scalar_t(-0.353553385f) * wi_74 * xj_10 * yk_9;
        sum_v_284 += scalar_t(0.456435442f) * wi_74 * xj_10 * yk_11;
        sum_v_284 += scalar_t(0.577350259f) * wi_74 * xj_12 * yk_14;
        sum_v_284 += scalar_t(-0.456435442f) * wi_74 * xj_14 * yk_13;
        sum_v_284 += scalar_t(-0.353553385f) * wi_74 * xj_14 * yk_15;

        scalar_t sum_v_286 = scalar_t(0);
        sum_v_286 += scalar_t(-0.456435442f) * wi_74 * xj_10 * yk_10;
        sum_v_286 += scalar_t(0.288675129f) * wi_74 * xj_12 * yk_13;
        sum_v_286 += scalar_t(-0.707106769f) * wi_74 * xj_14 * yk_12;
        sum_v_286 += scalar_t(-0.456435442f) * wi_74 * xj_14 * yk_14;

        scalar_t sum_v_294 = scalar_t(0);
        sum_v_294 += scalar_t(0.353553385f) * wi_74 * xj_10 * yk_14;
        sum_v_294 += scalar_t(-0.866025388f) * wi_74 * xj_12 * yk_9;
        sum_v_294 += scalar_t(0.353553385f) * wi_74 * xj_14 * yk_10;

        scalar_t sum_v_367 = scalar_t(0);
        sum_v_367 += scalar_t(-0.182574183f) * wi_85 * xj_10 * yk_4;
        sum_v_367 += scalar_t(0.730296731f) * wi_85 * xj_12 * yk_7;
        sum_v_367 += scalar_t(0.632455528f) * wi_85 * xj_14 * yk_6;
        sum_v_367 += scalar_t(-0.182574183f) * wi_85 * xj_14 * yk_8;

        scalar_t sum_v_365 = scalar_t(0);
        sum_v_365 += scalar_t(-0.44721356f) * wi_85 * xj_10 * yk_5;
        sum_v_365 += scalar_t(0.774596691f) * wi_85 * xj_12 * yk_6;
        sum_v_365 += scalar_t(-0.44721356f) * wi_85 * xj_14 * yk_7;

        scalar_t sum_v_371 = scalar_t(0);
        sum_v_371 += scalar_t(-0.707106769f) * wi_85 * xj_10 * yk_4;
        sum_v_371 += scalar_t(0.707106769f) * wi_85 * xj_14 * yk_8;

        scalar_t sum_v_197 = scalar_t(0);
        sum_v_197 += scalar_t(-0.408248276f) * wi_59 * xj_10 * yk_4;
        sum_v_197 += scalar_t(0.408248276f) * wi_59 * xj_12 * yk_7;
        sum_v_197 += scalar_t(-0.707106769f) * wi_59 * xj_14 * yk_6;
        sum_v_197 += scalar_t(-0.408248276f) * wi_59 * xj_14 * yk_8;

        scalar_t sum_v_195 = scalar_t(0);
        sum_v_195 += scalar_t(0.408248276f) * wi_59 * xj_10 * yk_5;
        sum_v_195 += scalar_t(0.816496551f) * wi_59 * xj_12 * yk_8;
        sum_v_195 += scalar_t(-0.408248276f) * wi_59 * xj_14 * yk_7;

        scalar_t sum_v_199 = scalar_t(0);
        sum_v_199 += scalar_t(-0.707106769f) * wi_59 * xj_10 * yk_7;
        sum_v_199 += scalar_t(0.707106769f) * wi_59 * xj_14 * yk_5;

        scalar_t sum_v_118 = scalar_t(0);
        sum_v_118 += scalar_t(0.597614288f) * wi_42 * xj_10 * yk_9;
        sum_v_118 += scalar_t(0.154303357f) * wi_42 * xj_10 * yk_11;
        sum_v_118 += scalar_t(0.487950057f) * wi_42 * xj_12 * yk_14;
        sum_v_118 += scalar_t(-0.154303357f) * wi_42 * xj_14 * yk_13;
        sum_v_118 += scalar_t(0.597614288f) * wi_42 * xj_14 * yk_15;

        scalar_t sum_v_116 = scalar_t(0);
        sum_v_116 += scalar_t(0.487950057f) * wi_42 * xj_10 * yk_10;
        sum_v_116 += scalar_t(0.617213428f) * wi_42 * xj_12 * yk_13;
        sum_v_116 += scalar_t(-0.377964467f) * wi_42 * xj_14 * yk_12;
        sum_v_116 += scalar_t(0.487950057f) * wi_42 * xj_14 * yk_14;

        scalar_t sum_v_114 = scalar_t(0);
        sum_v_114 += scalar_t(0.534522474f) * wi_42 * xj_10 * yk_11;
        sum_v_114 += scalar_t(0.654653668f) * wi_42 * xj_12 * yk_12;
        sum_v_114 += scalar_t(0.534522474f) * wi_42 * xj_14 * yk_13;

        scalar_t sum_v_69 = scalar_t(0);
        sum_v_69 += scalar_t(-0.316227794f) * wi_31 * xj_10 * yk_6;
        sum_v_69 += scalar_t(-0.547722578f) * wi_31 * xj_10 * yk_8;
        sum_v_69 += scalar_t(0.547722578f) * wi_31 * xj_12 * yk_5;
        sum_v_69 += scalar_t(0.547722578f) * wi_31 * xj_14 * yk_4;

        scalar_t sum_v_71 = scalar_t(0);
        sum_v_71 += scalar_t(0.547722578f) * wi_31 * xj_10 * yk_5;
        sum_v_71 += scalar_t(0.632455587f) * wi_31 * xj_12 * yk_6;
        sum_v_71 += scalar_t(0.547722578f) * wi_31 * xj_14 * yk_7;

        scalar_t sum_v_106 = scalar_t(0);
        sum_v_106 += scalar_t(0.707106769f) * wi_40 * xj_12 * yk_3;
        sum_v_106 += scalar_t(0.707106769f) * wi_40 * xj_14 * yk_2;

        scalar_t sum_v_100 = scalar_t(0);
        sum_v_100 += scalar_t(0.707106769f) * wi_40 * xj_10 * yk_3;
        sum_v_100 += scalar_t(0.707106769f) * wi_40 * xj_14 * yk_1;

        scalar_t sum_v_40 = scalar_t(0);
        sum_v_40 += scalar_t(0.707106769f) * wi_20 * xj_10 * yk_2;
        sum_v_40 += scalar_t(-0.707106769f) * wi_20 * xj_12 * yk_1;

        scalar_t sum_v_64 = scalar_t(0);
        sum_v_64 += wi_30 * xj_11 * yk_0;

        scalar_t sum_v_66 = scalar_t(0);
        sum_v_66 += wi_30 * xj_13 * yk_0;

        scalar_t sum_v_68 = scalar_t(0);
        sum_v_68 += wi_30 * xj_15 * yk_0;

        scalar_t sum_v_62 = scalar_t(0);
        sum_v_62 += scalar_t(0.707106769f) * wi_28 * xj_5 * yk_2;
        sum_v_62 += scalar_t(-0.707106769f) * wi_28 * xj_7 * yk_1;

        scalar_t sum_v_9 = scalar_t(0);
        sum_v_9 += scalar_t(0.577350259f) * wi_9 * xj_5 * yk_1;
        sum_v_9 += scalar_t(0.577350259f) * wi_9 * xj_7 * yk_2;
        sum_v_9 += scalar_t(0.577350259f) * wi_9 * xj_9 * yk_3;

        scalar_t sum_v_35 = scalar_t(0);
        sum_v_35 += scalar_t(0.547722578f) * wi_19 * xj_5 * yk_4;
        sum_v_35 += scalar_t(0.547722578f) * wi_19 * xj_7 * yk_7;
        sum_v_35 += scalar_t(-0.316227794f) * wi_19 * xj_9 * yk_6;
        sum_v_35 += scalar_t(0.547722578f) * wi_19 * xj_9 * yk_8;

        scalar_t sum_v_91 = scalar_t(0);
        sum_v_91 += scalar_t(0.408248276f) * wi_39 * xj_5 * yk_5;
        sum_v_91 += scalar_t(0.816496551f) * wi_39 * xj_7 * yk_8;
        sum_v_91 += scalar_t(-0.408248276f) * wi_39 * xj_9 * yk_7;

        scalar_t sum_v_97 = scalar_t(0);
        sum_v_97 += scalar_t(0.707106769f) * wi_39 * xj_5 * yk_6;
        sum_v_97 += scalar_t(-0.408248276f) * wi_39 * xj_5 * yk_8;
        sum_v_97 += scalar_t(-0.408248276f) * wi_39 * xj_7 * yk_5;
        sum_v_97 += scalar_t(0.408248276f) * wi_39 * xj_9 * yk_4;

        scalar_t sum_v_60 = scalar_t(0);
        sum_v_60 += scalar_t(-0.707106769f) * wi_28 * xj_5 * yk_3;
        sum_v_60 += scalar_t(0.707106769f) * wi_28 * xj_9 * yk_1;

        scalar_t sum_v_180 = scalar_t(0);
        sum_v_180 += scalar_t(-0.408248276f) * wi_56 * xj_5 * yk_1;
        sum_v_180 += scalar_t(0.816496551f) * wi_56 * xj_7 * yk_2;
        sum_v_180 += scalar_t(-0.408248276f) * wi_56 * xj_9 * yk_3;

        scalar_t sum_v_186 = scalar_t(0);
        sum_v_186 += scalar_t(-0.154303357f) * wi_58 * xj_5 * yk_13;
        sum_v_186 += scalar_t(-0.597614288f) * wi_58 * xj_5 * yk_15;
        sum_v_186 += scalar_t(0.487950057f) * wi_58 * xj_7 * yk_10;
        sum_v_186 += scalar_t(0.597614288f) * wi_58 * xj_9 * yk_9;
        sum_v_186 += scalar_t(-0.154303357f) * wi_58 * xj_9 * yk_11;

        scalar_t sum_v_188 = scalar_t(0);
        sum_v_188 += scalar_t(-0.377964467f) * wi_58 * xj_5 * yk_12;
        sum_v_188 += scalar_t(-0.487950057f) * wi_58 * xj_5 * yk_14;
        sum_v_188 += scalar_t(0.617213428f) * wi_58 * xj_7 * yk_11;
        sum_v_188 += scalar_t(0.487950057f) * wi_58 * xj_9 * yk_10;

        scalar_t sum_v_271 = scalar_t(0);
        sum_v_271 += scalar_t(0.577350259f) * wi_73 * xj_5 * yk_7;
        sum_v_271 += scalar_t(0.577350259f) * wi_73 * xj_7 * yk_4;
        sum_v_271 += scalar_t(0.577350259f) * wi_73 * xj_9 * yk_5;

        scalar_t sum_v_277 = scalar_t(0);
        sum_v_277 += scalar_t(-0.182574183f) * wi_73 * xj_5 * yk_4;
        sum_v_277 += scalar_t(0.730296731f) * wi_73 * xj_7 * yk_7;
        sum_v_277 += scalar_t(0.632455528f) * wi_73 * xj_9 * yk_6;
        sum_v_277 += scalar_t(-0.182574183f) * wi_73 * xj_9 * yk_8;

        scalar_t sum_v_279 = scalar_t(0);
        sum_v_279 += scalar_t(-0.577350259f) * wi_73 * xj_5 * yk_5;
        sum_v_279 += scalar_t(0.577350259f) * wi_73 * xj_7 * yk_8;
        sum_v_279 += scalar_t(0.577350259f) * wi_73 * xj_9 * yk_7;

        scalar_t sum_v_281 = scalar_t(0);
        sum_v_281 += scalar_t(-0.707106769f) * wi_73 * xj_5 * yk_4;
        sum_v_281 += scalar_t(0.707106769f) * wi_73 * xj_9 * yk_8;

        scalar_t sum_v_348 = scalar_t(0);
        sum_v_348 += scalar_t(-0.353553385f) * wi_84 * xj_5 * yk_9;
        sum_v_348 += scalar_t(0.456435442f) * wi_84 * xj_5 * yk_11;
        sum_v_348 += scalar_t(0.577350259f) * wi_84 * xj_7 * yk_14;
        sum_v_348 += scalar_t(-0.456435442f) * wi_84 * xj_9 * yk_13;
        sum_v_348 += scalar_t(-0.353553385f) * wi_84 * xj_9 * yk_15;

        scalar_t sum_v_350 = scalar_t(0);
        sum_v_350 += scalar_t(-0.456435442f) * wi_84 * xj_5 * yk_10;
        sum_v_350 += scalar_t(0.288675129f) * wi_84 * xj_7 * yk_13;
        sum_v_350 += scalar_t(-0.707106769f) * wi_84 * xj_9 * yk_12;
        sum_v_350 += scalar_t(-0.456435442f) * wi_84 * xj_9 * yk_14;

        scalar_t sum_v_358 = scalar_t(0);
        sum_v_358 += scalar_t(0.353553385f) * wi_84 * xj_5 * yk_14;
        sum_v_358 += scalar_t(-0.866025388f) * wi_84 * xj_7 * yk_9;
        sum_v_358 += scalar_t(0.353553385f) * wi_84 * xj_9 * yk_10;

        scalar_t sum_v_352 = scalar_t(0);
        sum_v_352 += scalar_t(-0.707106769f) * wi_84 * xj_5 * yk_13;
        sum_v_352 += scalar_t(0.707106769f) * wi_84 * xj_9 * yk_11;

        scalar_t sum_v_347 = scalar_t(0);
        sum_v_347 += scalar_t(-0.353553385f) * wi_83 * xj_4 * yk_9;
        sum_v_347 += scalar_t(0.456435442f) * wi_83 * xj_4 * yk_11;
        sum_v_347 += scalar_t(0.577350259f) * wi_83 * xj_6 * yk_14;
        sum_v_347 += scalar_t(-0.456435442f) * wi_83 * xj_8 * yk_13;
        sum_v_347 += scalar_t(-0.353553385f) * wi_83 * xj_8 * yk_15;

        scalar_t sum_v_349 = scalar_t(0);
        sum_v_349 += scalar_t(-0.456435442f) * wi_83 * xj_4 * yk_10;
        sum_v_349 += scalar_t(0.288675129f) * wi_83 * xj_6 * yk_13;
        sum_v_349 += scalar_t(-0.707106769f) * wi_83 * xj_8 * yk_12;
        sum_v_349 += scalar_t(-0.456435442f) * wi_83 * xj_8 * yk_14;

        scalar_t sum_v_357 = scalar_t(0);
        sum_v_357 += scalar_t(0.353553385f) * wi_83 * xj_4 * yk_14;
        sum_v_357 += scalar_t(-0.866025388f) * wi_83 * xj_6 * yk_9;
        sum_v_357 += scalar_t(0.353553385f) * wi_83 * xj_8 * yk_10;

        scalar_t sum_v_272 = scalar_t(0);
        sum_v_272 += scalar_t(0.632455528f) * wi_72 * xj_4 * yk_6;
        sum_v_272 += scalar_t(0.182574183f) * wi_72 * xj_4 * yk_8;
        sum_v_272 += scalar_t(0.730296731f) * wi_72 * xj_6 * yk_5;
        sum_v_272 += scalar_t(-0.182574183f) * wi_72 * xj_8 * yk_4;

        scalar_t sum_v_270 = scalar_t(0);
        sum_v_270 += scalar_t(0.577350259f) * wi_72 * xj_4 * yk_7;
        sum_v_270 += scalar_t(0.577350259f) * wi_72 * xj_6 * yk_4;
        sum_v_270 += scalar_t(0.577350259f) * wi_72 * xj_8 * yk_5;

        scalar_t sum_v_193 = scalar_t(0);
        sum_v_193 += scalar_t(0.597614288f) * wi_57 * xj_4 * yk_9;
        sum_v_193 += scalar_t(0.154303357f) * wi_57 * xj_4 * yk_11;
        sum_v_193 += scalar_t(0.487950057f) * wi_57 * xj_6 * yk_14;
        sum_v_193 += scalar_t(-0.154303357f) * wi_57 * xj_8 * yk_13;
        sum_v_193 += scalar_t(0.597614288f) * wi_57 * xj_8 * yk_15;

        scalar_t sum_v_191 = scalar_t(0);
        sum_v_191 += scalar_t(0.487950057f) * wi_57 * xj_4 * yk_10;
        sum_v_191 += scalar_t(0.617213428f) * wi_57 * xj_6 * yk_13;
        sum_v_191 += scalar_t(-0.377964467f) * wi_57 * xj_8 * yk_12;
        sum_v_191 += scalar_t(0.487950057f) * wi_57 * xj_8 * yk_14;

        scalar_t sum_v_189 = scalar_t(0);
        sum_v_189 += scalar_t(0.534522474f) * wi_57 * xj_4 * yk_11;
        sum_v_189 += scalar_t(0.654653668f) * wi_57 * xj_6 * yk_12;
        sum_v_189 += scalar_t(0.534522474f) * wi_57 * xj_8 * yk_13;

        scalar_t sum_v_177 = scalar_t(0);
        sum_v_177 += scalar_t(0.707106769f) * wi_55 * xj_4 * yk_2;
        sum_v_177 += scalar_t(0.707106769f) * wi_55 * xj_6 * yk_1;

        scalar_t sum_v_183 = scalar_t(0);
        sum_v_183 += scalar_t(-0.707106769f) * wi_55 * xj_4 * yk_1;
        sum_v_183 += scalar_t(0.707106769f) * wi_55 * xj_8 * yk_3;

        scalar_t sum_v_98 = scalar_t(0);
        sum_v_98 += scalar_t(0.408248276f) * wi_38 * xj_4 * yk_7;
        sum_v_98 += scalar_t(-0.816496551f) * wi_38 * xj_6 * yk_4;
        sum_v_98 += scalar_t(0.408248276f) * wi_38 * xj_8 * yk_5;

        scalar_t sum_v_92 = scalar_t(0);
        sum_v_92 += scalar_t(-0.408248276f) * wi_38 * xj_4 * yk_4;
        sum_v_92 += scalar_t(0.408248276f) * wi_38 * xj_6 * yk_7;
        sum_v_92 += scalar_t(-0.707106769f) * wi_38 * xj_8 * yk_6;
        sum_v_92 += scalar_t(-0.408248276f) * wi_38 * xj_8 * yk_8;

        scalar_t sum_v_30 = scalar_t(0);
        sum_v_30 += scalar_t(-0.316227794f) * wi_18 * xj_4 * yk_6;
        sum_v_30 += scalar_t(-0.547722578f) * wi_18 * xj_4 * yk_8;
        sum_v_30 += scalar_t(0.547722578f) * wi_18 * xj_6 * yk_5;
        sum_v_30 += scalar_t(0.547722578f) * wi_18 * xj_8 * yk_4;

        scalar_t sum_v_32 = scalar_t(0);
        sum_v_32 += scalar_t(0.547722578f) * wi_18 * xj_4 * yk_5;
        sum_v_32 += scalar_t(0.632455587f) * wi_18 * xj_6 * yk_6;
        sum_v_32 += scalar_t(0.547722578f) * wi_18 * xj_8 * yk_7;

        scalar_t sum_v_57 = scalar_t(0);
        sum_v_57 += scalar_t(0.707106769f) * wi_27 * xj_6 * yk_3;
        sum_v_57 += scalar_t(-0.707106769f) * wi_27 * xj_8 * yk_2;

        scalar_t sum_v_94 = scalar_t(0);
        sum_v_94 += scalar_t(-0.707106769f) * wi_38 * xj_4 * yk_7;
        sum_v_94 += scalar_t(0.707106769f) * wi_38 * xj_8 * yk_5;

        scalar_t sum_v_25 = scalar_t(0);
        sum_v_25 += wi_17 * xj_5 * yk_0;

        scalar_t sum_v_27 = scalar_t(0);
        sum_v_27 += wi_17 * xj_7 * yk_0;

        scalar_t sum_v_29 = scalar_t(0);
        sum_v_29 += wi_17 * xj_9 * yk_0;

        scalar_t sum_v_15 = scalar_t(0);
        sum_v_15 += wi_15 * xj_3 * yk_1;

        scalar_t sum_v_23 = scalar_t(0);
        sum_v_23 += wi_15 * xj_3 * yk_3;

        scalar_t sum_v_18 = scalar_t(0);
        sum_v_18 += wi_14 * xj_2 * yk_2;

        scalar_t sum_v_6 = scalar_t(0);
        sum_v_6 += wi_6 * xj_2 * yk_0;

        scalar_t sum_v_13 = scalar_t(0);
        sum_v_13 += wi_13 * xj_1 * yk_1;

        scalar_t sum_v_21 = scalar_t(0);
        sum_v_21 += wi_13 * xj_1 * yk_3;

        scalar_t sum_v_16 = scalar_t(0);
        sum_v_16 += wi_12 * xj_0 * yk_2;

        scalar_t sum_v_4 = scalar_t(0);
        sum_v_4 += wi_4 * xj_0 * yk_0;

        scalar_t sum_v_2 = scalar_t(0);
        sum_v_2 += scalar_t(0.44721359f) * wi_2 * xj_16 * yk_4;
        sum_v_2 += scalar_t(0.44721359f) * wi_2 * xj_17 * yk_5;
        sum_v_2 += scalar_t(0.44721359f) * wi_2 * xj_18 * yk_6;
        sum_v_2 += scalar_t(0.44721359f) * wi_2 * xj_19 * yk_7;
        sum_v_2 += scalar_t(0.44721359f) * wi_2 * xj_20 * yk_8;

        scalar_t sum_v_44 = scalar_t(0);
        sum_v_44 += scalar_t(0.316227764f) * wi_22 * xj_16 * yk_7;
        sum_v_44 += scalar_t(0.547722518f) * wi_22 * xj_17 * yk_6;
        sum_v_44 += scalar_t(0.316227764f) * wi_22 * xj_17 * yk_8;
        sum_v_44 += scalar_t(-0.547722518f) * wi_22 * xj_18 * yk_5;
        sum_v_44 += scalar_t(-0.316227764f) * wi_22 * xj_19 * yk_4;
        sum_v_44 += scalar_t(-0.316227764f) * wi_22 * xj_20 * yk_5;

        scalar_t sum_v_128 = scalar_t(0);
        sum_v_128 += scalar_t(0.462910056f) * wi_45 * xj_16 * yk_5;
        sum_v_128 += scalar_t(0.462910056f) * wi_45 * xj_17 * yk_4;
        sum_v_128 += scalar_t(0.267261237f) * wi_45 * xj_18 * yk_7;
        sum_v_128 += scalar_t(0.267261237f) * wi_45 * xj_19 * yk_6;
        sum_v_128 += scalar_t(0.462910056f) * wi_45 * xj_19 * yk_8;
        sum_v_128 += scalar_t(0.462910056f) * wi_45 * xj_20 * yk_7;

        scalar_t sum_v_125 = scalar_t(0);
        sum_v_125 += scalar_t(-0.534522474f) * wi_45 * xj_16 * yk_6;
        sum_v_125 += scalar_t(0.462910056f) * wi_45 * xj_17 * yk_7;
        sum_v_125 += scalar_t(-0.534522474f) * wi_45 * xj_18 * yk_4;
        sum_v_125 += scalar_t(0.462910056f) * wi_45 * xj_19 * yk_5;

        scalar_t sum_v_78 = scalar_t(0);
        sum_v_78 += scalar_t(-0.119522862f) * wi_34 * xj_16 * yk_13;
        sum_v_78 += scalar_t(-0.462910056f) * wi_34 * xj_16 * yk_15;
        sum_v_78 += scalar_t(-0.292769998f) * wi_34 * xj_17 * yk_12;
        sum_v_78 += scalar_t(-0.377964467f) * wi_34 * xj_17 * yk_14;
        sum_v_78 += scalar_t(0.414039344f) * wi_34 * xj_18 * yk_11;
        sum_v_78 += scalar_t(0.377964467f) * wi_34 * xj_19 * yk_10;
        sum_v_78 += scalar_t(0.462910056f) * wi_34 * xj_20 * yk_9;
        sum_v_78 += scalar_t(0.119522862f) * wi_34 * xj_20 * yk_11;

        scalar_t sum_v_79 = scalar_t(0);
        sum_v_79 += scalar_t(0.377964467f) * wi_34 * xj_16 * yk_10;
        sum_v_79 += scalar_t(0.478091449f) * wi_34 * xj_17 * yk_11;
        sum_v_79 += scalar_t(0.507092535f) * wi_34 * xj_18 * yk_12;
        sum_v_79 += scalar_t(0.478091449f) * wi_34 * xj_19 * yk_13;
        sum_v_79 += scalar_t(0.377964467f) * wi_34 * xj_20 * yk_14;

        scalar_t sum_v_75 = scalar_t(0);
        sum_v_75 += scalar_t(0.547722578f) * wi_33 * xj_16 * yk_3;
        sum_v_75 += scalar_t(0.547722578f) * wi_33 * xj_17 * yk_2;
        sum_v_75 += scalar_t(-0.316227794f) * wi_33 * xj_18 * yk_1;
        sum_v_75 += scalar_t(-0.547722578f) * wi_33 * xj_20 * yk_1;

        scalar_t sum_v_214 = scalar_t(0);
        sum_v_214 += scalar_t(-0.267261237f) * wi_62 * xj_16 * yk_12;
        sum_v_214 += scalar_t(0.327326834f) * wi_62 * xj_17 * yk_13;
        sum_v_214 += scalar_t(0.422577113f) * wi_62 * xj_17 * yk_15;
        sum_v_214 += scalar_t(-0.597614288f) * wi_62 * xj_18 * yk_10;
        sum_v_214 += scalar_t(-0.422577113f) * wi_62 * xj_19 * yk_9;
        sum_v_214 += scalar_t(0.327326834f) * wi_62 * xj_19 * yk_11;

        scalar_t sum_v_210 = scalar_t(0);
        sum_v_210 += scalar_t(0.422577113f) * wi_62 * xj_17 * yk_9;
        sum_v_210 += scalar_t(0.327326834f) * wi_62 * xj_17 * yk_11;
        sum_v_210 += scalar_t(0.597614288f) * wi_62 * xj_18 * yk_14;
        sum_v_210 += scalar_t(-0.327326834f) * wi_62 * xj_19 * yk_13;
        sum_v_210 += scalar_t(0.422577113f) * wi_62 * xj_19 * yk_15;
        sum_v_210 += scalar_t(0.267261237f) * wi_62 * xj_20 * yk_12;

        scalar_t sum_v_212 = scalar_t(0);
        sum_v_212 += scalar_t(-0.597614288f) * wi_62 * xj_16 * yk_14;
        sum_v_212 += scalar_t(-0.377964467f) * wi_62 * xj_17 * yk_13;
        sum_v_212 += scalar_t(0.377964467f) * wi_62 * xj_19 * yk_11;
        sum_v_212 += scalar_t(0.597614288f) * wi_62 * xj_20 * yk_10;

        scalar_t sum_v_209 = scalar_t(0);
        sum_v_209 += scalar_t(0.816496551f) * wi_61 * xj_16 * yk_2;
        sum_v_209 += scalar_t(-0.408248276f) * wi_61 * xj_17 * yk_3;
        sum_v_209 += scalar_t(-0.408248276f) * wi_61 * xj_19 * yk_1;

        scalar_t sum_v_300 = scalar_t(0);
        sum_v_300 += scalar_t(-0.387298346f) * wi_76 * xj_16 * yk_7;
        sum_v_300 += scalar_t(0.44721356f) * wi_76 * xj_17 * yk_6;
        sum_v_300 += scalar_t(-0.387298346f) * wi_76 * xj_17 * yk_8;
        sum_v_300 += scalar_t(-0.44721356f) * wi_76 * xj_18 * yk_5;
        sum_v_300 += scalar_t(0.387298346f) * wi_76 * xj_19 * yk_4;
        sum_v_300 += scalar_t(0.387298346f) * wi_76 * xj_20 * yk_5;

        scalar_t sum_v_299 = scalar_t(0);
        sum_v_299 += scalar_t(0.316227764f) * wi_76 * xj_16 * yk_8;
        sum_v_299 += scalar_t(-0.632455528f) * wi_76 * xj_17 * yk_7;
        sum_v_299 += scalar_t(0.632455528f) * wi_76 * xj_19 * yk_5;
        sum_v_299 += scalar_t(-0.316227764f) * wi_76 * xj_20 * yk_4;

        scalar_t sum_v_302 = scalar_t(0);
        sum_v_302 += scalar_t(0.49999997f) * wi_76 * xj_16 * yk_7;
        sum_v_302 += scalar_t(-0.49999997f) * wi_76 * xj_17 * yk_8;
        sum_v_302 += scalar_t(-0.49999997f) * wi_76 * xj_19 * yk_4;
        sum_v_302 += scalar_t(0.49999997f) * wi_76 * xj_20 * yk_5;

        scalar_t sum_v_384 = scalar_t(0);
        sum_v_384 += scalar_t(-0.288675129f) * wi_88 * xj_16 * yk_9;
        sum_v_384 += scalar_t(0.44721356f) * wi_88 * xj_16 * yk_11;
        sum_v_384 += scalar_t(0.353553385f) * wi_88 * xj_17 * yk_10;
        sum_v_384 += scalar_t(0.387298346f) * wi_88 * xj_18 * yk_13;
        sum_v_384 += scalar_t(0.182574183f) * wi_88 * xj_19 * yk_12;
        sum_v_384 += scalar_t(0.353553385f) * wi_88 * xj_19 * yk_14;
        sum_v_384 += scalar_t(0.44721356f) * wi_88 * xj_20 * yk_13;
        sum_v_384 += scalar_t(-0.288675129f) * wi_88 * xj_20 * yk_15;

        scalar_t sum_v_383 = scalar_t(0);
        sum_v_383 += scalar_t(-0.577350259f) * wi_88 * xj_16 * yk_10;
        sum_v_383 += scalar_t(0.182574183f) * wi_88 * xj_17 * yk_11;
        sum_v_383 += scalar_t(0.516397774f) * wi_88 * xj_18 * yk_12;
        sum_v_383 += scalar_t(0.182574183f) * wi_88 * xj_19 * yk_13;
        sum_v_383 += scalar_t(-0.577350259f) * wi_88 * xj_20 * yk_14;

        scalar_t sum_v_375 = scalar_t(0);
        sum_v_375 += scalar_t(-0.182574183f) * wi_87 * xj_16 * yk_3;
        sum_v_375 += scalar_t(0.730296731f) * wi_87 * xj_17 * yk_2;
        sum_v_375 += scalar_t(0.632455528f) * wi_87 * xj_18 * yk_1;
        sum_v_375 += scalar_t(0.182574183f) * wi_87 * xj_20 * yk_1;

        scalar_t sum_v_381 = scalar_t(0);
        sum_v_381 += scalar_t(-0.577350259f) * wi_88 * xj_16 * yk_12;
        sum_v_381 += scalar_t(0.353553385f) * wi_88 * xj_17 * yk_13;
        sum_v_381 += scalar_t(-0.456435442f) * wi_88 * xj_17 * yk_15;
        sum_v_381 += scalar_t(0.456435442f) * wi_88 * xj_19 * yk_9;
        sum_v_381 += scalar_t(0.353553385f) * wi_88 * xj_19 * yk_11;

        scalar_t sum_v_378 = scalar_t(0);
        sum_v_378 += scalar_t(-0.577350259f) * wi_87 * xj_17 * yk_1;
        sum_v_378 += scalar_t(0.577350259f) * wi_87 * xj_19 * yk_3;
        sum_v_378 += scalar_t(0.577350259f) * wi_87 * xj_20 * yk_2;

        scalar_t sum_v_205 = scalar_t(0);
        sum_v_205 += scalar_t(-0.408248276f) * wi_61 * xj_17 * yk_1;
        sum_v_205 += scalar_t(0.408248276f) * wi_61 * xj_19 * yk_3;
        sum_v_205 += scalar_t(-0.816496551f) * wi_61 * xj_20 * yk_2;

        scalar_t sum_v_207 = scalar_t(0);
        sum_v_207 += scalar_t(-0.707106769f) * wi_61 * xj_17 * yk_3;
        sum_v_207 += scalar_t(0.707106769f) * wi_61 * xj_19 * yk_1;

        scalar_t sum_v_53 = scalar_t(0);
        sum_v_53 += scalar_t(0.462910056f) * wi_25 * xj_26 * yk_4;
        sum_v_53 += scalar_t(0.377964467f) * wi_25 * xj_27 * yk_5;
        sum_v_53 += scalar_t(-0.119522862f) * wi_25 * xj_28 * yk_4;
        sum_v_53 += scalar_t(-0.292769998f) * wi_25 * xj_29 * yk_7;
        sum_v_53 += scalar_t(0.414039344f) * wi_25 * xj_30 * yk_6;
        sum_v_53 += scalar_t(-0.119522862f) * wi_25 * xj_30 * yk_8;
        sum_v_53 += scalar_t(0.377964467f) * wi_25 * xj_31 * yk_7;
        sum_v_53 += scalar_t(0.462910056f) * wi_25 * xj_32 * yk_8;

        scalar_t sum_v_86 = scalar_t(0);
        sum_v_86 += scalar_t(0.231455028f) * wi_36 * xj_26 * yk_14;
        sum_v_86 += scalar_t(0.298807144f) * wi_36 * xj_27 * yk_13;
        sum_v_86 += scalar_t(0.231455028f) * wi_36 * xj_27 * yk_15;
        sum_v_86 += scalar_t(0.462910056f) * wi_36 * xj_28 * yk_12;
        sum_v_86 += scalar_t(0.298807144f) * wi_36 * xj_28 * yk_14;
        sum_v_86 += scalar_t(-0.462910056f) * wi_36 * xj_29 * yk_11;
        sum_v_86 += scalar_t(-0.298807144f) * wi_36 * xj_30 * yk_10;
        sum_v_86 += scalar_t(-0.231455028f) * wi_36 * xj_31 * yk_9;
        sum_v_86 += scalar_t(-0.298807144f) * wi_36 * xj_31 * yk_11;
        sum_v_86 += scalar_t(-0.231455028f) * wi_36 * xj_32 * yk_10;

        scalar_t sum_v_11 = scalar_t(0);
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_26 * yk_9;
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_27 * yk_10;
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_28 * yk_11;
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_29 * yk_12;
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_30 * yk_13;
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_31 * yk_14;
        sum_v_11 += scalar_t(0.377964467f) * wi_11 * xj_32 * yk_15;

        scalar_t sum_v_233 = scalar_t(0);
        sum_v_233 += scalar_t(0.38575837f) * wi_66 * xj_26 * yk_10;
        sum_v_233 += scalar_t(0.38575837f) * wi_66 * xj_27 * yk_9;
        sum_v_233 += scalar_t(0.298807144f) * wi_66 * xj_27 * yk_11;
        sum_v_233 += scalar_t(0.298807144f) * wi_66 * xj_28 * yk_10;
        sum_v_233 += scalar_t(0.154303357f) * wi_66 * xj_29 * yk_13;
        sum_v_233 += scalar_t(0.154303357f) * wi_66 * xj_30 * yk_12;
        sum_v_233 += scalar_t(0.298807144f) * wi_66 * xj_30 * yk_14;
        sum_v_233 += scalar_t(0.298807144f) * wi_66 * xj_31 * yk_13;
        sum_v_233 += scalar_t(0.38575837f) * wi_66 * xj_31 * yk_15;
        sum_v_233 += scalar_t(0.38575837f) * wi_66 * xj_32 * yk_14;

        scalar_t sum_v_232 = scalar_t(0);
        sum_v_232 += scalar_t(-0.545544744f) * wi_66 * xj_26 * yk_9;
        sum_v_232 += scalar_t(0.327326834f) * wi_66 * xj_28 * yk_11;
        sum_v_232 += scalar_t(0.436435789f) * wi_66 * xj_29 * yk_12;
        sum_v_232 += scalar_t(0.327326834f) * wi_66 * xj_30 * yk_13;
        sum_v_232 += scalar_t(-0.545544744f) * wi_66 * xj_32 * yk_15;

        scalar_t sum_v_230 = scalar_t(0);
        sum_v_230 += scalar_t(-0.243975028f) * wi_66 * xj_26 * yk_13;
        sum_v_230 += scalar_t(-0.487950057f) * wi_66 * xj_27 * yk_12;
        sum_v_230 += scalar_t(0.377964467f) * wi_66 * xj_28 * yk_13;
        sum_v_230 += scalar_t(0.243975028f) * wi_66 * xj_28 * yk_15;
        sum_v_230 += scalar_t(-0.487950057f) * wi_66 * xj_29 * yk_10;
        sum_v_230 += scalar_t(-0.243975028f) * wi_66 * xj_30 * yk_9;
        sum_v_230 += scalar_t(0.377964467f) * wi_66 * xj_30 * yk_11;
        sum_v_230 += scalar_t(0.243975028f) * wi_66 * xj_32 * yk_11;

        scalar_t sum_v_406 = scalar_t(0);
        sum_v_406 += scalar_t(0.408248305f) * wi_91 * xj_26 * yk_13;
        sum_v_406 += scalar_t(0.408248305f) * wi_91 * xj_27 * yk_12;
        sum_v_406 += scalar_t(0.408248305f) * wi_91 * xj_28 * yk_15;
        sum_v_406 += scalar_t(-0.408248305f) * wi_91 * xj_29 * yk_10;
        sum_v_406 += scalar_t(-0.408248305f) * wi_91 * xj_30 * yk_9;
        sum_v_406 += scalar_t(-0.408248305f) * wi_91 * xj_32 * yk_11;

        scalar_t sum_v_404 = scalar_t(0);
        sum_v_404 += scalar_t(0.408248305f) * wi_91 * xj_26 * yk_15;
        sum_v_404 += scalar_t(-0.408248305f) * wi_91 * xj_27 * yk_14;
        sum_v_404 += scalar_t(-0.408248305f) * wi_91 * xj_28 * yk_13;
        sum_v_404 += scalar_t(0.408248305f) * wi_91 * xj_30 * yk_11;
        sum_v_404 += scalar_t(0.408248305f) * wi_91 * xj_31 * yk_10;
        sum_v_404 += scalar_t(-0.408248305f) * wi_91 * xj_32 * yk_9;

        scalar_t sum_v_402 = scalar_t(0);
        sum_v_402 += scalar_t(-0.408248305f) * wi_91 * xj_26 * yk_11;
        sum_v_402 += scalar_t(0.408248305f) * wi_91 * xj_28 * yk_9;
        sum_v_402 += scalar_t(0.408248305f) * wi_91 * xj_29 * yk_14;
        sum_v_402 += scalar_t(0.408248305f) * wi_91 * xj_30 * yk_15;
        sum_v_402 += scalar_t(-0.408248305f) * wi_91 * xj_31 * yk_12;
        sum_v_402 += scalar_t(-0.408248305f) * wi_91 * xj_32 * yk_13;

        scalar_t sum_v_407 = scalar_t(0);
        sum_v_407 += scalar_t(-0.408248305f) * wi_91 * xj_26 * yk_12;
        sum_v_407 += scalar_t(0.408248305f) * wi_91 * xj_27 * yk_13;
        sum_v_407 += scalar_t(-0.408248305f) * wi_91 * xj_28 * yk_14;
        sum_v_407 += scalar_t(0.408248305f) * wi_91 * xj_29 * yk_9;
        sum_v_407 += scalar_t(-0.408248305f) * wi_91 * xj_30 * yk_10;
        sum_v_407 += scalar_t(0.408248305f) * wi_91 * xj_31 * yk_11;

        scalar_t sum_v_326 = scalar_t(0);
        sum_v_326 += scalar_t(-0.288675129f) * wi_80 * xj_26 * yk_8;
        sum_v_326 += scalar_t(0.353553385f) * wi_80 * xj_27 * yk_7;
        sum_v_326 += scalar_t(0.387298346f) * wi_80 * xj_28 * yk_6;
        sum_v_326 += scalar_t(-0.44721356f) * wi_80 * xj_28 * yk_8;
        sum_v_326 += scalar_t(0.182574183f) * wi_80 * xj_29 * yk_5;
        sum_v_326 += scalar_t(0.44721356f) * wi_80 * xj_30 * yk_4;
        sum_v_326 += scalar_t(-0.353553385f) * wi_80 * xj_31 * yk_5;
        sum_v_326 += scalar_t(0.288675129f) * wi_80 * xj_32 * yk_4;

        scalar_t sum_v_324 = scalar_t(0);
        sum_v_324 += scalar_t(-0.645497203f) * wi_80 * xj_26 * yk_6;
        sum_v_324 += scalar_t(0.456435442f) * wi_80 * xj_27 * yk_7;
        sum_v_324 += scalar_t(-0.288675129f) * wi_80 * xj_28 * yk_8;
        sum_v_324 += scalar_t(-0.288675129f) * wi_80 * xj_30 * yk_4;
        sum_v_324 += scalar_t(0.456435442f) * wi_80 * xj_31 * yk_5;

        scalar_t sum_v_329 = scalar_t(0);
        sum_v_329 += scalar_t(0.456435442f) * wi_80 * xj_26 * yk_5;
        sum_v_329 += scalar_t(-0.353553385f) * wi_80 * xj_28 * yk_5;
        sum_v_329 += scalar_t(-0.577350259f) * wi_80 * xj_29 * yk_8;
        sum_v_329 += scalar_t(0.353553385f) * wi_80 * xj_30 * yk_7;
        sum_v_329 += scalar_t(0.456435442f) * wi_80 * xj_32 * yk_7;

        scalar_t sum_v_143 = scalar_t(0);
        sum_v_143 += scalar_t(-0.422577113f) * wi_48 * xj_26 * yk_8;
        sum_v_143 += scalar_t(0.377964467f) * wi_48 * xj_28 * yk_6;
        sum_v_143 += scalar_t(0.327326834f) * wi_48 * xj_28 * yk_8;
        sum_v_143 += scalar_t(-0.534522474f) * wi_48 * xj_29 * yk_5;
        sum_v_143 += scalar_t(-0.327326834f) * wi_48 * xj_30 * yk_4;
        sum_v_143 += scalar_t(0.422577113f) * wi_48 * xj_32 * yk_4;

        scalar_t sum_v_140 = scalar_t(0);
        sum_v_140 += scalar_t(-0.422577113f) * wi_48 * xj_26 * yk_5;
        sum_v_140 += scalar_t(-0.327326834f) * wi_48 * xj_28 * yk_5;
        sum_v_140 += scalar_t(-0.267261237f) * wi_48 * xj_29 * yk_8;
        sum_v_140 += scalar_t(0.327326834f) * wi_48 * xj_30 * yk_7;
        sum_v_140 += scalar_t(-0.597614288f) * wi_48 * xj_31 * yk_6;
        sum_v_140 += scalar_t(-0.422577113f) * wi_48 * xj_32 * yk_7;

        scalar_t sum_v_52 = scalar_t(0);
        sum_v_52 += scalar_t(0.377964467f) * wi_25 * xj_27 * yk_4;
        sum_v_52 += scalar_t(0.478091449f) * wi_25 * xj_28 * yk_5;
        sum_v_52 += scalar_t(0.507092535f) * wi_25 * xj_29 * yk_6;
        sum_v_52 += scalar_t(0.478091449f) * wi_25 * xj_30 * yk_7;
        sum_v_52 += scalar_t(0.377964467f) * wi_25 * xj_31 * yk_8;

        scalar_t sum_v_142 = scalar_t(0);
        sum_v_142 += scalar_t(-0.597614288f) * wi_48 * xj_27 * yk_8;
        sum_v_142 += scalar_t(-0.377964467f) * wi_48 * xj_28 * yk_7;
        sum_v_142 += scalar_t(0.377964467f) * wi_48 * xj_30 * yk_5;
        sum_v_142 += scalar_t(0.597614288f) * wi_48 * xj_31 * yk_4;

        scalar_t sum_v_229 = scalar_t(0);
        sum_v_229 += scalar_t(0.597614288f) * wi_65 * xj_26 * yk_1;
        sum_v_229 += scalar_t(0.154303357f) * wi_65 * xj_28 * yk_1;
        sum_v_229 += scalar_t(-0.154303357f) * wi_65 * xj_30 * yk_3;
        sum_v_229 += scalar_t(0.487950057f) * wi_65 * xj_31 * yk_2;
        sum_v_229 += scalar_t(0.597614288f) * wi_65 * xj_32 * yk_3;

        scalar_t sum_v_399 = scalar_t(0);
        sum_v_399 += scalar_t(-0.353553385f) * wi_90 * xj_26 * yk_3;
        sum_v_399 += scalar_t(0.577350259f) * wi_90 * xj_27 * yk_2;
        sum_v_399 += scalar_t(-0.456435442f) * wi_90 * xj_28 * yk_3;
        sum_v_399 += scalar_t(-0.456435442f) * wi_90 * xj_30 * yk_1;
        sum_v_399 += scalar_t(0.353553385f) * wi_90 * xj_32 * yk_1;

        scalar_t sum_v_396 = scalar_t(0);
        sum_v_396 += scalar_t(0.456435442f) * wi_90 * xj_27 * yk_1;
        sum_v_396 += scalar_t(0.707106769f) * wi_90 * xj_29 * yk_3;
        sum_v_396 += scalar_t(-0.288675129f) * wi_90 * xj_30 * yk_2;
        sum_v_396 += scalar_t(0.456435442f) * wi_90 * xj_31 * yk_3;

        scalar_t sum_v_394 = scalar_t(0);
        sum_v_394 += scalar_t(-0.353553385f) * wi_90 * xj_27 * yk_1;
        sum_v_394 += scalar_t(0.353553385f) * wi_90 * xj_31 * yk_3;
        sum_v_394 += scalar_t(-0.866025388f) * wi_90 * xj_32 * yk_2;

        scalar_t sum_v_226 = scalar_t(0);
        sum_v_226 += scalar_t(0.487950057f) * wi_65 * xj_27 * yk_3;
        sum_v_226 += scalar_t(0.617213428f) * wi_65 * xj_28 * yk_2;
        sum_v_226 += scalar_t(-0.377964467f) * wi_65 * xj_29 * yk_1;
        sum_v_226 += scalar_t(-0.487950057f) * wi_65 * xj_31 * yk_1;

        scalar_t sum_v_227 = scalar_t(0);
        sum_v_227 += scalar_t(0.534522474f) * wi_65 * xj_28 * yk_1;
        sum_v_227 += scalar_t(0.654653668f) * wi_65 * xj_29 * yk_2;
        sum_v_227 += scalar_t(0.534522474f) * wi_65 * xj_30 * yk_3;

        scalar_t sum_v_379 = scalar_t(0);
        sum_v_379 += scalar_t(-0.707106769f) * wi_87 * xj_16 * yk_1;
        sum_v_379 += scalar_t(0.707106769f) * wi_87 * xj_20 * yk_3;

        scalar_t sum_v_301 = scalar_t(0);
        sum_v_301 += scalar_t(0.707106769f) * wi_76 * xj_16 * yk_6;
        sum_v_301 += scalar_t(-0.707106769f) * wi_76 * xj_18 * yk_4;

        scalar_t sum_v_280 = scalar_t(0);
        sum_v_280 += scalar_t(-0.707106769f) * wi_72 * xj_4 * yk_4;
        sum_v_280 += scalar_t(0.707106769f) * wi_72 * xj_8 * yk_8;

        scalar_t sum_v_263 = scalar_t(0);
        sum_v_263 += wi_71 * xj_3 * yk_14;

        scalar_t sum_v_255 = scalar_t(0);
        sum_v_255 += wi_71 * xj_3 * yk_12;

        scalar_t sum_v_247 = scalar_t(0);
        sum_v_247 += wi_71 * xj_3 * yk_10;

        scalar_t sum_v_242 = scalar_t(0);
        sum_v_242 += wi_70 * xj_2 * yk_9;

        scalar_t sum_v_250 = scalar_t(0);
        sum_v_250 += wi_70 * xj_2 * yk_11;

        scalar_t sum_v_258 = scalar_t(0);
        sum_v_258 += wi_70 * xj_2 * yk_13;

        scalar_t sum_v_266 = scalar_t(0);
        sum_v_266 += wi_70 * xj_2 * yk_15;

        scalar_t sum_v_261 = scalar_t(0);
        sum_v_261 += wi_69 * xj_1 * yk_14;

        scalar_t sum_v_253 = scalar_t(0);
        sum_v_253 += wi_69 * xj_1 * yk_12;

        scalar_t sum_v_245 = scalar_t(0);
        sum_v_245 += wi_69 * xj_1 * yk_10;

        scalar_t sum_v_240 = scalar_t(0);
        sum_v_240 += wi_68 * xj_0 * yk_9;

        scalar_t sum_v_248 = scalar_t(0);
        sum_v_248 += wi_68 * xj_0 * yk_11;

        scalar_t sum_v_256 = scalar_t(0);
        sum_v_256 += wi_68 * xj_0 * yk_13;

        scalar_t sum_v_264 = scalar_t(0);
        sum_v_264 += wi_68 * xj_0 * yk_15;

        scalar_t sum_v_167 = scalar_t(0);
        sum_v_167 += wi_51 * xj_0 * yk_7;

        scalar_t sum_v_159 = scalar_t(0);
        sum_v_159 += wi_51 * xj_0 * yk_5;

        scalar_t sum_v_156 = scalar_t(0);
        sum_v_156 += wi_52 * xj_1 * yk_4;

        scalar_t sum_v_164 = scalar_t(0);
        sum_v_164 += wi_52 * xj_1 * yk_6;

        scalar_t sum_v_172 = scalar_t(0);
        sum_v_172 += wi_52 * xj_1 * yk_8;

        scalar_t sum_v_169 = scalar_t(0);
        sum_v_169 += wi_53 * xj_2 * yk_7;

        scalar_t sum_v_161 = scalar_t(0);
        sum_v_161 += wi_53 * xj_2 * yk_5;

        scalar_t sum_v_158 = scalar_t(0);
        sum_v_158 += wi_54 * xj_3 * yk_4;

        scalar_t sum_v_166 = scalar_t(0);
        sum_v_166 += wi_54 * xj_3 * yk_6;

        scalar_t sum_v_174 = scalar_t(0);
        sum_v_174 += wi_54 * xj_3 * yk_8;

        scalar_t sum_v_176 = scalar_t(0);
        sum_v_176 += scalar_t(0.707106769f) * wi_56 * xj_5 * yk_3;
        sum_v_176 += scalar_t(0.707106769f) * wi_56 * xj_9 * yk_1;

        scalar_t sum_v_182 = scalar_t(0);
        sum_v_182 += scalar_t(0.707106769f) * wi_56 * xj_7 * yk_3;
        sum_v_182 += scalar_t(0.707106769f) * wi_56 * xj_9 * yk_2;

        scalar_t sum_v_216 = scalar_t(0);
        sum_v_216 += wi_63 * xj_22 * yk_0;

        scalar_t sum_v_218 = scalar_t(0);
        sum_v_218 += wi_63 * xj_24 * yk_0;

        scalar_t sum_v_303 = scalar_t(0);
        sum_v_303 += scalar_t(0.707106769f) * wi_77 * xj_21 * yk_3;
        sum_v_303 += scalar_t(0.707106769f) * wi_77 * xj_25 * yk_1;

        scalar_t sum_v_334 = scalar_t(0);
        sum_v_334 += scalar_t(-0.707106769f) * wi_81 * xj_35 * yk_3;
        sum_v_334 += scalar_t(0.707106769f) * wi_81 * xj_37 * yk_1;

        scalar_t sum_v_410 = scalar_t(0);
        sum_v_410 += wi_92 * xj_35 * yk_0;

        scalar_t sum_v_408 = scalar_t(0);
        sum_v_408 += wi_92 * xj_33 * yk_0;

        scalar_t sum_v_412 = scalar_t(0);
        sum_v_412 += wi_92 * xj_37 * yk_0;

        scalar_t sum_v_414 = scalar_t(0);
        sum_v_414 += wi_92 * xj_39 * yk_0;

        scalar_t sum_v_388 = scalar_t(0);
        sum_v_388 += scalar_t(0.707106769f) * wi_89 * xj_23 * yk_8;
        sum_v_388 += scalar_t(-0.707106769f) * wi_89 * xj_25 * yk_6;

        scalar_t sum_v_322 = scalar_t(0);
        sum_v_322 += wi_79 * xj_31 * yk_0;

        scalar_t sum_v_320 = scalar_t(0);
        sum_v_320 += wi_79 * xj_29 * yk_0;

        scalar_t sum_v_318 = scalar_t(0);
        sum_v_318 += wi_79 * xj_27 * yk_0;

        scalar_t sum_v_124 = scalar_t(0);
        sum_v_124 += wi_44 * xj_20 * yk_0;

        scalar_t sum_v_122 = scalar_t(0);
        sum_v_122 += wi_44 * xj_18 * yk_0;

        scalar_t sum_v_120 = scalar_t(0);
        sum_v_120 += wi_44 * xj_16 * yk_0;

        // writeback
        atomicAdd(&out[o_base + (56LL << 5)], sum_v_56);
        atomicAdd(&out[o_base + (153LL << 5)], sum_v_153);
        atomicAdd(&out[o_base + (150LL << 5)], sum_v_150);
        atomicAdd(&out[o_base + (87LL << 5)], sum_v_87);
        atomicAdd(&out[o_base + (88LL << 5)], sum_v_88);
        atomicAdd(&out[o_base + (341LL << 5)], sum_v_341);
        atomicAdd(&out[o_base + (339LL << 5)], sum_v_339);
        atomicAdd(&out[o_base + (342LL << 5)], sum_v_342);
        atomicAdd(&out[o_base + (344LL << 5)], sum_v_344);
        atomicAdd(&out[o_base + (419LL << 5)], sum_v_419);
        atomicAdd(&out[o_base + (415LL << 5)], sum_v_415);
        atomicAdd(&out[o_base + (420LL << 5)], sum_v_420);
        atomicAdd(&out[o_base + (238LL << 5)], sum_v_238);
        atomicAdd(&out[o_base + (235LL << 5)], sum_v_235);
        atomicAdd(&out[o_base + (149LL << 5)], sum_v_149);
        atomicAdd(&out[o_base + (336LL << 5)], sum_v_336);
        atomicAdd(&out[o_base + (335LL << 5)], sum_v_335);
        atomicAdd(&out[o_base + (331LL << 5)], sum_v_331);
        atomicAdd(&out[o_base + (148LL << 5)], sum_v_148);
        atomicAdd(&out[o_base + (134LL << 5)], sum_v_134);
        atomicAdd(&out[o_base + (137LL << 5)], sum_v_137);
        atomicAdd(&out[o_base + (48LL << 5)], sum_v_48);
        atomicAdd(&out[o_base + (49LL << 5)], sum_v_49);
        atomicAdd(&out[o_base + (83LL << 5)], sum_v_83);
        atomicAdd(&out[o_base + (222LL << 5)], sum_v_222);
        atomicAdd(&out[o_base + (223LL << 5)], sum_v_223);
        atomicAdd(&out[o_base + (224LL << 5)], sum_v_224);
        atomicAdd(&out[o_base + (138LL << 5)], sum_v_138);
        atomicAdd(&out[o_base + (135LL << 5)], sum_v_135);
        atomicAdd(&out[o_base + (314LL << 5)], sum_v_314);
        atomicAdd(&out[o_base + (310LL << 5)], sum_v_310);
        atomicAdd(&out[o_base + (391LL << 5)], sum_v_391);
        atomicAdd(&out[o_base + (390LL << 5)], sum_v_390);
        atomicAdd(&out[o_base + (393LL << 5)], sum_v_393);
        atomicAdd(&out[o_base + (315LL << 5)], sum_v_315);
        atomicAdd(&out[o_base + (307LL << 5)], sum_v_307);
        atomicAdd(&out[o_base + (306LL << 5)], sum_v_306);
        atomicAdd(&out[o_base + (130LL << 5)], sum_v_130);
        atomicAdd(&out[o_base + (133LL << 5)], sum_v_133);
        atomicAdd(&out[o_base + (47LL << 5)], sum_v_47);
        atomicAdd(&out[o_base + (109LL << 5)], sum_v_109);
        atomicAdd(&out[o_base + (105LL << 5)], sum_v_105);
        atomicAdd(&out[o_base + (111LL << 5)], sum_v_111);
        atomicAdd(&out[o_base + (113LL << 5)], sum_v_113);
        atomicAdd(&out[o_base + (70LL << 5)], sum_v_70);
        atomicAdd(&out[o_base + (72LL << 5)], sum_v_72);
        atomicAdd(&out[o_base + (107LL << 5)], sum_v_107);
        atomicAdd(&out[o_base + (41LL << 5)], sum_v_41);
        atomicAdd(&out[o_base + (101LL << 5)], sum_v_101);
        atomicAdd(&out[o_base + (196LL << 5)], sum_v_196);
        atomicAdd(&out[o_base + (202LL << 5)], sum_v_202);
        atomicAdd(&out[o_base + (283LL << 5)], sum_v_283);
        atomicAdd(&out[o_base + (285LL << 5)], sum_v_285);
        atomicAdd(&out[o_base + (287LL << 5)], sum_v_287);
        atomicAdd(&out[o_base + (362LL << 5)], sum_v_362);
        atomicAdd(&out[o_base + (368LL << 5)], sum_v_368);
        atomicAdd(&out[o_base + (370LL << 5)], sum_v_370);
        atomicAdd(&out[o_base + (372LL << 5)], sum_v_372);
        atomicAdd(&out[o_base + (289LL << 5)], sum_v_289);
        atomicAdd(&out[o_base + (284LL << 5)], sum_v_284);
        atomicAdd(&out[o_base + (286LL << 5)], sum_v_286);
        atomicAdd(&out[o_base + (294LL << 5)], sum_v_294);
        atomicAdd(&out[o_base + (367LL << 5)], sum_v_367);
        atomicAdd(&out[o_base + (365LL << 5)], sum_v_365);
        atomicAdd(&out[o_base + (371LL << 5)], sum_v_371);
        atomicAdd(&out[o_base + (197LL << 5)], sum_v_197);
        atomicAdd(&out[o_base + (195LL << 5)], sum_v_195);
        atomicAdd(&out[o_base + (199LL << 5)], sum_v_199);
        atomicAdd(&out[o_base + (118LL << 5)], sum_v_118);
        atomicAdd(&out[o_base + (116LL << 5)], sum_v_116);
        atomicAdd(&out[o_base + (114LL << 5)], sum_v_114);
        atomicAdd(&out[o_base + (69LL << 5)], sum_v_69);
        atomicAdd(&out[o_base + (71LL << 5)], sum_v_71);
        atomicAdd(&out[o_base + (106LL << 5)], sum_v_106);
        atomicAdd(&out[o_base + (100LL << 5)], sum_v_100);
        atomicAdd(&out[o_base + (40LL << 5)], sum_v_40);
        atomicAdd(&out[o_base + (64LL << 5)], sum_v_64);
        atomicAdd(&out[o_base + (66LL << 5)], sum_v_66);
        atomicAdd(&out[o_base + (68LL << 5)], sum_v_68);
        atomicAdd(&out[o_base + (62LL << 5)], sum_v_62);
        atomicAdd(&out[o_base + (9LL << 5)], sum_v_9);
        atomicAdd(&out[o_base + (35LL << 5)], sum_v_35);
        atomicAdd(&out[o_base + (91LL << 5)], sum_v_91);
        atomicAdd(&out[o_base + (97LL << 5)], sum_v_97);
        atomicAdd(&out[o_base + (60LL << 5)], sum_v_60);
        atomicAdd(&out[o_base + (180LL << 5)], sum_v_180);
        atomicAdd(&out[o_base + (186LL << 5)], sum_v_186);
        atomicAdd(&out[o_base + (188LL << 5)], sum_v_188);
        atomicAdd(&out[o_base + (271LL << 5)], sum_v_271);
        atomicAdd(&out[o_base + (277LL << 5)], sum_v_277);
        atomicAdd(&out[o_base + (279LL << 5)], sum_v_279);
        atomicAdd(&out[o_base + (281LL << 5)], sum_v_281);
        atomicAdd(&out[o_base + (348LL << 5)], sum_v_348);
        atomicAdd(&out[o_base + (350LL << 5)], sum_v_350);
        atomicAdd(&out[o_base + (358LL << 5)], sum_v_358);
        atomicAdd(&out[o_base + (352LL << 5)], sum_v_352);
        atomicAdd(&out[o_base + (347LL << 5)], sum_v_347);
        atomicAdd(&out[o_base + (349LL << 5)], sum_v_349);
        atomicAdd(&out[o_base + (357LL << 5)], sum_v_357);
        atomicAdd(&out[o_base + (272LL << 5)], sum_v_272);
        atomicAdd(&out[o_base + (270LL << 5)], sum_v_270);
        atomicAdd(&out[o_base + (193LL << 5)], sum_v_193);
        atomicAdd(&out[o_base + (191LL << 5)], sum_v_191);
        atomicAdd(&out[o_base + (189LL << 5)], sum_v_189);
        atomicAdd(&out[o_base + (177LL << 5)], sum_v_177);
        atomicAdd(&out[o_base + (183LL << 5)], sum_v_183);
        atomicAdd(&out[o_base + (98LL << 5)], sum_v_98);
        atomicAdd(&out[o_base + (92LL << 5)], sum_v_92);
        atomicAdd(&out[o_base + (30LL << 5)], sum_v_30);
        atomicAdd(&out[o_base + (32LL << 5)], sum_v_32);
        atomicAdd(&out[o_base + (57LL << 5)], sum_v_57);
        atomicAdd(&out[o_base + (94LL << 5)], sum_v_94);
        atomicAdd(&out[o_base + (25LL << 5)], sum_v_25);
        atomicAdd(&out[o_base + (27LL << 5)], sum_v_27);
        atomicAdd(&out[o_base + (29LL << 5)], sum_v_29);
        atomicAdd(&out[o_base + (15LL << 5)], sum_v_15);
        atomicAdd(&out[o_base + (23LL << 5)], sum_v_23);
        atomicAdd(&out[o_base + (18LL << 5)], sum_v_18);
        atomicAdd(&out[o_base + (6LL << 5)], sum_v_6);
        atomicAdd(&out[o_base + (13LL << 5)], sum_v_13);
        atomicAdd(&out[o_base + (21LL << 5)], sum_v_21);
        atomicAdd(&out[o_base + (16LL << 5)], sum_v_16);
        atomicAdd(&out[o_base + (4LL << 5)], sum_v_4);
        atomicAdd(&out[o_base + (2LL << 5)], sum_v_2);
        atomicAdd(&out[o_base + (44LL << 5)], sum_v_44);
        atomicAdd(&out[o_base + (128LL << 5)], sum_v_128);
        atomicAdd(&out[o_base + (125LL << 5)], sum_v_125);
        atomicAdd(&out[o_base + (78LL << 5)], sum_v_78);
        atomicAdd(&out[o_base + (79LL << 5)], sum_v_79);
        atomicAdd(&out[o_base + (75LL << 5)], sum_v_75);
        atomicAdd(&out[o_base + (214LL << 5)], sum_v_214);
        atomicAdd(&out[o_base + (210LL << 5)], sum_v_210);
        atomicAdd(&out[o_base + (212LL << 5)], sum_v_212);
        atomicAdd(&out[o_base + (209LL << 5)], sum_v_209);
        atomicAdd(&out[o_base + (300LL << 5)], sum_v_300);
        atomicAdd(&out[o_base + (299LL << 5)], sum_v_299);
        atomicAdd(&out[o_base + (302LL << 5)], sum_v_302);
        atomicAdd(&out[o_base + (384LL << 5)], sum_v_384);
        atomicAdd(&out[o_base + (383LL << 5)], sum_v_383);
        atomicAdd(&out[o_base + (375LL << 5)], sum_v_375);
        atomicAdd(&out[o_base + (381LL << 5)], sum_v_381);
        atomicAdd(&out[o_base + (378LL << 5)], sum_v_378);
        atomicAdd(&out[o_base + (205LL << 5)], sum_v_205);
        atomicAdd(&out[o_base + (207LL << 5)], sum_v_207);
        atomicAdd(&out[o_base + (53LL << 5)], sum_v_53);
        atomicAdd(&out[o_base + (86LL << 5)], sum_v_86);
        atomicAdd(&out[o_base + (11LL << 5)], sum_v_11);
        atomicAdd(&out[o_base + (233LL << 5)], sum_v_233);
        atomicAdd(&out[o_base + (232LL << 5)], sum_v_232);
        atomicAdd(&out[o_base + (230LL << 5)], sum_v_230);
        atomicAdd(&out[o_base + (406LL << 5)], sum_v_406);
        atomicAdd(&out[o_base + (404LL << 5)], sum_v_404);
        atomicAdd(&out[o_base + (402LL << 5)], sum_v_402);
        atomicAdd(&out[o_base + (407LL << 5)], sum_v_407);
        atomicAdd(&out[o_base + (326LL << 5)], sum_v_326);
        atomicAdd(&out[o_base + (324LL << 5)], sum_v_324);
        atomicAdd(&out[o_base + (329LL << 5)], sum_v_329);
        atomicAdd(&out[o_base + (143LL << 5)], sum_v_143);
        atomicAdd(&out[o_base + (140LL << 5)], sum_v_140);
        atomicAdd(&out[o_base + (52LL << 5)], sum_v_52);
        atomicAdd(&out[o_base + (142LL << 5)], sum_v_142);
        atomicAdd(&out[o_base + (229LL << 5)], sum_v_229);
        atomicAdd(&out[o_base + (399LL << 5)], sum_v_399);
        atomicAdd(&out[o_base + (396LL << 5)], sum_v_396);
        atomicAdd(&out[o_base + (394LL << 5)], sum_v_394);
        atomicAdd(&out[o_base + (226LL << 5)], sum_v_226);
        atomicAdd(&out[o_base + (227LL << 5)], sum_v_227);
        atomicAdd(&out[o_base + (379LL << 5)], sum_v_379);
        atomicAdd(&out[o_base + (301LL << 5)], sum_v_301);
        atomicAdd(&out[o_base + (280LL << 5)], sum_v_280);
        atomicAdd(&out[o_base + (263LL << 5)], sum_v_263);
        atomicAdd(&out[o_base + (255LL << 5)], sum_v_255);
        atomicAdd(&out[o_base + (247LL << 5)], sum_v_247);
        atomicAdd(&out[o_base + (242LL << 5)], sum_v_242);
        atomicAdd(&out[o_base + (250LL << 5)], sum_v_250);
        atomicAdd(&out[o_base + (258LL << 5)], sum_v_258);
        atomicAdd(&out[o_base + (266LL << 5)], sum_v_266);
        atomicAdd(&out[o_base + (261LL << 5)], sum_v_261);
        atomicAdd(&out[o_base + (253LL << 5)], sum_v_253);
        atomicAdd(&out[o_base + (245LL << 5)], sum_v_245);
        atomicAdd(&out[o_base + (240LL << 5)], sum_v_240);
        atomicAdd(&out[o_base + (248LL << 5)], sum_v_248);
        atomicAdd(&out[o_base + (256LL << 5)], sum_v_256);
        atomicAdd(&out[o_base + (264LL << 5)], sum_v_264);
        atomicAdd(&out[o_base + (167LL << 5)], sum_v_167);
        atomicAdd(&out[o_base + (159LL << 5)], sum_v_159);
        atomicAdd(&out[o_base + (156LL << 5)], sum_v_156);
        atomicAdd(&out[o_base + (164LL << 5)], sum_v_164);
        atomicAdd(&out[o_base + (172LL << 5)], sum_v_172);
        atomicAdd(&out[o_base + (169LL << 5)], sum_v_169);
        atomicAdd(&out[o_base + (161LL << 5)], sum_v_161);
        atomicAdd(&out[o_base + (158LL << 5)], sum_v_158);
        atomicAdd(&out[o_base + (166LL << 5)], sum_v_166);
        atomicAdd(&out[o_base + (174LL << 5)], sum_v_174);
        atomicAdd(&out[o_base + (176LL << 5)], sum_v_176);
        atomicAdd(&out[o_base + (182LL << 5)], sum_v_182);
        atomicAdd(&out[o_base + (216LL << 5)], sum_v_216);
        atomicAdd(&out[o_base + (218LL << 5)], sum_v_218);
        atomicAdd(&out[o_base + (303LL << 5)], sum_v_303);
        atomicAdd(&out[o_base + (334LL << 5)], sum_v_334);
        atomicAdd(&out[o_base + (410LL << 5)], sum_v_410);
        atomicAdd(&out[o_base + (408LL << 5)], sum_v_408);
        atomicAdd(&out[o_base + (412LL << 5)], sum_v_412);
        atomicAdd(&out[o_base + (414LL << 5)], sum_v_414);
        atomicAdd(&out[o_base + (388LL << 5)], sum_v_388);
        atomicAdd(&out[o_base + (322LL << 5)], sum_v_322);
        atomicAdd(&out[o_base + (320LL << 5)], sum_v_320);
        atomicAdd(&out[o_base + (318LL << 5)], sum_v_318);
        atomicAdd(&out[o_base + (124LL << 5)], sum_v_124);
        atomicAdd(&out[o_base + (122LL << 5)], sum_v_122);
        atomicAdd(&out[o_base + (120LL << 5)], sum_v_120);
    }

}

// launcher helper
template <typename scalar_t>
void launch_uniform1d_combine_u32_path1490_fwd(
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
    uniform1d_combine_u32_path1490_fwd<scalar_t><<<grid, block, 0, stream>>>(
        w, x_all, y, out, src_idx, dst_idx, b_list,
        B, Iw, Ix, Ky, V, U);
}


torch::Tensor launcher_uniform1d_combine_u32_path1490_fwd(
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // [B,Ky,1]
    torch::Tensor src_idx,    // [B] int32
    torch::Tensor dst_idx,    // [B] int32
    torch::Tensor b_list,     // [B] int32
    int64_t V64)
{

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(dst_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "w/x_all/y must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");

    int B  = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);

    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto out = torch::zeros({S, V, U}, w.options());

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_combine_u32_path1490_fwd", [&] {

        launch_uniform1d_combine_u32_path1490_fwd<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U, stream);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

TORCH_LIBRARY(uniform1d_combine_u32_path1490_fwd_codegen, m) {
    m.def("run", &launcher_uniform1d_combine_u32_path1490_fwd);
}
