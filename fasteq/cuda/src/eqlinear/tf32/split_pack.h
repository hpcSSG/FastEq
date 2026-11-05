/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/split_pack.h
 *
 * M2 / Step A1: build the K-concatenated tf32x3 operands.
 *
 *   A3[b, a_col(op) + t*3K0 + s*K0 + m*U + k] = { big(x), res(x), big(x) }   for s = 0,1,2
 *   B3[n, b_col(op) + s*K0 + m*U + k]          = { big(w'), big(w'), res(w') } for s = 0,1,2
 *   with x = x[b, r0(op) + m*i_op + t, k] and w' = cg[p0+m] * W[p0+m, k, n]
 *
 * so that a single tf32 GEMM over K3 = 3*K0 computes
 *   A_big@B_big + A_res@B_big + A_big@B_res  ==  the 3-term tf32x3 emulation of A@B.
 *
 * The split uses cutlass::NumericConverterFastF32 -- the same primitive the Ampere 3xTF32 path
 * (and this project's M1) uses, so the numerics match bit-for-bit up to accumulation order.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include "prep_kernels.h"  // CgVals

namespace eqlin {

/// a3[b, col] from x; `src_index`/`is_res` have 3*I*U entries.
void launch_split_a(float const *x, float *a3,
                    int const *src_index, unsigned char const *is_res,
                    long long cols, int I, int U, int B, cudaStream_t stream);

/// b3[n, col] from w and cg; the tables have 3*P*U entries.
void launch_split_b(float const *w, float *b3,
                    int const *src_base, int const *p_of_col, unsigned char const *is_res,
                    CgVals cg, long long cols, int P, int U, int V, cudaStream_t stream);

/// M3-2 operand build: the two tf32 halves of B, written to separate buffers in one pass over W
/// (columns `cols` == P*U, the same per-row offsets the K-concatenated path uses).
///
///   b_big[n, col] = truncate_tf32(cg.v[p_of_col[col]] * w[src_base[col] + n])
///   b_res[n, col] = (that product) - b_big[n, col]
///
/// B is the weight side, so this runs once per weight load rather than per inference.
/// `rn` picks the split point: false truncates the big part (the historical behaviour, bit-identical
/// to `NumericConverterFastF32`'s default), true rounds it to nearest and measures the residual
/// against that.  The two differ only in where the half-ulp lands, which is worth 2x on the dropped
/// res*res term -- see TF32_PRECISION_REPORT.md §3.2.
void launch_pack_b_split(float const *w, float *b_big, float *b_res,
                         int const *src_base, int const *p_of_col, CgVals cg,
                         long long cols, int P, int U, int V, bool rn, cudaStream_t stream);

}  // namespace eqlin
