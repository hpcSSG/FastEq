/***************************************************************************************************
 * fasteq eqlinear backends/tf32/prep_kernels.h
 *
 * Preprocessing kernels turning the eqlinear inputs into the two packed tensors the grouped GEMM
 * consumes:
 *
 *   A_prep[B, I*U]  col-major-offset blocks, group g occupying columns [a_off[g], a_off[g+1])
 *                   A_prep[b, a_off[g] + t*K + m*U + k] = x[b, row_start[g] + m*i_g + t, k]
 *   B_prep[V, P*U]  B_prep[n, b_off[g] + m*U + k]      = cg[p0_g + m] * W[p0_g + m, k, n]
 *
 * Both are pure gathers driven by host-precomputed index tables, so the device side needs no
 * group search at all.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

namespace eqlin {

/// Upper bound on the number of input paths.  cg is passed to the pack kernel by value (the same
/// trick the original eqlinear reference uses with cg_T), so `prepare` needs no host->device copy for it.
constexpr int kMaxPaths = 32;

struct CgVals {
  float v[kMaxPaths];
};

/// a_prep[b, col] = x[b, src_index[col]] for col in [0, I*U);  src_index has I*U entries.
void launch_permute_a(float const *x, float *a_prep,
                      int const *src_index, int I, int U, int B, cudaStream_t stream);

/// b_prep[n, col] = cg.v[p_of_col[col]] * w[src_base[col] + n];  both tables have P*U entries.
void launch_pack_b(float const *w, float *b_prep,
                   int const *src_base, int const *p_of_col, CgVals cg,
                   int P, int U, int V, cudaStream_t stream);

}  // namespace eqlin
