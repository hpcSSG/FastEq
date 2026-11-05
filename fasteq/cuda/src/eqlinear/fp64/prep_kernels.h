/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/prep_kernels.h
 *
 * Preprocessing kernels turning the eqlinear inputs into the two packed tensors the grouped GEMM
 * consumes (both FP64):
 *
 *   A_prep[B, I*U]  A_prep[b, a_off[g] + t*K + m*U + k] = x[b, row_start[g] + m*i_g + t, k]
 *   B_prep[V, P*U]  B_prep[n, b_off[g] + m*U + k]      = cg[p0_g + m] * W[p0_g + m, k, n]
 *
 * Both are pure gathers driven by host-precomputed index tables, so the device side needs no
 * group search at all.  The A side is skipped entirely in activation-view mode (single-path
 * groups): there A_prep's column strip is bit-identical to a slice of x.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

namespace eqlin {

/// Upper bound on the number of input paths.  cg is passed to the pack kernel by value, so the
/// per-call path needs no host->device copy for it.
constexpr int kMaxPaths = 32;

struct CgVals {
  double v[kMaxPaths];
};

/// a_prep[b, col] = x[b, src_index[col]] for col in [0, I*U);  src_index has I*U entries.
void launch_permute_a(double const *x, double *a_prep,
                      int const *src_index, int I, int U, int B, cudaStream_t stream);

/// b_prep[n, col] = cg.v[p_of_col[col]] * w[src_base[col] + n];  both tables have P*U entries.
void launch_pack_b(double const *w, double *b_prep,
                   int const *src_base, int const *p_of_col, CgVals cg,
                   int P, int U, int V, cudaStream_t stream);

}  // namespace eqlin
