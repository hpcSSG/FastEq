/***************************************************************************************************
 * fasteq eqlinear backends/tf32/transpose_kt.h
 *
 * wgrad's operand preparation: turn a [B, rows, dim] slice of x or dy into the K-contiguous form the
 * TN-only tf32 MMA needs.
 *
 * wgrad contracts `dw[p] = cg_p * X_p^T Y_g` over K = B*i_g, with k = b*i_g + t.  Both x and dy are
 * stored [K, dim] with `dim` contiguous, so neither is a K-major A/B operand; this kernel writes the
 * transposed view once per call (`Xt[p] = [U, K]`, `Yt[g] = [V, K]`).
 *
 * The dy side is also where the tf32x3 split is materialised (`b_big`/`b_res`): the mainloop splits
 * its *A* operand in registers, but B must arrive pre-split.  Splitting the smaller tensor (dy, |J|
 * rows) rather than x (|I| rows) is what keeps this prep at one extra write instead of two.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

namespace eqlin {

/// One transposed slab: source rows `run_off .. run_off + i_g` of `src`, destination at `dst_off`.
struct KtPath {
  int run_off;
  int i_g;
  long long dst_off;
};

/// dst[dim, B*i_g] for every path, read from `src[B, ..., dim]` with row pitch `src_row_stride`.
///
/// `dst_res` non-null means "write the tf32x3 halves of the source instead of a plain copy"
/// (dst_big = the big part); it is the dy side.  `k_tiles_max` is the largest `ceil(B*i_g/32)` over
/// the paths, i.e. the grid extent.
void launch_transpose_kt(float const *src, float *dst_big, float *dst_res,
                         KtPath const *paths, int num_paths, int B, int dim,
                         long long src_row_stride, int k_tiles_max, cudaStream_t stream);

}  // namespace eqlin
