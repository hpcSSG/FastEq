/***************************************************************************************************
 * fasteq eqlinear backends/fp32/prep_bwd.h
 *
 * Backward shape prep.  These are pure layout transforms (allowed to be excluded from the core
 * timing), measured separately by the harness:
 *
 *   cgW[p,u,v]     = cg[p] * W[p,u,v]            (weight side; hoisted at create)
 *   Xt[p][u,k]     = x[b, in_prefix[p]+t, u]     (k = b*i_p + t)
 *   Yt[g][v,k]     = dy[b, out_prefix[g]+t, v]   (k = b*i_g + t)
 *
 * Xt/Yt make both wgrad operands K-contiguous, which is what the SIMT mainloop's cp.async copy
 * requires (the dgrad operands already are).
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include "bwd.h"
#include "prep_kernels.h"

namespace eqlin_fp32 {

void launch_pack_cgw(float const *w, float *cgw, CgVals cg, int P, int U, int V,
                     cudaStream_t stream);

/// x[B,I,U] -> Xt[P] slabs.  `src_batch_stride` = I*U; `max_rows` = max i_dims.
void launch_transpose_x(float const *x, float *xt, TransposeDesc const *descs, int num_desc, int B,
                        int src_batch_stride, int max_rows, int U, cudaStream_t stream);

/// dy[B,J,V] -> Yt[G] slabs.  `src_batch_stride` = J*V; `max_rows` = max i_dims.
void launch_transpose_dy(float const *dy, float *yt, TransposeDesc const *descs, int num_desc,
                         int B, int src_batch_stride, int max_rows, int V, cudaStream_t stream);

}  // namespace eqlin_fp32
