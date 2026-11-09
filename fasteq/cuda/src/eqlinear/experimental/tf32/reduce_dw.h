/***************************************************************************************************
 * fasteq eqlinear backends/tf32/reduce_dw.h
 *
 * wgrad's split-K reduction: sum the per-split partial tiles, scale by cg, write dw.
 *
 * The splits are separate *problems* in the grouped GEMM (no scheduler support, no atomics, no
 * fixup), so this is the only place the S partials meet.  Accumulating in fp64 costs nothing here
 * (the partials are ~50 MB) and is what keeps a K = 40k contraction at the tf32x3 error level.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include "prep_kernels.h"  // CgVals

namespace eqlin {

/// dw[p,u,v] = cg[p] * sum_s partials[off[p] + s*uv + (u*V+v)]
///
/// The split count is per path (`sp[p]`): a single global S would either over-split the paths with
/// a short K (wasting a pipeline fill per CTA) or under-split the long ones.
void launch_reduce_dw(float const *partials, CgVals cg, float *dw, int uv, int P,
                      int const *sp, long long const *off, cudaStream_t stream);

}  // namespace eqlin
