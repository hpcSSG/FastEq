/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/reduce_dw.h
 *
 * Deterministic split-K reduction for wgrad:
 *
 *   dw[p,u,v] = cg[p] * sum_{s=0}^{splits[p]-1} partials[split_off[p]+s][u,v]
 *
 * One thread per output element, splits accumulated in ascending order in fp64, no atomics.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include "prep_kernels.h"

namespace eqlin_fp32 {

/// `partials` is [num_problems][U][V]; `split_off`/`splits` have P entries.
void launch_reduce_dw(float const *partials, int const *split_off, int const *splits, CgVals cg,
                      float *dw, int P, int U, int V, cudaStream_t stream);

}  // namespace eqlin_fp32
