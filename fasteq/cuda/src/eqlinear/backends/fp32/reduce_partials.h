/***************************************************************************************************
 * fasteq eqlinear backends/fp32/reduce_partials.h
 *
 * M4: sum the per-path partial slabs produced by the split problem table.
 *
 *   out[b, j, :] = sum_{m = 0 .. pg[j]-1} partials[b, rbase[j] + m, :]
 *
 * in fp32, path order ascending -- the same accumulation torch's per-path loop performs.  One
 * thread per output element, no atomics, so the result is deterministic.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

namespace eqlin_fp32 {

/// `partials` is [B][R][V]; `rbase`/`pg` have J entries and describe each output row's contiguous
/// run of per-path problems.  V must be a multiple of 4 (float4 stores).
void launch_reduce_partials(float const *partials, int const *rbase, int const *pg, float *out,
                            int B, int J, int V, int R, cudaStream_t stream);

}  // namespace eqlin_fp32
