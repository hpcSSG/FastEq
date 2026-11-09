/***************************************************************************************************
 * fasteq eqlinear backends/tf32/reduce_fwd.h
 *
 * P1: the reduction half of the forward split-K path.
 *
 * `Sm90SplitKPlan` already turns "split this row's K into S slices" into "list S times as many
 * problems", and redirects their output into a `(J*S, B, V)` partial buffer.  Until now nothing
 * summed the S partials, because the only thing that existed was the ablation knob.  This is that
 * sum.  Numerically it is the cheap way to shorten the accumulation chain (`3K` -> `3K/S`) without
 * touching a single kernel register: the price is S times the output traffic instead.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

namespace eqlin {

/// `out[b, j, v] = sum_s partials[j*S + s, b, v]`, accumulated in fp64.
///
/// The partial buffer follows the plan's C layout: element (b, v) of problem `j'` sits at
/// `partials[j'*V + b*(J*S*V) + v]`.  Consecutive threads walk `v`, so both the S reads and the
/// single write are fully coalesced; the (j, s) -> j remap costs no extra traffic.
void launch_reduce_fwd(float const *partials, float *out, int J, int B, int V, int S,
                       cudaStream_t stream);

}  // namespace eqlin
