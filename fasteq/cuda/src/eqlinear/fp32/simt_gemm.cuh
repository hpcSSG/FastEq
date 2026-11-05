/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/simt_gemm.cuh
 *
 * M2's hand-written grouped SIMT GEMM (impl 1).  The stock CUTLASS 2.x SIMT path cannot express a
 * useful exact-fit N tile for V = 224 and burns 255 registers (with small spills) on a 128-
 * accumulator thread tile, which halves occupancy.  This kernel keeps the same maths -- one fp32
 * accumulator per output element, K ascending, so the result is bit-identical to the stock path --
 * but fixes the two things that matter:
 *
 *   1. the thread tile is a per-shape choice (8x16 for V=128, 8x12 for V=96, 8x14 for V=224), and
 *      every one of those is an exact fit for its V, so no N-padding work is done;
 *   2. the register budget is controlled by construction (no stock iterators/epilogue state), so a
 *      128-accumulator tile fits in a CTA that allows 2-3 CTAs per SM instead of 2 x 255 regs.
 *
 * Shared-memory bytes per FMA is the constraint this targets: (tm+tn)/(tm*tn) elements, i.e. 0.75
 * B/FMA at 8x16 vs 1.0 at the 8x8 tile M1 used.  Vector width does not change that (and the stock
 * path's warp LDS was already 128-bit); it only changes instruction count.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include "bwd.h"
#include "grouped_gemm_fp32.cuh"

namespace eqlin_fp32 {

/// Build the hand-written plan for this problem set.  The tile is chosen from V and B; all shapes
/// this project accepts divide their tiles exactly, which the plan asserts.
std::unique_ptr<GroupedGemmPlan::ImplBase> make_simt_custom_plan(
    int M, int N, int num_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, float const *a_data, float const *b_data,
    bool a_view = false);

/// Build the path-chained plan (M5): one tile covers an output row's `pg` per-path problems, which
/// it accumulates in smem between paths, so there is no partial slab and no reduction kernel.
/// `rows` holds the per-path problems; `rbase`/`pg` give output row j's contiguous run of them.
/// Returns nullptr when the fused grid would not fill the machine -- the caller should then use the
/// split plan plus `launch_reduce_partials`.
std::unique_ptr<GroupedGemmPlan::ImplBase> make_simt_fused_plan(
    int M, int N, int num_problems, int num_out_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, std::vector<int> const &rbase, std::vector<int> const &pg,
    float const *a_data, float const *b_data, bool a_view = false);

/// Backward plan: one tile per (problem, m, n), every problem carrying its own A/C offsets and
/// strides.  Reuses the same SIMT mainloop; the forward plans are untouched.
std::unique_ptr<BwdPlan> make_simt_bwd_plan(int N, int B,
                                            std::vector<BwdProblem> const &problems);

}  // namespace eqlin_fp32
