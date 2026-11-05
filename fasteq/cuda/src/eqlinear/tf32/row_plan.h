/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/row_plan.h
 *
 * One group of the block-sparse grouped GEMM == one output row j of the eqlinear.
 * Shared by the 2.x/SM80 path (grouped_gemm.cu) and the 3.x/SM90 path (sm90_grouped_gemm.cu).
 **************************************************************************************************/
#pragma once

namespace eqlin {

/// One GEMM problem.  Everything defaults to "use the plan-wide value", so a caller that only fills
/// {K, a_off, b_off} gets exactly the forward's behaviour.
///
/// The backward's fused plan (B3) mixes two kinds of problem in one row table -- dgrad writes into
/// the caller's dx, wgrad into our partial buffer, and they read different A/B tensors -- which is
/// what the per-row bases below are for.  The ptr-array kernel has always carried a pointer per
/// problem; these fields just let the host say so.
struct RowPlan {
  int K = 0;            // contraction length of this output row (g_op * U, or 3*g_op*U for K-concat)
  long long a_off = 0;  // column offset into this row's A tensor (element offset when A is a view)
  long long b_off = 0;  // column offset into this row's B tensor
  /// Leading dimension of this row's A strip; 0 means "the plan-wide lda".  Only the activation-view
  /// mode sets it, because there each row may be a slab of a differently pitched tensor.
  long long lda = 0;
  /// Leading dimension of this row's B strip; 0 means "the plan-wide ldb".  The backward's wgrad sets
  /// it: each group's transposed dy slab is [V, B*i_g] and i_g differs per group.
  long long ldb = 0;
  /// Leading dimension of this row's C tile; 0 means "the plan-wide ldc".  dgrad writes dx rows of
  /// `I*U` while wgrad's partial tiles are `V` apart, and the fused backward carries both.
  long long ldc = 0;
  /// Element offset of this row's C/D tile from this row's C base; **-1 means "j * ldc"** from the
  /// base passed to `launch`.  dgrad sets it because row j writes the input row i(j), which is not j
  /// (and that offset can legitimately be 0, hence the -1 sentinel rather than 0).
  long long c_off = -1;
  /// Per-row operand bases.  Null means "use the base the plan was built with" (A/B/B_res) or the
  /// `launch(out)` argument (C).  `a_base` and `c_base` are what makes a *fused* dgrad+wgrad launch
  /// possible: dgrad's A is the caller's dy (rebound every call) while wgrad's is our Xt, and their
  /// C tiles land in different allocations.
  void const *a_base = nullptr;
  void const *b_base = nullptr;
  void const *b_res_base = nullptr;
  void *c_base = nullptr;
  /// This row's GEMM M; 0 means "the plan-wide M".  A fused backward mixes dgrad (M = B) with
  /// wgrad (M = U) problems, so M has to be per problem.  N is not: dgrad's N is U and wgrad's is V,
  /// and the fused plan requires U == V because both share one tile-N instantiation.
  int M = 0;
};

}  // namespace eqlin
