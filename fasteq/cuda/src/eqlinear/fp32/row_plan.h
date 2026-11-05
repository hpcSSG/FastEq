/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/row_plan.h
 *
 * One group of the block-sparse grouped GEMM == one output row j of the eqlinear.
 **************************************************************************************************/
#pragma once

namespace eqlin_fp32 {

/// One GEMM problem.  M (= B) and N (= V) are plan-wide here; only the contraction length and the
/// two column offsets vary per output row.  Later versions that need per-row strides or bases (the
/// activation view, the fused backward) append fields to this struct -- they never repurpose these.
struct RowPlan {
  int K = 0;            // contraction length of this output row (g_op * U)
  long long a_off = 0;  // column offset into A_prep
  long long b_off = 0;  // column offset into B_prep
};

}  // namespace eqlin_fp32
