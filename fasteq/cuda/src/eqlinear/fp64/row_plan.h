/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/row_plan.h
 *
 * One group of the block-sparse grouped GEMM == one output row j of the eqlinear.
 *
 * The forward only ever needs these three per-row values; the tf32 line's fused-backward fields
 * (per-row bases / ldc / M) are deliberately absent here until FP64 has a backward.
 **************************************************************************************************/
#pragma once

namespace eqlin {

struct RowPlan {
  int K = 0;            // contraction length of this output row (g_op * U)
  long long a_off = 0;  // column offset into the packed A tensor (element offset when A is x)
  long long b_off = 0;  // column offset into the packed B tensor
  /// M2 (SM90 TMA mainloop): the A operand is read straight from x with K composed of
  /// (m, u).  `m` is the number of input paths in this row's group and `m_stride` is the element
  /// stride between consecutive path blocks (`i_g * U`).  Defaults are the flat single-path
  /// case (m = 1), which is what the 2.x plan and the A-view mode use.
  int m = 1;
  long long m_stride = 0;  // 0 means "U" (a single block; the value is irrelevant for m == 1)
  long long lda = 0, ldb = 0, ldc = 0;  // 0 uses the plan-wide stride
  long long c_off = -1;  // -1 uses row-index * output-column-stride
  int M = 0, N = 0;  // 0 uses the plan-wide extent
};

}  // namespace eqlin
