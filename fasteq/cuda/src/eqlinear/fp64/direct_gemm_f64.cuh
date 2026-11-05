/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/direct_gemm_f64.cuh
 *
 * M5-B: a hand-written grouped kernel with the 2.x occupancy structure (128 threads = 4 warps,
 * 3 CTAs/SM = 12 MMA warps) and the 3.x operand sourcing (A read straight from x through the
 * composed (U, m) K axis, so multi-path shapes need no permute).
 *
 * Why a new kernel instead of a stock one:
 *   * the SM90 cooperative kernel is pinned at 8 MMA warps (2 warpgroups, 232 regs/thread); 2 CTAs
 *     do not fit;
 *   * its TMA operand path cannot vectorise the FP64 DMMA fragment: a thread's A elements are
 *     (m0+8*v0+16*mi, k0+4*v1), so a K-major tile forces 32 scalar 8-byte loads per k-tile and 50 %
 *     bank-conflict replays (`FP64_M5_REPORT.md` §4);
 *   * the 2.x kernel has the right occupancy and a conflict-free smem layout (0.46 %) but cannot
 *     express the composed A, and its tile is pinned by macros.
 *
 * The layout below is a fragment-major variant of the 2.x "crosswise" idea: element (m, k) of a warp
 * block sits at
 *     off = 512*(k/4 + 4*(m/16)) + 16*((m%8)*4 + (k%4)) + 8*((m%16)/8)          [A, 32x16 block]
 *     off = 512*(2*(n/8) + (k%16)/8) + 16*((n%8)*4 + (k%4)) + 8*((k%8)/4)      [B, 16x32 block]
 * so each thread's 16 doubles are 8 contiguous 16-byte chunks, the 32 lanes of one LDS.128 cover a
 * contiguous 512-byte block (zero conflicts), and the global->shared side stays 8-byte cp.async
 * (16 B would need (m, k)/(m, k+1) adjacent, which contradicts the fragment pairing).
 * Verified in the `/tmp` spike: 16 LDS.128 per k-tile per thread, 0 conflicts, bit-exact.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "row_plan.h"

namespace eqlin {

/// One hand-written grouped GEMM over the same per-row plan the other backends use.
/// `a_data` may be null when the caller rebinds the A base per call (`bind_a`).
class DirectGemmPlan {
 public:
  struct ImplBase;

  DirectGemmPlan(int B, int V, int num_rows,
                 long long lda, long long ldb, long long ldc,
                 std::vector<RowPlan> const &rows,
                 double const *a_data, double const *b_data, int u, bool use_k8 = true);
  ~DirectGemmPlan();

  DirectGemmPlan(DirectGemmPlan const &) = delete;
  DirectGemmPlan &operator=(DirectGemmPlan const &) = delete;

  /// Launch, writing row j of the output to out + j*V (row stride ldc).
  int launch(double *out, cudaStream_t stream);

  /// Point every row's A at `a_base + row.a_off` (activation-view mode).
  void bind_a(double const *a_base, cudaStream_t stream);

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin
