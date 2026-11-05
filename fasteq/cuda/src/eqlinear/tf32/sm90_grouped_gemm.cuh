/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/sm90_grouped_gemm.cuh
 *
 * M2 backend: CUTLASS 3.x SM90 (WGMMA + TMA + warp-specialized) PtrArray grouped GEMM.
 *
 * Same group structure as the M1 path -- one group per eqlinear output row, M=B, N=V, K=g_op*U
 * (or 3*g_op*U for the K-concatenated tf32x3) -- but the compute runs on Hopper's WGMMA pipe
 * instead of SM80's mma.sync.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear.h"
#include "row_plan.h"

namespace eqlin {

/// True when this build can instantiate the SM90 TMA warp-specialized collective.
bool sm90_available();

/// Which mainloop the plan instantiates, and therefore what the caller must have packed.
enum class Sm90Operands {
  /// ElementA/ElementB = float: the TMA rounds fp32 to tf32 in flight, i.e. plain 1xTF32. Only a
  /// performance probe.
  kFp32,
  /// ElementA/ElementB = tfloat32_t, consumed verbatim: the caller must supply big/res-split,
  /// K-concatenated operands with K = 3*g*U.
  kTf32KConcat,
  /// ElementA = float (raw, one copy), ElementB = tfloat32_t.  The mainloop splits A into
  /// (big, residual) in shared memory and issues the three tf32x3 products itself, so `a_data` is a
  /// plain fp32 permute with K = g*U and `b_data`/`b_res_data` are the two pre-split B halves.
  kFp32InKernelSplit,
  /// Same operands and numerics as kFp32InKernelSplit, but the mainloop takes A from *registers*
  /// (`rs_` WGMMA) and derives the residual there, so no A_res tile is staged.  The host side is
  /// byte-for-byte identical to kFp32InKernelSplit -- the two are distinguishable only by schedule.
  kFp32RegSplit,
  /// P1: identical operands and traffic to kFp32RegSplit, but both halves of the split are rounded
  /// to nearest rather than truncated.  B is split by the host, so `launch_pack_b_split` needs the
  /// matching flag; A's split point moves inside the fork's mainloop (see
  /// `collective/dispatch_policy_tf32x3_staged.hpp`).
  kFp32RegSplitRn,
  /// P1 step 2: same operands and split as kFp32RegSplit, plus a second accumulator that grades the
  /// chain every `fold_k` K-tiles (runtime, `EQLIN_FOLD_K`).
  kFp32RegSplitFold,
  /// Both P1 switches at once.
  kFp32RegSplitRnFold,
};

/// What the runner can launch on the SM90 path.  An interface rather than a concrete type because
/// the split-K variant (M5-D) wraps a grouped plan instead of replacing one: it has the same
/// operands and the same launch contract, and differs only in how many problems the tile scheduler
/// sees and where the answer lands.
class Sm90Launch {
 public:
  virtual ~Sm90Launch() = default;
  virtual int launch(float *out, cudaStream_t stream) = 0;

  /// Point the A operand at `a_base` instead of the packed buffer (activation-view mode): every
  /// problem keeps the offset and leading dimension it was built with, only the base moves.  The
  /// base pointer array is uploaded once per distinct `a_base`, so a steady-state call is a no-op.
  virtual void bind_a(void const *a_base, cudaStream_t stream) { (void)a_base; (void)stream; }
};

/// Owns the device-side grouped-GEMM metadata for the SM90 path.  `b_res_data` is only read for
/// kFp32InKernelSplit.
class Sm90GroupedPlan : public Sm90Launch {
 public:
  /// Type-erased operand-element selection; defined in sm90_grouped_gemm.cu.
  struct ImplBase;

  Sm90GroupedPlan(Sm90Operands operands,
                  int B, int V, int num_rows,
                  long long lda, long long ldb, long long ldc,
                  std::vector<RowPlan> const &rows,
                  void const *a_data, void const *b_data, void const *b_res_data = nullptr);
  ~Sm90GroupedPlan();

  Sm90GroupedPlan(Sm90GroupedPlan const &) = delete;
  Sm90GroupedPlan &operator=(Sm90GroupedPlan const &) = delete;

  /// Launch, writing row j of the output to out + j*V (row stride ldc).
  int launch(float *out, cudaStream_t stream) override;

  /// Bind the A operand to an external base (activation-view mode); see Sm90Launch::bind_a.
  void bind_a(void const *a_base, cudaStream_t stream) override;

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin
