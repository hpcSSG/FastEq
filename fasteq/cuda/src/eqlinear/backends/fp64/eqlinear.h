/***************************************************************************************************
 * fasteq eqlinear backends/fp64/eqlinear.h
 *
 * Public interface of the CUTLASS-based FP64 eqlinear forward operator.
 *
 * CUTLASS-free: plain .cpp translation units (the torch binding) include this header, and CUTLASS
 * device headers do not survive a host-only compiler.
 *
 * The operator is the same one the tf32x3 line implements:
 *
 *   out[b, j, n] = sum_{p in P(j)} cg[p] * sum_k x[b, in_prefix[p] + t(j), k] * w[p, k, n]
 *
 * with x/w/cg/out all FP64 and the multiply-add done on FP64 tensor cores.  There is no tf32x3
 * emulation and no operand splitting: one GEMM pass, FP64 accumulation.
 *
 * Everything returns 0 on success, or a nonzero code renderable with status_string().
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear_group_table.h"

namespace eqlin {

using fasteq::eqlinear::GroupTable;
using fasteq::eqlinear::build_group_table;
using fasteq::eqlinear::out_total_i;

/// Backend selection.  Append-only, like the tf32 line's `Impl` contract; this is a separate
/// sequence from that line's ids.
///
///   k2xGrouped -- M1: CUTLASS 2.x device::GemmGrouped with double operands and accumulator,
///                 FP64 tensor-core (DMMA) instruction, one group per eqlinear output row.
///   kSm90Dmma  -- M2: CUTLASS 3.x SM90 PtrArray cooperative kernel with a custom TMA +
///                 warp-specialized FP64 DMMA mainloop (composite 32x32x16 atom, tile 128x64).
///   kAuto      -- M3-A production entry: multi-path shapes use kSm90Dmma with A read straight
///                 from x (no permute); single-path shapes use k2xGrouped (faster there).
///   kAuto2     -- M4: kAuto plus a per-shape 2.x tile (a 32-wide N strip on compute-bound shapes
///                 whose V is not a multiple of 64, where the 64-wide tile spends 12.5-33 % of the
///                 DMMA work on padding).  Same arithmetic order, so the output is bit-identical.
///   kDirect    -- M5-B: hand-written persistent kernel, 128-thread CTAs (3/SM = 12 MMA warps),
///                 composite 32x32x16 atom, fragment-major smem layout (16 B LDS, conflict-free),
///                 composed-K A straight from x, register epilogue.
///   kAuto3     -- M5-C: kAuto2 plus the hand-written kernel on the narrow-N shapes (V <= 128), where
///                 its 3 CTAs/SM beat both stock paths; wider N stays on kAuto2's choice.
///   kAuto4     -- M6: hand-written 12-warp kernel fed by TMA into a canonical SW128 tile; exact-N
///                 (128x32) on shapes whose V is not a multiple of 64, (64,64) elsewhere.
///   kAuto5     -- M6 production candidate: kAuto4 on the shapes where its TMA fill pays (per-row
///                 K <= 1024, i.e. <= 64 k-tiles), kAuto3's choice on the long-K shapes where the
///                 canonical 8-byte fetch costs more than the fill saves.
enum class F64Impl : int { k2xGrouped = 0, kSm90Dmma = 1, kAuto = 2, kAuto2 = 3, kDirect = 4,
                           kAuto3 = 5, kAuto4 = 6, kAuto5 = 7 };

/// Human-readable form of an error code returned by Runner.
char const *status_string(int code);

/// Prepared forward operator.  Buffer allocation and the grouped-GEMM metadata live here, so a
/// benchmark can time `run` (one kernel launch) separately from `prepare_act` (packing).
///
///   x[B, I, U]  @  W[P, U, V]  * cg[P]  ->  out[B, J, V]
class Runner {
 public:
  Runner(int B, int I, int U, int P, int V, std::vector<int> i_dims, F64Impl impl);
  ~Runner();

  Runner(Runner const &) = delete;
  Runner &operator=(Runner const &) = delete;

  /// J = out_total_i(i_dims)
  int j_rows() const;

  /// Weight-side preparation: cg-scaled, packed W.  O(V*P*U), independent of B, and inference
  /// reuses the same weights across calls -- so this belongs to setup, not to the per-call path.
  int prepare_weight(double const *w, double const *cg, cudaStream_t stream);

  /// Activation-side preparation: nothing to do when the A operand is bound straight to x (the
  /// activation-view mode, available when every output group has exactly one input path);
  /// otherwise it permutes x into the packed buffer.
  int prepare_act(double const *x, cudaStream_t stream);

  /// prepare_weight + prepare_act.
  int prepare(double const *x, double const *w, double const *cg, cudaStream_t stream);

  /// One grouped-GEMM launch writing out[B, J, V].
  int run(double *out, cudaStream_t stream);

  /// prepare + run.
  int operator()(double const *x, double const *w, double const *cg, double *out,
                 cudaStream_t stream);

  /// Debug: the packed operands.  `*numel` receives the element count (0 and nullptr in A-view
  /// mode, where no packed A exists).
  double const *a_prep(long long *numel) const;
  double const *b_prep(long long *numel) const;

  /// True when A is read in place from x (no packed A buffer).
  bool a_view() const;

 private:
  struct Impl_;
  std::unique_ptr<Impl_> p_;
};

}  // namespace eqlin
