/***************************************************************************************************
 * fasteq eqlinear backends/tf32/eqlinear.h
 *
 * Public interface of the CUTLASS-based tf32x3 eqlinear forward operator.
 *
 * This header is deliberately CUTLASS-free so that plain .cpp translation units (e.g. the torch
 * binding) can include it -- CUTLASS device headers do not survive a host-only compiler.
 *
 * Everything returns 0 on success, or a nonzero code renderable with status_string().
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace eqlin {

/// Backend / precision selection.
///
///   k3xTF32 / k1xTF32     -- M1: CUTLASS 2.x device::GemmGrouped on the SM80 mma.sync path
///   kSm90_1xTF32          -- M2 probe: stock SM90 PtrArray grouped GEMM (WGMMA + TMA + warp spec).
///                            Same packed operands as M1; plain 1xTF32 numerics, so it exists purely
///                            to measure what the Hopper path buys on these shapes.
///   kSm90_3xTF32          -- M2: SM90 grouped GEMM with tf32x3 expressed as a K-concatenation
///                            ([A_big|A_res|A_big] @ [[B_big];[B_big];[B_res]], K3 = 3*K0).
///   kSm90_3xTF32InKernel  -- M3-2: same numerics, but A is stored once (plain fp32 permute) and
///                            the big/res split happens inside the mainloop into a shared-memory
///                            A_res tile. Only B is pre-split (weights, a few MB, off the inference
///                            path).
///   kSm90_3xTF32RegSplit  -- M4-1: same numerics and the same operands as kSm90_3xTF32InKernel,
///                            but the mainloop issues an `rs_` WGMMA: A is read from shared memory
///                            into registers *once* and A_res is derived there, so neither an A_res
///                            tile nor the A-side operand re-reads exist. See M4_REPORT.md.
///   kSm90_3xTF32RegSplitRn -- P1: impl 5 with both halves of the tf32 split rounded to nearest
///                            instead of truncated.  Same operands and traffic; only the split
///                            point moves.  See TF32_PRECISION_REPORT.md §3.2.
enum class Impl : int { k3xTF32 = 0, k1xTF32 = 1, kSm90_1xTF32 = 2, kSm90_3xTF32 = 3,
                        kSm90_3xTF32InKernel = 4, kSm90_3xTF32RegSplit = 5,
                        kSm90_3xTF32RegSplitRn = 6,
                        kSm90_3xTF32RegSplitFold = 7,
                        kSm90_3xTF32RegSplitRnFold = 8 };

/// True when `impl` uses the SM90 (WGMMA/TMA) backend.
inline bool is_sm90(Impl impl) {
  return impl == Impl::kSm90_1xTF32 || impl == Impl::kSm90_3xTF32 ||
         impl == Impl::kSm90_3xTF32InKernel || impl == Impl::kSm90_3xTF32RegSplit ||
         impl == Impl::kSm90_3xTF32RegSplitRn || impl == Impl::kSm90_3xTF32RegSplitFold ||
         impl == Impl::kSm90_3xTF32RegSplitRnFold;
}

/// True when `impl` needs the tf32 (big,res) split + K-concatenation packing.
inline bool is_kconcat(Impl impl) { return impl == Impl::kSm90_3xTF32; }

/// True when `impl` splits A inside the mainloop, so the A operand is a plain fp32 permute.  Both
/// tf32x3 SM90 variants qualify -- they differ only in *where* in the kernel A_res is computed
/// (shared memory for 4, registers for 5), which the host side cannot see: the operand layout, the
/// preparation kernels and the accuracy contract are identical.
inline bool is_inkernel_tf32x3(Impl impl) {
  return impl == Impl::kSm90_3xTF32InKernel || impl == Impl::kSm90_3xTF32RegSplit ||
         impl == Impl::kSm90_3xTF32RegSplitRn || impl == Impl::kSm90_3xTF32RegSplitFold ||
         impl == Impl::kSm90_3xTF32RegSplitRnFold;
}

/// Number of output rows J: the sum of the distinct run-length-grouped i values.
int out_total_i(std::vector<int> const &i_dims);

/// Human-readable form of an error code returned by Runner.
char const *status_string(int code);

/// Prepared forward operator.  Buffer allocation and the grouped-GEMM metadata live here, so a
/// benchmark can time `run` (one kernel launch) separately from `prepare` (permute + pack).
///
///   x[B, I, U]  @  W[P, U, V]  * cg[P]  ->  out[B, J, V]
class Runner {
 public:
  Runner(int B, int I, int U, int P, int V, std::vector<int> i_dims, Impl impl);
  ~Runner();

  Runner(Runner const &) = delete;
  Runner &operator=(Runner const &) = delete;

  /// J = out_total_i(i_dims)
  int j_rows() const;

  /// Weight-side preparation: cg-scaled, packed W (split into big/res as the impl requires).
  /// Costs O(V*P*U) -- independent of B -- and inference reuses the same weights across calls, so
  /// this belongs to setup rather than to the per-call path.  See `prepare_act` for the other half.
  int prepare_weight(float const *w, double const *cg, cudaStream_t stream);

  bool a_view() const;

  /// Activation-side preparation: makes x readable by the GEMM.  Nothing to do when the A operand is
  /// bound straight to x (the activation-view mode); otherwise it permutes x into the packed buffer.
  int prepare_act(float const *x, cudaStream_t stream);

  /// Build A_prep (permuted x) and B_prep (cg-scaled, packed W) -- prepare_weight + prepare_act.
  int prepare(float const *x, float const *w, double const *cg, cudaStream_t stream);

  /// One grouped-GEMM launch writing out[B, J, V].
  int run(float *out, cudaStream_t stream);

  /// prepare + run.
  int operator()(float const *x, float const *w, double const *cg, float *out, cudaStream_t stream);

  /// Debug: the packed operands.  `*numel` receives the element count.
  float const *a_prep(long long *numel) const;
  float const *b_prep(long long *numel) const;

 private:
  struct Impl_;
  std::unique_ptr<Impl_> p_;
};

}  // namespace eqlin
