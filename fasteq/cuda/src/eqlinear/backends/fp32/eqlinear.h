/***************************************************************************************************
 * fasteq eqlinear backends/fp32/eqlinear.h
 *
 * Public interface of the FP32 (CUDA-core, FFMA) eqlinear forward operator.
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

namespace eqlin_fp32 {

/// Backend selection.  Append-only: `impl` is part of the torch op contract, so an id is never
/// renumbered or reused.
///
///   kSimt  -- CUTLASS 2.x `device::GemmGrouped` with `arch::OpClassSimt`: the mainloop issues plain
///             FFMA on the FP32 CUDA cores.  No tf32 conversion and no multi-pass emulation, so the
///             accuracy is that of an ordinary fp32 SGEMM (~1e-7 relative), and the ceiling is the
///             GPU's non-tensor FP32 peak (66.9 TFLOPS on H100 SXM5).
enum class Impl : int {
  kSimt = 0,        ///< CUTLASS 2.x grouped SIMT (the M1 backend; tile chosen by EQLIN_FP32_TILE_POLICY)
  kSimtCustom = 1,  ///< M2's hand-written grouped SIMT kernel (exact-fit N, larger thread tile)
  kAuto = 2,        ///< M2's per-shape choice: the custom kernel where it wins (V <= 96 or V > 128,
                    ///< i.e. no exact-fit 16x8 tile exists in the stock path), else stock tile policy 2
  kFused = 3,       ///< M5: kAuto, except that a multi-path configuration chains its per-path
                    ///< problems inside one CTA (summing them in smem) when the grid is wide
                    ///< enough, instead of writing a partial slab and reducing it in a second
                    ///< kernel.  Same numbers as kAuto (same per-path fp32 chains).
                    ///< EQLIN_FP32_FUSED=0 forces the split path for A/B.
};

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

  /// Build A_prep (permuted x) and B_prep (cg-scaled, packed W).
  /// Weight side only: `cg` is folded into the packed weights, so this is the half a caller hoists
  /// (weights are constant across inference calls).
  int prepare_weight(float const *w, double const *cg, cudaStream_t stream);

  /// Activation side only.  With zero-copy A this binds `x` in place (no kernel); otherwise it packs.
  /// **The bound stream must be the stream `run`/`run_act` is launched on.**
  int prepare_act(float const *x, cudaStream_t stream);

  /// One dispatch for the per-call path: `prepare_act` + `run` (requires a prior `prepare_weight`).
  int run_act(float const *x, float *out, cudaStream_t stream);

  /// Both halves in order.  Kept so existing callers (bench/preflight/ncu_probe) need no change.
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

}  // namespace eqlin_fp32
