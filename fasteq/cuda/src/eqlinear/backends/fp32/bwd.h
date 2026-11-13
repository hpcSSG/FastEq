/***************************************************************************************************
 * fasteq eqlinear backends/fp32/bwd.h
 *
 * FP32 eqlinear backward: dgrad (dx) + wgrad (dw).
 *
 * CUTLASS-free, like eqlinear.h, so the torch binding can include it from a host-only .cpp.
 *
 * The two gradients are the same kind of problem -- a grouped GEMM -- and differ only in
 * (M, N, K), where the operands live and where the C tile lands:
 *
 *   dgrad  dx[b,i,u] = sum_v dy[b,j,v] * cg_p * W[p,u,v]
 *          one problem per (group, path, component): M=B, N=U, K=V.  i(j) is a bijection, so
 *          there is no accumulation / scatter.
 *
 *   wgrad  dw[p,u,v] = cg_p * sum_{b,t} x[b,i,u] * dy[b,j,v]
 *          one problem per (path, split): M=U, N=V, K=B*i_g, split-K into partial slabs and a
 *          deterministic fp64 reduction.
 *
 * kB3 puts both problem families in ONE row table and launches one grouped GEMM, then the
 * reduction; kB1/kB2 are the dgrad-only / wgrad-only baselines used for A/B and reporting.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear.h"

namespace eqlin_fp32 {

/// Backward implementation id.  **Append-only and independent of the forward's `Impl`**: a
/// backward version is never implied by a forward one.  Only append, never renumber or reuse.
///
///   kB1  dgrad only (dx); no preprocessing, one grouped GEMM.
///   kB2  wgrad only (dw); transpose prep + split-K + deterministic fp64 reduction.
///   kB3  both, fused into one grouped-GEMM launch (one row table) + the reduction.
enum class BwdImpl : int {
  kB1 = 1,
  kB2 = 2,
  kB3 = 3,
};

/// One backward problem, host-side.  `kind` selects which caller-supplied base the kernel adds:
///   0 = dgrad: A base = dy, C base = dx
///   1 = wgrad: A base = Xt (packed transpose of x), C base = partials
/// `b` is a per-problem device pointer (cg*W for dgrad, Yt slice for wgrad).  All element offsets
/// are in floats.
struct BwdProblem {
  int kind = 0;
  int M = 0, N = 0, K = 0;
  float const *b = nullptr;
  long long a_off = 0;
  long long c_off = 0;
  int lda = 0, ldb = 0, ldc = 0;
  double scale = 1.;  // preserve the descriptor coefficient until the selected plan encodes it
  int weight_path = 0;
};

/// Per-call transpose prep descriptor.  One entry per input path (x -> Xt) or per output group
/// (dy -> Yt).  The source block is the contiguous [rows, cols] slice at `src_off` inside one
/// batch item; the destination is a [cols, B*rows] slab at `dst_off` with row stride `dst_ld`,
/// and batch item b starts at column b*rows.
struct TransposeDesc {
  int src_off = 0;
  int rows = 0;
  int cols = 0;
  long long dst_off = 0;
  int dst_ld = 0;
};

/// Type-erased backward launch: the four caller bases are passed per call so the plan never has to
/// rewrite a device pointer array (M5_REPORT.md §5.3's pinned-pointer race).
class BwdPlan {
 public:
  virtual ~BwdPlan() = default;
  virtual bool scales_after_dot() const { return false; }
  virtual bool transposes_weight() const { return false; }
  virtual int launch(float const *dy, float *dx, float const *xt, float *partials,
                     cudaStream_t stream) = 0;
};

/// Prepared backward operator.  Weight-side prep (cg*W) happens in the constructor; the per-call
/// transpose prep is measured separately (and excluded from the caller's core timing).
class BackwardRunner {
 public:
  BackwardRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims, BwdImpl impl,
                 float const *w, double const *cg, cudaStream_t stream);
  ~BackwardRunner();

  BackwardRunner(BackwardRunner const &) = delete;
  BackwardRunner &operator=(BackwardRunner const &) = delete;

  int j_rows() const;
  int num_paths() const;
  int u_cols() const;
  int v_cols() const;

  /// Number of wgrad split-K pieces of the longest path (0 for kB1).
  int splits() const;

  /// prep -> fused grouped GEMM -> reduction.  Writes whichever of dx/dw the impl covers.
  int run(float const *x, float const *dy, float *dx, float *dw, cudaStream_t stream);

 private:
  struct Impl_;
  std::unique_ptr<Impl_> p_;
};

}  // namespace eqlin_fp32
