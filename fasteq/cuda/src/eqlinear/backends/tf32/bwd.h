/***************************************************************************************************
 * fasteq eqlinear backends/tf32/bwd.h
 *
 * Backward (gradient) operators for the eqlinear forward, CUTLASS-free so plain .cpp callers can
 * include it.
 *
 * The backward is two contractions with completely different shapes, and only one of them is a
 * "GEMM-like" problem in the same sense as the forward:
 *
 *   dgrad  dx[b,i,u] = sum_v dy[b,j,v] * cg_p * W[p,u,v]
 *          -> one grouped-GEMM problem per (group, path, component).  (g,t,m) -> i is a bijection,
 *             so no accumulation, no scatter: A and C are plain 2-D slices of dy and dx.
 *
 *   wgrad  dw[p,u,v] = cg_p * sum_{b,t} x[b,i,u] * dy[b,j,v]
 *          -> M=U, N=V, K=B*i_g (up to tens of thousands): needs split-K and a reduction, and on
 *             Hopper both operands must first be made K-contiguous (tf32 WGMMA is TN-only).
 *
 * This file currently declares dgrad only; wgrad lands as a second class in the same header.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear.h"

namespace eqlin {

/// Backward implementation id.  A **separate, append-only series from the forward's `Impl`**: the
/// two lines evolve independently, and a backward version is never implied by a forward one.  The
/// naming is part of the contract the same way `Impl` is -- only append, never renumber or reuse.
///
///   B1  dgrad (dx): one grouped-GEMM problem per (group, path, component), tf32x3 on the RS
///       mainloop, no preprocessing at all.  See B1_REPORT.md.
///   B2  wgrad (dw): one grouped-GEMM problem per (path, split), tf32x3 on the same mainloop, with a
///       transpose prep (tf32 is TN-only, so both operands must arrive K-contiguous) and an fp64
///       split-K reduction.  See B2_REPORT.md.
///   B3  dgrad + wgrad fused into **one** grouped-GEMM launch (one row table carrying both kinds of
///       problem), then the fp64 reduction.  See B3_REPORT.md.
enum class BwdImpl : int {
  kB1 = 1,
  kB2 = 2,
  kB3 = 3,
};

/// dx = dy @ (cg*W)^T as I grouped-GEMM problems (M=B, N=U, K=V).
///
/// The weight side is cg-scaled and tf32-split once at construction (O(P*U*V), independent of B);
/// the per-call work is a pointer bind plus one grouped launch.  `dy[B,J,V]`, `w[P,U,V]`,
/// `dx[B,I,U]`, `i_dims` as everywhere else, `J = out_total_i(i_dims)`.
class DgradRunner {
 public:
  DgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims, BwdImpl bimpl,
              float const *w, double const *cg, cudaStream_t stream);
  ~DgradRunner();

  DgradRunner(DgradRunner const &) = delete;
  DgradRunner &operator=(DgradRunner const &) = delete;

  /// One grouped launch writing dx[B, I, U].
  int run(float const *dy, float *dx, cudaStream_t stream);

 private:
  struct Impl_;
  std::unique_ptr<Impl_> p_;
};

/// dw = cg-scaled wgrad, one grouped-GEMM problem per (path, split).
///
/// `dw[p,u,v] = cg_p * sum_{b,t} x[b,i,u] * dy[b,j,v]`, i.e. `M=U, N=V, K=B*i_g` per path with K up to
/// tens of thousands.  Two consequences drive the design:
///   * tf32 WGMMA is TN-only, so both operands are transposed into K-contiguous slabs first
///     (`Xt[p] = [U, K]`, `Yt[g] = [V, K]`, the latter also carried as tf32 big/res halves because
///     only A is split inside the mainloop);
///   * M,N are tiny, so the K axis is split into S pieces *as extra problems* -- no scheduler
///     support, no atomics -- and a separate kernel sums the S fp32 partials in fp64 and applies cg.
///
/// The prep is per-call work (x and dy both change), so unlike dgrad this operator has a real e2e
/// cost above the kernel.
class WgradRunner {
 public:
  /// `splits <= 0` lets the runner pick S from (P, U, K).
  WgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims, BwdImpl bimpl,
              int splits, double const *cg, cudaStream_t stream);
  ~WgradRunner();

  WgradRunner(WgradRunner const &) = delete;
  WgradRunner &operator=(WgradRunner const &) = delete;

  int splits() const;

  /// Transpose prep, one grouped launch, one fp64 reduction; writes dw[P, U, V].
  int run(float const *x, float const *dy, float *dw, cudaStream_t stream);

 private:
  struct Impl_;
  std::unique_ptr<Impl_> p_;
};

/// B3: both gradients as ONE grouped-GEMM launch.
///
/// dgrad's and wgrad's problems are the same kind of TN tf32x3 problem -- they differ only in
/// `(M,N,K)`, the operand tensors and where the C tile lands -- and the ptr-array kernel has always
/// carried a pointer and a problem shape per problem.  So the two row tables are concatenated into
/// one launch: wgrad's (longer) problems first, dgrad's (shorter) ones last, which is the
/// shortest-tail ordering.  This removes the full drain between the two GEMMs and one launch.
///
/// Requires `U == V` (both kinds share one tile-N instantiation).  The shape prep (wgrad's
/// transposes) still runs per call and is reported separately.
class BackwardRunner {
 public:
  BackwardRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims, BwdImpl bimpl,
                 int splits, float const *w, double const *cg, cudaStream_t stream);
  ~BackwardRunner();

  BackwardRunner(BackwardRunner const &) = delete;
  BackwardRunner &operator=(BackwardRunner const &) = delete;

  /// The largest per-path split count actually used.
  int splits() const;

  /// Shape prep -> one fused grouped launch -> fp64 reduction; writes dx[B,I,U] and dw[P,U,V].
  int run(float const *x, float const *dy, float *dx, float *dw, cudaStream_t stream);

 private:
  struct Impl_;
  std::unique_ptr<Impl_> p_;
};

}  // namespace eqlin
