/***************************************************************************************************
 * fasteq eqlinear backends/fp64/sm90_fp64_grouped_gemm.cuh
 *
 * M2 backend: CUTLASS 3.x SM90 PtrArray grouped GEMM with a custom FP64 DMMA mainloop (TMA +
 * warp specialization, composite 32x32x16 atom) and the stock cooperative kernel, tile scheduler
 * and epilogue.
 *
 * Same group structure as M1 -- one group per eqlinear output row, M=B, N=V, K=g_op*U -- and the
 * same packed operands; the only difference is where the math runs.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "row_plan.h"

namespace eqlin {

/// True when this build can instantiate the SM90 FP64 mainloop.
bool sm90_fp64_available();

/// Owns the grouped-GEMM metadata for the M2/M3 backend.  `composed` selects how A's K axis is
/// described: false = flat packed A (M, K); true = (M, U, m) read straight from x, with the path
/// index in the descriptor's batch mode (M3-A).  `u` and `m_max` are only used when composed.
/// `a_data` may be null when the caller binds the A base through `bind_a` before the first launch.
class Sm90Fp64Plan {
 public:
  struct ImplBase;

  Sm90Fp64Plan(int B, int V, int num_rows,
               long long lda, long long ldb, long long ldc,
               std::vector<RowPlan> const &rows,
               double const *a_data, double const *b_data,
               bool composed = false, int u = 0, int m_max = 1);
  ~Sm90Fp64Plan();

  Sm90Fp64Plan(Sm90Fp64Plan const &) = delete;
  Sm90Fp64Plan &operator=(Sm90Fp64Plan const &) = delete;

  int launch(double *out, cudaStream_t stream);
  void bind_a(double const *a_base, cudaStream_t stream);

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin
