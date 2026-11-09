/***************************************************************************************************
 * fasteq eqlinear backends/fp64/direct_gemm_f64_tma.cuh
 *
 * M6: the hand-written 12-warp kernel with a TMA fill.  Two tile instantiations: 64x64 (V % 64 == 0)
 * and 128x32 (exact-N for V=224 / V=96).  See direct_gemm_f64_tma.cu for the rationale.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "row_plan.h"

namespace eqlin {

class TmaGemmPlan {
 public:
  struct ImplBase;

  TmaGemmPlan(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
              std::vector<RowPlan> const &rows, double const *a_data, double const *b_data,
              int u, bool exact_n);
  ~TmaGemmPlan();

  TmaGemmPlan(TmaGemmPlan const &) = delete;
  TmaGemmPlan &operator=(TmaGemmPlan const &) = delete;

  int launch(double *out, cudaStream_t stream);
  void bind_a(double const *a_base, cudaStream_t stream);

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin
