/***************************************************************************************************
 * fasteq eqlinear backends/fp64/direct_gemm_f64_tma.cuh
 *
 * M6: the hand-written 12-warp kernel with a TMA fill.  Two tile instantiations: 64x64 (V % 64 == 0)
 * and 128x32 (exact-N for V=224 / V=96).  See direct_gemm_f64_tma.cu for the rationale.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear_common.h"
#include "prep_kernels.h"
#include "row_plan.h"

namespace eqlin {

class TmaGemmPlan {
 public:
  struct ImplBase;

  static inline void validate_shape(int num_rows, std::vector<RowPlan> const &rows, int u) {
    if (num_rows <= 0 || num_rows > kMaxPaths ||
        rows.size() < static_cast<std::size_t>(num_rows))
      throw std::invalid_argument("TmaGemmPlan: bad rows");
    // tma_fill divides by u/16 and every row consumes full 16-element K tiles.
    if (u <= 0 || u % 16 != 0)
      throw fasteq::eqlinear::UnsupportedConfiguration(
          "TmaGemmPlan: U must be positive and a multiple of 16");
    for (int j = 0; j < num_rows; ++j) {
      if (rows[j].K <= 0 || rows[j].K % 16 != 0)
        throw fasteq::eqlinear::UnsupportedConfiguration(
            "TmaGemmPlan: row K must be positive and a multiple of 16");
    }
  }

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
