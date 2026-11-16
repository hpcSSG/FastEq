#pragma once

#include <climits>
#include <memory>

#include "../../common/shape_plan.h"

namespace fasteq::eqlinear::fp32_shape {

// Host-only capability check. Keep the established square kernels and their
// numeric policy intact; scalar logical loads handle all other geometries.
inline bool required(const EqLinearProblemC& p) {
  if (p.paths || p.B <= 0 || p.I <= 0 || p.J <= 0 || p.P <= 0 || p.P > 32 ||
      !p.i_dims || p.U != p.V || p.U % 16 ||
      (p.V != 96 && p.V != 128 && p.V != 224))
    return true;
  int64_t input_rows = 0, output_rows = 0;
  for (int q = 0; q < p.P; ++q) {
    if (p.i_dims[q] <= 0) return true;
    input_rows += p.i_dims[q];
    if (!q || p.i_dims[q] != p.i_dims[q - 1]) output_rows += p.i_dims[q];
  }
  const int64_t lda = int64_t(p.I) * p.U;
  const int64_t ldb = int64_t(p.P) * p.U;
  const int64_t ldc = int64_t(p.J) * p.V;
  return input_rows != p.I || output_rows != p.J ||
         lda > INT_MAX || ldb > INT_MAX || ldc > INT_MAX ||
         lda % 4 || ldb % 4 || ldc % 4 || p.V % 4;
}

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream);

}  // namespace fasteq::eqlinear::fp32_shape
