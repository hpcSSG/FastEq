#pragma once

#include <cstdint>
#include <limits>

#include "../../common/shape_plan.h"

namespace fasteq::eqlinear::tf32_shape {

// CUTLASS-free, host-only policy: both forward and dgrad must fit the frozen
// kernels. Explicit maps always use the shape plan, including dense maps.
inline bool required(const EqLinearProblemC& p) {
  if (p.paths || !p.i_dims || p.B <= 0 || p.I <= 0 || p.U <= 0 || p.P <= 0 ||
      p.P > 32 || p.V <= 0 || p.J <= 0)
    return true;
  const int64_t lda = int64_t(p.I) * p.U;
  const int64_t ldb = int64_t(p.P) * p.U;
  const int64_t ldc = int64_t(p.J) * p.V;
  // U is dgrad's contiguous output dimension; V is its per-path K. The
  // original packers also store logical element offsets in int32 tables.
  if (p.U % 4 || p.V % 4 || lda % 4 || ldb % 4 || ldc % 4 ||
      lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max())
    return true;
  int64_t inputs = 0, outputs = 0;
  for (int first = 0; first < p.P;) {
    const int rows = p.i_dims[first];
    if (rows <= 0) return true;
    int end = first + 1;
    while (end < p.P && p.i_dims[end] == rows) ++end;
    const int64_t k = int64_t(end - first) * p.U;
    if (k % 4 || k > std::numeric_limits<int>::max()) return true;
    inputs += int64_t(end - first) * rows;
    outputs += rows;
    first = end;
  }
  return inputs != p.I || outputs != p.J;
}

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                  const double* cg, bool dgrad, cudaStream_t stream);

}  // namespace fasteq::eqlinear::tf32_shape
