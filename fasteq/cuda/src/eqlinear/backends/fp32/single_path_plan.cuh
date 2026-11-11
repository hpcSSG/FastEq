#pragma once

#include "../../common/shape_plan.h"

namespace fasteq::eqlinear::fp32_single {

// Host-only routing. The factory validates geometry and retains fp32_shape for
// sparse row mappings and geometries outside the specialized single problem.
inline bool required(const EqLinearProblemC& p, bool /*dgrad*/) {
  if (p.P != 1 || p.B <= 0 || p.I <= 0 || p.J <= 0 || p.U <= 0 || p.V <= 0 ||
      !p.i_dims) return false;
  // Preserve the established efficient SIMT path on dense, large square
  // problems while its replacement is being accuracy/performance gated. The
  // multiple-of-four restriction preserves the legacy short-chain policy on
  // tail batches; explicit/sparse geometries still use the shape factory.
  const bool square = p.U == p.V && (p.U == 96 || p.U == 128 || p.U == 224);
  if (square && !p.paths && p.I == 1 && p.J == 1 && p.i_dims[0] == 1 &&
      p.B >= 1024 && p.B % 4 == 0) return false;
  return true;
}

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream);

}  // namespace fasteq::eqlinear::fp32_single
