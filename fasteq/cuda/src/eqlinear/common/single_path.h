#pragma once
#include "shape_plan.h"

namespace fasteq::eqlinear {
// A single shared-weight contraction, with full logical operand strides.
// Kernels consume logical offsets directly; no activation packing is implied.
struct SinglePathGeometry {
  ShapeGeometry geometry;
  ShapePath path;
  explicit SinglePathGeometry(const EqLinearProblemC& p, const double* cg)
      : geometry(p, cg), path(geometry.paths.front()) {
    if (p.P != 1) throw std::invalid_argument("eqlinear: expected one path");
  }
  int64_t source_stride(bool backward) const {
    return backward ? int64_t(geometry.J)*geometry.V : int64_t(geometry.I)*geometry.U;
  }
  int64_t destination_stride(bool backward) const {
    return backward ? int64_t(geometry.I)*geometry.U : int64_t(geometry.J)*geometry.V;
  }
};
}  // namespace fasteq::eqlinear
