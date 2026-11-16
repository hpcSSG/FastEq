#pragma once

#include "../../common/single_path.h"

namespace fasteq::eqlinear::fp64_single {

// This policy precedes the legacy TMA U%16 guard; it does not weaken that guard.
// Geometry validation, including explicit row bounds, remains in the factory.
inline bool required(const EqLinearProblemC& p, bool /* dgrad */) {
  return p.P == 1;
}

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream);

}  // namespace fasteq::eqlinear::fp64_single
