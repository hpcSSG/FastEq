#pragma once

#include "../../common/single_path.h"

namespace fasteq::eqlinear::tf32_single {

// Direction is part of the factory policy so integration can select forward
// and dgrad independently. ShapeGeometry performs all metadata validation.
inline bool required(const EqLinearProblemC& p, bool /*dgrad*/) { return p.P == 1; }

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream);

}  // namespace fasteq::eqlinear::tf32_single
