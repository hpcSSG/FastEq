/***************************************************************************************************
 * fasteq eqlinear backends/fp32/prep_kernels.h
 *
 * FP32 uses the shared float prep kernels.  Kept as a backend-local alias header so call sites and
 * include order stay unchanged.
 **************************************************************************************************/
#pragma once

#include "eqlinear_prep_kernels.h"

namespace eqlin_fp32 {

using fasteq::eqlinear::CgValues;
using CgVals = CgValues<float>;
using fasteq::eqlinear::kMaxPaths;
using fasteq::eqlinear::launch_pack_b;
using fasteq::eqlinear::launch_permute_a;

}  // namespace eqlin_fp32
