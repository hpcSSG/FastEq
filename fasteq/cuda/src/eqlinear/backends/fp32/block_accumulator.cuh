#pragma once

#include <cuda_runtime.h>

namespace eqlin_fp32 {

/// Explicit rounding preserves the short-dot boundary under optimization.
/// There is no second output-tile array and no shared-memory partial sum.
struct BlockAccumulator {
  template <int K>
  __device__ __forceinline__ static void add(float &acc, float const (&a)[K],
                                            float const (&b)[K]) {
    float dot = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) dot = __fmaf_rn(a[k], b[k], dot);
    acc = __fadd_rn(acc, dot);
  }
};

}  // namespace eqlin_fp32
