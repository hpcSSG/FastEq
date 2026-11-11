#pragma once

#include <cuda_runtime.h>

namespace fasteq::eqlinear::fp32_single {

// FP32 error-free transforms, matching fp32_shape's arithmetic. Explicit RN
// operations protect the residuals even when the translation unit enables FMA.
struct FloatPair {
  float hi = 0.f, lo = 0.f;

  __device__ __forceinline__ static FloatPair two_sum(float a, float b) {
    const float hi = __fadd_rn(a, b);
    const float bv = __fsub_rn(hi, a);
    return {hi, __fadd_rn(__fsub_rn(a, __fsub_rn(hi, bv)), __fsub_rn(b, bv))};
  }

  __device__ __forceinline__ static FloatPair two_prod(float a, float b) {
    const float hi = __fmul_rn(a, b);
    return {hi, __fmaf_rn(a, b, -hi)};
  }

  __device__ __forceinline__ void add(FloatPair b) {
    const FloatPair s = two_sum(hi, b.hi);
    const FloatPair t = two_sum(lo, b.lo);
    const FloatPair u = two_sum(s.lo, t.hi);
    const FloatPair v = two_sum(s.hi, u.hi);
    *this = two_sum(v.hi, __fadd_rn(v.lo, __fadd_rn(u.lo, t.lo)));
  }

  __device__ __forceinline__ FloatPair multiply(FloatPair b) const {
    FloatPair result = two_prod(hi, b.hi);
    result.add(two_prod(hi, b.lo));
    result.add(two_prod(lo, b.hi));
    result.add(two_prod(lo, b.lo));
    return result;
  }

  __device__ __forceinline__ float scale(float2 cg) const {
    // Double-single multiplication: keep the exact leading product residual
    // and all three low cross terms. The correction is rounded only in FP32;
    // neither the dot nor the original double CG is truncated to one float.
    const FloatPair product = two_prod(hi, cg.x);
    float correction = __fmul_rn(lo, cg.y);
    correction = __fmaf_rn(lo, cg.x, correction);
    correction = __fmaf_rn(hi, cg.y, correction);
    correction = __fadd_rn(product.lo, correction);
    return __fadd_rn(product.hi, correction);
  }

  __device__ __forceinline__ float rounded() const { return __fadd_rn(hi, lo); }
};

// Compensated dot product (TwoProd + TwoSum): carry product and addition
// residuals separately until the end. This avoids a full pair/pair merge on
// every product while retaining the FP32 low-order correction.
struct DotAccumulator {
  float sum = 0.f, correction = 0.f;

  __device__ __forceinline__ void add_product(float a, float b) {
    const FloatPair product = FloatPair::two_prod(a, b);
    const FloatPair added = FloatPair::two_sum(sum, product.hi);
    correction = __fadd_rn(correction, __fadd_rn(added.lo, product.lo));
    sum = added.hi;
  }

  __device__ __forceinline__ FloatPair finish() const {
    return FloatPair::two_sum(sum, correction);
  }
};

}  // namespace fasteq::eqlinear::fp32_single
