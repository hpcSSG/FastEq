/***************************************************************************************************
 * fasteq eqlinear backends/fp32/tma_desc.cu
 **************************************************************************************************/
#include "tma_desc.h"

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

namespace eqlin_fp32 {
namespace {

using EncodeFn = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                              const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
                              const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

/// Resolved once: the driver symbol exists on any CUDA >= 12 driver, but the runtime entry point is
/// the portable way to reach it without linking libcuda.
EncodeFn encode_fn() {
  static EncodeFn fn = [] {
    void *p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) !=
            cudaSuccess ||
        p == nullptr) {
      return static_cast<EncodeFn>(nullptr);
    }
    return reinterpret_cast<EncodeFn>(p);
  }();
  return fn;
}

}  // namespace

bool encode_2d_f32(CUtensorMap *map, void const *base, uint64_t rows, uint64_t cols,
                   uint64_t row_stride_bytes, uint32_t box_rows, uint32_t box_cols) {
  EncodeFn fn = encode_fn();
  if (fn == nullptr || map == nullptr || base == nullptr) return false;
  // dim0 is the innermost (contiguous) dimension = the K axis, dim1 is the row axis.
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {row_stride_bytes};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t elem_stride[2] = {1, 1};
  CUresult r = fn(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void *>(base), gdim, gstride,
                  box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    // A failed encode is not fatal: the plan falls back to the cp.async mainloop.
    fprintf(stderr, "[eqlin] TMA encode failed (%d): box %u x %u, %llu x %llu, stride %llu\n",
            static_cast<int>(r), box_cols, box_rows, (unsigned long long)cols,
            (unsigned long long)rows, (unsigned long long)row_stride_bytes);
    return false;
  }
  return true;
}

}  // namespace eqlin_fp32
