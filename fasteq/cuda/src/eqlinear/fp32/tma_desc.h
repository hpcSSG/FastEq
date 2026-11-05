/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/tma_desc.h
 *
 * Thin host-side wrapper over `cuTensorMapEncodeTiled`, reached through the CUDA *runtime* entry
 * point so the .so needs no extra link dependency.  Only the 2-D fp32, no-interleave, no-swizzle
 * case is exposed -- that is what the mainloop needs, and `box_cols = 20` is what lets a TMA tile
 * land in exactly the padded shared-memory layout the LDS code already uses (row stride 20 floats,
 * of which k = 0..15 are consumed).  Verified bit-exact by the M6 probe; see M6_REPORT.md.
 **************************************************************************************************/
#pragma once

#include <cstdint>

#include <cuda.h>

namespace eqlin_fp32 {

/// Tensor is `rows x cols` fp32 with `row_stride_bytes` between consecutive rows; a tile is
/// `box_rows x box_cols` and lands in shared memory row-major with that same row length.
bool encode_2d_f32(CUtensorMap *map, void const *base, uint64_t rows, uint64_t cols,
                   uint64_t row_stride_bytes, uint32_t box_rows, uint32_t box_cols);

}  // namespace eqlin_fp32
