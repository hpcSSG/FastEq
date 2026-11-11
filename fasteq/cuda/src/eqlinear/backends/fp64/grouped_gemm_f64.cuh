/***************************************************************************************************
 * fasteq eqlinear backends/fp64/grouped_gemm_f64.cuh
 *
 * The CUTLASS side of the FP64 operator: one group per output row of the eqlinear.
 *
 * Each group is a plain (M=B, N=V, K=g_op*U) GEMM with FP64 operands and accumulator, reading from
 * the two packed tensors produced by prep_kernels.cu.  lda/ldb are the same constant for every
 * group (the packed tensors are single contiguous arrays); only K and the column offsets differ.
 *
 * The backend is CUTLASS 2.x device::GemmGrouped: FP64 has no 3.x SM90 collective in this CUTLASS
 * pin, and the FP64 tensor-core instruction is warp-level mma.sync (DMMA), which is exactly what
 * the 2.x multistage mainloop issues.  No splitting, no extra passes.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "row_plan.h"

namespace eqlin {

/// Compile-time tile configurations of the 2.x backend.
///
/// `kBaseline` reproduces the M1 knobs (`F64_*` macros) field for field and stays the default, so
/// impl 0 is bit-identical to M1.  The rest exist for the per-shape dispatch of M4: `kM128N32S3`
/// tiles N in 32-wide strips, which is exact for V=96/224 (V%64 != 0) and keeps 3 CTAs/SM = 12 warps
/// (128 threads, 3 stages, 61 KB smem); the others are controls for the sweep.
///
/// Changing the tile never changes the K-accumulation order, so every config must be bit-identical
/// to `kBaseline` on the same inputs.
enum class Gemm2xConfig {
  kBaseline,     // 64x64x16, warp 32x32, instr 16x8x4, 4 stages
  kM128N32S3,    // 128x32x16, warp 32x32, instr 16x8x4, 3 stages  (exact N for V=96/224)
  kM128N32S4,    // same tile, 4 stages (drops to 2 CTAs/SM: stage control)
  kM64N32S4,     // 64x32x16, warp 32x32, 4 stages (2 warps/CTA: warp-count control)
  kM64N64S3,     // baseline tile at 3 stages (isolates the stage count)
  kIns8,         // baseline tile with the 16x8x8 FP64 instruction
  kM32N128S3,    // 32x128x16: halves the A-strip count (A traffic ~ 1/TB_N), same DMMA work
  kM32N128S4,    // same tile at 4 stages (2 CTAs/SM: stage control)
  kM32N32S3,     // 32x32x16: small one-warp CTA for short-M grouped reductions
  kM128N128S3,   // 128x128x16: larger output tile to reduce A/B reloads
  kM128N64S3,    // 128x64x16: increase M reuse with a bounded accumulator tile
  kM64N128S3,    // 64x128x16: increase N reuse with a bounded accumulator tile
  kM32N64S3,     // 32x64x16: lower accumulator footprint to test occupancy
  kM128N64W32S3, // 128x64x16 with 32x32 warp tiles: lower per-thread accumulator pressure
  kHostSchedule, // baseline 64x64 tile with host-precomputed grouped problem schedule
};

/// Stable name of a 2.x tile config (`EQLIN_2X_TILE` vocabulary and debug output).
char const *gemm2x_config_name(Gemm2xConfig cfg);

/// Owns the device-side grouped-GEMM metadata (problem sizes, pointer/stride arrays, workspace).
/// `a_data` may be null when the caller will supply the A base through `bind_a` before the first
/// launch (activation-view mode).
class GroupedGemmPlan {
 public:
  /// Type-erased kernel selection; defined in grouped_gemm_f64.cu.
  struct ImplBase;

  GroupedGemmPlan(int B, int V, int num_rows,
                  long long lda, long long ldb, long long ldc,
                  std::vector<RowPlan> const &rows,
                  double const *a_data, double const *b_data,
                  Gemm2xConfig cfg = Gemm2xConfig::kBaseline);
  ~GroupedGemmPlan();

  GroupedGemmPlan(GroupedGemmPlan const &) = delete;
  GroupedGemmPlan &operator=(GroupedGemmPlan const &) = delete;

  /// Launch, writing row j of the output to out + j*V (row stride ldc).
  int launch(double *out, cudaStream_t stream);

  /// Point every row's A at `a_base + row.a_off` (activation-view mode).  A repeated call with the
  /// same base costs nothing, so a steady-state call is a no-op.
  void bind_a(double const *a_base, cudaStream_t stream);

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin
