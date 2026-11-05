/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/grouped_gemm_fp32.cuh
 *
 * M1 backend: CUTLASS 2.x `device::GemmGrouped` on the SIMT (FFMA) path.
 *
 * Same group structure as the tf32 line's M1: one group per eqlinear output row, M = B, N = V,
 * K = g_op * U, with the input paths concatenated along K.  The compute is plain FP32 FMA on the
 * CUDA cores -- there is no tf32 conversion and no multi-pass emulation.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "row_plan.h"

namespace eqlin_fp32 {

/// Owns the device-side grouped-GEMM metadata (problem shapes, pointer arrays, per-problem strides,
/// workspace) and the kernel object.  Construction does the one-time `initialize()`; a launch only
/// updates the C pointer array and calls `update()` + `run()`.
class GroupedGemmPlan {
 public:
  /// Type-erased kernel holder; the stock CUTLASS plan and the hand-written SIMT plan both
  /// implement it.  Defined here (not in the .cu) so a second implementation file can derive.
  struct ImplBase {
    virtual ~ImplBase() = default;
    virtual int launch(float *out, cudaStream_t stream) = 0;
    /// Re-point every problem's A at `x` (zero-copy A, M6/P1).  Default no-op: a plan whose A lives
    /// in its own buffer has nothing to rebind.  Returns nonzero if the rebind cannot be done --
    /// the caller must then not launch.
    virtual int bind_a(float const *x, cudaStream_t stream) {
      (void)x;
      (void)stream;
      return 0;
    }
  };

  /// `mode` selects the backend: 0 = stock CUTLASS with its tile read from
  /// `EQLIN_FP32_TILE_POLICY`, 1 = the hand-written SIMT kernel, 2 = per-shape choice between the
  /// two (custom where the stock path has no exact-fit 16x8 tile: V <= 96 or V > 128).
  /// `a_view` says `a_data` is a placeholder and the real A base arrives through `bind_a`.
  GroupedGemmPlan(int M, int N, int num_rows, long long lda, long long ldb, long long ldc,
                  std::vector<RowPlan> const &rows, float const *a_data, float const *b_data,
                  int mode = 0, bool a_view = false);
  ~GroupedGemmPlan();

  GroupedGemmPlan(GroupedGemmPlan const &) = delete;
  GroupedGemmPlan &operator=(GroupedGemmPlan const &) = delete;

  /// Launch, writing row j of the output to out + j*V (row stride ldc).
  int launch(float *out, cudaStream_t stream);

  /// Forward to the backend; see `ImplBase::bind_a`.
  int bind_a(float const *x, cudaStream_t stream);

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin_fp32
