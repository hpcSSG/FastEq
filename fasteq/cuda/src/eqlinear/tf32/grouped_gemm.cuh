/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/grouped_gemm.cuh
 *
 * The CUTLASS side of the operator: one group per output row of the eqlinear.
 *
 * Each group is a plain (M=B, N=V, K=g_op*U) GEMM reading from the two packed tensors produced by
 * prep_kernels.cu, with per-group strides (the packed tensors are single contiguous arrays, so
 * lda/ldb are the same constant for every group -- only K and the column offsets differ).
 *
 * tf32x3 comes from arch::OpMultiplyAddFastF32; the SM80 mma.sync path is the only one CUTLASS
 * offers for it (see the research report).
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear.h"
#include "row_plan.h"

namespace eqlin {

/// Owns the device-side grouped-GEMM metadata (problem sizes, pointer/strides arrays, workspace).
/// Lives beside the A_prep/B_prep buffers, so it is constructed once per (problem, tensors) pair
/// and `launch` is then a single kernel launch.
class GroupedGemmPlan {
 public:
  /// Type-erased kernel selection; defined in grouped_gemm.cu.
  struct ImplBase;

  GroupedGemmPlan(Impl impl,
                  int B, int V, int num_rows,
                  long long lda, long long ldb, long long ldc,
                  std::vector<RowPlan> const &rows,
                  float const *a_prep, float const *b_prep);
  ~GroupedGemmPlan();

  GroupedGemmPlan(GroupedGemmPlan const &) = delete;
  GroupedGemmPlan &operator=(GroupedGemmPlan const &) = delete;

  /// Rebind the activation view before launching on this stream.
  int bind_a(float const *a_base, cudaStream_t stream);

  /// Launch, writing row j of the output to out + j*V (row stride ldc).
  int launch(float *out, cudaStream_t stream);

 private:
  std::unique_ptr<ImplBase> p_;
};

}  // namespace eqlin
