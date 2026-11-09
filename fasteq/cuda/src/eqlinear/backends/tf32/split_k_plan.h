/***************************************************************************************************
 * fasteq eqlinear backends/tf32/split_k_plan.h
 *
 * Split-K decorator for the SM90 grouped path (M5-D).
 *
 * Why a decorator rather than a new mainloop: K is a purely host-side number here.  `RowPlan` holds
 * {K, a_off, b_off}, the operands are raw pointer+offset pairs, and the launch path maps problem j
 * to `out + j*V`.  So "split this row's K into S pieces" is the same transformation as "list S
 * problems whose K, a_off and b_off start at successive positions" -- nothing below the host layer
 * has to know, and the tile scheduler simply sees more, shorter tiles.
 *
 * The wrapper therefore expands the row list and redirects the output to its own (S*J, B, V)
 * partial buffer.  CUTLASS has no split-K for the PtrArray path in this pin (the array kernel
 * static_asserts a GroupScheduler, and the only group-compatible scheduler's fixup() is empty), so
 * the reduction is ours to do, not the kernel's.
 **************************************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "device_buffer.h"
#include "row_plan.h"
#include "sm90_grouped_gemm.cuh"

namespace eqlin {

/// Split every row's K into `splits` slices, each a whole number of `k_unit`-sized chunks and all
/// but the last as equal as possible; the last slice also takes any `K % k_unit` tail.
/// The result is laid out row-major over (row, slice), which is exactly the order the problem index
/// of the wrapped plan follows.
std::vector<RowPlan> split_k_rows(std::vector<RowPlan> const &rows, int splits, int k_unit);

/// Runs a split-K grouped GEMM into an internal partial buffer.
///
/// **Ablation mode is deliberately not a correct operator.**  Nothing here sums the S partials, so
/// `launch()` leaves `out` untouched and the answer is wrong -- the only thing this measures is what
/// the extra parallelism buys.  The shell that calls it is `EQLIN_SPLITK_ABLATION=S` in
/// `runner.cu`, and the reduction (impl 6) is a separate piece of work gated on this measurement.
class Sm90SplitKPlan : public Sm90Launch {
 public:
  Sm90SplitKPlan(Sm90Operands operands,
                 int B, int V, int num_rows,
                 long long lda, long long ldb,
                 std::vector<RowPlan> const &rows,
                 void const *a_data, void const *b_data, void const *b_res_data,
                 int splits, int k_unit, bool reduce = false);

  int launch(float *out, cudaStream_t stream) override;

  void bind_a(void const *a_base, cudaStream_t stream) override {
    inner_->bind_a(a_base, stream);
  }

  int splits() const { return splits_; }

 private:
  int splits_;
  bool reduce_ = false;
  int J_ = 0;
  int B_ = 0;
  int V_ = 0;
  int num_rows_;
  DeviceBuffer partials_;
  std::unique_ptr<Sm90GroupedPlan> inner_;
};

}  // namespace eqlin
