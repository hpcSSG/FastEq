/***************************************************************************************************
 * fasteq eqlinear backends/tf32/split_k_plan.cu
 **************************************************************************************************/
#include "split_k_plan.h"

#include "reduce_fwd.h"

#include <algorithm>
#include <stdexcept>

namespace eqlin {

std::vector<RowPlan> split_k_rows(std::vector<RowPlan> const &rows, int splits, int k_unit) {
  std::vector<RowPlan> out;
  out.reserve(rows.size() * static_cast<size_t>(splits));
  for (RowPlan const &r : rows) {
    int const chunks = r.K / k_unit;  // whole k_unit-tiles this row's K divides into
    int const s = std::min(splits, chunks);
    int const q = chunks / s;
    int const rem = chunks % s;
    long long off = 0;
    for (int i = 0; i < s; ++i) {
      long long len = static_cast<long long>(q + (i < rem ? 1 : 0)) * k_unit;
      if (i == s - 1) len = r.K - off;  // last slice also takes the K % k_unit tail
      out.push_back(RowPlan{static_cast<int>(len), r.a_off + off, r.b_off + off, r.lda});
      off += len;
    }
  }
  return out;
}

Sm90SplitKPlan::Sm90SplitKPlan(Sm90Operands operands, int B, int V, int num_rows, long long lda,
                               long long ldb, std::vector<RowPlan> const &rows, void const *a_data,
                               void const *b_data, void const *b_res_data, int splits,
                               int k_unit, bool reduce)
    : splits_(splits), reduce_(reduce), J_(num_rows), B_(B), V_(V), num_rows_(num_rows) {
  if (splits < 2) throw std::invalid_argument("Sm90SplitKPlan: splits must be >= 2");
  std::vector<RowPlan> const expanded = split_k_rows(rows, splits, k_unit);
  // The problem index is what routes the output, so every row must contribute exactly `splits`
  // slices -- otherwise problem j*S+s would land on the wrong partial.
  if (expanded.size() != rows.size() * static_cast<size_t>(splits)) {
    throw std::invalid_argument("Sm90SplitKPlan: a row has fewer K-tiles than the split count");
  }

  // Partial slice (j, s) is output row j*splits + s of the inner plan, which writes it at
  // `out + (j*splits+s)*V`; the tile's own row stride is therefore the whole partial buffer's.
  long long const ldc = static_cast<long long>(num_rows) * splits * V;
  if (partials_.reset(static_cast<size_t>(ldc) * B * sizeof(float))) {
    throw std::runtime_error("Sm90SplitKPlan: partial buffer allocation failed");
  }
  inner_ = std::make_unique<Sm90GroupedPlan>(operands, B, V, num_rows * splits, lda, ldb, ldc,
                                             expanded, a_data, b_data, b_res_data);
}

int Sm90SplitKPlan::launch(float *out, cudaStream_t stream) {
  int const rc = inner_->launch(partials_.as<float>(), stream);
  if (rc != 0) return rc;
  // Ablation mode leaves `out` untouched on purpose (see the note in split_k_plan.h); the graded
  // path sums the S partials in fp64 and is a correct operator.
  if (reduce_) launch_reduce_fwd(partials_.as<float>(), out, J_, B_, V_, splits_, stream);
  return 0;
}

}  // namespace eqlin
