#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../../common/shape_plan.h"

namespace fasteq::eqlinear::fp64_shape {

// Host-only policy: these are the frozen Runner/TmaGemmPlan bounds, not
// tensor-core requirements. In particular, grouped K alignment cannot fix U=10.
inline bool required(const EqLinearProblemC& p) {
  if (p.paths || p.U <= 0 || p.U % 16 || p.V <= 0 || p.V % 2 ||
      (int64_t(p.I) * p.U) % 2 || (int64_t(p.P) * p.U) % 2 ||
      (int64_t(p.J) * p.V) % 2 || p.J > 32)
    return true;
  if (!p.i_dims || p.P <= 0 || p.P > 32) return true;
  int64_t inputs = 0, outputs = 0;
  for (int q = 0; q < p.P; ++q) {
    if (p.i_dims[q] <= 0) return true;
    inputs += p.i_dims[q];
    if (!q || p.i_dims[q] != p.i_dims[q - 1]) outputs += p.i_dims[q];
  }
  // Full logical operands may contain unused rows even without an explicit map.
  return inputs != p.I || outputs != p.J;
}

namespace detail {

struct RowPath {
  int source_row, weight_path;
  double coefficient;
};

// CSR by destination row, retaining path order for deterministic summation.
// The reverse table also handles overlapping input ranges in explicit maps.
class RowTable {
 public:
  RowTable(const ShapeGeometry& geometry, bool dgrad)
      : offsets_(size_t(dgrad ? geometry.I : geometry.J) + 1, 0) {
    for (const auto& path : geometry.paths) {
      int start = dgrad ? path.input_row : path.output_row;
      for (int t = 0; t < path.rows; ++t) ++offsets_[size_t(start) + t + 1];
    }
    for (size_t r = 1; r < offsets_.size(); ++r) offsets_[r] += offsets_[r - 1];
    paths_.resize(static_cast<size_t>(offsets_.back()));
    auto cursor = offsets_;
    for (const auto& path : geometry.paths) {
      int start = dgrad ? path.input_row : path.output_row;
      int source = dgrad ? path.output_row : path.input_row;
      for (int t = 0; t < path.rows; ++t)
        paths_[cursor[size_t(start) + t]++] = {source + t, path.weight_path, path.coefficient};
    }
  }

  const std::vector<int64_t>& offsets() const { return offsets_; }
  const std::vector<RowPath>& paths() const { return paths_; }

 private:
  std::vector<int64_t> offsets_;
  std::vector<RowPath> paths_;
};

}  // namespace detail

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream);

}  // namespace fasteq::eqlinear::fp64_shape
