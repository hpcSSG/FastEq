#pragma once

#include <vector>

namespace fasteq::eqlinear {

/// Host-side description of how the P input paths collapse into J output rows.
///
/// Mirrors cutlass_impl/tests/eqlinear_ref.py::build_group_table, which is cross-validated against
/// the torch reference implementation. CUTLASS-free.
struct GroupTable {
  std::vector<int> i_dims;       // per input path multiplicity, length P
  std::vector<int> in_prefix;    // exclusive prefix sum of i_dims, length P
  std::vector<int> out_i_dims;   // distinct run-length i values, length G
  std::vector<int> out_prefix;   // exclusive prefix sum of out_i_dims, length G
  std::vector<int> path_count;   // input paths per output group, length G
  std::vector<int> path_istart;  // first input-path index of each group, length G
  std::vector<int> row_start;    // in_prefix[path_istart[g]]: first x row of each group
  int out_total_i = 0;           // J = sum(out_i_dims)

  int num_paths() const { return static_cast<int>(i_dims.size()); }
  int num_groups() const { return static_cast<int>(out_i_dims.size()); }

  /// K of output group g: path_count[g] * U.
  int group_k(int g, int U) const { return path_count[g] * U; }
};

/// Derive the grouping: maximal runs of consecutive equal values in i_dims.
/// Throws std::invalid_argument for an empty i_dims.
GroupTable build_group_table(std::vector<int> const &i_dims);

/// J: sum of the distinct run-length-grouped i values. Empty input returns 0.
int out_total_i(const std::vector<int> &i_dims);

}  // namespace fasteq::eqlinear
