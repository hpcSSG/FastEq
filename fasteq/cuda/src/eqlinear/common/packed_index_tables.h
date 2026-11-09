#pragma once

#include <vector>

#include "eqlinear_group_table.h"

namespace fasteq::eqlinear {

/// How many K-segments each operand carries. TF32 K-concatenation uses 3 and marks the residual
/// segments; FP32/FP64 use 1.
struct SegmentSpec {
  int count = 1;
  bool a_residual = false;
  bool b_residual = false;
};

/// Per-group offsets into the packed A/B column space. Backends use these to build their own
/// (deliberately different) RowPlan tables without re-walking the geometry.
struct GroupBlock {
  int i = 0;    // output rows (distinct i) in this group
  int m = 0;    // input paths feeding each output row
  int r0 = 0;   // first x row of the group
  int j0 = 0;   // first output row of the group
  int p0 = 0;   // first input path of the group
  int K0 = 0;   // m * U: one segment's contraction length
  int K = 0;    // count * K0
  long long a_base = 0;
  long long b_base = 0;
};

/// Shared A/B column-index mapping for the three eqlinear backends.
///
///   src_index[a_base + t*K + s*K0 + mm*U + k] = (r0 + mm*i + t)*U + k
///   b_src_base[b_base + s*K0 + mm*U + k]      = (p0 + mm)*U*V + k*V
///   b_p_of_col = p0 + mm
class PackedIndexTables {
 public:
  PackedIndexTables(GroupTable const &table, int U, int V, SegmentSpec spec = {});

  std::vector<int> const &src_index() const { return src_index_; }
  std::vector<int> const &b_src_base() const { return b_src_base_; }
  std::vector<int> const &b_p_of_col() const { return b_p_of_col_; }
  std::vector<unsigned char> const &a_is_res() const { return a_is_res_; }
  std::vector<unsigned char> const &b_is_res() const { return b_is_res_; }
  std::vector<GroupBlock> const &blocks() const { return blocks_; }
  long long a_cols() const { return a_cols_; }
  long long b_cols() const { return b_cols_; }

 private:
  std::vector<int> src_index_;
  std::vector<int> b_src_base_;
  std::vector<int> b_p_of_col_;
  std::vector<unsigned char> a_is_res_;
  std::vector<unsigned char> b_is_res_;
  std::vector<GroupBlock> blocks_;
  long long a_cols_ = 0;
  long long b_cols_ = 0;
};

}  // namespace fasteq::eqlinear
