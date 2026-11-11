#include "packed_index_tables.h"

#include <stdexcept>

namespace fasteq::eqlinear {

PackedIndexTables::PackedIndexTables(GroupTable const &table, int U, int V, SegmentSpec spec) {
  if (spec.count < 1 || U <= 0 || V <= 0) {
    throw std::invalid_argument("PackedIndexTables: invalid segment/geometry");
  }
  int const P = table.num_paths();
  int const I = table.in_prefix[P - 1] + table.i_dims[P - 1];
  a_cols_ = static_cast<long long>(spec.count) * I * U;
  b_cols_ = static_cast<long long>(spec.count) * P * U;
  src_index_.assign(static_cast<size_t>(a_cols_), 0);
  b_src_base_.assign(static_cast<size_t>(b_cols_), 0);
  b_p_of_col_.assign(static_cast<size_t>(b_cols_), 0);
  if (spec.a_residual) a_is_res_.assign(static_cast<size_t>(a_cols_), 0);
  if (spec.b_residual) b_is_res_.assign(static_cast<size_t>(b_cols_), 0);

  blocks_.reserve(static_cast<size_t>(table.num_groups()));
  long long a_col = 0;
  long long b_col = 0;
  for (int g = 0; g < table.num_groups(); ++g) {
    int const i = table.out_i_dims[g];
    int const m = table.path_count[g];
    int const r0 = table.row_start[g];
    int const j0 = table.out_prefix[g];
    int const p0 = table.path_istart[g];
    int const K0 = m * U;
    int const K = spec.count * K0;
    blocks_.push_back({i, m, r0, j0, p0, K0, K, a_col, b_col});

    for (int t = 0; t < i; ++t) {
      for (int s = 0; s < spec.count; ++s) {
        for (int mm = 0; mm < m; ++mm) {
          for (int k = 0; k < U; ++k) {
            long long const col = a_col + static_cast<long long>(t) * K + s * K0 + mm * U + k;
            src_index_[col] = (r0 + mm * i + t) * U + k;
            if (spec.a_residual) a_is_res_[col] = (s == 1) ? 1 : 0;
          }
        }
      }
    }
    for (int s = 0; s < spec.count; ++s) {
      for (int mm = 0; mm < m; ++mm) {
        for (int k = 0; k < U; ++k) {
          long long const col = b_col + static_cast<long long>(s) * K0 + mm * U + k;
          b_src_base_[col] = static_cast<long long>(p0 + mm) * U * V + static_cast<long long>(k) * V;
          b_p_of_col_[col] = p0 + mm;
          if (spec.b_residual) b_is_res_[col] = (s == 2) ? 1 : 0;
        }
      }
    }
    a_col += static_cast<long long>(i) * K;
    b_col += static_cast<long long>(spec.count) * K0;
  }
  if (a_col != a_cols_ || b_col != b_cols_) {
    throw std::runtime_error("PackedIndexTables: packed column count mismatch");
  }
}

}  // namespace fasteq::eqlinear
