/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/group_table.cu
 **************************************************************************************************/
#include "group_table.h"

#include <stdexcept>

namespace eqlin {

GroupTable build_group_table(std::vector<int> const &i_dims) {
  if (i_dims.empty()) {
    throw std::invalid_argument("build_group_table: i_dims must be non-empty");
  }
  GroupTable t;
  t.i_dims = i_dims;
  int const P = static_cast<int>(i_dims.size());

  t.in_prefix.assign(P, 0);
  for (int p = 1; p < P; ++p) {
    t.in_prefix[p] = t.in_prefix[p - 1] + i_dims[p - 1];
  }

  // maximal runs of consecutive equal i  (the original eqlinear reference:737-755, generalised)
  int run_start = 0;
  for (int p = 1; p <= P; ++p) {
    if (p == P || i_dims[p] != i_dims[run_start]) {
      t.out_i_dims.push_back(i_dims[run_start]);
      t.path_count.push_back(p - run_start);
      t.path_istart.push_back(run_start);
      run_start = p;
    }
  }

  int const G = static_cast<int>(t.out_i_dims.size());
  t.out_prefix.assign(G, 0);
  for (int g = 1; g < G; ++g) {
    t.out_prefix[g] = t.out_prefix[g - 1] + t.out_i_dims[g - 1];
  }

  t.row_start.resize(G);
  for (int g = 0; g < G; ++g) {
    t.row_start[g] = t.in_prefix[t.path_istart[g]];
  }

  t.out_total_i = 0;
  for (int g = 0; g < G; ++g) {
    t.out_total_i += t.out_i_dims[g];
  }
  return t;
}

}  // namespace eqlin
