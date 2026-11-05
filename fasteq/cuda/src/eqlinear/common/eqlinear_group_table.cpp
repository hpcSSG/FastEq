#include "eqlinear_group_table.h"

namespace fasteq::eqlinear {

int out_total_i(const std::vector<int>& i_dims) {
  int total = 0;
  for (std::size_t p = 0; p < i_dims.size();) {
    const int i = i_dims[p];
    std::size_t q = p + 1;
    while (q < i_dims.size() && i_dims[q] == i) {
      ++q;
    }
    total += i;
    p = q;
  }
  return total;
}

}  // namespace fasteq::eqlinear
