#pragma once

#include "../../common/pointer_table.cuh"
#include "row_plan.h"

namespace eqlin {
inline std::vector<PointerRow<float>> pointer_rows(std::vector<RowPlan> const &rows, long long stride) {
  std::vector<PointerRow<float>> result;
  result.reserve(rows.size());
  for (size_t j = 0; j < rows.size(); ++j) {
    auto const &r = rows[j];
    result.push_back({static_cast<float const *>(r.a_base), static_cast<float *>(r.c_base), r.a_off,
                      (r.c_base || r.c_off >= 0) ? r.c_off : static_cast<long long>(j) * stride});
  }
  return result;
}
}  // namespace eqlin
